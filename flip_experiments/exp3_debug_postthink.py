import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"
TARGETS = [
    (
        "4367330b55f72ace04b09a9addab85fcfad35f1cde4748b0191eb6bbeee83cbe/851e092b-4067-4e5c-91d2-e9dfb63d9ac4",
        "polarity",
    ),
    (
        "f1ac9bfbe98d4a47106c269ff3ebd482c0e8d52390bb9eeadcf33ff7d9da3120/fe4386a4-78da-4924-a081-3719ed84e4d9",
        "polarity",
    ),
]


def mirror_case(word, template):
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word.capitalize()


def find_answer_span(completion, which):
    tail = (
        completion.split("</think>", 1)[1] if "</think>" in completion else completion
    )
    offset = len(completion) - len(tail)
    ms = list(re.finditer(r"\b(yes|no)\b", tail, re.IGNORECASE))
    if not ms:
        return None
    m = ms[0] if which == "first" else ms[-1]
    return offset + m.start(), offset + m.end()


def locate(tok, full_text, abs_start, abs_end, consistent_label):
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids_full, offsets = enc["input_ids"], enc["offset_mapping"]
    covering = [i for i, (s, e) in enumerate(offsets) if s < abs_end and e > abs_start]
    if len(covering) != 1:
        return None
    idx = covering[0]
    s, e = offsets[idx]
    if not (s <= abs_start and e >= abs_end):
        return None
    real_token_id = ids_full[idx]
    prefix_text = full_text[:s]
    leading_ws = full_text[s:abs_start]
    observed_word = full_text[abs_start:abs_end]
    inconsistent_label = "NO" if consistent_label == "YES" else "YES"
    observed_label = "YES" if observed_word.upper() == "YES" else "NO"
    other_label = (
        inconsistent_label if observed_label == consistent_label else consistent_label
    )
    other_word = leading_ws + mirror_case(other_label, observed_word)
    ids_prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
    ids_with_other = tok(prefix_text + other_word, add_special_tokens=False)[
        "input_ids"
    ]
    oth_added = ids_with_other[len(ids_prefix) :]
    other_token_id = oth_added[0] if len(oth_added) == 1 else None
    consistent_token_id = (
        real_token_id if observed_label == consistent_label else other_token_id
    )
    inconsistent_token_id = (
        other_token_id if observed_label == consistent_label else real_token_id
    )
    return {
        "prefix_text": prefix_text,
        "pre_index": len(ids_prefix) - 1,
        "real_token_id": real_token_id,
        "consistent_token_id": consistent_token_id,
        "inconsistent_token_id": inconsistent_token_id,
    }


def main():
    payload = json.load(open("exp3_gpu_payload.json"))
    examples = payload["examples"]
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in payload["prompts"]}
    gens = {
        (g["example_id"], g["condition"]): g["completion"]
        for g in (json.loads(l) for l in open("exp3_local_generations.jsonl"))
    }

    print(f"loading {MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    input_device = model.get_input_embeddings().weight.device
    final_norm, lm_head = model.model.norm, model.lm_head

    for ex_id, iface in TARGETS:
        ex = examples[ex_id]
        consistent_label = ex["cot_supported_answer"]
        completion = gens[(ex_id, iface)]
        p = prompts_by_key[(ex_id, iface)]
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = chat_text + completion
        prompt_len = len(tok(chat_text, add_special_tokens=False)["input_ids"])

        span = find_answer_span(completion, "first")
        start, end = span
        abs_start, abs_end = len(chat_text) + start, len(chat_text) + end
        loc = locate(tok, full_text, abs_start, abs_end, consistent_label)
        pre_index = loc["pre_index"]
        k = pre_index - prompt_len + 1

        n_new = pre_index - prompt_len + 5
        prompt_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(input_device)
        print(
            f"\n{'='*80}\n{ex['qid'][:12]}.../{iface}/post_think  pre_index={pre_index}  k={k}  regenerating {n_new} tokens",
            flush=True,
        )
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=n_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        actual_next_id = int(out.sequences[0][pre_index + 1].item())
        print(
            f"actual next token in regenerated sequence: {actual_next_id} = {tok.decode([actual_next_id])!r}"
            f"  (expected real_token_id={loc['real_token_id']} = {tok.decode([loc['real_token_id']])!r})"
        )

        # ground truth: generate()'s OWN logits used to pick this exact token (move to CPU to
        # sidestep any multi-GPU device mismatch when comparing against our reconstruction)
        gt_logits = out.scores[k][0].float().cpu()
        gt_top10 = torch.topk(gt_logits, 10)
        print("\n-- outputs.scores (generate()'s ACTUAL decision logits) top 10 --")
        for val, idx in zip(gt_top10.values.tolist(), gt_top10.indices.tolist()):
            print(f"  {idx:8d}  {tok.decode([idx])!r:14s} logit={val:+.4f}")
        gt_argmax = int(gt_logits.argmax().item())

        raw_hs = out.hidden_states[k][n_layers][0, 0, :].to(
            final_norm.weight.device, dtype=final_norm.weight.dtype
        )
        print(
            f"\nraw hidden_states[-1] stats: mean={raw_hs.float().mean().item():+.4f} "
            f"std={raw_hs.float().std().item():.4f} L2norm={raw_hs.float().norm().item():.2f}"
        )

        for label, hs_to_head in (
            (
                "CACHED-gen hidden_state, WITH final_norm re-applied (assumes pre-norm)",
                final_norm(raw_hs),
            ),
            (
                "CACHED-gen hidden_state, WITHOUT re-applying norm (assumes already normed)",
                raw_hs,
            ),
        ):
            logits = lm_head(hs_to_head).float().cpu()
            top10 = torch.topk(logits, 10)
            argmax_id = int(logits.argmax().item())
            agree = argmax_id == gt_argmax
            max_diff = (gt_logits - logits).abs().max().item()
            print(f"\n-- {label} --")
            for val, idx in zip(top10.values.tolist(), top10.indices.tolist()):
                print(f"  {idx:8d}  {tok.decode([idx])!r:14s} logit={val:+.4f}")
            print(
                f"  argmax={argmax_id} ({tok.decode([argmax_id])!r})  agrees with GT: {agree}  "
                f"max_abs_diff_vs_GT={max_diff:.6f}"
            )
            print(
                f"  consistent={logits[loc['consistent_token_id']]:+.4f}  "
                f"inconsistent={logits[loc['inconsistent_token_id']]:+.4f}"
            )

        # third comparison: a PLAIN, non-cached forward pass over the exact same prefix
        # (this is what exp3_logit_lens.py's main figure and exp3_dual_probe.py both used --
        # need to know if IT also has a double-norm issue, or is unaffected)
        plain_input_ids = tok(
            loc["prefix_text"], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(input_device)
        with torch.no_grad():
            plain_out = model(plain_input_ids, output_hidden_states=True)
        plain_raw_hs = plain_out.hidden_states[n_layers][0, -1, :].to(
            final_norm.weight.device, dtype=final_norm.weight.dtype
        )
        print(
            f"\nplain (non-cached) forward pass, raw hidden_states[-1] stats: "
            f"mean={plain_raw_hs.float().mean().item():+.4f} std={plain_raw_hs.float().std().item():.4f} "
            f"L2norm={plain_raw_hs.float().norm().item():.2f}  (compare to cached-gen stats above)"
        )
        for label, hs_to_head in (
            ("PLAIN forward, WITH final_norm re-applied", final_norm(plain_raw_hs)),
            ("PLAIN forward, WITHOUT re-applying norm", plain_raw_hs),
        ):
            logits = lm_head(hs_to_head).float().cpu()
            argmax_id = int(logits.argmax().item())
            agree = argmax_id == gt_argmax
            max_diff = (gt_logits - logits).abs().max().item()
            print(
                f"-- {label} --  argmax={argmax_id} ({tok.decode([argmax_id])!r})  agrees with GT: {agree}  "
                f"max_abs_diff_vs_GT={max_diff:.6f}  consistent={logits[loc['consistent_token_id']]:+.4f}  "
                f"inconsistent={logits[loc['inconsistent_token_id']]:+.4f}"
            )


if __name__ == "__main__":
    main()
