import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"


def mirror_case(word: str, template: str) -> str:
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word.capitalize()


def find_answer_span(completion: str, which: str) -> tuple[int, int] | None:
    tail = (
        completion.split("</think>", 1)[1] if "</think>" in completion else completion
    )
    offset = len(completion) - len(tail)
    ms = list(re.finditer(r"\b(yes|no)\b", tail, re.IGNORECASE))
    if not ms:
        return None
    m = ms[0] if which == "first" else ms[-1]
    return offset + m.start(), offset + m.end()


def locate(tok, full_text: str, abs_start: int, abs_end: int, consistent_label: str):
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
    token_start_char = s
    prefix_text = full_text[:token_start_char]
    leading_ws = full_text[token_start_char:abs_start]
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
    oth_stable = ids_with_other[: len(ids_prefix)] == ids_prefix
    oth_added = ids_with_other[len(ids_prefix) :]
    if not (oth_stable and len(oth_added) == 1):
        return None
    other_token_id = oth_added[0]
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


def main() -> None:
    payload = json.load(open("exp3_gpu_payload.json"))
    examples, prompts_by_key = (
        payload["examples"],
        {(p["example_id"], p["interface"]): p for p in payload["prompts"]},
    )
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
    print(f"loaded. num_hidden_layers={n_layers}", flush=True)

    results, table_rows = [], []
    for ex_id, ex in examples.items():
        consistent_label = ex["cot_supported_answer"]
        for iface in ("original", "polarity"):
            completion = gens[(ex_id, iface)]
            p = prompts_by_key[(ex_id, iface)]
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = chat_text + completion
            prompt_len = len(tok(chat_text, add_special_tokens=False)["input_ids"])

            first_span, last_span = (
                find_answer_span(completion, "first"),
                find_answer_span(completion, "last"),
            )
            self_correction = (
                first_span is not None
                and last_span is not None
                and completion[first_span[0] : first_span[1]].upper()
                != completion[last_span[0] : last_span[1]].upper()
            )

            locs = {}
            for pos_name, span in (("post_think", first_span), ("final", last_span)):
                if span is None:
                    continue
                start, end = span
                abs_start, abs_end = len(chat_text) + start, len(chat_text) + end
                loc = locate(tok, full_text, abs_start, abs_end, consistent_label)
                if loc is not None:
                    locs[pos_name] = loc

            if not locs:
                print(f"{ex['qid'][:12]}.../{iface}: no usable positions -- skipping")
                continue

            max_pre_index = max(l["pre_index"] for l in locs.values())
            n_new = (
                max_pre_index - prompt_len + 5
            )  # small buffer past the later target position
            prompt_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(input_device)

            print(
                f"\n{ex['qid'][:12]}.../{iface}: regenerating {n_new} tokens (prompt_len={prompt_len}) ...",
                flush=True,
            )
            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=n_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            regen_ids = out.sequences[0][prompt_len:].tolist()
            recorded_ids = tok(completion, add_special_tokens=False)["input_ids"][
                : len(regen_ids)
            ]
            if regen_ids != recorded_ids:
                print(
                    "  *** REGENERATION MISMATCH -- text differs from the recorded completion in this range ***"
                )
                print(f"  regen : {tok.decode(regen_ids)!r}")
                print(f"  record: {tok.decode(recorded_ids)!r}")
            else:
                print(
                    f"  regeneration matches recorded completion exactly for these {len(regen_ids)} tokens -- OK"
                )

            hidden_states_per_step = (
                out.hidden_states
            )  # tuple, one entry per generated token

            def hidden_at(position: int, layer: int) -> torch.Tensor:
                k = position - prompt_len + 1
                if k == 0:
                    return hidden_states_per_step[0][layer][0, -1, :]
                return hidden_states_per_step[k][layer][0, 0, :]

            row_deltas = {}
            for pos_name, loc in locs.items():
                deltas = []
                with torch.no_grad():
                    for l in range(n_layers + 1):
                        hs = hidden_at(loc["pre_index"], l).to(
                            final_norm.weight.device, dtype=final_norm.weight.dtype
                        )
                        # hidden_states[-1] (l == n_layers) is already post-final-norm for
                        # this model -- confirmed via exact match against outputs.scores.
                        # Re-applying final_norm there double-normalizes. See exp3_debug_postthink.py.
                        normed = hs if l == n_layers else final_norm(hs)
                        logits = lm_head(normed)
                        deltas.append(
                            (
                                logits[loc["consistent_token_id"]]
                                - logits[loc["inconsistent_token_id"]]
                            )
                            .float()
                            .item()
                        )
                    full_logits = lm_head(
                        hidden_at(loc["pre_index"], n_layers).to(
                            final_norm.weight.device, dtype=final_norm.weight.dtype
                        )
                    )
                    argmax_id = int(full_logits.argmax().item())
                assertion2_ok = argmax_id == loc["real_token_id"]
                print(
                    f"  {pos_name}: pre_index={loc['pre_index']}  Delta_final_layer={deltas[-1]:+.3f}  "
                    f"assertion2={'OK' if assertion2_ok else '*** STILL FAILS ***'}"
                )
                row_deltas[pos_name] = deltas[-1]
                results.append(
                    {
                        "example_id": ex_id,
                        "qid": ex["qid"],
                        "condition": iface,
                        "position": pos_name,
                        "n_layers": n_layers,
                        "pre_index": loc["pre_index"],
                        "deltas": deltas,
                        "assertion2_ok": assertion2_ok,
                    }
                )
                open("exp3_incremental_probe.jsonl", "w").write(
                    "".join(json.dumps(r) + "\n" for r in results)
                )

            table_rows.append(
                {
                    "qid": ex["qid"][:8],
                    "condition": iface,
                    "delta_post_think": row_deltas.get("post_think"),
                    "delta_final": row_deltas.get("final"),
                    "self_correction": self_correction,
                }
            )

    print("\n" + "=" * 90)
    print(
        f"{'example':10}{'condition':10}{'Delta post-think':>18}{'Delta final':>14}{'self-correction?':>20}"
    )
    for r in table_rows:
        pt = (
            f"{r['delta_post_think']:+.2f}"
            if r["delta_post_think"] is not None
            else "  -"
        )
        fn = f"{r['delta_final']:+.2f}" if r["delta_final"] is not None else "  -"
        print(
            f"{r['qid']:10}{r['condition']:10}{pt:>18}{fn:>14}{str(r['self_correction']):>20}"
        )

    n_fail = sum(1 for r in results if not r["assertion2_ok"])
    print(
        f"\nassertion 2 (incremental/cached path): {len(results)-n_fail}/{len(results)} passed"
        + (f"  -- {n_fail} STILL FAILED" if n_fail else "")
    )
    print("-> exp3_incremental_probe.jsonl")


if __name__ == "__main__":
    main()
