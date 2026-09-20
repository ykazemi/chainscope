import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"
MAX_NEW_TOKENS = 4096
PAYLOAD = "exp3_gpu_payload.json"


def mirror_case(word: str, template: str) -> str:
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word.capitalize()


def parse_yn(text: str) -> str | None:
    ms = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
    return ms[-1].group(1).upper() if ms else None


def find_answer_span(completion: str) -> tuple[int, int] | None:
    tail = (
        completion.split("</think>", 1)[1] if "</think>" in completion else completion
    )
    offset = len(completion) - len(tail)
    ms = list(re.finditer(r"\b(yes|no)\b", tail, re.IGNORECASE))
    if not ms:
        return None
    m = ms[-1]
    return offset + m.start(), offset + m.end()


def main() -> None:
    payload = json.load(open(PAYLOAD))
    examples = payload["examples"]
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in payload["prompts"]}
    candidates = list(examples.keys())
    assert (
        len(candidates) == 2
    ), f"expected exactly 2 frozen candidates, got {len(candidates)}"

    print(f"loading {MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    input_device = model.get_input_embeddings().weight.device
    final_norm, lm_head = model.model.norm, model.lm_head
    print(
        f"loaded. num_hidden_layers={n_layers}  input_device={input_device}"
        f"  norm/head device={final_norm.weight.device}",
        flush=True,
    )

    # reuse already-verified completions if present -- avoids paying for a full
    # regeneration just to recompute logits (this script's fix only changes how
    # hidden_states are turned into logits, not the generation itself, and
    # do_sample=False makes these deterministic/already-verified anyway)
    cached_completions = {}
    try:
        for l in open("exp3_local_generations.jsonl"):
            d = json.loads(l)
            cached_completions[(d["example_id"], d["condition"])] = d["completion"]
        print(
            f"loaded {len(cached_completions)} cached completions from exp3_local_generations.jsonl "
            "-- will reuse instead of regenerating",
            flush=True,
        )
    except FileNotFoundError:
        pass

    generations, results = [], []
    for ex_id in candidates:
        ex = examples[ex_id]
        for iface in ("original", "polarity"):
            p = prompts_by_key[(ex_id, iface)]
            tag = f"{ex['qid'][:12]}.../{iface}"
            print(f"\n=== {tag} ===", flush=True)

            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tok(chat_text, return_tensors="pt", add_special_tokens=False).to(
                input_device
            )

            if (ex_id, iface) in cached_completions:
                completion = cached_completions[(ex_id, iface)]
                print("  reusing cached completion (no regeneration)")
            else:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id or tok.pad_token_id,
                    )
                completion = tok.decode(
                    out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
                )
            generations.append(
                {"example_id": ex_id, "condition": iface, "completion": completion}
            )
            open("exp3_local_generations.jsonl", "w").write(
                "".join(json.dumps(r) + "\n" for r in generations)
            )

            post_think = (
                completion.split("</think>", 1)[1]
                if "</think>" in completion
                else completion
            )
            local_answer = parse_yn(post_think)
            consistent_label = ex["cot_supported_answer"]
            want_consistent = iface == "polarity"
            got_consistent = local_answer == consistent_label
            baseline_ok = got_consistent == want_consistent
            print(
                f"  local greedy parsed={local_answer}  want_consistent={want_consistent} "
                f"got_consistent={got_consistent}  {'OK' if baseline_ok else '*** FAILS LOCALLY -- DROPPING ***'}"
            )
            if not baseline_ok:
                results.append(
                    {
                        "example_id": ex_id,
                        "condition": iface,
                        "status": "fails_local_greedy_baseline",
                    }
                )
                continue

            span = find_answer_span(completion)
            if span is None:
                print("  no answer span found -- dropping")
                results.append(
                    {
                        "example_id": ex_id,
                        "condition": iface,
                        "status": "no_answer_span",
                    }
                )
                continue
            start, end = span
            observed_word = completion[start:end]
            full_text = chat_text + completion
            abs_start, abs_end = len(chat_text) + start, len(chat_text) + end

            enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
            ids_full, offsets = enc["input_ids"], enc["offset_mapping"]
            covering = [
                i for i, (s, e) in enumerate(offsets) if s < abs_end and e > abs_start
            ]
            if len(covering) != 1 or not (
                offsets[covering[0]][0] <= abs_start
                and offsets[covering[0]][1] >= abs_end
            ):
                print(
                    f"  answer spans {len(covering)} real tokens -- multi-token, dropping"
                )
                results.append(
                    {
                        "example_id": ex_id,
                        "condition": iface,
                        "status": "multi_token_answer",
                    }
                )
                continue
            idx = covering[0]
            real_token_id = ids_full[idx]
            token_start_char = offsets[idx][0]
            prefix_text = full_text[:token_start_char]
            leading_ws = full_text[token_start_char:abs_start]

            inconsistent_label = "NO" if consistent_label == "YES" else "YES"
            observed_label = "YES" if observed_word.upper() == "YES" else "NO"
            other_label = (
                inconsistent_label
                if observed_label == consistent_label
                else consistent_label
            )
            other_word = leading_ws + mirror_case(other_label, observed_word)

            ids_prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
            ids_with_other = tok(prefix_text + other_word, add_special_tokens=False)[
                "input_ids"
            ]
            oth_stable = ids_with_other[: len(ids_prefix)] == ids_prefix
            oth_added = ids_with_other[len(ids_prefix) :]
            if not (oth_stable and len(oth_added) == 1):
                print(
                    f"  counterfactual {other_label!r} not single-token here -- dropping"
                )
                results.append(
                    {
                        "example_id": ex_id,
                        "condition": iface,
                        "status": "counterfactual_multi_token",
                    }
                )
                continue
            other_token_id = oth_added[0]

            consistent_token_id = (
                real_token_id if observed_label == consistent_label else other_token_id
            )
            inconsistent_token_id = (
                other_token_id if observed_label == consistent_label else real_token_id
            )
            pre_answer_index = len(ids_prefix) - 1
            print(
                f"  pre_answer_index={pre_answer_index}  consistent_token_id={consistent_token_id} "
                f"inconsistent_token_id={inconsistent_token_id}"
            )

            input_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(input_device)
            with torch.no_grad():
                fout = model(input_ids, output_hidden_states=True)
            hidden_states = fout.hidden_states
            assert len(hidden_states) == n_layers + 1

            deltas = []
            with torch.no_grad():
                for layer_idx, h in enumerate(hidden_states):
                    hs = h[0, pre_answer_index, :].to(
                        final_norm.weight.device, dtype=final_norm.weight.dtype
                    )
                    # hidden_states[-1] (layer_idx == n_layers) already has the final norm
                    # applied by this model/transformers version -- confirmed by an exact
                    # 0.000000 match against ground-truth outputs.scores when norm is NOT
                    # re-applied there. Re-applying it is a double-normalization bug.
                    normed = hs if layer_idx == n_layers else final_norm(hs)
                    logits = lm_head(normed)
                    deltas.append(
                        (logits[consistent_token_id] - logits[inconsistent_token_id])
                        .float()
                        .item()
                    )

            expected_positive = want_consistent
            sign_ok = (deltas[-1] > 0) == expected_positive
            print(
                f"  final-layer Delta={deltas[-1]:+.3f}  expected {'positive' if expected_positive else 'negative'}"
                f"  {'OK' if sign_ok else '*** SIGN MISMATCH -- CHECK PIPELINE ***'}"
            )

            results.append(
                {
                    "example_id": ex_id,
                    "qid": ex["qid"],
                    "condition": iface,
                    "status": "ok",
                    "n_layers": n_layers,
                    "pre_answer_index": pre_answer_index,
                    "consistent_token_id": int(consistent_token_id),
                    "inconsistent_token_id": int(inconsistent_token_id),
                    "deltas": deltas,
                    "final_delta_sign_ok": sign_ok,
                }
            )
            open("exp3_logit_lens.jsonl", "w").write(
                "".join(json.dumps(r) + "\n" for r in results)
            )

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{n_ok}/4 conditions produced a usable Delta_l curve")
    if n_ok == 0:
        print("nothing to plot")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {0: "#c0392b", 1: "#2980b9"}
    styles = {"original": "--", "polarity": "-"}
    for i, ex_id in enumerate(candidates):
        for r in results:
            if r["example_id"] == ex_id and r["status"] == "ok":
                L = r["n_layers"]
                xs = [l / L for l in range(L + 1)]
                ax.plot(
                    xs,
                    r["deltas"],
                    styles[r["condition"]],
                    color=colors[i],
                    label=f"{r['qid'][:8]}.../{r['condition']}",
                )
    ax.axhline(0, color="#999", lw=1)
    ax.axvline(0.75, color="#999", lw=1, ls=":")
    ax.text(
        0.755,
        ax.get_ylim()[1] * 0.9,
        "pre-registered\n'late' cutoff (75%)",
        fontsize=8,
        color="#666",
    )
    ax.set_xlabel("normalized layer depth (l / L)")
    ax.set_ylabel(r"$\Delta_\ell$ = logit(consistent) $-$ logit(inconsistent)")
    ax.set_title("Exp 3: layerwise logit-lens localization (QwQ-32B)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig("exp3_logit_lens.png", dpi=150)
    print("-> exp3_logit_lens.png")


if __name__ == "__main__":
    main()
