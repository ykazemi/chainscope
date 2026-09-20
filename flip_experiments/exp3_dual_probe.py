import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"
PAYLOAD = "exp3_gpu_payload.json"
GENERATIONS = "exp3_local_generations.jsonl"


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
    """Return None if not single-token here, else a dict with everything needed
    to compute Delta_l at this position (prefix text, position index, both
    token ids), independently re-deriving the (possibly whitespace-fused)
    token ids at THIS specific position rather than assuming they match
    another position in the same completion."""
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
        "observed_label": observed_label,
        "consistent_token_id": consistent_token_id,
        "inconsistent_token_id": inconsistent_token_id,
    }


def main() -> None:
    payload = json.load(open(PAYLOAD))
    examples, prompts_by_key = (
        payload["examples"],
        {(p["example_id"], p["interface"]): p for p in payload["prompts"]},
    )
    gens = {
        (g["example_id"], g["condition"]): g["completion"]
        for g in (json.loads(l) for l in open(GENERATIONS))
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

    results = []
    table_rows = []
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

            first_span = find_answer_span(completion, "first")
            last_span = find_answer_span(completion, "last")
            self_correction = (
                first_span is not None
                and last_span is not None
                and completion[first_span[0] : first_span[1]].upper()
                != completion[last_span[0] : last_span[1]].upper()
            )

            row_deltas = {}
            for pos_name, span in (("post_think", first_span), ("final", last_span)):
                tag = f"{ex['qid'][:12]}.../{iface}/{pos_name}"
                if span is None:
                    print(f"{tag}: no span found -- skipping")
                    continue
                start, end = span
                abs_start, abs_end = len(chat_text) + start, len(chat_text) + end
                loc = locate(tok, full_text, abs_start, abs_end, consistent_label)
                if loc is None:
                    print(
                        f"{tag}: multi-token or unresolvable at this position -- skipping"
                    )
                    continue

                input_ids = tok(
                    loc["prefix_text"], return_tensors="pt", add_special_tokens=False
                )["input_ids"].to(input_device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                hidden_states = out.hidden_states
                assert len(hidden_states) == n_layers + 1

                deltas = []
                with torch.no_grad():
                    for layer_idx, h in enumerate(hidden_states):
                        hs = h[0, loc["pre_index"], :].to(
                            final_norm.weight.device, dtype=final_norm.weight.dtype
                        )
                        # hidden_states[-1] (layer_idx == n_layers) is already post-final-norm
                        # for this model -- confirmed via exact match against outputs.scores.
                        # Re-applying final_norm there double-normalizes. See exp3_debug_postthink.py.
                        normed = hs if layer_idx == n_layers else final_norm(hs)
                        logits = lm_head(normed)
                        deltas.append(
                            (
                                logits[loc["consistent_token_id"]]
                                - logits[loc["inconsistent_token_id"]]
                            )
                            .float()
                            .item()
                        )

                    # assertion 2: full-vocabulary argmax at this position == real next token
                    full_logits = lm_head(
                        hidden_states[-1][0, loc["pre_index"], :].to(
                            final_norm.weight.device, dtype=final_norm.weight.dtype
                        )
                    )
                    argmax_id = int(full_logits.argmax().item())
                assertion2_ok = argmax_id == loc["real_token_id"]

                print(
                    f"{tag}: pre_index={loc['pre_index']}  Delta_final_layer={deltas[-1]:+.3f}  "
                    f"assertion2(argmax==real_next_token)={'OK' if assertion2_ok else '*** FAIL ***'}"
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
                        "consistent_token_id": int(loc["consistent_token_id"]),
                        "inconsistent_token_id": int(loc["inconsistent_token_id"]),
                        "deltas": deltas,
                        "assertion2_ok": assertion2_ok,
                    }
                )
                open("exp3_dual_probe.jsonl", "w").write(
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
        f"\nassertion 2: {len(results)-n_fail}/{len(results)} passed"
        + (f"  -- {n_fail} FAILED, do not trust those cells" if n_fail else "")
    )
    print("-> exp3_dual_probe.jsonl")


if __name__ == "__main__":
    main()
