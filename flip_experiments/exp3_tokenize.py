import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp2_score import parse_yn  # same last-match parser used everywhere else

OUT_DIR = Path(__file__).parent / "data"
CANDIDATES = [
    "4367330b55f72ace04b09a9addab85fcfad35f1cde4748b0191eb6bbeee83cbe/851e092b-4067-4e5c-91d2-e9dfb63d9ac4",
    "7729c35854ca897f19a2a6f820353b45eea2f46f3faa1a1401d3583940a0b47a/b1832d6d-d44d-4add-b72e-502f0f88953b",
    "f1ac9bfbe98d4a47106c269ff3ebd482c0e8d52390bb9eeadcf33ff7d9da3120/fe4386a4-78da-4924-a081-3719ed84e4d9",
]


def mirror_case(word: str, template: str) -> str:
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word.capitalize()


def find_answer_span(completion: str) -> tuple[int, int] | None:
    """Char span of the LAST yes/no match after </think>, as an absolute offset
    into `completion` (mirrors exp2_score.parse_yn's selection exactly). This is
    the bare word span (no leading whitespace) -- the real token boundary
    (whether whitespace fuses into the token or not) is resolved separately via
    the tokenizer's own offset mapping on the real text, not guessed here."""
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
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")

    examples = {
        json.loads(l)["example_id"]: json.loads(l)
        for l in (OUT_DIR / "exp2_examples.jsonl").read_text().splitlines()
    }
    prompts = [
        json.loads(l)
        for l in (OUT_DIR / "exp2_prompts.jsonl").read_text().splitlines()
        if json.loads(l)["cut"] == "think"
        and json.loads(l)["interface"] in ("original", "polarity")
    ]
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in prompts}
    completions_by_id = {}
    for l in (OUT_DIR / "exp2_completions.jsonl").read_text().splitlines():
        d = json.loads(l)
        completions_by_id[d["trial_id"]] = d["completion"]

    out_rows = []
    for ex_id in CANDIDATES:
        ex = examples[ex_id]
        for iface in ("original", "polarity"):
            p = prompts_by_key[(ex_id, iface)]
            trial_id = f'{ex["response_id"][:8]}_think_{iface}'
            completion = completions_by_id.get(trial_id)
            print(
                f"\n{'='*70}\n{ex['qid'][:12]}... / {iface}   (greedy trial_id={trial_id})"
            )

            if completion is None:
                print("  MISSING greedy completion for this trial_id -- skipping")
                out_rows.append(
                    {
                        "example_id": ex_id,
                        "condition": iface,
                        "status": "missing_greedy_completion",
                    }
                )
                continue

            post_think = (
                completion.split("</think>", 1)[1]
                if "</think>" in completion
                else completion
            )
            greedy_answer = parse_yn(post_think)
            want_consistent = iface == "polarity"
            got_consistent = greedy_answer == ex["cot_supported_answer"]
            baseline_ok = got_consistent == want_consistent
            print(
                f"  cot_supported_answer={ex['cot_supported_answer']}  greedy_parsed={greedy_answer}"
                f"  want_consistent={want_consistent}  got_consistent={got_consistent}"
                f"  {'OK' if baseline_ok else '*** BASELINE MISMATCH ***'}"
            )

            if not baseline_ok:
                out_rows.append(
                    {
                        "example_id": ex_id,
                        "qid": ex["qid"],
                        "condition": iface,
                        "status": "fails_greedy_baseline_check",
                        "cot_supported_answer": ex["cot_supported_answer"],
                        "greedy_parsed_answer": greedy_answer,
                    }
                )
                continue

            span = find_answer_span(completion)
            if span is None:
                print("  no yes/no match found in completion post-</think> -- skipping")
                out_rows.append(
                    {
                        "example_id": ex_id,
                        "condition": iface,
                        "status": "no_answer_span_found",
                    }
                )
                continue
            start, end = span  # bare word span, no leading whitespace
            observed_word = completion[start:end]
            chat_prefix = tok.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = chat_prefix + completion
            abs_start, abs_end = len(chat_prefix) + start, len(chat_prefix) + end

            # Tokenize the REAL text once and use its own offset mapping to find
            # which token(s) the real answer word actually landed in -- no
            # guessing about whether whitespace fuses into the token or not.
            enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
            ids_full, offsets = enc["input_ids"], enc["offset_mapping"]
            covering = [
                i for i, (s, e) in enumerate(offsets) if s < abs_end and e > abs_start
            ]
            answer_token_idx = covering[0] if covering else None
            real_single_token = (
                len(covering) == 1
                and offsets[answer_token_idx][0] <= abs_start
                and offsets[answer_token_idx][1] >= abs_end
            )

            consistent_label = ex["cot_supported_answer"]  # "YES"/"NO"
            inconsistent_label = "NO" if consistent_label == "YES" else "YES"
            observed_label = "YES" if observed_word.upper() == "YES" else "NO"
            other_label = (
                inconsistent_label
                if observed_label == consistent_label
                else consistent_label
            )

            if not real_single_token:
                print(
                    f"  answer word observed: {observed_word!r} -- spans {len(covering)} real tokens: "
                    f"{[tok.decode([ids_full[i]]) for i in covering]}  (NOT single-token here)"
                )
                out_rows.append(
                    {
                        "example_id": ex_id,
                        "qid": ex["qid"],
                        "condition": iface,
                        "status": "multi_token_answer",
                        "observed_word": observed_word,
                        "n_real_tokens": len(covering),
                    }
                )
                continue

            token_start_char = offsets[answer_token_idx][
                0
            ]  # true token start, whitespace-inclusive if fused
            real_token_id = ids_full[answer_token_idx]
            prefix_text = full_text[
                :token_start_char
            ]  # text up to the REAL token boundary
            leading_ws = full_text[
                token_start_char:abs_start
            ]  # whatever got fused in (e.g. "\n\n" or " " or "")
            other_word = leading_ws + mirror_case(other_label, observed_word)

            ids_prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
            ids_with_other = tok(prefix_text + other_word, add_special_tokens=False)[
                "input_ids"
            ]
            oth_stable = ids_with_other[: len(ids_prefix)] == ids_prefix
            oth_added = ids_with_other[len(ids_prefix) :]
            single_token = oth_stable and len(oth_added) == 1
            other_token_id = oth_added[0] if len(oth_added) == 1 else oth_added

            consistent_token_id = (
                real_token_id if observed_label == consistent_label else other_token_id
            )
            inconsistent_token_id = (
                other_token_id if observed_label == consistent_label else real_token_id
            )
            pre_answer_index = (
                len(ids_prefix) - 1
            )  # position whose next-token logits ARE the answer

            print(
                f"  answer word observed: {observed_word!r} (real token {real_token_id} = "
                f"{tok.decode([real_token_id])!r}, fused leading ws={leading_ws!r})"
                f"  [{observed_label}, ={'consistent' if observed_label==consistent_label else 'inconsistent'}]"
            )
            print(
                f"  counterfactual {other_label!r} -> stable={oth_stable} added={oth_added} "
                f"({'single token' if single_token else 'MULTI-TOKEN'})"
            )
            print(
                f"  prefix_len={len(ids_prefix)}  pre_answer_index={pre_answer_index}"
                f"  consistent_token_id={consistent_token_id}  inconsistent_token_id={inconsistent_token_id}"
            )

            decoded_tail = [tok.decode([t]) for t in ids_prefix[-25:]]
            print(f"  last 25 prefix tokens: {decoded_tail}")

            out_rows.append(
                {
                    "example_id": ex_id,
                    "qid": ex["qid"],
                    "condition": iface,
                    "status": "ok" if single_token else "multi_token_answer",
                    "prompt_length_tokens": len(
                        tok(chat_prefix, add_special_tokens=False)["input_ids"]
                    ),
                    "pre_answer_index": pre_answer_index,
                    "prefix_len": len(ids_prefix),
                    "consistent_answer": consistent_label,
                    "inconsistent_answer": inconsistent_label,
                    "consistent_token_id": consistent_token_id,
                    "inconsistent_token_id": inconsistent_token_id,
                    "answer_num_tokens_other": len(oth_added),
                    "observed_word": observed_word,
                    "leading_ws_fused": leading_ws,
                    "other_word": other_word,
                }
            )

    (OUT_DIR / "exp3_mech_metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in out_rows)
    )
    n_ok = sum(1 for r in out_rows if r.get("status") == "ok")
    print(f"\n{'='*70}\n{n_ok}/6 conditions: baseline OK and single-token confirmed")
    print("-> data/exp3_mech_metadata.jsonl")


if __name__ == "__main__":
    main()
