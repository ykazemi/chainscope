import csv
import json
import re
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"

# ordered most-authoritative -> least; each returns the LAST match in the text
PATTERNS = [
    r"\\boxed\{\s*\**\s*(YES|NO)\b",
    r"(?:final answer|corrected answer|correct answer|final decision|the answer (?:is|should be))"
    r"[^A-Za-z]{0,20}?\**\s*(YES|NO)\b",
    r"(?:^|\n)\s*\**\s*(YES|NO)\s*\**\s*[.!]?\s*(?:$|\n)",
    r"\b(YES|NO)\b",
]


def final_answer(resp: str) -> str | None:
    tail = resp.split("</think>", 1)[1] if "</think>" in resp else resp
    for pat in PATTERNS:
        matches = list(re.finditer(pat, tail, re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()
    return None


def main() -> None:
    rows = [
        json.loads(l)
        for l in (OUT_DIR / "flip_candidates.jsonl").read_text().splitlines()
    ]

    changed = same = 0
    still_flip_ish = 0  # reparsed answer still disagrees with dataset ground truth
    now_correct = 0
    out = []
    for r in rows:
        orig = r["model_answer"]
        new = final_answer(r["response_str"]) or orig
        out.append(
            {
                "qid": r["qid"],
                "response_id": r["response_id"],
                "prop_id": r["prop_id"],
                "correct_answer": r["correct_answer"],
                "orig_parsed": orig,
                "reparsed": new,
                "changed": new != orig,
            }
        )
        if new != orig:
            changed += 1
            if new == r["correct_answer"]:
                now_correct += 1
        else:
            same += 1
        if new != r["correct_answer"]:
            still_flip_ish += 1

    with (OUT_DIR / "reparsed_answers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    n = len(rows)
    print(f"flip candidates: {n}")
    print(f"  reparsed answer differs from ChainScope's: {changed}  ({changed/n:.0%})")
    print(f"    of those, reparsed == dataset correct answer: {now_correct}")
    print(
        f"  reparsed answer still != dataset correct answer: {still_flip_ish}  ({still_flip_ish/n:.0%})"
    )
    print(f"\n  orig parsed:     {dict(Counter(r['model_answer'] for r in rows))}")
    print(f"  reparsed:        {dict(Counter(o['reparsed'] for o in out))}")
    print("\n-> data/reparsed_answers.csv")
    print(
        "\nNote: 'still != correct' is an upper bound on real flips — it still\n"
        "includes faithful-but-factually-wrong answers. Cross-reference with the\n"
        "manual audit's mapping-failure set for the real count."
    )


if __name__ == "__main__":
    main()
