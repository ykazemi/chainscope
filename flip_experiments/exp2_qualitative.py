import argparse
import json
import random
import re
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"


def last_sentences(text: str, n: int) -> str:
    # drop a trailing short "Answer: X" style line, then take the last n sentences
    body = re.sub(
        r"\n\**\s*(final answer|answer)\s*[:\-]?\s*\**\s*(YES|NO)\.?\**\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    sents = re.split(r"(?<=[.!?])\s+", body.replace("\n", " "))
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[-n:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    examples = {
        r["example_id"]: r
        for r in map(
            json.loads, (OUT_DIR / "exp2_examples.jsonl").read_text().splitlines()
        )
        if r["kind"] == "flip"
    }
    semantic_full = {
        r["example_id"]: r
        for r in map(
            json.loads, (OUT_DIR / "exp2_scored.jsonl").read_text().splitlines()
        )
        if r["kind"] == "flip" and r["interface"] == "semantic" and r["cut"] == "full"
    }

    ids = sorted(examples)  # fixed order before sampling
    picked = random.Random(args.seed).sample(ids, min(args.n, len(ids)))

    lines = [
        f"Exp 2 qualitative spot-check — {len(picked)} of {len(ids)} mapping-failures, "
        f"random seed={args.seed} (not selected for effect)\n"
    ]
    for i, ex_id in enumerate(picked, 1):
        r = examples[ex_id]
        s = semantic_full.get(ex_id)
        q = r["q_str"].strip().splitlines()[-1].strip()
        tail = last_sentences(r["response_str"], 3)
        sem_out = (
            s["completion"].split("</think>")[-1].strip()
            if s
            else "(no semantic/full trial)"
        )
        sem_map = s["mapped_answer"] if s else None
        consistent = s["cot_consistent"] if s else None
        lines.append(f"[{i}] {q}")
        lines.append(f"    reasoning (last ~3 sentences): {tail}")
        lines.append(
            f"    original answer: {r['orig_model_answer']}   "
            f"(CoT-supported answer: {r['cot_supported_answer']})"
        )
        lines.append(
            f"    semantic re-elicit: {sem_out!r}  ->  mapped {sem_map}   "
            f"[{'CONSISTENT' if consistent else 'still inconsistent' if consistent is False else 'unparsed'}]"
        )
        lines.append("")

    out = OUT_DIR / "exp2_qualitative.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
