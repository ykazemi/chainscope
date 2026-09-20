import argparse
import json
import random
from pathlib import Path

import pandas as pd

from chainscope.typing import DATA_DIR

OUT_DIR = Path(__file__).parent / "data"
FLIP = "YES", "NO"


def _flip(a: str) -> str:
    return "NO" if a == "YES" else "YES"


def relation(q_str: str) -> str:
    q = q_str.lower()
    for probe, name in [
        ("fewer pages", "fewer-pages"),
        ("shorter total runtime", "shorter-runtime"),
        ("born earlier", "born-earlier"),
        ("born later", "born-later"),
        ("die at an earlier", "died-earlier"),
        ("die at a later", "died-later"),
        ("released earlier", "released-earlier"),
        ("released later", "released-later"),
        ("published later", "published-later"),
        ("published earlier", "published-earlier"),
        ("located south", "south-of"),
        ("located north", "north-of"),
    ]:
        if probe in q:
            return name
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-controls", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = {
        (r["qid"], r["response_id"]): r["my_label"]
        for r in __import__("csv").DictReader((OUT_DIR / "audit_labels.csv").open())
    }
    flips = [
        json.loads(l)
        for l in (OUT_DIR / "flip_candidates.jsonl").read_text().splitlines()
    ]
    ctrls = [
        json.loads(l)
        for l in (OUT_DIR / "nonflip_controls.jsonl").read_text().splitlines()
    ]

    df = pd.read_pickle(DATA_DIR / "df-wm-non-ambiguous-hard-2.pkl.gz")
    df = df[df.model_id == "qwen/qwq-32b"]
    names = {
        r.qid: (r.x_name, r.y_name, r.x_value, r.y_value)
        for r in df.itertuples()
        if r.qid
    }

    def enrich(r: dict, kind: str, cot_ans: str) -> dict | None:
        if r["qid"] not in names:
            return None
        xn, yn, xv, yv = names[r["qid"]]
        return {
            "example_id": f'{r["qid"]}/{r["response_id"]}',
            "qid": r["qid"],
            "response_id": r["response_id"],
            "kind": kind,
            "prop_id": r["prop_id"],
            "q_str": r["q_str"].strip(),
            "x_name": xn,
            "y_name": yn,
            "x_value": xv,
            "y_value": yv,
            "relation": relation(r["q_str"]),
            "correct_answer": r["correct_answer"],
            "orig_model_answer": r["model_answer"],
            "cot_supported_answer": cot_ans,
            "prompt_str": r["prompt_str"],
            "response_str": r["response_str"],
        }

    # flips = the 40 mapping-failures; CoT points opposite the emitted label
    mf = [
        r
        for r in flips
        if labels.get((r["qid"], r["response_id"])) == "mapping-failure"
    ]
    assert all(
        r["model_answer"] in FLIP for r in mf
    ), "a mapping-failure has a non-YES/NO emitted label — check by hand"
    flip_rows = [e for r in mf if (e := enrich(r, "flip", _flip(r["model_answer"])))]

    # matched controls: same questions as the flips, clean YES/NO, CoT == answer
    flip_qids = {r["qid"] for r in flip_rows}
    pool = [r for r in ctrls if r["qid"] in flip_qids and r["model_answer"] in FLIP]
    by_qid: dict[str, list] = {}
    for r in pool:
        by_qid.setdefault(r["qid"], []).append(r)
    rng = random.Random(args.seed)
    ctrl_rows = []
    for v in by_qid.values():
        pick = rng.choice(v)  # one control response per question
        e = enrich(
            pick, "control", pick["model_answer"]
        )  # CoT-supported == its own answer
        if e:
            ctrl_rows.append(e)
    rng.shuffle(ctrl_rows)
    ctrl_rows = ctrl_rows[: args.n_controls]

    rows = flip_rows + ctrl_rows
    (OUT_DIR / "exp2_examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    from collections import Counter

    print(
        f"flips (mapping-failures): {len(flip_rows)}   matched controls: {len(ctrl_rows)}"
    )
    print(
        f"  flip cot_supported_answer: {dict(Counter(r['cot_supported_answer'] for r in flip_rows))}"
    )
    print(f"  relations: {dict(Counter(r['relation'] for r in rows))}")
    miss = [r for r in mf if r["qid"] not in names]
    if miss:
        print(f"  WARNING: {len(miss)} flip qids missing from df (no x/y names)")
    print(f"-> data/exp2_examples.jsonl  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
