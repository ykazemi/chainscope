import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
CUT = "think"
IFACES = ("original", "polarity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="repeats per (example, interface)")
    args = ap.parse_args()

    all_rows = (
        json.loads(l) for l in (OUT_DIR / "exp2_prompts.jsonl").read_text().splitlines()
    )
    base = [
        r
        for r in all_rows
        if r["cut"] == CUT and r["interface"] in IFACES and r["kind"] == "flip"
    ]

    trials = []
    for r in base:
        for k in range(args.k):
            trials.append({**r, "trial_id": f'{r["trial_id"]}__r{k}', "resample_k": k})

    out = OUT_DIR / "exp3_prompts.jsonl"
    out.write_text("".join(json.dumps(t) + "\n" for t in trials))
    n_examples = len({r["example_id"] for r in base})
    print(
        f"{n_examples} mapping-failure examples x {len(IFACES)} interfaces x {args.k} repeats"
        f" = {len(trials)} trials"
    )
    print(f"-> {out}\nnext: python flip_experiments/exp3_run.py --backend openai")


if __name__ == "__main__":
    main()
