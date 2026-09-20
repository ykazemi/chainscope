import csv
import json
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
CATS = ["hidden-disagreement", "mapping-failure", "eval-false-positive", "ambiguous"]
COLORS = {
    "hidden-disagreement": "#c0392b",
    "mapping-failure": "#e67e22",
    "eval-false-positive": "#7f8c8d",
    "ambiguous": "#bdc3c7",
}


def main() -> None:
    cand = {
        (r["qid"], r["response_id"]): r
        for r in map(
            json.loads, (OUT_DIR / "flip_candidates.jsonl").read_text().splitlines()
        )
    }

    rows = list(csv.DictReader((OUT_DIR / "audit_labels.csv").open()))
    labelled = [r for r in rows if r["my_label"].strip()]
    if not labelled:
        print("no labels filled in yet — edit data/audit_labels.csv (my_label column)")
        return

    counts = Counter(r["my_label"].strip() for r in labelled)
    n = len(labelled)
    unknown = sorted(set(counts) - set(CATS))
    print(f"labelled {n}/{len(rows)}\n")
    for lab in CATS + unknown:
        c = counts.get(lab, 0)
        flag = "  <- unrecognised label" if lab in unknown else ""
        print(f"  {lab:20s} {c:3d}  ({c / n:5.1%}){flag}")

    print("\nby model's parsed answer:")
    for lab in CATS:
        if counts.get(lab):
            ma = Counter(
                r["model_answer"] for r in labelled if r["my_label"].strip() == lab
            )
            print(f"  {lab:20s} " + "  ".join(f"{k}={v}" for k, v in ma.items()))

    exp2 = [
        cand[(r["qid"], r["response_id"])]
        for r in labelled
        if r["my_label"].strip() in {"hidden-disagreement", "mapping-failure"}
        if (r["qid"], r["response_id"]) in cand
    ]
    (OUT_DIR / "flips_for_exp2.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in exp2)
    )
    print(f"\n-> data/flips_for_exp2.jsonl  ({len(exp2)} rows)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = [counts.get(l, 0) for l in CATS]
        fig, ax = plt.subplots(figsize=(6, 3.3))
        bars = ax.bar(range(len(CATS)), vals, color=[COLORS[l] for l in CATS])
        ax.set_xticks(range(len(CATS)))
        ax.set_xticklabels([l.replace("-", "-\n") for l in CATS], fontsize=9)
        ax.set_ylabel("responses")
        ax.set_title(f"Audit of answer_flipping=YES  (n={n}, qwq-32b)")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                str(v),
                ha="center",
                fontweight="bold",
            )
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "exp0_summary.png", dpi=150)
        print("-> data/exp0_summary.png")
    except ImportError:
        print("(matplotlib not available — skipped chart)")


if __name__ == "__main__":
    main()
