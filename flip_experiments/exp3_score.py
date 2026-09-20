import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp2_score import score_row  # reuse the exact same parser — no re-implementation

OUT_DIR = Path(__file__).parent / "data"


def main() -> None:
    src = OUT_DIR / "exp3_completions.jsonl"
    by_id = {}
    for l in src.read_text().splitlines():
        d = json.loads(l)
        cur = by_id.get(d["trial_id"])
        if cur is None or str(cur["completion"]).startswith("<<ERROR"):
            by_id[d["trial_id"]] = d
    rows = [score_row(d) for d in by_id.values()]
    errs = sum(1 for d in by_id.values() if str(d["completion"]).startswith("<<ERROR"))
    if errs:
        print(f"WARNING: {errs} trials still have errors (excluded below)\n")
    (OUT_DIR / "exp3_scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    by_example = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_example[r["example_id"]][r["interface"]].append(r)

    fieldnames = [
        "example_id",
        "qid",
        "k",
        "orig_n_inconsistent",
        "orig_n_parsed",
        "pol_n_consistent",
        "pol_n_parsed",
        "passes_frozen_rule",
    ]
    out_rows = []
    for ex_id, by_iface in sorted(by_example.items()):
        orig = by_iface.get("original", [])
        pol = by_iface.get("polarity", [])
        k = max(len(orig), len(pol))
        orig_inconsistent = sum(
            1 for r in orig if r["parsed"] and r["cot_consistent"] is False
        )
        orig_parsed = sum(1 for r in orig if r["parsed"])
        pol_consistent = sum(
            1 for r in pol if r["parsed"] and r["cot_consistent"] is True
        )
        pol_parsed = sum(1 for r in pol if r["parsed"])
        passes = (
            len(orig) == k
            and len(pol) == k
            and k > 0
            and orig_inconsistent == k
            and pol_consistent == k
        )
        out_rows.append(
            {
                "example_id": ex_id,
                "qid": orig[0]["qid"] if orig else pol[0]["qid"],
                "k": k,
                "orig_n_inconsistent": orig_inconsistent,
                "orig_n_parsed": orig_parsed,
                "pol_n_consistent": pol_consistent,
                "pol_n_parsed": pol_parsed,
                "passes_frozen_rule": passes,
            }
        )

    with (OUT_DIR / "exp3_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    n_pass = sum(r["passes_frozen_rule"] for r in out_rows)
    print(f"{len(out_rows)} mapping-failure examples resampled")
    print(
        f"{n_pass} pass the frozen rule (original K/K inconsistent AND polarity K/K consistent)"
    )
    print("-> data/exp3_candidates.csv")
    if n_pass:
        print("\ncandidates:")
        for r in out_rows:
            if r["passes_frozen_rule"]:
                print(f"  {r['example_id']}  (qid {r['qid'][:12]}..., k={r['k']})")


if __name__ == "__main__":
    main()
