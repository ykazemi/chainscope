import argparse
import csv
import html
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
LABELS = ["hidden-disagreement", "mapping-failure", "eval-false-positive", "ambiguous"]

CSS = """
body{max-width:60rem;margin:0 auto;padding:2rem 1.5rem;line-height:1.5;
     font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1a1a1a}
h1{font-size:1.4rem} .meta{color:#555;font-size:.9rem}
.card{border:1px solid #ddd;border-radius:8px;padding:1.1rem 1.3rem;margin:1.6rem 0}
.idx{font-weight:700;font-size:1.1rem;background:#111;color:#fff;border-radius:5px;
     padding:.1rem .5rem;margin-right:.5rem}
.q{font-weight:600;margin:.6rem 0}
.kv{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;font-size:.9rem;margin:.5rem 0}
.tag{background:#eef;border-radius:4px;padding:.05rem .45rem}
.spoiler{background:#f7f7f2;border-left:3px solid #c8a;padding:.5rem .8rem;margin:.7rem 0;font-size:.9rem}
details{margin:.6rem 0} summary{cursor:pointer;font-weight:600}
.cot{white-space:pre-wrap;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     background:#fbfbfb;border:1px solid #eee;border-radius:6px;padding:.8rem;max-height:34rem;overflow:auto}
.think{color:#666} .ans{background:#fff6d6;font-weight:600}
"""


def render_cot(resp: str) -> str:
    resp = resp.strip()
    if "</think>" in resp:
        think, _, tail = resp.partition("</think>")
        return (
            f'<span class="think">{html.escape(think.replace("<think>", "").strip())}</span>\n\n'
            f'<span class="ans">{html.escape(tail.strip())}</span>'
        )
    return html.escape(resp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--show-judge",
        action="store_true",
        help="reveal judge analysis + Claude draft label inline (not blind)",
    )
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in (OUT_DIR / "flip_candidates.jsonl").read_text().splitlines()
    ]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    print(
        f"sampled {len(sample)}/{len(rows)}  ({'SHOW-JUDGE' if args.show_judge else 'BLIND'})"
    )

    # blank template — never overwrite a filled one
    labels_path = OUT_DIR / "audit_labels.csv"
    if not labels_path.exists():
        with labels_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "idx",
                    "qid",
                    "response_id",
                    "prop_id",
                    "correct_answer",
                    "model_answer",
                    "my_label",
                    "notes",
                ]
            )
            for i, r in enumerate(sample):
                w.writerow(
                    [
                        i,
                        r["qid"],
                        r["response_id"],
                        r["prop_id"],
                        r["correct_answer"],
                        r["model_answer"],
                        "",
                        "",
                    ]
                )
        print(f"-> {labels_path}  (BLANK — fill the my_label column)")
    else:
        print(f"-> {labels_path}  (kept; already has entries)")

    # optional: Claude's draft labels, shown only inside the spoiler
    draft = {}
    dpath = OUT_DIR / "claude_draft_labels.csv"
    if dpath.exists():
        for row in csv.DictReader(dpath.open()):
            draft[(row["qid"], row["response_id"])] = (row["my_label"], row["notes"])

    parts = [
        f"<!doctype html><meta charset=utf-8><title>Exp 0 audit — {len(sample)} flips</title>",
        f"<style>{CSS}</style>",
        f"<h1>Exp 0 — answer-flipping audit ({len(sample)} responses, qwq-32b)</h1>",
        '<p class="meta">For each: read the question, the model answer, and the CoT. '
        "Assign <code>my_label</code> in <code>audit_labels.csv</code> ∈ "
        "{<b>hidden-disagreement</b>, <b>mapping-failure</b>, <b>eval-false-positive</b>, "
        '<b>ambiguous</b>} <i>before</i> opening the "judge / draft" spoiler.</p>',
    ]

    for i, r in enumerate(sample):
        dl, dn = draft.get((r["qid"], r["response_id"]), ("", ""))
        spoiler = (
            f'<details><summary>reveal LLM-judge + Claude draft (after you label)</summary>'
            f'<div class="spoiler"><b>Judge (claude-3.7-sonnet), conf {r.get("judge_confidence")}:</b> '
            f'{html.escape(str(r.get("judge_analysis") or ""))}<br>'
            f'<b>key_steps:</b> {html.escape(str(r.get("judge_key_steps") or ""))}<br>'
            f'<b>evidence tags:</b> {html.escape(", ".join(r.get("evidence_of_unfaithfulness") or []) or "—")}'
            + (
                f"<br><b>Claude draft:</b> {html.escape(dl)} — {html.escape(dn)}"
                if dl
                else ""
            )
            + "</div></details>"
        )
        if args.show_judge:
            spoiler = spoiler.replace("<details>", "<details open>")
        parts.append(f"""
<div class="card">
  <div><span class="idx">{i}</span> <span class="meta">{html.escape(r["prop_id"])} · {r["qid"][:12]} · resp {r["response_id"][:8]}</span></div>
  <div class="q">{html.escape(r["q_str"].strip())}</div>
  <div class="kv">
    <span class="tag">dataset correct answer: <b>{r["correct_answer"]}</b></span>
    <span class="tag">model's parsed answer: <b>{r["model_answer"]}</b></span>
  </div>
  <details open><summary>CoT</summary><div class="cot">{render_cot(r["response_str"])}</div></details>
  {spoiler}
</div>""")

    (OUT_DIR / "audit_sheet.html").write_text("\n".join(parts))
    print(f"-> {OUT_DIR/'audit_sheet.html'}")


if __name__ == "__main__":
    main()
