import json
import re
import sys
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"


def _last(pat: str, text: str, flags=0) -> str | None:
    ms = list(re.finditer(pat, text, flags))
    return ms[-1].group(1) if ms else None


def parse_yn(text: str) -> str | None:
    a = _last(r"\b(yes|no)\b", text, re.IGNORECASE)
    return a.upper() if a else None


def parse_ab(text: str) -> str | None:
    return _last(r"\b([AB])\b", text)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"‘’“”*().,?!:;]", "", s.lower())).strip()


def _frags(name: str) -> set[str]:
    """Candidate lowercase substrings that identify this entity: the full name,
    the possessor / title halves of "Possessor's Title", and the first / last
    few words of a long descriptive name ("Christian Ulrich I, Duke of ...")."""
    raw = name.lower()
    out = {_clean(name)}
    m = re.match(r"(.+?)'s (.+)", raw)  # split on the apostrophe first
    if m:
        out |= {_clean(m.group(1)), _clean(m.group(2))}
    words = _clean(name).split()
    if len(words) >= 4:
        out |= {" ".join(words[:3]), " ".join(words[-3:])}
    return {f for f in out if len(f) >= 4}


def parse_pick(text: str, x: str, y: str) -> str | None:
    t = _clean(text)
    fx, fy = _frags(x), _frags(y)
    shared = fx & fy  # fragments common to both aren't diagnostic
    fx, fy = fx - shared, fy - shared
    ix = max((t.rfind(f) for f in fx if f in t), default=-1)
    iy = max((t.rfind(f) for f in fy if f in t), default=-1)
    if ix >= 0 and iy < 0:
        return "x"
    if iy >= 0 and ix < 0:
        return "y"
    if ix >= 0 and iy >= 0:  # both named — later mention is the pick
        return "x" if ix > iy else "y"
    ab = parse_ab(text)  # model answered "A"/"B" to a name prompt
    return {"A": "x", "B": "y"}.get(ab)


def score_row(r: dict) -> dict:
    c = r.get("completion", "") or ""
    if "</think>" in c:  # QwQ opens a <think> block; score its answer
        c = c.split("</think>", 1)[1]
    if r["score_key"] == "yn":
        raw = parse_yn(c)
        mapped = raw
    else:  # pick_x_is_yes
        pick = parse_pick(c, r["x_name"], r["y_name"])
        raw = pick
        mapped = {"x": "YES", "y": "NO"}.get(pick)
    return {
        **r,
        "raw_answer": raw,
        "mapped_answer": mapped,
        "cot_consistent": None
        if mapped is None
        else mapped == r["cot_supported_answer"],
        "gt_correct": None if mapped is None else mapped == r["correct_answer"],
        "parsed": mapped is not None,
    }


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "exp2_completions.jsonl"
    by_id = {}  # dedupe: last non-error write per trial wins
    for l in src.read_text().splitlines():
        d = json.loads(l)
        cur = by_id.get(d["trial_id"])
        if cur is None or str(cur["completion"]).startswith("<<ERROR"):
            by_id[d["trial_id"]] = d
    rows = [score_row(d) for d in by_id.values()]
    errs = sum(1 for d in by_id.values() if str(d["completion"]).startswith("<<ERROR"))
    if errs:
        print(f"WARNING: {errs} trials still have errors (excluded from rates)\n")
    (OUT_DIR / "exp2_scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    # interface x kind: P(cot_consistent) among parsed, and parse rate
    agg = defaultdict(lambda: [0, 0, 0])  # consistent, parsed, total
    for r in rows:
        for key in [(r["kind"], r["interface"]), (r["kind"], r["interface"], r["cut"])]:
            a = agg[key]
            a[2] += 1
            if r["parsed"]:
                a[1] += 1
                a[0] += bool(r["cot_consistent"])

    print(f"scored {len(rows)} trials  ->  data/exp2_scored.jsonl\n")
    print(f"{'kind':8} {'interface':10} {'cut':6}  P(CoT-consistent | parsed)   parsed")
    for key in sorted(k for k in agg if len(k) == 3):
        cons, parsed, tot = agg[key]
        rate = f"{cons/parsed:.0%}" if parsed else "  -"
        print(
            f"{key[0]:8} {key[1]:10} {key[2]:6}  {rate:>6}  ({cons}/{parsed})".ljust(52)
            + f"   {parsed}/{tot}"
        )
    print("\nPOOLED over cuts:")
    for key in sorted(k for k in agg if len(k) == 2):
        cons, parsed, tot = agg[key]
        rate = f"{cons/parsed:.0%}" if parsed else "  -"
        print(f"{key[0]:8} {key[1]:10}         {rate:>6}  ({cons}/{parsed})")
    print(
        "\nPrimary contrast: (flip, semantic) vs (flip, original)  — and the same"
        "\ngap for controls. Semantic should lift flips much more than controls."
    )


if __name__ == "__main__":
    main()
