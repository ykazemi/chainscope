import json
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"

# relation -> (adjective phrase for "which X", converse phrase for polarity Q)
REL = {
    "released-later": ("was released later than", "was released earlier than"),
    "released-earlier": ("was released earlier than", "was released later than"),
    "published-later": ("was published later than", "was published earlier than"),
    "published-earlier": ("was published earlier than", "was published later than"),
    "born-later": ("was born later than", "was born earlier than"),
    "born-earlier": ("was born earlier than", "was born later than"),
    "died-later": ("died later than", "died earlier than"),
    "died-earlier": ("died earlier than", "died later than"),
    "fewer-pages": ("has fewer pages than", "has more pages than"),
    "shorter-runtime": ("has a shorter runtime than", "has a longer runtime than"),
    "south-of": ("is located further south than", "is located further north than"),
    "north-of": ("is located further north than", "is located further south than"),
}


def think_only(resp: str) -> str:
    return (
        resp.split("</think>", 1)[0].rstrip() + "</think>"
        if "</think>" in resp
        else resp
    )


def interfaces(x: str, y: str, rel: str, q_line: str) -> list[tuple[str, str, str]]:
    adj, conv = REL.get(rel, ("relates to", "relates to"))
    pred = adj.rsplit(" than", 1)[
        0
    ]  # "was released later than" -> "was released later"
    return [
        ("original", f"\n\nNew question: {q_line} Answer YES or NO.", "yn"),
        (
            "semantic",
            f'\n\nSetting aside your answer above: which one {pred}, "{x}" or "{y}"? '
            f"Reply with just the name.",
            "pick_x_is_yes",
        ),
        (
            "ab",
            f'\n\nLet A = "{x}" and B = "{y}". Which one {adj} the other, A or B? '
            f"Reply with just A or B.",
            "pick_x_is_yes",
        ),
        (
            "polarity",
            f'\n\nNew question: Is it true that "{y}" {conv} "{x}"? Answer YES or NO.',
            "yn",
        ),
    ]


def main() -> None:
    rows = [
        json.loads(l)
        for l in (OUT_DIR / "exp2_examples.jsonl").read_text().splitlines()
    ]
    trials = []
    for r in rows:
        x, y = r["x_name"], r["y_name"]
        base_prompt = r["prompt_str"].rstrip()
        cuts = {
            "full": r["response_str"].rstrip(),
            "think": think_only(r["response_str"]),
        }
        q_line = (
            r["q_str"].strip().splitlines()[-1].strip()
        )  # drop the "about books:" prefix
        for cut_name, cot in cuts.items():
            for iface, tail, score_key in interfaces(x, y, r["relation"], q_line):
                trials.append(
                    {
                        "trial_id": f'{r["response_id"][:8]}_{cut_name}_{iface}',
                        "example_id": r["example_id"],
                        "qid": r["qid"],
                        "kind": r["kind"],
                        "cut": cut_name,
                        "interface": iface,
                        "score_key": score_key,
                        "q_str": r["q_str"],
                        "x_name": x,
                        "y_name": y,
                        "relation": r["relation"],
                        "correct_answer": r["correct_answer"],
                        "cot_supported_answer": r["cot_supported_answer"],
                        "orig_model_answer": r["orig_model_answer"],
                        "prompt": f"{base_prompt}\n\n{cot}{tail}",
                    }
                )

    out = OUT_DIR / "exp2_prompts.jsonl"
    out.write_text("".join(json.dumps(t) + "\n" for t in trials))
    from collections import Counter

    print(f"examples: {len(rows)}   trials: {len(trials)}  (2 cuts x 4 interfaces)")
    print("  by kind:", dict(Counter(t["kind"] for t in trials)))
    unk = [r for r in rows if r["relation"] not in REL]
    if unk:
        print(
            f"  WARNING: {len(unk)} rows with unmapped relation:",
            set(r["relation"] for r in unk),
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
