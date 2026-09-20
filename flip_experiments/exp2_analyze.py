import json
import random
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
IFACES = ["original", "semantic", "ab", "polarity"]
CUTS = ["full", "think"]
N_BOOT = 10_000


def load():
    return [
        json.loads(l) for l in (OUT_DIR / "exp2_scored.jsonl").read_text().splitlines()
    ]


def cell(rows: list[dict]) -> dict:
    n = len(rows)
    parsed = [r for r in rows if r["parsed"]]
    cons = sum(bool(r["cot_consistent"]) for r in parsed)
    return {
        "n": n,
        "n_parsed": len(parsed),
        "n_consistent": cons,
        "lenient": cons / len(parsed) if parsed else None,  # P(consistent | parseable)
        "strict": cons / n if n else None,  # P(consistent AND parseable)
    }


def print_tables(rows: list[dict]) -> None:
    g = defaultdict(list)
    for r in rows:
        g[(r["kind"], r["cut"], r["interface"])].append(r)
        g[(r["kind"], "pooled", r["interface"])].append(r)

    for cut_label in CUTS + ["pooled"]:
        print(f"\n=== cut = {cut_label} ===")
        print(
            f"{'':18}{'lenient P(consistent|parseable)':>34}{'strict P(consistent & parseable)':>36}"
        )
        for kind in ("flip", "control"):
            for iface in IFACES:
                c = cell(g[(kind, cut_label, iface)])
                lo = f"{c['lenient']:.0%}" if c["lenient"] is not None else "  - "
                st = f"{c['strict']:.0%}"
                print(
                    f"  {kind:8}{iface:9} n={c['n']:3}  "
                    f"{lo:>10} ({c['n_consistent']}/{c['n_parsed']})".ljust(52)
                    + f"{st:>8} ({c['n_consistent']}/{c['n']})"
                )

    print("\npolarity, cuts NOT pooled (the one that moves a lot):")
    for kind in ("flip", "control"):
        for cut in CUTS:
            c = cell(g[(kind, cut, "polarity")])
            print(
                f"  {kind:8} {cut:6}  lenient={c['lenient']:.0%}  strict={c['strict']:.0%}"
            )


def bootstrap_interaction(
    rows: list[dict], cut_label: str
) -> tuple[float, float, float]:
    """[semantic-original]_flip - [semantic-original]_control, strict metric,
    resampled over examples (with replacement, within each kind)."""
    by_kind_example: dict[str, dict[str, dict[str, list[dict]]]] = {
        "flip": defaultdict(lambda: defaultdict(list)),
        "control": defaultdict(lambda: defaultdict(list)),
    }
    for r in rows:
        if cut_label != "pooled" and r["cut"] != cut_label:
            continue
        if r["interface"] not in ("original", "semantic"):
            continue
        by_kind_example[r["kind"]][r["example_id"]][r["interface"]].append(r)

    flip_ex = list(by_kind_example["flip"].keys())
    ctrl_ex = list(by_kind_example["control"].keys())

    def strict_rate(examples: list[str], kind: str, iface: str) -> float:
        trials = [
            t for ex in examples for t in by_kind_example[kind][ex].get(iface, [])
        ]
        return sum(bool(t["cot_consistent"]) for t in trials) / len(trials)

    point = (
        strict_rate(flip_ex, "flip", "semantic")
        - strict_rate(flip_ex, "flip", "original")
    ) - (
        strict_rate(ctrl_ex, "control", "semantic")
        - strict_rate(ctrl_ex, "control", "original")
    )

    rng = random.Random(0)
    boots = []
    for _ in range(N_BOOT):
        fb = [rng.choice(flip_ex) for _ in flip_ex]
        cb = [rng.choice(ctrl_ex) for _ in ctrl_ex]
        d = (
            strict_rate(fb, "flip", "semantic") - strict_rate(fb, "flip", "original")
        ) - (
            strict_rate(cb, "control", "semantic")
            - strict_rate(cb, "control", "original")
        )
        boots.append(d)
    boots.sort()
    lo = boots[int(0.025 * N_BOOT)]
    hi = boots[int(0.975 * N_BOOT)]
    return point, lo, hi


def make_figure(rows: list[dict]) -> None:
    """ONE figure: 4 interfaces x {flip, control}, strict metric, cuts pooled."""
    g = defaultdict(list)
    for r in rows:
        g[(r["kind"], r["interface"])].append(r)
    parse_rates = {}
    vals = {"flip": [], "control": []}
    for kind in ("flip", "control"):
        for iface in IFACES:
            c = cell(g[(kind, iface)])
            vals[kind].append(c["strict"])
            parse_rates[(kind, iface)] = c["n_parsed"] / c["n"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = range(len(IFACES))
    w = 0.36
    b1 = ax.bar(
        [i - w / 2 for i in x],
        vals["flip"],
        w,
        label="mapping-failures (n=39)",
        color="#a8631a",
    )
    b2 = ax.bar(
        [i + w / 2 for i in x],
        vals["control"],
        w,
        label="matched controls (n=29)",
        color="#5f6d7a",
    )
    for bars, kind in ((b1, "flip"), (b2, "control")):
        for bar, iface in zip(bars, IFACES):
            pr = parse_rates[(kind, iface)]
            note = f"{bar.get_height():.0%}" + (
                f"\n({pr:.0%} parsed)" if pr < 0.99 else ""
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                note,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(IFACES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("P(answer consistent AND parseable)\n[unparsed counted as failure]")
    ax.set_title(
        "Fixed CoT, vary the answer interface (Qwen/QwQ-32B)\ncuts pooled; strict metric",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(0.5, color="#ccc", lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp2_summary.png", dpi=150)
    print(
        "\n-> data/exp2_summary.png  (main figure: strict metric, cuts pooled, parse rate annotated)"
    )


def main() -> None:
    rows = load()
    print(f"{len(rows)} scored trials\n")
    print_tables(rows)

    print(
        "\n\n=== bootstrapped interaction (strict metric, 10,000 resamples over examples) ==="
    )
    print("[semantic - original]_flip - [semantic - original]_control")
    for cut_label in CUTS + ["pooled"]:
        pt, lo, hi = bootstrap_interaction(rows, cut_label)
        print(f"  {cut_label:8}  point={pt:+.0%}   95% CI [{lo:+.0%}, {hi:+.0%}]")

    make_figure(rows)


if __name__ == "__main__":
    main()
