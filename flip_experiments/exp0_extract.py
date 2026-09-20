import argparse
import json
from pathlib import Path

from chainscope.data_fetcher import DataFetcher
from chainscope.typing import DatasetParams

MODEL_ID = "qwen/qwq-32b"
DATASET_SUFFIX = "non-ambiguous-hard-2"
INSTR_ID = "instr-wm"

OUT_DIR = Path(__file__).parent / "data"


def _judge_analysis_index(fetcher: DataFetcher, prop_id: str) -> dict[str, dict]:
    """response_id -> {analysis, key_steps, confidence} from the pattern eval."""
    try:
        pe = fetcher._load_unfaithfulness_pattern_eval(prop_id)
    except FileNotFoundError:
        return {}
    idx: dict[str, dict] = {}
    for qid_analysis in pe.pattern_analysis_by_qid.values():
        for qa in (qid_analysis.q1_analysis, qid_analysis.q2_analysis):
            if qa is None:
                continue
            for rid, r in qa.responses.items():
                idx[rid] = {
                    "judge_analysis": r.answer_flipping_analysis,
                    "judge_key_steps": getattr(r, "key_steps", None),
                    "judge_confidence": getattr(r, "confidence", None),
                }
    return idx


def _row(pair, q, rid, r, judge: dict) -> dict:
    return {
        "prop_id": DatasetParams.from_id(q.dataset_id).prop_id,
        "dataset_id": q.dataset_id,
        "qid": q.qid,
        "response_id": rid,
        "q_str": q.q_str,
        "correct_answer": q.correct_answer,
        "model_answer": r.model_answer,
        "is_correct": r.is_correct,
        "answer_flipping_classification": r.answer_flipping_classification,
        "evidence_of_unfaithfulness": r.evidence_of_unfaithfulness,
        "pair_patterns": pair.unfaithfulness_patterns,
        "judge_analysis": judge.get("judge_analysis"),
        "judge_key_steps": judge.get("judge_key_steps"),
        "judge_confidence": judge.get("judge_confidence"),
        "prompt_str": r.prompt_str,
        "response_str": r.response_str,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    flip_path = OUT_DIR / "flip_candidates.jsonl"
    ctrl_path = OUT_DIR / "nonflip_controls.jsonl"
    if flip_path.exists() and not args.force:
        print(f"{flip_path} exists; pass --force to re-extract")
        return

    fetcher = DataFetcher(
        model_id=MODEL_ID, dataset_suffix=DATASET_SUFFIX, instr_id=INSTR_ID
    )
    pairs = fetcher.get_unfaithful_question_pairs()
    print(f"unfaithful pairs: {len(pairs)}")

    judge_idx_by_prop: dict[str, dict[str, dict]] = {}
    flips, ctrls = [], []
    for pair in pairs:
        for q in (pair.q1, pair.q2):
            prop_id = DatasetParams.from_id(q.dataset_id).prop_id
            if prop_id not in judge_idx_by_prop:
                judge_idx_by_prop[prop_id] = _judge_analysis_index(fetcher, prop_id)
            jidx = judge_idx_by_prop[prop_id]
            for rid, r in q.responses.items():
                cls = r.answer_flipping_classification
                if cls == "YES":
                    flips.append(_row(pair, q, rid, r, jidx.get(rid, {})))
                elif cls == "NO":
                    ctrls.append(_row(pair, q, rid, r, jidx.get(rid, {})))

    flip_qids = {row["qid"] for row in flips}
    ctrls_matched = [row for row in ctrls if row["qid"] in flip_qids]

    with flip_path.open("w") as f:
        for row in flips:
            f.write(json.dumps(row) + "\n")
    with ctrl_path.open("w") as f:
        for row in ctrls_matched:
            f.write(json.dumps(row) + "\n")

    print(f"flip candidates:           {len(flips)}  -> {flip_path}")
    print(f"non-flip controls (matched): {len(ctrls_matched)}  -> {ctrl_path}")
    print(f"  (non-flip pool, all questions: {len(ctrls)})")
    print(f"distinct flip questions: {len(flip_qids)}")


if __name__ == "__main__":
    main()
