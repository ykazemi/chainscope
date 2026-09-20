import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"

# the two failing (example_id, condition, position) cells from exp3_dual_probe.jsonl
FAILING = [
    (
        "4367330b55f72ace04b09a9addab85fcfad35f1cde4748b0191eb6bbeee83cbe/851e092b-4067-4e5c-91d2-e9dfb63d9ac4",
        "polarity",
    ),
    (
        "f1ac9bfbe98d4a47106c269ff3ebd482c0e8d52390bb9eeadcf33ff7d9da3120/fe4386a4-78da-4924-a081-3719ed84e4d9",
        "polarity",
    ),
]


def main() -> None:
    payload = json.load(open("exp3_gpu_payload.json"))
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in payload["prompts"]}
    gens = {
        (g["example_id"], g["condition"]): g["completion"]
        for g in (json.loads(l) for l in open("exp3_local_generations.jsonl"))
    }
    dual = [json.loads(l) for l in open("exp3_dual_probe.jsonl")]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    final_norm, lm_head = model.model.norm, model.lm_head

    for ex_id, iface in FAILING:
        row = [
            r
            for r in dual
            if r["example_id"] == ex_id
            and r["condition"] == iface
            and r["position"] == "post_think"
        ][0]
        p = prompts_by_key[(ex_id, iface)]
        completion = gens[(ex_id, iface)]
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = chat_text + completion
        enc = tok(full_text, add_special_tokens=False)
        prefix_ids = enc["input_ids"][: row["pre_index"] + 1]
        real_next_id = enc["input_ids"][row["pre_index"] + 1]

        input_ids = torch.tensor([prefix_ids]).to(input_device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            hs = out.hidden_states[-1][0, -1, :].to(
                final_norm.weight.device, dtype=final_norm.weight.dtype
            )
            logits = lm_head(final_norm(hs)).float()
        top5 = torch.topk(logits, 5)
        print(
            f"\n{ex_id[:12]}.../{iface}/post_think  (real next token was {real_next_id} = {tok.decode([real_next_id])!r})"
        )
        for val, idx in zip(top5.values.tolist(), top5.indices.tolist()):
            marker = " <-- actually generated" if idx == real_next_id else ""
            print(f"  {idx:8d}  {tok.decode([idx])!r:12s}  logit={val:+.4f}{marker}")
        real_logit = logits[real_next_id].item()
        top1_logit = top5.values[0].item()
        print(f"  gap (top1 - real_next_token logit) = {top1_logit - real_logit:.4f}")


if __name__ == "__main__":
    main()
