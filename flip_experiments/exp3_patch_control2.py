import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"
EX_ID = "f1ac9bfbe98d4a47106c269ff3ebd482c0e8d52390bb9eeadcf33ff7d9da3120/fe4386a4-78da-4924-a081-3719ed84e4d9"
DONOR_ID = "50e0c5d296fb551ff74b1d2fbad14adddc6595c3ded06275fac293849061af11/4d322cf1-8303-4c0d-b7af-34bb7bc8868a"
MAX_NEW_TOKENS = 4096


def mirror_case(word, template):
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word.capitalize()


def parse_yn(text):
    ms = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
    return ms[-1].group(1).upper() if ms else None


def find_answer_span(completion, which):
    tail = (
        completion.split("</think>", 1)[1] if "</think>" in completion else completion
    )
    offset = len(completion) - len(tail)
    ms = list(re.finditer(r"\b(yes|no)\b", tail, re.IGNORECASE))
    if not ms:
        return None
    m = ms[0] if which == "first" else ms[-1]
    return offset + m.start(), offset + m.end()


def locate(tok, full_text, abs_start, abs_end, consistent_label):
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids_full, offsets = enc["input_ids"], enc["offset_mapping"]
    covering = [i for i, (s, e) in enumerate(offsets) if s < abs_end and e > abs_start]
    if len(covering) != 1:
        return None
    idx = covering[0]
    s, e = offsets[idx]
    if not (s <= abs_start and e >= abs_end):
        return None
    real_token_id = ids_full[idx]
    prefix_text = full_text[:s]
    leading_ws = full_text[s:abs_start]
    observed_word = full_text[abs_start:abs_end]
    inconsistent_label = "NO" if consistent_label == "YES" else "YES"
    observed_label = "YES" if observed_word.upper() == "YES" else "NO"
    other_label = (
        inconsistent_label if observed_label == consistent_label else consistent_label
    )
    other_word = leading_ws + mirror_case(other_label, observed_word)
    ids_prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
    ids_with_other = tok(prefix_text + other_word, add_special_tokens=False)[
        "input_ids"
    ]
    oth_added = ids_with_other[len(ids_prefix) :]
    other_token_id = oth_added[0] if len(oth_added) == 1 else None
    consistent_token_id = (
        real_token_id if observed_label == consistent_label else other_token_id
    )
    inconsistent_token_id = (
        other_token_id if observed_label == consistent_label else real_token_id
    )
    return {
        "prefix_text": prefix_text,
        "pre_index": len(ids_prefix) - 1,
        "consistent_token_id": consistent_token_id,
        "inconsistent_token_id": inconsistent_token_id,
    }


def record_vectors(model, n_layers, input_ids, position):
    vectors = {}

    def make_recorder(layer_idx):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            vectors[layer_idx] = hs[0, position, :].detach().clone()

        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(make_recorder(0))]
    for i in range(n_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_recorder(i + 1)))
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()
    return vectors, out.logits[0, position, :].float()


def main():
    payload = json.load(open("exp3_gpu_payload_v2.json"))
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in payload["prompts"]}
    gens = {
        (g["example_id"], g["condition"]): g["completion"]
        for g in (json.loads(l) for l in open("exp3_local_generations.jsonl"))
    }

    print(f"loading {MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    input_device = model.get_input_embeddings().weight.device

    # --- recipient: f1ac9bfb / original (same as every prior patching run) ---
    recip_ex = payload["examples"][EX_ID]
    recip_completion = gens[(EX_ID, "original")]
    p = prompts_by_key[(EX_ID, "original")]
    recip_chat = tok.apply_chat_template(
        [{"role": "user", "content": p["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    recip_full = recip_chat + recip_completion
    start, end = find_answer_span(recip_completion, "first")
    abs_start, abs_end = len(recip_chat) + start, len(recip_chat) + end
    loc_recip = locate(
        tok, recip_full, abs_start, abs_end, recip_ex["cot_supported_answer"]
    )
    recip_ids = tok(
        loc_recip["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    recip_pos = loc_recip["pre_index"]
    consistent_id, inconsistent_id = (
        loc_recip["consistent_token_id"],
        loc_recip["inconsistent_token_id"],
    )
    print(
        f"recipient (f1ac9bfb/original) pos={recip_pos}  consistent(NO)={consistent_id} "
        f"({tok.decode([consistent_id])!r})  inconsistent(YES)={inconsistent_id} ({tok.decode([inconsistent_id])!r})"
    )

    with torch.no_grad():
        base_logits = model(recip_ids, use_cache=False).logits[0, recip_pos, :].float()
    delta_baseline = (base_logits[consistent_id] - base_logits[inconsistent_id]).item()
    print(f"baseline Delta_recipient (unpatched) = {delta_baseline:+.3f}\n")

    # --- donor: frozen YES-consistent example, polarity interface, generated fresh ---
    donor_ex = payload["examples"][DONOR_ID]
    dp = prompts_by_key[(DONOR_ID, "polarity")]
    donor_chat = tok.apply_chat_template(
        [{"role": "user", "content": dp["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    donor_inputs = tok(donor_chat, return_tensors="pt", add_special_tokens=False).to(
        input_device
    )
    print(
        f"generating donor (frozen: {DONOR_ID[:12]}.../polarity) completion, up to {MAX_NEW_TOKENS} tokens ...",
        flush=True,
    )
    with torch.no_grad():
        out = model.generate(
            **donor_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    donor_completion = tok.decode(
        out[0][donor_inputs["input_ids"].shape[1] :], skip_special_tokens=False
    )
    open("exp3_donor_generation.jsonl", "w").write(
        json.dumps(
            {
                "example_id": DONOR_ID,
                "condition": "polarity",
                "completion": donor_completion,
            }
        )
        + "\n"
    )

    donor_post_think = (
        donor_completion.split("</think>", 1)[1]
        if "</think>" in donor_completion
        else donor_completion
    )
    donor_answer = parse_yn(donor_post_think)
    print(
        f"donor greedy answer parsed = {donor_answer}  (donor's own consistent answer = "
        f"{donor_ex['cot_supported_answer']})  {'OK -- matches' if donor_answer==donor_ex['cot_supported_answer'] else '*** does NOT match ***'}"
    )

    donor_full = donor_chat + donor_completion
    dstart, dend = find_answer_span(donor_completion, "first")
    dabs_start, dabs_end = len(donor_chat) + dstart, len(donor_chat) + dend
    loc_donor = locate(
        tok, donor_full, dabs_start, dabs_end, donor_ex["cot_supported_answer"]
    )
    donor_ids = tok(
        loc_donor["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    donor_pos = loc_donor["pre_index"]

    donor_vectors, donor_logits = record_vectors(model, n_layers, donor_ids, donor_pos)
    donor_delta_own = (
        donor_logits[loc_donor["consistent_token_id"]]
        - donor_logits[loc_donor["inconsistent_token_id"]]
    ).item()
    validity_ok = donor_delta_own > 0
    print(
        f"donor's OWN post-think Delta (favors its own consistent=YES if positive) = {donor_delta_own:+.3f}  "
        f"{'VALID donor (pre-patch check passed)' if validity_ok else '*** donor does NOT favor its own answer post-think -- reporting anyway, not swapping ***'}\n"
    )

    # --- patch sweep: donor's vectors -> recipient's post-think position ---
    def make_patcher(pos, vec):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            hs[0, pos, :] = vec.to(hs.device, dtype=hs.dtype)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

        return hook

    results = []
    print("patching YES-donor vectors into f1ac9bfb/original recipient, all layers:")
    for layer_idx in range(n_layers + 1):
        module = (
            model.model.embed_tokens
            if layer_idx == 0
            else model.model.layers[layer_idx - 1]
        )
        h = module.register_forward_hook(
            make_patcher(recip_pos, donor_vectors[layer_idx])
        )
        with torch.no_grad():
            out = model(recip_ids, use_cache=False)
        h.remove()
        logits = out.logits[0, recip_pos, :].float()
        delta = (logits[consistent_id] - logits[inconsistent_id]).item()
        argmax_id = int(logits.argmax().item())
        print(
            f"  layer {layer_idx:3d}/{n_layers}  Delta_recipient={delta:+7.3f}  top1={tok.decode([argmax_id])!r}"
        )
        results.append(
            {
                "layer": layer_idx,
                "n_layers": n_layers,
                "delta_patched": delta,
                "top1_token": tok.decode([argmax_id]),
                "delta_baseline": delta_baseline,
                "donor_delta_own": donor_delta_own,
                "donor_validity_ok": validity_ok,
            }
        )
        open("exp3_patch_control2.jsonl", "w").write(
            "\n".join(json.dumps(r) for r in results) + "\n"
        )

    n_toward_no = sum(r["delta_patched"] > 0 for r in results)
    n_toward_yes = sum(r["delta_patched"] < 0 for r in results)
    print(
        f"\n{n_toward_no}/{n_layers+1} layers stay/move toward recipient-correct NO (Delta>0)"
    )
    print(f"{n_toward_yes}/{n_layers+1} layers move toward donor's-own YES (Delta<0)")
    print(
        f"final layer Delta_patched = {results[-1]['delta_patched']:+.3f}  "
        f"(baseline unpatched = {delta_baseline:+.3f})"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = [r["layer"] / n_layers for r in results]
    ax.plot(xs, [r["delta_patched"] for r in results], "-o", ms=3, color="#16a085")
    ax.axhline(0, color="#999", lw=1)
    ax.axhline(
        delta_baseline,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"Delta_recipient (unpatched) = {delta_baseline:+.2f}",
    )
    ax.set_xlabel("layer patched (l / L)")
    ax.set_ylabel(r"$\Delta_{recipient}$ = logit(NO) $-$ logit(YES)")
    ax.set_title(
        "Corrected control: YES-consistent unrelated donor -> f1ac9bfb/original"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("exp3_patch_control2.png", dpi=150)
    print("-> exp3_patch_control2.png")


if __name__ == "__main__":
    main()
