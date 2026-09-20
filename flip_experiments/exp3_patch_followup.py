import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"
EX_ID = "f1ac9bfbe98d4a47106c269ff3ebd482c0e8d52390bb9eeadcf33ff7d9da3120/fe4386a4-78da-4924-a081-3719ed84e4d9"
CONTROL_EX_ID = "4367330b55f72ace04b09a9addab85fcfad35f1cde4748b0191eb6bbeee83cbe/851e092b-4067-4e5c-91d2-e9dfb63d9ac4"


def mirror_case(word, template):
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word.capitalize()


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


def get_loc(tok, payload, gens, ex_id, iface):
    ex = payload["examples"][ex_id]
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in payload["prompts"]}
    completion = gens[(ex_id, iface)]
    p = prompts_by_key[(ex_id, iface)]
    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": p["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = chat_text + completion
    start, end = find_answer_span(completion, "first")
    abs_start, abs_end = len(chat_text) + start, len(chat_text) + end
    return locate(tok, full_text, abs_start, abs_end, ex["cot_supported_answer"])


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
    logits = out.logits[0, position, :].float()
    return vectors, logits


def patch_sweep(
    model,
    tok,
    n_layers,
    target_ids,
    target_pos,
    vectors_by_layer,
    consistent_id,
    inconsistent_id,
    label,
):
    def make_patcher(pos, vec):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            hs[0, pos, :] = vec.to(hs.device, dtype=hs.dtype)
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

        return hook

    results = []
    for layer_idx in range(n_layers + 1):
        module = (
            model.model.embed_tokens
            if layer_idx == 0
            else model.model.layers[layer_idx - 1]
        )
        h = module.register_forward_hook(
            make_patcher(target_pos, vectors_by_layer[layer_idx])
        )
        with torch.no_grad():
            out = model(target_ids, use_cache=False)
        h.remove()
        logits = out.logits[0, target_pos, :].float()
        delta = (logits[consistent_id] - logits[inconsistent_id]).item()
        argmax_id = int(logits.argmax().item())
        print(
            f"[{label}] layer {layer_idx:3d}/{n_layers}  Delta={delta:+7.3f}  top1={tok.decode([argmax_id])!r}"
        )
        results.append(
            {
                "layer": layer_idx,
                "n_layers": n_layers,
                "delta_patched": delta,
                "top1_token": tok.decode([argmax_id]),
            }
        )
    return results


def main():
    payload = json.load(open("exp3_gpu_payload.json"))
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

    loc_orig = get_loc(tok, payload, gens, EX_ID, "original")
    loc_pol = get_loc(tok, payload, gens, EX_ID, "polarity")
    orig_ids = tok(
        loc_orig["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    pol_ids = tok(
        loc_pol["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    orig_pos, pol_pos = loc_orig["pre_index"], loc_pol["pre_index"]
    consistent_id, inconsistent_id = (
        loc_orig["consistent_token_id"],
        loc_orig["inconsistent_token_id"],
    )
    print(
        f"f1ac9bfb  original pos={orig_pos} (len {orig_ids.shape[1]})  "
        f"polarity pos={pol_pos} (len {pol_ids.shape[1]})"
    )
    print(
        f"consistent={consistent_id} ({tok.decode([consistent_id])!r})  "
        f"inconsistent={inconsistent_id} ({tok.decode([inconsistent_id])!r})\n"
    )

    # ---- 1. reverse patch: original -> polarity ----
    print(
        "=" * 70 + "\nSTEP 1: recording ORIGINAL run vectors (to patch into polarity)"
    )
    orig_vectors, orig_logits = record_vectors(model, n_layers, orig_ids, orig_pos)
    delta_original = (orig_logits[consistent_id] - orig_logits[inconsistent_id]).item()

    _, pol_logits_baseline = record_vectors(model, n_layers, pol_ids, pol_pos)
    delta_polarity = (
        pol_logits_baseline[consistent_id] - pol_logits_baseline[inconsistent_id]
    ).item()
    print(
        f"baseline Delta_original={delta_original:+.3f}  Delta_polarity={delta_polarity:+.3f}\n"
    )

    print("STEP 2: reverse sweep -- patching ORIGINAL vectors into POLARITY run")
    reverse_results = patch_sweep(
        model,
        tok,
        n_layers,
        pol_ids,
        pol_pos,
        orig_vectors,
        consistent_id,
        inconsistent_id,
        "reverse",
    )
    for r in reverse_results:
        r["delta_original_baseline"] = delta_original
        r["delta_polarity_baseline"] = delta_polarity
        r["reversed"] = r["delta_patched"] < 0
    open("exp3_patch_reverse.jsonl", "w").write(
        "\n".join(json.dumps(r) for r in reverse_results) + "\n"
    )
    n_reversed = sum(r["reversed"] for r in reverse_results)
    print(
        f"\n{n_reversed}/{n_layers+1} layers reverse the sign (Delta_polarity>0 -> Delta_patched<0)\n"
    )

    # ---- 2. negative control: unrelated example's post-think vector -> f1ac9bfb original ----
    print(
        "=" * 70
        + "\nSTEP 3: recording CONTROL (4367330b, polarity) vectors -- unrelated question"
    )
    loc_ctrl = get_loc(tok, payload, gens, CONTROL_EX_ID, "polarity")
    ctrl_ids = tok(
        loc_ctrl["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    ctrl_pos = loc_ctrl["pre_index"]
    ctrl_vectors, _ = record_vectors(model, n_layers, ctrl_ids, ctrl_pos)
    print(f"control example post-think pos={ctrl_pos} (len {ctrl_ids.shape[1]})\n")

    print(
        "STEP 4: control sweep -- patching UNRELATED vectors into f1ac9bfb's ORIGINAL run"
    )
    control_results = patch_sweep(
        model,
        tok,
        n_layers,
        orig_ids,
        orig_pos,
        ctrl_vectors,
        consistent_id,
        inconsistent_id,
        "control",
    )
    for r in control_results:
        r["delta_original_baseline"] = delta_original
        r["rescued"] = r["delta_patched"] > 0
    open("exp3_patch_control.jsonl", "w").write(
        "\n".join(json.dumps(r) for r in control_results) + "\n"
    )
    n_rescued_ctrl = sum(r["rescued"] for r in control_results)
    print(
        f"\n{n_rescued_ctrl}/{n_layers+1} layers show a spurious rescue from the UNRELATED vector "
        f"(compare to 19/65 for the real polarity vector)\n"
    )

    # ---- plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    xs = [r["layer"] / n_layers for r in reverse_results]
    ax.plot(
        xs, [r["delta_patched"] for r in reverse_results], "-o", ms=3, color="#c0392b"
    )
    ax.axhline(0, color="#999", lw=1)
    ax.axhline(
        delta_polarity,
        color="#2980b9",
        ls="--",
        lw=1.2,
        label=f"Delta_polarity (unpatched) = {delta_polarity:+.2f}",
    )
    ax.axhline(
        delta_original,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"Delta_original (unpatched) = {delta_original:+.2f}",
    )
    ax.set_xlabel("layer patched (l / L)")
    ax.set_ylabel(r"$\Delta$ at polarity's post-think position")
    ax.set_title("Reverse: original -> polarity")
    ax.legend(fontsize=7)

    ax = axes[1]
    xs = [r["layer"] / n_layers for r in control_results]
    ax.plot(
        xs, [r["delta_patched"] for r in control_results], "-o", ms=3, color="#7f8c8d"
    )
    ax.axhline(0, color="#999", lw=1)
    ax.axhline(
        delta_original,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"Delta_original (unpatched) = {delta_original:+.2f}",
    )
    ax.set_xlabel("layer patched (l / L)")
    ax.set_ylabel(r"$\Delta$ at original's post-think position")
    ax.set_title("Negative control: unrelated example -> original")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("exp3_patch_followup.png", dpi=150)
    print("-> exp3_patch_followup.png")


if __name__ == "__main__":
    main()
