import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/QwQ-32B"
EX_ID = "f1ac9bfbe98d4a47106c269ff3ebd482c0e8d52390bb9eeadcf33ff7d9da3120/fe4386a4-78da-4924-a081-3719ed84e4d9"


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


def main():
    payload = json.load(open("exp3_gpu_payload.json"))
    ex = payload["examples"][EX_ID]
    prompts_by_key = {(p["example_id"], p["interface"]): p for p in payload["prompts"]}
    gens = {
        (g["example_id"], g["condition"]): g["completion"]
        for g in (json.loads(l) for l in open("exp3_local_generations.jsonl"))
    }
    consistent_label = ex["cot_supported_answer"]

    print(f"loading {MODEL_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    input_device = model.get_input_embeddings().weight.device
    print(f"loaded. num_hidden_layers={n_layers}", flush=True)

    def get_loc(iface):
        completion = gens[(EX_ID, iface)]
        p = prompts_by_key[(EX_ID, iface)]
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = chat_text + completion
        start, end = find_answer_span(completion, "first")
        abs_start, abs_end = len(chat_text) + start, len(chat_text) + end
        return locate(tok, full_text, abs_start, abs_end, consistent_label)

    loc_orig, loc_pol = get_loc("original"), get_loc("polarity")
    orig_ids = tok(
        loc_orig["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    pol_ids = tok(
        loc_pol["prefix_text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(input_device)
    orig_pos, pol_pos = loc_orig["pre_index"], loc_pol["pre_index"]
    consistent_token_id, inconsistent_token_id = (
        loc_orig["consistent_token_id"],
        loc_orig["inconsistent_token_id"],
    )
    print(
        f"original post_think pos={orig_pos} (len {orig_ids.shape[1]})  "
        f"polarity post_think pos={pol_pos} (len {pol_ids.shape[1]})"
    )
    print(
        f"consistent_token_id={consistent_token_id} ({tok.decode([consistent_token_id])!r})  "
        f"inconsistent_token_id={inconsistent_token_id} ({tok.decode([inconsistent_token_id])!r})"
    )

    # step 1: record the clean (polarity) run's residual stream at every layer, at its OWN post-think position
    clean_vectors = {}

    def make_recorder(layer_idx):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            clean_vectors[layer_idx] = hs[0, pol_pos, :].detach().clone()

        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(make_recorder(0))]
    for i in range(n_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_recorder(i + 1)))
    with torch.no_grad():
        clean_out = model(pol_ids, use_cache=False)
    for h in hooks:
        h.remove()
    assert len(clean_vectors) == n_layers + 1
    clean_logits = clean_out.logits[0, pol_pos, :].float()
    delta_clean = (
        clean_logits[consistent_token_id] - clean_logits[inconsistent_token_id]
    ).item()
    print(
        f"\nrecorded {len(clean_vectors)} clean vectors from polarity run. "
        f"Delta_polarity (unpatched, sanity check) = {delta_clean:+.3f}"
    )

    # baseline: unpatched original run
    with torch.no_grad():
        base_out = model(orig_ids, use_cache=False)
    base_logits = base_out.logits[0, orig_pos, :].float()
    delta_original = (
        base_logits[consistent_token_id] - base_logits[inconsistent_token_id]
    ).item()
    print(f"Delta_original (unpatched baseline) = {delta_original:+.3f}\n")

    # step 2: patch each layer, one at a time, into the original run
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
            make_patcher(orig_pos, clean_vectors[layer_idx])
        )
        with torch.no_grad():
            out = model(orig_ids, use_cache=False)
        h.remove()
        logits = out.logits[0, orig_pos, :].float()
        delta_patched = (
            logits[consistent_token_id] - logits[inconsistent_token_id]
        ).item()
        argmax_id = int(logits.argmax().item())
        rescued = delta_patched > 0
        behavioral_rescue = argmax_id == consistent_token_id
        print(
            f"layer {layer_idx:3d}/{n_layers}  Delta_patched={delta_patched:+7.3f}  "
            f"rescued(sign)={rescued}  behavioral_argmax_rescue={behavioral_rescue}  "
            f"(top token: {tok.decode([argmax_id])!r})"
        )
        results.append(
            {
                "layer": layer_idx,
                "n_layers": n_layers,
                "delta_patched": delta_patched,
                "rescued": rescued,
                "behavioral_rescue": behavioral_rescue,
                "delta_original": delta_original,
                "delta_clean": delta_clean,
            }
        )
        open("exp3_patch_results.jsonl", "w").write(
            "\n".join(json.dumps(r) for r in results) + "\n"
        )

    n_rescued = sum(r["rescued"] for r in results)
    n_behavioral = sum(r["behavioral_rescue"] for r in results)
    print(f"\n{'='*70}")
    print(
        f"Delta_original (unpatched) = {delta_original:+.3f}   Delta_polarity (unpatched) = {delta_clean:+.3f}"
    )
    print(
        f"rescue criterion (Delta_original<0 -> Delta_patched>0): {n_rescued}/{n_layers+1} layers rescue the sign"
    )
    print(
        f"secondary behavioral rescue (argmax==consistent token): {n_behavioral}/{n_layers+1} layers"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["layer"] / n_layers for r in results]
    ys = [r["delta_patched"] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, "-o", ms=3, color="#8e44ad", label="Delta_patched(layer)")
    ax.axhline(0, color="#999", lw=1)
    ax.axhline(
        delta_original,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"Delta_original (unpatched) = {delta_original:+.2f}",
    )
    ax.axhline(
        delta_clean,
        color="#2980b9",
        ls="--",
        lw=1.2,
        label=f"Delta_polarity (unpatched) = {delta_clean:+.2f}",
    )
    ax.set_xlabel("layer patched (l / L)")
    ax.set_ylabel(
        r"$\Delta$ at original's post-think position after patching layer $l$"
    )
    ax.set_title(
        "Exp 3 patching: polarity -> original residual stream (f1ac9bfb, post-think position)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("exp3_patch.png", dpi=150)
    print("-> exp3_patch.png")


if __name__ == "__main__":
    main()
