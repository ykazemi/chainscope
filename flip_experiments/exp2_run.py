import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    for _p in (
        _HERE / ".env",
        _HERE.parent / ".env",
    ):  # flip_experiments/.env or repo-root .env
        if _p.exists():
            load_dotenv(_p)
except Exception:  # noqa: BLE001
    pass

OUT_DIR = _HERE / "data"

# QwQ-32B is no longer on OpenRouter; DeepInfra still serves it. Auto-pick based
# on which key is present, or override everything with EXP2_* vars.
_DEEPINFRA = (os.environ.get("DEEPINFRA_API_KEY") or "").strip()
_OPENROUTER = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
if os.environ.get("EXP2_API_KEY"):
    BASE_URL = (
        os.environ.get("EXP2_BASE_URL") or "https://api.deepinfra.com/v1/openai"
    ).strip()
    MODEL = (os.environ.get("EXP2_MODEL") or "Qwen/QwQ-32B").strip()
    API_KEY = os.environ["EXP2_API_KEY"].strip()
elif _DEEPINFRA:
    BASE_URL, MODEL, API_KEY = (
        "https://api.deepinfra.com/v1/openai",
        "Qwen/QwQ-32B",
        _DEEPINFRA,
    )
elif (
    _OPENROUTER
):  # fallback: a *different* model (QwQ gone from OpenRouter) — note the caveat
    BASE_URL, MODEL, API_KEY = (
        "https://openrouter.ai/api/v1",
        "qwen/qwen3-32b",
        _OPENROUTER,
    )
else:
    BASE_URL = MODEL = API_KEY = ""
BASE_URL = (os.environ.get("EXP2_BASE_URL") or BASE_URL).strip()
MODEL = (os.environ.get("EXP2_MODEL") or MODEL).strip()
PROVIDER = os.environ.get("EXP2_PROVIDER", "").strip()  # OpenRouter: pin one upstream
PROMPTS = OUT_DIR / "exp2_prompts.jsonl"
COMPLETIONS = OUT_DIR / "exp2_completions.jsonl"
# QwQ-32B always opens a <think> block, so we can't cap at a few tokens; let it
# think briefly then answer, and parse the post-</think> text in exp2_score.py.
MAX_TOKENS = 4096
CONNECT_TIMEOUT, READ_TIMEOUT = 10, 150  # hard-ish: fail a stalled request and retry


def backend_dummy(prompt: str) -> str:
    # deterministic-ish stub for wiring tests only
    return "YES" if len(prompt) % 2 else "NO"


def backend_openai(prompt: str) -> str:
    import requests  # uses certifi CA bundle; avoids macOS system-cert issues

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    if PROVIDER:
        payload["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
    url = BASE_URL.rstrip("/") + "/chat/completions"
    hdr = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/cot-flip-audit",
        "X-Title": "cot-flip-exp2",
    }
    last = None
    for attempt in range(4):
        try:
            resp = requests.post(
                url, json=payload, headers=hdr, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data:
                raise RuntimeError(f"no choices: {json.dumps(data)[:300]}")
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2**attempt + 1, 20))
    raise RuntimeError(f"4 attempts failed: {last}")


BACKENDS = {"dummy": backend_dummy, "openai": backend_openai}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6, help="concurrent requests")
    args = ap.parse_args()
    call = BACKENDS[args.backend]
    if args.backend == "openai":
        if not API_KEY:
            raise SystemExit(
                "no API key. Put one line in flip_experiments/.env (or repo-root .env):\n"
                "  DEEPINFRA_API_KEY=...   (serves the audited Qwen/QwQ-32B)"
            )
        print(
            f"backend: {BASE_URL}  model: {MODEL}"
            + (f"  provider: {PROVIDER}" if PROVIDER else "")
        )
        if "qwen3" in MODEL.lower():
            print(
                "  NOTE: qwen3-32b != the audited QwQ-32B — Exp 2 result carries a model-swap caveat"
            )

    done = set()
    if COMPLETIONS.exists():
        for l in COMPLETIONS.read_text().splitlines():
            d = json.loads(l)
            if not str(d.get("completion", "")).startswith(
                "<<ERROR"
            ):  # retry errors on re-run
                done.add(d["trial_id"])
    trials = [
        json.loads(l)
        for l in PROMPTS.read_text().splitlines()
        if json.loads(l)["trial_id"] not in done
    ]
    if args.limit:
        trials = trials[: args.limit]
    print(f"{len(done)} already done · {len(trials)} to run · {args.workers} workers")

    lock = threading.Lock()
    f = COMPLETIONS.open("a")
    n_done = [0]

    def work(t: dict) -> None:
        try:
            comp = call(t["prompt"])
        except Exception as e:  # noqa: BLE001 - keep going, record the failure
            comp = f"<<ERROR: {e}>>"
        row = {**{k: v for k, v in t.items() if k != "prompt"}, "completion": comp}
        with lock:
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_done[0] += 1
            if n_done[0] % 25 == 0 or n_done[0] == len(trials):
                print(f"  {n_done[0]}/{len(trials)}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for fut in as_completed(ex.submit(work, t) for t in trials):
            fut.result()
    f.close()

    errs = sum(
        1 for l in COMPLETIONS.read_text().splitlines() if '"completion": "<<ERROR' in l
    )
    print(
        f"-> {COMPLETIONS}   ({errs} errors — re-run to retry them)"
        f"\nnext: python flip_experiments/exp2_score.py"
    )


if __name__ == "__main__":
    main()
