"""Zero-shot baseline through the Hugging Face inference router (OpenAI-compatible chat API, billed to the HF
account behind HF_TOKEN): same renders and full-feature prompt as `scripts/claude_baseline.py submit --rich`.
For open-weight models that HF providers serve (e.g. deepseek-ai/DeepSeek-V4.1-Flash).

  python scripts/router_baseline.py --model deepseek-ai/DeepSeek-V4.1-Flash --rich data/rich/real --modes view4 view1 \
      --out results/deepseek_real/preds.pkl
Writes the rich preds layout (preds["<tag>:<mode>"], scored by scripts/rich_iou.py); resumes from an existing file and
saves every --save_every answers. Token usage is summed in P["usage"].
"""
import argparse
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="deepseek")
    ap.add_argument("--rich", required=True)
    ap.add_argument("--modes", nargs="*", default=["view4"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max_tokens", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    import requests
    from scripts.claude_baseline import TASK_RICH, VIEW1, VIEWS, extract_code, grid_png
    df = pd.concat([pd.read_parquet(os.path.join(a.rich, sh, "parts.parquet")).assign(shard=sh)
                    for sh in sorted(os.listdir(a.rich)) if os.path.exists(os.path.join(a.rich, sh, "parts.parquet"))],
                   ignore_index=True).head(a.n)
    stems = [os.path.join(a.rich, sh, "renders", u) for sh, u in zip(df.shard, df.uid)]
    if os.path.exists(a.out):
        P = pickle.load(open(a.out, "rb"))
    else:
        P = {"uids": list(df.uid), "dup": [False] * len(df), "family": list(df.family), "gt_code": list(df.code),
             "stems": stems, "kind": "cadquery", "preds": {}, "raw": {}, "usage": {}, "model": a.model}
    for m in a.modes:
        P["preds"].setdefault(f"{a.tag}:{m}", [None] * len(df))
        P["raw"].setdefault(f"{a.tag}:{m}", [None] * len(df))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lock, done = threading.Lock(), [0]
    headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}

    def one(m, i):
        body = {"model": a.model, "max_tokens": a.max_tokens, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + grid_png(stems[i], m)}},
            {"type": "text", "text": (VIEWS if m.startswith("view4") else VIEW1) + "\n\n" + TASK_RICH}]}]}
        for attempt in range(3):
            try:
                r = requests.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=body,
                                  timeout=a.timeout)
                if r.status_code == 200:
                    d = r.json()
                    text = d["choices"][0]["message"].get("content") or ""
                    with lock:
                        P["raw"][f"{a.tag}:{m}"][i] = text
                        P["preds"][f"{a.tag}:{m}"][i] = extract_code(text) if text else None
                        for k, v in (d.get("usage") or {}).items():
                            if isinstance(v, int):
                                P["usage"][k] = P["usage"].get(k, 0) + v
                        done[0] += 1
                        if done[0] % a.save_every == 0:
                            pickle.dump(P, open(a.out, "wb"))
                            print(f"[router] {done[0]} answers, usage {P['usage']}", flush=True)
                    return
                print(f"[router] {m} {i}: HTTP {r.status_code} {r.text[:200]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[router] {m} {i}: {type(e).__name__}", flush=True)
            time.sleep(10 * (attempt + 1))

    todo = [(m, i) for m in a.modes for i in range(len(df)) if P["raw"][f"{a.tag}:{m}"][i] is None]
    print(f"[router] {a.model}: {len(todo)} requests to go", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda x: one(*x), todo))
    pickle.dump(P, open(a.out, "wb"))
    print(f"[done] {a.out}; usage {P['usage']}", flush=True)


if __name__ == "__main__":
    main()
