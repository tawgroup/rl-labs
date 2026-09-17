"""Download mini datasets for rl-labs. Laptop-first: --mini tải vài MB."""
import argparse, json, os

RAW = "data/raw"

def _save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} rows -> {path}")

def dl_tinystories(mini=True):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if mini:
        ds = ds.select(range(2000))
    rows = [{"text": t["text"]} for t in ds]
    _save_jsonl(f"{RAW}/tinystories.jsonl", rows)

def dl_gsm8k(mini=True):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if mini:
        ds = ds.select(range(200))
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    _save_jsonl(f"{RAW}/gsm8k.jsonl", rows)

def dl_mbpp(mini=True):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    if mini:
        ds = ds.select(range(100))
    rows = [{"task_id": r["task_id"], "prompt": r["prompt"], "code": r["code"], "test_list": r["test_list"]} for r in ds]
    _save_jsonl(f"{RAW}/mbpp.jsonl", rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--gsm8k", action="store_true")
    ap.add_argument("--mbpp", action="store_true")
    ap.add_argument("--mini", action="store_true", default=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mini = not a.full
    do_all = a.all or not (a.tiny or a.gsm8k or a.mbpp)
    if do_all or a.tiny: dl_tinystories(mini)
    if do_all or a.gsm8k: dl_gsm8k(mini)
    if do_all or a.mbpp: dl_mbpp(mini)
