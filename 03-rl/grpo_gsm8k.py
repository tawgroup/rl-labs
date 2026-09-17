"""GRPO skeleton for GSM8K-mini with verifiable reward (numeric exact match).
Requires GPU (Colab free). See README for the idea; TRL API may drift — pin trl==0.9.6.
"""
import argparse, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

def parse_number(t: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return m[-1] if m else None

def gold_number(answer: str):
    # GSM8K format: "...#### 42"
    if "####" in answer:
        return answer.split("####")[-1].strip().split()[0]
    return parse_number(answer)

def reward_fn(completions, answer=None, **kwargs):
    out = []
    for c, g in zip(completions, answer):
        out.append(1.0 if parse_number(c) == gold_number(g) else 0.0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True)
    ds = load_dataset("json", data_files="data/raw/gsm8k.jsonl", split="train")
    ds = ds.map(lambda r: {"prompt": f"Solve: {r['question']}\nAnswer with a number.", "answer": r["answer"]})

    cfg = GRPOConfig(output_dir="checkpoints/grpo-gsm8k", max_steps=a.steps,
                     per_device_train_batch_size=2, report_to="none")
    GRPOTrainer(model=model, args=cfg, train_dataset=ds,
                reward_funcs=reward_fn).train()

if __name__ == "__main__":
    main()
