"""GRPO skeleton for MBPP-mini: reward = run test_list in subprocess (2s timeout).
DANGER: executes model-generated code — only run locally sandbox-style, never on prod.
"""
import argparse, subprocess, tempfile, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

def run_tests(code: str, tests: list) -> float:
    prog = code + "\n" + "\n".join(tests)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=2)
        return 1.0 if r.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        os.unlink(path)

def reward_fn(completions, test_list=None, **kwargs):
    return [run_tests(c, t) for c, t in zip(completions, test_list)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True)
    ds = load_dataset("json", data_files="data/raw/mbpp.jsonl", split="train")
    ds = ds.map(lambda r: {"prompt": r["prompt"], "test_list": r["test_list"]})

    cfg = GRPOConfig(output_dir="checkpoints/grpo-mbpp", max_steps=a.steps,
                     per_device_train_batch_size=2, report_to="none")
    GRPOTrainer(model=model, args=cfg, train_dataset=ds,
                reward_funcs=reward_fn).train()

if __name__ == "__main__":
    main()
