"""Tiny eval: exact-match on GSM8K-mini + pass rate on MBPP-mini (subset)."""
import argparse, json, re

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    print("eval skeleton: wire your checkpoint into 03-rl reward fns and score here.")
    print("GSM8K-mini rows:", len(load("data/raw/gsm8k.jsonl")))
    print("MBPP-mini rows:", len(load("data/raw/mbpp.jsonl")))
