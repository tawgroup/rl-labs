"""Tiny GPT smoke-test: char-level is too slow to converge meaningfully on CPU,
so this uses gpt2 tokenizer + tiny GPT2 config. Laptop-first (500 steps)."""
import argparse
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    ds = load_dataset("json", data_files="data/raw/tinystories.jsonl", split="train")
    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=128)
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])

    cfg = GPT2Config(n_layer=4, n_head=4, n_embd=256, vocab_size=tok.vocab_size)
    model = GPT2LMHeadModel(cfg)
    args = TrainingArguments(
        output_dir="checkpoints/tiny-gpt", max_steps=a.steps,
        per_device_train_batch_size=4, logging_steps=50,
        save_steps=500, report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()

if __name__ == "__main__":
    main()
