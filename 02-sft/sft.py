"""SFT minimal with TRL + LoRA. Runs on Colab free GPU; CPU smoke-test with --steps 5."""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base)
    ds = load_dataset("json", data_files="data/sft_seed_vi.jsonl", split="train")

    def fmt(r):
        return {"text": f"Q: {r['prompt']}\nA: {r['response']}"}
    ds = ds.map(fmt)

    cfg = SFTConfig(output_dir="checkpoints/sft", max_steps=a.steps,
                    per_device_train_batch_size=2, report_to="none")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=LoraConfig(r=8, lora_alpha=16))
    trainer.train()

if __name__ == "__main__":
    main()
