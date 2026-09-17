# 02-sft — học làm theo lệnh (Base → Instruct)

Mục tiêu: hiểu khác biệt duy nhất giữa model ngọng và model chat được:
**format Q&A**. SFT 500 câu seed là đủ thấy đổi hành vi.

```bash
python 02-sft/sft.py --base gpt2 --steps 50
```

Dùng TRL + LoRA để laptop/Colab đều chạy. Data: `data/sft_seed_vi.jsonl`
(nhân bản lên 500 dòng rồi sửa dần — chất lượng seed quyết giọng model).
