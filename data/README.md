# Data — combo nhẹ, verify được

## 3 dataset mini

1. **TinyStories** (pretrain): truyện 3-4 tuổi, vocab nhỏ. Dùng sample ~2MB là đủ thấy loss xuống trên laptop.
   Nguồn: `roneneldan/TinyStories` trên HF.
2. **GSM8K-mini** (RL toán): 200 bài trích từ `openai/gsm8k`, đáp án là 1 số → reward = so số sau khi parse, dùng `sympy`.
3. **MBPP-mini** (RL code Python): 100 bài trích từ `google-research-datasets/mbpp` (sanitized), mỗi bài có `test_list` → reward = chạy `pytest`-style trong sandbox subprocess timeout 2s.

## Tải

```bash
python data/download.py --all --mini
# raw nằm ở data/raw/, processed ở data/processed/
```

Full-size khi cần: bỏ flag `--mini`.

## Tự tạo SFT seed (02-sft dùng)

`data/sft_seed_vi.jsonl` — 500 dòng mẫu, bạn tự nhân bản/sửa:
`{"prompt": "2 + 3 = ?", "response": "5"}`
Mục tiêu: cho model thấy format hỏi-đáp, không cần hay.
