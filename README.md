# rl-labs — tiny pretrain → SFT → RL, laptop-first

Hiểu full flow ngành LLM bằng tay, không chỉ đọc báo:
`pretrain (học nói) → SFT (học làm theo lệnh) → RL với reward verify được (học suy luận đúng)`.

Repo này chạy được trên **laptop (CPU/MPS) + Colab free**, không cần cụm GPU.

## 3 chặng

| # | Thư mục | Làm gì | Data | Chạy ở đâu |
|---|---------|--------|------|------------|
| 01 | `01-pretrain/` | Train base LM tí hon, hiểu next-token prediction | TinyStories sample (~2MB, tự download) | Laptop OK |
| 02 | `02-sft/` | Dạy format Q&A (Base → Instruct) | 500 câu tự chế + Alpaca-mini | Laptop/Colab |
| 03 | `03-rl/` | RL với reward máy chấm (toán + Python) | GSM8K-mini + MBPP-mini | Colab free GPU |

## Quickstart

```bash
pip install -r requirements.txt
python data/download.py --all --mini   # tải 3 dataset mini
python 01-pretrain/train_tiny.py --steps 500   # smoke test laptop
python 02-sft/sft.py --base gpt2 --steps 50
python 03-rl/grpo_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 50
python eval/eval.py --all
```

## Vì sao combo này?

- **TinyStories**: pretrain rẻ nhất mà vẫn ra chữ mượt. Train xong nó chỉ biết kể tiếp, chưa biết trả lời → động lực làm bước 02.
- **GSM8K-mini (toán)**: đáp án verify bằng code (`sympy` / so số), không cần giám khảo người.
- **MBPP-mini (Python)**: reward = `pytest pass/fail`, chuẩn bài Good Start Labs: game/engine làm nguồn reward.

## Kiến trúc: dense trước, MoE sau

Mặc định dùng dense tiny (GPT-2 / Qwen-0.5B). MoE chỉ lợi khi serve lớn
(tổng params to nhưng active/token nhỏ) — laptop chứa không nổi tổng params,
nên để ở `docs/moe.md` đọc thêm, không train ở đây.

## Lộ trình đề xuất (1 weekend)

1. Chạy hết quickstart ở chế độ mini, thấy loss xuống là được.
2. Đọc `01-pretrain/README.md` → hiểu vì sao base LM không biết trả lời.
3. Đọc `02-sft/README.md` → Base vs Instruct khác nhau đúng 1 bước này.
4. Đọc `03-rl/README.md` → reward verifiable + teacher-dense-reward là trend hiện tại.
