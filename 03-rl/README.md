# 03-rl — học suy luận đúng (reward máy chấm)

Đây là trend hiện tại (DeepSeek R1, Good Start Labs): SFT chỉ bắt chước,
RL mới dạy tự thử-sai với **verifiable reward**.

2 task mini trong repo:

1. **Toán (`grpo_gsm8k.py`)**: reward = parse số cuối + so với đáp án (exact match).
   Không cần giám khảo người.
2. **Code (`grpo_mbpp.py`)**: reward = chạy `test_list` trong subprocess timeout 2s,
   pass hết = +1. Đây là bản mini của "game engine làm nguồn reward".

```bash
python 03-rl/grpo_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 50
python 03-rl/grpo_mbpp.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 50
```

Nâng cao (giống bài báo): dùng model xịn làm teacher cho **dense reward từng bước**
(VD chấm "có đọc test trước khi sửa không"), cộng với reward cứng từ test.
Reward cứng luôn là chính — teacher chỉ phụ, tránh model nịnh teacher.
