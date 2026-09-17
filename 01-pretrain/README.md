# 01-pretrain — học nói (base LM)

Mục tiêu: hiểu **next-token prediction**. Train GPT tí hon (~10M params)
trên TinyStories-mini, laptop CPU/MPS chạy được vài trăm steps.

Chạy smoke test:

```bash
python 01-pretrain/train_tiny.py --steps 500
# xong gõ gì nó cũng "kể tiếp", hỏi 2+3 nó vẫn kể tiếp -> đúng, vì chưa qua SFT
```

Bài học duy nhất cần nhớ: **base LM không biết trả lời, nó chỉ biết viết tiếp.**
Muốn Q&A phải sang `02-sft/`.
