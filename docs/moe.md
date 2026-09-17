# MoE — vì sao inference nhẹ mà laptop vẫn chịu

MoE (Mixture of Experts, kiểu DeepSeek V3): tổng 671B params nhưng mỗi token
router chỉ chọn 2-3 experts → active compute ~37B. Tính toán nhẹ như model 37B,
kiến thức rộng như 671B.

Vì sao laptop vẫn không train/serve được:
- VRAM phải chứa (hoặc offload) **toàn bộ** experts, không chỉ active phần.
- Routing, load-balancing, all-to-all comms phức tạp, train dễ sập.
- Offload qua RAM/disk làm inference chậm hẳn.

Kết luận cho repo này: học dense tiny trước. Khi cần MoE, thuê GPU + dùng
vLLM/SGLang đã tối ưu sẵn, đừng tự train MoE trên laptop.
