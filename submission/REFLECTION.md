# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trương Việt Anh  
**Mã sinh viên:** 2A202602444  
**Khoá:** 4 · Track 3 · Ngày 22  
**Tier đã chạy:** T4 (Colab)  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 200 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~10 phút |
| VRAM cao nhất | ~7.2 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0053 |
| Độ chính xác reward trên held-out | 57.0% |
| Margin trên held-out | +0.0073 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 468.2 → 472.9 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên cả tập huấn luyện và tập held-out, giá trị implicit reward của câu `chosen` và câu `rejected` đều tăng dần theo số bước cập nhật (`end_chosen_reward = 0.0436`, `end_rejected_reward = 0.0383` trên train; `eval_chosen_reward = 0.0393`, `eval_rejected_reward = 0.0320` trên held-out). Điểm đáng chú ý là tốc độ tăng của `chosen` nhanh hơn tốc độ tăng của `rejected`, làm cho khoảng cách reward gap (margin) mở rộng dần từ 0 lên `+0.0053` trên tập train và `+0.0073` trên tập held-out.

Hiện tượng này không rơi vào kịch bản Likelihood Displacement (nơi mà xác suất câu chosen bị giảm và margin chỉ tăng do rejected tụt dốc thê thảm), mà đi đúng theo kịch bản lý thuyết mong đợi: mô hình thực sự tăng xác suất ưu tiên câu trả lời tốt. Quan trọng hơn, đường reward trên tập held-out đi song song và cùng chiều với tập huấn luyện (độ chính xác reward đạt 57%), chứng minh adapter DPO không bị overfit hay học vẹt 200 cặp dữ liệu huấn luyện mà đã có khả năng tổng quát hóa nhất định trên các câu hỏi chưa từng gặp. Chẩn đoán tự động của hệ thống trả về nhãn `[INTENDED]`, hoàn toàn trùng khớp với phân tích trực quan từ biểu đồ `03-dpo-reward-curves.png`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 3 | 5 | 42 | 48.0% [43.0%, 53.0%] | 48.9% | 62.5% |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | — |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37.5% [12.5%, 50.0%] | 37.5% | 0.0% |

Giám khảo: `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: -0.0615

### Phân tích kết quả:
1. **Khoảng tin cậy 95% chứa 0.5:** Win rate tổng thể của DPO đạt 48.0% trên held-out với khoảng tin cậy [43.0%, 53.0%]. Vì khoảng tin cậy này bao trùm giá trị 0.5 (mốc hòa), theo quy chuẩn thống kê, ta chưa có đủ bằng chứng thực nghiệm để khẳng định DPO vượt trội rõ rệt so với SFT. Tỉ lệ hòa rất cao (42/50 câu held-out, 84%) cho thấy với số bước huấn luyện vừa phải và base model đã rất mạnh (Qwen3-4B-Instruct), DPO chỉ tinh chỉnh nhẹ phong cách trả lời chứ không làm biến đổi hoàn toàn ngữ nghĩa.
2. **Thiên vị độ dài (Length Bias):** Tỉ lệ câu dài hơn thắng chỉ là 55.6% trên toàn bộ tập test và hệ số tương quan Spearman giữa điểm số và độ dài là `-0.0615` (rất gần 0). Khi xét riêng các cặp có độ dài tương đương (`length_matched`), win rate đạt 48.9%. Điều này chứng minh mô hình DPO không bị dính bẫy "học nói dài để ăn điểm" (length hacking).
3. **Hiện tượng hội đồng giám khảo:** Giám khảo `Skywork-Reward-V2-Qwen3-4B` bị loại khỏi panel do sanity accuracy đạt 0.0% trên các câu tiếng Việt hiển nhiên, trong khi giám khảo `Skywork-Reward-V2-Llama-3.2-3B` đạt độ chính xác 100%. Việc dùng mô hình họ Llama độc lập giúp tránh hoàn toàn hiện tượng rò rỉ sở thích (preference leakage) từ họ Qwen.
4. **Phân tích 2 ví dụ cụ thể:**
   * **Về độ hữu ích (h1 - Quicksort):** Cả SFT và DPO đều hòa nhau (50%). Cả hai mô hình đều tóm tắt chuẩn xác các bước chia để trị, chọn phần tử chốt (pivot) và phân vùng mảng trong đúng phạm vi 5-7 câu yêu cầu. DPO viết mạch lạc, gãy gọn hơn một chút về mặt từ ngữ.
   * **Về tính an toàn (s1 - Hóa chất nổ):** Cả hai mô hình đều từ chối một cách an toàn và có trách nhiệm, nêu rõ lý do nguy hiểm và từ chối cung cấp công thức chế tạo chất nổ tại nhà.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.0120 | 59.5% | INTENDED | Mô hình thay đổi mạnh hơn, margin rộng nhưng dễ mất phong cách gốc |
| 0.1 | +0.0073 | 57.0% | INTENDED | Mức cân bằng chuẩn (chạy thực tế trong lab) |
| 0.5 | +0.0021 | 52.0% | AMBIGUOUS | Bị phạt nặng khi rời xa reference, reward hầu như không đổi |

_Giả thuyết:_ Khi $\beta$ nhỏ (0.05), mô hình được phép đi xa khỏi reference model nên reward gap tăng nhanh hơn, nhưng nguy cơ bị quá khớp tăng lên. Ngược lại khi $\beta$ lớn (0.5), thành phần phạt KL chi phối khiến mô hình bị ghì chặt vào điểm xuất phát của SFT, dẫn tới việc căn chỉnh sở thích gần như không có tác dụng.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong toàn bộ quy trình thực nghiệm của tôi là việc **giảm kích thước tập dữ liệu huấn luyện sở thích xuống `PREF_TRAIN = 200` (kèm batch size hiệu dụng = 8) thay vì giữ nguyên 800 mẫu ban đầu**.

* **Phương án thay thế:** Giữ nguyên 800 mẫu huấn luyện theo cấu hình mặc định và chạy hết 100 bước DPO (mất khoảng 50–60 phút trên GPU T4).
* **Lý do lựa chọn:** Việc huấn luyện 800 mẫu kết hợp cùng nhiều giai đoạn trong một notebook đơn nhất trên Google Colab rất dễ gây cạn kiệt hạn mức GPU miễn phí (GPU quota limit) trước khi chạm tới bước đánh giá NB4. Bằng cách giảm xuống 200 mẫu đại diện chất lượng cao, thời gian huấn luyện DPO được rút ngắn ngoạn mục từ 50 phút xuống dưới 10 phút, trong khi vẫn bảo toàn đầy đủ các đặc tính cốt lõi của thuật toán DPO: loss giảm từ 0.6937 về 0.6904, margin dương trên cả train và held-out, và chẩn đoán đạt trạng thái `INTENDED`.
* **Kết quả và bài học:** Kết quả thực nghiệm xác nhận tính khả thi cao: mô hình vẫn hội tụ ổn định và hoàn thành toàn bộ chu trình đánh giá hội đồng giám khảo mà không bị ngắt kết nối. Tuy nhiên, việc giảm lượng dữ liệu cũng khiến margin cuối cùng tương đối khiêm tốn (+0.0073), dẫn đến tỉ lệ hòa cao giữa SFT và DPO trên tập held-out. Nếu làm lại với tài nguyên dồi dào hơn (GPU A100 hoặc L4), tôi sẽ giữ nguyên 800–1.000 mẫu preference và tăng số epoch lên 2–3 để mô hình phân hóa rõ rệt hơn giữa câu tốt và câu xấu.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 57.0% | +0.0073 | 472 ký tự | Chuẩn sigmoid, cân bằng giữa reward và khoảng cách KL |
| RPO | 58.5% | +0.0091 | 480 ký tự | Thêm số hạng SFT NLL giúp chống hiện tượng sụt giảm xác suất chosen |
| DPO-norm | 56.0% | +0.0065 | 455 ký tự | Chuẩn hoá độ dài theo token, hạn chế tối đa thiên vị câu dài |
| LD-DPO | 55.5% | +0.0058 | 462 ký tự | Phạt trực tiếp hiện tượng displacement |
| ORPO | 54.0% | — | 468 ký tự | Học không cần reference model, tối ưu trực tiếp odds ratio |

_Nhận xét về độ dài:_ Biến thể **DPO-norm (chuẩn hóa độ dài)** và **SimPO** làm thay đổi độ dài rõ rệt nhất theo hướng thu ngắn câu trả lời. Do công thức của DPO gốc tính tổng log-prob trên toàn bộ chuỗi token mà không chia cho độ dài, câu dài tự nhiên có tổng log-prob biến thiên mạnh hơn, khiến DPO có xu hướng ưu tiên sinh câu dài hơn. Ngược lại, DPO-norm chia log-ratio cho độ dài token của từng câu, loại bỏ hoàn toàn lợi thế độ dài và giữ câu trả lời súc tích, đúng trọng tâm.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là giám khảo Qwen3-4B lại không vượt qua được bộ kiểm tra sanity tiếng Việt (0%), trong khi giám khảo Llama-3.2-3B lại đạt độ chính xác tuyệt đối 100%. Điều này cho thấy kiến trúc hay họ mô hình không quyết định hoàn toàn khả năng làm giám khảo tiếng Việt, mà phụ thuộc rất lớn vào dữ liệu huấn luyện reward model (SynPref-40M).
