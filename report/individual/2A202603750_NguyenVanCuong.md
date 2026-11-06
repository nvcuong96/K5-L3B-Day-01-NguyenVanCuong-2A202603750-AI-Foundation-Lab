# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603750**
- **Họ và tên:Nguyễn Văn Cường**
- **SEED 9836 ** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | v |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*SEED=9836 · thử thách #3
Bộ kiểm tra: 9/15 đạt
  ✅ 1.1 vectorize
  ✅ 1.2 sigmoid / predict_proba
  ✅ 1.3 train_logreg
  ✅ 1.4 metrics
  ✅ 2.1 build_mlp
  ✅ 2.2 train_mlp
  ✅ 3.1–3.2 cosine / top_k_similar
  ✅ 3.3 knn_predict
  ✅ 4.2–4.3 prompt / parse_label
  ❌ 4.1 + 4.4 call_openai / llm_classify_batch
  ❌ tính tay: baseline_accuracy
  ❌ tính tay: sigmoid_0
  ❌ tính tay: f1_example
  ❌ tính tay: cosine_example
  ❌ tính tay: llm_cost_per_day

Dự đoán đã ghi:
  [16:33:25] ml_f1: F_1 8.5 precision cao hơn recall vì số lượng của spam trên tập test ít
  [16:33:32] ml_lr: lr=0.001 thì loss sẽ hội tụ chậm lr=50 thì gấp khúc
  [16:33:47] dl_curve: loss sẽ giảm đến một mức nhất đinh rồi chậm dần, hơn Phần 1
  [16:33:57] dl_relu: Nếu xóa thì nó tương đương với mô hình phần 1, F1 sẽ giảm
  [16:37:09] emb_search: có thể sẽ là spam
  [16:38:35] emb_tricky: 3.b sẽ đoán đúng nhiều hơn
  [16:38:37] llm_models: model F1 của model lớn cao hơn, model lớn đắt gấp đôi, có thể thắng được]
  [16:41:09] llm_fewshot: Nếu dùng tập train vào prompt thì có thể F1 giảm do gây nhiễu prompt, chi phí tăng gấp 5 lần do thêm 3000 từ
  [16:41:40] tradeoff: Trên đường Pareto: Bag of Words + Logistic Regression, lớn hơn 50ms: Embedding + kNN, LLM
  [16:41:42] hard_eval: 1. sẽ còn khoảng 0.5, 2. sẽ giảm ít 0.9, 3. giảm ít 0.9, 4. sẽ giảm mạnh
Thiếu dự đoán: không

                                     f1  f1_SUB
1. BoW + LogReg (tự code)        0.905   0.919
2. BoW + MLP (deep learning)     0.949   0.974
3a. Encoder (MiniLM) + kNN       0.924   1.000
3b. Encoder (MiniLM) + LogReg    0.962   0.919
3c. Encoder (MiniLM) + MLP head  0.972   0.919
4a. LLM nhỏ zero-shot            0.667   0.667
4b. LLM lớn zero-shot            0.667   0.667
4c. LLM nhỏ few-shot             0.667   0.667

💾 Đã lưu lab1_report.json (chỉ chứa SEED, không chứa MSSV)*

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
