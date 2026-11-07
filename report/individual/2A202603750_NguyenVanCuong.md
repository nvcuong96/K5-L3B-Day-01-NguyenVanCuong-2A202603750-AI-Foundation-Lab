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

SEED=9836 · thử thách #3
Bộ kiểm tra: 15/15 đạt
  ✅ 1.1 vectorize
  ✅ 1.2 sigmoid / predict_proba
  ✅ 1.3 train_logreg
  ✅ 1.4 metrics
  ✅ 2.1 build_mlp
  ✅ 2.2 train_mlp
  ✅ 3.1–3.2 cosine / top_k_similar
  ✅ 3.3 knn_predict
  ✅ 4.2–4.3 prompt / parse_label
  ✅ 4.1 + 4.4 call_openai / llm_classify_batch
  ✅ tính tay: baseline_accuracy
  ✅ tính tay: sigmoid_0
  ✅ tính tay: f1_example
  ✅ tính tay: cosine_example
  ✅ tính tay: llm_cost_per_day

Dự đoán đã ghi:
  [14:07:09] ml_f1: F_1 8.5 precision cao hơn recall vì số lượng của spam trên tập test ít
  [14:07:22] ml_lr: lr=0.001 thì loss sẽ hội tụ chậm lr=50 thì gấp khúc
  [14:07:54] dl_curve: loss sẽ giảm đến một mức nhất đinh rồi chậm dần, hơn Phần 1
  [14:08:16] dl_relu: Nếu xóa thì nó tương đương với mô hình phần 1, F1 sẽ giảm
  [14:10:38] emb_search: có thể sẽ là spam
  [14:11:56] emb_tricky: 3.b sẽ đoán đúng nhiều hơn
  [14:12:00] llm_models: model F1 của model lớn cao hơn, model lớn đắt gấp đôi, có thể thắng được]
  [14:14:01] llm_fewshot: Nếu dùng tập train vào prompt thì có thể F1 giảm do gây nhiễu prompt, chi phí tăng gấp 5 lần do thêm 3000 từ
  [14:14:39] tradeoff: Trên đường Pareto: Bag of Words + Logistic Regression, lớn hơn 50ms: Embedding + kNN, LLM
  [14:14:40] hard_eval: 1. sẽ còn khoảng 0.5, 2. sẽ giảm ít 0.9, 3. giảm ít 0.9, 4. sẽ giảm mạnh
Thiếu dự đoán: không

                                     f1  f1_SUB
1. BoW + LogReg (tự code)        0.905   0.919
2. BoW + MLP (deep learning)     0.949   0.974
3a. Encoder (MiniLM) + kNN       0.924   1.000
3b. Encoder (MiniLM) + LogReg    0.962   0.919
3c. Encoder (MiniLM) + MLP head  0.972   0.919
4a. LLM nhỏ zero-shot            0.930   0.930
4b. LLM lớn zero-shot            0.974   0.974
4c. LLM nhỏ few-shot             1.000   1.000

💾 Đã lưu lab1_report.json (chỉ chứa SEED, không chứa MSSV)

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
Việc thêm ví dụ mẫu (few-shot) từ tập train vào prompt sẽ làm phình chi phí gấp nhiều lần nhưng chất lượng có thể bị nhiễu. Thực tế, mặc dù F1-score trên SUB chỉ tăng lên chút, chi phí tính toán tăng hơn gấp đôi ( 0.00099lên 0.00200) do số lượng token đầu vào phình to. Điều này chỉ ra rằng trong môi trường sản xuất, việc tối ưu prompt tinh gọn quan trọng hơn nhồi nhét ví dụ thô.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

* BoW thường giảm nhiều nhất
* Ở Các nhóm né từ khóa tránh được các tự được xác định sẵn

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

Cách làm: tận dùng các hàm đã được viết sẵn ở phần 2 DL như build_mlp(), train_mlp(), predict_mlp(), metrics()

Kết quả:

Cấu hình 4 nơ-ron lớp ẩn:
  - Số tham số: 8,009
  - Thời gian huấn luyện: 2.26 giây
  - F1-score trên tập test: 0.949

Cấu hình 512 nơ-ron lớp ẩn:
  - Số tham số: 1,025,025
  - Thời gian huấn luyện: 27.60 giây
  - F1-score trên tập test: 0.949
Ta nên ưu tiên chọn mạng nhỏ (ví dụ lớp ẩn ít nơ-ron hơn) trong các trường hợp sau:

Tài nguyên phần cứng hạn chế (Resource Constrained): Chạy trên thiết bị di động, IoT, Edge devices nơi RAM và năng lượng tiêu thụ cực kỳ eo hẹp.
Yêu cầu độ trễ thấp (Low Latency): Khi cần tốc độ suy luận (Inference Speed) cực nhanh cho dịch vụ thời gian thực (Real-time).
Tránh Overfitting: Khi dữ liệu huấn luyện quá ít, mạng quá lớn sẽ dễ học vẹt (overfit). Mạng nhỏ đóng vai trò như một bộ điều khiển (regularizer) tự nhiên giúp tăng khả năng tổng quát hóa.
Tiết kiệm chi phí lưu trữ: Kích thước file trọng số (model checkpoint) nhỏ hơn nhiều lần giúp việc cập nhật qua mạng (OTA) dễ dàng hơn.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

Đáp ứng SLA độ trễ (p95 < 50 ms): Chỉ có các mô hình dựa trên Bag-of-Words đạt yêu cầu này.
Đáp ứng cả chất lượng và độ trễ: BoW + MLP (Deep Learning) là lựa chọn tối ưu nhất. Mô hình này vừa đáp ứng cực tốt SLA độ trễ (p95 chỉ khoảng ~4 ms), vừa đem lại chất lượng phân loại cao ( F1 -score đạt 0.949).
Ước tính chi phí/tháng khi sử dụng LLM:
Với 10 triệu tin nhắn/ngày, tương đương khoảng 300 triệu tin nhắn/tháng.
Sử dụng mô hình nhỏ gemini-3.5-flash-lite với đơn giá ước tính trong bảng kết quả là 0.025 đô / 1.000 tin nhắn.
Chi phí mỗi ngày: 250 đô/ngày.
Chi phí mỗi tháng: 7500 đo/tháng (khoảng 190 triệu VND/tháng).

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
