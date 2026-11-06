# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026 (giờ Việt Nam).

**Runtime Colab:** GPU NVIDIA T4, device CUDA.

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`.

**Ngưỡng detection/segmentation tạo JSON:** 0.35. So sánh detection thêm ở 0.20 và 0.60 theo notebook.

**Thay đổi so với notebook nguồn:** Giữ nguyên các ô suy luận, checkpoint, package, ảnh và threshold của bản v1.0.3. Thêm một ô phụ để in record kitchen và tải ZIP bằng chứng trước khi hoàn thiện báo cáo. Ô upload ban đầu được hủy khi chưa có báo cáo; việc này không thay đổi JSON/PNG đã validation. Báo cáo được hoàn thiện sau khi đọc JSON và xem cả ba PNG; có hỗ trợ AI trong thao tác và soạn thảo.

## 1. Phân loại ảnh

Nguồn: [classification_predictions.json](day1_lab_outputs/classification_predictions.json), [classification_top5.png](day1_lab_outputs/visuals/classification_top5.png).

1. Record `sample_id=traffic`: `class_id=468`, `class_name=cab`, `rank=1`, `score=0.510915`, `taxonomy_name=ImageNet-1K`. Ảnh COCO 210273, kích thước 640 × 428 pixel.
2. Đây là prediction cho **toàn ảnh**: classification xếp hạng các lớp mà không trả box định vị từng xe. Ảnh có nhiều xe buýt, ô tô và người; nhãn cab không mô tả đầy đủ cảnh giao thông và chưa được xác nhận đúng.
3. Class list đi kèm taxonomy/checkpoint ImageNet-1K, không do model tự tạo tên lớp mới khi suy luận.
4. `class_id` là mã trong bảng lớp; `class_name` giúp người đọc hiểu; `taxonomy_name` xác định hệ lớp để tránh hiểu nhầm mã/tên giữa các bộ dữ liệu. Mã 468 chỉ có nghĩa khi biết taxonomy và ánh xạ lớp tương ứng.
5. Guideline single-label phải nêu đơn vị phân loại, phạm vi lớp, tiêu chí ưu tiên chủ thể/chủ đề, cách xử lý nhiều chủ thể, hòa lựa chọn và trường hợp không có lớp phù hợp. Với cảnh traffic, cần quy định chọn theo cảnh hay phương tiện chủ đạo; không tự chọn cab chỉ vì model xếp cao nhất.
6. Score chỉ là giá trị model dùng xếp hạng, không chứng minh nhãn đúng và không đo chất lượng nhãn người. Ground truth cần người đối chiếu ảnh theo guideline, được reviewer kiểm tra. Không thể diễn giải 0.510915 là “ground truth đúng 51%”.

## 2. Phát hiện vật thể

Nguồn: [detection_predictions.json](day1_lab_outputs/detection_predictions.json), [detection_predictions.png](day1_lab_outputs/visuals/detection_predictions.png), output ô so sánh threshold.

1. Chọn record `sample_id=kitchen`, COCO 397133, ảnh 640 × 427: `class_id=0`, `class_name=person`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68`, taxonomy COCO-80, threshold 0.35.
2. Gốc tọa độ ở góc trên trái; x tăng sang phải, y tăng xuống dưới. Box của người đứng bên phải từ góc trên trái (385.33, 69.24) đến góc dưới phải (498.92, 348.92), rộng 113.58 và cao 279.68 pixel theo JSON. Trừ các tọa độ đã làm tròn cho chiều rộng 113.59; sai lệch 0.01 xuất hiện vì notebook tính kích thước từ tọa độ thô rồi mới làm tròn.
3. Số prediction trên **kitchen**, không phải tổng ba ảnh:

   | Threshold | Prediction |
   | --- | ---: |
   | 0.20 | 17 |
   | 0.35 | 11 |
   | 0.60 | 6 |

   Ở 0.20 có thêm các prediction như spoon, potted plant, dining table, bottle; ở 0.60 còn person, bowl, oven. Đây là thay đổi lọc prediction, không chứng minh các vật bị loại không tồn tại.
4. Khi hạ ngưỡng, reviewer phải đối chiếu thêm box có score thấp, kiểm tra sai lớp, vật giả, box trùng và độ khít box. Đồng thời vẫn tìm object bị bỏ sót; không chỉ kiểm tra những gì model đưa ra.
5. **Đề xuất quy tắc box (visible box):** mỗi object thuộc phạm vi có một box ôm sát phần nhìn thấy, chứa đầy đủ phần đó, không thêm nền không cần thiết và không vượt khung ảnh. Với vật bị che, không tự suy đoán phần khuất. Đây là quy tắc đề xuất; nếu guideline chính thức dùng amodal box thì phải theo quy định chính thức.
6. Ở mép trái phía dưới ảnh kitchen, chỉ thấy một phần người và tay; phần còn lại nằm ngoài ảnh. Nếu guideline đã quy định object cắt mép thì áp dụng; nếu chưa rõ có gán người chỉ lộ một phần hay không, escalation cho Lab Coach để chốt tiêu chí. Không bỏ đối tượng chỉ vì bị cắt hoặc score thấp, cũng không vẽ thêm cơ thể ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng

Nguồn: [segmentation_predictions.json](day1_lab_outputs/segmentation_predictions.json), [segmentation_prediction.png](day1_lab_outputs/visuals/segmentation_prediction.png).

1. Record `sample_id=kitchen`, `instance_id=kitchen-001`, `class_id=0`, `class_name=person`, `score=0.899318`, `polygon_point_count=348`. Năm điểm đầu `polygon_xy`: `[[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]]`. Đơn vị pixel, taxonomy COCO-80, threshold 0.35.
2. Polygon mô tả đường biên của từng instance, cho biết hình dạng và vùng ảnh thuộc object. Box chỉ là hình chữ nhật bao quanh nên chứa thêm nền, ví dụ khoảng trống giữa hai chân người.
3. `instance_id` phân biệt các object riêng trong bộ output. Hai người có thể cùng class ID 0 nhưng khác instance ID. Số thứ tự kitchen-001 không xác định lớp và không bảo đảm theo dõi cùng người qua frame hay lần chạy khác; đây không phải tracking ID.
4. **Đề xuất quy tắc mask:** bám phần biên nhìn thấy ở độ phóng đại đủ rõ; tách từng instance, tránh ăn sang nền hoặc vật khác, không tự nối qua vùng bị che; không đưa vùng nền giữa hai chân vào mask người. Phần che khuất, lỗ và vùng rời phải theo quy ước mask chính thức; chỗ chưa có quy định cần hỏi trước khi gán.
5. Ở vùng bàn phía trước bên trái, nhiều bát, khăn và dụng cụ đặt sát/che mặt bàn. Mask dining table phủ lên một số vùng khăn và đĩa, cần đối chiếu quy ước visible mask và cách biểu diễn lỗ/vùng khuất. Reviewer cần phóng to so với ảnh gốc, không kết luận mask đúng vì vùng màu nhìn liền nhau. Nếu guideline chưa quy định tách vật đặt trên bàn, escalation cho Lab Coach; không tự coi toàn bộ vùng trong box là mặt bàn.

## 4. Vòng đời và kiểm tra chất lượng

Trong production: ảnh thô được kiểm tra phạm vi và quyền sử dụng; guideline thống nhất lớp/quy tắc hình học; annotator tạo ground truth; reviewer QC và trả rework; nhãn được chấp nhận mới dùng huấn luyện; prediction của model được đối chiếu để phát hiện lỗi. Lab này chỉ suy luận bằng checkpoint có sẵn, không tạo bộ ground truth và không huấn luyện.

| Tác vụ | Đơn vị ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cấp ảnh theo guideline single-label | traffic chứa nhiều phương tiện; cab hạng 1 score 0.510915 không đại diện chắc chắn cho cả cảnh | Chọn theo quy tắc chủ thể/chủ đề; hỏi nếu taxonomy không phù hợp | So ảnh và taxonomy; kiểm tra lý do chọn lớp, không duyệt bằng score |
| Phát hiện vật thể | Class và box riêng mỗi object | Người chỉ lộ một phần ở mép trái dưới ảnh kitchen; threshold 0.20 giữ 17 box so với 6 ở 0.60 | Gán phần nhìn thấy theo quy định, đánh dấu ca cắt mép cần hỏi | Kiểm tra bỏ sót, trùng, sai lớp, tọa độ và tiêu chí object cắt mép |
| Instance segmentation | Class và mask/polygon riêng mỗi instance | Vật đặt trên bàn che mặt bàn; mask dining table có vùng cần kiểm tra quanh khăn/đĩa | Tách biên và instance theo guideline; không đoán phần khuất | So mask với ảnh gốc, kiểm tra nền/lỗ, vùng chồng, biên người và vật tiếp xúc; trả rework nếu sai |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai cố định trong notebook, giữ attribution và giấy phép; không bổ sung ảnh cá nhân, dữ liệu nội bộ, thông tin định danh hoặc token vào báo cáo/output công khai.
2. Nếu gặp ảnh/dữ liệu ngoài phạm vi, dừng xử lý và upload, báo Lab Coach/giảng viên qua kênh được phép, không gửi dữ liệu đó lên repo để hỏi.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` đã điền và trích đúng record traffic/kitchen.
- [x] `day1_lab_outputs/classification_predictions.json` — 15 record (5 mỗi ảnh).
- [x] `day1_lab_outputs/detection_predictions.json` — 53 record tổng ba ảnh.
- [x] `day1_lab_outputs/segmentation_predictions.json` — 49 record tổng ba ảnh.
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`.
- [x] `day1_lab_outputs/visuals/classification_top5.png`.
- [x] `day1_lab_outputs/visuals/detection_predictions.png`.
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`.
- [x] Ô validation báo `PASS: đủ 3 JSON, 3 PNG, attribution và instance_id duy nhất.`
- [x] Checksum ba model và ba ảnh đều PASS; SHA-256 model đầy đủ có trong JSON, ảnh được cố định bằng sample ID/COCO ID và checksum trong notebook nguồn.
- [x] Đã xem cả ba PNG và nêu điểm mơ hồ cùng hành động annotator/reviewer.
- [x] Không thêm họ tên, MSSV, email hoặc dữ liệu nhạy cảm vào báo cáo/output.
- [x] Đã nộp link repository trên LMS.
