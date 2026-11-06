# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** … / … / `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=`cab` ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=`ImageNet-1K`

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   Cả bức ảnh. Record có `task` là `image_classification` và dùng model `yolo11n-cls.pt`, bản này chỉ cho điểm theo lớp chứ không cho vị trí vật thể. Trong record cũng không có box, còn `image_width` = 640 và `image_height` = 428 là kích thước của ảnh gốc.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   Đi kèm taxonomy/checkpoint. Danh sách lớp của ImageNet-1K được cố định từ lúc huấn luyện và lưu trong checkpoint. Khi chạy, model chỉ chấm điểm các lớp đó, code tra `class_id` trong `result.names` để lấy tên, nên model không trả về được lớp nào ngoài danh sách.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   Vì chỉ có tên lớp thì khó biết model đã chọn đầu ra nào và tên đó thuộc hệ nhãn nào. Ví dụ `cab` ở đây là lớp số 468 của ImageNet-1K, còn model detection lại dùng COCO-80 là một danh sách lớp khác.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   Guideline cần nói rõ thế nào là chủ thể chính của ảnh, để hai người gán độc lập vẫn ra cùng một lớp. Có thể xét lần lượt: vật được lấy nét, vật chiếm diện tích lớn nhất, rồi vật gần giữa ảnh nhất. Nếu vẫn không chọn được thì gán `ambiguous` hoặc `none_of_the_above` chứ không đoán. Các lớp dễ nhầm như `cab` và `minibus` cũng phải có định nghĩa.

6. Vì sao không được lấy `score` của model làm ground truth?

   Score chỉ là mức tự tin của model, nó có thể rất tự tin mà vẫn sai. `cab` được 0,51 chỉ có nghĩa là model chấm lớp này cao nhất, chưa chứng minh trong ảnh có taxi. Ground truth dùng để chấm model nên phải do người gán theo guideline; lấy dự đoán của model làm đáp án thì model lúc nào cũng đúng và không tìm ra được lỗi. Score chỉ nên dùng để chọn ra những ca cần xem lại.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=`person` ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   Tính từ góc trên-trái của ảnh (x tăng sang phải, y tăng xuống dưới), góc trên-trái của box cách mép trái khoảng 385 pixel và cách mép trên khoảng 69 pixel. Góc dưới-phải cách mép trái khoảng 499 pixel và cách mép trên khoảng 349 pixel. Box rộng khoảng 114 pixel, cao khoảng 280 pixel, đó là người đeo tạp dề đang đứng quay lưng.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   Mình chọn `0.35` và `0.60`, đếm các record `kitchen` trong JSON:

   | Ngưỡng | Số prediction | Các lớp |
   | --- | --- | --- |
   | `0.35` | 11 | 2 `person`, 5 `bowl`, 2 `oven`, 2 `cup` |
   | `0.60` | 6 | 2 `person`, 2 `bowl`, 2 `oven` |

   Lên `0.60` thì mất 3 `bowl` và 2 `cup`. Trong đó có cả vật có thật, như tô thức ăn ở `[59.29, 288.92, 133.46, 328.98]` (score 0.499948). Vì vậy không thể dùng ngưỡng để quyết định vật nào được gán nhãn trong ground truth.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   Reviewer phải xem nhiều hộp hơn, nhất là hộp điểm thấp: hộp đó có đúng là vật thật không, có trùng với hộp khác không, lớp có đúng không, có ôm sát vật không. Hạ ngưỡng cũng không có nghĩa là model đã tìm đủ, nên vẫn phải soi xem còn vật nào bị bỏ sót.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   Mỗi vật thuộc danh sách lớp được vẽ một box ôm sát phần nhìn thấy, không đoán thêm phần bị che hoặc nằm ngoài ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   Bàn tay ở mép trái ảnh, thân người nằm ngoài khung hình (`person`, score 0.610931, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78], `x_min` gần bằng 0). Mình sẽ viết thành quy tắc trong guideline vì vật bị cắt mép gặp ở nhiều ảnh: nếu còn nhận ra được lớp thì vẫn gán, box chỉ vẽ tới mép ảnh và đánh dấu `truncated`. Nếu phần nhìn thấy ít đến mức không biết là vật gì thì mình mới hỏi Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=`kitchen-001` ; `class_name`=`person` ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   Box chỉ là hình chữ nhật bao quanh vật, còn polygon đi theo đường viền nên thấy được hình dạng thật. Với `kitchen-001`, polygon chỉ chiếm khoảng 57% diện tích box, phần còn lại của box không thuộc về người. Vật nằm sát nhau cũng tách được rõ hơn: box của `kitchen-003` và `kitchen-006` giao nhau khoảng 714 pixel², còn polygon chỉ giao nhau khoảng 41 pixel².

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   `instance_id` dùng để phân biệt từng vật trong output. Class ID chỉ cho biết vật thuộc lớp nào: `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010` đều có `class_id` = 45 (`bowl`) nhưng là 4 cái khác nhau. Tracking ID thì theo một vật qua nhiều khung hình, còn `instance_id` chỉ là số thứ tự trong kết quả của một ảnh, nên `001` ở ảnh khác hay lần chạy khác không phải cùng một vật.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   Vẽ biên sát phần nhìn thấy của từng instance, không lấn sang nền hoặc vật khác, và mỗi pixel chỉ thuộc một instance.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   Tô thức ăn `kitchen-003` chạm đĩa bánh `kitchen-006` trên bàn, hai polygon chồng nhau khoảng 41 pixel. Mình dùng guideline vì hai vật chạm nhau rất hay gặp: pixel ở chỗ chồng nhau thuộc về vật nằm phía trước, và hai vật vẫn là hai instance dù cùng lớp `bowl`. Nếu không nhìn ra vật nào ở trước thì mình mới hỏi Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho cả ảnh | Ảnh `traffic` ra top-1 là `cab` (0.510915) nhưng trên ảnh chủ yếu là xe buýt và ô tô con; detection cùng ảnh ra 14 `car`, 8 `bus`, 4 `person`. Nhiều xe như vậy nên khó chọn một lớp | Xem cả ảnh, chọn chủ thể chính theo guideline rồi gán một lớp; không chắc thì escalation | Nhãn có đúng guideline không, nhất là lớp dễ nhầm như `cab`/`minibus` |
| Phát hiện vật thể | Một lớp và một box cho mỗi vật | Ảnh `kitchen`: box `person` `[0.12, 263.18, 61.55, 310.78]` chỉ thấy bàn tay vì thân người ở ngoài ảnh. Tô thức ăn (0.499948) là vật thật nhưng bị loại ở ngưỡng `0.60` | Vẽ box sát phần nhìn thấy cho mọi vật trong danh sách lớp, kể cả vật model bỏ sót | Vật bị sót, box thừa hoặc trùng, sai lớp, box không sát; sai thì trả lại làm lại |
| Instance segmentation | Một lớp và một mask/polygon riêng cho mỗi instance | Ảnh `kitchen`: polygon `kitchen-003` và `kitchen-006` chồng nhau khoảng 41 pixel. Polygon bàn `kitchen-005` có đường thẳng dài nối các mảng rời và lan xuống gầm bàn | Vẽ biên sát vật, tách riêng từng instance; không rõ vật nào ở trước thì escalation | Biên có sát không, instance có bị gộp hay tách sai không, có pixel nào thuộc hai instance không; sai thì trả lại làm lại |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Mình chỉ dùng ba ảnh COCO có sẵn trong notebook, không đưa ảnh riêng hay dữ liệu nhạy cảm vào Colab và không ghi thông tin cá nhân vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Mình sẽ dừng lại, không xử lý tiếp và không đẩy lên repo, rồi báo Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` 
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
