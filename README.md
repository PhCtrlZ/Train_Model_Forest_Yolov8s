#  Train_Model_Forest_YOLOv8s

Dự án này tập trung vào **huấn luyện mô hình YOLOv8s** cho bài toán **Object Detection / Segmentation trong môi trường rừng (Forest)**. Project được xây dựng phục vụ mục đích **nghiên cứu – học tập – NCKH**, sử dụng Python và thư viện Ultralytics YOLOv8.

---

##  Mục tiêu dự án

* Xây dựng dataset Forest (ảnh + nhãn)
* Huấn luyện mô hình **YOLOv8s** cho:

  * Object Detection
  * Segmentation (các phiên bản forest_seg)
* Đánh giá mô hình thông qua **mAP, IoU**
* Làm nền tảng cho các ứng dụng:

  * Giám sát rừng
  * Phát hiện đối tượng trong môi trường tự nhiên
  * Nghiên cứu thị giác máy tính

---

## 📂 Cấu trúc thư mục

```
Forest/
│── train/                # Ảnh + label training
│── valid/                # Ảnh + label validation
│── test/                 # Ảnh + label test
│
│── forest.v1i.yolov8/    # Dataset YOLOv8 (version 1)
│── forest_seg/           # Dataset segmentation chính
│── forest_seg_v2/        # Dataset segmentation thử nghiệm để giám loss và tăng mAP
│── forest_seg_v3/        # Dataset segmentation thử nghiệm để giám loss và tăng mAP
│
│── runs/                 # Kết quả train (YOLO auto-generate)
│── outputs/              # Output inference
│
│── data.yaml             # File cấu hình dataset YOLO
│── main.py               # File train / inference chính
│── download.py           # Script tải dataset (Roboflow)
│
│── README.dataset.txt    # Thông tin dataset
│── README.roboflow.txt   # Thông tin Roboflow
```

---

##  Công nghệ sử dụng

* **Python 3.9+**
* **Ultralytics YOLOv8**
* OpenCV
* Roboflow (dataset)
* Git & GitHub

---

## ⚙️ Cài đặt môi trường

```bash
pip install ultralytics opencv-python matplotlib
```

Kiểm tra YOLOv8:

```bash
yolo --version
```

---

##  Huấn luyện mô hình YOLOv8

Ví dụ train YOLOv8s:

```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640
```

Với Segmentation:

```bash
yolo task=segment mode=train model=yolov8s-seg.pt data=data.yaml epochs=100 imgsz=640
```

---

## Đánh giá mô hình

Các chỉ số chính:

* **mAP@0.5**
* **mAP@0.5:0.95**
* **IoU (Intersection over Union)**

Kết quả được lưu tại:

```
runs/detect/
runs/segment/
```

---

##  Inference (dự đoán ảnh)

```bash
yolo task=detect mode=predict model=best.pt source=test/images
```

Hoặc bằng Python:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("image.jpg", show=True)
```

---

##  Kết quả

* Mô hình học tốt các đặc trưng trong môi trường rừng
* Bounding box và segmentation mask bám sát đối tượng
* Có thể mở rộng cho video real-time



