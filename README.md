# Driver Monitoring AI

<div align="center">

**Language / Ngôn Ngữ:** 
[Tiếng Việt](#vietnamese) | [English](#english)

</div>

---

<a id="vietnamese"></a>

# Tiếng Việt

## Driver Monitoring AI

Hệ thống giám sát tài xế bằng trí tuệ nhân tạo để phát hiện buồn ngủ và ngáp khi lái xe, giúp nâng cao an toàn giao thông.

---

## Mục Đích

Dự án này phát triển một hệ thống AI tiên tiến giúp:
- **Phát hiện buồn ngủ** (Closed Eyes) - Nhắm mắt liên tục > 20 frames
- **Phát hiện ngáp** (Yawn) - Ngáp liên tục > 35 frames
- **Cảnh báo thời gian thực** bằng âm thanh và hình ảnh
- **Giao diện web** để xem lịch sử và cài đặt

---

## Tính Năng Chính

- Phát hiện mắt đóng với độ chính xác 90.69%
- Phát hiện ngáp với độ chính xác 95.14%
- Cảnh báo âm thanh (tần số khác nhau cho mắt và ngáp)
- Giao diện web Flask hiển thị realtime
- Tính toán EAR (Eye Aspect Ratio) trên giao diện
- Calibrate threshold tự động
- Debug tool để kiểm tra vùng nhận diện

---

## Công Nghệ Sử Dụng

| Công Nghệ | Phần Trăm | Mục Đích |
|-----------|----------|---------|
| **Python** | 51% | Backend AI, xử lý video |
| **HTML** | 28.3% | Giao diện web |
| **JavaScript** | 16.1% | Interactivity, chart |
| **CSS** | 4.6% | Styling |

### Thư Viện Chính
- **OpenCV** (cv2): Xử lý video & nhận diện khuôn mặt
- **MediaPipe**: Landmark detection (mắt, miệng)
- **Scikit-learn** (joblib): SVM models cho eye/yawn
- **Flask**: Web server
- **NumPy**: Xử lý dữ liệu

---

## Cấu Hình Hiện Tại

### Phát Hiện Mắt
- **Brightness Threshold**: < 102 → Mắt đóng
- **Frame Threshold**: 20 frames liên tục
- **Confidence Min**: 0.6
- **Accuracy**: 90.69% | Precision: 89.36% | Recall: 94.03%

### Phát Hiện Ngáp
- **Mouth ROI**: y=[0.72:0.82], x=[0.28:0.72]
- **Frame Threshold**: 35 frames liên tục
- **Confidence Min**: 1.15
- **Accuracy**: 95.14% | Precision: 96.12% | Recall: 94.66%

---

## Hướng Dẫn Cài Đặt

### Yêu Cầu
- Python 3.8+
- Webcam
- Thư viện trong requirements.txt

### Bước 1: Clone Repository
```bash
git clone https://github.com/CMTran2005/Driver_Monitoring_AI.git
cd Driver_Monitoring_AI
```

### Bước 2: Tạo Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn Bị Model
Đảm bảo thư mục `trainer/` chứa:
- `eye_model.pkl` - Model phát hiện mắt
- `yawn_model.pkl` - Model phát hiện ngáp

---

## Cách Sử Dụng

### Script Chính: detect.py
Chạy hệ thống phát hiện realtime:
```bash
python detect.py
```

Chức năng:
- Detect mắt & ngáp realtime từ webcam
- Hiển thị cảnh báo khi phát hiện
- Phát âm thanh cảnh báo (2500Hz cho mắt, 1500Hz cho ngáp)
- Nhấn 'q' để thoát

---

### Script Debug: diagnostic.py
Kiểm tra vùng nhận diện:
```bash
python diagnostic.py
```

Hiển thị:
- **XANH LÒNG**: Vùng mắt được detect
- **CYAN**: Vùng fallback check brightness
- **VÀNG**: Vùng miệng (mouth ROI)
- Confidence score & brightness value

---

### Script Calibrate: calibrate_eyes.py
Tìm threshold mắt mới:
```bash
python calibrate_eyes.py
```

Chức năng:
- Ghi nhận brightness khi MỞ & NHẮM mắt
- Tính toán threshold tự động
- Đề xuất giá trị ngưỡng mới

---

### Web App: app.py
Chạy giao diện web:
```bash
python app.py
```

Truy cập: http://localhost:5000

Tính năng:
- Realtime video feed từ webcam
- Trạng thái tài xế (tỉnh táo / buồn ngủ)
- EAR indicator
- Lịch sử & cài đặt

---

## Cấu Trúc Dự Án

```
Driver_Monitoring_AI/
├── detect.py              # Script chính phát hiện
├── app.py                 # Web server Flask
├── test_inference.py      # Test model
├── diagnostic.py          # Debug tool
├── calibrate_eyes.py      # Calibrate threshold
├── CONFIG.py              # Cấu hình tham số
├── SETUP.md               # Hướng dẫn chạy chi tiết
├── requirements.txt       # Python dependencies
├── trainer/               # Thư mục chứa models
│   ├── eye_model.pkl      # Model phát hiện mắt
│   └── yawn_model.pkl     # Model phát hiện ngáp
├── templates/             # HTML templates
│   ├── index.html
│   ├── history.html
│   └── settings.html
├── static/                # CSS, JS, images
└── database/              # Dữ liệu lưu trữ
```

---

## Tùy Chỉnh Cấu Hình

### Vùng Mắt Không Chính Xác
Chỉnh sửa trong `CONFIG.py` hoặc `detect.py`:
```python
EYE_CONF_THRESH = 0.6      # Tăng để chặt chẽ hơn
EYE_LIMIT = 20             # Tăng nếu quá nhạy
```

### Vùng Miệng Không Chính Xác
```python
mouth_roi = {
    'y_start': 0.72,   # Điều chỉnh vị trí Y
    'y_end': 0.82,
    'x_start': 0.28,   # Điều chỉnh vị trí X
    'x_end': 0.72,
}
```

### Cảnh Báo Quá Nhạy/Chậm
```python
EYE_LIMIT = 20         # Tăng → cảnh báo chậm hơn
YAWN_LIMIT = 35        # Tăng → cảnh báo chậm hơn
```

---

## Hiệu Năng

**Eye Detection:**
- Accuracy: 90.69%
- Precision: 89.36%
- Recall: 94.03%
- F1-Score: 91.64%

**Yawn Detection:**
- Accuracy: 95.14%
- Precision: 96.12%
- Recall: 94.66%
- F1-Score: 95.38%

---

## Cộng Tác Viên

| Tên | GitHub | Vai Trò |
|-----|--------|---------|
| CMTran2005 | [@CMTran2005](https://github.com/CMTran2005) | Project Lead + AI Training |
| Trần Quang Huy | [@huydz252](https://github.com/huydz252) | Data & Design |
| Thịnh | [@thinhk16k5](https://github.com/thinhk16k5) | Web & UI |

---

## Quy Trình Thử Nghiệm

1. **Chạy script chính:**
   ```bash
   python detect.py
   ```
   - Mở mắt bình thường → kiểm tra
   - Nhắm mắt 2-3 giây → xem có báo không
   - Ngáp 2-3 lần → xem có báo không

2. **Nếu có vấn đề, dùng diagnostic:**
   ```bash
   python diagnostic.py
   ```

3. **Nếu cần calibrate lại:**
   ```bash
   python calibrate_eyes.py
   ```

---

## Đóng Góp

Chúng tôi hoan nghênh các đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## License

Dự án này được cấp phép dưới [MIT License](LICENSE)

---

## Liên Hệ

- **Project Lead + AI Training**: CMTran2005 - [@CMTran2005](https://github.com/CMTran2005)
- **Data & Design**: Trần Quang Huy - [@huydz252](https://github.com/huydz252)
- **Web & UI**: Thịnh - [@thinhk16k5](https://github.com/thinhk16k5)

---

## Lưu Ý Về An Toàn

Hệ thống này được phát triển để **hỗ trợ** tài xế, không phải để **thay thế** sự chú ý của họ. Luôn tuân thủ luật giao thông và quy định địa phương.

---

Nếu bạn thích dự án này, vui lòng cho nó một star!

---

---

<a id="english"></a>

# English

## Driver Monitoring AI

An AI-based driver monitoring system that detects drowsiness and yawning while driving to enhance road safety.

---

## Purpose

This project develops an advanced AI system that helps:
- **Detect drowsiness** (Closed Eyes) - Continuous eye closure > 20 frames
- **Detect yawning** - Continuous yawning > 35 frames
- **Real-time alerts** with sound and visual warnings
- **Web interface** for history and settings management

---

## Key Features

- Eye detection with 90.69% accuracy
- Yawn detection with 95.14% accuracy
- Audio alerts (different frequencies for eyes and yawn)
- Flask web interface with realtime streaming
- EAR (Eye Aspect Ratio) calculation
- Automatic threshold calibration
- Debug tool for region inspection

---

## Technology Stack

| Technology | Percentage | Purpose |
|-----------|-----------|---------|
| **Python** | 51% | Backend AI, video processing |
| **HTML** | 28.3% | Web interface |
| **JavaScript** | 16.1% | Interactivity, charts |
| **CSS** | 4.6% | Styling |

### Main Libraries
- **OpenCV** (cv2): Video processing & face detection
- **MediaPipe**: Landmark detection (eyes, mouth)
- **Scikit-learn** (joblib): SVM models for eye/yawn
- **Flask**: Web server
- **NumPy**: Data processing

---

## Current Configuration

### Eye Detection
- **Brightness Threshold**: < 102 → Eyes closed
- **Frame Threshold**: 20 consecutive frames
- **Confidence Min**: 0.6
- **Accuracy**: 90.69% | Precision: 89.36% | Recall: 94.03%

### Yawn Detection
- **Mouth ROI**: y=[0.72:0.82], x=[0.28:0.72]
- **Frame Threshold**: 35 consecutive frames
- **Confidence Min**: 1.15
- **Accuracy**: 95.14% | Precision: 96.12% | Recall: 94.66%

---

## Installation Guide

### Requirements
- Python 3.8+
- Webcam
- Libraries from requirements.txt

### Step 1: Clone Repository
```bash
git clone https://github.com/CMTran2005/Driver_Monitoring_AI.git
cd Driver_Monitoring_AI
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Models
Ensure `trainer/` folder contains:
- `eye_model.pkl` - Eye detection model
- `yawn_model.pkl` - Yawn detection model

---

## Usage

### Main Script: detect.py
Run the realtime detection system:
```bash
python detect.py
```

Features:
- Realtime eye & yawn detection from webcam
- Visual alerts when detected
- Audio alerts (2500Hz for eyes, 1500Hz for yawn)
- Press 'q' to exit

---

### Debug Script: diagnostic.py
Inspect detection regions:
```bash
python diagnostic.py
```

Displays:
- **GREEN**: Eye detection region
- **CYAN**: Fallback brightness check region
- **YELLOW**: Mouth ROI
- Confidence scores & brightness values

---

### Calibrate Script: calibrate_eyes.py
Find new eye threshold:
```bash
python calibrate_eyes.py
```

Features:
- Record brightness with OPEN eyes
- Record brightness with CLOSED eyes
- Calculate optimal threshold automatically

---

### Web App: app.py
Run web interface:
```bash
python app.py
```

Access: http://localhost:5000

Features:
- Realtime webcam feed
- Driver status (alert/awake)
- EAR indicator
- History & settings

---

## Project Structure

```
Driver_Monitoring_AI/
├── detect.py              # Main detection script
├── app.py                 # Flask web server
├── test_inference.py      # Model testing
├── diagnostic.py          # Debug tool
├── calibrate_eyes.py      # Threshold calibration
├── CONFIG.py              # Configuration parameters
├── SETUP.md               # Detailed setup guide
├── requirements.txt       # Python dependencies
├── trainer/               # Models folder
│   ├── eye_model.pkl      # Eye detection model
│   └── yawn_model.pkl     # Yawn detection model
├── templates/             # HTML templates
│   ├── index.html
│   ├── history.html
│   └── settings.html
├── static/                # CSS, JS, images
└── database/              # Data storage
```

---

## Customization

### Eye Region Not Accurate
Edit in `CONFIG.py` or `detect.py`:
```python
EYE_CONF_THRESH = 0.6      # Increase for stricter detection
EYE_LIMIT = 20             # Increase if too sensitive
```

### Mouth Region Not Accurate
```python
mouth_roi = {
    'y_start': 0.72,   # Adjust Y position
    'y_end': 0.82,
    'x_start': 0.28,   # Adjust X position
    'x_end': 0.72,
}
```

### Alerts Too Sensitive/Slow
```python
EYE_LIMIT = 20         # Increase → slower alerts
YAWN_LIMIT = 35        # Increase → slower alerts
```

---

## Performance

**Eye Detection:**
- Accuracy: 90.69%
- Precision: 89.36%
- Recall: 94.03%
- F1-Score: 91.64%

**Yawn Detection:**
- Accuracy: 95.14%
- Precision: 96.12%
- Recall: 94.66%
- F1-Score: 95.38%

---

## Contributors

| Name | GitHub | Role |
|------|--------|------|
| CMTran2005 | [@CMTran2005](https://github.com/CMTran2005) | Project Lead + AI Training |
| Trần Quang Huy | [@huydz252](https://github.com/huydz252) | Data & Design |
| Thịnh | [@thinhk16k5](https://github.com/thinhk16k5) | Web & UI |

---

## Testing Procedure

1. **Run main script:**
   ```bash
   python detect.py
   ```
   - Open eyes normally → check
   - Close eyes for 2-3 seconds → verify alert
   - Yawn 2-3 times → verify alert

2. **If issues arise, use diagnostic:**
   ```bash
   python diagnostic.py
   ```

3. **If calibration needed:**
   ```bash
   python calibrate_eyes.py
   ```

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under [MIT License](LICENSE)

---

## Contact

- **Project Lead + AI Training**: CMTran2005 - [@CMTran2005](https://github.com/CMTran2005)
- **Data & Design**: Trần Quang Huy - [@huydz252](https://github.com/huydz252)
- **Web & UI**: Thịnh - [@thinhk16k5](https://github.com/thinhk16k5)

---

## Safety Notice

This system is developed to **assist** drivers, not to **replace** their attention. Always comply with traffic laws and local regulations.

---

If you like this project, please give it a star!

---

<div align="center">

[Back to Top](#driver-monitoring-ai)

</div>
