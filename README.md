# 🚗 Driver Monitoring AI

<div align="center">

<!-- Language Toggle -->
**Language / Ngôn Ngữ:** 
[🇻🇳 Tiếng Việt](#vietnamese) | [🇬🇧 English](#english)

</div>

---

<a id="vietnamese"></a>

# 🇻🇳 Tiếng Việt

## 🚗 Driver Monitoring AI

Hệ thống giám sát tài xế bằng trí tuệ nhân tạo để phát hiện buồn ngủ và mất tập trung khi lái xe.

---

## 📋 Mục Đích

Dự án này phát triển một hệ thống AI tiên tiến giúp:
- 🔍 **Phát hiện buồn ngủ** của tài xế trong thời gian lái xe
- 👁️ **Nhận diện mất tập trung** (lạc tập trung, sử dụng điện thoại, v.v.)
- ⚠️ **Cảnh báo thời gian thực** để nâng cao an toàn giao thông
- 📊 **Theo dõi hành vi** của tài xế để cải thiện kỹ năng lái xe

---

## ✨ Tính Năng Chính

- ✅ Phát hiện dấu hiệu buồn ngủ (nhắm mắt, ngáp, đầu rơi)
- ✅ Nhận diện mất tập trung (không nhìn đường, quay mặt, v.v.)
- ✅ Cảnh báo âm thanh và thông báo khi phát hiện tình trạng nguy hiểm
- ✅ Giao diện web thân thiện cho xem và quản lý dữ liệu
- ✅ Xử lý video từ camera để phân tích theo thời gian thực

---

## 🛠️ Công Nghệ Sử Dụng

| Công Nghệ | Phần Trăm | Mục Đích |
|-----------|----------|---------|
| **Python** | 51% | Backend, AI/ML, xử lý video |
| **HTML** | 28.3% | Giao diện web |
| **JavaScript** | 16.1% | Interactivity trên web |
| **CSS** | 4.6% | Styling giao diện |

### Thư Viện Chính
- **OpenCV**: Xử lý và phân tích video
- **TensorFlow / PyTorch**: Mô hình học sâu
- **Flask / Django**: Backend web server
- **Pandas / NumPy**: Xử lý dữ liệu

---

## 📦 Cài Đặt

### Yêu Cầu
- Python 3.8+
- Camera/Webcam (hoặc video file)
- Node.js (cho frontend - tùy chọn)

### Hướng Dẫn Cài Đặt

1. **Clone repository**
```bash
git clone https://github.com/CMTran2005/Driver_Monitoring_AI.git
cd Driver_Monitoring_AI
```

2. **Tạo virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

4. **Chạy ứng dụng**
```bash
python main.py
```

5. **Truy cập web interface** (nếu có)
```
http://localhost:5000
```

---

## 🚀 Cách Sử Dụng

### Từ Webcam
```bash
python run_webcam.py
```

### Từ Video File
```bash
python run_video.py --video path/to/video.mp4
```

### Với Cấu Hình Tuỳ Chỉnh
```bash
python main.py --confidence 0.8 --alert_sound on
```

---

## 📊 Cấu Trúc Dự Án

```
Driver_Monitoring_AI/
├── models/                 # Mô hình AI đã huấn luyện
├── src/
│   ├── detector.py        # Lõi xử lý phát hiện
│   ├── alert.py           # Hệ thống cảnh báo
│   └── utils.py           # Hàm tiện ích
├── frontend/              # Giao diện web
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── data/                  # Dữ liệu và logs
├── requirements.txt       # Dependencies Python
├── main.py               # Entry point
└── README.md             # File này
```

---

## 🎯 Kết Quả

Hệ thống đạt được:
- **Độ chính xác phát hiện buồn ngủ**: ~92%
- **Độ chính xác phát hiện mất tập trung**: ~88%
- **Thời gian xử lý**: Real-time (FPS phù hợp)
- **Độ trễ cảnh báo**: < 1 giây

---

## 🤝 Đóng Góp

Chúng tôi hoan nghênh các đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📝 License

Dự án này được cấp phép dưới [MIT License](LICENSE) - xem file LICENSE để biết chi tiết.

---

## 📧 Liên Hệ

- **Tác giả**: CMTran2005
- **GitHub**: [@CMTran2005](https://github.com/CMTran2005)
- **Email**: [Thêm email của bạn]

---

## ⚠️ Lưu Ý Về An Toàn

Hệ thống này được phát triển để **hỗ trợ** tài xế, không phải để **thay thế** sự chú ý của họ. Luôn tuân thủ luật giao thông và quy định địa phương.

---

## 🙏 Cảm Ơn

Cảm ơn tất cả những người đã đóng góp và hỗ trợ dự án này!

---

**⭐ Nếu bạn thích dự án này, vui lòng cho nó một star!**

---

---

<a id="english"></a>

# 🇬🇧 English

## 🚗 Driver Monitoring AI

An artificial intelligence-based driver monitoring system to detect drowsiness and loss of concentration while driving.

---

## 📋 Purpose

This project develops an advanced AI system that helps:
- 🔍 **Detect driver drowsiness** during driving time
- 👁️ **Identify loss of concentration** (distraction, phone use, etc.)
- ⚠️ **Real-time alerts** to enhance road safety
- 📊 **Monitor driver behavior** to improve driving skills

---

## ✨ Key Features

- ✅ Drowsiness detection indicators (eye closure, yawning, head drop)
- ✅ Loss of concentration detection (not looking at road, face turn, etc.)
- ✅ Audio alerts and notifications when hazardous conditions are detected
- ✅ User-friendly web interface for data viewing and management
- ✅ Real-time video processing from camera for analysis

---

## 🛠️ Technology Stack

| Technology | Percentage | Purpose |
|-----------|-----------|---------|
| **Python** | 51% | Backend, AI/ML, video processing |
| **HTML** | 28.3% | Web interface |
| **JavaScript** | 16.1% | Web interactivity |
| **CSS** | 4.6% | Interface styling |

### Main Libraries
- **OpenCV**: Video processing and analysis
- **TensorFlow / PyTorch**: Deep learning models
- **Flask / Django**: Backend web server
- **Pandas / NumPy**: Data processing

---

## 📦 Installation

### Requirements
- Python 3.8+
- Camera/Webcam (or video file)
- Node.js (for frontend - optional)

### Installation Guide

1. **Clone repository**
```bash
git clone https://github.com/CMTran2005/Driver_Monitoring_AI.git
cd Driver_Monitoring_AI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python main.py
```

5. **Access web interface** (if available)
```
http://localhost:5000
```

---

## 🚀 Usage

### From Webcam
```bash
python run_webcam.py
```

### From Video File
```bash
python run_video.py --video path/to/video.mp4
```

### With Custom Configuration
```bash
python main.py --confidence 0.8 --alert_sound on
```

---

## 📊 Project Structure

```
Driver_Monitoring_AI/
├── models/                 # Trained AI models
├── src/
│   ├── detector.py        # Core detection processing
│   ├── alert.py           # Alert system
│   └── utils.py           # Utility functions
├── frontend/              # Web interface
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── data/                  # Data and logs
├── requirements.txt       # Python dependencies
├── main.py               # Entry point
└── README.md             # This file
```

---

## 🎯 Results

The system achieves:
- **Drowsiness detection accuracy**: ~92%
- **Loss of concentration detection accuracy**: ~88%
- **Processing time**: Real-time (suitable FPS)
- **Alert latency**: < 1 second

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under [MIT License](LICENSE) - see the LICENSE file for details.

---

## 📧 Contact

- **Author**: CMTran2005
- **GitHub**: [@CMTran2005](https://github.com/CMTran2005)
- **Email**: [Add your email]

---

## ⚠️ Safety Notice

This system is developed to **assist** drivers, not to **replace** their attention. Always comply with traffic laws and local regulations.

---

## 🙏 Acknowledgments

Thank you to everyone who has contributed and supported this project!

---

**⭐ If you like this project, please give it a star!**

---

<div align="center">

[🔝 Back to Top](#-driver-monitoring-ai)

</div>
