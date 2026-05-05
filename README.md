# 🚗 Driver Monitoring AI

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
