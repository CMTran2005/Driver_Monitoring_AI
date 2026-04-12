# 🚀 HƯỚNG DẪN CHẠY HỆ THỐNG

## **Cấu hình hiện tại:**
- **Eye Threshold**: brightness < 102 → mắt đóng
- **Mouth ROI**: y=[0.72:0.82], x=[0.28:0.72]
- **Confidence Threshold**: Eye=0.5, Yawn=0.6
- **Noise Filter**: Eye=8 frames, Yawn=20 frames

---

## **3 Script chính:**

### **1. 🎯 Chính - Chạy hệ thống**
```bash
python test_inference.py
```
**Chức năng:**
- Detect mắt & ngáp realtime
- Hiển thị cảnh báo nếu nhắm > 8 frames liên tục
- Hiển thị cảnh báo nếu ngáp > 20 frames liên tục

**Điều khiển:**
- ESC hoặc Ctrl+C: Thoát

---

### **2. 🔍 Debug - Xem vùng nhận diện**
```bash
python diagnostic.py
```
**Chức năng:**
- **XANH LÒNG** = Vùng mắt được detect
- **CYAN** = Vùng fallback check brightness (khi không detect được)
- **VÀNG** = Vùng miệng
- Hiển thị confidence score & brightness value

**Có ích khi:**
- Vùng miệng bị sai lệch
- Muốn kiểm tra vùng extract

---

### **3. 📊 Calibrate - Tìm threshold mới**
```bash
python calibrate_eyes.py
```
**Chức năng:**
- Ghi nhận brightness khi MỞ & NHẮM mắt
- Tính toán threshold tự động
- Đề xuất logic & ngưỡng mới

**Khi dùng:**
- Nếu muốn calibrate lại (ví dụ: camera mới, ánh sáng khác)

---

## **Tùy chỉnh nếu cần:**

### **Vùng mắt không chính xác:**
```python
# File: test_inference.py, dòng ~75
if brightness < 102:  # Điều chỉnh nếu cần
    current_eye_pred = 1
```

### **Vùng miệng không chính xác:**
```python
# File: test_inference.py, dòng ~92
mouth_y_start = int(h * 0.72)  # Hạ xuống nếu quá cao
mouth_y_end = int(h * 0.82)    # Hạ/tăng để điều chỉnh cao độ
mouth_x_start = int(w * 0.28)  # Điều chỉnh chiều ngang
mouth_x_end = int(w * 0.72)
```

### **Cảnh báo quá nhạy/chậm:**
```python
# File: test_inference.py, dòng ~15-17
EYE_THRESH = 8       # Tăng nếu quá nhạy (phát hiện quá nhanh)
YAWN_THRESH = 20    # Tăng nếu quá nhạy
EYE_CONFIDENCE_THRESH = 0.5    # Tăng để chặt chẽ hơn
YAWN_CONFIDENCE_THRESH = 0.6   # Tăng để chặt chẽ hơn
```

---

## **Quy trình thử nghiệm:**

1. **Chạy chính:**
   ```bash
   python test_inference.py
   ```
   - Mở mắt bình thường → kiểm tra
   - Nhắm mắt 2-3 giây → xem có báo không
   - Ngáp 2-3 lần → xem có báo không

2. **Nếu có vấn đề, dùng diagnostic:**
   ```bash
   python diagnostic.py
   ```
   - Xem vùng mắt/miệng có chính xác không
   - Điều chỉnh coordinates nếu cần

3. **Nếu cần calibrate lại:**
   ```bash
   python calibrate_eyes.py
   ```

---

## **Files:**
- `test_inference.py` - Chính
- `diagnostic.py` - Debug vùng nhận diện
- `calibrate_eyes.py` - Calibrate threshold mắt
- `check_model_accuracy.py` - Kiểm tra độ chính xác model
- `SETUP.md` - Hướng dẫn này

---

**Bắt đầu thử nghiệm bây giờ! 🚀**
```bash
python test_inference.py
```
