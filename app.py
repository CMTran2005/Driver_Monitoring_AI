from flask import Flask, render_template, Response, jsonify
import cv2
import joblib # Thêm thư viện của AI
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. LOAD MÔ HÌNH AI & BIẾN TOÀN CỤC
# ==========================================
try:
    eye_model = joblib.load('trainer/eye_model.pkl')
    yawn_model = joblib.load('trainer/yawn_model.pkl')
    print("✅ Đã load thành công mô hình AI!")
except Exception as e:
    print("❌ Lỗi: Không tìm thấy file pkl. Hãy kiểm tra lại thư mục 'trainer'. Lỗi:", e)
    eye_model = None
    yawn_model = None

# Biến này để lưu trạng thái hiện tại gửi sang Frontend
current_ear = 0.3
driver_status = "tỉnh táo"

# ==========================================
# 2. HÀM DỰ ĐOÁN CỦA BÊN AI
# ==========================================
def predict_status(roi_gray, model):
    if model is None:
        return 0 # Nếu không có model thì mặc định là tỉnh táo
    
    # Tiền xử lý vùng mắt/miệng giống hệt lúc train
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_flatten = roi_gray.flatten().reshape(1, -1)
    
    # Dự đoán: 1 là bất thường (đóng mắt/ngáp), 0 là bình thường
    prediction = model.predict(roi_flatten)
    return prediction[0] 

# ==========================================
# 3. XỬ LÝ CAMERA VÀ LỒNG GHÉP AI
# ==========================================
camera = cv2.VideoCapture(0)
# Dùng bộ nhận diện khuôn mặt có sẵn của OpenCV để tìm khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    global current_ear, driver_status
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Quét tìm khuôn mặt trong khung hình
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # Trạng thái mặc định nếu không thấy mặt
            temp_status = "tỉnh táo"
            temp_ear = 0.3
            
            for (x, y, w, h) in faces:
                # Vẽ khung xanh quanh mặt cho ngầu
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Lấy vùng khuôn mặt (roi_gray) đưa cho AI
                roi_gray = gray[y:y+h, x:x+w]
                
                # Gọi hàm AI để dự đoán
                eye_prediction = predict_status(roi_gray, eye_model)
                
                if eye_prediction == 1: # AI phán: Đang nhắm mắt!
                    temp_status = "buồn ngủ"
                    temp_ear = 0.15 # Ép EAR xuống ngưỡng báo động đỏ cho biểu đồ tụt xuống
                    cv2.putText(frame, "CANH BAO!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    temp_ear = 0.35 # Mở mắt thì biểu đồ giữ mức cao
            
            # Cập nhật kết quả ra biến toàn cục để Frontend gọi API lấy về
            current_ear = temp_ear
            driver_status = temp_status
            
            # Đóng gói khung hình
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==========================================
# 4. ĐỊNH TUYẾN CÁC TRANG WEB
# ==========================================
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/history.html')
def history():
    return render_template('history.html')

@app.route('/settings.html')
def settings():
    return render_template('settings.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 5. API ĐỂ FRONTEND JS LẤY DỮ LIỆU BÁO ĐỘNG
# ==========================================
@app.route('/api/status')
def get_status():
    return jsonify({
        "ear": current_ear,
        "status": driver_status
    })

if __name__ == "__main__":
    app.run(debug=True)