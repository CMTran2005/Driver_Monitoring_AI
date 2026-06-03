from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)

# ==========================================
# MEDIAPIPE SETUP (Tasks API - v0.10+)
# ==========================================
USE_MEDIAPIPE = False
face_landmarker = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request, pathlib

    MODEL_PATH = 'face_landmarker.task'
    if not pathlib.Path(MODEL_PATH).exists():
        print("📥 Đang tải face_landmarker.task (~6MB)...")
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            MODEL_PATH
        )
        print("✅ Tải xong model!")

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    USE_MEDIAPIPE = True
    print("✅ MediaPipe Tasks ready!")
except Exception as e:
    print("❌ MediaPipe lỗi:", e)
    print("⚠️  Dùng Haar Cascade fallback.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Điểm mốc EAR/MAR (478-landmark model)
LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH_TOP    = 13;  MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78;  MOUTH_RIGHT  = 308

# ==========================================
# BIẾN TOÀN CỤC
# ==========================================
current_ear   = 0.30
current_mar   = 0.0
driver_status = "tỉnh táo"
alert_history = []
eye_counter   = 0
yawn_counter  = 0
last_alert_time = None

# ==========================================
# CÀI ĐẶT
# ==========================================
SETTINGS_FILE = 'settings.json'

def load_settings():
    defaults = {
        "soundEnabled": True, "sensitivity": 2.0,
        "earThresh": 0.22, "marThresh": 0.60,
        "eyeFrames": 20, "yawnFrames": 10
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return {**defaults, **json.load(f)}
        except:
            pass
    return defaults

def save_settings_to_file(data):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(data, f)

settings = load_settings()

# ==========================================
# TÍNH EAR & MAR
# ==========================================
def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)

def calc_mar(lm, w, h):
    top    = np.array([lm[MOUTH_TOP].x * w,    lm[MOUTH_TOP].y * h])
    bottom = np.array([lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h])
    left   = np.array([lm[MOUTH_LEFT].x * w,   lm[MOUTH_LEFT].y * h])
    right  = np.array([lm[MOUTH_RIGHT].x * w,  lm[MOUTH_RIGHT].y * h])
    return np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)

# ==========================================
# LOG LỊCH SỬ
# ==========================================
def log_alert(alert_type, level, level_text):
    global last_alert_time
    now = datetime.now()
    if last_alert_time is None or (now - last_alert_time).total_seconds() >= 10:
        last_alert_time = now
        alert_history.insert(0, {
            "time": now.strftime("%d/%m/%Y %H:%M:%S"),
            "type": alert_type,
            "level": level,
            "level_text": level_text
        })
        if len(alert_history) > 100:
            alert_history.pop()

# ==========================================
# CAMERA & DETECTION
# ==========================================
camera = cv2.VideoCapture(0)

def generate_frames():
    global current_ear, current_mar, driver_status, eye_counter, yawn_counter

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        EAR_THRESH  = float(settings.get("earThresh",  0.22))
        MAR_THRESH  = float(settings.get("marThresh",  0.60))
        EYE_FRAMES  = int(settings.get("eyeFrames",  20))
        YAWN_FRAMES = int(settings.get("yawnFrames", 35))

        ear = 0.30; mar = 0.0
        temp_status = "tỉnh táo"
        face_found  = False

        if USE_MEDIAPIPE:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = face_landmarker.detect(mp_image)

            if result.face_landmarks:
                face_found = True
                lm = result.face_landmarks[0]

                left_ear  = calc_ear(lm, LEFT_EYE,  w, h)
                right_ear = calc_ear(lm, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0
                mar = calc_mar(lm, w, h)

                for idx in LEFT_EYE + RIGHT_EYE:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 180), -1)

                eye_counter  = eye_counter + 1 if ear < EAR_THRESH else max(0, eye_counter - 2)
                yawn_counter = yawn_counter + 1 if mar > MAR_THRESH else max(0, yawn_counter - 1)

                if eye_counter >= EYE_FRAMES:
                    temp_status = "buồn ngủ"
                    log_alert("Nhắm mắt quá lâu", "danger", "Nguy hiểm")
                    cv2.putText(frame, "!!! BUON NGU !!!", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                elif yawn_counter >= YAWN_FRAMES:
                    temp_status = "đang ngáp"
                    log_alert("Phát hiện ngáp", "warning", "Cảnh báo")
                    cv2.putText(frame, "DANG NGAP", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2)

                ear_color = (0, 0, 255) if ear < EAR_THRESH else (0, 255, 100)
                mar_color = (0, 0, 255) if mar > MAR_THRESH  else (0, 255, 100)
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, h-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, h-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mar_color, 2)
                cv2.putText(frame, f"Eye:{eye_counter}/{EYE_FRAMES}  Yawn:{yawn_counter}/{YAWN_FRAMES}",
                            (w-270, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        else:
            # Fallback Haar Cascade
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                face_found = True
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 200, 255), 2)
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                if len(eyes) == 0:
                    eye_counter += 1
                    brightness = cv2.mean(roi_gray[0:int(fh*0.4), :])[0]
                    ear = 0.15 if brightness < 100 else 0.25
                else:
                    eye_counter = max(0, eye_counter - 2)
                    ear = 0.32
                if eye_counter >= EYE_FRAMES:
                    temp_status = "buồn ngủ"
                    log_alert("Nhắm mắt quá lâu", "danger", "Nguy hiểm")
                    cv2.putText(frame, "!!! BUON NGU !!!", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        if not face_found:
            eye_counter  = max(0, eye_counter - 1)
            yawn_counter = max(0, yawn_counter - 1)

        current_ear   = round(ear, 3)
        current_mar   = round(mar, 3)
        driver_status = temp_status

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/history.html')
def history():
    return render_template('history.html')

@app.route('/settings.html')
def settings_page():
    return render_template('settings.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify({"ear": current_ear, "mar": current_mar, "status": driver_status})

@app.route('/api/history')
def get_history():
    return jsonify(alert_history)

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    global alert_history, last_alert_time
    alert_history = []
    last_alert_time = None
    return jsonify({"ok": True})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global settings
    data = request.get_json()
    settings.update(data)
    save_settings_to_file(settings)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)