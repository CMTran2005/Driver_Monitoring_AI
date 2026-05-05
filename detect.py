import cv2
import joblib
import numpy as np
import mediapipe as mp
import winsound

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Tắt log rác của TensorFlow
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Tắt mọi UserWarning

# --- CẤU HÌNH ---
EYE_LIMIT = 20        # Giảm một chút vì có check confidence sẽ khắt khe hơn
YAWN_LIMIT = 35
EYE_CONF_THRESH = 0.6  # Ngưỡng tin cậy cho mắt
YAWN_CONF_THRESH = 1.15 # Ngưỡng tin cậy cho ngáp 

eye_counter = 0
yawn_counter = 0

# 1. Load model
eye_model = joblib.load('trainer/eye_model.pkl')
yawn_model = joblib.load('trainer/yawn_model.pkl')

# 2. Khởi tạo MediaPipe (Dùng cách import an toàn)
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except:
    import mediapipe.solutions.face_mesh as mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Chỉ số điểm mốc
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
MOUTH = [13, 14, 78, 308] 

def get_roi_and_predict(frame, landmarks, indices, model, img_size=(64, 64)):
    h, w, _ = frame.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*coords)
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    padding = 10
    roi = frame[max(0, min_y-padding):min(h, max_y+padding), 
                max(0, min_x-padding):min(w, max_x+padding)]
    
    if roi.size == 0: return None, 0
    
    # Tiền xử lý
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, img_size)
    data = roi_resized.flatten().reshape(1, -1)
    
    # Dự đoán và tính Confidence
    pred = model.predict(data)[0]
    # abs() vì decision_function trả về khoảng cách đại số (âm/dương tùy class)
    conf = abs(np.max(model.decision_function(data)))
    
    return pred, conf

cap = cv2.VideoCapture(0)

eye_conf_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # --- XỬ LÝ MẮT ---
            eye_pred, eye_conf = get_roi_and_predict(frame, landmarks, LEFT_EYE, eye_model)
            eye_conf_history.append(eye_conf)
            if len(eye_conf_history) > 5: # Chỉ giữ lại 5 khung hình gần nhất
                eye_conf_history.pop(0) 
            if eye_pred == 0 and eye_conf >= EYE_CONF_THRESH:
                eye_counter += 1
            else:
                eye_counter = 0

            # --- XỬ LÝ NGÁP ---
            yawn_pred, yawn_conf = get_roi_and_predict(frame, landmarks, MOUTH, yawn_model)
            if yawn_pred == 1 and yawn_conf >= YAWN_CONF_THRESH:
                yawn_counter += 1
            else:
                yawn_counter = 0

            # --- HIỂN THỊ VÀ CẢNH BÁO ---
            color_eye = (0, 0, 255) if eye_counter >= EYE_LIMIT else (0, 255, 0)
            color_yawn = (0, 0, 255) if yawn_counter >= YAWN_LIMIT else (0, 255, 0)

            cv2.putText(frame, f"Eye: {'CLOSED' if eye_pred==0 else 'OPEN'} | Conf: {eye_conf:.2f} | Count: {eye_counter}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_eye, 2)
            cv2.putText(frame, f"Mouth: {'YAWN' if yawn_pred==1 else 'NORMAL'} (Conf: {yawn_conf:.2f})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_yawn, 2)

            if eye_counter >= EYE_LIMIT:
                cv2.putText(frame, "!!! CANH BAO BUON NGU !!!", (150, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                winsound.Beep(2500, 100)

            if yawn_counter >= YAWN_LIMIT:
                cv2.putText(frame, "!!! CANH BAO NGAP !!!", (150, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                winsound.Beep(1500, 100)

    cv2.imshow('Driver Monitoring AI Pro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()