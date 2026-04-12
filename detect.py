import cv2
import joblib
import numpy as np
import mediapipe as mp
import winsound

# --- CẤU HÌNH ---
EYE_LIMIT = 25 
YAWN_LIMIT = 40
eye_counter = 0
yawn_counter = 0

# 1. Load model
eye_model = joblib.load('trainer/eye_model.pkl')
yawn_model = joblib.load('trainer/yawn_model.pkl')

# 2. Khởi tạo MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Chỉ số điểm mốc MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144] 
MOUTH = [13, 14, 78, 308] 

def get_roi(frame, landmarks, indices, img_size=(64, 64)):
    h, w, _ = frame.shape
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*coords)
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    padding = 10
    roi = frame[max(0, min_y-padding):min(h, max_y+padding), 
                max(0, min_x-padding):min(w, max_x+padding)]
    
    if roi.size == 0: return None
    
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, img_size)
    return roi_resized.flatten().reshape(1, -1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            #  MẮT ---
            eye_data = get_roi(frame, landmarks, LEFT_EYE)
            if eye_data is not None:
                eye_pred = eye_model.predict(eye_data)[0]
                if eye_pred == 0: # 0 là đóng mắt
                    eye_counter += 1
                else:
                    eye_counter = 0

            #  NGÁP ---
            yawn_data = get_roi(frame, landmarks, MOUTH)
            if yawn_data is not None:
                yawn_pred = yawn_model.predict(yawn_data)[0]
                if yawn_pred == 1: # 1 là ngáp
                    yawn_counter += 1
                else:
                    yawn_counter = 0

            # CẢNH BÁO ---
            if eye_counter >= EYE_LIMIT:
                cv2.putText(frame, "CANH BAO: BUON NGU!", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                winsound.Beep(2500, 200) #thời gian beep

            if yawn_counter >= YAWN_LIMIT:
                cv2.putText(frame, "BAN DANG NGAP!", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                winsound.Beep(1500, 200)

    cv2.imshow('Driver Monitoring AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()