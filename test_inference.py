import cv2
import joblib
import numpy as np

eye_model = joblib.load('trainer/eye_model.pkl')
yawn_model = joblib.load('trainer/yawn_model.pkl')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

eye_counter = 0
yawn_counter = 0
EYE_THRESH = 8
YAWN_THRESH = 35
EYE_CONFIDENCE_THRESH = 0.5
YAWN_CONFIDENCE_THRESH = 0.75

cap = cv2.VideoCapture(0)

def predict_roi(roi, model, size=(64, 64)):
    if roi.size == 0: return 0, 0.0
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, size)
    roi_flatten = roi_resized.flatten().reshape(1, -1)
    
    prediction = model.predict(roi_flatten)[0]
    confidence = np.max(model.decision_function(roi_flatten))
    return prediction, abs(confidence)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(roi_face, cv2.COLOR_BGR2GRAY)
        
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 10)
        current_eye_pred = 0
        eye_confidence = 0.0
        is_eye_detected = False
        
        if len(eyes) > 0:
            is_eye_detected = True
            (ex, ey, ew, eh) = eyes[0]
            eye_roi = roi_face[ey:ey+eh, ex:ex+ew]
            current_eye_pred, eye_confidence = predict_roi(eye_roi, eye_model)
        else:
            eye_region_top = roi_face[0:int(h*0.35), int(w*0.1):int(w*0.9)]
            brightness = cv2.mean(eye_region_top)[0]

            if brightness < 102:
                current_eye_pred = 1 
                eye_confidence = 0.8
            else:
                current_eye_pred = 0
                eye_confidence = 0.4

        if current_eye_pred == 1 and eye_confidence >= EYE_CONFIDENCE_THRESH:
            eye_counter += 1
        else:
            eye_counter = 0

        mouth_y_start = int(h * 0.68)
        mouth_y_end = int(h * 0.85)
        mouth_x_start = int(w * 0.25)
        mouth_x_end = int(w * 0.75)
        
        mouth_roi = roi_face[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        current_yawn_pred, yawn_confidence = predict_roi(mouth_roi, yawn_model)

        if current_yawn_pred == 1 and yawn_confidence >= YAWN_CONFIDENCE_THRESH:
            yawn_counter += 1
        else:
            yawn_counter = 0

        is_eye_closed = eye_counter >= EYE_THRESH
        is_yawning = yawn_counter >= YAWN_THRESH

        eye_text = f"MAT DONG! ({eye_counter}/{EYE_THRESH})" if is_eye_closed else f"MAT: MO ({eye_counter})"
        yawn_text = f"NGAP! ({yawn_counter}/{YAWN_THRESH})" if is_yawning else f"MIENG: BINH THUONG ({yawn_counter})"
        
        color_eye = (0, 0, 255) if is_eye_closed else (0, 255, 0)
        color_yawn = (0, 0, 255) if is_yawning else (0, 255, 0)

        cv2.putText(frame, eye_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_eye, 2)
        cv2.putText(frame, yawn_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_yawn, 2)
        
        cv2.rectangle(roi_face, (mouth_x_start, mouth_y_start), (mouth_x_end, mouth_y_end), (255, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('He thong Giam sat Tai xe - Debug Mode', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()