import joblib
import cv2

# 1. Load mô hình đã train
eye_model = joblib.load('trainer/eye_model.pkl')
yawn_model = joblib.load('trainer/yawn_model.pkl')

def predict_status(roi_gray, model):
    # Tiền xử lý vùng mắt/miệng giống hệt lúc train
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_flatten = roi_gray.flatten().reshape(1, -1)
    
    # Dự đoán: 1 là bất thường (đóng mắt/ngáp), 0 là bình thường
    prediction = model.predict(roi_flatten)
    return prediction[0]