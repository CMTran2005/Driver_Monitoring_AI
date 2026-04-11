import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

IMG_SIZE = (64, 64)

def load_data_from_folders(folder_positive, folder_negative):
    images = []
    labels = []

    for folder, label in [(folder_positive, 1), (folder_negative, 0)]:
        path = os.path.join('trainer/dataset', folder)
        for file in os.listdir(path):
            try:
                img_path = os.path.join(path, file)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img.flatten())
                labels.append(label)
            except Exception as e:
                print(f"Lỗi file {file}: {e}")
                
    return np.array(images), np.array(labels)

def train_and_save(X, y, model_name):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Đang huấn luyện {model_name}...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    joblib.dump(model, f'trainer/{model_name}.pkl')
    print(f"Đã lưu: {model_name}.pkl\n")

X_eye, y_eye = load_data_from_folders('closed', 'open')
train_and_save(X_eye, y_eye, 'eye_model')

X_yawn, y_yawn = load_data_from_folders('yawn', 'no_yawn')
train_and_save(X_yawn, y_yawn, 'yawn_model')