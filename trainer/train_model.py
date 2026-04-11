import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def load_data(data_path):
    images = []
    labels = []
    for file in os.listdir(data_path):

        img = cv2.imread(os.path.join(data_path, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))

        if "_" in file:
            label = 1 if "closed" in file else 0 
        else:
            label = 1 if int(file.split('.')[0]) > 500 else 0
            
        images.append(img.flatten())
        labels.append(label)
    return np.array(images), np.array(labels)

X, y = load_data('trainer/dataset/')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SVC(probability=True)
model.fit(X_train, y_train)

joblib.dump(model, 'trainer/drowsiness_model.pkl')
print("Huấn luyện xong và đã lưu model!")