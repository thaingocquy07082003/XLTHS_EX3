from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier  # Đây chỉ là ví dụ về một mô hình
# Tạo arrays X và y từ dữ liệu MFCC và nhãn tương ứng
X = []  # Array chứa các vectors MFCC
y = []  # Array chứa các nhãn tương ứng ('a','e','i','o','u')
# Đọc và trích xuất MFCC cho mỗi file
for file_path, label in dataset:  # dataset là một list các tuple (file_path, label)
    signal, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(signal, sr=sr)
    X.append(mfccs.mean(axis=1))  # Lấy trung bình các hệ số MFCC theo thời gian
    y.append(label)
X = np.array(X)
y = np.array(y)
# Chia tập dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Tạo và huấn luyện mô hình
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)
# Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Dự đoán cho file .wav mới
new_signal, sr = librosa.load(new_file_path, sr=None)
new_mfccs = librosa.feature.mfcc(new_signal, sr=sr)
new_feature = new_mfccs.mean(axis=1)
new_feature = new_feature.reshape(1, -1)  # Reshape cho phù hợp với model input
predicted_label = model.predict(new_feature)
print(f"Predicted label for the new file: {predicted_label}")

