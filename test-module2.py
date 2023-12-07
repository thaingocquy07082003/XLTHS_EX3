import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# cac K-mean dc code o day
def extract_mfcc_from_wav(file_path):
    # Đọc file wav và lấy mẫu
    threshold = 0.04
    signal, Fs = librosa.load(file_path, sr=None)
    frame_length = int(0.015 * Fs)
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_length)
    # Tính STE từng khung
    ste = np.sum(np.square(frames), axis=0)
    ste_normalized = (ste - np.min(ste)) / (np.max(ste) - np.min(ste))
    # Phân loại thành tiếng nói và khoảng lặng
    is_speech = ste_normalized > threshold
    is_speech_full = np.repeat(is_speech, frame_length)[:len(signal)]
    is_speech_full = np.pad(is_speech_full, (0, len(signal) - len(is_speech_full)), constant_values=False)
    # Tìm danh sách khoảng lặng
    silence_segments = librosa.effects.split(signal, top_db=threshold)
    # Bỏ đi các khoảng lặng < 300 ms
    for start, end in silence_segments:
        duration = librosa.samples_to_time(end - start, sr=Fs)
        if duration < 0.2:
            is_speech_full[start:end] = True
    # Trả về tín hiệu chỉ chứa nguyên âm hay tiếng nói
    vowel = signal[is_speech_full]
    frame_length = int(0.03 * Fs)   # 0.025
    hop_length = int(0.01 * Fs)     # 0.011
    frames = librosa.util.frame(vowel, frame_length=frame_length, hop_length=hop_length)
    #Số khung
    N = frames.shape[1]
    start = N//3
    end =  2*start  #3*start
    for i in range(start,end):  # Lặp từ start đến end``
        frame = frames[:, i]  # Lấy frame thứ i
        mfccs = librosa.feature.mfcc(y=frame, sr=Fs, n_mfcc=13, n_fft=2048, hop_length=512)
        # Trích xuất MFCC
        feature_matrix = np.vstack(mfccs)
    # Chuẩn hóa các vector đặc trưng
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    # mfccs = librosa.feature.mfcc(y=vowel, sr=Fs, n_mfcc=13, n_fft=2048, hop_length=512)
    # để tạo thành một vector đặc trưng duy nhất cho toàn bộ đoạn âm thanh
    mfccs_mean = np.mean(normalized_features, axis=1)
    return mfccs_mean

def build_model_a(K):
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/a.wav'
    vectors_a = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_a.append(MFCC)
    # avg_vector_a = np.mean(vectors_a,axis=0)
    mfcc_features = np.array(vectors_a)
    num_clusters = K  # K =3
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(mfcc_features)
    labels = kmeans.labels_
    # Tâm cụm (centroids) cho mỗi cụm.
    centroids = kmeans.cluster_centers_
    # Tính trung bình của các vector trong mỗi cụm.

    mean_vectors = []
    for i in range(num_clusters):
        # Lấy chỉ số của các vector trong cụm hiện tại.
        cluster_indices = np.where(labels == i)[0]
        # Lấy tất cả vector trong cụm.
        cluster_vectors = mfcc_features[cluster_indices]
        # Tính trung bình của các vector (chỉ lấy K vector đầu tiên nếu có nhiều hơn K).
        mean_vector = np.mean(cluster_vectors[:K], axis=0) if len(cluster_vectors) >= K else np.mean(cluster_vectors,
                                                                                                     axis=0)
        mean_vectors.append(mean_vector)

    for i, mean_vec in enumerate(mean_vectors):
        print(f"K vector trung bình cho cụm {i + 1}: {mean_vec}")
    # centroids = kmeans.cluster_centers_
    # Tạo đồ thị và trực quan hóa kết quả
    plt.figure(figsize=(10, 6))
    # Vẽ các điểm dữ liệu
    for i in range(num_clusters):
        # Lấy các điểm thuộc cùng một cụm
        cluster_points = mfcc_features[labels == i]
        # Trực quan hóa các điểm đó
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'group {i}')
    # Vẽ các tâm cụm
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='*')
    # Thêm một số tính năng cho đồ thị
    plt.title('Points in each group a')
    plt.xlabel('MFCC Feature Dim 1')
    plt.ylabel('MFCC Feature Dim 2')
    plt.legend()
    # plt.grid(True)
    # Hiển thị đồ thị
    plt.show()
    return mean_vectors
    # tra va danh sach cac means cua k cum nguyen am a

def build_model_e(K):
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/e.wav'
    vectors_e = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_e.append(MFCC)
    # avg_vector_e = np.mean(vectors_e,axis=0)
    mfcc_features = np.array(vectors_e)
    num_clusters = K  # K =3
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(mfcc_features)
    labels = kmeans.labels_
    # Tâm cụm (centroids) cho mỗi cụm.
    centroids = kmeans.cluster_centers_
    # Tính trung bình của các vector trong mỗi cụm.
    mean_vectors = []
    for i in range(num_clusters):
        # Lấy chỉ số của các vector trong cụm hiện tại.
        cluster_indices = np.where(labels == i)[0]
        # Lấy tất cả vector trong cụm.
        cluster_vectors = mfcc_features[cluster_indices]
        # Tính trung bình của các vector (chỉ lấy K vector đầu tiên nếu có nhiều hơn K).
        mean_vector = np.mean(cluster_vectors[:K], axis=0) if len(cluster_vectors) >= K else np.mean(cluster_vectors,axis=0)
        mean_vectors.append(mean_vector)

    for i, mean_vec in enumerate(mean_vectors):
        print(f"K vector trung bình cho cụm {i + 1}: {mean_vec}")
    # centroids = kmeans.cluster_centers_
    # Tạo đồ thị và trực quan hóa kết quả
    plt.figure(figsize=(10, 6))
    # Vẽ các điểm dữ liệu
    for i in range(num_clusters):
        # Lấy các điểm thuộc cùng một cụm
        cluster_points = mfcc_features[labels == i]
        # Trực quan hóa các điểm đó
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'group {i}')
    # Vẽ các tâm cụm
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='*')
    # Thêm một số tính năng cho đồ thị
    plt.title('Points in each group e')
    plt.xlabel('MFCC Feature Dim 1')
    plt.ylabel('MFCC Feature Dim 2')
    plt.legend()
    # plt.grid(True)
    # Hiển thị đồ thị
    plt.show()
    return mean_vectors
    # tra va danh sach cac means cua k cum nguyen am a

def build_model_i(K):
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/i.wav'
    vectors_i = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_i.append(MFCC)
    # avg_vector_i = np.mean(vectors_i,axis=0)
    mfcc_features = np.array(vectors_i)
    num_clusters = K  # K =3
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(mfcc_features)
    labels = kmeans.labels_
    # Tâm cụm (centroids) cho mỗi cụm.
    centroids = kmeans.cluster_centers_
    # Tính trung bình của các vector trong mỗi cụm.
    mean_vectors = []
    for i in range(num_clusters):
        # Lấy chỉ số của các vector trong cụm hiện tại.
        cluster_indices = np.where(labels == i)[0]
        # Lấy tất cả vector trong cụm.
        cluster_vectors = mfcc_features[cluster_indices]
        # Tính trung bình của các vector (chỉ lấy K vector đầu tiên nếu có nhiều hơn K).
        mean_vector = np.mean(cluster_vectors[:K], axis=0) if len(cluster_vectors) >= K else np.mean(cluster_vectors,
                                                                                                     axis=0)
        mean_vectors.append(mean_vector)

    for i, mean_vec in enumerate(mean_vectors):
        print(f"K vector trung bình cho cụm {i + 1}: {mean_vec}")
    # centroids = kmeans.cluster_centers_
    # Tạo đồ thị và trực quan hóa kết quả
    plt.figure(figsize=(10, 6))
    # Vẽ các điểm dữ liệu
    for i in range(num_clusters):
        # Lấy các điểm thuộc cùng một cụm
        cluster_points = mfcc_features[labels == i]
        # Trực quan hóa các điểm đó
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'group {i}')
    # Vẽ các tâm cụm
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='*')
    # Thêm một số tính năng cho đồ thị
    plt.title('Points in each group i')
    plt.xlabel('MFCC Feature Dim 1')
    plt.ylabel('MFCC Feature Dim 2')
    plt.legend()
    # plt.grid(True)
    # Hiển thị đồ thị
    plt.show()
    return mean_vectors
    # tra va danh sach cac means cua k cum nguyen am a

def build_model_o(K):
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/o.wav'
    vectors_o = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_o.append(MFCC)
    # avg_vector_o = np.mean(vectors_o,axis=0)
    mfcc_features = np.array(vectors_o)
    num_clusters = K  # K =3
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(mfcc_features)
    labels = kmeans.labels_
    # Tâm cụm (centroids) cho mỗi cụm.
    centroids = kmeans.cluster_centers_
    # Tính trung bình của các vector trong mỗi cụm.
    mean_vectors = []
    for i in range(num_clusters):
        # Lấy chỉ số của các vector trong cụm hiện tại.
        cluster_indices = np.where(labels == i)[0]
        # Lấy tất cả vector trong cụm.
        cluster_vectors = mfcc_features[cluster_indices]
        # Tính trung bình của các vector (chỉ lấy K vector đầu tiên nếu có nhiều hơn K).
        mean_vector = np.mean(cluster_vectors[:K], axis=0) if len(cluster_vectors) >= K else np.mean(cluster_vectors,
                                                                                                     axis=0)
        mean_vectors.append(mean_vector)

    for i, mean_vec in enumerate(mean_vectors):
        print(f"K vector trung bình cho cụm {i + 1}: {mean_vec}")
    # centroids = kmeans.cluster_centers_
    # Tạo đồ thị và trực quan hóa kết quả
    plt.figure(figsize=(10, 6))
    # Vẽ các điểm dữ liệu
    for i in range(num_clusters):
        # Lấy các điểm thuộc cùng một cụm
        cluster_points = mfcc_features[labels == i]
        # Trực quan hóa các điểm đó
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'group {i}')
    # Vẽ các tâm cụm
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='*')
    # Thêm một số tính năng cho đồ thị
    plt.title('Points in each group o')
    plt.xlabel('MFCC Feature Dim 1')
    plt.ylabel('MFCC Feature Dim 2')
    plt.legend()
    # plt.grid(True)
    # Hiển thị đồ thị
    plt.show()
    return mean_vectors
    # tra va danh sach cac means cua k cum nguyen am a

def build_model_u(K):
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/u.wav'
    vectors_u = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_u.append(MFCC)
    # avg_vector_u = np.mean(vectors_u,axis=0)
    mfcc_features = np.array(vectors_u)
    num_clusters = K  # K =3
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(mfcc_features)
    labels = kmeans.labels_
    # Tâm cụm (centroids) cho mỗi cụm.
    centroids = kmeans.cluster_centers_
    # Tính trung bình của các vector trong mỗi cụm.
    mean_vectors = []
    for i in range(num_clusters):
        # Lấy chỉ số của các vector trong cụm hiện tại.
        cluster_indices = np.where(labels == i)[0]
        # Lấy tất cả vector trong cụm.
        cluster_vectors = mfcc_features[cluster_indices]
        # Tính trung bình của các vector (chỉ lấy K vector đầu tiên nếu có nhiều hơn K).
        mean_vector = np.mean(cluster_vectors[:K], axis=0) if len(cluster_vectors) >= K else np.mean(cluster_vectors,
                                                                                                     axis=0)
        mean_vectors.append(mean_vector)

    for i, mean_vec in enumerate(mean_vectors):
        print(f"K vector trung bình cho cụm {i + 1}: {mean_vec}")
    # centroids = kmeans.cluster_centers_
    # Tạo đồ thị và trực quan hóa kết quả
    plt.figure(figsize=(10, 6))
    # Vẽ các điểm dữ liệu
    for i in range(num_clusters):
        # Lấy các điểm thuộc cùng một cụm
        cluster_points = mfcc_features[labels == i]
        # Trực quan hóa các điểm đó
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'group {i}')
    # Vẽ các tâm cụm
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='*')
    # Thêm một số tính năng cho đồ thị
    plt.title('Points in each group u')
    plt.xlabel('MFCC Feature Dim 1')
    plt.ylabel('MFCC Feature Dim 2')
    plt.legend()
    # plt.grid(True)
    # Hiển thị đồ thị
    plt.show()
    return mean_vectors  # tra va danh sach cac means cua k cum nguyen am a

def Model_Of_speak():
    K= 3  # k = ?
    model_a = build_model_a(K)
    model_e = build_model_e(K)
    model_i = build_model_i(K)
    model_o = build_model_o(K)
    model_u = build_model_u(K)
    model = np.array([model_a, model_e, model_i, model_o, model_u])
    print(model)
    # plt.figure(figsize=(10, 6))
    # for i in range(5):
    #     plt.plot(model[i, :], label=f'MFCC {i + 1}')
    # # Thêm title, legend và labels
    # plt.title(' 5 MFCC Vectors Over Time')
    # plt.xlabel('Frame Index')
    # plt.ylabel('MFCC Coefficient Value')
    # plt.legend()
    # plt.show()
    return model


def readSignals_and_extractionMFCC(list_path):
    mfcc_vectors = []
    for file_path in list_path:
        MFCC = extract_mfcc_from_wav(file_path)
        mfcc_vectors.append(MFCC)
    return mfcc_vectors

def test(x_test, y_test, model):
    """ Hàm dự đoán 1 tập dữ liệu kiểm thử
        x_test: file kiểm thử với kiểu dữ liệu là .......
        y_test: nhãn của tập huấn luyện dùng để chấm

        Trả về:
        - Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai
        - Độ chính xác nhận dạng tổng hợp (%)
    """
    index = ["a","e","i","o","u"]
    column1= []  # ten file
    column2 = []   # du doan
    column3 = []   # dung hay sai
    i=0
    y_pred = []
    test_fft_vectors = readSignals_and_extractionMFCC(x_test)
    for i in range(len(test_fft_vectors)):
        one_predict = matching(test_fft_vectors[i], model)
        y_pred.append(one_predict)
        check = (y_test[i] == one_predict)
        print(f"{x_test[i]} /{one_predict}/ -> {check}")
        column1.append(x_test[i])
        column2.append(one_predict)
        column3.append(check)
        # dis_a = distanceA(test_fft_vectors[i],model[0])
        # dis_e = distanceE(test_fft_vectors[i], model[1])
        # dis_i = distanceI(test_fft_vectors[i], model[2])
        # dis_o = distanceO(test_fft_vectors[i], model[3])
        # dis_u = distanceU(test_fft_vectors[i], model[4])
        # one_predict_dis = min(dis_a,dis_e,dis_i,dis_o,dis_u)
        # if dis_a == one_predict_dis:
        #     one_predict = "a"
        # elif dis_e == one_predict_dis:
        #     one_predict = "e"
        # elif dis_i == one_predict_dis:
        #     one_predict = "i"
        # elif dis_o == one_predict_dis:
        #     one_predict = "o"
        # elif dis_u == one_predict_dis:
        #     one_predict = "u"
        # check = (y_test[i] == one_predict)
        # y_pred.append(one_predict)
        # print(f"{x_test[i]} /{one_predict}/ -> {check}")
        # column1.append(x_test[i])
        # column2.append(one_predict)
        # column3.append(check)
    data = {'Index': index,
            'Column1': column1,
            'Column2': column2,
            'Column3': column3}
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy ,data

def matching(vector_x, model_vectors):
    """Hàm so khớp vector_x (input) và model (các vector tham số của 5 nguyên âm)
        Trả về kết quả là Nguyên âm có khoảng cách Euclid nhỏ nhất
    """
    # Danh sách các vector tham số fft của 5 nguyên âm
    # model_vectors[0] = vector_a,
    # model_vectors[1] = vector_e,
    # model_vectors[2] = vector_i,
    # model_vectors[3] = vector_o,
    # model_vectors[4] = vector_u]
    # Các nhãn tương ứng
    vowels = ['a', 'e', 'i', 'o', 'u']
    # Tính khoảng cách Euclidean giữa vector_x và từng vector trong model
    distances = [np.linalg.norm(vector_x - model_vector) for model_vector in model_vectors]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    # Kết quả nhận dạng
    result = vowels[min_distance_index]
    return result

def distanceA(mfcc,model_mfccs):
    distances = [np.linalg.norm(mfcc - model_vector) for model_vector in model_mfccs]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    return  min_distance_index

def distanceE(mfcc,model_mfccs):
    distances = [np.linalg.norm(mfcc - model_vector) for model_vector in model_mfccs]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    return  min_distance_index

def distanceI(mfcc,model_mfccs):
    distances = [np.linalg.norm(mfcc - model_vector) for model_vector in model_mfccs]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    return  min_distance_index

def distanceO(mfcc,model_mfccs):
    distances = [np.linalg.norm(mfcc - model_vector) for model_vector in model_mfccs]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    return  min_distance_index

def distanceU(mfcc,model_mfccs):
    distances = [np.linalg.norm(mfcc - model_vector) for model_vector in model_mfccs]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    return  min_distance_index

test_folders = ['01MDA', '02FVA', '03MAB', '04MHB', '05MVB', '06FTB', '07FTC', '08MLD', '09MPD', '10MSD', '11MVD','12FTD', '14FHH', '15MMH', '16FTH', '17MTH', '18MNK', '19MXK', '20MVK', '21MTL', '22MHL']
vowel_labels = ['a', 'e', 'i', 'o', 'u']
file_path_template = 'signals/NguyenAmKiemThu-16k/{}/{}.wav'
x_test = []  # Lưu đường dẫn từng file test
y_test = []  # Lưu nhãn
for folder in test_folders:
    for label in vowel_labels:
        file_path = file_path_template.format(folder,label)
        x_test.append(file_path)
        y_test.append(label)
for x_name in x_test:
    print(x_name)

model = Model_Of_speak()
y_pred, accuracy ,data = test(x_test, y_test, model)
confusion = confusion_matrix(y_test, y_pred)
class_names = np.unique(y_test)
df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



