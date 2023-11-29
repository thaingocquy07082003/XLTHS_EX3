from operator import le
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
import time


#Tách nguyên âm - dùng lại
def segment_vowel_silence(audio, Fs, threshold = 0.04, min_duration=0.3):

    # Chia khung tín hiệu, mỗi khung độ dài 
    frame_length = int(0.025 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)
    # Tính STE từng khung
    ste = np.sum(np.square(frames), axis=0)

    # Chuẩn hóa STE
    ste_normalized = (ste - np.min(ste)) / (np.max(ste) - np.min(ste))

    # Phân loại thành tiếng nói và khoảng lặng
    is_speech = ste_normalized > threshold

    is_speech_full = np.repeat(is_speech, frame_length)[:len(audio)]
    is_speech_full = np.pad(is_speech_full, (0, len(audio) - len(is_speech_full)), constant_values=False) 
    
    # Tìm danh sách khoảng lặng
    silence_segments = librosa.effects.split(audio, top_db=threshold)

    # Bỏ đi các khoảng lặng < 300 ms
    for start, end in silence_segments:
        duration = librosa.samples_to_time(end - start, sr=Fs)
        if duration < min_duration:
            is_speech_full[start:end] = True

    # Tìm vị trí của các khung nguyên âm
    vowel_indices = np.where(is_speech_full)[0]

    # Chia thành 3 đoạn và lấy đoạn giữa
    if len(vowel_indices) >= 3:
        start_index = vowel_indices[len(vowel_indices) // 3]
        end_index = vowel_indices[2 * len(vowel_indices) // 3]
        vowel_middle_segment = audio[start_index:end_index]
    else:
        vowel_middle_segment = audio

    return vowel_middle_segment

def nomalizing_value(mfcc_vector):
    return (mfcc_vector - np.mean(mfcc_vector)) / np.std(mfcc_vector)
    
#MFCC 1 ng âm, 1 ng, xong
def MFCC_1vowel_1speaker(audio, Fs):
    """
    Hàm trích xuất vector MFCC của 1 nguyên âm, 1 người nói (1 audio input)
    Đầu vào: đoạn nguyên âm ổn định và Fs
    Trả về: vector MFCC 13 chiều của 1 nguyên âm, 1 người nói
    """
    frame_length = int(0.03* Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length//2)

    # Tính vector MFCC từng khung
    mfcc_frames = []
    for frame in (frames.T):
        mfcc_result = librosa.feature.mfcc(y=frame, sr=Fs, n_mfcc=13, n_fft=frame_length, hop_length = frame_length//2)
        mfcc_frames.append(mfcc_result)

    mfccs = np.mean(mfcc_frames, axis=2)
    normalized_mfccs = []
    for mfcc_frame in mfccs:
        normalized_mfcc_frame = nomalizing_value(mfcc_frame)
        normalized_mfccs.append(normalized_mfcc_frame)

    # Tính giá trị trung bình của các frame MFCC chuẩn hóa
    mfccs_mean = np.mean(normalized_mfccs, axis=0)
    return mfccs_mean #(13 ,)chiều

def Mi_MFCC_1vowel_1speaker(audio, Fs):
    """
    Hàm Trích xuất vector MFCC của 1 nguyên âm 1 người (1 audio input)
    Đầu vào: đoạn nguyên âm ổn định và Fs
        Tín hiệu đầu vào được chia ra làm M khung, mỗi khung sẽ tính được 1 MFCC, nhưng không tính avg
    Trả về: M vector MFCC ứng với M khung
    Ý nghĩa: để thực hiện K-means trên tất cả các vector MFCC của tất cả các khung, của 21 người nói (xem hàm tiếp theo)
    """
    frame_length = int(0.03* Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length//2)

    # Tính vector MFCC từng khung
    mfcc_frames = []
    for frame in (frames.T):
        mfcc_result = librosa.feature.mfcc(y=frame, sr=Fs, n_mfcc=13,n_fft=frame_length, hop_length = frame_length//2)
        mfcc_frames.append(mfcc_result) 

    mfccs = np.mean(mfcc_frames, axis=2)
    normalized_mfccs = []
    for mfcc_frame in mfccs:
        normalized_mfcc_frame = nomalizing_value(mfcc_frame)
        normalized_mfccs.append(normalized_mfcc_frame)

    return normalized_mfccs

#Tính vector mfcc model ĐÃ kết hợp K-means vào
def MFCC_1vowel_nspeaker(vowel_label, K=3):
    """ Hàm tính vector đặc trưng MFCC cho 1 nguyên âm (không phụ thuộc người nói)
        * Đầu vào là 1 ký hiệu nguyên âm ('a',.., 'u') = tên tệp
        * Thuật toán:
            - chương trình đọc qua từng tệp,
            - tệp thứ i có tín hiệu chia ra làm M[i] khung
            - tính được M[i] vector MFCC của mỗi khung (hàm trên) rồi trả về.
            - Hàm này sẽ thực hiện phân cụm K-means trên tập hợp M[0] + M[1] + ... + M[20] (21 người)
             để cho ra K vector MFCC ứng với mỗi cụm và có chung 1 nhãn /a/ hoặc /e/ hoặc ... 
        * Trả về K vector MFCC cuối cùng ---> để bỏ vào model
    """
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",\
         "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/{}.wav'
    all_vectors = []
    
    for foldername in name_folders:
        file_path = file_path_template.format(foldername, vowel_label)
        print(file_path)
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs)
        mi_mfcc = Mi_MFCC_1vowel_1speaker(vowel, Fs) # Mi vector MFCC ứng với mỗi khung, Đã chuẩn hóa
        all_vectors.extend(mi_mfcc)
      
    # Giải thích: mi_mfcc là 1 list chứa mi vector mfcc của mi khung của 1 vowel (1 audio, 1 người)
    # all_vectors là list dài chứa liên tiếp các vector của mi_mfcc
    print(len(all_vectors))
    print(all_vectors)

    kmeans = KMeans(n_clusters=K) #Có thể thay hạt giống
    kmeans.fit(np.array(all_vectors))
    K_vectors = kmeans.cluster_centers_

    print(f"Đã xong chữ {vowel_label}, len_mfcc = {K_vectors.shape}")
    print(K_vectors)
    print(type(K_vectors))

    return K_vectors

def matching(vector_x, model_vectors):
    """Hàm so khớp vector_x (input) và model (các vector tham số của 5 nguyên âm)
    * Đầu vào:
      - vector_x: chính là vector MFCC đặc trưng của 1 nguyên âm kiểm thử x
      - model là Bộ Vector tham số mfcc của 5 nguyên âm,
        + mỗi Nguyên âm được biểu diễn bằng K vector
      Như vậy model là 5 ma trận (K,13)
    * Đầu ra: nhãn
    """
    # Các nhãn
    vowels = ['a', 'e', 'i', 'o', 'u']
    # Tính khoảng cách Euclidean giữa vector_x và từng vector trong model_vectors
    distances = [np.linalg.norm(vector_x - vector_clus) for vowel_vectors in model_vectors for vector_clus in vowel_vectors]
    # Xác định nguyên âm có khoảng cách nhỏ nhất
    min_distance_index = np.argmin(distances)
    # Kết quả nhận dạng
    result = vowels[min_distance_index //len(model_vectors[0])]  # Lấy phần nguyên để xác định nguyên âm
    return result

def build_model(K):
    print("Trích xuất các vector MFCC kết hợp thuật toán K-means, với K= ?")
    vector_a = MFCC_1vowel_nspeaker("a", K)
    vector_e = MFCC_1vowel_nspeaker("e", K)
    vector_i = MFCC_1vowel_nspeaker("i", K)
    vector_o = MFCC_1vowel_nspeaker("o", K)
    vector_u = MFCC_1vowel_nspeaker("u", K)
    model_vectors = [vector_a, vector_e, vector_i, vector_o, vector_u]
    return model_vectors

def readSignals_and_extraction_MFCC(list_path):
    """Đọc tín hiệu rồi trích xuất vector đặc trưng --> dùng cho hàm test (kiểm thử)
    """
    mfcc_vectors = []
    for file_path in list_path:
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs)
        mfcc1 = MFCC_1vowel_1speaker(vowel, Fs)
        mfcc_vectors.append(mfcc1)
    return mfcc_vectors

def test(x_test, y_test, model): 
    """ Hàm dự đoán 1 tập dữ liệu kiểm thử
        x_test: tập kiểm thử với kiểu dữ liệu là .......
        y_test: nhãn của tập kiểm thử

        Trả về:
        - Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai
        - Độ chính xác nhận dạng tổng hợp (%)
    """
    y_pred = []
    test_mfcc_vectors = readSignals_and_extraction_MFCC(x_test)
    for i in range(len(test_mfcc_vectors)):
        one_predict = matching(test_mfcc_vectors[i], model)
        y_pred.append(one_predict)
        check = (y_test[i] == one_predict)
        print(f"{x_test[i]} /{one_predict}/ -> {check}")
        
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

def plot_clustered_vectors(clustered_vectors, labels, K):
    # Hàm vẽ K vector của mỗi nguyên âm
    colors = ['red', 'purple', 'blue', 'yellow', 'green']
    for i, (vowels_vectors, label, color) in enumerate(zip(clustered_vectors, labels, colors)):
        for j in range(K):
            plt.plot(vowels_vectors[j], label=f'{label}' if j == 0 else "", color=color)
    plt.xlabel('Dimension')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_clustered_vectors_subplots(clustered_vectors, labels, K):
    # Hàm vẽ K vector của mỗi nguyên âm trong các subplot
    colors = ['red', 'purple', 'blue', 'green', 'yellow'] 
    fig, axes = plt.subplots(nrows=len(labels), ncols=1, figsize=(8, 12)) 
    for i, (vowels_vectors, label, color, ax) in enumerate(zip(clustered_vectors, labels, colors, axes)):
        ax.set_prop_cycle('color', [color])  # Đặt màu sắc duy nhất cho toàn bộ subplot
        for j in range(K):
            ax.plot(vowels_vectors[j], label=f'{label}' if j == 0 else "")
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    test_folders = ['01MDA', '02FVA', '03MAB', '04MHB', '05MVB', '06FTB', '07FTC', '08MLD', '09MPD', '10MSD', '11MVD', \
        '12FTD', '14FHH', '15MMH', '16FTH', '17MTH', '18MNK', '19MXK', '20MVK', '21MTL', '22MHL']
    vowel_labels = ['a', 'e', 'i', 'o', 'u']
    file_path_template = 'signals/NguyenAmKiemThu-16k/{}/{}.wav'
    x_test = [] #Lưu đường dẫn từng file test
    y_test = [] #Lưu nhãn
    for folder in test_folders:
        for label in vowel_labels:
            file_path = file_path_template.format(folder, label)
            x_test.append(file_path)
            y_test.append(label)

    K = 5 
    model1 = build_model(K = K) 

    #------------Vẽ đồ thị các vector------------------
    plot_clustered_vectors_subplots(model1, labels=vowel_labels, K=K)
    plot_clustered_vectors(model1, labels=vowel_labels, K=K)
    #---------------Kiểm thử-------------------------------
    print("Nhận dạng với K =", K)
    starttime = time.time()
    y_pred1, accuracy1 = test(x_test, y_test, model1)
    endtime = time.time()
    print("Accuracy:",accuracy1)
    print(f"Thời gian chạy với K = {K}: {endtime - starttime} s")
    confusion = confusion_matrix(y_test, y_pred1)

    class_names = np.unique(y_test)
    df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()