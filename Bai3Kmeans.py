import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

#Tách nguyên âm - dùng lại
def segment_vowel_silence(audio, Fs, threshold = 0.03, min_duration=0.3):

    # Chia khung tín hiệu, mỗi khung độ dài 20ms
    frame_length = int(0.02 * Fs)
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

    # Trả về tín hiệu chỉ chứa nguyên âm hay tiếng nói
    vowel = audio[is_speech_full]
    return vowel 
#MFCC 1 ng âm, 1 ng, xong
def MFCC_1vowel_1speaker(audio, Fs, K=2):
    """
    Trả về K vector, biểu diễn cho đặc trưng MFCC của 1 nguyên âm 1 người (1 audio input)
    """
    frame_length = int(0.03 * Fs)
    hop_length = int(0.02 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    #Số khung
    N = frames.shape[1]
    #Chọn vùng ở giữa, M = N//3 khung
    M = N//3

    # Tính vector MFCC từng khung
    mfcc_frames = []
    for frame in frames[M:2*M]:
        mfcc_result = librosa.feature.mfcc(y=frame, sr=Fs, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_frames.append(np.squeeze(mfcc_result))

    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(np.array(mfcc_frames))
    K_vectors = kmeans.cluster_centers_

    return K_vectors

def Mi_MFCC_1vowel_1speaker(audio, Fs):
    """
    Hàm Trích xuất vector MFCC của 1 nguyên âm 1 người (1 audio input)
    Trả về M vector, mỗi vector là 1 mfcc của chỉ 1 khung
    """
    frame_length = int(0.03 * Fs)
    hop_length = int(0.02 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    #Số khung
    N = frames.shape[1]
    #Chọn vùng ở giữa, M = N//3 khung
    M = N//3

    # Tính vector MFCC từng khung
    mfcc_frames = []
    for frame in frames[M:2*M]:
        mfcc_result = librosa.feature.mfcc(y=frame, sr=Fs, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_frames.append(np.squeeze(mfcc_result))
    # # Tính trung bình cộng của M vector MFCC
    # avg_mfcc = np.mean(mfcc_frames, axis=0)
    return mfcc_frames

#2 c, d cần kết hợp K-means vào
def MFCC_1vowel_nspeaker(vowel_label, K=2):
    """ Hàm tính vector đặc trưng MFCC cho 1 nguyên âm (không phụ thuộc người nói)
        - Đầu vào là 1 ký hiệu nguyên âm ('a',.., 'u') = tên tệp
        - Bằng cách tính trung bình cộng của 21 người nói khác nhau
        - Trả về 1 vector MFCC cuối cùng ---> để bỏ vào model
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
        mi_mfcc = Mi_MFCC_1vowel_1speaker(vowel, Fs)
        all_vectors.extend(mi_mfcc)
    # Giải thích: mi_mfcc là 1 list chứa mi vector mfcc của mi khung của 1 vowel (1 audio, 1 người)
    # all_vectors là list dài chứa liên tiếp các vector của mi_mfcc
    
    print(all_vectors)

    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(np.array(all_vectors))
    K_vectors = kmeans.cluster_centers_

    print(f"Đã xong chữ {vowel_label}, len_mfcc = {K_vectors.shape}")
    print(K_vectors)

    return K_vectors

#Cần kết hợp k-means vào
def matching(vectors_x, model_vectors):
    """Hàm so khớp vector_x (input) và model (các vector tham số của 5 nguyên âm)
    * Đầu vào:
      - vectors_x: Bao gồm K vector đặc trưng của 1 nguyên âm kiểm thử x
      - model là Bộ Vector tham số mfcc của 5 nguyên âm,
        + mỗi Nguyên âm được biểu diễn bằng K vector
      Như vậy cả X thực ra là 1 ma trận numpy (K, 13)
      Và model là 5 ma trận (K,13)
    * Đầu ra: nhãn
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    distances = np.zeros((len(vowels), vectors_x.shape[0]))

    for i, model_vector in enumerate(model_vectors):
        distances[i, :] = np.linalg.norm(vectors_x - model_vector, axis=1)

    min_distance_indices = np.argmin(distances, axis=0)

    result = vowels[min_distance_indices[0]]
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

def readSignals_and_extraction_MFCC(list_path, K):
    """Đọc tín hiệu rồi trích xuất vector đặc trưng --> dùng cho kiểm thử
    """
    mfcc_vectors = []
    for file_path in list_path:
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs)
        mfcc1 = MFCC_1vowel_1speaker(vowel, Fs, K)
        mfcc_vectors.append(mfcc1)
    return mfcc_vectors

def test(x_test, y_test, model, K): 
    """ Hàm dự đoán 1 tập dữ liệu kiểm thử
        x_test: tập kiểm thử với kiểu dữ liệu là .......
        y_test: nhãn của tập kiểm thử

        Trả về:
        - Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai
        - Độ chính xác nhận dạng tổng hợp (%)
    """
    print("Nhận dạng với K =?")
    y_pred = []
    test_mfcc_vectors = readSignals_and_extraction_MFCC(x_test, K)
    for i in range(len(test_mfcc_vectors)):
        one_predict = matching(test_mfcc_vectors[i], model)
        y_pred.append(one_predict)
        check = (y_test[i] == one_predict)
        print(f"{x_test[i]} /{one_predict}/ -> {check}")
        
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

if __name__ == "__main__":
    #Đọc tên từng file, bỏ vào x_test và y_test
    # train_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",\
    #      "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
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

    model1 = build_model(K=3) #Thay K chỗ này

    #------------Vẽ đồ thị các vector------------------

    #---------------Kiểm thử-------------------------------
    y_pred1, accuracy1 = test(x_test, y_test, model1, K=3)

    print("Accuracy:",accuracy1)

    confusion = confusion_matrix(y_test, y_pred1)
    # if (accuracy1 > accuracy2 and accuracy1 > accuracy3):
    #     confusion = confusion_matrix(y_test, y_pred1)
    # elif (accuracy2 > accuracy3):
    #     confusion = confusion_matrix(y_test, y_pred2)
    # else:
    #     confusion = confusion_matrix(y_test, y_pred3)
    
    class_names = np.unique(y_test)
    df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()