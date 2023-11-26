import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA


def segment_vowel_silence(audio, fs, threshold=0.03, min_duration=0.3):
    # Chia khung tín hiệu, mỗi khung độ dài 20ms
    frame_length = int(0.01 * fs)
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
        duration = librosa.samples_to_time(end - start, sr=fs)
        if duration < min_duration:
            is_speech_full[start:end] = True

    # Trả về tín hiệu chỉ chứa nguyên âm hay tiếng nói
    vowel = audio[is_speech_full]
    return vowel


def MFCC_1vowel_1speaker(audio, fs):
    """
    Hàm Trích xuất vector FFT của 1 nguyên âm 1 người (1 audio input)
    """
    # Chia khung tín hiệu, mỗi khung độ dài 20ms
    frame_length = int(0.01 * fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)
    # Số khung
    N = frames.shape[1]
    # Chọn vùng ở giữa, M = N//3 khung
    M = N // 3

    # Tính biến đổi Fourier nhanh (FFT) từng khung
    mfcc_frames = []
    for frame in frames[0:3 * M]:
        mfcc_result = librosa.feature.mfcc(y=frame, sr=fs, n_mfcc=13)
        mfcc_frames.append(mfcc_result)
    # Tính trung bình cộng của M vector FFT
    avg_mfcc = np.mean(mfcc_frames, axis=0)
    return avg_mfcc


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def MFCC_1vowel_nspeaker(vowelchar):
    """ Hàm tính vector đặc trưng fft cho 1 nguyên âm (không phụ thuộc người nói)
        - Đầu vào là 1 ký hiệu nguyên âm ('a',.., 'u') = tên tệp
        - Bằng cách tính trung bình cộng của 21 người nói khác nhau
        - Trả về 1 vector fft cuối cùng ---> để bỏ vào model
    """
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/{}.wav'
    vectors = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername, vowelchar)
        print(file_path)  # Dòng này sau này xóa
        audio, fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, fs, threshold=0.03, min_duration=0.3)
        MFCC1 = MFCC_1vowel_1speaker(vowel, fs)
        vectors.append(MFCC1)
    vector_MFCC = np.mean(vectors, axis=0)
    print(f"Đã xong chữ {vowelchar}, len(vector_fft) = {len(vector_MFCC)}")
    return vector_MFCC


def build_model():
    print("Trích xuất các vector với N_FFT= ")
    vector_a = MFCC_1vowel_nspeaker("a")
    vector_e = MFCC_1vowel_nspeaker("e")
    vector_i = MFCC_1vowel_nspeaker("i")
    vector_o = MFCC_1vowel_nspeaker("o")
    vector_u = MFCC_1vowel_nspeaker("u")
    model_vectors = np.array([vector_a, vector_e, vector_i, vector_o, vector_u])

    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(model_vectors[i, :], label=f'MFCC {i + 1}')
    # Thêm title, legend và labels
    plt.title('First 5 MFCC Vectors Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('MFCC Coefficient Value')
    plt.legend()
    plt.show()
    return model_vectors  # model de huan luyen


def extract_mfcc_from_wav(file_path):
    # Đọc file wav và lấy mẫu
    signal, sr = librosa.load(file_path, sr=None)
    # Trích xuất MFCC
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # 13 là số lượng hệ số MFCC mặc định
    # để tạo thành một vector đặc trưng duy nhất cho toàn bộ đoạn âm thanh
    mfccs_mean = np.mean(mfccs, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(mfccs_mean, label=f'MFCC {0 + 1}')
    plt.title('First 5 MFCC Vectors Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('MFCC Coefficient Value')
    plt.legend()
    plt.show()
    return mfccs_mean


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


def readSignals_and_extractionFFT(list_path):
    fft_vectors = []
    for file_path in list_path:
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs)
        fft1 = MFCC_1vowel_1speaker(vowel,Fs)
        fft_vectors.append(fft1)
    return fft_vectors


def test(x_test, y_test, model):
    """ Hàm dự đoán 1 tập dữ liệu kiểm thử
        x_test: tập kiểm thử với kiểu dữ liệu là .......
        y_test: nhãn của tập kiểm thử

        Trả về:
        - Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai
        - Độ chính xác nhận dạng tổng hợp (%)
    """
    y_pred = []
    test_fft_vectors = readSignals_and_extractionFFT(x_test)
    for i in range(len(test_fft_vectors)):
        one_predict = matching(test_fft_vectors[i], model)
        y_pred.append(one_predict)
        check = (y_test[i] == one_predict)
        print(f"{x_test[i]} /{one_predict}/ -> {check}")

    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy


# file_path = r'signals/NguyenAmKiemThu-16k/09MPD/u.wav'
# # Đọc file âm thanh
# audio, Fs = librosa.load(file_path, sr=None)
# # Gọi hàm để thực hiện segment và nhận đoạn tín hiệu nguyên âm
# vowel = segment_vowel_silence(audio, Fs, threshold=0.04, min_duration=0.3)
#
# model_vector = build_model()
# extract_mfcc_from_wav(file_path)
# # Vector MFCC của tín hiệu đầu vào
# input_mfcc_vector = MFCC_1vowel_1speaker(vowel,Fs)
# # Tập hợp các vector MFCC đặc trưng cho mỗi nguyên âm từ tập huấn luyện
# training_vectors = {'a' : model_vector[0] , 'e': model_vector[1] , 'i': model_vector[2] , 'o': model_vector[3] , 'u' : model_vector[4]}  # e.g., {'a': vector_a, 'e': vector_e, ..., 'u': vector_u}
# # Khởi tạo biến để lưu khoảng cách nhỏ nhất và nguyên âm tương ứng
# min_distance = float('inf')
# recognized_vowel = None
# # Tính khoảng cách và tìm khoảng cách nhỏ nhất
# for vowel, feature_vector in training_vectors.items():
#     distance = euclidean_distance(input_mfcc_vector, feature_vector)
#     if distance < min_distance:
#         min_distance = distance
#         recognized_vowel = vowel
# # Kết quả nhận dạng
# print(f"Nguyên âm nhận dạng là: {recognized_vowel} với khoảng cách: {min_distance}")

def test_a():
    test_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/a.wav'
    vectors=[]
    for folder in test_folders:
            file_path = file_path_template.format(folder)
            print(file_path)  # Dòng này sau này xóa
            audio, fs = librosa.load(file_path, sr=None)
            vowel = segment_vowel_silence(audio, fs, threshold=0.03, min_duration=0.3)
            MFCC1 = MFCC_1vowel_1speaker(vowel, fs)
            vectors.append(MFCC1)
    plt.figure(figsize=(10, 6))
    for i in range(21):
        plt.plot(vectors[i], label=f'MFCC {i + 1}')
    # Thêm title, legend và labels
    plt.title('First 21 MFCC Vectors a Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('MFCC Coefficient Value')
    plt.legend()
    plt.show()


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
model = build_model()
y_pred, accuracy = test(x_test, y_test, model)
confusion = confusion_matrix(y_test, y_pred)
class_names = np.unique(y_test)
df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
# test_a()