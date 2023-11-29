import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import time


def segment_vowel_silence(audio, Fs, threshold = 0.04, min_duration=0.3):
    print("check", Fs)
    # Chia khung tín hiệu, mỗi khung độ dài 25ms
    frame_length = int(0.03 * Fs)
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
#  trich xuat vector dac trung cho tung file tin hieu
def nomalizing_value(mfcc_vector):
    """
    Hàm chuẩn hóa vector MFCC về cùng thang đo
    """
    return (mfcc_vector - np.mean(mfcc_vector)) / np.std(mfcc_vector)
def extract_mfcc_from_wav(file_path):
    # Đọc file wav và lấy mẫu
    signal, Fs = librosa.load(file_path, sr=None)
    vowel = segment_vowel_silence(signal, Fs, threshold=0.04, min_duration=0.3)
    frame_length = int(0.03 * Fs)
    frames = librosa.util.frame(vowel, frame_length=frame_length, hop_length=frame_length // 2)

    mfcc_frames = []
    for frame in frames.T:
        mfcc_result = librosa.feature.mfcc(y=frame, sr=Fs, n_mfcc=13, n_fft=frame_length, hop_length=frame_length//2)
        mfcc_frames.append(mfcc_result)

    # Trích xuất MFCC
    mfccs = np.mean(mfcc_frames, axis=2)  # Take mean along the time axis
    # được sử dụng để lấy giá trị trung bình dọc theo trục thời gian (axis=2), 
    # tính trung bình một cách hiệu quả các hệ số MFCC theo thời gian cho mỗi khung.
    # Nói cách khác, nó tính giá trị trung bình của các hệ số MFCC theo thời gian cho từng khung, 
    # tạo ra một mảng 2D trong đó mỗi hàng tương ứng với một hệ số MFCC và mỗi cột tương ứng với một khung. 
    # Điều này cho phép bạn có được bộ hệ số MFCC đại diện cho từng khung hình trong âm thanh đầu vào.

    # Chuẩn hóa MFCCs
    normalized_mfccs = []
    for mfcc_frame in mfccs:
        normalized_mfcc_frame = nomalizing_value(mfcc_frame)
        normalized_mfccs.append(normalized_mfcc_frame)

    # Tính giá trị trung bình của các frame MFCC chuẩn hóa
    mfccs_mean = np.mean(normalized_mfccs, axis=0)
    return mfccs_mean

def build_model_a():
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/a.wav'
    vectors_a = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_a.append(MFCC)
    avg_vector_a = np.mean(vectors_a,axis=0)
    return avg_vector_a

def build_model_e():
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/e.wav'
    vectors_e = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_e.append(MFCC)
    avg_vector_e = np.mean(vectors_e,axis=0)
    return avg_vector_e

def build_model_i():
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/i.wav'
    vectors_i = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_i.append(MFCC)
    avg_vector_i = np.mean(vectors_i,axis=0)
    return avg_vector_i

def build_model_o():
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/o.wav'
    vectors_o = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_o.append(MFCC)
    avg_vector_o = np.mean(vectors_o,axis=0)
    return avg_vector_o

def build_model_u():
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/u.wav'
    vectors_u = []
    for foldername in name_folders:
        file_path = file_path_template.format(foldername)
        MFCC = extract_mfcc_from_wav(file_path)
        vectors_u.append(MFCC)
    avg_vector_u = np.mean(vectors_u,axis=0)
    return avg_vector_u

def Model_Of_speak():
    model_a = build_model_a()
    model_e = build_model_e()
    model_i = build_model_i()
    model_o = build_model_o()
    model_u = build_model_u()
    model = np.array([model_a, model_e, model_i, model_o, model_u])
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(model[i, :], label=f'MFCC {i + 1}')
    # Thêm title, legend và labels
    plt.title(' 5 MFCC Vectors Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('MFCC Coefficient Value')
    plt.legend()
    plt.show()
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
        y_test: nhãn của tập kiểm thử

        Trả về:
        - Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai
        - Độ chính xác nhận dạng tổng hợp (%)
    """
    y_pred = []
    test_fft_vectors = readSignals_and_extractionMFCC(x_test)
    for i in range(len(test_fft_vectors)):
        one_predict = matching(test_fft_vectors[i], model)
        y_pred.append(one_predict)
        check = (y_test[i] == one_predict)
        print(f"{x_test[i]} /{one_predict}/ -> {check}")

    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy

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
starttime = time.time()
y_pred, accuracy = test(x_test, y_test, model)
endtime = time.time()
print(f"Thời gian chạy: {endtime - starttime} s")
print("Accuracy:",accuracy)
confusion = confusion_matrix(y_test, y_pred)
class_names = np.unique(y_test)
df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
plt.figure(figsize=(8, 6))
sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
