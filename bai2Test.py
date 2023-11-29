import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import hamming
import scipy.signal.windows


def segment_vowel_silence(audio, Fs, threshold = 0.04, min_duration=0.3):
    print("check", Fs)
    # Chia khung tín hiệu, mỗi khung độ dài 25ms
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

def nomalizing_value(fft_vector):
    """
    Hàm chuẩn hóa vector FFT về cùng thang đo
    """
    # Tính phổ biên độ bằng cách lấy giá trị tuyệt đối
    magnitude_spectrum = np.abs(fft_vector) 
    normalized_spectrum = magnitude_spectrum / np.sum(magnitude_spectrum) 
    return normalized_spectrum
    # Hàm nomalizing_value được sử dụng để chuẩn hóa vector tần số sau khi thực hiện FFT. 
    # Điều này giúp đảm bảo rằng các giá trị của vector tần số nằm trong cùng một phạm vi [0,1]
    # giúp quá trình huấn luyện mô hình hoặc phân loại sau này trở nên ổn định hơn.

def FFT_1vowel_1speaker(audio, Fs , N_FFT):
    """
    Hàm Trích xuất vector FFT của 1 nguyên âm 1 người (1 audio input)
    """
    frame_length = int(0.025 * Fs)
    frames = librosa.util.frame(x=audio, frame_length=frame_length, hop_length= frame_length//2)
    sum_fft = np.zeros(N_FFT, dtype=complex)
    # tạo ra một mảng của các số phức có độ dài là N_FFT
    # sử dụng để tích lũy (accumulate) kết quả của phép biến đổi Fourier (FFT) từ nhiều frame khác nhau.

    for frame in frames.T:  # .T Chuyển đổi khung để lặp lại chính xác
        windowing_frame = frame * scipy.signal.windows.hamming(frame_length) # hamming dùng để giảm rò rỉ quang phổ được áp dụng cho từng khung
        sum_fft += np.abs(np.fft.fft(windowing_frame, N_FFT))
    # dùng np.fft.fft để tính FFT cho từng khung, sau đó tính tổng của các FFT khung này, mỗi vector có độ dài là N_FFT, có cả phần thực và ảo
    avg_fft = sum_fft / len(frames[0])  # Chia cho số lượng khung hình
    return nomalizing_value(avg_fft)[:N_FFT // 2]
    # lấy N_FFT // 2 vì đối xứng qua trục tần số, nên chỉ cần lấy nửa đầu tiên, là phần dương, phần âm sẽ bị trùng lặp
    # có thang đo từ 0 đến 1, có độ dài là N_FFT // 2
    # avg_fft là kết quả trả về vector tần số sau phép biển đổi sau khi biến đổi Fourier   
    # Hàm FFT_1vowel_1speaker thực hiện quá trình FFT để chuyển đổi một tín hiệu âm thanh từ miền thời gian sang miền tần số. 
    # Kết quả của hàm này là một vector biểu diễn tần số của âm thanh, được sau đó chuẩn hóa bằng cách sử dụng hàm nomalizing_value

def FFT_1vowel_nspeaker(vowelchar, N_FFT):  
    """ Hàm tính vector đặc trưng fft cho 1 nguyên âm (không phụ thuộc người nói)
        - Đầu vào là 1 ký hiệu nguyên âm ('a',.., 'u') = tên tệp
        - Bằng cách tính trung bình cộng của 21 người nói khác nhau
        - Trả về 1 vector fft cuối cùng ---> để bỏ vào model
    """
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",\
         "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/{}.wav'
    vectors = []
    
    for foldername in name_folders:
        file_path = file_path_template.format(foldername, vowelchar)
        # print(file_path) #Dòng này sau này xóa
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs, threshold = 0.04, min_duration=0.3)
        fft1 = FFT_1vowel_1speaker(vowel,Fs, N_FFT=N_FFT)
        vectors.append(fft1)

    vector_fft = np.mean(vectors, axis=0)
    print(f"Đã xong chữ {vowelchar}, len(vector_fft) = {len(vector_fft)}")
    return vector_fft 
# vector_fft là tổng trung bình của 21 người nói khác nhau của mỗi nguyên âm

# Caau2. ý 3: ở câu 2c ta tính được nguyên âm 1 file của 1 người nói, 
# giờ ta tính 5 file nguyên âm của 1 người nói và so sánh với trung bình của nguyên âm tương ứng bằng hàm matching để tính khoảng cách Euclidean
def matching(vector_x, model_vectors):
    """Hàm so khớp vector_x (input) và model (các vector tham số của 5 nguyên âm)
        Trả về kết quả là Nguyên âm có khoảng cách Euclidean nhỏ nhất
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

def build_model(N_FFT):
    print("Trích xuất các vector với N_FFT= ",N_FFT)
    vector_a = FFT_1vowel_nspeaker("a", N_FFT)
    vector_e = FFT_1vowel_nspeaker("e", N_FFT)
    vector_i = FFT_1vowel_nspeaker("i", N_FFT)
    vector_o = FFT_1vowel_nspeaker("o", N_FFT)
    vector_u = FFT_1vowel_nspeaker("u", N_FFT)
    model_vectors = [vector_a, vector_e, vector_i, vector_o, vector_u]
    return model_vectors

def readSignals_and_extractionFFT(list_path, N_FFT):
    fft_vectors = []
    for file_path in list_path:
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs, threshold = 0.065, min_duration=0.3)
        fft1 = FFT_1vowel_1speaker(vowel,Fs, N_FFT=N_FFT)
        fft_vectors.append(fft1)
    return fft_vectors
# Hàm Test Dự đoán nhãn của tập dữ liệu kiểm thử và tính độ chính xác
# Gọi hàm readSignals_and_extractionFFT để tính vector đặc trưng cho tất cả các file trong tập kiểm thử.
# Dùng hàm matching để dự đoán nhãn của từng file.
# In kết quả dự đoán và tính độ chính xác sử dụng accuracy_score.
def test(x_test, y_test, model, N_FFT): 
    """ Hàm dự đoán 1 tập dữ liệu kiểm thử
        x_test: tập kiểm thử với kiểu dữ liệu là .......
        y_test: nhãn của tập kiểm thử

        Trả về:
        - Kết quả nhận dạng (dự đoán) nhãn nguyên âm của mỗi file test (/a/, …,/u/), Đúng/Sai
        - Độ chính xác nhận dạng tổng hợp (%)
    """
    print("Nhận dạng với N_FFT =", N_FFT)
    y_pred = []
    test_fft_vectors = readSignals_and_extractionFFT(x_test, N_FFT)
    for i in range(len(test_fft_vectors)):
        one_predict = matching(test_fft_vectors[i], model)
        y_pred.append(one_predict)
        check = (y_test[i] == one_predict)
        print(f"{x_test[i]} /{one_predict}/ -> {check}")
        
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy


if __name__ == "__main__":
    
    #Đọc tên từng file, bỏ vào x_test và y_test
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

    model1 = build_model(512)
    model2 = build_model(1024)
    model3 = build_model(2048)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.real(model1))
    Y = ['a', 'e', 'i', 'o', 'u']
    label_to_color = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    Y_colors = [label_to_color[label] for label in Y]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_colors, cmap='viridis', edgecolor='k', s=50)
    plt.title('Các vector FFT (N_FFT = 512) sau khi giảm chiều')
    for i, label in enumerate(Y):
        plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    X_pca = pca.fit_transform(np.real(model2))
    Y = ['a', 'e', 'i', 'o', 'u']
    label_to_color = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    Y_colors = [label_to_color[label] for label in Y]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_colors, cmap='viridis', edgecolor='k', s=50)
    plt.title('Các vector FFT (N_FFT = 1024) sau khi giảm chiều')
    for i, label in enumerate(Y):
        plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    X_pca = pca.fit_transform(np.real(model3))
    Y = ['a', 'e', 'i', 'o', 'u']
    label_to_color = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    Y_colors = [label_to_color[label] for label in Y]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_colors, cmap='viridis', edgecolor='k', s=50)
    plt.title('Các vector FFT (N_FFT = 2048) sau khi giảm chiều')
    for i, label in enumerate(Y):
        plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), textcoords="offset points", xytext=(0,5), ha='center')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    y_pred1, accuracy1 = test(x_test, y_test, model1, 512)
    y_pred2, accuracy2 = test(x_test, y_test, model2, 1024)
    y_pred3, accuracy3 = test(x_test, y_test, model3, 2048)

    print(accuracy1, accuracy2, accuracy3, sep='\n')

    confusion = None
    if (accuracy1 > accuracy2 and accuracy1 > accuracy3):
        confusion = confusion_matrix(y_test, y_pred1)
    elif (accuracy2 > accuracy3):
        confusion = confusion_matrix(y_test, y_pred2)
    else:
        confusion = confusion_matrix(y_test, y_pred3)
    
    class_names = np.unique(y_test)
    df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
