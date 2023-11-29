import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
from scipy.signal import windows
import time

def segment_vowel_silence(audio, Fs, threshold = 0.04, min_duration=0.3):

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
    magnitude_spectrum = np.abs(fft_vector)
    return magnitude_spectrum / np.sum(magnitude_spectrum)

def FFT_1vowel_1speaker(audio, Fs, N_FFT):
    """
    Hàm Trích xuất vector FFT của 1 nguyên âm 1 người (1 audio input)
    """

    # Chia khung tín hiệu, mỗi khung độ dài 25ms
    frame_length = int(0.01 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length // 2)
    sum_fft = np.zeros(N_FFT, dtype=complex)
    for frame in (frames.T):  
        windowing_frame = frame * windows.hamming(frame_length)
        sum_fft += np.abs(np.fft.fft(windowing_frame, N_FFT))

    avg_fft = sum_fft / len(frames[0]) 

    return nomalizing_value(avg_fft)[:N_FFT // 2] 

def FFT_1vowel_nspeaker(vowel_label, N_FFT):
    """ Hàm tính vector đặc trưng fft cho 1 nguyên âm (không phụ thuộc người nói)
        - Đầu vào là 1 ký hiệu nguyên âm = nhãn ('a',.., 'u')  = tên tệp
        - Bằng cách tính trung bình cộng của 21 người nói khác nhau
        - Trả về 1 vector fft cuối cùng ---> để bỏ vào model
    """
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",\
         "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmHuanLuyen-16k/{}/{}.wav'
    vectors = []
    
    for foldername in name_folders:
        file_path = file_path_template.format(foldername, vowel_label)
        audio, Fs = librosa.load(file_path, sr=None)
        vowel = segment_vowel_silence(audio, Fs)
        fft1 = FFT_1vowel_1speaker(vowel, Fs, N_FFT=N_FFT)
        vectors.append(fft1)

    vector_fft = np.mean(vectors, axis=0)
    print(f"Đã xong chữ {vowel_label}, len(vector_fft) = {len(vector_fft)}")
    return vector_fft

def matching(vector_x, model_vectors):
    """Hàm so khớp vector_x (input) và model (các vector tham số của 5 nguyên âm)
    * Đầu vào:
      - vector_x: vector đặc trưng của 1 file tín hiệu kiểm thử
      - model là Bộ Vector tham số fft của 5 nguyên âm
        # model_vectors[0] = vector_a,
        # model_vectors[1] = vector_e,
        # model_vectors[2] = vector_i,
        # model_vectors[3] = vector_o,
        # model_vectors[4] = vector_u]
    * Đầu ra:
        Trả về kết quả là Nguyên âm có khoảng cách Euclid nhỏ nhất
    """
    # Các nhãn
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
        vowel = segment_vowel_silence(audio, Fs)
        fft1 = FFT_1vowel_1speaker(vowel, Fs, N_FFT=N_FFT)
        fft_vectors.append(fft1)
    return fft_vectors

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
        print( f"{x_test[i]} /{one_predict}/ -> {check}")
        
    accuracy = accuracy_score(y_test, y_pred)
    print("check %",{N_FFT},accuracy)
    return y_pred, accuracy

def plot_vector(vector, label):
    #Hàm vẽ 1 vector
    plt.plot(np.real(vector), label=label)
    plt.xlabel('Dimension')
    plt.legend()

def plot_all_vectors(vectors, labels):
    #Hàm vẽ 5 vector/1 đồ thị
    for i, vector in enumerate(vectors):
        plt.plot(np.real(vector), label=labels[i])
    plt.xlabel('Dimension')
    plt.legend()

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

    model1 = build_model(512)
    model2 = build_model(1024)
    model3 = build_model(2048)

    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    # Vẽ mỗi vector trên 1 subplot
    for i, vector in enumerate(model1):
        plt.sca(axs[i])
        plot_vector(vector, label=vowel_labels[i])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    # Xuất 5 vector trên 1 đồ thị
    plot_all_vectors(model1, labels=vowel_labels)
    plt.show()

    start1 = time.time()
    y_pred1, accuracy1 = test(x_test, y_test, model1, 512)
    start2 = time.time()
    y_pred2, accuracy2 = test(x_test, y_test, model2, 1024)
    start3 = time.time()
    y_pred3, accuracy3 = test(x_test, y_test, model3, 2048)
    end3 = time.time()

    print(f"Thời gian chạy với N_FFT = 512: {start2 - start1} s")
    print(f"Thời gian chạy với N_FFT = 1024: {start3 - start2} s")
    print(f"Thời gian chạy với N_FFT = 2048: {end3 - start3} s")

    print(accuracy1, accuracy2, accuracy3, sep='\n')

    confusion = confusion = confusion_matrix(y_test, y_pred1)
    
    class_names = np.unique(y_test)
    df_confusion = pd.DataFrame(confusion, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_confusion, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()