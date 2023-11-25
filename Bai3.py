import librosa
import numpy as np
import matplotlib.pyplot as plt

def part_1():
    # Đọc file .wav
    audio_signal, sr = librosa.load(r"C:\Users\thain\PycharmProjects\XLTHS_EX3\signals\NguyenAmHuanLuyen-16k\23MTL\a.wav", sr=None)  # sr (sample rate) is kept as original
    # Preprocess the signal (for example, remove noise)
    audio_signal_preprocessed = librosa.effects.preemphasis(audio_signal)
    # Detect silent sections
    audio_signal_trimmed, trim_index = librosa.effects.trim(audio_signal_preprocessed, top_db=20)  # The top_db value is based on the volume level
    # Plotting
    times = librosa.times_like(audio_signal_preprocessed, sr=sr)
    plt.figure(figsize=(15, 5))
    plt.plot(times, audio_signal_preprocessed, label='Audio Signal')
    if trim_index.any():
        plt.axvspan(times[trim_index[0]], times[trim_index[1]], color='red', alpha=0.5, label='Trimmed Silence if red')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal with Trimmed Sections')
    plt.legend()
    plt.show()


def segment_vowel_silence(audio, Fs, threshold=0.04, min_duration=0.3):
    print(len(audio))

    # Chia khung tín hiệu, mỗi khung độ dài 20ms
    frame_length = int(0.02 * Fs)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length)
    # print(len(frames))
    # Tính STE từng khung
    ste = np.sum(np.square(frames), axis=0)
    # Chuẩn hóa STE
    ste_normalized = (ste - np.min(ste)) / (np.max(ste) - np.min(ste))
    print(ste_normalized)
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

    num_speech_frames = np.sum(is_speech)
    speech_start = np.where(is_speech)[0][0]
    speech_end = np.where(is_speech)[0][-1]
    # Lấy đoạn giữa của phần nguyên âm
    middle_third_start = speech_start + num_speech_frames // 3
    middle_third_end = speech_end - num_speech_frames // 3
    middle_part = audio[middle_third_start:middle_third_end]
    N_MFCC = 13
    # Trích xuất MFCC từ đoạn âm thanh (ví dụ là middle_part từ bước 2)
    mfccs = librosa.feature.mfcc(y=middle_part, sr=Fs, n_mfcc=N_MFCC)
    print(mfccs)
    return mfccs


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
        audio, Fs = librosa.load(file_path, sr=None)
        MFCC1 = segment_vowel_silence(audio, Fs,threshold=0.04, min_duration=0.3)
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

def MFCC_1vowel_nspeaker_test(vowelchar):
    """ Hàm tính vector đặc trưng fft cho 1 nguyên âm (không phụ thuộc người nói)
        - Đầu vào là 1 ký hiệu nguyên âm ('a',.., 'u') = tên tệp
        - Bằng cách tính trung bình cộng của 21 người nói khác nhau
        - Trả về 1 vector fft cuối cùng ---> để bỏ vào model
    """
    name_folders = ["23MTL", "24FTL", "25MLM", "27MCM", "28MVN", "29MHN", "30FTN", "32MTP", "33MHP", "34MQP", "35MMQ",
                    "36MAQ", "37MDS", "38MDS", "39MTS", "40MHS", "41MVS", "42FQT", "43MNT", "44MTT", "45MDV"]
    file_path_template = 'signals/NguyenAmKiemThu-16k/{}/{}.wav'
    vectors = np.array()
    for foldername in name_folders:
        file_path = file_path_template.format(foldername, vowelchar)
        print(file_path)  # Dòng này sau này xóa
        audio, Fs = librosa.load(file_path, sr=None)
        fft1 = segment_vowel_silence(audio, Fs,threshold=0.04, min_duration=0.3)
        vectors.append(fft1)
    vector_MFCC = np.mean(vectors, axis=0)
    print(f"Đã xong chữ {vowelchar}, len(vector_fft) = {len(vector_MFCC)}")
    return vector_MFCC

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def extract_mfcc_from_wav(file_path):
    # Đọc file wav và lấy mẫu
    signal, sr = librosa.load(file_path, sr=None)
    # Trích xuất MFCC
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)  # 13 là số lượng hệ số MFCC mặc định
    # để tạo thành một vector đặc trưng duy nhất cho toàn bộ đoạn âm thanh
    mfccs_mean = np.mean(mfccs, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(mfccs_mean, label=f'MFCC {0 + 1}')
    plt.title('First 5 MFCC Vectors Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('MFCC Coefficient Value')
    plt.legend()
    plt.show()
    return mfccs_mean


file_path = r'signals/NguyenAmHuanLuyen-16k/23MTL/a.wav'
# Đọc file âm thanh
audio, Fs = librosa.load(file_path, sr=None)
# Gọi hàm để thực hiện segment và nhận đoạn tín hiệu nguyên âm
vowel = segment_vowel_silence(audio, Fs, threshold=0.04, min_duration=0.3)

model_vector = build_model()
extract_mfcc_from_wav(file_path)
# Vector MFCC của tín hiệu đầu vào
input_mfcc_vector = vowel
# Tập hợp các vector MFCC đặc trưng cho mỗi nguyên âm từ tập huấn luyện
training_vectors = {'a' : model_vector[0] , 'e': model_vector[1] , 'i': model_vector[2] , 'o': model_vector[3] , 'u' : model_vector[4]}  # e.g., {'a': vector_a, 'e': vector_e, ..., 'u': vector_u}
# Khởi tạo biến để lưu khoảng cách nhỏ nhất và nguyên âm tương ứng
min_distance = float('inf')
recognized_vowel = None
# Tính khoảng cách và tìm khoảng cách nhỏ nhất
for vowel, feature_vector in training_vectors.items():
    distance = euclidean_distance(input_mfcc_vector, feature_vector)
    if distance < min_distance:
        min_distance = distance
        recognized_vowel = vowel
# Kết quả nhận dạng
print(f"Nguyên âm nhận dạng là: {recognized_vowel} với khoảng cách: {min_distance}")






