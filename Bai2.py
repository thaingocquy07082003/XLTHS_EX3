import os
import numpy as np
import pandas as pd
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from math import floor
from scipy.signal import hamming

#--------------------variables--------------------

#---------------------------vector_file:

vector_filelist = ['FFT.txt']
#-----------note
'''
FFT.txt: N_FFT = 1024, frame_len = 0.02, frame_shift = 0.02
'''
#-----------end note

vector_file = vector_filelist[0]
#---------------------------N_FFT values:

N_FFT_list = [512, 1024, 2048]
N_FFT = N_FFT_list[1]

#---------------------------frame_len values:

frame_len_value_list = [0.01, 0.02 ,0.03]
frame_len_value = frame_len_value_list[1]

#---------------------------frame_shift values:

frame_shift_value_list = [0.01, 0.02, 0.03]
frame_shift_value = frame_shift_value_list[1]

#---------------------------folder:

#---------tập huấn luyện:
listfileHL =[
        "signals/NguyenAmHuanLuyen-16k/23MTL/",
        "signals/NguyenAmHuanLuyen-16k/24FTL/",
        "signals/NguyenAmHuanLuyen-16k/25MLM/",
        "signals/NguyenAmHuanLuyen-16k/27MCM/",
        "signals/NguyenAmHuanLuyen-16k/28MVN/",
        "signals/NguyenAmHuanLuyen-16k/29MHN/",
        "signals/NguyenAmHuanLuyen-16k/30FTN/",
        "signals/NguyenAmHuanLuyen-16k/32MTP/",
        "signals/NguyenAmHuanLuyen-16k/33MHP/",
        "signals/NguyenAmHuanLuyen-16k/34MQP/",
        "signals/NguyenAmHuanLuyen-16k/35MMQ/",
        "signals/NguyenAmHuanLuyen-16k/36MAQ/",
        "signals/NguyenAmHuanLuyen-16k/37MDS/",
        "signals/NguyenAmHuanLuyen-16k/38MDS/",
        "signals/NguyenAmHuanLuyen-16k/39MTS/",
        "signals/NguyenAmHuanLuyen-16k/40MHS/",
        "signals/NguyenAmHuanLuyen-16k/41MVS/",
        "signals/NguyenAmHuanLuyen-16k/42FQT/",
        "signals/NguyenAmHuanLuyen-16k/43MNT/",
        "signals/NguyenAmHuanLuyen-16k/44MTT/",
        "signals/NguyenAmHuanLuyen-16k/45MDV/"]
#---------tập kiểm thử:
listfileKT = [
        "signals/NguyenAmKiemThu-16k/01MDA/",
        "signals/NguyenAmKiemThu-16k/02FVA/",
        "signals/NguyenAmKiemThu-16k/03MAB/",
        "signals/NguyenAmKiemThu-16k/04MHB/",
        "signals/NguyenAmKiemThu-16k/05MVB/",
        "signals/NguyenAmKiemThu-16k/06FTB/",
        "signals/NguyenAmKiemThu-16k/07FTC/",
        "signals/NguyenAmKiemThu-16k/08MLD/",
        "signals/NguyenAmKiemThu-16k/09MPD/",
        "signals/NguyenAmKiemThu-16k/10MSD/",
        "signals/NguyenAmKiemThu-16k/11MVD/",
        "signals/NguyenAmKiemThu-16k/12FTD/",
        "signals/NguyenAmKiemThu-16k/14FHH/",
        "signals/NguyenAmKiemThu-16k/15MMH/",
        "signals/NguyenAmKiemThu-16k/16FTH/",
        "signals/NguyenAmKiemThu-16k/17MTH/",
        "signals/NguyenAmKiemThu-16k/18MNK/",
        "signals/NguyenAmKiemThu-16k/19MXK/",
        "signals/NguyenAmKiemThu-16k/20MVK/",
        "signals/NguyenAmKiemThu-16k/21MTL/",
        "signals/NguyenAmKiemThu-16k/22MHL/"]

#----------------------end----------------------

def get_vector_from_file():
    
    FFT = np.loadtxt(vector_file, delimiter='\t', skiprows=1)
    FFTa = FFT[:, 0]
    FFTe = FFT[:, 1]
    FFTi = FFT[:, 2]
    FFTo = FFT[:, 3]
    FFTu = FFT[:, 4]
    
    return FFTa, FFTe, FFTi, FFTo, FFTu

def display_matrix_as_table(matrix, dcx, dcxa, dcxe, dcxi, dcxo, dcxu):
    columns = ['a', 'e', 'i', 'o', 'u']
    index = ['a', 'e', 'i', 'o', 'u']
    
    accuracy_column = [dcxa / 21 * 100, dcxe / 21 * 100, dcxi / 21 * 100, dcxo / 21 * 100, dcxu / 21 * 100]
    df = pd.DataFrame(matrix, columns=columns, index=index)
    df.insert(loc=len(columns), column='Độ chính xác (%)', value=accuracy_column)
    print("\n")
    print("Kết quả với N_FFT = " + str(N_FFT) + ", frame_len = " + str(frame_len_value) + ", frame_shift = " + str(frame_shift_value) + ":")
    print("=================================================")
    print(df)
    print("=================================================")
    print('Độ chính xác tổng thể: ', dcx / 105 * 100, '%')
    print("\n")

def Show_Vector2():
    FFTa, FFTe, FFTi, FFTo, FFTu = get_vector_from_file()

    vowels = ['a', 'e', 'i', 'o', 'u']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Đặc trưng của các nguyên âm')
    ax.set_xlabel('Mẫu FFT')
    ax.set_ylabel('Giá trị FFT')
    ax.grid(True)

    for i, vowel in enumerate(vowels):
        ax.plot(eval(f'FFT{vowel}'), '-', linewidth=1, label=f'{vowel}')

    ax.legend(['/a/', '/e/', '/i/', '/o/', '/u/'])
    ax.set_title('Đồ thị N_FFT bằng ' + str(N_FFT) + ', frame_len bằng ' + str(frame_len_value) + ', frame_shift bằng ' + str(frame_shift_value) + ':')
    
    plt.tight_layout()
    plt.show()

def Show_Vector():

    FFTa, FFTe, FFTi, FFTo, FFTu = get_vector_from_file()

    vowels = ['a', 'e', 'i', 'o', 'u']
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))

    for i in range(5):
        ax = axs[i]
        ax.set_title(f'Vector dac trung cua {vowels[i]}')
        ax.set_xlabel('')
        ax.set_ylabel('FFT Value')
        ax.grid(True)

    for i in range(5):
        axs[0].plot(FFTa, '-', linewidth=1, label=f'a{i}')
        axs[1].plot(FFTe, '-', linewidth=1, label=f'e{i}')
        axs[2].plot(FFTi, '-', linewidth=1, label=f'i{i}')
        axs[3].plot(FFTo, '-', linewidth=1, label=f'o{i}')
        axs[4].plot(FFTu, '-', linewidth=1, label=f'u{i}')

    plt.tight_layout()
    plt.show()

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def find_vowel(center_vectors):
    
    vector_a, vector_e, vector_i, vector_o, vector_u = get_vector_from_file()
    
    distances = {
        'a': euclidean_distance(center_vectors, vector_a),
        'e': euclidean_distance(center_vectors, vector_e),
        'i': euclidean_distance(center_vectors, vector_i),
        'o': euclidean_distance(center_vectors, vector_o),
        'u': euclidean_distance(center_vectors, vector_u),
    }

    vowel = min(distances, key=distances.get)
    
    return vowel

def third_voice(time_start, time_end):
    tb = np.round((time_end - time_start) / 3.0)
    t1 = time_start + tb
    t2 = time_end - tb
    return t1, t2

def normalize(energy, power, x):
    energy = energy / np.max(energy)
    power = power / np.max(power)
    x = x / np.max(np.abs(x))
    return energy, power, x

def confusion_matrix(right_vowel, vowel, matrix):
    list_vowel = ['a', 'e', 'i', 'o', 'u']
    for i in range(len(list_vowel)):
        if right_vowel == list_vowel[i]:
            for j in range(len(list_vowel)):
                if vowel == list_vowel[j]:
                    matrix[i][j] += 1
    return matrix

def short_time_energy(x, fs, fd, f_overlap):
    f_size = round(fd * fs)
    f_overlap = round(f_overlap * fs)
    n = len(x)
    lenF = int(np.floor((n - f_overlap) / (f_size - f_overlap)))
    temp = 0
    Energy = np.zeros(n)
    Power = np.zeros(lenF)
    for i in range(lenF):
        if temp + f_size - f_overlap <= n:
            frame = 0
            for j in range(temp, temp + f_size):
                frame = frame + np.abs(x[j]) ** 2
            Energy[temp: temp + f_size] = frame
            Power[i] = frame
            temp = round(temp + f_size - f_overlap)
    return Energy, Power, f_size, lenF

def remove_silence(x, lenF, Power, ThresholdSTE, f_size, fs):
    f_overlap = 0.02
    voice = np.zeros(lenF)
    for i in range(lenF):
        if Power[i] > ThresholdSTE:
            voice[i] = 1
        else:
            voice[i] = 0

    n_len = 0
    vowel = np.array([])
    j = 1
    for i in range(lenF - 1):
        if i == 0:
            n_len = n_len + f_size / fs
        else:
            n_len = n_len + f_size / fs - f_overlap

        if voice[i] == 0 and voice[i + 1] == 1:
            vowel = np.append(vowel, n_len)
            j = j + 1

    j = 1
    n_len = 0
    for i in range(lenF - 1):
        if i == 0:
            n_len = n_len + f_size / fs
        else:
            n_len = n_len + f_size / fs - f_overlap

        if voice[i] == 1 and voice[i + 1] == 0:
            vowel = np.append(vowel, n_len)
            j = j + 1

    check_speech = np.ones(len(vowel))
    for i in range(1, len(vowel) - 1, 2):
        if abs(vowel[i + 1] - vowel[i]) < 0.3:
            check_speech[i] = 0
            check_speech[i + 1] = 0

    speech = np.array([])
    j = 1
    for i in range(len(vowel)):
        if check_speech[i] == 1:
            speech = np.append(speech, vowel[i])
            j = j + 1

    return speech

def detected_silence(file_input):
    x, Fs = sf.read(file_input)
    Energy, Power, f_size, lenF = short_time_energy(x, Fs, 0.03, 0.02)
    ThresholdSTE = 0.3
    Energy, Power, x = normalize(Energy, Power, x)
    vowel = remove_silence(x, lenF, Power, ThresholdSTE, f_size, Fs)
    pointstart = vowel[0]
    pointend = vowel[1]
    return pointstart, pointend


def find_fft(y, time_start, time_end, fs):
    frame_shift = (frame_shift_value * fs)
    frame_len = int(frame_len_value * fs)
    number_frame = int((time_end - time_start) / frame_shift)
    hamming_wd = hamming((frame_len))
    ffts = []

    for frame_index in range(1, number_frame + 1):
        start = (frame_index - 1) * frame_shift + time_start
        finish = min(len(y), start + frame_len)
        f_size = np.arange(start, finish).astype(int)
        frame = hamming_wd * y[f_size]
        p = np.abs(np.fft.fft(frame, N_FFT))
        p1 = p[:len(p) // 2]
        ffts.append(p1)

    ffts = np.mean(np.array(ffts), axis=0)
    return ffts

def Trich_Vecto_Dac_Trung_1_Nguoi(file, start, finish):
    y, Fs = sf.read(file)
    time_start = np.round(start * Fs).astype(int)
    time_end = np.round(finish * Fs).astype(int)
    t_start = time_start + ((time_end - time_start) // 3)
    t_end = time_start + 2 * ((time_end - time_start) // 3)
    ffts = find_fft(y, t_start, t_end, Fs)
    center_vectors = ffts
    return center_vectors

def Trich_Vecto_Dac_Trung(id):
    FFTs = []
    for person in range(len(listfileHL)):
        if id == 1:
            file = listfileHL[person] + 'a.wav'
        elif id == 2:
            file = listfileHL[person] + 'e.wav'
        elif id == 3:
            file = listfileHL[person] + 'i.wav'
        elif id == 4:
            file = listfileHL[person] + 'o.wav'
        elif id == 5:
            file = listfileHL[person] + 'u.wav'
    
        y, Fs = sf.read(file)
        start, end = detected_silence(file)
        time_start = np.round(start * Fs).astype(int)
        time_end = np.round(end * Fs).astype(int)
        T1, T2 = third_voice(time_start, time_end)
        ffts = find_fft(y, T1, T2, Fs)
        FFTs.append(ffts)
            
    center_vectors = np.mean(np.array(FFTs), axis=0)

    return center_vectors

def save_file():
    
    FFTa = Trich_Vecto_Dac_Trung(1)
    FFTe = Trich_Vecto_Dac_Trung(2)
    FFTi = Trich_Vecto_Dac_Trung(3)
    FFTo = Trich_Vecto_Dac_Trung(4)
    FFTu = Trich_Vecto_Dac_Trung(5)
    # print(FFTa)
    # print(FFTe)
    # print(FFTi)
    # print(FFTo)
    # print(FFTu)
    
    np.savetxt(vector_file, np.column_stack((FFTa, FFTe, FFTi, FFTo, FFTu)), fmt="%s", delimiter='\t', header="FFTa\tFFTe\tFFTi\tFFTo\tFFTu")

def process_vowel(vowel, suffix, counters, matrix):

    for person, filepath in enumerate(listfileKT):
        filename = filepath + suffix
        t_start, t_end = detected_silence(filename)
        center_vectors = Trich_Vecto_Dac_Trung_1_Nguoi(filename, t_start, t_end)
        recognized_vowel = find_vowel(center_vectors)

        if recognized_vowel == vowel:
            counters['dcx'] += 1
            counters[f'dcx{vowel}'] += 1

        matrix = confusion_matrix(vowel, recognized_vowel, matrix)

    return counters, matrix

def vowel_recognition():
    counters = {'dcx': 0, 'dcxa': 0, 'dcxe': 0, 'dcxi': 0, 'dcxo': 0, 'dcxu': 0}
    matrix = np.zeros((5, 5))
    
    for i, vowel in enumerate(['a', 'e', 'i', 'o', 'u'], start=1):
        suffix = vowel + '.wav'
        counters, matrix = process_vowel(vowel, suffix, counters, matrix)
        
    display_matrix_as_table(matrix, counters['dcx'], counters['dcxa'], counters['dcxe'], counters['dcxi'], counters['dcxo'], counters['dcxu'])

if __name__ == "__main__":
    # Trích vector đặc trưng từ NguyenAmHuanLuyen và lưu vào file
    # save_file()
    
    # Nhận diện nguyên âm từ NguyenAmKiemThu
    vowel_recognition()
    
    # Hiển thị vector đặc trưng
    # Show_Vector()
    Show_Vector2()