from scipy import io
import numpy as np
import librosa

def read_mat(f):
    mat = io.loadmat(f)
    
    sampling_rate = int(mat['dataStruct'][0, 0][1][0, 0])
    n_samples = int(mat['dataStruct'][0, 0][2][0, 0])
    ch_indexes = np.int8(mat['dataStruct'][0, 0][3][0])
    
    data = mat['dataStruct'][0, 0][0].T
    waves = np.zeros([len(ch_indexes), n_samples], dtype=np.float32)
    
    for i in range(len(ch_indexes)):
        waves[i] = data[ch_indexes[i] - 1]  
        
    return waves

# compute msg for a wave
def compute_msg(wave, \
                sr, hop_length, \
                n_fft, n_mels, \
                desired_msg_w):
    
    # pad wave if neccessary to get the desired msg width
    desired_wave_len = hop_length * (desired_msg_w - 1)
    if desired_wave_len > len(wave):
        wave = np.pad(wave, (0, desired_wave_len - len(wave)), \
                      'constant', constant_values=(0))
    
    msg = librosa.feature.melspectrogram(y=wave, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    msg = librosa.logamplitude(msg ** 2, ref_power=1.)
    
    return msg