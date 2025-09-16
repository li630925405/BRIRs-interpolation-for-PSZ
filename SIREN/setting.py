import numpy as np
import scipy.io as scio
import torch
import random
import os

num_x = 111
src_idx = 5
original_time_len = 400
data_type = "brir_r"
model_type = "siren"
sr = 44100
training_step = 2000
pinn_weight = 5 * 1e-10
collocation = num_x * 2 - 1
c = 340
hidden_features = 128
folder = "./res/"
max_freq = 7000
min_freq = 0
omega = 10

seed_value = 3407   # 设定随机数种子
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True


if data_type == "brir":
    m = num_x * 2 - 1
    truth = scio.loadmat('./brir_plaster_sprayed/brir_' + str(m) + '.mat')['brir'][:, :, src_idx, 0]  # (400, 41, 6)
elif data_type == "brir_r":
    m = num_x * 2 - 1
    truth = scio.loadmat('./brir_plaster_sprayed/brir_' + str(m) + '.mat')['brir'][:, :, src_idx, 1]  # (400, 41, 6)


truth = truth.transpose()[:, :original_time_len]  # (41, 400)
rir = truth[::2, :original_time_len]  # (21, 400)

freq_resolution = sr / original_time_len  # Hz per bin
max_freq_bin = int(max_freq / freq_resolution) + 1  # +1 to include the bin containing max_freq
min_freq_bin = int(min_freq / freq_resolution)  # bin containing min_freq

# time_len = int(original_time_len * 2 * max_freq / sr)
time_len = original_time_len

rir_fft_full = np.fft.rfft(rir, axis=-1)
rir_fft_filtered = rir_fft_full[:, min_freq_bin:max_freq_bin]
rir = np.fft.irfft(rir_fft_filtered, n=time_len)

truth_fft_full = np.fft.rfft(truth, axis=-1)
truth_fft_filtered = truth_fft_full[:, min_freq_bin:max_freq_bin]
truth = np.fft.irfft(truth_fft_filtered, n=time_len)


name = f"{data_type}_{model_type}_{original_time_len}_{time_len}time_{num_x}mic_src{src_idx}_plaster_sprayed"
params = f"{data_type}_{original_time_len}_{time_len}time_{num_x}mic_src{src_idx}_plaster_sprayed"
name = folder + name