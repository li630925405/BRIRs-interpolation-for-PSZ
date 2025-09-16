from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse
from numpy import linalg as LA
import scipy.io as scio
from scipy.signal import savgol_filter


mic_num = 4
ls_num = 6
beta = 0.01
sr = 44100
time_w_len = 400
freq_w_len = time_w_len // 2 + 1
mic_idx = 41
min_freq = 2000
max_freq = 7000
num_x = 21
m = num_x * 2 - 1
dmic = int(0.6 / (4 / m))
dmic = dmic if dmic % 2 == 0 else dmic + 1


def load_tf(rir_type):
    """Load transfer function for given rir_type, num_x, and specific mic_index"""
    m = num_x * 2 - 1
    rir = np.zeros((m, ls_num, time_w_len))
    if rir_type == "siren":
        for n in range(ls_num):
            rir_data = np.load(f"rir/siren_brir_r_400_400time_{num_x}mic_src{n}_plaster_sprayed.npy")
            rir[:, n, :] = np.pad(rir_data, ((0, 0), (0, time_w_len - rir_data.shape[1])), mode='constant')
    elif rir_type == "truth":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 1].transpose(1, 2, 0)
    elif rir_type == "bilinear":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{num_x}.mat')['brir'][:time_w_len, :, :, 1].transpose(1, 2, 0)
        rir = ndimage.zoom(rir, zoom=[m/num_x, 1, 1], order=1)

    tf = np.fft.rfft(rir, axis=-1)
    tf = tf.transpose(2, 1, 0)
    tf_l = tf[:, :, [mic_idx, mic_idx+dmic]]

    rir = np.zeros((m, ls_num, time_w_len))
    if rir_type == "siren":
        for n in range(ls_num):
            rir_data = np.load(f"rir/siren_brir_400_400time_{num_x}mic_src{n}_plaster_sprayed.npy")
            rir[:, n, :] = np.pad(rir_data, ((0, 0), (0, time_w_len - rir_data.shape[1])), mode='constant')
    elif rir_type == "truth":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 0].transpose(1, 2, 0)
    elif rir_type == "bilinear":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{num_x}.mat')['brir'][:time_w_len, :, :, 0].transpose(1, 2, 0)
        rir = ndimage.zoom(rir, zoom=[m/num_x, 1, 1], order=1)

    tf = np.fft.rfft(rir, axis=-1)
    tf = tf.transpose(2, 1, 0)
    tf_r = tf[:, :, [mic_idx, mic_idx+dmic]]

    estimate_tf = np.array([tf_l[:, :, 0], tf_r[:, :, 0], tf_l[:, :, 1], tf_r[:, :, 1]])
    return estimate_tf


def get_hybrid_tf(turning_point_bin=None):
    """Create hybrid transfer function: bilinear for low freq, siren for high freq"""
    bi_tf = load_tf("bilinear")  # Shape: (4, 201, 6)
    siren_tf = load_tf("siren")  # Shape: (4, 201, 6)
    freq_resolution = sr / time_w_len  # Hz per bin
    max_freq_bin = int(max_freq / freq_resolution) + 1  # +1 to include the bin containing max_freq

    # Use dynamic turning point if provided, otherwise fallback to min_freq
    if turning_point_bin is not None:
        switch_bin = turning_point_bin
    else:
        switch_bin = int(min_freq / freq_resolution)  # fallback to min_freq

    # Convert to (freq, ls_num, mic_num) format
    bi_tf = bi_tf.transpose(1, 2, 0)  # (201, 6, 4)
    siren_tf = siren_tf.transpose(1, 2, 0)  # (201, 6, 4)

    estimate_tf = np.zeros((freq_w_len, ls_num, mic_num), complex)
    estimate_tf[:switch_bin, :, :] = bi_tf[:switch_bin, :, :]
    estimate_tf[switch_bin:, :, :] = siren_tf[switch_bin:, :, :]

    # Convert back to (mic_num, freq, ls_num) format
    return estimate_tf.transpose(2, 0, 1)


def calculate_nmse_per_frequency(truth_tf, estimate_tf):
    """Calculate NMSE for each frequency point"""
    # Convert to (freq, ls_num, mic_num) format
    truth_tf = truth_tf.transpose(1, 2, 0)  # (201, 6, 4)
    estimate_tf = estimate_tf.transpose(1, 2, 0)  # (201, 6, 4)
    
    freq_resolution = sr / time_w_len
    max_freq_bin = int(max_freq / freq_resolution) + 1
    freq_points = min(max_freq_bin, freq_w_len)
    
    nmse_per_freq = np.zeros(freq_points)
    
    for f in range(freq_points):
        # Calculate NMSE for this frequency point across all speakers and microphones
        truth_f = truth_tf[f, :, :]  # (6, 4)
        estimate_f = estimate_tf[f, :, :]  # (6, 4)
        
        # Calculate MSE
        mse = np.mean(np.abs(estimate_f - truth_f)**2)
        # Calculate signal power
        signal_power = np.mean(np.abs(truth_f)**2)
        
        # Calculate NMSE and convert to dB
        nmse = mse / signal_power
        nmse_per_freq[f] = 10 * np.log10(nmse + 1e-12)  # Convert to dB, add small value to avoid log(0)
    
    return nmse_per_freq


def main():
    # Calculate mic indices for 1m to 4m range
    start_pos = int((1.5 - 0.5) / (4.5 - 0.5) * (m - 1))  # 1m position
    start_pos = start_pos if start_pos % 2 == 0 else start_pos + 1
    end_pos = int((3.5 - 0.5) / (4.5 - 0.5) * (m - 1))    # 4m position
    mic_indices = range(start_pos + 1, end_pos - dmic + 1, 2)

    # First pass: calculate siren and bilinear to find turning point
    all_siren_nmse = []
    all_bilinear_nmse = []

    for mic_idx in mic_indices:
        globals()['mic_idx'] = mic_idx

        truth_tf = load_tf("truth")
        siren_tf = load_tf("siren")
        bilinear_tf = load_tf("bilinear")

        # Calculate NMSE for each frequency point
        siren_nmse = calculate_nmse_per_frequency(truth_tf, siren_tf)
        bilinear_nmse = calculate_nmse_per_frequency(truth_tf, bilinear_tf)

        # Store NMSE data for each microphone
        all_siren_nmse.append(siren_nmse)
        all_bilinear_nmse.append(bilinear_nmse)

    # Convert to numpy arrays for easier computation
    all_siren_nmse = np.array(all_siren_nmse)
    all_bilinear_nmse = np.array(all_bilinear_nmse)

    # Calculate average NMSE across all microphone positions
    siren_mean = np.mean(all_siren_nmse, axis=0)
    bilinear_mean = np.mean(all_bilinear_nmse, axis=0)

    # Find turning point where siren NMSE becomes lower than bilinear
    freq_resolution = sr / time_w_len
    start_freq = 100  # Hz
    start_bin = int(start_freq / freq_resolution)
    diff = bilinear_mean - siren_mean  # Positive when siren is better (lower NMSE)
    turning_indices = np.where(diff[start_bin:] > 0)[0]

    if len(turning_indices) > 0:
        turning_point_bin = turning_indices[0] + start_bin
        turning_point_freq = turning_point_bin * freq_resolution
        print(f"Dynamic turning point found at bin {turning_point_bin}, frequency: {turning_point_freq:.1f} Hz")
        print(f"At turning point: siren NMSE = {siren_mean[turning_point_bin]:.4f}, bilinear NMSE = {bilinear_mean[turning_point_bin]:.4f}")
    else:
        turning_point_bin = int(min_freq / freq_resolution)  # fallback
        turning_point_freq = min_freq
        print(f"No turning point found, using fallback frequency: {turning_point_freq:.1f} Hz")

    # Second pass: calculate hybrid with dynamic turning point
    all_hybrid_nmse = []

    for mic_idx in mic_indices:
        globals()['mic_idx'] = mic_idx

        truth_tf = load_tf("truth")
        hybrid_tf = get_hybrid_tf(turning_point_bin)
        hybrid_nmse = calculate_nmse_per_frequency(truth_tf, hybrid_tf)
        all_hybrid_nmse.append(hybrid_nmse)

    # Convert to numpy arrays and calculate statistics
    all_hybrid_nmse = np.array(all_hybrid_nmse)
    siren_std = np.std(all_siren_nmse, axis=0)
    bilinear_std = np.std(all_bilinear_nmse, axis=0)
    hybrid_mean = np.mean(all_hybrid_nmse, axis=0)
    hybrid_std = np.std(all_hybrid_nmse, axis=0)
    
    # 打印每个方法的平均NMSE和标准差
    print(f"Average NMSE values:")
    print(f"  SIREN: {np.mean(siren_mean):.4f} ± {np.mean(siren_std):.4f} dB")
    print(f"  Bilinear: {np.mean(bilinear_mean):.4f} ± {np.mean(bilinear_std):.4f} dB")
    print(f"  Hybrid: {np.mean(hybrid_mean):.4f} ± {np.mean(hybrid_std):.4f} dB")

    # Print turning point details (already calculated above)
    print(f"Hybrid switching point: {turning_point_freq:.1f} Hz (bin {turning_point_bin})")
    print(f"At turning point: siren NMSE = {siren_mean[turning_point_bin]:.4f}, bilinear NMSE = {bilinear_mean[turning_point_bin]:.4f}")

    # Print key frequency points
    key_freqs = [100, 200, 500, 1000, 1500, 2000, 2500]
    print("\nKey frequency points NMSE:")
    for freq in key_freqs:
        bin_idx = int(freq / freq_resolution)
        if bin_idx < len(siren_mean):
            print(f"{freq:4d} Hz: siren = {siren_mean[bin_idx]:.4f}, bilinear = {bilinear_mean[bin_idx]:.4f}, hybrid = {hybrid_mean[bin_idx]:.4f}")
    print("="*50)
    
    # Set font and style for publication
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    # Create visualization
    freq_resolution = sr / time_w_len
    max_freq_bin = int(max_freq / freq_resolution) + 1
    freq_points = min(max_freq_bin, freq_w_len)
    xx = np.arange(freq_points) * freq_resolution

    fig = plt.figure(figsize=(4, 2.5))
    axes = fig.add_subplot(111)

    # 绘制平均值曲线
    axes.semilogx(xx, siren_mean, color='blue', linewidth=2,
                  label="SIREN")
    axes.semilogx(xx, bilinear_mean, color='green', linewidth=2,
                  label="linear")
    axes.semilogx(xx, hybrid_mean, color='red', linewidth=2,
                  label="hybrid")

    axes.fill_between(xx, bilinear_mean - bilinear_std, bilinear_mean + bilinear_std,
                      alpha=0.2, color='green')
    axes.fill_between(xx, siren_mean - siren_std, siren_mean + siren_std,
                      alpha=0.2, color='blue')
    axes.fill_between(xx, hybrid_mean - hybrid_std, hybrid_mean + hybrid_std,
                      alpha=0.4, color='red')

    axes.axvline(x=turning_point_freq, color='red', linestyle='--', linewidth=1.5)

    plt.ylabel("NMSE [dB]")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(100, max_freq)  # Set x-axis limit to focus on the frequency range of interest
    plt.grid(True, which="both", ls="-", alpha=0.3)
    # Move legend to the right side
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"nmse_frequency_{num_x}_{start_pos}_{end_pos}_plaster_sprayed.svg", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()