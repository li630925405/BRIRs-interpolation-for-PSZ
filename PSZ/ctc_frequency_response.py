from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import signal


mic_num = 4
ls_num = 6
beta = 0.01
sr = 44100
time_w_len = 400
freq_w_len = time_w_len // 2 + 1
mic_idx = 1
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
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 1].transpose(1, 2, 0) 
        rir = rir[::2, :, :]
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
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 0].transpose(1, 2, 0) 
        rir = rir[::2, :, :]
        rir = ndimage.zoom(rir, zoom=[m/num_x, 1, 1], order=1)

    tf = np.fft.rfft(rir, axis=-1)
    tf = tf.transpose(2, 1, 0)
    tf_r = tf[:, :, [mic_idx, mic_idx+dmic]]

    estimate_tf = np.array([tf_l[:, :, 0], tf_r[:, :, 0], tf_l[:, :, 1], tf_r[:, :, 1]])
    return estimate_tf


def get_combine_tf(turning_point_bin=None):
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

    return estimate_tf


def calculate_filter(estimate_tf, method):
    g_a = estimate_tf[:, :, 0:2]
    g_b = estimate_tf[:, :, 2:4]

    q_matrix = np.zeros((freq_w_len, ls_num), complex)

    if method == "PM":
        for i in range(freq_w_len):
            d_a = [1, 1]  # 2 mics in bright zone
            d_b = [0, 0]  # 2 mics in dark zone

            ga = g_a[i].T  # (2, 6) - 2 mics, 6 speakers
            gb = g_b[i].T  # (2, 6) - 2 mics, 6 speakers

            h = np.concatenate((ga, gb), 0)  # (4, 6) - 4 mics, 6 speakers
            h_h = h.conj().T  # (6, 4) - 6 speakers, 4 mics
            d = np.array(d_a + d_b)  # [1, 0]
            m = h_h @ h + beta * np.identity(h.shape[1])
            q = np.linalg.solve(h_h @ h + beta * np.identity(h.shape[1]), h_h @ d)
            q_matrix[i, :] = q

    return q_matrix


def calculate_contrst(q_matrix, smooth=False):
    truth_tf = load_tf("truth")
    # Convert to (freq, ls_num, mic_num) format
    truth_tf = truth_tf.transpose(1, 2, 0)
    g_a = truth_tf[:, :, 0:2]  # (201, 6, 2)
    g_b = truth_tf[:, :, 2:4]  # (201, 6, 2)

    bright = np.zeros((freq_w_len), complex)
    dark = np.zeros((freq_w_len), complex)

    for i in range(ls_num):
        # For each speaker, sum over all microphones in bright/dark zones
        bright += q_matrix[:, i] * np.sum(g_a[:, i, :], axis=1)  # (201,) * (201,) -> (201,)
        dark += q_matrix[:, i] * np.sum(g_b[:, i, :], axis=1)    # (201,) * (201,) -> (201,)

    contrast = 20 * np.log10(np.abs(bright) / np.abs(dark))
    
    freq_resolution = sr / time_w_len
    max_freq_bin = int(max_freq / freq_resolution) + 1
    freq_points = min(max_freq_bin, freq_w_len)
    contrast = contrast[:freq_points]
    
    return contrast


def downsample_tf(tf, new_time_len):
    """
    Downsample transfer function to match SIREN's time resolution
    Args:
        tf: Transfer function in shape (mic_num, freq_w_len, ls_num) or (freq_w_len, ls_num, mic_num)
        new_time_len: Target time length (from SIREN: int(time_len * 2 * max_freq / sr))
    Returns:
        Downsampled transfer function
    """
    # Calculate new frequency length
    new_freq_len = new_time_len // 2 + 1
    
    # Handle different input shapes
    if tf.shape[1] == freq_w_len:  # Shape: (mic_num, freq_w_len, ls_num)
        # Take first new_freq_len frequency bins
        return tf[:, :new_freq_len, :]
    elif tf.shape[0] == freq_w_len:  # Shape: (freq_w_len, ls_num, mic_num)
        # Take first new_freq_len frequency bins
        return tf[:new_freq_len, :, :]
    else:
        raise ValueError(f"Unexpected tf shape: {tf.shape}")


def calculate_tf_metrics(truth_tf, estimate_tf):
    freq_resolution = sr / time_w_len  # Hz per bin
    max_freq_bin = int(max_freq / freq_resolution) + 1  # +1 to include the bin containing max_freq
    min_freq_bin = int(min_freq / freq_resolution)  # bin containing min_freq
    freq_len = max_freq_bin - min_freq_bin

    truth_tf = truth_tf[min_freq_bin:max_freq_bin, :, :]
    estimate_tf = estimate_tf[min_freq_bin:max_freq_bin, :, :]

    signal_power = np.mean(np.abs(truth_tf)**2)
    noise_power = np.mean(np.abs(estimate_tf - truth_tf)**2)

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def main():
    method = "PM"
    smooth = False
    # Calculate mic indices for 1m to 4m range
    # Total array length: 4m (0.5m to 4.5m), with m=81 positions
    # Position 0 = 0.5m, position 80 = 4.5m
    # 1m corresponds to position: (1-0.5)/(4.5-0.5) * 80 = 0.5/4 * 80 = 10
    # 4m corresponds to position: (4-0.5)/(4.5-0.5) * 80 = 3.5/4 * 80 = 70

    start_pos = int((1.5 - 0.5) / (4.5 - 0.5) * (m - 1))  # 1m position
    start_pos = start_pos if start_pos % 2 == 0 else start_pos + 1
    end_pos = int((3.5 - 0.5) / (4.5 - 0.5) * (m - 1))    # 4m position
    mic_indices = range(start_pos + 1, end_pos - dmic + 1, 2) 

    # First pass: calculate siren and bilinear to find turning point
    all_siren_contrast = []
    all_bilinear_contrast = []

    for mic_idx in mic_indices:
        globals()['mic_idx'] = mic_idx

        truth_tf = load_tf("truth")
        siren_tf = load_tf("siren")
        bilinear_tf = load_tf("bilinear")

        # Convert to (freq, ls_num, mic_num) format for calculate_filter
        siren_tf_for_filter = siren_tf.transpose(1, 2, 0)
        bilinear_tf_for_filter = bilinear_tf.transpose(1, 2, 0)

        siren_q = calculate_filter(siren_tf_for_filter, method)
        bilinear_q = calculate_filter(bilinear_tf_for_filter, method)

        siren_contrast = calculate_contrst(siren_q, smooth)
        bilinear_contrast = calculate_contrst(bilinear_q, smooth)

        # 存储每个麦克风的contrast数据
        all_siren_contrast.append(siren_contrast)
        all_bilinear_contrast.append(bilinear_contrast)

    # Calculate turning point from average contrast
    all_siren_contrast = np.array(all_siren_contrast)
    all_bilinear_contrast = np.array(all_bilinear_contrast)

    siren_mean = np.mean(all_siren_contrast, axis=0)
    bilinear_mean = np.mean(all_bilinear_contrast, axis=0)

    # Find turning point
    freq_resolution = sr / time_w_len
    start_freq = 100  # Hz
    start_bin = int(start_freq / freq_resolution)
    diff = siren_mean - bilinear_mean
    turning_indices = np.where(diff[start_bin:] > 0)[0]

    if len(turning_indices) > 0:
        turning_point_bin = turning_indices[0] + start_bin
        turning_point_freq = turning_point_bin * freq_resolution
        print(f"Dynamic turning point found at bin {turning_point_bin}, frequency: {turning_point_freq:.1f} Hz")
    else:
        turning_point_bin = int(min_freq / freq_resolution)  # fallback
        turning_point_freq = min_freq
        print(f"No turning point found, using fallback frequency: {turning_point_freq:.1f} Hz")

    # Second pass: calculate hybrid with dynamic turning point
    all_combine_contrast = []

    for mic_idx in mic_indices:
        globals()['mic_idx'] = mic_idx

        combine_tf = get_combine_tf(turning_point_bin)
        combine_q = calculate_filter(combine_tf, method)
        combine_contrast = calculate_contrst(combine_q, smooth)
        all_combine_contrast.append(combine_contrast)


    all_combine_contrast = np.array(all_combine_contrast)

    siren_std = np.std(all_siren_contrast, axis=0)
    bilinear_std = np.std(all_bilinear_contrast, axis=0)
    combine_mean = np.mean(all_combine_contrast, axis=0)
    combine_std = np.std(all_combine_contrast, axis=0)
    
    print(f"Average Contrast values:")
    print(f"  SIREN: {np.mean(siren_mean):.2f} ± {np.mean(siren_std):.2f} dB")
    print(f"  Bilinear: {np.mean(bilinear_mean):.2f} ± {np.mean(bilinear_std):.2f} dB")
    print(f"  Hybrid: {np.mean(combine_mean):.2f} ± {np.mean(combine_std):.2f} dB")

    # Print turning point details (already calculated above)
    print(f"Hybrid switching point: {turning_point_freq:.1f} Hz (bin {turning_point_bin})")
    print(f"At turning point: siren = {siren_mean[turning_point_bin]:.2f} dB, bilinear = {bilinear_mean[turning_point_bin]:.2f} dB")

    key_freqs = [100, 200, 500, 1000, 1500, 2000, 2500]
    print("\nKey frequency points:")
    for freq in key_freqs:
        bin_idx = int(freq / freq_resolution)
        if bin_idx < len(siren_mean):
            print(f"{freq:4d} Hz: siren = {siren_mean[bin_idx]:6.2f} dB, bilinear = {bilinear_mean[bin_idx]:6.2f} dB, hybrid = {combine_mean[bin_idx]:6.2f} dB")
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


    freq_resolution = sr / time_w_len
    max_freq_bin = int(max_freq / freq_resolution) + 1
    freq_points = min(max_freq_bin, freq_w_len)  # 取最小值，避免超出数据范围
    xx = np.arange(freq_points) * freq_resolution  # 使用实际的频率bin
    fig = plt.figure(figsize=(4, 2.5))
    axes = fig.add_subplot(111)
    
    axes.semilogx(xx, siren_mean, color='blue', linewidth=2,
                  label="SIREN")
    axes.semilogx(xx, bilinear_mean, color='green', linewidth=2,
                  label="linear")
    axes.semilogx(xx, combine_mean, color='red', linewidth=2,
                  label="hybrid")

    axes.fill_between(xx, bilinear_mean - bilinear_std, bilinear_mean + bilinear_std,
                      alpha=0.2, color='green')
    axes.fill_between(xx, siren_mean - siren_std, siren_mean + siren_std,
                      alpha=0.2, color='blue')
    axes.fill_between(xx, combine_mean - combine_std, combine_mean + combine_std,
                      alpha=0.4, color='red')


    axes.axvline(x=turning_point_freq, color='red', linestyle='--', linewidth=1.5)

    plt.ylabel("Acosutic Contrast [dB]")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(100, max_freq)  # Set x-axis limit to focus on the frequency range of interest
    # plt.title("Average Contrast across {} Microphones (up to {}kHz)".format(len(mic_indices), max_freq//1000))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    # Move legend to the right side
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"contrast_frequency_{num_x}_7000_{start_pos}_{end_pos}_plaster_sprayed.svg", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
