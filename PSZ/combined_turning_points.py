from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import json


# Global parameters
mic_num = 4
ls_num = 6
beta = 0.01
sr = 44100
time_w_len = 400
freq_w_len = time_w_len // 2 + 1
mic_idx = 41
min_freq = 2000
max_freq = 7000
mic_distance = 0.6


def load_tf_with_mic(rir_type, num_x, mic_index):
    """Load transfer function for given rir_type, num_x, and specific mic_index"""
    m = num_x * 2 - 1
    dmic = int(mic_distance / (4 / m))
    # Ensure dmic is always even (multiple of 2)
    dmic = dmic if dmic % 2 == 0 else dmic + 1
    
    rir = np.zeros((m, ls_num, time_w_len))
    if rir_type == "siren":
        for n in range(ls_num):
            rir[:, n, :] = np.load(f"rir/siren_brir_r_400_400time_{num_x}mic_src{n}_plaster_sprayed.npy")
    elif rir_type == "truth":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 1].transpose(1, 2, 0) 
    elif rir_type == "bilinear":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 1].transpose(1, 2, 0) 
        rir = rir[::2, :, :]
        rir = ndimage.zoom(rir, zoom=[m/num_x, 1, 1], order=1)

    tf = np.fft.rfft(rir, axis=-1)
    tf = tf.transpose(2, 1, 0)
    tf_l = tf[:, :, [mic_index, mic_index+dmic]]

    rir = np.zeros((m, ls_num, time_w_len))
    if rir_type == "siren":
        for n in range(ls_num):
            rir[:, n, :] = np.load(f"rir/siren_brir_400_400time_{num_x}mic_src{n}_plaster_sprayed.npy")
    elif rir_type == "truth":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 0].transpose(1, 2, 0)
    elif rir_type == "bilinear":
        rir = scio.loadmat(f'rir/plaster_sprayed/brir_{m}.mat')['brir'][:time_w_len, :, :, 0].transpose(1, 2, 0) 
        rir = rir[::2, :, :]
        rir = ndimage.zoom(rir, zoom=[m/num_x, 1, 1], order=1)

    tf = np.fft.rfft(rir, axis=-1)
    tf = tf.transpose(2, 1, 0)
    tf_r = tf[:, :, [mic_index, mic_index+dmic]]

    estimate_tf = np.array([tf_l[:, :, 0], tf_r[:, :, 0], tf_l[:, :, 1], tf_r[:, :, 1]])
    return estimate_tf


def calculate_filter(estimate_tf, method):
    """Calculate filter using given method"""
    # Convert from (4, 201, 6) to (201, 6, 4) format
    estimate_tf = estimate_tf.transpose(1, 2, 0)
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
            d = np.array(d_a + d_b)  # [1, 1, 0, 0]
            m = h_h @ h + beta * np.identity(h.shape[1])
            q = np.linalg.solve(h_h @ h + beta * np.identity(h.shape[1]), h_h @ d)
            q_matrix[i, :] = q

    return q_matrix


def calculate_contrast_with_mic(q_matrix, num_x, mic_index):
    """Calculate contrast using truth transfer function for specific mic_index"""
    tf = load_tf_with_mic("truth", num_x, mic_index)
    
    # Convert from (4, 201, 6) to (201, 6, 4) format
    tf = tf.transpose(1, 2, 0)
    g_a = tf[:, :, 0:2]  # (201, 6, 2)
    g_b = tf[:, :, 2:4]  # (201, 6, 2)

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


def calculate_average_contrast_full_freq(num_x, method_type):
    """Calculate average contrast across all microphones for full frequency range"""
    method = "PM"
    
    m = num_x * 2 - 1
    dmic = int(0.6 / (4 / m))
    start_pos = int((1.5 - 0.5) / (4.5 - 0.5) * (m - 1))  # 1m position
    start_pos = start_pos if start_pos % 2 == 0 else start_pos + 1
    end_pos = int((3.5 - 0.5) / (4.5 - 0.5) * (m - 1))    # 4m position
    mic_indices = range(start_pos + 1, end_pos - dmic + 1, 2) 
    
    # Initialize storage for all microphone contrasts
    all_contrasts = []
    
    for mic_index in mic_indices:
        # Load transfer functions with current mic_index
        tf = load_tf_with_mic(method_type, num_x, mic_index)
        
        # Calculate filters
        q = calculate_filter(tf, method)
        
        # Calculate contrasts using truth data with same mic_index
        contrast = calculate_contrast_with_mic(q, num_x, mic_index)
        
        if contrast is not None:
            all_contrasts.append(contrast)
    
    if len(all_contrasts) == 0:
        return None
    
    # Convert to numpy arrays and calculate mean
    all_contrasts = np.array(all_contrasts)
    mean_contrast = np.mean(all_contrasts, axis=0)
    
    return mean_contrast


def calculate_average_nmse_full_freq(num_x, method_type):
    """Calculate average NMSE across all microphones for full frequency range"""
    m = num_x * 2 - 1
    dmic = int(0.6 / (4 / m))
    start_pos = int((1.5 - 0.5) / (4.5 - 0.5) * (m - 1))  # 1m position
    start_pos = start_pos if start_pos % 2 == 0 else start_pos + 1
    end_pos = int((3.5 - 0.5) / (4.5 - 0.5) * (m - 1))    # 4m position
    mic_indices = range(start_pos + 1, end_pos - dmic + 1, 2) 
    
    # Initialize storage for all microphone NMSE data
    all_nmse = []
    
    for mic_index in mic_indices:
        # Load transfer functions
        truth_tf = load_tf_with_mic("truth", num_x, mic_index)
        estimate_tf = load_tf_with_mic(method_type, num_x, mic_index)
        
        # Calculate NMSE for each frequency point
        nmse = calculate_nmse_per_frequency(truth_tf, estimate_tf)
        
        if nmse is not None:
            all_nmse.append(nmse)
    
    if len(all_nmse) == 0:
        return None
    
    # Convert to numpy arrays and calculate mean
    all_nmse = np.array(all_nmse)
    mean_nmse = np.mean(all_nmse, axis=0)
    
    return mean_nmse


def find_contrast_turning_point(bilinear_contrast, siren_contrast, consecutive_points=5):
    """Find the turning point where siren contrast consistently exceeds bilinear contrast"""
    freq_resolution = sr / time_w_len
    start_freq = 100  # Hz
    start_bin = int(start_freq / freq_resolution)
    
    diff = siren_contrast - bilinear_contrast
    
    # Find consecutive points where siren > bilinear
    for i in range(start_bin, len(diff) - consecutive_points + 1):
        if np.all(diff[i:i+consecutive_points] > 0):
            turning_point_bin = i
            turning_point_freq = turning_point_bin * freq_resolution
            return turning_point_freq, turning_point_bin
    
    return None, None


def find_nmse_turning_point(bilinear_nmse, siren_nmse, consecutive_points=5):
    """Find the turning point where siren NMSE is consistently lower than bilinear NMSE"""
    freq_resolution = sr / time_w_len
    start_freq = 100  # Hz
    start_bin = int(start_freq / freq_resolution)
    
    diff = bilinear_nmse - siren_nmse  # Positive when siren is better (lower NMSE)
    
    # Find consecutive points where siren < bilinear (diff > 0)
    for i in range(start_bin, len(diff) - consecutive_points + 1):
        if np.all(diff[i:i+consecutive_points] > 0):
            turning_point_bin = i
            turning_point_freq = turning_point_bin * freq_resolution
            return turning_point_freq, turning_point_bin
    
    return None, None


def save_cached_data(data, filename):
    """Save computed data to cache file"""
    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            json_data[key] = value.tolist()
        elif isinstance(value, dict):
            json_data[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    json_data[key][k] = v.tolist()
                else:
                    json_data[key][k] = v
        else:
            json_data[key] = value
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {filename}")


def load_cached_data(filename):
    """Load cached data from file"""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        json_data = json.load(f)
    
    # Convert lists back to numpy arrays
    data = {}
    for key, value in json_data.items():
        if isinstance(value, list):
            data[key] = np.array(value)
        elif isinstance(value, dict):
            data[key] = {}
            for k, v in value.items():
                # Convert string keys back to integers for numeric dictionaries
                if k.isdigit():
                    dict_key = int(k)
                else:
                    dict_key = k
                
                if isinstance(v, list):
                    data[key][dict_key] = np.array(v)
                else:
                    data[key][dict_key] = v
        else:
            data[key] = value
    
    return data


def main():
    """Main function to analyze both contrast and NMSE turning points"""
    num_x_values = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111]
    consecutive_points = 5  # Number of consecutive points required for turning point
    cache_filename = f'combined_turning_points_cache_consecutive{consecutive_points}_plaster_sprayed.json'
    
    # Try to load cached data
    cached_data = load_cached_data(cache_filename)
    
    if cached_data is not None:
        print("Found cached data, loading from file...")
        contrast_turning_points = cached_data['contrast_turning_points']
        nmse_turning_points = cached_data['nmse_turning_points']
        print("Cached data loaded successfully!")
    else:
        print("No cached data found, computing from scratch...")
        
        # Storage for results
        contrast_turning_points = {}
        nmse_turning_points = {}
        
        print("Analyzing turning points for different num_x values...")
        print(f"Using consecutive points algorithm: {consecutive_points} consecutive points required for turning point")
        
        for num_x in num_x_values:
            print(f"\nProcessing num_x = {num_x}...")
            
            # Calculate contrast turning points
            print("  Calculating contrast data...")
            bilinear_contrast = calculate_average_contrast_full_freq(num_x, "bilinear")
            siren_contrast = calculate_average_contrast_full_freq(num_x, "siren")
            
            if bilinear_contrast is not None and siren_contrast is not None:
                contrast_tp_freq, _ = find_contrast_turning_point(bilinear_contrast, siren_contrast, consecutive_points)
                contrast_turning_points[num_x] = contrast_tp_freq
                
                if contrast_tp_freq is not None:
                    print(f"  Contrast turning point: {contrast_tp_freq:.1f} Hz")
                else:
                    print(f"  No contrast turning point found")
            else:
                contrast_turning_points[num_x] = None
                print(f"  Failed to calculate contrast data")
            
            # Calculate NMSE turning points
            print("  Calculating NMSE data...")
            bilinear_nmse = calculate_average_nmse_full_freq(num_x, "bilinear")
            siren_nmse = calculate_average_nmse_full_freq(num_x, "siren")
            
            if bilinear_nmse is not None and siren_nmse is not None:
                nmse_tp_freq, _ = find_nmse_turning_point(bilinear_nmse, siren_nmse, consecutive_points)
                nmse_turning_points[num_x] = nmse_tp_freq
                
                if nmse_tp_freq is not None:
                    print(f"  NMSE turning point: {nmse_tp_freq:.1f} Hz")
                else:
                    print(f"  No NMSE turning point found")
            else:
                nmse_turning_points[num_x] = None
                print(f"  Failed to calculate NMSE data")
        
        # Save computed data to cache
        cache_data = {
            'contrast_turning_points': contrast_turning_points,
            'nmse_turning_points': nmse_turning_points
        }
        save_cached_data(cache_data, cache_filename)
    
    # Prepare data for plotting
    valid_contrast_points = []
    valid_contrast_num_x = []
    valid_nmse_points = []
    valid_nmse_num_x = []
    
    print(f"\nSummary of turning points:")
    print("num_x\tContrast TP\tNMSE TP")
    print("-" * 35)
    
    for num_x in num_x_values:
        contrast_tp = contrast_turning_points.get(num_x, None)
        nmse_tp = nmse_turning_points.get(num_x, None)
        
        contrast_str = f"{contrast_tp:.0f}Hz" if contrast_tp is not None else "N/A"
        nmse_str = f"{nmse_tp:.0f}Hz" if nmse_tp is not None else "N/A"
        
        print(f"{num_x}\t{contrast_str}\t\t{nmse_str}")
        
        if contrast_tp is not None:
            valid_contrast_points.append(contrast_tp)
            valid_contrast_num_x.append(num_x)
        
        if nmse_tp is not None:
            valid_nmse_points.append(nmse_tp)
            valid_nmse_num_x.append(num_x)
    
    # Set font sizes for publication
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'savefig.format': 'svg'
    })
    
    # Create the combined plot
    fig, ax = plt.subplots(1, 1, figsize=(3.15, 2.5))
    
    # Plot contrast turning points
    if valid_contrast_points:
        ax.plot(valid_contrast_num_x, valid_contrast_points, 'go-', linewidth=3, markersize=8, label='Contrast')
    
    # Plot NMSE turning points  
    if valid_nmse_points:
        ax.plot(valid_nmse_num_x, valid_nmse_points, 'bo-', linewidth=3, markersize=8, label='NMSE')
    
    ax.set_xlabel('$Z$')
    ax.set_ylabel('$f_t$ [Hz]')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 125)
    ax.axhline(y=2000, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'combined_turning_points_consecutive{consecutive_points}_vs_num_mics_plaster_sprayed.svg', bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'combined_turning_points_consecutive{consecutive_points}_vs_num_mics_plaster_sprayed.svg'")
    print(f"Used consecutive_points = {consecutive_points} for turning point detection")


if __name__ == "__main__":
    main()