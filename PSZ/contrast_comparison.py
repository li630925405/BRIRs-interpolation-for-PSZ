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
            d = np.array(d_a + d_b)  # [1, 0, 0, 0]
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


def calculate_average_contrast_full_freq(num_x, method_type):
    """Calculate average contrast across all microphones for full frequency range"""
    method = "PM"
    
    m = num_x * 2 - 1
    dmic = int(0.6 / (4 / m))
    # dmic = dmic if dmic % 2 == 0 else dmic + 1
    start_pos = int((1.5 - 0.5) / (4.5 - 0.5) * (m - 1))  # 1m position
    start_pos = start_pos if start_pos % 2 == 0 else start_pos + 1
    end_pos = int((3.5 - 0.5) / (4.5 - 0.5) * (m - 1))    # 4m position
    mic_indices = range(start_pos + 1, end_pos - dmic + 1, 2) 
    
    
    # Initialize storage for all microphone contrasts
    all_contrasts = []
    
    print(f"  Processing {len(mic_indices)} microphones for {method_type}...")
    
    for mic_index in mic_indices:
        # Load transfer functions with current mic_index
        tf = load_tf_with_mic(method_type, num_x, mic_index)
        
        # Calculate filters
        q = calculate_filter(tf, method)
        
        # Calculate contrasts using truth data with same mic_index
        contrast = calculate_contrast_with_mic(q, num_x, mic_index)
        
        
        if contrast is not None:
            all_contrasts.append(contrast)
        else:
            print(f"    mic_index={mic_index}: contrast calculation failed!")
    
    if len(all_contrasts) == 0:
        print(f"  Warning: No valid contrast data obtained for {method_type}")
        return None
    
    # Convert to numpy arrays and calculate mean
    all_contrasts = np.array(all_contrasts)
    mean_contrast = np.mean(all_contrasts, axis=0)
    
    return mean_contrast


def find_turning_point(bilinear_contrast, siren_contrast, consecutive_points=5):
    """Find the turning point where siren consistently exceeds bilinear"""
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


def combine_contrasts(bilinear_contrast, siren_contrast, turning_point_bin):
    """Combine bilinear and siren contrasts at the turning point"""
    if turning_point_bin is None:
        # If no turning point, use original 2000Hz boundary
        freq_resolution = sr / time_w_len
        turning_point_bin = int(min_freq / freq_resolution)
    
    combined_contrast = np.copy(bilinear_contrast)
    combined_contrast[turning_point_bin:] = siren_contrast[turning_point_bin:]
    
    return combined_contrast


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
    """Main function to analyze contrast differences for different num_x values"""
    num_x_values = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111]
    consecutive_points = 5  # Number of consecutive points required for turning point
    cache_filename = f'contrast_comparison_cache_consecutive{consecutive_points}_plaster_sprayed.json'
    
    # Try to load cached data
    cached_data = load_cached_data(cache_filename)
    
    if cached_data is not None:
        print("Found cached data, loading from file...")
        bilinear_contrasts = cached_data['bilinear_contrasts']
        siren_contrasts = cached_data['siren_contrasts']
        combined_contrasts_fixed = cached_data['combined_contrasts_fixed']
        combined_contrasts_adaptive = cached_data['combined_contrasts_adaptive']
        turning_points = cached_data['turning_points']
        avg_differences_fixed = cached_data['avg_differences_fixed']
        avg_differences_adaptive = cached_data['avg_differences_adaptive']
        print("Cached data loaded successfully!")
    else:
        print("No cached data found, computing from scratch...")
        # Storage for results
        bilinear_contrasts = {}
        siren_contrasts = {}
        combined_contrasts_fixed = {}
        combined_contrasts_adaptive = {}
        turning_points = {}
        avg_differences_fixed = []
        avg_differences_adaptive = []
        
        print("Analyzing contrast differences for different num_x values...")
        print(f"Using consecutive points algorithm: {consecutive_points} consecutive points required for turning point")
        print("Step 1: Calculate full-frequency contrasts for bilinear and siren methods")
        
        for num_x in num_x_values:
            print(f"\nProcessing num_x = {num_x}...")
            
            # Calculate full-frequency contrasts
            bilinear_contrast = calculate_average_contrast_full_freq(num_x, "bilinear")
            siren_contrast = calculate_average_contrast_full_freq(num_x, "siren")
            
            if bilinear_contrast is not None and siren_contrast is not None:
                bilinear_contrasts[num_x] = bilinear_contrast
                siren_contrasts[num_x] = siren_contrast
                
                # Find turning point
                turning_point_freq, turning_point_bin = find_turning_point(bilinear_contrast, siren_contrast, consecutive_points)
                turning_points[num_x] = turning_point_freq
                
                if turning_point_freq is not None:
                    print(f"  Turning point found at: {turning_point_freq:.1f} Hz")
                else:
                    print(f"  No turning point found (siren never exceeds bilinear)")
                
                # Create combined contrasts
                # Method 1: Fixed boundary at 2000Hz
                freq_resolution = sr / time_w_len
                fixed_boundary_bin = int(min_freq / freq_resolution)
                combined_fixed = combine_contrasts(bilinear_contrast, siren_contrast, fixed_boundary_bin)
                combined_contrasts_fixed[num_x] = combined_fixed
                
                # Method 2: Adaptive boundary at turning point
                combined_adaptive = combine_contrasts(bilinear_contrast, siren_contrast, turning_point_bin)
                combined_contrasts_adaptive[num_x] = combined_adaptive
                
                # Calculate average differences
                diff_fixed = combined_fixed - bilinear_contrast
                diff_adaptive = combined_adaptive - bilinear_contrast
                
                avg_diff_fixed = np.mean(diff_fixed)
                avg_diff_adaptive = np.mean(diff_adaptive)
                
                avg_differences_fixed.append(avg_diff_fixed)
                avg_differences_adaptive.append(avg_diff_adaptive)
                
                print(f"  Fixed boundary (2000Hz) - Average improvement: {avg_diff_fixed:.2f} dB")
                print(f"  Adaptive boundary - Average improvement: {avg_diff_adaptive:.2f} dB")
            else:
                avg_differences_fixed.append(np.nan)
                avg_differences_adaptive.append(np.nan)
                print(f"  Failed to calculate contrasts")
        
        # Save computed data to cache
        cache_data = {
            'bilinear_contrasts': bilinear_contrasts,
            'siren_contrasts': siren_contrasts,
            'combined_contrasts_fixed': combined_contrasts_fixed,
            'combined_contrasts_adaptive': combined_contrasts_adaptive,
            'turning_points': turning_points,
            'avg_differences_fixed': avg_differences_fixed,
            'avg_differences_adaptive': avg_differences_adaptive
        }
        save_cached_data(cache_data, cache_filename)
    
    # Create frequency axis
    freq_resolution = sr / time_w_len
    max_freq_bin = int(max_freq / freq_resolution) + 1
    freq_points = min(max_freq_bin, freq_w_len)
    frequencies = np.arange(freq_points) * freq_resolution
    
    # Set font sizes for publication - 9pt absolute size
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'savefig.format': 'svg'
    })
    
    # Plot 1: Average improvement vs num_x
    valid_indices_fixed = ~np.isnan(avg_differences_fixed)
    valid_indices_adaptive = ~np.isnan(avg_differences_adaptive)
    
    valid_num_x_fixed = np.array(num_x_values)[valid_indices_fixed]
    valid_fixed_diffs = np.array(avg_differences_fixed)[valid_indices_fixed]
    
    valid_num_x_adaptive = np.array(num_x_values)[valid_indices_adaptive]
    valid_adaptive_diffs = np.array(avg_differences_adaptive)[valid_indices_adaptive]
    
    # Create first figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.15, 2.5))
    ax1.plot(valid_num_x_fixed, valid_fixed_diffs, 'ro-', linewidth=3, markersize=8, label='Fixed')
    ax1.plot(valid_num_x_adaptive, valid_adaptive_diffs, 'bo-', linewidth=3, markersize=8, label='Adaptive')
    ax1.set_xlabel('$Z$')
    ax1.set_ylabel('Acoustic Contrast [dB]')
    # ax1.set_title('Combine vs Bilinear: Average Contrast Improvement', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(5, 125)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(f'contrast_improvement_consecutive{consecutive_points}_vs_num_mics_plaster_sprayed.svg', bbox_inches='tight')
    plt.show()
    
    # Plot 2: Turning points vs num_x
    valid_turning_points = []
    valid_turning_num_x = []
    print(f"\nDebugging turning points:")
    for num_x in num_x_values:
        tp = turning_points.get(num_x, "Not found")
        print(f"  num_x={num_x}: turning_point={tp}")
        if num_x in turning_points and turning_points[num_x] is not None:
            valid_turning_points.append(turning_points[num_x])
            valid_turning_num_x.append(num_x)
    
    print(f"Valid turning points found: {len(valid_turning_points)}")
    if valid_turning_points:
        print(f"Valid turning points: {valid_turning_points}")
        print(f"Valid num_x values: {valid_turning_num_x}")
    
    # Create second figure
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.15, 2.5))
    if valid_turning_points:
        ax2.plot(valid_turning_num_x, valid_turning_points, 'go-', linewidth=3, markersize=8)
    ax2.set_xlabel('$Z$')
    ax2.set_ylabel('$f_t$ [Hz]')
    # ax2.set_title('Calculated Turning Points vs Number of Microphones', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 125)
    ax2.axhline(y=2000, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Fixed 2000Hz')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'turning_points_consecutive{consecutive_points}_vs_num_mics_plaster_sprayed.svg', bbox_inches='tight')
    
    # Print summary
    print("\nSummary:")
    print("num_x\tTurning Point\tFixed Boundary\tAdaptive Boundary")
    print("-" * 65)
    for i, num_x in enumerate(num_x_values):
        tp_str = f"{turning_points.get(num_x, np.nan):.0f}Hz" if turning_points.get(num_x) is not None else "N/A"
        fixed_str = f"{avg_differences_fixed[i]:.2f}dB" if not np.isnan(avg_differences_fixed[i]) else "Failed"
        adaptive_str = f"{avg_differences_adaptive[i]:.2f}dB" if not np.isnan(avg_differences_adaptive[i]) else "Failed"
        print(f"{num_x}\t{tp_str}\t\t{fixed_str}\t\t{adaptive_str}")


if __name__ == "__main__":
    main()