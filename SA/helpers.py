import numpy as np
from scipy.signal import welch


def get_column_data(df, potential_names, file_label, data_type):
    for col in potential_names:
        if col in df.columns:
            return df[col].values
    print(f"Warning: Could not find '{data_type}' column in {file_label}.")
    return None

def tile_signal(time_single, data_single, target_duration):
    """Repeats a single cycle to cover target_duration with C0 closure."""
    if len(time_single) < 2: return time_single, data_single
    
    # --- NEW: Ensure the last point matches the first to remove the vertical jump ---
    data_closed = data_single.copy()
    data_closed[-1] = data_closed[0]
    
    period = time_single[-1] - time_single[0]
    t_norm = time_single - time_single[0]
    n_repeats = int(np.ceil(target_duration / period)) + 2 
    
    t_full, data_full = [], []
    for i in range(n_repeats):
        t_full.append(t_norm + (i * period))
        data_full.append(data_closed) # Use the closed data
        
    t_final = np.concatenate(t_full)
    data_final = np.concatenate(data_full)
    mask = t_final <= target_duration
    return t_final[mask], data_final[mask]
    
def detect_valve_timings(t_0d, v_0d, t_acc, acc_3d):
    """Detects AO and AC timings."""
    # 1. AO from Volume (Start of decrease)
    dv = np.diff(v_0d)
    mask_ao_vol = (t_0d[:-1] > 0.04) & (dv < -0.1) 
    t_ao_0d = t_0d[np.where(mask_ao_vol)[0][0]] if np.any(mask_ao_vol) else np.nan

    # 2. AO from Acc (Max peak in 0.05-0.15s)
    mask_ao_acc = (t_acc >= 0.05) & (t_acc <= 0.15)
    if np.any(mask_ao_acc):
        idx_local = np.argmax(acc_3d[mask_ao_acc])
        t_ao_acc = t_acc[mask_ao_acc][idx_local]
    else:
        t_ao_acc = np.nan

    # 3. AC from Volume (Minimum Volume)
    t_ac_0d = t_0d[np.argmin(v_0d)]

    # 4. AC from Acc (Min in Diastole > 0.25s)
    mask_ac_acc = (t_acc > 0.25)
    if np.any(mask_ac_acc):
        idx_local = np.argmin(acc_3d[mask_ac_acc])
        t_ac_acc = t_acc[mask_ac_acc][idx_local]
    else:
        t_ac_acc = np.nan

    return t_ao_0d, t_ao_acc, t_ac_0d, t_ac_acc


def compute_psd(time, signal, fs=None):
    """
    Computes Power Spectral Density (PSD) using Welch's method.
    If fs (sampling freq) is not provided, it is estimated from time array.
    """
    # Estimate sampling frequency if not given
    if fs is None:
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt
    
    # Compute PSD
    # nperseg defines the window size. 
    # For short signals (0.8s), we need a small window to get any result, 
    # but a larger window gives better frequency resolution.
    # We'll use min(length, 256) or similar dynamic sizing.
    n_per_seg = min(len(signal), 512)
    
    freqs, psd = welch(signal, fs=fs, nperseg=n_per_seg)
    return freqs, psd