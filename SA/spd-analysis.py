import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import cycle
from scipy.stats import pearsonr

# lib
from helpers import *

# cases
CASE = "csv-data-rho05"

# torso sa
CASES = {
    #"Baseline": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/base.csv",
    #"Savgol": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/savgol.csv",
    #"IncrElastance": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/incr-ea.csv",
    #"DecrElastance": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/decr-ea.csv",
    #"IncrResistance": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/incr-r.csv",
    #"DecrResistance": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/decr-r.csv",
    "5em4" : "/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/5em4.csv",
    "2em4" : "/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/2em4.csv",
    "1em4" : "/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/1em4.csv"
}

# torso conv
# CASES = {
#     "5em4": f"/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/5em4.csv",
#     "2em4": f"/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/2em4.csv",
#     "1em4": f"/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/1em4.csv"
# }

# Column Names
COLS_TIME_3D = ['Time', 'time', 'Actual_Time', 't']
COLS_ACC_NORM = ['avg(normal_a)', 'normal_acceleration', 'acc_normal', 'avg(acceleration_normal)']
COLS_TIME_0D = ['Time', 'time', 't']
COLS_VOL_LV  = ['VLV', 'V_LV', 'volume_lv', 'Volume_LV']
COLS_PRES_LV = ['pLV', 'P_LV', 'pressure_lv', 'Pressure_LV']

# Time Limits
TIME_LIMITS_3D = [0.0, 0.8] 

# processing loop
processed_data = {}

print(f"--- Processing {len(CASES)} Simulation Cases ---\n")

for label, path in CASES.items():
    print(f"Loading: {label}...")
    
    # 1. Load 3D Data
    try:
        df_3d = pd.read_csv(path)
        t_3d = get_column_data(df_3d, COLS_TIME_3D, label, "3D Time")
        acc_n = get_column_data(df_3d, COLS_ACC_NORM, label, "Normal Acceleration")
        if t_3d is None or acc_n is None: continue
    except Exception as e:
        print(f"  Error 3D: {e}"); continue

    # 3. Filter 3D & Time Alignment
    if TIME_LIMITS_3D:
        mask_3d = (t_3d >= TIME_LIMITS_3D[0]) & (t_3d <= TIME_LIMITS_3D[1])
        t_3d = t_3d[mask_3d]
        acc_n = acc_n[mask_3d]
    
    # CRITICAL: Shift 3D time to 0
    t_3d = t_3d - t_3d[0]
    target_duration = t_3d[-1]

    # 6. Store Data
    processed_data[label] = {}

    # frequ analysis
    fs_est = 1.0 / np.mean(np.diff(t_3d))
    
    f, p = compute_psd(t_3d, acc_n, fs=fs_est)
    
    # Store
    processed_data[label]['freq'] = f
    processed_data[label]['psd'] = p

if not processed_data:
    print("No data loaded. Exiting.")
    sys.exit()

# frequ plot
fig4, ax_psd = plt.subplots(figsize=(10, 6))
colors = cycle(plt.cm.tab10.colors) # Reset colors

for label, data in processed_data.items():
    c = next(colors)
    # Plot PSD (Log Scale on Y is standard for dB-like comparison)
    # or Linear scale if you want to see raw energy dominance
    ax_psd.semilogy(data['freq'], data['psd'], label=label, color=c, linewidth=1.5)

ax_psd.set_title('Power Spectral Density (SCG Acceleration)')
ax_psd.set_xlabel('Frequency [Hz]')
ax_psd.set_ylabel('Power/Frequency [(m/s^2)^2/Hz]')
ax_psd.set_xlim(0, 50)  # Focus on physiological range (0-50Hz usually contains 95% of energy)
ax_psd.grid(True, which="both", linestyle='--', alpha=0.6)
ax_psd.legend()

plt.tight_layout()
plt.show()

