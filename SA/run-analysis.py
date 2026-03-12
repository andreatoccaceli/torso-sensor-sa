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
CASES = {
    # "Baseline": {
    #     "3d": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/base.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_base.csv"
    # },
    "Baseline": {
        "3d": f"/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_smooth.csv"
    },
    # "Baseline-Smooth_03": {
    #     "3d": f"/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho03.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_smooth.csv"
    # },
    "Baseline-Smooth_05_periodic": {
        "3d": f"/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_periodic.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_smooth.csv"
    },
    # "Savgol": {
    #     "3d": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/savgol.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_base.csv"
    # },
    # "IncrElastance": {
    #     "3d": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/incr-ea.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_increase_Ea.csv"
    # },
    # "DecrElastance": {
    #     "3d": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/decr-ea.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_decrease_Ea.csv"
    # },
    # "IncrResistance": {
    #     "3d": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/incr-r.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_increase_R.csv"
    # },
    # "DecrResistance": {
    #     "3d": f"/home/andrea/Scrivania/phd/dev/torso-sensor-sa/SA/{CASE}/decr-r.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_decrease_R.csv"
    # }
}

# Column Names
COLS_TIME_3D = ['Time', 'time', 'Actual_Time', 't']
COLS_ACC_NORM = ['avg(normal_a)', 'normal_acceleration', 'acc_normal', 'avg(acceleration_normal)']
COLS_TIME_0D = ['Time', 'time', 't']
COLS_VOL_LV  = ['VLV_smooth', 'VLV', 'V_LV', 'volume_lv', 'Volume_LV']
COLS_PRES_LV = ['pLV', 'P_LV', 'pressure_lv', 'Pressure_LV']

# Time Limits
TIME_LIMITS_3D = None 
TIME_LIMITS_0D = (39.2, 40.0) 
DT_SIMULATIONS = 1e-3

# processing loop
processed_data = {}

print(f"--- Processing {len(CASES)} Simulation Cases ---\n")

for label, paths in CASES.items():
    print(f"Loading: {label}...")
    
    # 1. Load 3D Data
    try:
        df_3d = pd.read_csv(paths['3d'])
        t_3d = get_column_data(df_3d, COLS_TIME_3D, label, "3D Time")
        acc_n = get_column_data(df_3d, COLS_ACC_NORM, label, "Normal Acceleration")
        if t_3d is None or acc_n is None: continue
    except Exception as e:
        print(f"  Error 3D: {e}"); continue

    # 2. Load 0D Data
    try:
        df_0d = pd.read_csv(paths['0d'])
        t_0d = get_column_data(df_0d, COLS_TIME_0D, label, "0D Time")
        vol_lv = get_column_data(df_0d, COLS_VOL_LV, label, "LV Volume")
        pres_lv = get_column_data(df_0d, COLS_PRES_LV, label, "LV Pressure")
        if t_0d is None or vol_lv is None: continue
    except Exception as e:
        print(f"  Error 0D: {e}"); continue

    # 3. Filter 3D & Time Alignment
    if TIME_LIMITS_3D:
        mask_3d = (t_3d >= TIME_LIMITS_3D[0]) & (t_3d <= TIME_LIMITS_3D[1])
        t_3d = t_3d[mask_3d]
        acc_n = acc_n[mask_3d]
    
    # CRITICAL: Shift 3D time to 0
    t_3d = t_3d - t_3d[0]
    target_duration = t_3d[-1]

    # 4. Filter 0D
    if TIME_LIMITS_0D:
        mask_0d = (t_0d >= TIME_LIMITS_0D[0]) & (t_0d <= TIME_LIMITS_0D[1])
        t_0d = t_0d[mask_0d]
        vol_lv = vol_lv[mask_0d]
        pres_lv = pres_lv[mask_0d]

    # 5. Tile 0D
    t_0d_tiled, vol_tiled = tile_signal(t_0d, vol_lv, target_duration)
    _, pres_tiled = tile_signal(t_0d, pres_lv, target_duration)


    # 6. Store Data
    processed_data[label] = {
        't_3d': t_3d,
        'acc_n': acc_n,
        't_0d': t_0d_tiled,
        'vol_lv': vol_tiled,
        'pres_lv': pres_tiled
    }

if not processed_data:
    print("No data loaded. Exiting.")
    sys.exit()

# plot accelerations
colors = cycle(plt.cm.tab10.colors)
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=False)


# 1. Define Cycle Info
CYCLE_DURATION = 0.8 # seconds
MAX_TIME = 2.5 # Limit for the lines (or use your own variable)

# 2. Plot Data (Same as before)
for label, data in processed_data.items():
    c = next(colors)
    ax1.plot(data['t_3d'], data['acc_n'], label=label, color=c, linewidth=1.5)
    ax2.plot(data['t_0d'], data['vol_lv'], label=label, color=c, linewidth=1.5)
    ax3.plot(data['t_0d'], data['pres_lv'], label=label, color=c, linestyle='-.')

# 3. Add Vertical Cycle Lines
# We create a list of times: [0.8, 1.6, 2.4, ...]
cycle_times = np.arange(CYCLE_DURATION, MAX_TIME, CYCLE_DURATION)

for ax in [ax1, ax2, ax3]:
    # Draw the lines
    for t_cycle in cycle_times:
        ax.axvline(x=t_cycle, color='black', linestyle=':', linewidth=2.0, alpha=0.7)

    # Optional: Add text label only on the top plot
    if ax == ax1:
        for i, t_cycle in enumerate(cycle_times):
            ax.text(t_cycle, ax.get_ylim()[1], f' End Heartbeat',rotation=90, verticalalignment='top', fontsize=10, color='black')

    # Standard styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.autoscale(enable=True, axis='x', tight=True) # each plot has its range values

# Styling
ax1.set_ylabel(r'Norm. Accel [$m/s^2$]')
ax1.set_title('Sensitivity Analysis')
ax1.legend(loc='upper right', fontsize='small', framealpha=0.9)
ax2.set_ylabel(r'LV Volume [mL]')
ax3.set_ylabel(r'LV Pressure [mmHg]')
ax3.set_xlabel('Time [s]') # You might want this label on all axes now

plt.tight_layout()

# quantitative metrics
BASELINE_LABEL = "Baseline"
if BASELINE_LABEL not in processed_data:
    print(f"Error: Baseline '{BASELINE_LABEL}' not found.")
    sys.exit()

t_base = processed_data[BASELINE_LABEL]['t_3d']
a_base = processed_data[BASELINE_LABEL]['acc_n']

metrics_list = []

print(f"\n{'Label':<20} | {'RMSE':<8} | {'Corr':<8} | {'AO_0D':<8} | {'AO_Acc':<8} | {'Diff_AO':<8} | {'AC_0D':<8} | {'AC_Acc':<8}")
print("-" * 110)

for label, data in processed_data.items():
    t_test = data['t_3d']
    a_test = data['acc_n']

    # Physiological Metrics
    t_ao_0d, t_ao_acc, t_ac_0d, t_ac_acc = detect_valve_timings(
        data['t_0d'], data['vol_lv'], t_test, data['acc_n']
    )

    # if label == BASELINE_LABEL:
    #     print(f"{label:<20} | {} | {} | {t_ao_0d:<8.3f} | {t_ao_acc:<8.3f} | {t_ao_acc-t_ao_0d:<8.3f} | {t_ac_0d:<8.3f} | {t_ac_acc:<8.3f}") 
    #     continue
    
    # Interpolate if needed
    if len(a_base) != len(a_test):
        a_test = np.interp(t_base, t_test, a_test)

    # Standard Metrics
    rmse = np.sqrt(np.mean((a_test - a_base)**2))
    corr, _ = pearsonr(a_base, a_test)
    amp_diff = np.ptp(a_test) - np.ptp(a_base)

    metrics_list.append({
        'Label': label,
        'RMSE': rmse,
        'Correlation': corr,
        'Amp_Diff': amp_diff,
        'AO_Diff': t_ao_acc - t_ao_0d,
        'AC_Diff': t_ac_acc - t_ac_0d
    })

    print(f"{label:<20} | {rmse:<8.4f} | {corr:<8.4f} | {t_ao_0d:<8.3f} | {t_ao_acc:<8.3f} | {t_ao_acc-t_ao_0d:<8.3f} | {t_ac_0d:<8.3f} | {t_ac_acc:<8.3f}")

# ==========================================
# --- 6. PLOT METRICS (BAR CHARTS) ---
# ==========================================
if metrics_list:
    df_metrics = pd.DataFrame(metrics_list)
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(df_metrics['Label'], df_metrics['RMSE'], color='salmon')
    axes[0].set_title('RMSE vs Baseline')
    
    axes[1].bar(df_metrics['Label'], df_metrics['Correlation'], color='skyblue')
    axes[1].set_title('Correlation vs Baseline')
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(df_metrics['Label'], df_metrics['Amp_Diff'], color='lightgreen')
    axes[2].set_title('Amplitude Difference')
    axes[2].axhline(0, color='black', linewidth=0.8)

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()