import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import cycle
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================

# *** DEFINITION OF CASES ***
# Dictionary structure: 
# "Label Name": {"3d": "path_to_acceleration.csv", "0d": "path_to_0d_solution.csv"}
CASES = {
    "Baseline": {
        "3d": "/home/andrea/Scrivania/phd/dev/heart-torso/SA/csv-data-rho05/base.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/heart-torso/0D-volumes/circulation_base.csv"
    },
    "IncrElastance": {
        "3d": "/home/andrea/Scrivania/phd/dev/heart-torso/SA/csv-data-rho05/incr-ea.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/heart-torso/0D-volumes/circulation_increase_Ea.csv"
    },
    "DecrElastance": {
        "3d": "/home/andrea/Scrivania/phd/dev/heart-torso/SA/csv-data-rho05/decr-ea.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/heart-torso/0D-volumes/circulation_decrease_Ea.csv"
    },
    "IncrResistance": {
        "3d": "/home/andrea/Scrivania/phd/dev/heart-torso/SA/csv-data-rho05/incr-r.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/heart-torso/0D-volumes/circulation_increase_R.csv"
    },
    "DecrResistance": {
        "3d": "/home/andrea/Scrivania/phd/dev/heart-torso/SA/csv-data-rho05/decr-r.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/heart-torso/0D-volumes/circulation_decrease_R.csv"
    }
    # Add your other 2 cases here...
}

# --- Column Name Searching ---
# The script will look for these column names in order.
# Adjust these lists if your CSVs have different headers.

# 3D FILE COLUMNS
COLS_TIME_3D = ['Time', 'time', 'Actual_Time', 't']
COLS_ACC_NORM = ['avg(normal_a)', 'normal_acceleration', 'acc_normal', 'avg(acceleration_normal)']

# 0D FILE COLUMNS
COLS_TIME_0D = ['Time', 'time', 't']
COLS_VOL_LV  = ['VLV', 'V_LV', 'volume_lv', 'Volume_LV']  # LV Volume
COLS_PRES_LV = ['pLV', 'P_LV', 'pressure_lv', 'Pressure_LV'] # LV Pressure

# --- Units & Scaling ---
# Assuming 0D volume is in mL, Pressure in mmHg, Acceleration in m/s^2
TIME_LIMITS_3D = None # Set tuple (start, end) to crop time, or None to use all
TIME_LIMITS_0D = (39.2, 40.0) # Set tuple (start, end) to crop time, or None to use all

def detect_valve_timings(t_0d, v_0d, t_acc, acc_3d):
    """
    Detects AO and AC timings from 0D Volume and 3D Acceleration.
    """
    # --- 1. Aortic Opening (AO) ---
    
    # A. From 0D Volume (Start of decrease)
    # We look for the point where volume derivative becomes significantly negative
    # or simply the first point after t=0.04 where V[i] < V[i-1]
    dv = np.diff(v_0d)
    # Find start of ejection: first index where dV/dt is negative after IVC (approx > 0.04s)
    # We use a mask to skip the very first few steps
    mask_ao_vol = (t_0d[:-1] > 0.04) & (dv < -0.1) # Threshold to avoid noise
    if np.any(mask_ao_vol):
        idx_ao_0d = np.where(mask_ao_vol)[0][0]
        t_ao_0d = t_0d[idx_ao_0d]
    else:
        t_ao_0d = np.nan

    # B. From Acceleration (Max peak in 0.05 - 0.15s)
    mask_ao_acc = (t_acc >= 0.05) & (t_acc <= 0.15)
    if np.any(mask_ao_acc):
        # Subset of data in the window
        t_window = t_acc[mask_ao_acc]
        a_window = acc_3d[mask_ao_acc]
        # Find index of max value in this window
        idx_local = np.argmax(a_window) 
        t_ao_acc = t_window[idx_local]
    else:
        t_ao_acc = np.nan

    # --- 2. Aortic Closure (AC) ---

    # A. From 0D Volume (End of Ejection = Minimum Volume)
    # We look for the global minimum volume
    idx_min_vol = np.argmin(v_0d)
    t_ac_0d = t_0d[idx_min_vol]

    # B. From Acceleration (Local Min in Diastole)
    # We define diastole start roughly > 0.25s to avoid the AO peak
    mask_ac_acc = (t_acc > 0.25) 
    if np.any(mask_ac_acc):
        t_window = t_acc[mask_ac_acc]
        a_window = acc_3d[mask_ac_acc]
        # Find global minimum in the diastolic phase
        idx_local = np.argmin(a_window)
        t_ac_acc = t_window[idx_local]
    else:
        t_ac_acc = np.nan

    return t_ao_0d, t_ao_acc, t_ac_0d, t_ac_acc

def get_column_data(df, potential_names, file_label, data_type):
    """Helper to find a column from a list of potential names."""
    for col in potential_names:
        if col in df.columns:
            return df[col].values
    print(f"Warning: Could not find '{data_type}' column in {file_label}. Checked: {potential_names}")
    return None

def tile_signal(time_single, data_single, target_duration):
    """
    Repeats a single cycle of data to cover the target duration.
    Assumes time_single represents exactly one period.
    """
    if len(time_single) < 2:
        return time_single, data_single
        
    # 1. Calculate Period (duration of the single cycle provided)
    # We assume the input is exactly one cycle (e.g. 39.2 to 40.0 -> 0.8s)
    period = time_single[-1] - time_single[0]
    
    # Small adjustment: if the data is discrete, the "true" period might need
    # to account for one time step, but usually max - min is sufficient for plotting.
    # If your cycle cuts are perfect, this works.
    
    # 2. Normalize time to start at 0
    t_norm = time_single - time_single[0]
    
    # 3. Calculate how many repeats are needed
    n_repeats = int(np.ceil(target_duration / period)) + 1
    
    # 4. Create lists to hold appended data
    t_full = []
    data_full = []
    
    for i in range(n_repeats):
        # Shift time for the i-th cycle
        t_shifted = t_norm + (i * period)
        
        # Append
        t_full.append(t_shifted)
        data_full.append(data_single)
        
    # 5. Concatenate
    t_final = np.concatenate(t_full)
    data_final = np.concatenate(data_full)
    
    # 6. Crop to exact target duration to keep plots clean
    mask = t_final <= target_duration
    return t_final[mask], data_final[mask]

processed_data = {}

print(f"--- Processing {len(CASES)} Simulation Cases ---\n")

for label, paths in CASES.items():
    print(f"Processing: {label}")
    
    # -----------------------------
    # 1. Process 3D Acceleration File
    # -----------------------------
    try:
        df_3d = pd.read_csv(paths['3d'])
        
        # Get Time
        t_3d = get_column_data(df_3d, COLS_TIME_3D, label, "3D Time")
        
        # Get Normal Acceleration
        acc_n = get_column_data(df_3d, COLS_ACC_NORM, label, "Normal Acceleration")
        
        if t_3d is None or acc_n is None:
            print(f"  -> Skipping {label} due to missing 3D data.")
            continue
            
    except Exception as e:
        print(f"  -> Error loading 3D file for {label}: {e}")
        continue

    # -----------------------------
    # 2. Process 0D Circulation File
    # -----------------------------
    try:
        df_0d = pd.read_csv(paths['0d'])
        
        # Get Time
        t_0d = get_column_data(df_0d, COLS_TIME_0D, label, "0D Time")
        
        # Get Volume
        vol_lv = get_column_data(df_0d, COLS_VOL_LV, label, "LV Volume")
        
        # Get Pressure
        pres_lv = get_column_data(df_0d, COLS_PRES_LV, label, "LV Pressure")
        
        if t_0d is None or vol_lv is None or pres_lv is None:
            print(f"  -> Skipping {label} due to missing 0D data.")
            continue

    except Exception as e:
        print(f"  -> Error loading 0D file for {label}: {e}")
        continue

    # -----------------------------
    # 3. Filter by Time (Using separate limits)
    # -----------------------------
    
    # A. Filter 3D Data (Target Duration)
    if TIME_LIMITS_3D:
        mask_3d = (t_3d >= TIME_LIMITS_3D[0]) & (t_3d <= TIME_LIMITS_3D[1])
        t_3d = t_3d[mask_3d]
        acc_n = acc_n[mask_3d]
        
        # Determine the maximum time we need to fill with 0D data
        target_duration = TIME_LIMITS_3D[1]
    else:
        target_duration = t_3d[-1] # Fallback if no limit set

    # B. Filter 0D Data (Template Cycle)
    if TIME_LIMITS_0D:
        mask_0d = (t_0d >= TIME_LIMITS_0D[0]) & (t_0d <= TIME_LIMITS_0D[1])
        t_0d = t_0d[mask_0d]
        vol_lv = vol_lv[mask_0d]
        pres_lv = pres_lv[mask_0d]
        
        # -----------------------------
        # 4. Tile (Repeat) 0D Data
        # -----------------------------
        # This repeats the single 0D cycle to match the 3D duration
        print(f"  -> Tiling 0D cycle (Window: {TIME_LIMITS_0D}) to match {target_duration}s...")
        
        # Tile Volume
        t_0d_tiled, vol_tiled = tile_signal(t_0d, vol_lv, target_duration)
        
        # Tile Pressure (using same time basis)
        _, pres_tiled = tile_signal(t_0d, pres_lv, target_duration)
        
        # Update variables to use the tiled versions
        t_0d = t_0d_tiled
        vol_lv = vol_tiled
        pres_lv = pres_tiled

    # --- PHYSIOLOGICAL METRICS (New) ---
    t_ao_0d, t_ao_acc, t_ac_0d, t_ac_acc = detect_valve_timings(t_0d_curr, v_0d_curr, t_test, data['acc_n'])
    
    # Calculate delay between physical event (0D) and signal peak (Acc)
    # Positive diff means Acceleration peak happens AFTER volume change (expected)
    diff_ao = t_ao_acc - t_ao_0d

    # Store for plotting
    metrics_list.append({
        'Label': label,
        'RMSE': rmse,
        'Correlation': corr,
        'AO_Time_0D': t_ao_0d,
        'AO_Time_Acc': t_ao_acc,
        'AC_Time_0D': t_ac_0d,
        'AC_Time_Acc': t_ac_acc
    })

    print(f"{label:<20} | {rmse:<8.4f} | {corr:<8.4f} | {t_ao_0d:<8.3f} | {t_ao_acc:<8.3f} | {diff_ao:<8.3f} | {t_ac_0d:<8.3f} | {t_ac_acc:<8.3f}")

if not processed_data:
    print("No data loaded. Exiting.")
    sys.exit()

# ==========================================
# --- 3. PLOTTING ---
# ==========================================
colors = cycle(plt.cm.tab10.colors) 
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

# 1. Define Cycle Info
CYCLE_DURATION = 0.8  # seconds
MAX_TIME = 2.5        # Limit for the lines (or use your own variable)

# 2. Plot Data (Same as before)
for label, data in processed_data.items():
    c = next(colors)
    ax1.plot(data['t_3d'], data['acc_n'], label=label, color=c, linewidth=1.5)
    ax2.plot(data['t_0d'], data['vol_lv'], label=label, color=c, linestyle='--')
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
            ax.text(t_cycle, ax.get_ylim()[1], f' End Heartbeat', 
                    rotation=90, verticalalignment='top', fontsize=10, color='black')

    # Standard styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.autoscale(enable=True, axis='x', tight=True) # each plot has its range values

# Styling
ax1.set_ylabel(r'Norm. Accel [$m/s^2$]')
ax1.set_title('Sensitivity Analysis')
ax1.legend(loc='upper right', fontsize='small', framealpha=0.9)

ax2.set_ylabel(r'LV Volume [mL]')

ax3.set_ylabel(r'LV Pressure [mmHg]')
ax3.set_xlabel('Time [s]')  # You might want this label on all axes now

plt.tight_layout()

# --- PLOT 2: PV LOOPS ---
fig2, ax_pv = plt.subplots(figsize=(8, 8))
# Reset colors for consistency between figures
colors = cycle(plt.cm.tab10.colors)

for label, data in processed_data.items():
    c = next(colors)
    # X=Volume, Y=Pressure
    ax_pv.plot(data['vol_lv'], data['pres_lv'], label=label, color=c, linewidth=2)

ax_pv.set_title('Left Ventricle PV Loops')
ax_pv.set_xlabel(r'Volume [mL]')
ax_pv.set_ylabel(r'Pressure [mmHg]')
ax_pv.grid(True, linestyle=':', alpha=0.6)
ax_pv.legend()

plt.tight_layout()
plt.show()


# ==========================================
# --- 4. QUANTITATIVE ANALYSIS ---
# ==========================================

BASELINE_LABEL = "Baseline"  # MUST match one of your keys in CASES

if BASELINE_LABEL not in processed_data:
    print(f"Error: Baseline label '{BASELINE_LABEL}' not found in loaded cases.")
    sys.exit()

# Get Baseline Data (interpolated to common time if needed, but assuming same dt here)
# Note: If time steps differ, you must interpolate A_test to A_base time points.
t_base = processed_data[BASELINE_LABEL]['t_3d']
a_base = processed_data[BASELINE_LABEL]['acc_n']

metrics_list = []

print(f"\n--- Quantitative Comparison vs {BASELINE_LABEL} ---")
print(f"{'Case':<20} | {'RMSE':<10} | {'Corr(r)':<10} | {'Rel.Err(%)':<12} | {'Max Amp Diff':<12}")
print("-" * 75)

for label, data in processed_data.items():
    if label == BASELINE_LABEL:
        continue

    t_test = data['t_3d']
    a_test = data['acc_n']

    # 1. Check lengths (Must be equal for element-wise operations)
    if len(a_base) != len(a_test):
        # Simple interpolation to match Baseline length
        a_test = np.interp(t_base, t_test, a_test)

    # --- METRIC 1: RMSE ---
    rmse = np.sqrt(np.mean((a_test - a_base)**2))

    # --- METRIC 2: Pearson Correlation ---
    corr, _ = pearsonr(a_base, a_test)

    # --- METRIC 3: Relative L2 Error (%) ---
    norm_diff = np.linalg.norm(a_test - a_base)
    norm_base = np.linalg.norm(a_base)
    rel_error = (norm_diff / norm_base) * 100

    # --- METRIC 4: Peak-to-Peak Amplitude Difference ---
    range_test = np.ptp(a_test) # max - min
    range_base = np.ptp(a_base)
    amp_diff = range_test - range_base

    # Aortic Valve opening/closing

    # Store for plotting
    metrics_list.append({
        'Label': label,
        'RMSE': rmse,
        'Correlation': corr,
        'Rel_Error': rel_error,
        'Amp_Diff': amp_diff
    })

    print(f"{label:<20} | {rmse:<10.4f} | {corr:<10.4f} | {rel_error:<12.2f} | {amp_diff:<12.4f}")

# ==========================================
# --- 5. PLOT METRICS (Bar Chart) ---
# ==========================================
if metrics_list:
    df_metrics = pd.DataFrame(metrics_list)
    
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot RMSE
    axes[0].bar(df_metrics['Label'], df_metrics['RMSE'], color='salmon')
    axes[0].set_title('RMSE (Lower is better)')
    axes[0].set_ylabel('Error [m/s^2]')
    axes[0].tick_params(axis='x', rotation=15)

    # Plot Correlation
    axes[1].bar(df_metrics['Label'], df_metrics['Correlation'], color='skyblue')
    axes[1].set_title('Correlation (Closer to 1 is better)')
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis='x', rotation=15)

    # Plot Amplitude Diff
    axes[2].bar(df_metrics['Label'], df_metrics['Amp_Diff'], color='lightgreen')
    axes[2].set_title('Peak-to-Peak Diff (vs Baseline)')
    axes[2].set_ylabel('Difference [m/s^2]')
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.show()