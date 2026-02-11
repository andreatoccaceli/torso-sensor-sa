import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_FILE = "0D-volumes/circulation_decrease_R.csv"
TIME_START = 39.2
TIME_END = 40.0

# Target simulation time step (fine grid for evaluating derivatives)
DT_SIM = 1e-4  

# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"--- Reading {CSV_FILE} ---")
df = pd.read_csv(CSV_FILE)

# Filter Time
df = df[(df['time'] >= TIME_START) & (df['time'] <= TIME_END)].copy()
raw_t = df['time'].values
raw_v = df['VLV'].values

# Create a fine time grid for comparison
t_fine = np.arange(raw_t[0], raw_t[-1], DT_SIM)

results = {}

# ==========================================
# 3. METHOD 1: RAW DATA (Linear Interpolation)
# ==========================================
# We use finite differences on the raw data
dt_raw = np.mean(np.diff(raw_t))
acc_raw = np.gradient(np.gradient(raw_v, dt_raw), dt_raw)

results['Raw (Finite Diff)'] = {
    'time': raw_t,
    'vol': raw_v,
    'acc': acc_raw,
    'color': 'red',
    'style': 'o',
    'alpha': 0.3,
    'linewidth': 0
}

# ==========================================
# 4. METHOD 2: CUBIC SPLINE (Interpolation)
# ==========================================
# Forces the curve to hit every single point. C2 continuous.
cs = CubicSpline(raw_t, raw_v)
vol_spline = cs(t_fine)
acc_spline = cs(t_fine, 2)  # 2nd derivative

results['Cubic Spline (Interp)'] = {
    'time': t_fine,
    'vol': vol_spline,
    'acc': acc_spline,
    'color': 'blue',
    'style': '-',
    'alpha': 1.0,
    'linewidth': 1.5
}

# ==========================================
# 5. METHOD 3: SAVITZKY-GOLAY (Light Smoothing)
# ==========================================
# Window = 11. Softens noise but keeps sharp corners mostly intact.
win_light = 11
poly_light = 3
vol_sg_light = savgol_filter(raw_v, window_length=win_light, polyorder=poly_light)

# We fit a spline to the SMOOTHED data to get analytical derivatives
cs_light = CubicSpline(raw_t, vol_sg_light)
acc_sg_light = cs_light(t_fine, 2)

results[f'Sav-Golay (Window {win_light})'] = {
    'time': t_fine,
    'vol': cs_light(t_fine),
    'acc': acc_sg_light,
    'color': 'green',
    'style': '--',
    'alpha': 0.9,
    'linewidth': 2
}

# ==========================================
# 6. METHOD 4: SAVITZKY-GOLAY (Heavy Smoothing)
# ==========================================
# Window = 31. Aggressively rounds off corners.
win_heavy = 7
poly_heavy = 3
vol_sg_heavy = savgol_filter(raw_v, window_length=win_heavy, polyorder=poly_heavy)

cs_heavy = CubicSpline(raw_t, vol_sg_heavy)
acc_sg_heavy = cs_heavy(t_fine, 2)

results[f'Sav-Golay (Window {win_heavy})'] = {
    'time': t_fine,
    'vol': cs_heavy(t_fine),
    'acc': acc_sg_heavy,
    'color': 'purple',
    'style': '-',
    'alpha': 0.9,
    'linewidth': 2.5
}

# ==========================================
# 7. PLOTTING & ANALYSIS
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

print("\n--- MAX ACCELERATION METRICS ---")
print(f"{'Method':<30} | {'Max Acc (mL/s^2)':<20} | {'Reduction'}")
print("-" * 65)

# Get the max acceleration of the Spline method to use as baseline for reduction %
base_max = np.max(np.abs(results['Cubic Spline (Interp)']['acc']))

for name, data in results.items():
    # Plot Volume
    ax1.plot(data['time'], data['vol'], 
             data['style'], color=data['color'], 
             alpha=data['alpha'], linewidth=data.get('linewidth', 1),
             label=name)
    
    # Plot Acceleration
    ax2.plot(data['time'], data['acc'], 
             data['style'], color=data['color'], 
             alpha=data['alpha'], linewidth=data.get('linewidth', 1),
             label=name)
    
    # Calculate Metric
    max_acc = np.max(np.abs(data['acc']))
    reduction = (1 - max_acc / base_max) * 100
    if name == 'Cubic Spline (Interp)': reduction = 0.0
    
    print(f"{name:<30} | {max_acc:.2f}             | -{reduction:.1f}%")

# Styling
ax1.set_title("1. Volume Comparison (Check for shape distortion)", fontsize=14)
ax1.set_ylabel("Volume (mL)")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

ax2.set_title("2. Acceleration Comparison (Check for spike reduction)", fontsize=14)
ax2.set_ylabel("Acceleration ($d^2V/dt^2$)", fontsize=12)
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()