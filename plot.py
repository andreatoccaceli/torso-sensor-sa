import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import savgol_filter 
from itertools import cycle

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================

# *** ADD YOUR FILES HERE ***
DISPLACEMENT_FILES = {
    #"DirichletTorsoBase-NeumannAll": '/home/andrea/Scrivania/phd/dev/heart-torso/torso_scg_neumann_all_homdir_torsobase.csv',
    #"Robin-NeumannBase": '/home/andrea/Scrivania/phd/dev/heart-torso/torso_scg_neumann_base_robin_all.csv',
    #"Robin-NeumannBase-DirichletColumn": '/home/andrea/Scrivania/phd/dev/heart-torso/torso_scg_neumann_base_robin_all_hom_dirichlet.csv',
    #"5em3": "/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/5em3.csv",
    #"1em3": "/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/1em3.csv",
    #"5em4": "/home/andrea/Scrivania/phd/simulations/heart-torso/conv_torso/5em4.csv"
    "Base-rho07" : "/home/andrea/Scrivania/phd/dev/heart-torso/SA/csv-data/base.csv",
    "Savgol-rho05" : "/home/andrea/Scrivania/phd/simulations/heart-torso/savgol.csv"
}

# --- Standard Time Config ---
# We generally assume Time is in column 1 (index 1) or named 'Time'.
# The script will try to find 'Time' or 'time', otherwise fallback to this index.
DEFAULT_TIME_IDX = 1  

# --- Volume Data Config ---
FILE_VOLUME = "/home/andrea/Scrivania/phd/dev/half-sphere-benchmark/circulation.csv" 
VOL_TIME_START = 39.2  
VOL_TIME_END = 40.0   

# --- Physics / Unit Constants ---
CONST_3_OVER_2PI = 3.0 / (2.0 * np.pi)
ML_TO_M3 = 1e-6 

# rescaled factor should be 1e-3 to move from mm to m cause in the sim are rescaled as mm but then 
# when visualizing should be rescaled. If acceleration is read then it isn't necessary to rescale cause was already
# computed with rescaled value in the sim
RESCALED_FACTOR = 1
RESCALED_FACTOR_A = 1
RESCALED_FACTOR_V = 1

# to compute acceleration and velocity
DELTA_T = 1e-4 

# ==========================================
# --- 2. PROCESS FILES LOOP ---
# ==========================================

processed_dfs = {} 

print(f"--- Processing {len(DISPLACEMENT_FILES)} displacement files ---")

for label, file_path in DISPLACEMENT_FILES.items():
    print(f"\n" + "="*60)
    print(f"PROCESSING: {label}")
    print(f"FILE: {file_path}")
    print("="*60)
    
    # --- LOAD CSV ---
    try:
        df_disp = pd.read_csv(file_path, header=0)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Skipping.")
        continue

    # --- 2.1 GET TIME ---
    # Try standard names first, else use default index
    time_col_found = False
    for t_name in ['Time', 'time', 'Actual_Time']:
        if t_name in df_disp.columns:
            df_disp['Actual_Time'] = df_disp[t_name]
            time_col_found = True
            print(f"  -> Found time column: '{t_name}'")
            break
    
    if not time_col_found:
        try:
            print(f"  -> Time column by name not found. Using default index {DEFAULT_TIME_IDX}.")
            df_disp['Actual_Time'] = df_disp.iloc[:, DEFAULT_TIME_IDX]
        except IndexError:
            print(f"Error: Default Time index {DEFAULT_TIME_IDX} is out of bounds. Skipping file.")
            continue

    # --- 2.2 GET DISPLACEMENT ---
    print(f"  -> Looking for Displacement data...")
    
    # Standard Names
    col_d0 = "avg(d (0))"
    col_d1 = "avg(d (1))"
    col_d2 = "avg(d (2))"
    col_dm = "avg(d (Magnitude))"
    col_dn = "avg(normal_d)"
    
    d_x, d_y, d_z, d_n = None, None, None, None

    # Check if standard columns exist
    if col_d0 in df_disp.columns and col_d1 in df_disp.columns and col_d2 in df_disp.columns:
        print(f"     Found standard columns: {col_d0}, {col_d1}, {col_d2}")
        # Apply Rescaling
        d_x = df_disp[col_d0] * RESCALED_FACTOR
        d_y = df_disp[col_d1] * RESCALED_FACTOR
        d_z = df_disp[col_d2] * RESCALED_FACTOR

        if col_dn in df_disp.columns:
            d_n = df_disp[col_dn] * RESCALED_FACTOR
        else:
            print(f"Normal displacement not found, set t 0")
            d_n = None
        
        # Magnitude
        if col_dm in df_disp.columns:
            df_disp['displacement_magnitude'] = df_disp[col_dm] * RESCALED_FACTOR
        else:
            df_disp['displacement_magnitude'] = np.sqrt(d_x**2 + d_y**2 + d_z**2)
            
    else:
        # Fallback: Ask user
        print(f"     Warning: Standard displacement columns ('{col_d0}'...) NOT found.")
        print("     Please enter column INDICES manually (integer, 0-based).")
        try:
            idx_x = int(input(f"     Index for Disp X: "))
            idx_y = int(input(f"     Index for Disp Y: "))
            idx_z = int(input(f"     Index for Disp Z: "))
            
            # Apply Rescaling
            d_x = df_disp.iloc[:, idx_x] * RESCALED_FACTOR
            d_y = df_disp.iloc[:, idx_y] * RESCALED_FACTOR
            d_z = df_disp.iloc[:, idx_z] * RESCALED_FACTOR
            d_n = None
            df_disp['displacement_magnitude'] = np.sqrt(d_x**2 + d_y**2 + d_z**2)
            
        except (ValueError, IndexError) as e:
            print(f"Error: Invalid input or index out of bounds ({e}). Skipping file.")
            continue

    # Save to df (in meters)
    df_disp['disp_x_m'] = d_x
    df_disp['disp_y_m'] = d_y
    df_disp['disp_z_m'] = d_z
    df_disp['disp_n_m'] = d_n


    # --- 2.3 GET/COMPUTE VELOCITY ---
    while True:
        accel_choice = input(f"  -> Velocity source? (c=COMPUTE, r=READ from csv): ").strip().lower()
        if accel_choice in ['c', 'r']: break
        print("     Invalid input. Enter 'c' or 'r'.")

    v_x, v_y, v_z = np.zeros(len(d_x)), np.zeros(len(d_y)), np.zeros(len(d_z))

    if accel_choice == 'r':
        # --- OPTION A: READ ---
        print(f"     Attempting to read velocity columns...")
        col_v0 = "avg(velocity (0))"
        col_v1 = "avg(velocity (1))"
        col_v2 = "avg(velocity (2))"
        
        if col_v0 in df_disp.columns and col_v1 in df_disp.columns and col_v2 in df_disp.columns:
            print(f"     Found standard columns: {col_v0}, {col_v1}, {col_v2}")
            v_x = df_disp[col_v0].values * RESCALED_FACTOR_V
            v_y = df_disp[col_v1].values * RESCALED_FACTOR_V
            v_z = df_disp[col_v2].values * RESCALED_FACTOR_V
        else:
            print(f"     Warning: Standard velocity columns NOT found.")
            try:
                idx_ax = int(input(f"     Index for Accel X: "))
                idx_ay = int(input(f"     Index for Accel Y: "))
                idx_az = int(input(f"     Index for Accel Z: "))
                v_x = df_disp.iloc[:, idx_ax].values
                v_y = df_disp.iloc[:, idx_ay].values
                v_z = df_disp.iloc[:, idx_az].values
            except (ValueError, IndexError) as e:
                print(f"Error reading velocity columns: {e}. Skipping file.")
                continue

    else:
        # --- OPTION B: COMPUTE ---
        if len(d_x) < 3:
            print(f"Warning: Not enough points to compute velocity. Skipping.")
            continue
        
        print(f"     Computing velocity (Finite Differences)...")
        dt = DELTA_T
        
        # First 2 points (Forward/Central)
        for k in [0, 1]:
            v_x[k] = (d_x[2] - d_x[0]) / dt
            v_y[k] = (d_y[2] - d_y[0]) / dt
            v_z[k] = (d_z[2] - d_z[0]) / dt
        
        # Rest (Backward)
        for i in range(2, len(d_x)):
            v_x[i] = (d_x[i] - d_x[i-1]) / dt
            v_y[i] = (d_y[i] - d_y[i-1]) / dt
            v_z[i] = (d_z[i] - d_z[i-1]) / dt

    # Store
    df_disp['velocity_x'] = v_x
    df_disp['velocity_y'] = v_y
    df_disp['velocity_z'] = v_z
    df_disp['velocity_magnitude'] = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    processed_dfs[label] = df_disp
    print(f"  -> Done with {label}.")

    # --- 2.3 GET/COMPUTE ACCELERATION ---
    while True:
        accel_choice = input(f"  -> Acceleration source? (c=COMPUTE, r=READ from csv)): ").strip().lower()
        if accel_choice in ['c', 'r']: break
        print("     Invalid input. Enter 'c' or 'r'.")

    a_x, a_y, a_z = np.zeros(len(d_x)), np.zeros(len(d_y)), np.zeros(len(d_z))

    if accel_choice == 'r':
        # --- OPTION A: READ ---
        print(f"     Attempting to read acceleration columns...")
        col_a0 = "avg(acceleration (0))"
        col_a1 = "avg(acceleration (1))"
        col_a2 = "avg(acceleration (2))"
        col_an = "avg(normal_a)"
        
        if col_a0 in df_disp.columns and col_a1 in df_disp.columns and col_a2 in df_disp.columns:
            print(f"     Found standard columns: {col_a0}, {col_a1}, {col_a2}, {col_an}")
            a_x = df_disp[col_a0].values * RESCALED_FACTOR_A
            a_y = df_disp[col_a1].values * RESCALED_FACTOR_A
            a_z = df_disp[col_a2].values * RESCALED_FACTOR_A

            accel_choice = input(f"  -> Read normal acceleration? (y or n)): ").strip().lower()
            if accel_choice == 'y':
                a_n = df_disp[col_an].values * RESCALED_FACTOR_A
            else:
                a_n = None
        else:
            print(f"     Warning: Standard acceleration columns NOT found. Normal acceleration cannot be computed")
            try:
                idx_ax = int(input(f"     Index for Accel X: "))
                idx_ay = int(input(f"     Index for Accel Y: "))
                idx_az = int(input(f"     Index for Accel Z: "))
                idx_an = int(input(f"     Index for Accel Normal: "))

                a_x = df_disp.iloc[:, idx_ax].values
                a_y = df_disp.iloc[:, idx_ay].values
                a_z = df_disp.iloc[:, idx_az].values
                a_n = df_disp.iloc[:, idx_an].values
            except (ValueError, IndexError) as e:
                print(f"Error reading acceleration columns: {e}. Skipping file.")
                continue

    else:
        # --- OPTION B: COMPUTE ---
        if len(d_x) < 3:
            print(f"Warning: Not enough points to compute acceleration. Skipping.")
            continue
        
        print(f"     Computing acceleration (Finite Differences)...")
        dt_squared = DELTA_T * DELTA_T
        
        # First 2 points (Forward/Central)
        for k in [0, 1]:
            a_x[k] = (d_x[2] - 2*d_x[1] + d_x[0]) / dt_squared * RESCALED_FACTOR_A
            a_y[k] = (d_y[2] - 2*d_y[1] + d_y[0]) / dt_squared * RESCALED_FACTOR_A
            a_z[k] = (d_z[2] - 2*d_z[1] + d_z[0]) / dt_squared * RESCALED_FACTOR_A
        
        # Rest (Backward)
        for i in range(2, len(d_x)):
            a_x[i] = (d_x[i] - 2*d_x[i-1] + d_x[i-2]) / dt_squared * RESCALED_FACTOR_A
            a_y[i] = (d_y[i] - 2*d_y[i-1] + d_y[i-2]) / dt_squared * RESCALED_FACTOR_A 
            a_z[i] = (d_z[i] - 2*d_z[i-1] + d_z[i-2]) / dt_squared * RESCALED_FACTOR_A

        a_n = None # normal acceleration can just be saved

    # Store
    df_disp['acceleration_x'] = a_x
    df_disp['acceleration_y'] = a_y
    df_disp['acceleration_z'] = a_z
    df_disp['acceleration_normal'] = a_n
    df_disp['acceleration_magnitude'] = np.sqrt(a_x**2 + a_y**2 + a_z**2)

    processed_dfs[label] = df_disp
    print(f"  -> Done with {label}.")

if not processed_dfs:
    print("\nNo files processed. Exiting.")
    sys.exit(1)


# ==========================================
# --- 3. LOAD VOLUME DATA (Global) ---
# ==========================================
print(f"\nLoading volume data form '{FILE_VOLUME}'...")
try:
    df_raw_vol = pd.read_csv(FILE_VOLUME)
    df_filtered_vol = df_raw_vol[
        (df_raw_vol['time'] >= VOL_TIME_START) & (df_raw_vol['time'] <= VOL_TIME_END)
    ].copy()
    
    if df_filtered_vol.empty:
        raise ValueError("Empty volume data in range")
        
    time_points_vol_original = df_filtered_vol['time'].values
    volume_values = df_filtered_vol['VLV'].values 
    
    # Unit correction
    volume_values_m3 = volume_values * ML_TO_M3
    r_t_m = np.cbrt(CONST_3_OVER_2PI * volume_values_m3)

    # extend volume and radius to all time duration of sim data
    # Retrieve simulation time from the first processed file
    if processed_dfs:
        print("Extending vol")
        first_key = list(processed_dfs.keys())[0]
        sim_time = processed_dfs[first_key]['Actual_Time'].values
        target_len = len(sim_time)
        
        # Calculate how many times to repeat the single period to cover the full sim time
        # We use ceil to ensure we cover the end, then slice back.
        n_repeats = int(np.ceil(target_len / len(volume_values)))
        n_repeats = 3

        print(f"N repeats: {n_repeats}")
        
        # Tile (copy-paste) the arrays and slice to exact simulation length
        volume_values = np.tile(volume_values, n_repeats)
        r_t_m = np.tile(r_t_m, n_repeats)
        
        # Update the time reference to match simulation time (for plotting alignment)
        original_dt = np.mean(np.diff(time_points_vol_original))
        total_points = len(volume_values)
        start_time = time_points_vol_original[0]
        time_points_vol_original = (start_time + np.arange(total_points) * original_dt) - VOL_TIME_START

    # 0D Acceleration
    if len(r_t_m) > 2:
        dt_array = np.diff(time_points_vol_original)
        dt = np.mean(dt_array)
        print(dt)
        dt_squared = dt * dt
        a_r = np.zeros(len(r_t_m))
        a_r[0] = (r_t_m[2] - 2*r_t_m[1] + r_t_m[0]) / dt_squared
        a_r[1] = (r_t_m[2] - 2*r_t_m[1] + r_t_m[0]) / dt_squared
        for i in range(2, len(r_t_m)):
            a_r[i] = (r_t_m[i] - 2*r_t_m[i-1] + r_t_m[i-2]) / dt_squared
    else:
        a_r = np.zeros(len(r_t_m))
        
except Exception as e:
    print(f"Warning: Issue with volume data ({e}). Volume plots will be empty.")
    time_points_vol_original = np.array([])
    volume_values = np.array([])
    r_t_m = np.array([]) 
    a_r = np.array([])


# ==========================================
# --- 4. PLOTTING ---
# ==========================================
print("\n--- Plot Configuration ---")

def get_plot_components(data_type_name):
    components = []
    while not components:
        choice_str = input(f"  -> {data_type_name}: Plot which components? (e.g., 'xyznmagn' or 'x', 'n'): ").strip().lower()
        valid_comps = []
        for char in choice_str:
            if char in ['x', 'y', 'z', 'm', 'n'] and char not in valid_comps: 
                if char == 'm' and 'magn' in choice_str: valid_comps.append('magn')
                elif char in ['x','y','z', 'n']: valid_comps.append(char)
        if 'magn' in choice_str and 'magn' not in valid_comps: valid_comps.append('magn')
        
        if valid_comps: return valid_comps
        print(f"--- No valid components found in '{choice_str}'. Try again. ---")

def get_plot_choice(prompt_text):
    plot_options = {'1': '3d_accel', '2': '0d_vol', '3': '0d_accel', '4': '0d_radius', '5': '3d_disp', '6' : '3d_vel'}
    print(f"\n{prompt_text}")
    print("  1: 3D Acceleration (m/s^2)")
    print("  2: 0D Volume (mL)")
    print("  3: 0D Radial Acceleration (m/s^2)")
    print("  4: 0D Radius (m)")
    print("  5: 3D Displacement (m)") 
    print("  6: 3D Velocity (m)") 
    
    while True:
        choice = input("Enter choice (1-5): ").strip()
        if choice in plot_options:
            if choice == '1': return ('3d_accel', get_plot_components("3D Acceleration"))
            elif choice == '5': return ('3d_disp', get_plot_components("3D Displacement"))
            elif choice == '6': return ('3d_vel', get_plot_components("3D Velocity"))
            else: return (plot_options[choice], None)
        print("Invalid choice.")

def plot_on_axis(ax, plot_info, all_dfs, time_vol, volume_data, accel_0d_data, radius_0d_data):
    ax.clear() 
    plot_type, components = plot_info
    
    # Linestyles cycle
    lines = cycle(['-', '--', ':', '-.'])
    lines_color = cycle(['tab:blue', 'tab:green', 'tab:red', 'black'])

    if plot_type == '3d_accel':
        ax.set_ylabel('3D Acceleration (m/s^2)')
        for label, df_curr in all_dfs.items():
            cl = next(lines_color)
            if 'x' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_x'], color=cl, linestyle="-", label=f'{label} - Acceleration X')
            if 'y' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_y'], color=next(lines_color), linestyle="-", label=f'{label} - Acceleration Y')
            if 'z' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_z'], color=next(lines_color), linestyle="-", label=f'{label} - Acceleration Z')
            if 'n' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_normal'], color=cl, linestyle="-", label=f'{label} - Acceleration Normal')
            if 'magn' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_magnitude'], color=next(lines_color), linestyle="-", label=f'{label} - Acceleration Mag')
        ax.set_title(f'3D Acceleration Comparison ')

    elif plot_type == '3d_disp':
        ax.set_ylabel('3D Displacement (m)') 
        for label, df_curr in all_dfs.items():
            ls = next(lines)
            if 'x' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_x_m'], color='tab:blue', linestyle=ls, label=f'{label} - Displacement X')
            if 'y' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_y_m'], color='tab:red', linestyle=ls, label=f'{label} - Displacement Y')
            if 'z' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_z_m'], color='tab:green', linestyle=ls, label=f'{label} - Displacement Z')
            if 'n' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_n_m'], color='tab:green', linestyle=ls, label=f'{label} - Displacement Normal')
            if 'magn' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['displacement_magnitude'], color='black', linestyle=ls, label=f'{label} - Displacement Mag')
        ax.set_title(f'3D Displacement Comparison ')
    
    elif plot_type == '3d_vel':
        ax.set_ylabel('3D Velocity (m/s)') 
        for label, df_curr in all_dfs.items():
            ls = next(lines)
            cl = next(lines_color)
            if 'x' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_x'], color=cl, linestyle="-", label=f'{label} - Velocity X')
            if 'y' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_y'], color=next(lines_color), linestyle="-", label=f'{label} - Velocity Y')
            if 'z' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_z'], color=next(lines_color), linestyle="-", label=f'{label} - Velocity Z')
            if 'magn' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_magnitude'], color=next(lines_color), linestyle="-", label=f'{label} - Velocity Mag')
        ax.set_title(f'3D Velocity Comparison ')

    # --- 0D PLOTS ---
    elif plot_type == '0d_vol':
        ax.set_ylabel('Volume (mL)', color='tab:purple')
        ax.plot(time_vol, volume_data, color='tab:purple', label='0D Volume', linestyle='--')
        ax.tick_params(axis='y', labelcolor='tab:purple')

    elif plot_type == '0d_accel':
        ax.set_ylabel('Radial Accel (m/s^2)', color='tab:orange')
        ax.plot(time_vol, accel_0d_data, color='tab:orange', label='0D Rad Accel', linestyle=':')
        ax.tick_params(axis='y', labelcolor='tab:orange')
    
    elif plot_type == '0d_radius':
        ax.set_ylabel('Radius (m)', color='tab:cyan')
        ax.plot(time_vol, radius_0d_data, color='tab:cyan', label='0D Radius', linestyle='-.')
        ax.tick_params(axis='y', labelcolor='tab:cyan')

    ax.set_xlabel('Time (s)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize='small')


# --- EXECUTE ---
plot_type_top = get_plot_choice("Select data for TOP plot (ax1):")
plot_type_bottom = get_plot_choice("Select data for BOTTOM plot (ax2):")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
plot_on_axis(ax1, plot_type_top, processed_dfs, time_points_vol_original, volume_values, a_r, r_t_m)
plot_on_axis(ax2, plot_type_bottom, processed_dfs, time_points_vol_original, volume_values, a_r, r_t_m)

# always plot 0D LV volume
plot_on_axis(ax3, ('0d_vol', None), processed_dfs, time_points_vol_original, volume_values, a_r, r_t_m)


fig.suptitle('Multi-File Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
