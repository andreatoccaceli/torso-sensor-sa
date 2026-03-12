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
    # "rho05": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_base.csv",
    #     "VOL_TIME_START" : 39.2, 
    #     "VOL_TIME_END" : 40.0,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-E5e4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_reduced_E5e4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-cm": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_reduced_cm.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-cm-all": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_reduced_cm_all.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-cm-all-new-code": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_reduced_cm_all_new_code.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-bspline-all-new-code": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_reduced_bspline_all_new_code.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-no-bc-vertebral-column-rho05": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_bc_vertebral_column.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    "rho05-cbspline-no-bc-vertebral-column-no-interp": {
        "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
        "VOL_TIME_START" : 78.4, 
        "VOL_TIME_END" :  79.2,
        "VOL_0D_NAME" : "VLV"
    },
    "rho05-cbspline-no-bc-vertebral-column-no-interp-periodicity": {
        "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_periodicity.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
        "VOL_TIME_START" : 78.4, 
        "VOL_TIME_END" :  79.2,
        "VOL_0D_NAME" : "VLV"
    },
    #     "rho05-cbspline-no-bc-vertebral-column-no-interp-new": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_new.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "mf2": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/mf2_cubic_no_vertebral_bc.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "1em3": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/new-conv/1em3.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "5em4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/new-conv/5em4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "2em4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/new-conv/2em4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # }
    # "rho05-cbspline-reduced-bspline-out5-new-code": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_reduced_bspline_out5_new_code.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-linear-all-new-code": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_reduced_linear_all_new_code.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-reduced-cm-rho-04": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_cm_rho_04.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-optim-more": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_optim_more_long.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-5em4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_5em4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-optim-more-5em4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_optim_more_5em4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-non-optim-more-5em4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/rho05_cubic_spline_non_optim_more_5em4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "rho05-cbspline-E-5e4": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/E5e4.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # }
}

# --- Standard Time Config ---
DEFAULT_TIME_IDX = 1  

# --- Global Volume Data Config (Fallback) ---
VOL_TIME_START_GLOBAL = 39.2  
VOL_TIME_END_GLOBAL = 40.0   

# --- Physics / Unit Constants ---
CONST_3_OVER_2PI = 3.0 / (2.0 * np.pi)
ML_TO_M3 = 1e-6 

RESCALED_FACTOR = 1
RESCALED_FACTOR_A = 1
RESCALED_FACTOR_V = 1

DELTA_T = 1e-4

# ==========================================
# --- 2. PROCESS FILES LOOP ---
# ==========================================

processed_dfs = {} 

print(f"--- Processing {len(DISPLACEMENT_FILES)} displacement files ---")

for label, file_path_dict in DISPLACEMENT_FILES.items():
    
    path_3d = file_path_dict["3d"] if isinstance(file_path_dict, dict) else file_path_dict
    
    print(f"\n" + "="*60)
    print(f"PROCESSING: {label}")
    print(f"FILE: {path_3d}")
    print("="*60)
    
    # --- LOAD CSV ---
    try:
        df_disp = pd.read_csv(path_3d, header=0)
    except FileNotFoundError:
        print(f"Error: File not found at '{path_3d}'. Skipping.")
        continue

    # --- 2.1 GET TIME ---
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
    
    col_d0 = "avg(d (0))"
    col_d1 = "avg(d (1))"
    col_d2 = "avg(d (2))"
    col_dm = "avg(d (Magnitude))"
    col_dn = "avg(normal_d)"
    
    d_x, d_y, d_z, d_n = None, None, None, None

    if col_d0 in df_disp.columns and col_d1 in df_disp.columns and col_d2 in df_disp.columns:
        print(f"     Found standard columns: {col_d0}, {col_d1}, {col_d2}")
        d_x = df_disp[col_d0] * RESCALED_FACTOR
        d_y = df_disp[col_d1] * RESCALED_FACTOR
        d_z = df_disp[col_d2] * RESCALED_FACTOR

        if col_dn in df_disp.columns:
            d_n = df_disp[col_dn] * RESCALED_FACTOR
        else:
            print(f"     Normal displacement not found, set to 0")
            d_n = None
        
        if col_dm in df_disp.columns:
            df_disp['displacement_magnitude'] = df_disp[col_dm] * RESCALED_FACTOR
        else:
            df_disp['displacement_magnitude'] = np.sqrt(d_x**2 + d_y**2 + d_z**2)
            
    else:
        print(f"     Warning: Standard displacement columns ('{col_d0}'...) NOT found.")
        print("     Please enter column INDICES manually (integer, 0-based).")
        try:
            idx_x = int(input(f"     Index for Disp X: "))
            idx_y = int(input(f"     Index for Disp Y: "))
            idx_z = int(input(f"     Index for Disp Z: "))
            
            d_x = df_disp.iloc[:, idx_x] * RESCALED_FACTOR
            d_y = df_disp.iloc[:, idx_y] * RESCALED_FACTOR
            d_z = df_disp.iloc[:, idx_z] * RESCALED_FACTOR
            d_n = None
            df_disp['displacement_magnitude'] = np.sqrt(d_x**2 + d_y**2 + d_z**2)
            
        except (ValueError, IndexError) as e:
            print(f"Error: Invalid input or index out of bounds ({e}). Skipping file.")
            continue

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
                idx_ax = int(input(f"     Index for Vel X: "))
                idx_ay = int(input(f"     Index for Vel Y: "))
                idx_az = int(input(f"     Index for Vel Z: "))
                v_x = df_disp.iloc[:, idx_ax].values
                v_y = df_disp.iloc[:, idx_ay].values
                v_z = df_disp.iloc[:, idx_az].values
            except (ValueError, IndexError) as e:
                print(f"Error reading velocity columns: {e}. Skipping file.")
                continue

    else:
        if len(d_x) < 3:
            print(f"Warning: Not enough points to compute velocity. Skipping.")
            continue
        
        print(f"     Computing velocity (Finite Differences)...")
        dt = DELTA_T
        
        for k in [0, 1]:
            v_x[k] = (d_x[2] - d_x[0]) / dt
            v_y[k] = (d_y[2] - d_y[0]) / dt
            v_z[k] = (d_z[2] - d_z[0]) / dt
        
        for i in range(2, len(d_x)):
            v_x[i] = (d_x[i] - d_x[i-1]) / dt
            v_y[i] = (d_y[i] - d_y[i-1]) / dt
            v_z[i] = (d_z[i] - d_z[i-1]) / dt

    df_disp['velocity_x'] = v_x
    df_disp['velocity_y'] = v_y
    df_disp['velocity_z'] = v_z
    df_disp['velocity_magnitude'] = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    # --- 2.4 GET/COMPUTE ACCELERATION ---
    while True:
        accel_choice = input(f"  -> Acceleration source? (c=COMPUTE, r=READ from csv)): ").strip().lower()
        if accel_choice in ['c', 'r']: break
        print("     Invalid input. Enter 'c' or 'r'.")

    a_x, a_y, a_z = np.zeros(len(d_x)), np.zeros(len(d_y)), np.zeros(len(d_z))

    if accel_choice == 'r':
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
            print(f"     Warning: Standard acceleration columns NOT found.")
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
        if len(d_x) < 3:
            print(f"Warning: Not enough points to compute acceleration. Skipping.")
            continue
        
        print(f"     Computing acceleration (Finite Differences)...")
        dt_squared = DELTA_T * DELTA_T
        
        for k in [0, 1]:
            a_x[k] = (d_x[2] - 2*d_x[1] + d_x[0]) / dt_squared * RESCALED_FACTOR_A
            a_y[k] = (d_y[2] - 2*d_y[1] + d_y[0]) / dt_squared * RESCALED_FACTOR_A
            a_z[k] = (d_z[2] - 2*d_z[1] + d_z[0]) / dt_squared * RESCALED_FACTOR_A
        
        for i in range(2, len(d_x)):
            a_x[i] = (d_x[i] - 2*d_x[i-1] + d_x[i-2]) / dt_squared * RESCALED_FACTOR_A
            a_y[i] = (d_y[i] - 2*d_y[i-1] + d_y[i-2]) / dt_squared * RESCALED_FACTOR_A 
            a_z[i] = (d_z[i] - 2*d_z[i-1] + d_z[i-2]) / dt_squared * RESCALED_FACTOR_A

        a_n = None

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
# --- 3. LOAD VOLUME DATA (Per-File) ---
# ==========================================
print("\n--- Loading 0D Volume Data ---")

processed_0d = {}

for label, file_path_dict in DISPLACEMENT_FILES.items():
    if not isinstance(file_path_dict, dict) or "0d" not in file_path_dict:
        print(f"No 0D file specified for '{label}'. Skipping 0D load.")
        continue
        
    path_0d = file_path_dict["0d"]
    
    # Extract specific time windows, fallback to global if missing
    t_start = file_path_dict.get("VOL_TIME_START", VOL_TIME_START_GLOBAL)
    t_end = file_path_dict.get("VOL_TIME_END", VOL_TIME_END_GLOBAL)
    
    print(f"Loading 0D data for '{label}' from: {path_0d} (Time: {t_start} to {t_end})")
    
    try:
        df_raw_vol = pd.read_csv(path_0d)
        df_filtered_vol = df_raw_vol[
            (df_raw_vol['time'] >= t_start) & (df_raw_vol['time'] <= t_end)
        ].copy()
        
        if df_filtered_vol.empty:
            raise ValueError(f"Empty volume data in range {t_start} - {t_end}")
            
        time_points_vol_original = df_filtered_vol['time'].values
        volume_values = df_filtered_vol[file_path_dict['VOL_0D_NAME']].values 
        
        volume_values_m3 = volume_values * ML_TO_M3
        r_t_m = np.cbrt(CONST_3_OVER_2PI * volume_values_m3)

        # Retrieve simulation time strictly for THIS specific 3D file to match its length
        if label in processed_dfs:
            sim_time = processed_dfs[label]['Actual_Time'].values
            target_len = len(sim_time)
            
            n_repeats = int(np.ceil(target_len / len(volume_values)))
            n_repeats = 3 
            
            volume_values = np.tile(volume_values, n_repeats)
            r_t_m = np.tile(r_t_m, n_repeats)
            
            original_dt = np.mean(np.diff(time_points_vol_original))
            total_points = len(volume_values)
            start_time = time_points_vol_original[0]
            # Adjust the time array so it always starts at 0 for plotting comparison
            time_points_vol_original = (start_time + np.arange(total_points) * original_dt) - t_start

        # 0D Acceleration
        a_r = np.zeros(len(r_t_m))
        if len(r_t_m) > 2:
            dt_array = np.diff(time_points_vol_original)
            dt = np.mean(dt_array)
            dt_squared = dt * dt
            
            a_r[0] = (r_t_m[2] - 2*r_t_m[1] + r_t_m[0]) / dt_squared
            a_r[1] = (r_t_m[2] - 2*r_t_m[1] + r_t_m[0]) / dt_squared
            for i in range(2, len(r_t_m)):
                a_r[i] = (r_t_m[i] - 2*r_t_m[i-1] + r_t_m[i-2]) / dt_squared
                
        # Store all arrays into a dictionary mapped to the label
        processed_0d[label] = {
            'time': time_points_vol_original,
            'vol': volume_values,
            'radius': r_t_m,
            'accel': a_r
        }
        
    except Exception as e:
        print(f"Warning: Issue with volume data for {label} ({e}). 0D plots for this label will be skipped.")


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
        choice = input("Enter choice (1-6): ").strip()
        if choice in plot_options:
            if choice == '1': return ('3d_accel', get_plot_components("3D Acceleration"))
            elif choice == '5': return ('3d_disp', get_plot_components("3D Displacement"))
            elif choice == '6': return ('3d_vel', get_plot_components("3D Velocity"))
            else: return (plot_options[choice], None)
        print("Invalid choice.")

def plot_on_axis(ax, plot_info, all_dfs, all_0d_dfs):
    ax.clear() 
    plot_type, components = plot_info
    
    lines = cycle(['-', '--', ':', '-.'])
    lines_color = cycle(['tab:blue', 'tab:green', 'tab:red', 'black', 'tab:orange', 'tab:purple'])

    if plot_type == '3d_accel':
        ax.set_ylabel('3D Acceleration (m/s^2)')
        for label, df_curr in all_dfs.items():
            cl = next(lines_color)
            if 'x' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_x'], color=cl, linestyle="-", label=f'{label} - Accel X')
            if 'y' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_y'], color=next(lines_color), linestyle="-", label=f'{label} - Accel Y')
            if 'z' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_z'], color=next(lines_color), linestyle="-", label=f'{label} - Accel Z')
            if 'n' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_normal'], color=cl, linestyle="-", label=f'{label} - Accel Norm')
            if 'magn' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_magnitude'], color=next(lines_color), linestyle="-", label=f'{label} - Accel Mag')
        ax.set_title('3D Acceleration Comparison')

    elif plot_type == '3d_disp':
        ax.set_ylabel('3D Displacement (m)') 
        for label, df_curr in all_dfs.items():
            ls = next(lines)
            if 'x' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_x_m'], color='tab:blue', linestyle=ls, label=f'{label} - Disp X')
            if 'y' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_y_m'], color='tab:red', linestyle=ls, label=f'{label} - Disp Y')
            if 'z' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_z_m'], color='tab:green', linestyle=ls, label=f'{label} - Disp Z')
            if 'n' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['disp_n_m'], color='tab:orange', linestyle=ls, label=f'{label} - Disp Norm')
            if 'magn' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['displacement_magnitude'], color='black', linestyle=ls, label=f'{label} - Disp Mag')
        ax.set_title('3D Displacement Comparison')
    
    elif plot_type == '3d_vel':
        ax.set_ylabel('3D Velocity (m/s)') 
        for label, df_curr in all_dfs.items():
            cl = next(lines_color)
            if 'x' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_x'], color=cl, linestyle="-", label=f'{label} - Vel X')
            if 'y' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_y'], color=next(lines_color), linestyle="-", label=f'{label} - Vel Y')
            if 'z' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_z'], color=next(lines_color), linestyle="-", label=f'{label} - Vel Z')
            if 'magn' in components:
                ax.plot(df_curr['Actual_Time'], df_curr['velocity_magnitude'], color=next(lines_color), linestyle="-", label=f'{label} - Vel Mag')
        ax.set_title('3D Velocity Comparison')

    # --- 0D PLOTS ---
    elif plot_type == '0d_vol':
        ax.set_ylabel('Volume (mL)')
        for label, data_0d in all_0d_dfs.items():
            ls = next(lines)
            cl = next(lines_color)
            ax.plot(data_0d['time'], data_0d['vol'], color=cl, label=f'{label} - 0D Volume', linestyle=ls)

    elif plot_type == '0d_accel':
        ax.set_ylabel('Radial Accel (m/s^2)')
        for label, data_0d in all_0d_dfs.items():
            ls = next(lines)
            cl = next(lines_color)
            ax.plot(data_0d['time'], data_0d['accel'], color=cl, label=f'{label} - 0D Rad Accel', linestyle=ls)
    
    elif plot_type == '0d_radius':
        ax.set_ylabel('Radius (m)')
        for label, data_0d in all_0d_dfs.items():
            ls = next(lines)
            cl = next(lines_color)
            ax.plot(data_0d['time'], data_0d['radius'], color=cl, label=f'{label} - 0D Radius', linestyle=ls)

    ax.set_xlabel('Time (s)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left', fontsize='small')


# --- EXECUTE ---
plot_type_top = get_plot_choice("Select data for TOP plot (ax1):")
plot_type_bottom = get_plot_choice("Select data for BOTTOM plot (ax2):")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
plot_on_axis(ax1, plot_type_top, processed_dfs, processed_0d)
plot_on_axis(ax2, plot_type_bottom, processed_dfs, processed_0d)

# always plot 0D LV volume
plot_on_axis(ax3, ('0d_vol', None), processed_dfs, processed_0d)

fig.suptitle('Multi-File Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()