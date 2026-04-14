import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
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
    # "Old-CSV": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_new.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "New-CSV": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_new_csv.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "FinalFrame-OldIdx": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_new_csv_final_frame.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "FinalFrame-NewIdx": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_new_csv_fframe_nidx.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "LongHorizon": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/long_horizon.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "LongHorizon-Until": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/long_horizon_until.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "LongHorizon-Until-new": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/scg_signal_normal.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "Sensor-Radius-2cm": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/new_loc_probe_scg_sphere.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "Half-Single-Point-new": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/probe_scg.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    "Below-Apex": {
        "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/below_apex.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
        "VOL_TIME_START" : 78.4, 
        "VOL_TIME_END" :  79.2,
        "VOL_0D_NAME" : "VLV"
    },
    "Scale-factor-07": {
        "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/scale_factor_07.csv",
        "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
        "VOL_TIME_START" : 78.4, 
        "VOL_TIME_END" :  79.2,
        "VOL_0D_NAME" : "VLV"
    },
    # "Old-Wrong-with-pvsm": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/old_wrong_with_filter.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circ_limit_cycle_less_out.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
    # "Using-T-per": {
    #     "3d": "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/csv-data/rho05_cubic_spline_no_interp_periodicity.csv",
    #     "0d": "/home/andrea/Scrivania/phd/dev/torso-sensor-sa/0D-volumes/circulation_cubic_spline_reduced.csv",
    #     "VOL_TIME_START" : 78.4, 
    #     "VOL_TIME_END" :  79.2,
    #     "VOL_0D_NAME" : "VLV"
    # },
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
HEARTBEAT_PERIOD = 0.8 

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
        accel_choice = input(f"  -> Acceleration source? (c=COMPUTE, r=READ from csv): ").strip().lower()
        if accel_choice in ['c', 'r']: break
        print("     Invalid input. Enter 'c' or 'r'.")

    a_x, a_y, a_z = np.zeros(len(d_x)), np.zeros(len(d_y)), np.zeros(len(d_z))

    if accel_choice == 'r':
        print(f"     Attempting to read acceleration columns...")
        col_a0 = "avg(acceleration (0))"
        col_a1 = "avg(acceleration (1))"
        col_a2 = "avg(acceleration (2))"
        col_an = "avg(normal_a)"
        
        if col_a1 in df_disp.columns and col_a2 in df_disp.columns:
            print(f"     Found standard columns: {col_a0}, {col_a1}, {col_a2}")
            a_x = df_disp[col_a0].values * RESCALED_FACTOR_A
            a_y = df_disp[col_a1].values * RESCALED_FACTOR_A
            a_z = df_disp[col_a2].values * RESCALED_FACTOR_A

            read_normal = input(f"  -> Read normal acceleration? (y or n): ").strip().lower()
            if read_normal == 'y':
                if col_an in df_disp.columns:
                    if 'Area' in df_disp.columns:
                        print("     Trovata colonna 'Area': Calcolo accelerazione media...")
                        a_n = (df_disp[col_an].values / df_disp['Area'].values) * RESCALED_FACTOR_A
                    else:
                        print("     Nessuna colonna 'Area' (Modalita' PUNTO).")
                        a_n = df_disp[col_an].values * RESCALED_FACTOR_A
                else:
                    a_n = None
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
# --- 3. LOAD VOLUME DATA (0D) ---
# ==========================================
print("\n--- Loading 0D Volume Data ---")

processed_0d = {}

for label, file_path_dict in DISPLACEMENT_FILES.items():
    if not isinstance(file_path_dict, dict) or "0d" not in file_path_dict:
        continue
        
    path_0d = file_path_dict["0d"]
    t_start = file_path_dict.get("VOL_TIME_START", VOL_TIME_START_GLOBAL)
    t_end = file_path_dict.get("VOL_TIME_END", VOL_TIME_END_GLOBAL)
    
    print(f"Loading 0D data for '{label}' from: {path_0d} (Time: {t_start} to {t_end})")
    
    try:
        df_raw_vol = pd.read_csv(path_0d)
        df_filtered_vol = df_raw_vol[(df_raw_vol['time'] >= t_start) & (df_raw_vol['time'] <= t_end)].copy()
        
        if df_filtered_vol.empty:
            raise ValueError(f"Empty volume data in range {t_start} - {t_end}")
            
        time_points_vol_original = df_filtered_vol['time'].values
        volume_values = df_filtered_vol[file_path_dict['VOL_0D_NAME']].values 
        
        volume_values_m3 = volume_values * ML_TO_M3
        r_t_m = np.cbrt(CONST_3_OVER_2PI * volume_values_m3)

        if label in processed_dfs:
            sim_time = processed_dfs[label]['Actual_Time'].values
            start_time_3d = sim_time[0] 
            original_dt = np.mean(np.diff(time_points_vol_original))
            total_points = len(volume_values)
            
            time_points_vol_original = start_time_3d + np.arange(total_points) * original_dt

        a_r = np.zeros(len(r_t_m))
        if len(r_t_m) > 2:
            dt_array = np.diff(time_points_vol_original)
            dt = np.mean(dt_array)
            dt_squared = dt * dt
            
            a_r[0] = (r_t_m[2] - 2*r_t_m[1] + r_t_m[0]) / dt_squared
            a_r[1] = (r_t_m[2] - 2*r_t_m[1] + r_t_m[0]) / dt_squared
            for i in range(2, len(r_t_m)):
                a_r[i] = (r_t_m[i] - 2*r_t_m[i-1] + r_t_m[i-2]) / dt_squared
                
        processed_0d[label] = {
            'time': time_points_vol_original,
            'vol': volume_values,
            'radius': r_t_m,
            'accel': a_r
        }
        
    except Exception as e:
        print(f"Warning: Issue with volume data for {label} ({e}). Skipping.")

# ==========================================
# --- 3.5 MAGIC PERIODIC EXTENSION ---
# ==========================================
print("\n--- Applying Periodic Extension to Short Signals ---")

global_max_time = 0.0
for df in processed_dfs.values():
    global_max_time = max(global_max_time, df['Actual_Time'].max())
for d0 in processed_0d.values():
    global_max_time = max(global_max_time, d0['time'].max())

print(f"Global Maximum Time found across all files: {global_max_time:.2f}s")

# Estendi i file 3D corti
for label, df in processed_dfs.items():
    local_max = df['Actual_Time'].max()
    if local_max < global_max_time - (HEARTBEAT_PERIOD / 2):
        copies = []
        current_max = local_max
        shift_multiplier = 1
        
        while current_max < global_max_time - 1e-3:
            df_copy = df.copy()
            df_copy['Actual_Time'] += shift_multiplier * HEARTBEAT_PERIOD
            copies.append(df_copy)
            current_max = df_copy['Actual_Time'].max()
            shift_multiplier += 1
            
        if copies:
            processed_dfs[label] = pd.concat([df] + copies, ignore_index=True)
            print(f"  -> [3D] '{label}' extended: {local_max:.2f}s is too short. Appended {len(copies)} periodic copies (shift = {HEARTBEAT_PERIOD}s) to reach {processed_dfs[label]['Actual_Time'].max():.2f}s.")

# Estendi i file 0D corti
for label, d0 in processed_0d.items():
    local_max = d0['time'].max()
    if local_max < global_max_time - (HEARTBEAT_PERIOD / 2):
        copies_t, copies_v, copies_r, copies_a = [], [], [], []
        current_max = local_max
        shift_multiplier = 1
        
        while current_max < global_max_time - 1e-3:
            copies_t.append(d0['time'] + shift_multiplier * HEARTBEAT_PERIOD)
            copies_v.append(d0['vol'])
            copies_r.append(d0['radius'])
            copies_a.append(d0['accel'])
            current_max = copies_t[-1][-1]
            shift_multiplier += 1
            
        if copies_t:
            d0['time'] = np.concatenate([d0['time']] + copies_t)
            d0['vol'] = np.concatenate([d0['vol']] + copies_v)
            d0['radius'] = np.concatenate([d0['radius']] + copies_r)
            d0['accel'] = np.concatenate([d0['accel']] + copies_a)
            print(f"  -> [0D] '{label}' extended: {local_max:.2f}s is too short. Appended {len(copies_t)} periodic copies (shift = {HEARTBEAT_PERIOD}s) to reach {d0['time'].max():.2f}s.")

print("--- Data Processing Complete ---")

# ==========================================
# --- 4. PLOTTING ---
# ==========================================
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

    max_time = 0 
    min_time = float('inf')

    for df in all_dfs.values():
        max_time = max(max_time, df['Actual_Time'].max())
        min_time = min(min_time, df['Actual_Time'].min())
    for d0 in all_0d_dfs.values():
        max_time = max(max_time, d0['time'].max())
        min_time = min(min_time, d0['time'].min())

    if min_time == float('inf'): min_time = 0

    if plot_type == '3d_accel':
        ax.set_ylabel('3D Acceleration (m/s^2)')
        for label, df_curr in all_dfs.items():
            cl = next(lines_color)
            if 'x' in components: ax.plot(df_curr['Actual_Time'], df_curr['acceleration_x'], color=cl, linestyle="-", label=f'{label} - Accel X')
            if 'y' in components: ax.plot(df_curr['Actual_Time'], df_curr['acceleration_y'], color=next(lines_color), linestyle="-", label=f'{label} - Accel Y')
            if 'z' in components: ax.plot(df_curr['Actual_Time'], df_curr['acceleration_z'], color=next(lines_color), linestyle="-", label=f'{label} - Accel Z')
            if 'n' in components and df_curr['acceleration_normal'] is not None: 
                ax.plot(df_curr['Actual_Time'], df_curr['acceleration_normal'], color=cl, linestyle="-", label=f'{label} - Accel Norm')
            if 'magn' in components: ax.plot(df_curr['Actual_Time'], df_curr['acceleration_magnitude'], color=next(lines_color), linestyle="-", label=f'{label} - Accel Mag')
        ax.set_title('3D Acceleration Comparison')

    elif plot_type == '3d_disp':
        ax.set_ylabel('3D Displacement (m)') 
        for label, df_curr in all_dfs.items():
            ls = next(lines)
            if 'x' in components: ax.plot(df_curr['Actual_Time'], df_curr['disp_x_m'], color='tab:blue', linestyle=ls, label=f'{label} - Disp X')
            if 'y' in components: ax.plot(df_curr['Actual_Time'], df_curr['disp_y_m'], color='tab:red', linestyle=ls, label=f'{label} - Disp Y')
            if 'z' in components: ax.plot(df_curr['Actual_Time'], df_curr['disp_z_m'], color='tab:green', linestyle=ls, label=f'{label} - Disp Z')
            if 'n' in components and df_curr['disp_n_m'] is not None: 
                ax.plot(df_curr['Actual_Time'], df_curr['disp_n_m'], color='tab:orange', linestyle=ls, label=f'{label} - Disp Norm')
            if 'magn' in components: ax.plot(df_curr['Actual_Time'], df_curr['displacement_magnitude'], color='black', linestyle=ls, label=f'{label} - Disp Mag')
        ax.set_title('3D Displacement Comparison')
    
    elif plot_type == '3d_vel':
        ax.set_ylabel('3D Velocity (m/s)') 
        for label, df_curr in all_dfs.items():
            cl = next(lines_color)
            if 'x' in components: ax.plot(df_curr['Actual_Time'], df_curr['velocity_x'], color=cl, linestyle="-", label=f'{label} - Vel X')
            if 'y' in components: ax.plot(df_curr['Actual_Time'], df_curr['velocity_y'], color=next(lines_color), linestyle="-", label=f'{label} - Vel Y')
            if 'z' in components: ax.plot(df_curr['Actual_Time'], df_curr['velocity_z'], color=next(lines_color), linestyle="-", label=f'{label} - Vel Z')
            if 'magn' in components: ax.plot(df_curr['Actual_Time'], df_curr['velocity_magnitude'], color=next(lines_color), linestyle="-", label=f'{label} - Vel Mag')
        ax.set_title('3D Velocity Comparison')

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

    t_line = min_time + HEARTBEAT_PERIOD
    first_line = True
    while t_line <= max_time:
        line_label = f'End of Beat ({HEARTBEAT_PERIOD}s)' if first_line else None
        ax.axvline(x=t_line, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label=line_label)
        t_line += HEARTBEAT_PERIOD
        first_line = False

    ax.set_xlabel('Time (s)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left', fontsize='small')


# --- EXECUTE ---
plot_type_top = get_plot_choice("Select data for TOP plot (ax1):")
plot_type_bottom = get_plot_choice("Select data for BOTTOM plot (ax2):")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

plot_on_axis(ax1, plot_type_top, processed_dfs, processed_0d)
plot_on_axis(ax2, plot_type_bottom, processed_dfs, processed_0d)
plot_on_axis(ax3, ('0d_vol', None), processed_dfs, processed_0d)

fig.suptitle('Multi-File Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()