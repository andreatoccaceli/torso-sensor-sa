import pyvista as pv
import os
import numpy as np

# --- CONFIGURATION ---
input_pattern = "/home/andrea/Scrivania/phd/simulations/4CH-EM-coarse/solution_{:06d}.xdmf"
start_idx = 0
end_idx = 102000
target_field_name = "d" 

# Output .vtp (PolyData) for surface files
output_filename = "/home/andrea/Scrivania/phd/dev/heart-torso/displacement_heart_relative.vtp"
output_basename = "dref"
num_digits = 6
step_stride = 200 
# ---------------------

def load_and_merge(filename):
    """Reads file, combines blocks, and merges duplicate points."""
    mesh = pv.read(filename)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    
    # FIX: clean() returns a NEW mesh, it cannot be done inplace.
    # We assign the result back to 'mesh'.
    mesh = mesh.clean() 
    return mesh

def main():
    print(f"Processing... Calculating relative field '{target_field_name}(t) - {target_field_name}(0)' on the SURFACE.")
    
    # ---------------------------------------------------------
    # 1. SETUP REFERENCE GEOMETRY (Step 0)
    # ---------------------------------------------------------
    first_file = input_pattern.format(start_idx)
    print(f"Loading reference mesh (t=0) from {first_file}...")
    
    # Load and CLEAN the full volume mesh at t=0
    vol_mesh_0 = load_and_merge(first_file)
    
    # Extract d(0) - The Initial Displacement
    if target_field_name not in vol_mesh_0.point_data:
        raise ValueError(f"Field '{target_field_name}' not found in {first_file}")

    d_0_vol = vol_mesh_0.point_data[target_field_name].copy()
    
    # --- STEP 1: WARP GEOMETRY ---
    # We move the unloaded nodes by d(0) to get the Reference Configuration
    print("  Warping mesh by d(0) to match Simulation Reference State...")
    
    # FIX: Use 'factor' instead of 'scale_factor'
    vol_mesh_0.warp_by_vector(vectors=target_field_name, factor=1.0, inplace=True)
    
    # --- STEP 2: EXTRACT SURFACE ---
    # pass_pointid=True ensures we keep 'vtkOriginalPointIds' to map back to volume
    print("  Extracting surface...")
    surf_mesh = vol_mesh_0.extract_surface(pass_pointid=True)
    
    # Get the indices that map Surface Nodes -> Volume Nodes
    if "vtkOriginalPointIds" in surf_mesh.point_data:
        surf_indices = surf_mesh.point_data["vtkOriginalPointIds"]
    else:
        print("Warning: vtkOriginalPointIds not found. Relying on implicit ordering (risky).")
        surf_indices = None

    # Slice d(0) to get only surface values
    if surf_indices is not None:
        d_0_surf = d_0_vol[surf_indices]
    else:
        d_0_surf = surf_mesh.point_data[target_field_name]

    # Clean up the surface mesh (remove old arrays we don't need)
    surf_mesh.clear_data()
    
    # ---------------------------------------------------------
    # 2. PROCESS TIME SERIES
    # ---------------------------------------------------------
    
    # Process t=0 manually
    new_name = f"{output_basename}{start_idx:0{num_digits}d}"
    surf_mesh.point_data[new_name] = np.zeros_like(d_0_surf)
    print(f"  [Step {start_idx}] Added '{new_name}' (All zeros)")

    # Loop
    for i in range(start_idx + step_stride, end_idx + 1, step_stride):
        filename = input_pattern.format(i)
        if not os.path.exists(filename):
            continue
            
        if i % 1000 == 0: print(f"  [Step {i}] Processing...")

        # Load and CLEAN the volume mesh (must clean to match indices of vol_mesh_0)
        temp_mesh = load_and_merge(filename)
        d_t_vol = temp_mesh.point_data[target_field_name]
        
        # EXTRACT SURFACE DATA (Fast Mapping)
        if surf_indices is not None:
            # We trust that temp_mesh.clean() produced the exact same point order as vol_mesh_0.clean()
            d_t_surf = d_t_vol[surf_indices]
        else:
            # Fallback if indices missing
            temp_surf = temp_mesh.extract_surface()
            d_t_surf = temp_surf.point_data[target_field_name]

        # CALCULATE RELATIVE: d(t) - d(0)
        d_relative = d_t_surf - d_0_surf
        
        step_name = f"{output_basename}{i:0{num_digits}d}"
        surf_mesh.point_data[step_name] = d_relative
        
        del temp_mesh 

    # ---------------------------------------------------------
    # 3. SAVE
    # ---------------------------------------------------------
    print(f"Writing optimized surface file to {output_filename}...")
    surf_mesh.save(output_filename, binary=True)
    print("Done.")

if __name__ == "__main__":
    main()