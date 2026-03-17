from paraview.simple import *
import glob
import sys
import os

# ==========================================
# 1. CONFIGURAZIONE UTENTE
# ==========================================
# Cartella contenente i file .xdmf
FOLDER_NAME = "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/torso-reduced-long-horizon" # Cambia con la cartella che ti serve
OUTPUT_CSV = "scg_signal_normal.csv"

# LE TUE COORDINATE ESATTE SUL PETTO (X, Y, Z)
# Sostituisci questi valori con il punto che vuoi tracciare
PROBE_COORDS = [0.03648213669657707, 1.3178693056106567, -0.010187415406107903]

# Nome dell'array di accelerazione (di solito 'acceleration' in lifex/deal.II)
ACCEL_ARRAY_NAME = "acceleration"

# ==========================================
# 2. INIZIALIZZAZIONE PIPELINE
# ==========================================
# Trova tutti i file xdmf e ordinali numericamente
file_list = sorted(glob.glob(f"{FOLDER_NAME}/solution_*.xdmf"))

if not file_list:
    print(f"ERRORE: Nessun file .xdmf trovato in '{FOLDER_NAME}'.")
    sys.exit(1)

print(f"Trovati {len(file_list)} time-step. Costruzione della pipeline...")

# Disabilita cache per salvare RAM
paraview.simple._DisableFirstRenderCameraReset()

# 1. Lettore Dati
reader = XDMFReader(registrationName='SimData', FileNames=file_list)

# 2. Estrai la superficie (necessario per calcolare le normali correttamente)
surface = ExtractSurface(registrationName='Surface', Input=reader)

# 3. Calcola le Normali alla superficie
normals = GenerateSurfaceNormals(registrationName='Normals', Input=surface)
#normals.ComputeCellNormals = 0
#normals.ComputePointNormals = 1 # Vogliamo le normali sui nodi

# 4. Calcolatore: Prodotto Scalare tra Accelerazione e Normale
# In ParaView, dot(A, B) fa esattamente a_x*n_x + a_y*n_y + a_z*n_z
calc = Calculator(registrationName='NormalAccelCalc', Input=normals)
calc.AttributeType = 'Point Data'
calc.ResultArrayName = 'normal_a'
calc.Function = f'dot({ACCEL_ARRAY_NAME}, Normals)'

# 5. Sonda (Probe) al punto esatto
print(f"Posizionamento sensore SCG virtuale in coordinate: {PROBE_COORDS}")
probe = ProbeLocation(registrationName='SCG_Sensor', Input=calc, ProbeType='Fixed Radius Point Source')
probe.ProbeType.Center = PROBE_COORDS

# 6. Estrai i dati nel tempo (sfruttiamo il lettore per ciclare sui frame)
print("Estrazione del segnale SCG nel tempo. Questa operazione richiede qualche minuto...")
pdot = PlotDataOverTime(registrationName='PlotTime', Input=probe)

# ==========================================
# 3. SALVATAGGIO
# ==========================================
# Salva il risultato nel CSV
SaveData(OUTPUT_CSV, proxy=pdot, PointDataArrays=['acceleration_normal', ACCEL_ARRAY_NAME])

print(f"\n[SUCCESSO] Segnale SCG salvato correttamente in: {OUTPUT_CSV}")