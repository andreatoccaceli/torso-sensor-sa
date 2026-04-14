from paraview.simple import *
import glob
import sys
import os

# ==========================================
# 1. CONFIGURAZIONE UTENTE
# ==========================================
# Cartella contenente i file .xdmf
# FOLDER_NAME = "/home/andrea/Scrivania/phd/simulations/heart-torso/disp_torso/torso-reduced-long-horizon" # Cambia con la cartella che ti serve
FOLDER_NAME = "/home/andrea/Scrivania/phd/simulations/heart-torso/SA_torso/incr_ea" # Cambia con la cartella che ti serve
OUTPUT_CSV = "/home/andrea/Scrivania/phd/simulations/heart-torso/csv-data/sa_incr_ea.csv"

# LE TUE COORDINATE ESATTE SUL PETTO (X, Y, Z)
# Sostituisci questi valori con il punto che vuoi tracciare
PROBE_COORDS = [0.03648213669657707, 1.3178693056106567, -0.010187415406107903] # below the apex
#PROBE_COORDS = [-0.00811256, 1.32603, 0.0498535] # at the center
#PROBE_COORDS = [-0.0827177, 1.49353, 0.0127231] # with pvsm, wrong position since is slighty above the base


# Nome dell'array di accelerazione (di solito 'acceleration' in lifex/deal.II)
ACCEL_ARRAY_NAME = "acceleration"

# RAGGIO DEL SENSORE (in metri, adatta alla scala della tua mesh)
# Es. 0.015 significa un sensore con raggio 1.5 cm
USE_RADIUS = False
SENSOR_RADIUS = 0.02

# ==========================================
# 2. INIZIALIZZAZIONE PIPELINE
# ==========================================
file_list = sorted(glob.glob(f"{FOLDER_NAME}/solution_*.xdmf"))
if not file_list:
    print(f"ERRORE: Nessun file trovato.")
    sys.exit(1)

print(f"Trovati {len(file_list)} time-step. Costruzione pipeline per area media...")
paraview.simple._DisableFirstRenderCameraReset()

reader = XDMFReader(registrationName='SimData', FileNames=file_list)
surface = ExtractSurface(registrationName='Surface', Input=reader)

# Calcola le Normali
normals = GenerateSurfaceNormals(registrationName='Normals', Input=surface)

# Calcola il prodotto scalare (Accelerazione Normale su ogni nodo)
calc = Calculator(registrationName='NormalAccelCalc', Input=normals)
calc.AttributeType = 'Point Data'
calc.ResultArrayName = 'normal_a'
calc.Function = f'dot({ACCEL_ARRAY_NAME}, Normals)'

# ==========================================
# LA MAGIA DEL SENSORE FISICO: TAGLIO E INTEGRAZIONE
# ==========================================
if USE_RADIUS:
    print(f"Modalita' AREA: Creazione area sensore con raggio {SENSOR_RADIUS}m...")

    # 1. Taglia la mesh mantenendo solo la "toppa" dentro la sfera
    clip = Clip(registrationName='SensorPatch', Input=calc)
    clip.ClipType = 'Sphere'
    clip.ClipType.Center = PROBE_COORDS
    clip.ClipType.Radius = SENSOR_RADIUS
    clip.Invert = 0 # 0 = Mantieni solo i dati DENTRO la sfera

    # 2. Integra le variabili su tutta la toppa
    extraction_filter = IntegrateVariables(registrationName='Integrator', Input=clip)
    
else:
    print(f"Modalita' PUNTO: Posizionamento sensore SCG virtuale in coordinate: {PROBE_COORDS}")
    extraction_filter = ProbeLocation(registrationName='SCG_Sensor', Input=calc, ProbeType='Fixed Radius Point Source')
    extraction_filter.ProbeType.Center = PROBE_COORDS

    

# Estrai nel tempo
print("Estrazione nel tempo in corso (questa operazione elabora tutti i frame)...")
pdot = PlotDataOverTime(registrationName='PlotTime', Input=extraction_filter)

# ==========================================
# 3. SALVATAGGIO
# ==========================================
print("Salvataggio dati...")
SaveData(OUTPUT_CSV, proxy=pdot)
print(f"\n[SUCCESSO] Dati integrati salvati in: {OUTPUT_CSV}")