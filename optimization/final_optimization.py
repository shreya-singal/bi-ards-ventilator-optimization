import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "optimization" / "optimal_settings_final.csv"

print("=" * 70)
print("BI+ARDS Ventilator Optimization")
print("=" * 70)

rf_pc = joblib.load(MODELS_DIR / "rf_model_pc.pkl")
rf_vc = joblib.load(MODELS_DIR / "rf_model_vc.pkl")

PC_FEATURES = rf_pc.feature_names_in_.tolist()
VC_FEATURES = rf_vc.feature_names_in_

FIO2_VALUES = [0.3, 0.6, 1.0]
PEEP_VALUES = [6, 10, 14]
IP_VALUES = [8, 14, 20]
RR_VALUES = [12, 18, 24]

SUBMODES = [
    "Assisted Control",
    "Continuous Mandatory"
]

SEVERITY_GRID = [
    (0.3, 0.3, "Volume Control"),
    (0.3, 0.7, "Volume Control"),
    (0.6, 0.3, "Pressure Control"),
    (0.6, 0.7, "Pressure Control"),
    (0.9, 0.3, "Pressure Control"),
    (0.9, 0.7, "Pressure Control")
]


def predict_pc(ards, bi, fio2, peep, ip, rr):

    features = pd.DataFrame([
        {
            "ards_severity": ards,
            "bi_severity": bi,
            "fio2": fio2,
            "peep": peep,
            "inspiratory_pressure": ip,
            "set_rr": rr
        }
    ])

    features = features[PC_FEATURES]

    score = rf_pc.predict(features)[0]

    return float(np.clip(score, 0, 100))


def predict_vc(ards, bi, fio2, peep, rr):
print("\nOptimal settings saved successfully."