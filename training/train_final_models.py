# final_optimization.py
"""
FINAL CORRECT OPTIMIZATION - BI+ARDS
TV removed from VC features — it is a patient response (continuous),
not a controllable ventilator setting.
"""

import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("FINAL OPTIMIZATION - BI+ARDS")
print("=" * 70)

# Load models
rf_pc  = joblib.load("rf_model_pc.pkl")
lgb_vc = joblib.load("rf_model_vc.pkl")

PC_FEATURES = rf_pc.feature_names_in_.tolist()
VC_FEATURES = lgb_vc.feature_name_

print(f"PC features: {PC_FEATURES}")
print(f"VC features: {VC_FEATURES}")

# Action space
FIO2     = [0.3, 0.6, 1.0]
PEEP     = [6, 10, 14]
IP       = [8, 14, 20]
RR       = [12, 18, 24]       # ← fixed: all 3 levels
SUBMODES = ['Assisted Control', 'Continuous Mandatory']

# Severity grid with clinical mode preference
# Pressure Control for moderate-severe ARDS (better pressure limiting)
# Volume Control for mild ARDS
SEVERITIES = [
    (0.3, 0.3, "Volume Control"),
    (0.3, 0.7, "Volume Control"),
    (0.6, 0.3, "Pressure Control"),
    (0.6, 0.7, "Pressure Control"),
    (0.9, 0.3, "Pressure Control"),
    (0.9, 0.7, "Pressure Control"),
]


def find_best_pc(ards, bi):
    best_score = -1
    best = None
    for fio2 in FIO2:
        for peep in PEEP:
            for ip in IP:
                for rr in RR:
                    for sm in SUBMODES:
                        features = {
                            'ards_severity':        ards,
                            'bi_severity':          bi,
                            'fio2':                 fio2,
                            'peep':                 peep,
                            'inspiratory_pressure': ip,
                            'set_rr':               rr,
                        }
                        X     = pd.DataFrame([features])[PC_FEATURES]
                        score = float(np.clip(rf_pc.predict(X)[0], 0, 100))
                        if score > best_score:
                            best_score = score
                            best = {
                                'ARDS': ards, 'BI': bi,
                                'Mode': 'Pressure Control',
                                'SubMode': sm,
                                'FiO2': fio2,
                                'PEEP': peep,
                                'Inspiratory_Pressure': ip,
                                'Tidal_Volume': 'N/A',
                                'RR': rr,
                                'Score': round(score, 1),
                            }
    return best


def find_best_vc(ards, bi):
    """TV is a patient response, not a setting — not searched or passed."""
    best_score = -1
    best = None
    for fio2 in FIO2:
        for peep in PEEP:
            for rr in RR:
                for sm in SUBMODES:
                    features = {
                        'ards_severity': ards,
                        'bi_severity':   bi,
                        'fio2':          fio2,
                        'peep':          peep,
                        'set_rr':        rr,
                    }
                    X     = pd.DataFrame([features])[VC_FEATURES]
                    score = float(np.clip(lgb_vc.predict(X)[0], 0, 100))
                    if score > best_score:
                        best_score = score
                        best = {
                            'ARDS': ards, 'BI': bi,
                            'Mode': 'Volume Control',
                            'SubMode': sm,
                            'FiO2': fio2,
                            'PEEP': peep,
                            'Inspiratory_Pressure': 'N/A',
                            'Tidal_Volume': 'N/A',
                            'RR': rr,
                            'Score': round(score, 1),
                        }
    return best


results = []

print("\nSearching optimal settings with clinical rules...")
print("-" * 70)

for ards, bi, recommended_mode in SEVERITIES:
    best_action = find_best_pc(ards, bi) if recommended_mode == "Pressure Control" else find_best_vc(ards, bi)
    results.append(best_action)
    ip_str = f"IP={best_action['Inspiratory_Pressure']}" if best_action['Mode'] == 'Pressure Control' else "VC mode"
    print(f"ARDS={ards}, BI={bi}: {best_action['Mode']} | "
          f"FiO2={best_action['FiO2']} | PEEP={best_action['PEEP']} | "
          f"{ip_str} | RR={best_action['RR']} | Score={best_action['Score']}")

df_results = pd.DataFrame(results)
df_results.to_csv("optimal_settings_final_correct.csv", index=False)

print("\n" + "=" * 70)
print("FINAL OPTIMAL SETTINGS TABLE")
print("=" * 70)
print(df_results.to_string(index=False))

print("\n" + "=" * 70)
print("CLINICAL RECOMMENDATION")
print("=" * 70)
print("""
For Severe/Moderate ARDS (0.6-0.9): Use PRESSURE CONTROL
  - Limits peak airway pressures (lung-protective)
  - Better for sicker, stiff-lung patients

For Mild ARDS (0.3): Use VOLUME CONTROL
  - Guaranteed tidal volume delivery
  - Simpler to manage at milder severity
""")