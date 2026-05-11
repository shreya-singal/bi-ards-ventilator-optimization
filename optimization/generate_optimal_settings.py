"""
generate_optimal_settings.py — BI+ARDS
========================================
Searches all ventilator setting combinations to find the one
that maximises the predicted health score for each severity pair.
"""

import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("OPTIMIZATION ALGORITHM — BI+ARDS")
print("=" * 70)

# ── Load models ──────────────────────────────────────────────────────────────
rf_pc  = joblib.load("rf_model_pc.pkl")   # Pressure Control: Random Forest
lgb_vc = joblib.load("rf_model_vc.pkl")   # Volume Control:   LightGBM

with open("rf_features_pc.txt") as f:
    PC_FEATURES = [l.strip() for l in f]
with open("rf_features_vc.txt") as f:
    VC_FEATURES = [l.strip() for l in f]

print(f"PC features: {PC_FEATURES}")
print(f"VC features: {VC_FEATURES}")

# ── Severity grid ─────────────────────────────────────────────────────────────
SEVERITIES = [
    (0.3, 0.3), (0.3, 0.7),
    (0.6, 0.3), (0.6, 0.7),
    (0.9, 0.3), (0.9, 0.7),
]

# ── Action space (must match your Pulse simulation DOE) ───────────────────────
FIO2 = [0.3, 0.6, 1.0]
PEEP = [6, 10, 14]          # cmH2O — higher PEEP appropriate for ARDS
IP   = [8, 14, 20]          # cmH2O — inspiratory pressure (PC only)
RR   = [12, 18, 24]         # breaths/min — full 3 levels

PC_SUBMODES = ['Assisted Pressure Control', 'Continuous Mandatory Pressure Control']
VC_SUBMODES = ['Assisted Volume Control',   'Continuous Mandatory Volume Control']


def clip_score(score):
    return float(np.clip(score, 0, 100))


results = []

print("\nSearching optimal settings...")
print("-" * 70)

for ards_sev, bi_sev in SEVERITIES:
    print(f"\nARDS severity={ards_sev}, BI severity={bi_sev}")
    best_score  = -np.inf
    best_action = None

    # ── Pressure Control search ───────────────────────────────────────────
    for submode in PC_SUBMODES:
        for fio2 in FIO2:
            for peep in PEEP:
                for ip in IP:
                    for rr in RR:
                        features = {
                            'ards_severity':        ards_sev,
                            'bi_severity':          bi_sev,
                            'fio2':                 fio2,
                            'peep':                 peep,
                            'inspiratory_pressure': ip,
                            'set_rr':               rr,
                        }
                        X     = pd.DataFrame([features])[PC_FEATURES]
                        score = clip_score(rf_pc.predict(X)[0])

                        if score > best_score:
                            best_score  = score
                            best_action = {
                                'ARDS_Severity':          ards_sev,
                                'BI_Severity':            bi_sev,
                                'Recommended_Mode':       'Pressure Control',
                                'Sub_Mode':               submode,
                                'FiO2':                   fio2,
                                'PEEP':                   peep,
                                'Inspiratory_Pressure':   ip,
                                'Respiratory_Rate':       rr,
                                'Predicted_Health_Score': score,
                            }

    # ── Volume Control search (no TV — it's a patient response, not a setting) ─
    for submode in VC_SUBMODES:
        for fio2 in FIO2:
            for peep in PEEP:
                for rr in RR:
                    features = {
                        'ards_severity': ards_sev,
                        'bi_severity':   bi_sev,
                        'fio2':          fio2,
                        'peep':          peep,
                        'set_rr':        rr,
                    }
                    X     = pd.DataFrame([features])[VC_FEATURES]
                    score = clip_score(lgb_vc.predict(X)[0])

                    if score > best_score:
                        best_score  = score
                        best_action = {
                            'ARDS_Severity':          ards_sev,
                            'BI_Severity':            bi_sev,
                            'Recommended_Mode':       'Volume Control',
                            'Sub_Mode':               submode,
                            'FiO2':                   fio2,
                            'PEEP':                   peep,
                            'Inspiratory_Pressure':   'N/A',
                            'Respiratory_Rate':       rr,
                            'Predicted_Health_Score': score,
                        }

    results.append(best_action)
    print(f"   ✅ {best_action['Recommended_Mode']} ({best_action['Sub_Mode']}) | "
          f"FiO2={best_action['FiO2']} | PEEP={best_action['PEEP']} | "
          f"Score={best_action['Predicted_Health_Score']:.1f}")


df_results = pd.DataFrame(results).sort_values(['ARDS_Severity', 'BI_Severity'])

print("\n" + "=" * 70)
print("OPTIMAL VENTILATOR SETTINGS TABLE")
print("=" * 70)
print(df_results.to_string(index=False))

df_results.to_csv("optimal_settings_final.csv", index=False)
print("\n✅ Saved: optimal_settings_final.csv")

# ── Clinical summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLINICAL SUMMARY — BI+ARDS")
print("=" * 70)

ards_label = {0.3: 'Mild', 0.6: 'Moderate', 0.9: 'Severe'}
bi_label   = {0.3: 'Mild BI', 0.7: 'Severe BI'}

for _, row in df_results.iterrows():
    a_label = ards_label.get(row['ARDS_Severity'], f"ARDS={row['ARDS_Severity']}")
    b_label = bi_label.get(row['BI_Severity'],     f"BI={row['BI_Severity']}")
    ip_str  = (f"IP={row['Inspiratory_Pressure']} cmH₂O"
               if row['Inspiratory_Pressure'] != 'N/A'
               else "Volume Control mode")

    print(f"  {a_label} ARDS + {b_label}: "
          f"{row['Recommended_Mode']} ({row['Sub_Mode']}) | "
          f"FiO₂={row['FiO2']} | PEEP={row['PEEP']} | {ip_str} | "
          f"RR={row['Respiratory_Rate']} | Score={row['Predicted_Health_Score']:.1f}")