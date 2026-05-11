import joblib
FIO2 = [0.3, 0.6, 1.0]
PEEP = [6, 10, 14]
IP = [8, 14, 20]
RR = [12, 18, 24]


severity_pairs = [
    (0.3, 0.3),
    (0.3, 0.7),
    (0.6, 0.3),
    (0.6, 0.7),
    (0.9, 0.3),
    (0.9, 0.7)
]


results = []


for ards, bi in severity_pairs:
    best_score = -1
    best_config = None

    for fio2 in FIO2:
        for peep in PEEP:
            for ip in IP:
                for rr in RR:

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

                    score = rf_pc.predict(features)[0]
                    score = float(np.clip(score, 0, 100))

                    if score > best_score:
                        best_score = score
                        best_config = {
                            "ARDS_Severity": ards,
                            "BI_Severity": bi,
                            "Recommended_Mode": "Pressure Control",
                            "FiO2": fio2,
                            "PEEP": peep,
                            "Inspiratory_Pressure": ip,
                            "Respiratory_Rate": rr,
                            "Predicted_Health_Score": round(score, 1)
                        }

    results.append(best_config)


results_df = pd.DataFrame(results)
results_df.to_csv("optimal_settings_final.csv", index=False)

print(results_df)