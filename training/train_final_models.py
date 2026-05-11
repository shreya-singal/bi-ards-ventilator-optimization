import warnings
]


pc_df = final_df[
    final_df["ventilation_mode"] == "pressure control"
]

X_pc = pc_df[pc_features]
y_pc = pc_df["health_score"]


rf_pc = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_pc.fit(X_pc, y_pc)

pc_predictions = rf_pc.predict(X_pc)
pc_r2 = rf_pc.score(X_pc, y_pc)
pc_mae = mean_absolute_error(y_pc, pc_predictions)

joblib.dump(rf_pc, "../models/rf_model_pc.pkl")


vc_df = final_df[
    final_df["ventilation_mode"] == "volume control"
]

X_vc = vc_df[vc_features]
y_vc = vc_df["health_score"]


lgb_vc = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbose=-1
)

lgb_vc.fit(X_vc, y_vc)

vc_predictions = lgb_vc.predict(X_vc)
vc_r2 = lgb_vc.score(X_vc, y_vc)
vc_mae = mean_absolute_error(y_vc, vc_predictions)

joblib.dump(lgb_vc, "../models/rf_model_vc.pkl")


print("\nModel Summary")
print("-" * 50)
print(f"Pressure Control  | Random Forest | R²: {pc_r2:.4f} | MAE: {pc_mae:.4f}")
print(f"Volume Control    | LightGBM      | R²: {vc_r2:.4f} | MAE: {vc_mae:.4f}")