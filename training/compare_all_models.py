"""
compare_all_models.py
======================
Compare 9 regression models for Pressure Control and Volume Control separately.
Loads from bi_ards_dataset_with_scores.csv (output of calculate_health_score.py).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("9 MODEL COMPARISON — PRESSURE CONTROL vs VOLUME CONTROL")
print("=" * 70)

# ── Load data (with health scores already computed) ───────────────────────────
df = pd.read_csv("bi_ards_dataset_with_scores.csv")   # ← fixed
final_df = df[df['time_hours'] == df['time_hours'].max()].copy()

# Split by ventilation mode
pc_df = final_df[final_df['ventilation_mode'] == 'pressure control'].copy()
vc_df = final_df[final_df['ventilation_mode'] == 'volume control'].copy()

print(f"\nPressure Control rows : {len(pc_df)}")
print(f"Volume Control rows   : {len(vc_df)}")

# ── Feature sets ──────────────────────────────────────────────────────────────
pc_features = ['ards_severity', 'bi_severity', 'fio2', 'peep',
               'inspiratory_pressure', 'set_rr']

vc_features = ['ards_severity', 'bi_severity', 'fio2', 'peep',
               'set_rr', 'TidalVolume']

# ── Model definitions ─────────────────────────────────────────────────────────
models = {
    'Random Forest':    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost':          xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'LightGBM':         lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'Decision Tree':    DecisionTreeRegressor(random_state=42),
    'Ridge':            Ridge(alpha=1.0),
    'Lasso':            Lasso(alpha=0.01),
    'SVR':              SVR(kernel='rbf'),
    'MLP':              MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}


def evaluate_models(data, features, mode_name):
    """Train and evaluate all 9 models; return sorted results DataFrame."""
    X = data[features]
    y = data['health_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    for name, model in models.items():
        start = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred    = model.predict(X_test)
            train_r2  = model.score(X_train, y_train)
            test_r2   = r2_score(y_test, y_pred)
            test_mae  = mean_absolute_error(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            elapsed   = time.time() - start

            results.append({
                'Model':        name,
                'Train R²':     round(train_r2, 4),
                'Test R²':      round(test_r2, 4),
                'Test MAE':     round(test_mae, 4),
                'CV R² (mean)': round(cv_scores.mean(), 4),
                'CV R² (std)':  round(cv_scores.std(), 4),
                'Time (s)':     round(elapsed, 2),
            })
            print(f"  ✅ {name:<20} Test R²={test_r2:.4f}  MAE={test_mae:.4f}")
        except Exception as e:
            print(f"  ❌ {name} failed: {str(e)[:60]}")

    return pd.DataFrame(results).sort_values('Test R²', ascending=False)


# ── Run for Pressure Control ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PRESSURE CONTROL — Model Comparison")
print("=" * 70)
pc_results = evaluate_models(pc_df, pc_features, "Pressure Control")
print("\n", pc_results[['Model', 'Train R²', 'Test R²', 'Test MAE', 'CV R² (mean)']].to_string(index=False))

# ── Run for Volume Control ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VOLUME CONTROL — Model Comparison")
print("=" * 70)
vc_results = evaluate_models(vc_df, vc_features, "Volume Control")
print("\n", vc_results[['Model', 'Train R²', 'Test R²', 'Test MAE', 'CV R² (mean)']].to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
pc_results.to_csv("pc_model_comparison.csv", index=False)
vc_results.to_csv("vc_model_comparison.csv", index=False)
print("\n✅ Saved: pc_model_comparison.csv, vc_model_comparison.csv")

# ── Recommendation ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BEST MODELS")
print("=" * 70)
best_pc = pc_results.iloc[0]
best_vc = vc_results.iloc[0]
print(f"  Pressure Control → {best_pc['Model']:<20} Test R²={best_pc['Test R²']}  MAE={best_pc['Test MAE']}")
print(f"  Volume Control   → {best_vc['Model']:<20} Test R²={best_vc['Test R²']}  MAE={best_vc['Test MAE']}")
print("\n  Use these model names in train_final_models.py")