# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BI+ARDS Ventilator Optimization",
    page_icon="🫁",
    layout="wide"
)

st.title("BI+ARDS Ventilator Optimization Tool")

st.markdown(
    """
    This tool predicts patient health scores for BI+ARDS patients
    using machine learning models trained on Pulse Physiology
    simulations.
    """
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

DATA_PATH = BASE_DIR / "bi_ards_dataset_with_scores.csv"

# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():

    rf_pc = joblib.load(
        MODELS_DIR / "rf_model_pc.pkl"
    )

    rf_vc = joblib.load(
        MODELS_DIR / "rf_model_vc.pkl"
    )

    return rf_pc, rf_vc


@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


try:

    rf_pc, rf_vc = load_models()

    optimal_df = load_dataset()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(f"Error loading files: {e}")

# ─────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────

ards_labels = {
    0.3: "Mild ARDS",
    0.6: "Moderate ARDS",
    0.9: "Severe ARDS"
}

bi_labels = {
    0.3: "Mild Brain Injury",
    0.7: "Severe Brain Injury"
}

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Patient Configuration")

ards_val = st.sidebar.select_slider(
    "ARDS Severity",
    options=[0.3, 0.6, 0.9],
    value=0.6,
    format_func=lambda x: ards_labels[x]
)

bi_val = st.sidebar.select_slider(
    "Brain Injury Severity",
    options=[0.3, 0.7],
    value=0.3,
    format_func=lambda x: bi_labels[x]
)

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Ventilation Mode",
    ["Pressure Control", "Volume Control"]
)

sub_mode = st.sidebar.selectbox(
    "Sub Mode",
    ["Assisted Control", "Continuous Mandatory"]
)

fio2 = st.sidebar.slider(
    "FiO₂",
    0.3,
    1.0,
    0.6,
    0.05
)

peep = st.sidebar.slider(
    "PEEP (cmH₂O)",
    6,
    14,
    10
)

rr = st.sidebar.slider(
    "Respiratory Rate",
    12,
    24,
    18
)

ip = None

if mode == "Pressure Control":

    ip = st.sidebar.slider(
        "Inspiratory Pressure",
        8,
        20,
        14
    )

predict_clicked = st.sidebar.button(
    "Predict Health Score",
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# Prediction functions
# ─────────────────────────────────────────────────────────────

def predict_pc():

    features = pd.DataFrame(
        [[
            ards_val,
            bi_val,
            fio2,
            peep,
            ip,
            rr
        ]],
        columns=[
            "ards_severity",
            "bi_severity",
            "fio2",
            "peep",
            "inspiratory_pressure",
            "set_rr"
        ]
    )

    features = features[
        rf_pc.feature_names_in_
    ]

    prediction = rf_pc.predict(features)[0]

    return round(
        float(np.clip(prediction, 0, 100)),
        1
    )


def predict_vc():

    features = pd.DataFrame(
        [[
            ards_val,
            bi_val,
            fio2,
            peep,
            rr
        ]],
        columns=[
            "ards_severity",
            "bi_severity",
            "fio2",
            "peep",
            "set_rr"
        ]
    )

    features = features[
        rf_vc.feature_names_in_
    ]

    prediction = rf_vc.predict(features)[0]

    return round(
        float(np.clip(prediction, 0, 100)),
        1
    )

# ─────────────────────────────────────────────────────────────
# Find best configuration
# ─────────────────────────────────────────────────────────────

def find_optimal(ards, bi):

    matches = optimal_df[
        (optimal_df["ards_severity"] == ards) &
        (optimal_df["bi_severity"] == bi)
    ]

    if len(matches) == 0:
        return None

    best = matches.sort_values(
        "health_score",
        ascending=False
    )

    return best.iloc[0]

# ─────────────────────────────────────────────────────────────
# Recommended configuration
# ─────────────────────────────────────────────────────────────

if models_loaded:

    optimal = find_optimal(
        ards_val,
        bi_val
    )

    if optimal is not None:

        st.subheader("Recommended Ventilator Settings")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### Mode")

            st.write(
                optimal["ventilation_mode"]
            )

            st.write(
                optimal["sub_mode"]
            )

        with col2:

            st.markdown("### Key Parameters")

            st.write(
                f"FiO₂: {optimal['fio2']:.1f}"
            )

            st.write(
                f"PEEP: {optimal['peep']:.0f} cmH₂O"
            )

            st.write(
                f"Respiratory Rate: "
                f"{optimal['set_rr']:.0f}/min"
            )

        with col3:

            st.markdown("### Additional Settings")

            if (
                optimal["ventilation_mode"]
                == "Pressure Control"
            ):

                st.write(
                    f"Inspiratory Pressure: "
                    f"{optimal['inspiratory_pressure']:.0f} cmH₂O"
                )

            else:

                st.write(
                    "Volume Control ventilation"
                )

        st.markdown("---")

        best_score = optimal["health_score"]

        if best_score >= 80:

            st.success(
                f"Predicted Health Score: "
                f"{best_score:.1f}/100"
            )

        elif best_score >= 60:

            st.warning(
                f"Predicted Health Score: "
                f"{best_score:.1f}/100"
            )

        else:

            st.error(
                f"Predicted Health Score: "
                f"{best_score:.1f}/100"
            )

# ─────────────────────────────────────────────────────────────
# Manual prediction
# ─────────────────────────────────────────────────────────────

if predict_clicked and models_loaded:

    st.markdown("---")

    st.subheader("Manual Ventilator Evaluation")

    if mode == "Pressure Control":

        predicted_score = predict_pc()

    else:

        predicted_score = predict_vc()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Selected Settings")

        st.write(f"Mode: {mode}")

        st.write(f"Sub Mode: {sub_mode}")

        st.write(f"FiO₂: {fio2}")

        st.write(f"PEEP: {peep} cmH₂O")

        st.write(f"Respiratory Rate: {rr}/min")

        if mode == "Pressure Control":

            st.write(
                f"Inspiratory Pressure: "
                f"{ip} cmH₂O"
            )

    with col2:

        st.markdown("### Predicted Outcome")

        if predicted_score >= 80:

            st.success(
                f"Health Score: "
                f"{predicted_score}/100"
            )

        elif predicted_score >= 60:

            st.warning(
                f"Health Score: "
                f"{predicted_score}/100"
            )

        else:

            st.error(
                f"Health Score: "
                f"{predicted_score}/100"
            )

# ─────────────────────────────────────────────────────────────
# Reference table
# ─────────────────────────────────────────────────────────────

if models_loaded:

    st.markdown("---")

    st.subheader("Top Performing Configurations")

    display_df = (
        optimal_df
        .sort_values(
            "health_score",
            ascending=False
        )
        .head(15)
    )

    display_df = display_df[
        [
            "ards_severity",
            "bi_severity",
            "ventilation_mode",
            "sub_mode",
            "fio2",
            "peep",
            "inspiratory_pressure",
            "set_rr",
            "health_score"
        ]
    ].copy()

    display_df.columns = [
        "ARDS Severity",
        "BI Severity",
        "Ventilation Mode",
        "Sub Mode",
        "FiO₂",
        "PEEP",
        "Inspiratory Pressure",
        "Respiratory Rate",
        "Health Score"
    ]

    display_df["Health Score"] = (
        display_df["Health Score"]
        .round(1)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# ─────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────

if models_loaded:

    st.markdown("---")

    st.subheader("Feature Importance")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Pressure Control Model"
        )

        pc_imp = pd.DataFrame({

            "Feature":
                rf_pc.feature_names_in_,

            "Importance":
                (
                    rf_pc.feature_importances_
                    * 100
                ).round(1)
        })

        pc_imp = pc_imp.sort_values(
            "Importance",
            ascending=True
        )

        fig1 = px.bar(
            pc_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        fig1.update_layout(
            height=400,
            margin=dict(
                l=0,
                r=0,
                t=30,
                b=0
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### Volume Control Model"
        )

        vc_imp = pd.DataFrame({

            "Feature":
                rf_vc.feature_names_in_,

            "Importance":
                (
                    rf_vc.feature_importances_
                    * 100
                ).round(1)
        })

        vc_imp = vc_imp.sort_values(
            "Importance",
            ascending=True
        )

        fig2 = px.bar(
            vc_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Greens"
        )

        fig2.update_layout(
            height=400,
            margin=dict(
                l=0,
                r=0,
                t=30,
                b=0
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────

st.markdown("---")

st.caption(
    "BI+ARDS Ventilator Optimization using "
    "Pulse Physiology simulations and "
    "Machine Learning"
)