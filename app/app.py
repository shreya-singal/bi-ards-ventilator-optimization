import streamlit as st

recommended = optimal_df[
    (optimal_df["ARDS_Severity"] == ards)
    & (optimal_df["BI_Severity"] == bi)
].iloc[0]


st.subheader("Recommended Ventilator Settings")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mode", recommended["Recommended_Mode"])
    st.metric("FiO2", recommended["FiO2"])
    st.metric("PEEP", recommended["PEEP"])

with col2:
    st.metric("Respiratory Rate", recommended["Respiratory_Rate"])
    st.metric(
        "Predicted Health Score",
        round(recommended["Predicted_Health_Score"], 1)
    )


if predict:
    st.subheader("Prediction Result")

    if mode == "Pressure Control":
        score = predict_pressure_control(
            ards,
            bi,
            fio2,
            peep,
            inspiratory_pressure,
            rr
        )
    else:
        score = predict_volume_control(
            ards,
            bi,
            fio2,
            peep,
            rr
        )

    st.metric("Predicted Health Score", f"{score}/100")


st.subheader("Feature Importance")

pc_importance = pd.DataFrame({
    "Feature": rf_pc.feature_names_in_,
    "Importance": rf_pc.feature_importances_
})

fig = px.bar(
    pc_importance.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)