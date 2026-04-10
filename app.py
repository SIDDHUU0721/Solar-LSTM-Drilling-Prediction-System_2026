import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from io import BytesIO

# Load LSTM model

model = load_model("lstm_model.keras")
input_shape = model.input_shape  # (batch_size, timesteps, features)
n_features = input_shape[2]

# Feature names
feature_names = ['IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']


# Streamlit Layout

st.set_page_config(page_title="Solar Drilling Prediction", layout="wide")
st.title("🚀 Solar AC Power & Drilling Predictor")

st.markdown(
    """
    Enter values for the three features or upload a CSV for batch predictions.
    The app will predict AC_POWER and give a drilling recommendation.
    """
)

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Input", "Batch CSV", "Visualizations"])


# TAB 1: Single Input

with tab1:
    st.subheader("Single Sample Prediction")
    input_data = []
    cols = st.columns(len(feature_names))
    for i, feature in enumerate(feature_names):
        with cols[i]:
            val = st.number_input(feature, min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            input_data.append(val)

    if st.button("Predict Single Sample"):
        data = np.array(input_data).reshape(1, 1, n_features)
        prediction = model.predict(data)
        predicted_power = prediction[0][0]

        # Calculate thresholds
        # You need to load your dataset df to get quantiles
        # Example: df = pd.read_csv("your_dataset.csv")
        # Here we just simulate quantiles for demonstration
        high = 75
        medium = 40

        # Drilling decision
        def drilling_mode(power):
            if power >= high:
                return "Full Drilling"
            elif power >= medium:
                return "Moderate Drilling"
            else:
                return "Low Drilling"

        decision = drilling_mode(predicted_power)

        st.success(f"Predicted AC_POWER: {predicted_power:.2f} W")
        st.info(f"Drilling Recommendation: {decision}")

        # Chart
        st.subheader("Single Prediction Chart")
        st.bar_chart(pd.DataFrame({'AC_POWER': [predicted_power]}))


# TAB 2: Batch CSV

with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Check feature names
        if list(df.columns[:n_features]) != feature_names:
            st.error(f"CSV must have columns: {feature_names}")
        else:
            st.write("Preview of uploaded CSV:", df.head())

            batch_data = df.values.reshape((df.shape[0], 1, n_features))
            batch_pred = model.predict(batch_data)
            df['AC_POWER'] = batch_pred

            # Simulate thresholds
            high = 75
            medium = 40

            def drilling_mode(power):
                if power >= high:
                    return "Full Drilling"
                elif power >= medium:
                    return "Moderate Drilling"
                else:
                    return "Low Drilling"

            df['Drilling_Recommendation'] = [drilling_mode(p) for p in df['AC_POWER']]

            st.write("Batch Predictions with Drilling Recommendations:", df)

            # Download predictions CSV
            towrite = BytesIO()
            df.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button(
                label="Download Predictions CSV",
                data=towrite,
                file_name="predictions_with_decision.csv",
                mime="text/csv"
            )


# TAB 3: Visualizations

with tab3:
    st.subheader("Prediction Visualizations")

    # Single prediction chart
    if 'prediction' in locals():
        st.markdown("**Single Sample Prediction**")
        st.bar_chart(pd.DataFrame({'AC_POWER': [predicted_power]}))

    # Batch predictions chart
    if uploaded_file and 'AC_POWER' in df.columns:
        st.markdown("**Batch Predictions**")
        st.bar_chart(pd.DataFrame({'AC_POWER': df['AC_POWER']}))
