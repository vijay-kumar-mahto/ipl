import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("model.pkl")                # Trained Model
le = joblib.load("label_encoder.pkl")           # LabelEncoder used on BattingTeam
y_pred = joblib.load("y_pred.pkl")              # Predictions from training script
y_test = joblib.load("y_test.pkl")              # Ground truth from test set

# Create team mapping from encoder
teams = le.classes_
team_mapping = dict(zip(le.transform(teams), teams))

# Streamlit app UI
st.title("IPL Team Score Predictor")

st.write("This app predicts the total score for a team in a given innings based on IPL match data.")

# Inputs
team_name = st.selectbox("Team", list(teams))
innings = st.selectbox("Innings", [1, 2])
mae = st.slider("Margin of Error (MAE)", min_value=1, max_value=100, value=40)

# Predict when user clicks button
if st.button("Predict"):
    team_id = le.transform([team_name])[0]
    sample_input = pd.DataFrame([[team_id, innings]], columns=['BattingTeam', 'innings'])
    predicted_score = int(model.predict(sample_input)[0])

    within_margin = np.abs(y_pred - y_test) <= mae
    confidence = np.mean(within_margin) * 100

    st.subheader("Prediction Result")
    st.write("Team:", team_name)
    st.write("Innings:", innings)
    st.write("Predicted Score:", f"{predicted_score} ± {mae} runs")
    st.write("Confidence (within margin):", f"{confidence:.2f}%")