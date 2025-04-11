# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and column headers
model = joblib.load('ipl_score_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="IPL Run Predictor", layout="centered")
st.title("🏏 IPL Next Ball Run Predictor")

# UI Inputs
st.subheader("Match Situation")

batting_team = st.selectbox("Select Batting Team", [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
    'Rajasthan Royals', 'Sunrisers Hyderabad', 'Lucknow Super Giants',
    'Gujarat Titans'
])

bowler = st.text_input("Enter Bowler Name", "Jasprit Bumrah")
batter = st.text_input("Enter Batter Name", "Virat Kohli")
non_striker = st.text_input("Enter Non-Striker Name", "Faf du Plessis")

overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
ballnumber = st.number_input("Ball Number in Current Over", min_value=1, max_value=6, step=1)

if st.button("🔍 Predict Runs for Next Ball"):
    # Construct input DataFrame
    input_dict = {
        'batting_team': batting_team,
        'bowler': bowler,
        'batter': batter,
        'non_striker': non_striker,
        'overs': overs,
        'ballnumber': ballnumber
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode and align with training columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    predicted_run = model.predict(input_encoded)[0]

    st.markdown("### 🎯 Predicted Runs:")
    st.success(f"👉 **{round(predicted_run, 2)}** runs expected on this ball.")
