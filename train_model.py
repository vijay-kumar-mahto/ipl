# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("IPL.csv")

df.dropna(inplace=True)

# Features for next ball prediction
df = df[['BattingTeam', 'bowler', 'batter', 'non-striker', 'overs', 'ballnumber', 'total_run']]
df.rename(columns={
    'BattingTeam': 'batting_team',
    'batter': 'batter',
    'non-striker': 'non_striker',
    'bowler': 'bowler'
}, inplace=True)

X = df[['batting_team', 'bowler', 'batter', 'non_striker', 'overs', 'ballnumber']]
y = df['total_run']  # runs scored in that specific ball

# One-hot encode all categorical
X_encoded = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

joblib.dump(model, 'ipl_score_model.pkl')
joblib.dump(X_encoded.columns, 'model_columns.pkl')
