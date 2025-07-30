import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    results = pd.read_csv("results.csv")
    races = pd.read_csv("races.csv")
    drivers = pd.read_csv("drivers.csv")
    constructors = pd.read_csv("constructors.csv")
    circuits = pd.read_csv("circuits.csv")

    # Drop conflicting columns (like 'url') if they exist
    for df in [races, drivers, constructors, circuits]:
        if 'url' in df.columns:
            df.drop(columns=['url'], inplace=True)

    # Merge dataframes
    df = results.merge(races, on='raceId', suffixes=('', '_race')) \
                .merge(drivers, on='driverId') \
                .merge(constructors, on='constructorId') \
                .merge(circuits, on='circuitId')

    # Feature engineering
    df['driver_age'] = df['year'] - pd.to_numeric(df['dob'].str[:4], errors='coerce')
    df['is_home_race'] = (df['nationality_x'] == df['nationality_y']).astype(int)
    df['top3'] = (df['positionOrder'] <= 3).astype(int)

    return df

# Load and prepare data
df = load_data()
features = ['grid', 'driver_age', 'positionOrder', 'points', 'is_home_race']
df_model = df[['raceId', 'driverId', 'surname'] + features + ['top3']].dropna()

# Train model
X = df_model[features]
y = df_model['top3']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --------------------------
# Streamlit UI
# --------------------------
st.title("🏎️ F1 Podium Predictor App")
st.markdown("Predict which drivers are likely to finish in the **Top 3** for a given race.")

# Dropdown for raceId
race_ids = df_model['raceId'].unique()
selected_race = st.selectbox("Select a race ID:", sorted(race_ids))

# Filter and predict
race_drivers = df_model[df_model['raceId'] == selected_race].copy()
X_race = race_drivers[features]
race_drivers['Top 3 Probability'] = model.predict_proba(X_race)[:, 1]
race_drivers['Predicted Top 3'] = model.predict(X_race)

# Sort and display
race_drivers_sorted = race_drivers.sort_values(by='Top 3 Probability', ascending=False)
st.subheader(f"Predictions for Race ID {selected_race}")
st.dataframe(
    race_drivers_sorted[['surname', 'grid', 'Top 3 Probability', 'Predicted Top 3']],
    use_container_width=True
)
