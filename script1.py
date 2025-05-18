import pandas as pd
import numpy as np
import pickle
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# 1. THINGSPEAK DETAILS 

THING_SPEAK_CHANNEL_ID = "2849586"   # Example Channel ID
READ_API_KEY = "XVZ5FB3IFVVZ6Y2"     # Example Read API Key
# Fetch only the latest entry from the channel:
THING_SPEAK_URL = f"https://api.thingspeak.com/channels/{THING_SPEAK_CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"


# 2. DATASET LOADING & PREPROCESSING

@st.cache_data
def load_data():
    """
    Reads the CSV file containing cattle health data.
    Returns a DataFrame.
    """
    df = pd.read_csv("synthetic_cattle_health_data_3.csv")
    return df

def preprocess_data(df):
    """
    Splits the data into features (X) and target (y),
    then performs a train-test split.
    """
    # Features: sensor readings
    X = df[['Heart_Rate_bpm', 'Temperature_C', 'Humidity_percent', 'Respiratory_Rate_breaths_min']]
    # Target: disease/health status
    y = df['Health_Status']
    # Split data into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# 3. MODEL TRAINING & SAVING

def train_and_save_model(X_train, y_train):
    """
    Trains a RandomForestClassifier on the training data,
    then saves the trained model to 'model.pkl'.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


# 4. MODEL LOADING

def load_model():
    """
    Loads the trained model from 'model.pkl' for prediction.
    """
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


# 5. FETCH LATEST SENSOR DATA FROM THINGSPEAK

def fetch_sensor_data():
    """
    Sends a GET request to ThingSpeak to fetch the latest sensor data.
    Returns a NumPy array shaped (1,4) if successful, else None.
    """
    response = requests.get(THING_SPEAK_URL)
    if response.status_code == 200:
        data = response.json()
        feeds = data.get("feeds", [])
        if feeds:
            latest_entry = feeds[-1]
            try:
                # Convert each field to float, then to int if needed
                heart_rate    = float(latest_entry["field1"])
                temperature   = float(latest_entry["field2"])
                humidity      = float(latest_entry["field3"])
                respiratory   = float(latest_entry["field4"])
                
                # Reshape into 2D array for model prediction
                return np.array([[heart_rate, temperature, humidity, respiratory]])
            except (ValueError, TypeError):
                # If fields are empty or invalid
                return None
    return None


# 6. PREDICTION LOGIC

def predict_disease(model, input_data):
    """
    Uses the loaded model to predict disease given sensor input_data.
    Returns the predicted label as a string.
    """
    prediction = model.predict(input_data)
    return prediction[0]  # Extract the label


# 7. STREAMLIT APP

def main():
    st.title("🐄 Automated Cow Disease Prediction System")
    st.markdown("This app **automatically** fetches the latest sensor data from ThingSpeak and predicts the cow's health status using a RandomForest model.")

    # 7.1. Fetch latest data from ThingSpeak
    sensor_data = fetch_sensor_data()
    
    if sensor_data is not None:
        # 7.2. Load the trained model
        model = load_model()
        
        # 7.3. Predict disease
        disease_prediction = predict_disease(model, sensor_data)
        
        # 7.4. Display fetched data and prediction
        st.subheader("Latest Sensor Data from ThingSpeak")
        st.write(f"- **Heart Rate (bpm)**: {sensor_data[0][0]}")
        st.write(f"- **Temperature (°C)**: {sensor_data[0][1]}")
        st.write(f"- **Humidity (%)**: {sensor_data[0][2]}")
        st.write(f"- **Respiratory Rate**: {sensor_data[0][3]}")
        
        st.subheader("Predicted Health Status")
        st.write(f"**{disease_prediction}**")

    else:
        st.error("Failed to fetch valid data from ThingSpeak. Please check your Read API Key, Channel ID, and fields.")


# 8. MAIN EXECUTION

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train model if 'model.pkl' doesn't exist
    try:
        open("model.pkl", "rb").close()
    except FileNotFoundError:
        train_and_save_model(X_train, y_train)
    
    # Run the Streamlit app
    main()
