import pickle
from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import pandas as pd

# ----------------------------
# Input schema
# ----------------------------
class WeatherSample(BaseModel):
    temperature: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    precipitation: float | None = None
    cloud_cover: str | None = None  # String, nem lista
    atmospheric_pressure: float | None = None
    uv_index: float | None = None
    season: str | None = None
    visibility: float | None = None
    location: str | None = None

# ----------------------------
# Output schema
# ----------------------------
class PredictResponse(BaseModel):
    weather_type: str

# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="Weather Type Prediction")

# Load trained model and encoders from file
with open("random_forest_model_pipeline.pkl", "rb") as f_in:
    pipeline, label_encoders = pickle.load(f_in)

# ----------------------------
# Prediction logic
# ----------------------------
def preprocess_data(sample: Dict) -> pd.DataFrame:
    """
    Preprocess the input sample before passing it to the model.
    This includes encoding categorical features and handling missing values.
    """
    # Convert input sample to DataFrame
    df = pd.DataFrame([sample])

    df.rename(columns={
        'cloud_cover': 'Cloud Cover',
        'season': 'Season',
        'location': 'Location'
    }, inplace=True)

    # Encoding categorical features (cloud_cover, season, location) using the fitted LabelEncoders
    for column in ["Cloud Cover", "Season", "Location"]:
        if column in df.columns and df[column].notna().any():
            if df[column].iloc[0] in label_encoders[column].classes_:
                df[column] = label_encoders[column].transform(df[column])
            else:
                # Use the most frequent category for unseen values
                default_value = label_encoders[column].classes_[0]  # Default to the first class
                df[column] = label_encoders[column].transform([default_value])[0]

    # Fill missing values with the median (or use any other strategy if necessary)
    df = df.fillna(df.median())
    
    return df

def predict_single(sample: Dict) -> str:
    """
    Receives a dictionary, converts to DataFrame, preprocesses the data,
    and returns predicted weather type.
    """
    # Preprocess the data
    df = preprocess_data(sample)
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    return str(prediction)

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(sample: WeatherSample):
    pred = predict_single(sample.dict())
    return PredictResponse(weather_type=pred)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
