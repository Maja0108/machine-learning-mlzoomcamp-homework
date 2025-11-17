import pickle
from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import pandas as pd

# ----------------------------
# Input schema
# ----------------------------
class AirSample(BaseModel):
    CO_GT: float | None = None           # például ha van előző mérés
    PT08_S1_CO: float | None = None
    NMHC_GT: float | None = None
    C6H6_GT: float | None = None
    PT08_S2_NMHC: float | None = None
    NOx_GT: float | None = None
    PT08_S3_NOx: float | None = None
    NO2_GT: float | None = None
    PT08_S4_NO2: float | None = None
    PT08_S5_O3: float | None = None
    T: float | None = None
    RH: float | None = None
    AH: float | None = None

# ----------------------------
# Output schema
# ----------------------------
class PredictResponse(BaseModel):
    predicted_CO: float

# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="CO Concentration Prediction")

# Load trained model pipeline
with open("svr_model_pipeline.pkl", "rb") as f_in:
    pipeline = pickle.load(f_in)

# ----------------------------
# Prediction logic
# ----------------------------
def predict_single(sample: Dict) -> float:
    """
    Receives a dictionary, converts to DataFrame,
    and returns predicted CO concentration.
    """
    df = pd.DataFrame([sample])
    # fill missing values with median (or use the pipeline's preprocessing)
    df = df.fillna(df.median())
    prediction = pipeline.predict(df)[0]
    return float(prediction)

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(sample: AirSample):
    pred = predict_single(sample.model_dump())
    return PredictResponse(predicted_CO=pred)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
