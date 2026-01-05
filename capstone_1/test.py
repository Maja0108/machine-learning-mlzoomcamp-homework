import pytest
from fastapi.testclient import TestClient
from predict import app  # Importáljuk a FastAPI alkalmazást, amelyet az előző kódban készítettünk

# ----------------------------
# Test Setup
# ----------------------------
client = TestClient(app)

# ----------------------------
# Test the /predict endpoint
# ----------------------------

def test_predict_sunny():
    payload = {
        "temperature": 22.5,
        "humidity": 78.0,
        "wind_speed": 15.0,
        "precipitation": 20.0,
        "cloud_cover": "Partly Cloudy",  # Itt helyesen küldjük a stringet
        "atmospheric_pressure": 1012.0,
        "uv_index": 5.0,
        "season": "Summer",
        "visibility": 10.0,
        "location": "Urban"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"weather_type": "Sunny"}  # Itt a predikciót ellenőrizzük

def test_predict_rainy():
    payload = {
        "temperature": 18.0,
        "humidity": 85.0,
        "wind_speed": 10.0,
        "precipitation": 80.0,
        "cloud_cover": "Cloudy",  # Itt is stringként kell küldeni
        "atmospheric_pressure": 1015.0,
        "uv_index": 3.0,
        "season": "Spring",
        "visibility": 8.0,
        "location": "Rural"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"weather_type": "Rainy"}  # Itt a predikciót ellenőrizzük

def test_predict_cloudy():
    payload = {
        "temperature": 15.0,
        "humidity": 70.0,
        "wind_speed": 5.0,
        "precipitation": 10.0,
        "cloud_cover": "Cloudy",  # Itt is stringként kell küldeni
        "atmospheric_pressure": 1010.0,
        "uv_index": 2.0,
        "season": "Fall",
        "visibility": 12.0,
        "location": "Urban"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"weather_type": "Cloudy"}  # Itt a predikciót ellenőrizzük

def test_predict_snowy():
    payload = {
        "temperature": -5.0,
        "humidity": 90.0,
        "wind_speed": 12.0,
        "precipitation": 100.0,
        "cloud_cover": "Snowy",  # Itt is stringként kell küldeni
        "atmospheric_pressure": 1000.0,
        "uv_index": 0.0,
        "season": "Winter",
        "visibility": 3.0,
        "location": "Rural"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"weather_type": "Snowy"}  # Itt a predikciót ellenőrizzük

def test_invalid_input():
    payload = {
        "temperature": "invalid",  # Hibás adat, nem float
        "humidity": 80.0,
        "wind_speed": 10.0,
        "precipitation": 50.0,
        "cloud_cover": "Clear",
        "atmospheric_pressure": 1015.0,
        "uv_index": 5.0,
        "season": "Summer",
        "visibility": 10.0,
        "location": "Urban"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # 422: Unprocessable Entity, ha hibás adatot küldünk
