Problem Description

Air quality monitoring plays a crucial role in understanding environmental conditions and assessing potential health risks in urban areas. One important pollutant is carbon monoxide (CO), a toxic gas produced primarily by traffic and industrial activity. The AirQualityUCI dataset provides hourly measurements of CO concentration along with various meteorological variables (such as temperature, humidity, and absolute humidity) and other atmospheric pollutants like NO₂ and benzene. The central problem is to determine how these environmental and pollution-related factors influence CO levels and to predict future CO concentrations based on patterns present in historical data. This task is both practically relevant—supporting early warning systems and environmental planning—and analytically interesting due to the dataset’s multivariate and temporal structure.

How a Model Could Be Used

A regression model can be used to estimate CO concentration by learning relationships between CO levels and the dataset’s predictor variables. Traditional linear regression can reveal linear associations between CO and other pollutant metrics, while more advanced models—such as Random Forests, Gradient Boosting, or Neural Networks—can capture nonlinear interactions and temporal dependencies. By training such a model on historical measurements, we can forecast CO levels for future time periods, identify which features contribute most to variations in CO concentration, and evaluate how environmental conditions interact to influence air quality. This predictive capability could support real-time air quality monitoring, allow authorities to anticipate pollution peaks, and inform public health interventions.

EDA of dataset

Some preliminray evaluation: EDA_airquality.ipynb 

Evaluation of the model comparison (modelling.ipynb)

The regression experiments show that the SVR model achieved the best overall performance, obtaining the lowest MSE (0.165) and the highest R² score (0.907). This indicates that the Support Vector Regressor was most effective at capturing the underlying nonlinear relationships in the CO concentration data. The Gradient Boosting model performed similarly well, with an R² of 0.899, demonstrating that ensemble boosting methods also model the pollution patterns accurately.

In contrast, the Random Forest model performed slightly worse, suggesting that while it captures general structure, it may be less sensitive to fine-grained variations in the CO values compared to SVR and boosting. The KNN regressor achieved competitive but lower accuracy, reflecting its limitations in high-dimensional or noisy environmental datasets. Overall, the results highlight that models capable of learning smooth nonlinear boundaries (like SVR and Gradient Boosting) tend to provide the most accurate predictions for CO concentration, making them strong candidates for air-quality forecasting applications.

# CO Concentration Prediction API

This project provides a FastAPI-based web service to predict air quality by estimating carbon monoxide (CO) concentrations using a trained regression model.

---

## Problem Statement

The goal of this project is to predict **CO concentration** in the air based on various environmental and pollutant measurements. This is a regression problem, where the target variable, `CO concentration`, is continuous. Predicting CO levels is important for environmental monitoring and public health.

---

## Dataset Description

The model is trained on the **Air Quality Dataset** from the UCI Machine Learning Repository.  

**Features include**:

- `PT08.S1(CO)` – sensor 1 (CO)  
- `PT08.S2(NMHC)` – sensor 2 (non-methane hydrocarbons)  
- `PT08.S3(NOx)` – sensor 3 (nitrogen oxides)  
- `PT08.S4(NO2)` – sensor 4 (nitrogen dioxide)  
- `PT08.S5(O3)` – sensor 5 (ozone)  
- `T` – temperature  
- `RH` – relative humidity  
- `AH` – absolute humidity  
- `Date` and `Time` – timestamp of measurement  

**Target variable**:

- `CO(GT)` – true CO concentration (in mg/m³)

**Notes on preprocessing**:

- Date and Time columns were combined into a `Datetime` column for easier time-based analysis.  
- Missing values were handled using median imputation.  
- Some sensor measurements were filtered or cleaned to ensure numerical consistency.

---

## Exploratory Data Analysis (EDA) Summary

- CO concentrations are right-skewed, with occasional high spikes.  
- Temperature and humidity show correlations with CO levels.  
- Some sensors are strongly correlated, suggesting redundancy in features.  
- Missing values were present but handled via median imputation to ensure compatibility with regression models.

---

## Modeling Approach & Metrics

### Approach

Several regression models were evaluated to predict CO concentration:

| Model                        | MSE      | R²       |
|-------------------------------|----------|----------|
| Support Vector Regressor (SVR)| 0.165    | 0.907    |
| Random Forest                 | 0.217    | 0.879    |
| Gradient Boosting             | 0.179    | 0.899    |
| K-Nearest Neighbors           | 0.188    | 0.895    |

The **SVR model** provided the best performance in terms of R² and was selected for the final API.

### Preprocessing

- **Numerical features**: median imputation for missing values, scaling with StandardScaler.  
- **Datetime features**: converted into numeric representations if used in modeling.  
- **Pipeline**: SVR model wrapped in a scikit-learn pipeline for clean preprocessing and prediction.

### Metrics

- **Mean Squared Error (MSE)** – measures the average squared difference between predicted and true CO values.  
- **R²** – measures the proportion of variance explained by the model.

---

## How to Run Locally and via Docker

### Prerequisites

- **uv package manager** installed.  
- **Docker** and **docker-compose** (for Docker deployment).

---

### Local Setup

1. Clone the repository and navigate to the project directory.  
2. Set up the environment:

uv sync
3. Train the model (optional, as a pre-trained model is included):
uv run python train.py
4. Run the FastAPI server:
uv run uvicorn predict:app --reload --host 0.0.0.0 --port 9696
5. Access the API:
    Web API: http://localhost:9696

### Docker setup
1. Build Docker image
docker build -t co-prediction -f Dockerfile .

2. Run the Docker container:
docker run -it -p 9696:9696 co-prediction

3. Access API via via http://localhost:9696

### Use of API
POST request to /predict:
´´´
curl -X POST "http://localhost:9696/predict" \
-H "Content-Type: application/json" \
-d '{
    "PT08.S1(CO)": 2.3,
    "PT08.S2(NMHC)": 150,
    "PT08.S3(NOx)": 100,
    "PT08.S4(NO2)": 80,
    "PT08.S5(O3)": 25,
    "T": 20.5,
    "RH": 45,
    "AH": 0.9
}'
´´´
Example Response:
{
  "prediction": 2.87
}

### Known Limitations / Next Steps

Model performance depends heavily on training data quality.

Current handling of missing data is simple; more advanced imputation could improve predictions.

Model is trained on historical data and may not generalize to future trends or new environmental conditions.

Future improvements could include temporal features (hour of day, season) or additional pollutants.