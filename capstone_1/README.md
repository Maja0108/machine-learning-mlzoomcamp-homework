## Problem Description

Weather classification plays a significant role in helping meteorologists and decision-makers understand weather patterns, enabling them to make predictions and warnings regarding weather-related phenomena. In this project, we focus on predicting weather types such as **Sunny**, **Rainy**, **Cloudy**, and **Snowy** based on a range of environmental and atmospheric factors. These include temperature, humidity, wind speed, and more. The goal is to build a model that can predict the weather type for a given set of input features, which can be used in applications such as weather forecasting, agriculture, or event planning.

The problem at hand involves **multi-class classification**, where the objective is to assign one of the following weather types to the provided data:
- **Sunny**
- **Rainy**
- **Cloudy**
- **Snowy**

---

## How the Model Could Be Used

A classification model can be trained to predict the **Weather Type** based on the provided features. Traditional models like **Logistic Regression** can help identify linear relationships between features and the target variable. However, more complex models such as **Random Forests**, **Gradient Boosting**, or **Neural Networks** can capture intricate, nonlinear relationships and interactions between the features.

By training the model on historical weather data, it can predict future weather types based on various environmental inputs. This capability can be integrated into systems that assist in:
- **Weather forecasting**,
- **Disaster preparedness**,
- **Decision-making in agriculture**,
- **Planning for outdoor events**, and more.

---

## Dataset Description

The model is trained on a weather dataset that contains environmental features along with the corresponding **Weather Type**. 

#### **Features:**
- **Temperature**: Temperature in degrees Celsius.
- **Humidity**: Humidity percentage.
- **Wind Speed**: Wind speed in km/h.
- **Precipitation**: Precipitation in percentage.
- **Cloud Cover**: The type of cloud cover, such as "Clear", "Partly Cloudy", "Cloudy", etc.
- **Atmospheric Pressure**: Atmospheric pressure in hPa.
- **UV Index**: The ultraviolet index.
- **Season**: The season during which the data was recorded.
- **Visibility**: Visibility in kilometers.
- **Location**: Location of the observation, e.g., "Urban", "Rural".

#### **Target variable:**
- **Weather Type**: The weather condition, which can be one of:
  - Sunny
  - Rainy
  - Cloudy
  - Snowy

---

## Exploratory Data Analysis (EDA) Summary

- **Feature Relationships**: Temperature, humidity, and wind speed are expected to show correlations with the weather type, as they directly influence whether it’s sunny, rainy, or snowy.
- **Categorical Features**: Cloud cover, season, and location are categorical and need to be encoded.
- **Missing Values**: Handle missing values appropriately, likely through median imputation for numerical values and mode imputation for categorical ones.
- **Class Distribution**: Ensure the weather types are balanced; if the distribution is skewed, consider techniques like SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

---

## Modeling Approach & Evaluation Metrics

#### **Approach**:
Several classification models are trained and evaluated to predict the **Weather Type**:

| Model                        | Accuracy | F1-Score | Precision | Recall  |
|------------------------------|----------|----------|-----------|---------|
| Logistic Regression           | 84.92%   | 0.86     | 0.85      | 0.87    |
| Decision Tree Classifier      | 90.68%   | 0.91     | 0.90      | 0.93    |
| K-Nearest Neighbors (KNN)     | 88.60%   | 0.89     | 0.88      | 0.89    |
| Support Vector Machine (SVM)  | 82.73%   | 0.83     | 0.84      | 0.81    |
| Naive Bayes Classifier       | 86.29%   | 0.87     | 0.86      | 0.88    |
| Random Forest Classifier      | 91.17%   | 0.92     | 0.91      | 0.93    |
| AdaBoost Classifier           | 87.80%   | 0.88     | 0.89      | 0.88    |
| Gradient Boosting Classifier  | 90.80%   | 0.91     | 0.90      | 0.92    |

**Best Model**: Based on accuracy and other metrics, **Random Forest Classifier** emerged as the best model.

#### **Preprocessing**:
- **Categorical features** like `cloud_cover`, `season`, and `location` are **one-hot encoded**.
- **Numerical features** are **scaled** to ensure they are on a comparable scale for models like SVM or KNN.
- **Missing values** are handled using **imputation** strategies (either median for numerical or mode for categorical).

#### **Evaluation Metrics**:
- **Accuracy**: The proportion of correct predictions.
- **F1-Score**: A balanced measure of precision and recall.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positives that were correctly identified.

---

## How to Run Locally and via Docker

#### **Local Setup**:

1. Clone the repository and navigate to the project directory.
2. Set up the environment:
```
   uv sync
```
3. Train the model (if a pre-trained model isn't available):
```
    uv run python train.py
```
4. Run the FastAPI server:
```
    uv run uvicorn predict:app --reload --host 0.0.0.0 --port 9696
```
5. WebAPI: http://localhost:9696/

Docker Setup:

1. Build the Docker image:
```
    docker build -t weather-prediction -f Dockerfile .
```
2. Run the Docker container:
```
    docker run -it -p 9696:9696 weather-prediction
```
3. Access the API via:
    http://localhost:9696/

Using the API

To make a prediction, send a POST request to /predict with the following JSON payload:
```
curl -X POST "http://localhost:9696/predict" \
-H "Content-Type: application/json" \
-d '{
    "temperature": 22.5,
    "humidity": 78.0,
    "wind_speed": 15.0,
    "precipitation": 20.0,
    "cloud_cover": "Partly Cloudy",  
    "atmospheric_pressure": 1012.0,
    "uv_index": 5.0,
    "season": "Summer",
    "visibility": 10.0,
    "location": "Urban"
}'
```
Example Response:
```
{
  "weather_type": "Sunny"
}
```

Known Limitations and Future Steps:

In some cases Rainy and Snowy classes are not well modelled. Later I try to find the route cause. Samples pronlematic data in test.py to documnet these situations.

Class Imbalance: If the weather types are imbalanced (e.g., many more sunny days than rainy ones), consider using techniques like SMOTE to address this issue.

Feature Engineering: More advanced techniques such as time-series analysis (e.g., hour of day, seasonality) could be incorporated into future models to improve prediction accuracy.

Generalization: The model may struggle to generalize to unseen weather patterns or new locations. Consider retraining periodically with updated data.