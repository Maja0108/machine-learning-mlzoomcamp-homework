Problem Description

Air quality monitoring plays a crucial role in understanding environmental conditions and assessing potential health risks in urban areas. One important pollutant is carbon monoxide (CO), a toxic gas produced primarily by traffic and industrial activity. The AirQualityUCI dataset provides hourly measurements of CO concentration along with various meteorological variables (such as temperature, humidity, and absolute humidity) and other atmospheric pollutants like NO₂ and benzene. The central problem is to determine how these environmental and pollution-related factors influence CO levels and to predict future CO concentrations based on patterns present in historical data. This task is both practically relevant—supporting early warning systems and environmental planning—and analytically interesting due to the dataset’s multivariate and temporal structure.

How a Model Could Be Used

A regression model can be used to estimate CO concentration by learning relationships between CO levels and the dataset’s predictor variables. Traditional linear regression can reveal linear associations between CO and other pollutant metrics, while more advanced models—such as Random Forests, Gradient Boosting, or Neural Networks—can capture nonlinear interactions and temporal dependencies. By training such a model on historical measurements, we can forecast CO levels for future time periods, identify which features contribute most to variations in CO concentration, and evaluate how environmental conditions interact to influence air quality. This predictive capability could support real-time air quality monitoring, allow authorities to anticipate pollution peaks, and inform public health interventions.

EDA of dataset

Some preliminray evaluation: EDA_airquality.ipynb 

Evaluation of the model comparison (modelling.ipynb)

The regression experiments show that the SVR model achieved the best overall performance, obtaining the lowest MSE (0.165) and the highest R² score (0.907). This indicates that the Support Vector Regressor was most effective at capturing the underlying nonlinear relationships in the CO concentration data. The Gradient Boosting model performed similarly well, with an R² of 0.899, demonstrating that ensemble boosting methods also model the pollution patterns accurately.

In contrast, the Random Forest model performed slightly worse, suggesting that while it captures general structure, it may be less sensitive to fine-grained variations in the CO values compared to SVR and boosting. The KNN regressor achieved competitive but lower accuracy, reflecting its limitations in high-dimensional or noisy environmental datasets. Overall, the results highlight that models capable of learning smooth nonlinear boundaries (like SVR and Gradient Boosting) tend to provide the most accurate predictions for CO concentration, making them strong candidates for air-quality forecasting applications.