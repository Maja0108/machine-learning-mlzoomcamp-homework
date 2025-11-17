import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import pickle
import zipfile
from io import BytesIO
import requests

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')

# ----------------------------
# Load AirQualityUCI data
# ----------------------------
def load_data():
    # Data download
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(BytesIO(r.content))

    # Opening CSV
    with z.open('AirQualityUCI.csv') as f:
        df = pd.read_csv(f, sep=';', decimal=',', engine='python')
    df = df.iloc[:, :-2]  # delete 2 empty columns

    # Change -200 to NAN
    df.replace(-200, pd.NA, inplace=True)

    # Choose numeric columns
    numeric_cols = df.columns[2:]  # Leave out Date and Time
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Change missing values to median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S').dt.hour  

    # New feaures
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Delete original Date column
    df = df.drop(columns=['Date'])
    return df

# ----------------------------
# Train SVR model
# ----------------------------
def train_model(df):
    # Target: CO(GT)
    feature_cols = df.columns.drop('CO(GT)')
    X = df[feature_cols].values  # numerikus mátrix
    y = df['CO(GT)'].values

    pipeline = make_pipeline(
    SimpleImputer(strategy='median'),   # NaN -> median
    StandardScaler(),                    # scaler
    SVR(C=100, epsilon=0.1, gamma='scale')  # SVR
)
    pipeline.fit(X, y)
    return pipeline

# ----------------------------
# Save model
# ----------------------------
def save_model(pipeline, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

# ----------------------------
# Main
# ----------------------------
df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'svr_model_pipeline.pkl')

print("SVR model saved to svr_model_pipeline.pkl")
