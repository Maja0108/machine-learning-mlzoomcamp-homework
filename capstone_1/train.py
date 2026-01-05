import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

# ----------------------------
# Load data
# ----------------------------
def load_data():
    df = pd.read_csv("./weather_classification_data.csv")
    encoded_df = df.copy()

    # Encode categorical variables
    encoders = {}  # Dictionary to store the encoders
    cat_col = ['Cloud Cover', 'Season', 'Location']  # List of categorical columns
    for var in cat_col:
        encode = LabelEncoder()
        encoded_df[var] = encode.fit_transform(df[var])  # Fit and transform on the data
        encoders[var] = encode  # Save the encoder for later use

    return encoded_df, encoders

# ----------------------------
# Train Random Forest model
# ----------------------------
def train_model(df):
    # Choose target column
    X = df.drop(columns=['Weather Type'])
    y = df['Weather Type']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest with pipeline
    pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),  # Fill missing values
        StandardScaler(),                         # Standardize the features
        RandomForestClassifier(max_depth=10, 
                               min_samples_leaf=4, 
                               min_samples_split=2, 
                               n_estimators=50, random_state=42)  # Random Forest model
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    return pipeline

# ----------------------------
# Save model and encoders
# ----------------------------
def save_model(pipeline, encoders, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump((pipeline, encoders), f_out)
    print(f"Model and encoders saved to {filename}")

# ----------------------------
# Main
# ----------------------------
df, encoders = load_data()  # Load and encode data
pipeline = train_model(df)

# Save the model and the encoders separately
save_model(pipeline, encoders, 'random_forest_model_pipeline.pkl')
