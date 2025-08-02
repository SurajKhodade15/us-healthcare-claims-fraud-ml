# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from given path"""
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def preprocess_features(df: pd.DataFrame):
    """Split features and target, apply scaling"""
    X = df.drop(columns=['Is_Fraudulent'])
    y = df['Is_Fraudulent']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    # Assuming input_data has the same structure as training data
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    return pd.DataFrame(input_scaled, columns=input_data.columns)