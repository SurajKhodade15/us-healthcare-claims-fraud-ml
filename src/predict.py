# src/predict.py
import joblib
import numpy as np

def load_model(model_path="models/cat_boost_model.pkl"):
    return joblib.load(model_path)

def predict_new(model, X_new):
    return model.predict(X_new), model.predict_proba(X_new)[:, 1]
