# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config, log
from src.data_preprocessing import load_data, preprocess_features
from src.train_model import train_catboost

# Load configuration
config = load_config("config/settings.yaml")

# Step 1: Load dataset
log("Loading dataset...")
df = load_data(config['data']['processed_path'])
log('Dataset loaded successfully.')

# Step 2: Preprocess and split features/target
log("Preprocessing features...")
X, y, scaler = preprocess_features(df)

# Step 3: Train/test split
log("Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config['train_test_split']['test_size'],
    stratify=y if config['train_test_split']['stratify'] else None,
    random_state=config['train_test_split']['random_state']
)

# Step 4: Model training with CatBoost and hyperparameter tuning
log("Starting model training...")
model, metrics = train_catboost(X_train, y_train, X_test, y_test, config['model']['save_path'])

# Step 5: Output performance metrics
log("Training complete. Model performance:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

log("✅ End-to-end ML training pipeline completed successfully.")
