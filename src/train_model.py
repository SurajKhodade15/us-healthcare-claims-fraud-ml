# src/train_model.py
import joblib
import logging
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import numpy as np

from .evaluate_model import calculate_comprehensive_metrics
from .utils import create_directory, log

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_catboost(X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray, 
                  model_path: str = "models/cat_boost_model.pkl") -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train CatBoost with GridSearch and save best model with comprehensive error handling
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray): Test features  
        y_test (np.ndarray): Test target
        model_path (str): Path to save the trained model
        
    Returns:
        Tuple[Pipeline, Dict[str, Any]]: Best model and metrics
        
    Raises:
        ValueError: If training data is invalid
        Exception: For other training errors
    """
    try:
        # Validate inputs
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training data is empty")
            
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Training features and target have different lengths")
            
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("Test features and target have different lengths")
            
        logger.info(f"Starting CatBoost training...")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if model_dir and not create_directory(model_dir):
            raise Exception(f"Failed to create model directory: {model_dir}")
        
        # Define pipeline
        pipeline = Pipeline([
            ('model', CatBoostClassifier(
                random_state=42, 
                verbose=0,
                eval_metric='AUC',
                early_stopping_rounds=50
            ))
        ])

        # Define parameter grid
        params = {
            'model__iterations': [100, 200],
            'model__learning_rate': [0.1, 0.15],
            'model__depth': [5, 7]
        }

        logger.info(f"Starting GridSearchCV with {len(params['model__iterations']) * len(params['model__learning_rate']) * len(params['model__depth'])} parameter combinations...")
        
        # Perform grid search
        grid = GridSearchCV(
            pipeline, 
            param_grid=params, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
            scoring='roc_auc', 
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        logger.info(f"✅ Grid search completed. Best score: {grid.best_score_:.4f}")
        logger.info(f"Best parameters: {grid.best_params_}")

        # Get best model
        best_model = grid.best_estimator_
        
        # Save model
        joblib.dump(best_model, model_path)
        logger.info(f"✅ Model saved to: {model_path}")

        # Make predictions
        logger.info("Generating predictions on test set...")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        logger.info("Calculating performance metrics...")
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba)
        metrics['Best_Params'] = grid.best_params_
        metrics['Best_CV_Score'] = grid.best_score_
        metrics['Training_Samples'] = X_train.shape[0]
        metrics['Test_Samples'] = X_test.shape[0]
        metrics['Features'] = X_train.shape[1]

        logger.info("✅ Model training completed successfully!")
        return best_model, metrics
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def train_simple_catboost(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray,
                         model_path: str = "models/cat_boost_model.pkl") -> Tuple[CatBoostClassifier, Dict[str, Any]]:
    """
    Train a simple CatBoost model without grid search (faster for development)
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target
        model_path (str): Path to save the model
        
    Returns:
        Tuple[CatBoostClassifier, Dict[str, Any]]: Model and metrics
    """
    try:
        logger.info("Training simple CatBoost model...")
        
        # Create model directory
        model_dir = os.path.dirname(model_path)
        if model_dir:
            create_directory(model_dir)
        
        # Train model with default parameters
        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=0,
            eval_metric='AUC'
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"✅ Simple model saved to: {model_path}")
        
        # Generate predictions and metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba)
        metrics['Model_Type'] = 'Simple CatBoost'
        metrics['Training_Samples'] = X_train.shape[0]
        metrics['Test_Samples'] = X_test.shape[0]
        
        logger.info("✅ Simple model training completed!")
        return model, metrics
        
    except Exception as e:
        logger.error(f"Simple training failed: {e}")
        raise

def validate_model_performance(metrics: Dict[str, Any], min_auc: float = 0.7) -> bool:
    """
    Validate if model performance meets minimum requirements
    
    Args:
        metrics (Dict[str, Any]): Model performance metrics
        min_auc (float): Minimum required AUC score
        
    Returns:
        bool: True if model meets requirements
    """
    try:
        auc_score = metrics.get('ROC_AUC', 0)
        
        if auc_score < min_auc:
            logger.warning(f"Model AUC ({auc_score:.3f}) below minimum threshold ({min_auc})")
            return False
            
        logger.info(f"✅ Model performance validation passed (AUC: {auc_score:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"Error validating model performance: {e}")
        return False
