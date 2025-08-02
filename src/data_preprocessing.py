# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from given path with error handling
    
    Args:
        path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
        Exception: For other loading errors
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
            
        logger.info(f"Loading data from {path}...")
        df = pd.read_csv(path)
        
        if df.empty:
            raise pd.errors.EmptyDataError(f"Dataset is empty: {path}")
            
        logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise

def preprocess_features(df: pd.DataFrame, target_column: str = 'Is_Fraudulent') -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Split features and target, apply scaling with proper error handling
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        
    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: X_scaled, y, scaler
        
    Raises:
        KeyError: If target column doesn't exist
        ValueError: If no features remain after preprocessing
    """
    try:
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset")
            
        logger.info("Starting feature preprocessing...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            logger.info(f"Encoding {len(categorical_columns)} categorical columns: {list(categorical_columns)}")
            
            # Use label encoding for categorical variables
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                
        # Check for remaining features
        if X.shape[1] == 0:
            raise ValueError("No features remaining after preprocessing")
            
        # Apply scaling
        logger.info("Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"✅ Preprocessing completed. Features: {X_scaled.shape[1]}, Samples: {X_scaled.shape[0]}")
        return X_scaled, y.values, scaler
        
    except KeyError as e:
        logger.error(f"Column error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in preprocessing: {e}")
        raise

def preprocess_input(input_data: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> pd.DataFrame:
    """
    Preprocess input data for prediction with error handling
    
    Args:
        input_data (pd.DataFrame): Input data to preprocess
        scaler (Optional[StandardScaler]): Pre-fitted scaler
        
    Returns:
        pd.DataFrame: Preprocessed input data
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        if input_data.empty:
            raise ValueError("Input data is empty")
            
        logger.info(f"Preprocessing input data with shape: {input_data.shape}")
        
        # Handle categorical variables (same as training)
        categorical_columns = input_data.select_dtypes(include=['object']).columns
        input_processed = input_data.copy()
        
        if len(categorical_columns) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_columns)}")
            
            for col in categorical_columns:
                # Simple label encoding (in production, should use the same encoder from training)
                le = LabelEncoder()
                input_processed[col] = le.fit_transform(input_processed[col].astype(str))
        
        # Apply scaling if scaler provided
        if scaler is not None:
            logger.info("Applying scaling to input data...")
            input_scaled = scaler.transform(input_processed)
            return pd.DataFrame(input_scaled, columns=input_processed.columns)
        else:
            # If no scaler provided, use StandardScaler
            logger.warning("No scaler provided, creating new StandardScaler")
            new_scaler = StandardScaler()
            input_scaled = new_scaler.fit_transform(input_processed)
            return pd.DataFrame(input_scaled, columns=input_processed.columns)
            
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

def validate_dataset(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """
    Validate dataset integrity
    
    Args:
        df (pd.DataFrame): Dataset to validate
        required_columns (Optional[list]): Required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if df.empty:
            logger.error("Dataset is empty")
            return False
            
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        
        if not high_missing.empty:
            logger.warning(f"Columns with >50% missing values: {list(high_missing.index)}")
            
        logger.info("✅ Dataset validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return False