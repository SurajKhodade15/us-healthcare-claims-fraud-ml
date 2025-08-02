# src/utils.py
import yaml
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with error handling
    
    Args:
        path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            
        logger.info(f"Configuration loaded successfully from {path}")
        return config
        
    except FileNotFoundError as e:
        logger.error(f"Config file error: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise

def log(message: str, level: str = "INFO") -> None:
    """
    Enhanced logging function with different levels
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR, DEBUG)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level = level.upper()
    
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)
    else:
        logger.info(message)
    
    print(f"[{level}] {timestamp} - {message}")

def validate_file_path(file_path: str) -> bool:
    """
    Validate if file path exists and is accessible
    
    Args:
        file_path (str): Path to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        return os.path.exists(file_path) and os.path.isfile(file_path)
    except Exception as e:
        logger.error(f"Error validating file path {file_path}: {e}")
        return False

def create_directory(directory_path: str) -> bool:
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path (str): Directory path to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory ensured: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False
