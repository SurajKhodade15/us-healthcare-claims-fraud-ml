# src/evaluate_model.py
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics with error handling
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray): Prediction probabilities
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics
        
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        # Validate inputs
        if len(y_true) == 0 or len(y_pred) == 0 or len(y_proba) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
            raise ValueError("Input arrays must have the same length")
            
        if not all(isinstance(x, (int, float, np.integer, np.floating)) for x in y_true):
            raise ValueError("y_true must contain only numeric values")
            
        logger.info("Calculating comprehensive metrics...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle case where confusion matrix might not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            logger.warning("Unusual confusion matrix shape, using fallback calculations")
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate basic metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_proba),
            'PR_AUC': average_precision_score(y_true, y_proba),
        }
        
        # Calculate advanced metrics
        metrics.update({
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'False_Positive_Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'False_Negative_Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'True_Positives': int(tp),
            'True_Negatives': int(tn),
            'False_Positives': int(fp),
            'False_Negatives': int(fn),
            'Total_Samples': len(y_true),
            'Positive_Samples': int(np.sum(y_true == 1)),
            'Negative_Samples': int(np.sum(y_true == 0))
        })
        
        # Calculate balanced accuracy
        metrics['Balanced_Accuracy'] = (metrics['Sensitivity'] + metrics['Specificity']) / 2
        
        # Calculate MCC (Matthews Correlation Coefficient)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['MCC'] = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0
        
        logger.info(f"✅ Metrics calculated successfully")
        logger.info(f"ROC AUC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
        
        return metrics
        
    except ValueError as e:
        logger.error(f"Validation error in metrics calculation: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def generate_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_proba: np.ndarray, save_dir: str = "plots") -> None:
    """
    Generate and save evaluation plots
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray): Prediction probabilities
        save_dir (str): Directory to save plots
    """
    try:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # 1. Confusion Matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Fraud', 'Fraud'],
                   yticklabels=['Non-Fraud', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. ROC Curve
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✅ Evaluation plots saved to {save_dir}/model_evaluation.png")
        
    except Exception as e:
        logger.error(f"Error generating evaluation plots: {e}")

def print_classification_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted classification summary
    
    Args:
        metrics (Dict[str, Any]): Evaluation metrics dictionary
    """
    try:
        print("\n" + "="*60)
        print("🎯 MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"📊 Overall Performance:")
        print(f"   Accuracy:           {metrics.get('Accuracy', 0):.4f}")
        print(f"   Balanced Accuracy:  {metrics.get('Balanced_Accuracy', 0):.4f}")
        print(f"   ROC AUC:           {metrics.get('ROC_AUC', 0):.4f}")
        print(f"   PR AUC:            {metrics.get('PR_AUC', 0):.4f}")
        
        print(f"\n🎯 Classification Metrics:")
        print(f"   Precision:         {metrics.get('Precision', 0):.4f}")
        print(f"   Recall (Sensitivity): {metrics.get('Recall', 0):.4f}")
        print(f"   F1-Score:          {metrics.get('F1_Score', 0):.4f}")
        print(f"   Specificity:       {metrics.get('Specificity', 0):.4f}")
        
        print(f"\n📈 Confusion Matrix Breakdown:")
        print(f"   True Positives:    {metrics.get('True_Positives', 0)}")
        print(f"   True Negatives:    {metrics.get('True_Negatives', 0)}")
        print(f"   False Positives:   {metrics.get('False_Positives', 0)}")
        print(f"   False Negatives:   {metrics.get('False_Negatives', 0)}")
        
        print(f"\n🔍 Error Analysis:")
        print(f"   False Positive Rate: {metrics.get('False_Positive_Rate', 0):.4f}")
        print(f"   False Negative Rate: {metrics.get('False_Negative_Rate', 0):.4f}")
        print(f"   Matthews Correlation: {metrics.get('MCC', 0):.4f}")
        
        # Performance interpretation
        auc = metrics.get('ROC_AUC', 0)
        if auc >= 0.9:
            performance = "Excellent"
        elif auc >= 0.8:
            performance = "Good"
        elif auc >= 0.7:
            performance = "Fair"
        else:
            performance = "Poor"
            
        print(f"\n✅ Overall Assessment: {performance} (AUC: {auc:.3f})")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error printing classification summary: {e}")

def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        y_proba: np.ndarray) -> bool:
    """
    Validate prediction arrays for consistency
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels  
        y_proba (np.ndarray): Prediction probabilities
        
    Returns:
        bool: True if validation passes
    """
    try:
        # Check array lengths
        if not (len(y_true) == len(y_pred) == len(y_proba)):
            logger.error("Prediction arrays have different lengths")
            return False
            
        # Check value ranges
        if not all(label in [0, 1] for label in np.unique(y_true)):
            logger.error("y_true contains values other than 0 and 1")
            return False
            
        if not all(label in [0, 1] for label in np.unique(y_pred)):
            logger.error("y_pred contains values other than 0 and 1")
            return False
            
        if not all(0 <= prob <= 1 for prob in y_proba):
            logger.error("y_proba contains values outside [0, 1] range")
            return False
            
        logger.info("✅ Prediction validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating predictions: {e}")
        return False
