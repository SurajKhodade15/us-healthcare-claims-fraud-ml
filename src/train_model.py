# src/train_model.py
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from .evaluate_model import calculate_comprehensive_metrics

def train_catboost(X_train, y_train, X_test, y_test, model_path="models/cat_boost_model.pkl"):
    """Train CatBoost with GridSearch and save best model"""
    pipeline = Pipeline([
        ('model', CatBoostClassifier(random_state=42, verbose=0))
    ])

    params = {
        'model__iterations': [100, 200],
        'model__learning_rate': [0.1, 0.15],
        'model__depth': [5, 7]
    }

    grid = GridSearchCV(pipeline, param_grid=params, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, model_path)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba)
    metrics['Best_Params'] = grid.best_params_

    return best_model, metrics
