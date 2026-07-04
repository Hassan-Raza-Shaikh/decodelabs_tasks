import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading highly imbalanced dataset from CSV...")
    df = pd.read_csv('fraud_dataset.csv')
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Class 0 (Legitimate): {np.sum(y==0)}, Class 1 (Fraud): {np.sum(y==1)}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set Fraud cases: {np.sum(y_train==1)}")
    print(f"Test set Fraud cases: {np.sum(y_test==1)}")

    # 1. Logistic Regression Pipeline with SMOTE
    print("\n--- Training Logistic Regression ---")
    lr_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Hyperparameter tuning for Logistic Regression
    lr_param_grid = {
        'lr__C': [0.01, 0.1, 1, 10]
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    
    print("Best LR parameters:", lr_grid.best_params_)
    
    # 2. Random Forest Pipeline with SMOTE
    print("\n--- Training Random Forest ---")
    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter tuning for Random Forest
    # Keeping the grid slightly small for faster execution
    rf_param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [10, 20, None]
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    print("Best RF parameters:", rf_grid.best_params_)

    # 3. Evaluation (Discard "Accuracy", use Strict Precision, Recall, and ROC-AUC)
    print("\n--- Model Evaluation ---")
    
    models = {
        "Logistic Regression": lr_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Discarding accuracy intentionally per requirements
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n[{name}] Performance Metrics:")
        print(f"Precision: {precision:.4f}  (Out of all predicted fraud, how many were actually fraud?)")
        print(f"Recall:    {recall:.4f}  (Out of all actual fraud, how many were detected?)")
        print(f"ROC-AUC:   {roc_auc:.4f}  (Model's ability to distinguish between classes)")

if __name__ == "__main__":
    main()
