import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading datasets...")
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found. Please place them in the directory or run generate_dummy_data.py")
        return

    # Assuming Target variable is 'Class' (standard for Kaggle Credit Card Fraud dataset)
    if 'Class' not in train_df.columns:
        print("Error: Target column 'Class' missing from datasets.")
        return

    # Handle missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Drop unnecessary columns if required (Time might not be highly predictive but often kept; we can keep it as part of features)
    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']
    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']

    print("Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler immediately
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Handling Class Imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("Training Models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42) # reduced n_estimators for speed
    }

    results = {}
    best_model_name = None
    best_recall = -1
    best_model = None

    for name, model in models.items():
        print(f"--> Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on Test data ONLY
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-score': round(f1, 4),
            'Confusion_Matrix': cm
        }
        
        # Primary comparison metric is Recall, as specified
        if rec > best_recall:
            best_recall = rec
            best_model_name = name
            best_model = model

    print(f"\nTraining complete! Best Model Selected: {best_model_name} (Recall: {best_recall:.4f})")

    # Save the best model using pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    # Save results to a json file for Streamlit to consume
    results['Best_Model'] = best_model_name
    results['Features'] = X_train.columns.tolist()  # Save feature definitions for dynamic form rendering
    
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Files saved: model.pkl, scaler.pkl, model_results.json")

if __name__ == "__main__":
    main()
