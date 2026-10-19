"""
Machine Learning Model Training for Crop Recommendation
Trains a Random Forest Classifier on the 2,200 agricultural field records dataset.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'crop_recommendation.csv')
    model_dir = os.path.join(base_dir, 'ml')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'crop_model.pkl')

    print(f">> Loading agricultural dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Features: N, P, K, temperature, humidity, ph, rainfall
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    y = df['label']

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f">> Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(">> Training Random Forest Classifier...")

    # We use 100 trees with fixed random state for 100% reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f">> Model Training Complete! Test Accuracy: {accuracy * 100:.2f}%")

    # Save model and feature list
    artifacts = {
        'model': model,
        'features': feature_cols,
        'classes': list(model.classes_),
        'accuracy': round(accuracy * 100, 2)
    }
    joblib.dump(artifacts, model_path)
    print(f">> Saved trained model artifact to: {model_path}")
    return accuracy

if __name__ == '__main__':
    train()
