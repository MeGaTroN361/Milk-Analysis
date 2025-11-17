# model_pipeline.py
"""
Train fat regression and disease classification models from a CSV dataset.
Usage:
    python model_pipeline.py --in training_dataset.csv
Outputs:
    - fat_model.pkl
    - disease_model.pkl
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
import sys

def train_from_csv(dataset_csv, save_fat="fat_model.pkl", save_disease="disease_model.pkl", random_state=42):
    if not os.path.exists(dataset_csv):
        print(f"Error: dataset file '{dataset_csv}' not found.")
        sys.exit(1)

    df = pd.read_csv(dataset_csv)
    required_cols = {'temperature','ph','turbidity','conductivity','heart_rate','fat_content','status'}
    missing = required_cols - set(df.columns)
    if missing:
        print("Error: dataset missing required columns:", missing)
        sys.exit(1)

    features = ['temperature','ph','turbidity','conductivity','heart_rate']
    X = df[features]
    y_reg = df['fat_content']
    y_clf = df['status']

    # Fat regressor
    print("Training fat content regressor...")
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.2, random_state=random_state)
    fat_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    fat_model.fit(X_tr_r, y_tr_r)
    preds_r = fat_model.predict(X_te_r)
    mse = mean_squared_error(y_te_r, preds_r)
    rmse = np.sqrt(mse)
    print(f"Fat regressor trained. RMSE: {rmse:.4f}")

    # Disease classifier
    print("Training disease classifier...")
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X, y_clf, test_size=0.2, random_state=random_state, stratify=y_clf)
    disease_model = RandomForestClassifier(n_estimators=250, random_state=random_state, class_weight='balanced')
    disease_model.fit(X_tr_c, y_tr_c)
    acc = accuracy_score(y_te_c, disease_model.predict(X_te_c))
    print(f"Disease classifier trained. Accuracy: {acc:.4f}")

    # Save models
    joblib.dump(fat_model, save_fat)
    joblib.dump(disease_model, save_disease)
    print(f"Saved fat model to '{save_fat}' and disease model to '{save_disease}'")

    # Optional: return objects if called programmatically
    return fat_model, disease_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models from CSV dataset")
    parser.add_argument('--in', dest='infile', type=str, required=True, help='Input CSV dataset file')
    parser.add_argument('--fat_out', type=str, default='fat_model.pkl', help='Output filename for fat model')
    parser.add_argument('--disease_out', type=str, default='disease_model.pkl', help='Output filename for disease model')
    parser.add_argument('--seed', type=int, default=42, help='Random state for reproducibility')
    args = parser.parse_args()

    train_from_csv(args.infile, save_fat=args.fat_out, save_disease=args.disease_out, random_state=args.seed)
