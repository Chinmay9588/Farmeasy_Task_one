# model_train.py
"""
Train pipeline for Chickpea disease classification.
Assumes target column name is 'Disease'. If different, edit TARGET_NAME below.

Usage:
    python model_train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

CSV_PATH = "Chickpea_Crop_Disease_With_Realistic_Sowing_Date.csv"
MODEL_OUT = "model_pipeline.joblib"
LABEL_ENCODER_OUT = "label_encoder.joblib"

# ---- CONFIG: change if needed ----
TARGET_NAME = "Disease"   # <-- If your CSV's target column has another name, change this.
# ----------------------------------

def add_days_since_sowing(df):
    """If CSV has a sowing date column, create days_since_sowing numeric column."""
    sow_cols = [c for c in df.columns if 'sow' in c.lower() or 'sowing' in c.lower()]
    if not sow_cols:
        return df
    sow_col = sow_cols[0]
    # try to find an explicit current_date column
    cur_cols = [c for c in df.columns if 'current' in c.lower() or 'date' in c.lower() and c != sow_col]
    cur_col = cur_cols[0] if cur_cols else None

    def parse_date(val):
        if pd.isna(val): return None
        for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(str(val), fmt)
            except:
                continue
        try:
            return pd.to_datetime(val)
        except:
            return None

    days = []
    for _, row in df.iterrows():
        s = parse_date(row[sow_col])
        if cur_col:
            c = parse_date(row[cur_col])
        else:
            c = datetime.now()
        if s is None or c is None:
            days.append(np.nan)
        else:
            days.append((c - s).days)
    df['days_since_sowing'] = days
    return df

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Put your CSV in the project root.")

    df = pd.read_csv(CSV_PATH)
    print("Loaded CSV. Columns:", df.columns.tolist())
    print(df.head())

    # create days_since_sowing if sowing date present
    df = add_days_since_sowing(df)

    if TARGET_NAME not in df.columns:
        raise KeyError(f"Target '{TARGET_NAME}' not found in CSV. Edit TARGET_NAME in this file accordingly.")

    # drop rows with missing target
    df = df.dropna(subset=[TARGET_NAME]).reset_index(drop=True)

    # Features are everything except target
    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME].astype(str)  # treat labels as strings

    # Keep feature columns that are numeric or object (we will encode object)
    # Drop overly verbose text columns if any (like long notes) — user can adjust
    # We select numeric and object columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    object_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Remove original date/sowing columns if days_since_sowing exists
    # Identify original date-like columns (exclude the derived 'days_since_sowing')
    date_like = [c for c in X.columns if (('date' in c.lower() or 'sow' in c.lower()) and c != 'days_since_sowing')]
    if 'days_since_sowing' in X.columns:
        for c in date_like:
            if c in numeric_cols: numeric_cols.remove(c)
            if c in object_cols: object_cols.remove(c)
            if c in X.columns:
                X = X.drop(columns=[c])

    # Ensure days_since_sowing included if created
    if 'days_since_sowing' in df.columns and 'days_since_sowing' not in numeric_cols:
        numeric_cols.append('days_since_sowing')

    print("Numeric features:", numeric_cols)
    print("Categorical features:", object_cols)

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, object_cols)
        ], remainder='drop'
    )

    # Label encode y
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, LABEL_ENCODER_OUT)
    print(f"Saved label encoder to {LABEL_ENCODER_OUT}")

    # Build full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    pipeline.fit(X_train, y_train)

    # Evaluation
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = pipeline.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Save pipeline
    joblib.dump(pipeline, MODEL_OUT)
    print(f"Saved model pipeline to {MODEL_OUT}")

if __name__ == "__main__":
    main()
