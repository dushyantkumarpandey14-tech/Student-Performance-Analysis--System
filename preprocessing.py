"""
================================================
 Student Performance Analysis
 File: preprocessing.py
 Handles: Missing values, Encoding, Scaling,
          Outlier Detection
================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")


def load_data(path="data/student_performance.csv"):
    df = pd.read_csv(path)
    print("=" * 50)
    print("STEP 1 — DATA LOADING")
    print("=" * 50)
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\n{df.head()}")
    return df


def handle_missing_values(df):
    print("\n" + "=" * 50)
    print("STEP 2 — MISSING VALUE TREATMENT")
    print("=" * 50)
    print("Before:\n", df.isnull().sum()[df.isnull().sum() > 0])

    # Numerical → Median | Categorical → Mode
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  [Num] '{col}' filled with median")

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  [Cat] '{col}' filled with mode")

    print(f"After : {df.isnull().sum().sum()} missing values")
    return df


def encode_categorical(df):
    print("\n" + "=" * 50)
    print("STEP 3 — ENCODING CATEGORICAL FEATURES")
    print("=" * 50)
    le = LabelEncoder()
    for col in ["gender","parental_education","lunch","test_preparation","pass_status","grade"]:
        if col in df.columns:
            df[col+"_enc"] = le.fit_transform(df[col])
            print(f"  '{col}' encoded")
    return df


def detect_outliers(df):
    print("\n" + "=" * 50)
    print("STEP 4 — OUTLIER DETECTION (IQR Method)")
    print("=" * 50)
    for col in ["attendance","study_hours","math_score","reading_score","writing_score","avg_score"]:
        if col not in df.columns: continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n = ((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum()
        print(f"  '{col}': {n} outliers detected")


def scale_features(df):
    print("\n" + "=" * 50)
    print("STEP 5 — FEATURE SCALING (StandardScaler)")
    print("=" * 50)
    cols = ["attendance","study_hours","math_score","reading_score","writing_score"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols].dropna())
    print(f"  Scaled columns: {cols}")
    print(f"  Mean ≈ 0: {scaled.mean(axis=0).round(3)}")
    return scaler


def run_preprocessing(path="data/student_performance.csv"):
    df = load_data(path)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    detect_outliers(df)
    scaler = scale_features(df)
    df.to_csv("data/cleaned_data.csv", index=False)
    print("\n[OK] Saved: data/cleaned_data.csv")
    return df, scaler


if __name__ == "__main__":
    run_preprocessing()
