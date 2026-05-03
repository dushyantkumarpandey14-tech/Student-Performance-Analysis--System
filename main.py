"""
================================================
 Student Performance Analysis
 File: main.py
 RUN THIS FILE — it runs everything automatically
================================================
"""

import os

# Create folders
os.makedirs("data",   exist_ok=True)
os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

print("╔══════════════════════════════════════════════╗")
print("║  STUDENT PERFORMANCE ANALYSIS SYSTEM        ║")
print("║  BTech 3rd Year | 6th Semester              ║")
print("║  Tools: Python, Pandas, Matplotlib,         ║")
print("║         Seaborn, Scikit-learn               ║")
print("╚══════════════════════════════════════════════╝\n")

# STEP 1 — Generate Dataset
print(">>> STEP 1: Generating Dataset...")
from generate_dataset import generate_dataset
df_raw = generate_dataset()
df_raw.to_csv("data/student_performance.csv", index=False)
print(f"    {df_raw.shape[0]} students, {df_raw.shape[1]} features\n")

# STEP 2 — Preprocessing
print(">>> STEP 2: Preprocessing Data...")
from preprocessing import run_preprocessing
df, scaler = run_preprocessing()

# STEP 3 — EDA
print("\n>>> STEP 3: Generating Charts (EDA)...")
from eda import run_eda
run_eda(df)

# STEP 4 — ML Models
print("\n>>> STEP 4: Training ML Models...")
from models import run_models
run_models(df)

print("\n" + "="*50)
print("  ALL DONE!")
print("="*50)
print("  Plots  → plots/  (10 PNG files)")
print("  Models → models/ (.pkl files)")
print("  Data   → data/   (.csv files)")
print("\n  To open web dashboard:")
print("  → streamlit run app.py")
