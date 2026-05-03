"""
================================================
 Student Performance Analysis
 File: models.py
 Models: Linear Regression, Decision Tree,
         Random Forest
 Targets: avg_score (regression)
          pass_status (classification)
================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, classification_report, confusion_matrix)
import pickle

BG   = "#0a0f1e"
CARD = "#111827"
plt.style.use("dark_background")


def prepare(df):
    """Feature engineering + train/test split."""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)

    df = df.copy()

    # New features
    df["total_score"]   = df["math_score"] + df["reading_score"] + df["writing_score"]
    df["verbal_score"]  = (df["reading_score"] + df["writing_score"]) / 2
    df["attend_group"]  = pd.cut(df["attendance"],
                                 bins=[0,60,75,85,100], labels=[0,1,2,3]).astype(float)

    print("  Created: total_score, verbal_score, attend_group")

    # Encode
    le = LabelEncoder()
    df["gender_enc"]   = le.fit_transform(df["gender"])
    df["lunch_enc"]    = le.fit_transform(df["lunch"])
    df["prep_enc"]     = le.fit_transform(df["test_preparation"])
    df["pass_enc"]     = (df["pass_status"] == "Pass").astype(int)
    edu_map = {"some high school":0,"high school":1,"some college":2,
               "associate's degree":3,"bachelor's degree":4,"master's degree":5}
    df["edu_enc"] = df["parental_education"].map(edu_map)

    FEATURES = ["attendance","study_hours","assignments","gender_enc",
                "lunch_enc","prep_enc","edu_enc","attend_group"]

    df = df.dropna(subset=FEATURES + ["avg_score","pass_enc"])
    X  = df[FEATURES]
    yr = df["avg_score"]
    yc = df["pass_enc"]

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, yr, test_size=0.25, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(X, yc, test_size=0.25, random_state=42)

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    print(f"  Train: {len(X_tr)} | Test: {len(X_te)} | Features: {FEATURES}")
    return X_tr, X_te, X_tr_s, X_te_s, yr_tr, yr_te, yc_tr, yc_te, scaler, FEATURES


def model_linear_regression(X_tr_s, X_te_s, yr_tr, yr_te):
    print("\n" + "="*50)
    print("MODEL 1 — LINEAR REGRESSION")
    print("="*50)
    m = LinearRegression()
    m.fit(X_tr_s, yr_tr)
    p = m.predict(X_te_s)
    rmse = np.sqrt(mean_squared_error(yr_te, p))
    mae  = mean_absolute_error(yr_te, p)
    r2   = r2_score(yr_te, p)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return m, p, {"rmse":rmse,"mae":mae,"r2":r2}


def model_decision_tree(X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te):
    print("\n" + "="*50)
    print("MODEL 2 — DECISION TREE")
    print("="*50)

    # Regression
    dtr = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
    dtr.fit(X_tr, yr_tr)
    pr  = dtr.predict(X_te)
    rmse= np.sqrt(mean_squared_error(yr_te, pr))
    r2  = r2_score(yr_te, pr)
    print(f"  [Regression]     RMSE={rmse:.4f}  R²={r2:.4f}")

    # Classification
    dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
    dtc.fit(X_tr, yc_tr)
    pc  = dtc.predict(X_te)
    acc = accuracy_score(yc_te, pc)
    print(f"  [Classification] Accuracy = {acc*100:.2f}%")
    print(classification_report(yc_te, pc, target_names=["Fail","Pass"]))

    return dtr, dtc, pr, pc, {"rmse":rmse,"r2":r2}, {"accuracy":acc}


def model_random_forest(X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, features):
    print("\n" + "="*50)
    print("MODEL 3 — RANDOM FOREST (Best Model)")
    print("="*50)

    # Regression
    rfr = RandomForestRegressor(n_estimators=100, max_depth=8,
                                min_samples_leaf=3, random_state=42)
    rfr.fit(X_tr, yr_tr)
    pr  = rfr.predict(X_te)
    rmse= np.sqrt(mean_squared_error(yr_te, pr))
    r2  = r2_score(yr_te, pr)
    print(f"  [Regression]     RMSE={rmse:.4f}  R²={r2:.4f}")

    fi = pd.Series(rfr.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n  Feature Importances:\n{fi.round(4).to_string()}")

    # Classification
    rfc = RandomForestClassifier(n_estimators=100, max_depth=6,
                                  min_samples_leaf=3, random_state=42)
    rfc.fit(X_tr, yc_tr)
    pc  = rfc.predict(X_te)
    acc = accuracy_score(yc_te, pc)
    print(f"\n  [Classification] Accuracy = {acc*100:.2f}%")
    print(classification_report(yc_te, pc, target_names=["Fail","Pass"]))

    return rfr, rfc, pr, pc, {"rmse":rmse,"r2":r2}, {"accuracy":acc}, fi


def plot_comparison(lr_m, dt_m, rf_m, yr_te, lr_p, dt_p, rf_p, fi):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG)
    fig.suptitle("Model Comparison & Evaluation", fontsize=15, fontweight="bold", color="white")
    axes = axes.flatten()
    clrs = ["#00d4ff","#7c3aed","#10b981"]

    # R² bar
    ax = axes[0]; ax.set_facecolor(CARD)
    r2s = [lr_m["r2"], dt_m["r2"], rf_m["r2"]]
    bars = ax.bar(["Linear Reg","Decision Tree","Random Forest"], r2s, color=clrs, edgecolor="white", lw=0.5)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{v:.4f}", ha="center", color="white", fontweight="bold")
    ax.set_title("R² Score (Higher = Better)", color="white", fontweight="bold")
    ax.set_ylim(0, 1.1); ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # RMSE bar
    ax = axes[1]; ax.set_facecolor(CARD)
    rmses = [lr_m["rmse"], dt_m["rmse"], rf_m["rmse"]]
    bars = ax.bar(["Linear Reg","Decision Tree","Random Forest"], rmses, color=clrs, edgecolor="white", lw=0.5)
    for b, v in zip(bars, rmses):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f"{v:.3f}", ha="center", color="white", fontweight="bold")
    ax.set_title("RMSE (Lower = Better)", color="white", fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # Feature importance
    ax = axes[2]; ax.set_facecolor(CARD)
    clr_fi = plt.cm.cool(np.linspace(0.2, 0.9, len(fi)))
    fi.plot(kind="barh", ax=ax, color=clr_fi, edgecolor="white", lw=0.4)
    ax.set_title("Feature Importance (Random Forest)", color="white", fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # Actual vs Predicted — 3 models
    for ax, pred, name, color in zip(axes[3:],
        [lr_p, dt_p, rf_p],
        ["Linear Regression","Decision Tree","Random Forest"], clrs):
        ax.set_facecolor(CARD)
        ax.scatter(yr_te, pred, alpha=0.6, color=color, s=25)
        lims = [min(yr_te.min(), pred.min()), max(yr_te.max(), pred.max())]
        ax.plot(lims, lims, "w--", lw=1.2, label="Perfect")
        ax.set_title(f"{name}: Actual vs Predicted", color="white", fontweight="bold", fontsize=10)
        ax.set_xlabel("Actual", color="#94a3b8", fontsize=9)
        ax.set_ylabel("Predicted", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.legend(fontsize=7, labelcolor="white", facecolor=CARD)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/10_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[SAVED] plots/10_model_comparison.png")


def print_summary(lr_m, dt_m, rf_m, dt_c, rf_c):
    print("\n" + "="*60)
    print(" FINAL MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<22} {'RMSE':>8} {'R²':>8} {'Accuracy':>10}")
    print("-"*60)
    print(f"{'Linear Regression':<22} {lr_m['rmse']:>8.4f} {lr_m['r2']:>8.4f} {'N/A':>10}")
    print(f"{'Decision Tree':<22} {dt_m['rmse']:>8.4f} {dt_m['r2']:>8.4f} {dt_c['accuracy']*100:>9.2f}%")
    print(f"{'Random Forest ★':<22} {rf_m['rmse']:>8.4f} {rf_m['r2']:>8.4f} {rf_c['accuracy']*100:>9.2f}%")
    print("="*60)
    print("  ★ BEST: Random Forest — best overall performance")


def run_models(df):
    (X_tr, X_te, X_tr_s, X_te_s,
     yr_tr, yr_te, yc_tr, yc_te,
     scaler, features) = prepare(df)

    lr,  lr_p,  lr_m                           = model_linear_regression(X_tr_s, X_te_s, yr_tr, yr_te)
    dtr, dtc, dt_p, dt_cp, dt_m, dt_c          = model_decision_tree(X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te)
    rfr, rfc, rf_p, rf_cp, rf_m, rf_c, fi      = model_random_forest(X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, features)

    plot_comparison(lr_m, dt_m, rf_m, yr_te, lr_p, dt_p, rf_p, fi)
    print_summary(lr_m, dt_m, rf_m, dt_c, rf_c)

    # Save models
    with open("models/rf_regressor.pkl","wb") as f:  pickle.dump(rfr, f)
    with open("models/rf_classifier.pkl","wb") as f: pickle.dump(rfc, f)
    with open("models/scaler.pkl","wb") as f:        pickle.dump(scaler, f)
    with open("models/features.pkl","wb") as f:      pickle.dump(features, f)
    print("\n[OK] Models saved to models/ folder")

    return rfr, rfc, scaler, features


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    run_models(df)
