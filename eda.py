"""
================================================
 Student Performance Analysis
 File: eda.py
 All charts using Matplotlib & Seaborn
 Saves 8 plots to plots/ folder
================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

BG   = "#0a0f1e"
CARD = "#111827"
plt.style.use("dark_background")
COLORS = ["#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444","#ec4899"]


def save(name, caption=""):
    plt.tight_layout()
    plt.savefig(f"plots/{name}", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[SAVED] plots/{name}  —  {caption}")


# ── PLOT 1: Score Distributions ──────────────────────────────────────────────
def plot_score_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle("Score Distributions", fontsize=14, fontweight="bold", color="white")
    for ax, col, name, color in zip(axes,
        ["math_score","reading_score","writing_score"],
        ["Math Score","Reading Score","Writing Score"],
        COLORS):
        ax.set_facecolor(CARD)
        ax.hist(df[col].dropna(), bins=20, color=color, alpha=0.85, edgecolor="white", lw=0.3)
        ax.axvline(df[col].mean(), color="#f59e0b", lw=2, linestyle="--",
                   label=f"Mean: {df[col].mean():.1f}")
        ax.set_title(name, color="white", fontweight="bold")
        ax.set_xlabel("Score", color="#94a3b8")
        ax.set_ylabel("Frequency", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.legend(labelcolor="white", facecolor=CARD)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save("01_score_distributions.png", "Distribution of Math, Reading, Writing scores")


# ── PLOT 2: Grade & Pass/Fail ─────────────────────────────────────────────────
def plot_grade_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Grade & Pass/Fail Distribution", fontsize=14, fontweight="bold", color="white")

    # Bar chart
    ax1 = axes[0]
    ax1.set_facecolor(CARD)
    order  = ["A+","A","B","C","D","F"]
    counts = df["grade"].value_counts().reindex(order, fill_value=0)
    clrs   = ["#00d4ff","#10b981","#6ee7b7","#f59e0b","#fb923c","#ef4444"]
    bars   = ax1.bar(counts.index, counts.values, color=clrs, edgecolor="white", lw=0.5)
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 str(int(b.get_height())), ha="center", color="white", fontweight="bold")
    ax1.set_title("Grade Distribution", color="white", fontweight="bold")
    ax1.set_xlabel("Grade", color="#94a3b8")
    ax1.set_ylabel("Students", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for sp in ["top","right"]: ax1.spines[sp].set_visible(False)

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    pf = df["pass_status"].value_counts()
    wedges, texts, auto = ax2.pie(pf.values, labels=pf.index, autopct="%1.1f%%",
        startangle=140, colors=["#10b981","#ef4444"],
        wedgeprops={"edgecolor": BG, "linewidth": 2})
    for t in texts: t.set_color("white")
    for a in auto:  a.set_color("white"); a.set_fontweight("bold")
    ax2.set_title("Pass vs Fail", color="white", fontweight="bold")

    save("02_grade_distribution.png", "Grade counts and Pass/Fail ratio")


# ── PLOT 3: Attendance vs Marks ───────────────────────────────────────────────
def plot_attendance_vs_marks(df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    pm = df["pass_status"] == "Pass"
    ax.scatter(df.loc[pm,"attendance"],  df.loc[pm,"avg_score"],  color="#10b981", alpha=0.7, label="Pass", s=40)
    ax.scatter(df.loc[~pm,"attendance"], df.loc[~pm,"avg_score"], color="#ef4444", alpha=0.7, label="Fail", s=40)
    x = df["attendance"].dropna()
    y = df.loc[x.index,"avg_score"]
    m, b = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, m*xl+b, color="#f59e0b", lw=2, linestyle="--", label=f"Trend (slope={m:.2f})")
    ax.axvline(75, color="#00d4ff", linestyle=":", alpha=0.6, label="75% attendance")
    ax.set_title("Attendance vs Average Score", color="white", fontweight="bold", fontsize=13)
    ax.set_xlabel("Attendance (%)", color="#94a3b8")
    ax.set_ylabel("Average Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.legend(labelcolor="white", facecolor=CARD, edgecolor="#1e2d45")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    corr = df["attendance"].corr(df["avg_score"])
    save("03_attendance_vs_marks.png", f"Correlation = {corr:.3f}")


# ── PLOT 4: Assignments vs Marks ──────────────────────────────────────────────
def plot_assignments_vs_marks(df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    sc = ax.scatter(df["assignments"], df["avg_score"], c=df["avg_score"],
                    cmap="cool", alpha=0.7, s=45, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Avg Score").ax.tick_params(colors="white")
    m, b = np.polyfit(df["assignments"].dropna(), df.loc[df["assignments"].notna(),"avg_score"], 1)
    xl = np.linspace(df["assignments"].min(), df["assignments"].max(), 100)
    ax.plot(xl, m*xl+b, color="#f59e0b", lw=2, linestyle="--", label="Trend")
    ax.set_title("Assignment Score vs Average Score", color="white", fontweight="bold", fontsize=13)
    ax.set_xlabel("Assignment Score", color="#94a3b8")
    ax.set_ylabel("Average Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.legend(labelcolor="white", facecolor=CARD)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save("04_assignments_vs_marks.png", "Assignment score correlation with marks")


# ── PLOT 5: Correlation Heatmap ───────────────────────────────────────────────
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    ax.set_facecolor(CARD)
    cols = ["attendance","study_hours","assignments","math_score",
            "reading_score","writing_score","avg_score"]
    corr = df[cols].corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax,
                annot_kws={"size":11,"color":"white"},
                linewidths=0.5, linecolor="#1e2d45")
    ax.set_title("Feature Correlation Heatmap", color="white", fontweight="bold", fontsize=13)
    ax.tick_params(colors="white", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save("05_correlation_heatmap.png", "Heatmap of all numerical features")


# ── PLOT 6: Test Prep Effect ──────────────────────────────────────────────────
def plot_test_prep(df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    data  = df.groupby("test_preparation")[["math_score","reading_score","writing_score"]].mean()
    x     = np.arange(3)
    width = 0.35
    for i, (label, color) in enumerate(zip(data.index, ["#00d4ff","#ef4444"])):
        bars = ax.bar(x + i*width - width/2, data.loc[label], width,
                      label=label, color=color, alpha=0.85, edgecolor="white", lw=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{b.get_height():.1f}", ha="center", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Math","Reading","Writing"], color="#94a3b8")
    ax.set_title("Effect of Test Preparation on Scores", color="white", fontweight="bold", fontsize=13)
    ax.set_ylabel("Average Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.legend(labelcolor="white", facecolor=CARD, edgecolor="#1e2d45")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save("06_test_prep_effect.png", "Completed vs None — score comparison")


# ── PLOT 7: Gender Comparison ────────────────────────────────────────────────
def plot_gender(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=BG)
    fig.suptitle("Score Comparison by Gender", fontsize=14, fontweight="bold", color="white")
    for ax, col, name in zip(axes,
        ["math_score","reading_score","writing_score"],["Math","Reading","Writing"]):
        ax.set_facecolor(CARD)
        dm = df[df["gender"]=="Male"][col].dropna()
        df_ = df[df["gender"]=="Female"][col].dropna()
        bp  = ax.boxplot([dm, df_], labels=["Male","Female"], patch_artist=True,
            boxprops=dict(facecolor="#00d4ff", color="white", alpha=0.7),
            medianprops=dict(color="#f59e0b", linewidth=2),
            whiskerprops=dict(color="#94a3b8"),
            capprops=dict(color="#94a3b8"),
            flierprops=dict(marker="o", color="#ef4444", alpha=0.5))
        bp["boxes"][1].set_facecolor("#7c3aed")
        ax.set_title(f"{name} Score", color="white", fontweight="bold")
        ax.set_ylabel("Score", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save("07_gender_comparison.png", "Box plot comparison by gender")


# ── PLOT 8: Study Hours vs Marks ─────────────────────────────────────────────
def plot_study_hours(df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    sc = ax.scatter(df["study_hours"], df["avg_score"], c=df["avg_score"],
                    cmap="cool", alpha=0.7, s=50)
    plt.colorbar(sc, ax=ax, label="Score").ax.tick_params(colors="white")
    m, b = np.polyfit(df["study_hours"].dropna(),
                      df.loc[df["study_hours"].notna(),"avg_score"], 1)
    xl = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
    ax.plot(xl, m*xl+b, color="#f59e0b", lw=2, linestyle="--",
            label=f"y = {m:.1f}x + {b:.1f}")
    ax.set_title("Study Hours vs Average Score", color="white", fontweight="bold", fontsize=13)
    ax.set_xlabel("Study Hours / Day", color="#94a3b8")
    ax.set_ylabel("Average Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.legend(labelcolor="white", facecolor=CARD)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save("08_study_hours.png", "Study hours correlation with performance")


# ── PLOT 9: Performance Trends (Top 10) ──────────────────────────────────────
def plot_performance_trends(df):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    top10 = df.nlargest(10, "avg_score")[["name","avg_score","math_score",
                                           "reading_score","writing_score"]]
    x = np.arange(len(top10))
    w = 0.22
    for i, (col, color, label) in enumerate(zip(
        ["math_score","reading_score","writing_score","avg_score"],
        ["#00d4ff","#7c3aed","#10b981","#f59e0b"],
        ["Math","Reading","Writing","Average"])):
        ax.bar(x + i*w - 1.5*w, top10[col], w, label=label,
               color=color, alpha=0.85, edgecolor="white", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(top10["name"], rotation=30, ha="right", color="#94a3b8")
    ax.set_title("Performance Trends — Top 10 Students", color="white", fontweight="bold", fontsize=13)
    ax.set_ylabel("Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.legend(labelcolor="white", facecolor=CARD, edgecolor="#1e2d45")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    save("09_performance_trends.png", "Top 10 student score breakdown")


def run_eda(df):
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    plot_score_distributions(df)
    plot_grade_distribution(df)
    plot_attendance_vs_marks(df)
    plot_assignments_vs_marks(df)
    plot_heatmap(df)
    plot_test_prep(df)
    plot_gender(df)
    plot_study_hours(df)
    plot_performance_trends(df)
    print("\n[OK] All 9 plots saved to plots/ folder")


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    run_eda(df)
