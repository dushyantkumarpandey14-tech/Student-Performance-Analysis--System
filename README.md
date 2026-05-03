# 📚 Student Performance Analysis System

> **BTech 3rd Year | 6th Semester | Data Analytics Project**

A complete Machine Learning project that analyzes and predicts student academic performance using Python, Pandas, Matplotlib, Seaborn, and Scikit-learn.

---

## 🎯 Project Overview

This project analyzes student performance data to:
- Identify patterns and trends in academic performance
- Find correlation between attendance, study hours, assignments and marks
- Predict whether a student will **Pass or Fail**
- Predict the expected **grade and score**
- Provide an interactive **web dashboard** for real-time predictions

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Pandas** | Data loading, cleaning & manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Charts and visualizations |
| **Seaborn** | Statistical plots (heatmap, boxplot) |
| **Scikit-learn** | Machine Learning models |
| **Streamlit** | Interactive web dashboard |

---

## 📁 Project Structure

```
Student-Performance-Analysis-System/
│
├── main.py                  ← Run this file (runs everything)
├── generate_dataset.py      ← Creates student dataset (300 records)
├── preprocessing.py         ← Data cleaning & encoding
├── eda.py                   ← 9 charts & visualizations
├── models.py                ← ML models training & evaluation
├── app.py                   ← Streamlit web dashboard
│
├── data/                    ← Generated automatically
│   ├── student_performance.csv
│   └── cleaned_data.csv
│
├── plots/                   ← Generated automatically (10 charts)
│   ├── 01_score_distributions.png
│   ├── 02_grade_distribution.png
│   ├── 03_attendance_vs_marks.png
│   ├── 04_assignments_vs_marks.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_test_prep_effect.png
│   ├── 07_gender_comparison.png
│   ├── 08_study_hours.png
│   ├── 09_performance_trends.png
│   └── 10_model_comparison.png
│
└── models/                  ← Saved ML models (.pkl files)
```

---

## 📊 Dataset Features

- **Name, Gender** — Student details
- **Parental Education** — 6 levels (high school to master's)
- **Lunch** — Socioeconomic indicator
- **Test Preparation** — Completed / None
- **Attendance %** — Class attendance
- **Study Hours** — Daily study hours
- **Assignment Score** — Assignment performance
- **Math, Reading, Writing Score** — Subject-wise marks
- **Average Score** — Overall performance
- **Grade** — A+, A, B, C, D, F
- **Pass/Fail Status** — Target variable

---

## 🤖 Machine Learning Models

| Model | RMSE | R² Score | Accuracy |
|-------|------|----------|----------|
| Linear Regression | 8.22 | 0.532 | N/A |
| Decision Tree | 9.68 | 0.352 | 94.37% |
| **Random Forest ★** | **8.63** | **0.484** | **94.37%** |

**Best Model: Random Forest** — highest accuracy, best generalization

---

## 📈 Key Insights

- ✅ **Attendance** is the most important feature (16.2% importance)
- ✅ **Study hours** strongly correlate with performance
- ✅ **Test preparation** boosts scores by 8-12 points
- ✅ **Parental education** significantly impacts student results
- ✅ **94.37% accuracy** in predicting Pass/Fail

---

## ▶️ How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/dushyantkumarpandey14-tech/Student-Performance-Analysis--System.git
cd Student-Performance-Analysis--System
```

### Step 2 — Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Step 3 — Run the full pipeline
```bash
python main.py
```
This automatically:
- Generates the dataset
- Cleans and preprocesses data
- Creates all 10 charts
- Trains 3 ML models
- Saves everything to folders

### Step 4 — Launch the web dashboard
```bash
streamlit run app.py
```
Opens in your browser automatically at `http://localhost:8501`

---

## 🖥️ Web Dashboard Features

- 📊 **Dashboard** — Stats, charts, at-risk students
- 🔮 **Predict** — Enter student data → get score + grade prediction
- 📋 **Data Explorer** — Browse full dataset, download CSV
- ℹ️ **About** — Project info and tech stack

---
