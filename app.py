"""
================================================
 Student Performance Analysis
 File: app.py
 Dashboard + Prediction System
 Run: streamlit run app.py
================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0a0f1e; color: #e2e8f0; }
h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        rfr      = pickle.load(open("models/rf_regressor.pkl","rb"))
        rfc      = pickle.load(open("models/rf_classifier.pkl","rb"))
        scaler   = pickle.load(open("models/scaler.pkl","rb"))
        features = pickle.load(open("models/features.pkl","rb"))
        return rfr, rfc, scaler, features
    except:
        return None, None, None, None


@st.cache_data
def load_data():
    try: return pd.read_csv("data/cleaned_data.csv")
    except: return None


rfr, rfc, scaler, features = load_models()
df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("", ["🏠 Dashboard","🔮 Predict","📊 Data Explorer","ℹ About"])

# ── DASHBOARD ────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("📚 Student Performance Analytics")
    st.markdown("**BTech 6th Semester — Data Analytics Project**")
    st.divider()

    if df is not None:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Students",  len(df))
        c2.metric("Pass Rate",       f"{(df['pass_status']=='Pass').mean()*100:.1f}%")
        c3.metric("Average Score",   f"{df['avg_score'].mean():.1f}")
        c4.metric("Avg Attendance",  f"{df['attendance'].mean():.1f}%")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Grade Distribution")
            BG, CARD = "#0a0f1e","#111827"
            fig, ax = plt.subplots(figsize=(6,4), facecolor=BG)
            ax.set_facecolor(CARD)
            order  = ["A+","A","B","C","D","F"]
            gc     = df["grade"].value_counts().reindex(order, fill_value=0)
            colors = ["#00d4ff","#10b981","#6ee7b7","#f59e0b","#fb923c","#ef4444"]
            bars   = ax.bar(gc.index, gc.values, color=colors, edgecolor="white", lw=0.5)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                        str(int(b.get_height())), ha="center", color="white", fontweight="bold")
            ax.tick_params(colors="#94a3b8"); ax.set_ylabel("Students", color="#94a3b8")
            for sp in ["top","right"]: ax.spines[sp].set_visible(False)
            st.pyplot(fig)

        with col2:
            st.subheader("Attendance vs Score")
            fig2, ax2 = plt.subplots(figsize=(6,4), facecolor=BG)
            ax2.set_facecolor(CARD)
            pm = df["pass_status"]=="Pass"
            ax2.scatter(df.loc[pm,"attendance"],  df.loc[pm,"avg_score"],  color="#10b981", alpha=0.6, s=20, label="Pass")
            ax2.scatter(df.loc[~pm,"attendance"], df.loc[~pm,"avg_score"], color="#ef4444", alpha=0.6, s=20, label="Fail")
            ax2.set_xlabel("Attendance %", color="#94a3b8"); ax2.set_ylabel("Avg Score", color="#94a3b8")
            ax2.tick_params(colors="#94a3b8"); ax2.legend(labelcolor="white", facecolor=CARD)
            for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
            st.pyplot(fig2)

        st.subheader("⚠ At-Risk Students (Score < 50)")
        at_risk = df[df["avg_score"]<50][["name","gender","attendance","study_hours","avg_score","pass_status"]].sort_values("avg_score")
        st.dataframe(at_risk, use_container_width=True)
    else:
        st.error("Run python main.py first to generate data!")


# ── PREDICT ──────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("🔮 Predict Student Performance")
    st.markdown("Enter student details to predict score and pass/fail status.")
    st.divider()

    with st.form("form"):
        c1, c2 = st.columns(2)
        with c1:
            gender       = st.selectbox("Gender", ["Male","Female"])
            parental_edu = st.selectbox("Parental Education", [
                "some high school","high school","some college",
                "associate's degree","bachelor's degree","master's degree"])
            lunch        = st.selectbox("Lunch Type", ["standard","free/reduced"])
            test_prep    = st.selectbox("Test Preparation", ["completed","none"])
        with c2:
            attendance   = st.slider("Attendance (%)", 40, 100, 75)
            study_hours  = st.slider("Study Hours/Day", 0.5, 8.0, 3.0, 0.5)
            assignments  = st.slider("Assignment Score", 0, 100, 70)

        submitted = st.form_submit_button("🔮 Predict Now", use_container_width=True)

    if submitted:
        if rfr is None:
            st.error("Run python main.py first!")
        else:
            edu_map = {"some high school":0,"high school":1,"some college":2,
                       "associate's degree":3,"bachelor's degree":4,"master's degree":5}
            gender_enc = 1 if gender=="Male" else 0
            lunch_enc  = 1 if lunch=="standard" else 0
            prep_enc   = 0 if test_prep=="completed" else 1
            edu_enc    = edu_map[parental_edu]
            att_group  = 0 if attendance<60 else 1 if attendance<75 else 2 if attendance<85 else 3

            X = np.array([[attendance, study_hours, assignments,
                           gender_enc, lunch_enc, prep_enc, edu_enc, att_group]])

            pred_score = rfr.predict(X)[0]
            pred_pass  = rfc.predict(X)[0]
            prob       = rfc.predict_proba(X)[0][1]*100

            def grade(s):
                if s>=90: return "A+"
                elif s>=80: return "A"
                elif s>=70: return "B"
                elif s>=60: return "C"
                elif s>=50: return "D"
                else: return "F"

            st.divider()
            st.subheader("📋 Result")
            c1,c2,c3 = st.columns(3)
            c1.metric("Predicted Score", f"{pred_score:.1f}/100")
            c2.metric("Expected Grade",  grade(pred_score))
            c3.metric("Pass Probability",f"{prob:.1f}%")

            if pred_pass==1:
                st.success(f"✅ PREDICTED: PASS — Pass probability {prob:.1f}%")
            else:
                st.error(f"❌ PREDICTED: FAIL — Only {prob:.1f}% chance of passing")

            st.subheader("💡 Recommendations")
            if attendance < 75: st.warning("📌 Attendance below 75% — attend more classes")
            if study_hours < 3: st.warning("📌 Study at least 3-4 hours daily")
            if test_prep=="none": st.info("📌 Complete the test preparation course (+8-12 marks)")
            if assignments < 60: st.warning("📌 Improve assignment submission score")


# ── DATA EXPLORER ─────────────────────────────────────────────────────────────
elif page == "📊 Data Explorer":
    st.title("📊 Dataset Explorer")
    if df is not None:
        st.write(f"**{len(df)} students | {df.shape[1]} features**")
        st.dataframe(df, use_container_width=True)
        st.divider()
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "student_data.csv","text/csv")
    else:
        st.error("Run python main.py first!")


# ── ABOUT ─────────────────────────────────────────────────────────────────────
elif page == "ℹ About":
    st.title("ℹ About This Project")
    st.markdown("""
    ## Student Performance Analysis System
    **BTech 3rd Year | 6th Semester | Data Analytics Project**

    ---
    ### Tools Used (as per project requirement)
    | Tool | Purpose |
    |------|---------|
    | **Python** | Core language |
    | **Pandas** | Data loading, cleaning, manipulation |
    | **Matplotlib** | Charts and visualizations |
    | **Seaborn** | Statistical plots (heatmap, boxplot) |
    | **Scikit-learn** | ML models — Linear Regression, Decision Tree, Random Forest |
    | **Streamlit** | This web dashboard |

    ---
    ### Dataset Features
    - Marks (Math, Reading, Writing)
    - Attendance %
    - Assignment scores
    - Study hours
    - Gender, Parental Education, Lunch, Test Prep

    ---
    ### How to Run
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn streamlit
    python main.py
    streamlit run app.py
    ```
    """)
