"""
================================================
 Student Performance Analysis
 File: generate_dataset.py
 Creates a realistic student dataset (300 rows)
================================================
"""

import pandas as pd
import numpy as np

np.random.seed(42)

def generate_dataset(n=300):

    names_male   = ["Aarav","Rohit","Karan","Arjun","Vikram","Siddharth","Rahul","Nikhil",
                     "Aditya","Varun","Harshit","Pranav","Kunal","Manav","Yash","Akash",
                     "Rohan","Dev","Saurabh","Ritesh","Amit","Suresh","Ravi","Deepak",
                     "Gaurav","Ankit","Sachin","Mohit","Vivek","Ajay"]
    names_female = ["Priya","Sneha","Ananya","Meera","Divya","Kavya","Pooja","Riya",
                     "Shreya","Tanvi","Nandini","Ishita","Swati","Deepika","Anjali",
                     "Shivani","Pallavi","Kritika","Sonam","Bhavna","Neha","Sunita",
                     "Rekha","Geeta","Sonal","Mansi","Komal","Nisha","Preeti","Jyoti"]

    gender, name = [], []
    for i in range(n):
        if i % 2 == 0:
            gender.append("Male")
            name.append(names_male[i % len(names_male)])
        else:
            gender.append("Female")
            name.append(names_female[i % len(names_female)])

    parental_education = np.random.choice(
        ["some high school","high school","some college",
         "associate's degree","bachelor's degree","master's degree"],
        n, p=[0.05, 0.20, 0.25, 0.20, 0.20, 0.10]
    )

    edu_map = {"some high school":0,"high school":1,"some college":2,
               "associate's degree":3,"bachelor's degree":4,"master's degree":5}
    edu_score = np.array([edu_map[e] for e in parental_education])

    lunch        = np.random.choice(["standard","free/reduced"], n, p=[0.65, 0.35])
    lunch_num    = (np.array(lunch) == "standard").astype(int)
    test_prep    = np.random.choice(["completed","none"], n, p=[0.40, 0.60])
    prep_num     = (np.array(test_prep) == "completed").astype(int)

    attendance   = np.clip(60 + edu_score*3 + prep_num*5 + np.random.normal(0,8,n), 40, 100).astype(int)
    study_hours  = np.clip(2 + prep_num*1.5 + np.random.normal(0,1.5,n), 0.5, 8).round(1)
    assignments  = np.clip(50 + prep_num*10 + edu_score*3 + np.random.normal(0,10,n), 0, 100).astype(int)

    base = (40 + edu_score*3 + prep_num*8 + lunch_num*5 +
            (attendance-60)*0.4 + study_hours*3 + np.random.normal(0,8,n))

    math_score    = np.clip(base + np.random.normal(0,5,n), 0, 100).astype(int)
    reading_score = np.clip(base + 5 + np.random.normal(0,4,n), 0, 100).astype(int)
    writing_score = np.clip(base + 4 + np.random.normal(0,4,n), 0, 100).astype(int)
    avg_score     = ((math_score + reading_score + writing_score) / 3).round(2)

    def grade(s):
        if s >= 90: return "A+"
        elif s >= 80: return "A"
        elif s >= 70: return "B"
        elif s >= 60: return "C"
        elif s >= 50: return "D"
        else: return "F"

    df = pd.DataFrame({
        "name": name, "gender": gender,
        "parental_education": parental_education,
        "lunch": lunch, "test_preparation": test_prep,
        "attendance": attendance, "study_hours": study_hours,
        "assignments": assignments,
        "math_score": math_score, "reading_score": reading_score,
        "writing_score": writing_score, "avg_score": avg_score,
        "grade": [grade(s) for s in avg_score],
        "pass_status": ["Pass" if s >= 50 else "Fail" for s in avg_score]
    })

    # Add ~3% missing values for realism
    for col in ["attendance","study_hours","math_score"]:
        mask = np.random.choice([True,False], n, p=[0.03,0.97])
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/student_performance.csv", index=False)
    print(f"[OK] Dataset saved — Shape: {df.shape}")
    print(df.head())
