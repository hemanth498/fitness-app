import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

import warnings
warnings.filterwarnings('ignore')

# Apply custom background CSS
page_bg_img = """
<style>
body {
    background-image: url("https://source.unsplash.com/1600x900/?fitness,exercise");
    background-size: cover;
    background-attachment: fixed;
}
.sidebar .sidebar-content {
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 10px;
}
h1, h2, h3 {
    font-family: 'Arial', sans-serif;
    color: #FF5733;
}
.css-1aumxhk {
    color: black !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------------- Title & Introduction ----------------------
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🔥 Personal Fitness Tracker 🔥</h1>", unsafe_allow_html=True)

st.write("### Welcome to your Personal Fitness Tracker! 🚀")
st.markdown("""
Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp 
and then you will see the predicted value of kilocalories burned. 🎯
""")

# ---------------------- Sidebar ----------------------
st.sidebar.header("⚡ User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("🎂 Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("⚖️ BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("⏳ Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("💓 Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡️ Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("🚻 Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    return pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration], 
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]
    })

df = user_input_features()

# ---------------------- Display User Parameters ----------------------
st.markdown("### Your Input Parameters 📝")
st.write(df)

# Loading Data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merging and preprocessing
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)

# Prepare data
exercise_train_data = pd.get_dummies(exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]], drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]], drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

# Train Model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# ---------------------- Display Prediction ----------------------
st.markdown("### 🔮 Predicted Calories Burned 🔥")
st.success(f"💪 You will burn approximately **{round(prediction[0], 2)} kilocalories**.")

# ---------------------- Similar Results ----------------------
st.markdown("### 📊 Similar Results from the Dataset")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.dataframe(similar_data.sample(5))

# ---------------------- Insights for User ----------------------
st.markdown("### 📈 General Information Based on Your Input")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.info(f"📌 You are **older** than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of users.")
st.info(f"📌 Your **exercise duration** is longer than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of users.")
st.info(f"📌 Your **heart rate** is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of users.")
st.info(f"📌 Your **body temperature** is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of users.")

st.write("🚀 Keep pushing forward in your fitness journey! 💪")
