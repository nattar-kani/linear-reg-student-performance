import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://nattar:MJN@natt.nqgtssr.mongodb.net/?retryWrites=true&w=majority&appName=natt"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db["student_pred"]

def load_model():
    with open("Student-Performance-LR.pkl", "rb") as file:
        model,scaler,le = pickle.load(file)
    return model, scaler, le


def preprocessing_inpuData(data, scaler, le):
    df = pd.DataFrame([data])
    df['Extracurricular Activities'] = le.transform(df['Extracurricular Activities'].values)
    df_transform = scaler.transform(df)
    return df_transform

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_inpuData(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student performance prediction")
    st.write("Enter your data to get a Performance prediction score")

    readingHours = st.number_input("Hours studied", min_value=1, max_value=15, value=3)
    score = st.number_input("Previous score", max_value=100)
    activities = st.selectbox("Extracurricular activities", ['Yes', 'No'])
    sleepingTime = st.number_input("Sleeping hours", min_value=4, value=4)
    qpSolved = st.number_input("Number of Question paper solved", min_value=1,value=1)

    if st.button("Predict your performance"):
        user_data = {
            "Hours Studied": readingHours,
            "Previous Scores": score,
            "Extracurricular Activities": activities,
            "Sleep Hours": sleepingTime,
            "Sample Question Papers Practiced": qpSolved
        }
        prediction = predict_data(user_data)
        st.success(f"Your predicted performance index is {prediction}")
        user_data['prediction'] = round(float(prediction[0]),2)
        
        collection.insert_one(user_data)
    

if __name__ == "__main__":
    main()
