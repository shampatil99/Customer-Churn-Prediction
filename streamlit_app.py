import streamlit as st
import requests
import pandas as pd
from io import BytesIO

API_BASE = st.sidebar.text_input("API base URL", value="http://localhost:8000")

st.title("Customer Churn Prediction Platform")

st.header("1. Upload dataset and inspect")
uploaded = st.file_uploader("Upload CSV", type=["csv"] , key="upload")
if uploaded:
    filestr = uploaded.read()
    response = requests.post(f"{API_BASE}/upload", files={"file": (uploaded.name, filestr)}, data={})
    st.write(response.json())

st.header("2. Data cleaning")
uploaded2 = st.file_uploader("Upload CSV for cleaning", type=["csv"], key="clean")
missing = st.selectbox("Missing value handling", ["drop", "mean", "median", "mode"], index=0)
encoding = st.selectbox("Categorical encoding", ["label", "onehot"], index=0)
scaling = st.selectbox("Scaling", [None, "standard", "minmax"], index=0)
if uploaded2 and st.button("Run cleaning"):
    filestr = uploaded2.read()
    data = {"missing_strategy": missing, "categorical_encoding": encoding, "scaling": scaling}
    response = requests.post(f"{API_BASE}/data_cleaning", files={"file": (uploaded2.name, filestr)}, data=data)
    st.write(response.json())

st.header("3. Train models")
uploaded3 = st.file_uploader("Upload CSV for training", type=["csv"], key="train")
target_col = st.text_input("Target column name (e.g., Churn)")
models_select = st.multiselect("Models", ["logistic", "random_forest", "xgboost"], default=["logistic"])
if uploaded3 and target_col and st.button("Train"):
    filestr = uploaded3.read()
    models_str = ",".join(models_select)
    data = {"target_column": target_col, "models": models_str}
    response = requests.post(f"{API_BASE}/train_model", files={"file": (uploaded3.name, filestr)}, data={"target_column": target_col, "models": models_str})
    st.write(response.json())

st.header("4. Predict + GenAI explanations")
uploaded4 = st.file_uploader("Upload CSV for prediction", type=["csv"], key="predict")
model_name = st.text_input("Which trained model name to use (e.g., logistic_regression)")
if uploaded4 and model_name and st.button("Run prediction"):
    filestr = uploaded4.read()
    response = requests.post(f"{API_BASE}/predict", files={"file": (uploaded4.name, filestr)}, data={"model_name": model_name})
    j = response.json()
    if "predictions" in j:
        df = pd.DataFrame(j["predictions"])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv, file_name=f"predictions_{model_name}.csv")
    else:
        st.write(j)
