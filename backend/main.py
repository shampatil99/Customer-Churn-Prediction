from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import io
import joblib
from typing import List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .utils import get_dataset_metadata, clean_dataframe, prepare_features
from .genai import generate_explanations

app = FastAPI(title="Churn Prediction API")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODELS_DIR, exist_ok=True)

class CleanConfig(BaseModel):
    missing_strategy: Optional[str] = "drop"  # drop / mean / median / mode
    categorical_encoding: Optional[str] = "label"  # label / onehot
    scaling: Optional[str] = None  # standard / minmax / none

class TrainConfig(BaseModel):
    target_column: str
    models: List[str]  # e.g., ["logistic", "random_forest", "xgboost"]
    test_size: float = 0.2
    random_state: int = 42

class PredictConfig(BaseModel):
    model_name: str


@app.post("/upload")
async def upload(file: UploadFile = File(...), target_column: Optional[str] = Form(None)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    metadata = get_dataset_metadata(df)
    response = {"n_rows": df.shape[0], "n_cols": df.shape[1], "columns": metadata}
    if target_column and target_column in df.columns:
        response["target_column"] = target_column
    response["preview"] = df.head(5).to_dict(orient="records")
    return response


@app.post("/data_cleaning")
async def data_cleaning(file: UploadFile = File(...), missing_strategy: str = Form("drop"), categorical_encoding: str = Form("label"), scaling: Optional[str] = Form(None)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    cfg = {
        "missing_strategy": missing_strategy,
        "categorical_encoding": categorical_encoding,
        "scaling": scaling,
    }
    cleaned = clean_dataframe(df.copy(), cfg)
    # Save cleaned data to models/cleaned_data.csv for reuse
    cleaned_path = os.path.join(MODELS_DIR, "cleaned_data.csv")
    cleaned.to_csv(cleaned_path, index=False)
    return {
        "preview": cleaned.head(5).to_dict(orient="records"),
        "shape": cleaned.shape,
        "saved_path": cleaned_path
    }


@app.post("/train_model")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...), models: str = Form(...), test_size: float = Form(0.2)):
    # models passed as comma-separated string
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    X, y, feature_names = prepare_features(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    results = {}

    for m in model_list:
        if m.lower() in ("logistic", "logistic_regression", "lr"):
            model = LogisticRegression(max_iter=1000)
            model_name = "logistic_regression"
        elif m.lower() in ("random_forest", "rf"):
            model = RandomForestClassifier(n_estimators=100)
            model_name = "random_forest"
        elif m.lower() in ("xgboost", "xgb"):
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model_name = "xgboost"
            except Exception as e:
                results[m] = {"error": "xgboost not installed"}
                continue
        else:
            results[m] = {"error": "unknown model"}
            continue

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        model_filename = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        joblib.dump({"model": model, "feature_names": feature_names, "X_mean": X.mean(axis=0).tolist()}, model_filename)

        results[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "saved_path": model_filename,
        }

    return results


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return {"error": "model not found", "model_path": model_path}
    data = joblib.load(model_path)
    model = data["model"]
    feature_names = data.get("feature_names")
    X_mean = data.get("X_mean")

    X = df.copy()
    if feature_names is not None:
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            return {"error": "missing features in test file", "missing": missing}
        Xr = X[feature_names]
    else:
        Xr = X.select_dtypes(include=["number"])
        feature_names = Xr.columns.tolist()

    preds = model.predict(Xr)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xr)[:, 1].tolist()

    explanations = generate_explanations(Xr, model, feature_names, X_mean)

    out = []
    for i, row in X.iterrows():
        r = row.to_dict()
        r["prediction"] = int(preds[i])
        if proba is not None:
            r["score"] = float(proba[i])
        r["explanation"] = explanations[i]
        out.append(r)

    # Save predictions to a CSV in models folder
    out_df = pd.DataFrame(out)
    out_path = os.path.join(MODELS_DIR, f"predictions_{model_name}.csv")
    out_df.to_csv(out_path, index=False)

    return {"n_predictions": len(out), "predictions_file": out_path, "predictions": out}
