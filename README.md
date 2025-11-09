# Customer Churn Prediction Platform (FastAPI + Streamlit)

This repository contains a FastAPI backend and a Streamlit frontend to:
- Upload datasets and inspect metadata
- Run configurable data cleaning
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Run batch predictions and generate natural-language explanations (via OpenAI or a simple local heuristic)

Quick start

1. Create a virtual environment and install requirements:

```powershell
python -m venv churn_venv; .\churn_venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the FastAPI app:

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. In another terminal run the Streamlit UI:

```powershell
streamlit run ..\streamlit_app.py
```

Notes on GenAI explanations

- If you set the environment variable `OPENAI_API_KEY`, the app will attempt to call OpenAI's completion API to generate richer explanations.
- If no API key is present or OpenAI is not installed, the backend will use a simple heuristic using model coefficients / feature importances and deviation from mean.

Files
- `backend/main.py` - FastAPI app
- `backend/utils.py` - preprocessing helpers
- `backend/genai.py` - explanation generator wrapper
- `streamlit_app.py` - lightweight Streamlit UI
- `requirements.txt` - Python dependencies
- `sample_data.csv` - small sample dataset

GenAI prompt structure

When OpenAI is available, the prompt includes a comma-separated list of feature=value pairs and asks the model to provide a concise reason why the customer may churn or stay, focusing on contributing features.

License: MIT
