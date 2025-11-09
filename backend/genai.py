import os
from typing import List
import numpy as np

# Optional OpenAI import; only used if API key present
try:
    import openai
except Exception:
    openai = None


def simple_explain(X_row, model, feature_names, X_mean=None, top_k=2):
    # Fallback explanation method using coefficients or feature importances and deviation from mean
    arr = X_row.values.flatten()
    if X_mean is not None:
        mean = np.array(X_mean)
    else:
        mean = np.mean(arr)

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
        contrib = (arr - mean) * importances
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_).flatten()
        contrib = arr * coef
    else:
        contrib = np.abs(arr - mean)

    idx = np.argsort(-np.abs(contrib))[:top_k]
    parts = []
    for i in idx:
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        parts.append(f"{fname}={arr[i]:.3f}")
    reason = " and ".join(parts)
    return f"Likely drivers: {reason}."


def generate_explanations(X, model, feature_names, X_mean=None) -> List[str]:
    explanations = []
    # If OpenAI available and API key set, use it to generate nicer explanations
    use_openai = openai is not None and os.getenv("OPENAI_API_KEY")
    if use_openai:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    for i in range(len(X)):
        row = X.iloc[[i]]
        if use_openai:
            prompt = _build_prompt(row, model, feature_names)
            try:
                resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=60)
                text = resp.choices[0].text.strip()
            except Exception:
                text = simple_explain(row, model, feature_names, X_mean)
        else:
            text = simple_explain(row, model, feature_names, X_mean)
        explanations.append(text)
    return explanations


def _build_prompt(row, model, feature_names):
    parts = []
    for i, f in enumerate(feature_names):
        parts.append(f"{f}={row.iloc[0,i]}")
    input_str = ", ".join(parts)
    prompt = (
        "You are an expert data scientist. Given a customer record with features: "
        f"{input_str}. Provide a concise reason why this customer is likely to churn or stay. "
        "Be specific about which features contribute."
    )
    return prompt
