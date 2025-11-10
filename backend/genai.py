import os
from typing import List
import numpy as np
import openai


def generate_explanations(X, model, feature_names, X_mean=None) -> List[str]:
    explanations = []
    # If OpenAI available and API key set, use it to generate nicer explanations
    use_openai = openai is not None and os.getenv("OPENAI_API_KEY")
    if use_openai:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    for i in range(len(X)):
        row = X.iloc[[i]]
        prompt = generate_prompt(row, model, feature_names)
        resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=60)
        text = resp.choices[0].text.strip()
        explanations.append(text)
    return explanations

def generate_prompt(row, model, feature_names):
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
