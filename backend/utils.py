import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from typing import Tuple, List


def get_dataset_metadata(df: pd.DataFrame):
    cols = {}
    for c in df.columns:
        cols[c] = {
            "dtype": str(df[c].dtype),
            "n_missing": int(df[c].isnull().sum()),
            "n_unique": int(df[c].nunique(dropna=True)),
        }
    return cols


def clean_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Missing
    strategy = cfg.get("missing_strategy", "drop")
    if strategy == "drop":
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype.kind in 'biufc':
                    if strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median":
                        df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Categorical encoding
    enc = cfg.get("categorical_encoding", "label")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if enc == "label":
        for c in cat_cols:
            try:
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))
            except Exception:
                pass
    elif enc == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scaling
    scaling = cfg.get("scaling")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if scaling == "standard":
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    elif scaling == "minmax":
        df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])

    return df


def prepare_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df2 = df.copy()
    if target_column not in df2.columns:
        raise ValueError("target column missing")
    y = df2[target_column]
    X = df2.drop(columns=[target_column])

    # Basic cleaning: drop non-numeric id columns
    # Keep numeric and encoded categorical columns
    X = X.select_dtypes(include=[np.number, 'float64', 'int64'])
    # If no numeric columns left, attempt label-encoding of object columns
    if X.shape[1] == 0:
        obj_cols = df2.drop(columns=[target_column]).select_dtypes(include=['object']).columns.tolist()
        for c in obj_cols:
            X[c] = pd.Categorical(df2[c]).codes

    feature_names = X.columns.tolist()
    return X, y.astype(int), feature_names
