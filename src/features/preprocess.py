"""Leakage-safe preprocessing pipeline.

Fit ONLY on training data. The same fitted object is saved with the model,
so training and inference use identical preprocessing.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.build_features import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features,
)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        # Blank TotalCharges <=> tenure == 0 (new, not yet billed) -> 0 is logical.
        # add_indicator keeps a flag so the model can see which rows were filled.
        ("impute", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("columns", columns),
    ])
