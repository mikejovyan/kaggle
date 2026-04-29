from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict

if TYPE_CHECKING:
    import pandas as pd


def adversarial_score(
    data: pd.DataFrame,
    model: Any,
    target: str = "Set",
    cv: int = 5,
) -> float:
    X = data.drop(columns=target)
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    X[cat_cols] = X[cat_cols].astype("category")
    y = data[target] == "train"

    preds = cross_val_predict(model, X, y, cv=cv)
    return balanced_accuracy_score(y_true=y, y_pred=preds)
