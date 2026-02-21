import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def plot_confusion_matrix(cm: np.ndarray, target: str, class_labels: list[str]) -> None:
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        data=pd.DataFrame(cm, columns=class_labels, index=class_labels),
        annot=True,
        fmt="d",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
    )
    plt.title(f"Confusion Matrix: {target}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    train_data: pd.DataFrame,
    features: list[str],
    cmap: str = "coolwarm",
    figsize: tuple | None = None,
) -> None:
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if not isinstance(features, list):
        raise TypeError("features must be a list")
    if not features:
        raise ValueError("features list cannot be empty")

    missing_features = [col for col in features if col not in train_data.columns]
    if missing_features:
        raise ValueError(f"Features not found in train_data: {missing_features}")

    correlation_matrix = train_data[features].corr()

    if figsize is None:
        width = max(len(features) * 0.75, 5)
        height = max(len(features) * 0.75, 4)
        figsize = (width, height)

    plt.figure(figsize=figsize)
    sns.heatmap(
        data=correlation_matrix,
        mask=np.triu(correlation_matrix),
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(
    train_data: pd.DataFrame,
    features: list[str],
    test_data: pd.DataFrame | None = None,
    categorical_features: list[str] | None = None,
    figsize: tuple | None = None,
    ncols: int = 2,
) -> None:
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if not isinstance(features, list):
        raise TypeError("features must be a list")

    if not features:
        raise ValueError("features list cannot be empty")

    if not isinstance(ncols, int) or ncols <= 0:
        raise ValueError("ncols must be a positive integer")

    categorical_features = categorical_features or []

    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        raise ValueError(f"Features not found in train_data: {missing_features}")

    train_copy = train_data.copy()
    train_copy["set"] = "train"

    if test_data is None:
        combined_data = train_copy
    else:
        if not isinstance(test_data, pd.DataFrame):
            raise TypeError("test_data must be a pandas DataFrame")

        missing_features_test = [f for f in features if f not in test_data.columns]
        if missing_features_test:
            raise ValueError(
                f"Features not found in test_data: {missing_features_test}"
            )

        test_copy = test_data.copy()
        test_copy["set"] = "test"
        combined_data = pd.concat([train_copy, test_copy])

    nrows = int(np.ceil(len(features) / ncols))

    if figsize is None:
        width = 12
        height = 4 * nrows
        figsize = (width, height)

    plt.figure(figsize=figsize)

    for i, feature in enumerate(features, start=1):
        plt.subplot(nrows, ncols, i)
        if feature in categorical_features:
            sns.countplot(data=combined_data, x=feature, hue="set", edgecolor="black")
            plt.title(f"{feature} Count")
        else:
            for dataset_name in combined_data["set"].unique():
                selection = combined_data.loc[combined_data["set"] == dataset_name]
                sns.histplot(data=selection, x=feature, label=dataset_name)
            plt.legend(title="set")
            plt.grid(axis="y", linestyle="--")
            plt.title(f"{feature} Distribution")

    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.show()


def plot_model_metrics(
    results: pd.DataFrame, metrics: list[str] | str, figsize: tuple | None = None
) -> None:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame")

    if not metrics:
        raise ValueError("metrics cannot be empty")

    if isinstance(metrics, str):
        metrics = [metrics]

    missing_metrics = [m for m in metrics if m not in results.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in results: {missing_metrics}")

    if figsize is None:
        width = max(results.shape[0] * 0.7, 5)
        height = max(results.shape[0] * 0.35, 4)
        figsize = (width, height)

    plt.figure(figsize=figsize)
    sns.lineplot(data=results[metrics])
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_predictions(data: pd.Series, figsize: tuple | None = None) -> None:
    if not isinstance(data, pd.Series):
        raise TypeError(f"data must be a pandas Series, got {type(data).__name__}")

    if data.empty:
        raise ValueError("data is empty")

    if figsize is None:
        width = 5
        height = 4
        figsize = (width, height)

    plt.figure(figsize=figsize)
    sns.histplot(x=data, stat="count", discrete=True, fill=True)
    plt.title("Prediction Counts")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_probability_distribution(
    data: pd.Series, figsize: tuple | None = None, target: str | None = None
) -> None:
    if not isinstance(data, pd.Series):
        raise TypeError(f"data must be a pandas Series, got {type(data).__name__}")

    if data.empty:
        raise ValueError("data is empty")

    if figsize is None:
        width = 5
        height = 4
        figsize = (width, height)

    plt.figure(figsize=figsize)
    sns.kdeplot(x=data, fill=True)
    plt.title("Probability Distribution")
    plt.xlabel(target or "Probability")
    plt.ylabel("Density")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_target_correlations(
    train_data: pd.DataFrame,
    target: str,
    features: list[str],
    figsize: tuple | None = None,
) -> None:
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if not isinstance(target, str):
        raise TypeError("target must be a string")

    if not isinstance(features, list):
        raise TypeError("features must be a list")

    if not features:
        raise ValueError("features list cannot be empty")

    if target not in train_data.columns:
        raise ValueError(f"Target column '{target}' not found in train_data")

    missing_features = [col for col in features if col not in train_data.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in train_data: {missing_features}")

    correlations = (
        train_data[features].corrwith(train_data[target]).sort_values(ascending=False)
    )

    if figsize is None:
        max_label_length = max(len(label) for label in features)
        width = max(12, 5)
        height = max(max_label_length * 0.2, 5)
        figsize = (width, height)

    plt.figure(figsize=figsize)
    sns.barplot(data=correlations.reset_index(), y=correlations.values, x="index")
    plt.xlabel("Feature")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--")
    plt.title(f"Correlation with {target}")
    plt.tight_layout()
    plt.show()


def validate_train_test_distribution(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: BaseEstimator,
    features: list | None = None,
    cv: int = 5,
    threshold: float = 0.6,
) -> None:
    """
    Validate that train and test sets have similar distributions.

    Uses a discriminator model to try to distinguish train from test samples.
    If the model can't distinguish them well (ROC-AUC < threshold), they're
    likely from the same distribution.
    """
    if features:
        common_columns = features
    else:
        common_columns = train_data.columns.intersection(test_data.columns)
        if len(common_columns) == 0:
            raise ValueError("Train and test sets have no common columns")

    train_valid = train_data[common_columns].copy()
    test_valid = test_data[common_columns].copy()

    X_valid = pd.concat([test_valid, train_valid], axis=0, ignore_index=True)
    y_valid = [0] * len(test_valid) + [1] * len(train_valid)

    cv_preds = cross_val_predict(
        model, X_valid, y_valid, cv=cv, n_jobs=-1, method="predict_proba"
    )

    score = roc_auc_score(y_true=y_valid, y_score=cv_preds[:, 1])

    if score < threshold:
        comparison = "<"
        message = "train and test sets indistinguishable"
    else:
        comparison = ">="
        message = (
            "train and test sets distinguishable (potential distribution mismatch)"
        )

    print(
        f"Validation score = {score:.2f} {comparison} "
        f"threshold = {threshold} → {message}"
    )
