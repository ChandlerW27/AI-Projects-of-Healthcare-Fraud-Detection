
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_provider_features.csv"
)
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
EXAMPLE_DIR = PROJECT_ROOT / "app" / "examples"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_PATH)

if "FraudLabel" in train_df.columns:
    y = pd.to_numeric(train_df["FraudLabel"], errors="coerce")
else:
    y = train_df["PotentialFraud"].map({"No": 0, "Yes": 1})

X = (
    train_df
    .drop(
        columns=["Provider", "PotentialFraud", "FraudLabel"],
        errors="ignore",
    )
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

validation_scaler = StandardScaler()
X_train_scaled = validation_scaler.fit_transform(X_train)
X_valid_scaled = validation_scaler.transform(X_valid)

validation_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
validation_model.fit(X_train_scaled, y_train)

valid_probability = validation_model.predict_proba(X_valid_scaled)[:, 1]

metrics = {
    "pr_auc": float(
        average_precision_score(y_valid, valid_probability)
    ),
    "roc_auc": float(
        roc_auc_score(y_valid, valid_probability)
    ),
    "threshold": 0.50,
}

# Train the deployment model on all labeled rows.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
model.fit(X_scaled, y)

joblib.dump(model, MODEL_DIR / "logistic_model.joblib")
joblib.dump(scaler, MODEL_DIR / "standard_scaler.joblib")

metadata = {
    "feature_names": X.columns.tolist(),
    "feature_defaults": {
        key: float(value)
        for key, value in X.median().to_dict().items()
    },
    "feature_means": {
        key: float(value)
        for key, value in X.mean().to_dict().items()
    },
    "feature_stds": {
        key: float(value)
        for key, value in (
            X.std(ddof=0).replace(0, 1).to_dict()
        ).items()
    },
    "threshold": 0.50,
    "model_type": "LogisticRegression",
    "prediction_level": "Provider-level fraud risk",
    "validation_metrics": metrics,
}

(MODEL_DIR / "model_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

coefficients = (
    pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
        "AbsoluteCoefficient": np.abs(model.coef_[0]),
    })
    .sort_values("AbsoluteCoefficient", ascending=False)
)

coefficients.to_csv(
    MODEL_DIR / "logistic_coefficients.csv",
    index=False,
)

pd.DataFrame([X.median().to_dict()]).to_csv(
    EXAMPLE_DIR / "example_provider_input.csv",
    index=False,
)

pd.DataFrame([metrics]).to_csv(
    REPORT_DIR / "logistic_validation_metrics.csv",
    index=False,
)

print("Training complete.")
print("PR-AUC:", round(metrics["pr_auc"], 4))
print("ROC-AUC:", round(metrics["roc_auc"], 4))
print("Saved model files to:", MODEL_DIR.resolve())
