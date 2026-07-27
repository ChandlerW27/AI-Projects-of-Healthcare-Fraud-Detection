
import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    COEFFICIENTS_PATH,
    LOGISTIC_MODEL_PATH,
    METADATA_PATH,
    SCALER_PATH,
)


class FraudPredictor:
    """Load the trained Logistic Regression model and make predictions."""

    def __init__(self):
        required_files = [
            LOGISTIC_MODEL_PATH,
            SCALER_PATH,
            METADATA_PATH,
            COEFFICIENTS_PATH,
        ]

        missing = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Model files are missing. Run "
                "`python src/train_logistic.py` first.\n\n"
                + "\n".join(missing)
            )

        self.model = joblib.load(LOGISTIC_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )
        self.coefficients = pd.read_csv(COEFFICIENTS_PATH)

        self.feature_names = self.metadata["feature_names"]
        self.defaults = self.metadata["feature_defaults"]
        self.means = pd.Series(self.metadata["feature_means"])
        self.stds = (
            pd.Series(self.metadata["feature_stds"])
            .replace(0, 1)
        )
        self.threshold = float(
            self.metadata.get("threshold", 0.50)
        )
        self.metrics = self.metadata.get(
            "validation_metrics",
            {},
        )

    def build_input(self, values):
        """Create one complete model row from partial input."""
        row = {
            feature: self.defaults.get(feature, 0.0)
            for feature in self.feature_names
        }
        row.update(values)
        return pd.DataFrame([row], columns=self.feature_names)

    def prepare_input(self, data):
        """Clean and align input features."""
        prepared = data.copy()

        for feature in self.feature_names:
            if feature not in prepared.columns:
                prepared[feature] = self.defaults.get(feature, 0.0)

        prepared = prepared[self.feature_names]
        prepared = prepared.apply(pd.to_numeric, errors="coerce")
        prepared = prepared.replace([np.inf, -np.inf], np.nan)

        return prepared.fillna(pd.Series(self.defaults))

    def predict(self, data):
        """Return fraud probability, label, and prepared input."""
        prepared = self.prepare_input(data)
        probability = float(
            self.model.predict_proba(
                self.scaler.transform(prepared)
            )[0, 1]
        )

        return {
            "probability": probability,
            "is_fraud": probability >= self.threshold,
            "threshold": self.threshold,
            "input": prepared,
        }

    def explain_contributions(self, prepared_input, top_n=6):
        """Return the strongest positive Logistic Regression contributions."""
        coefficients = (
            self.coefficients
            .set_index("Feature")["Coefficient"]
        )

        standardized = (
            prepared_input.iloc[0][self.means.index]
            - self.means
        ) / self.stds

        contributions = (
            standardized
            .reindex(coefficients.index)
            .fillna(0)
            * coefficients
        ).sort_values(ascending=False)

        return contributions[
            contributions > 0
        ].head(top_n)
