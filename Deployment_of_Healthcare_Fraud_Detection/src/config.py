from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.joblib"
SCALER_PATH = MODEL_DIR / "standard_scaler.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
COEFFICIENTS_PATH = MODEL_DIR / "logistic_coefficients.csv"
