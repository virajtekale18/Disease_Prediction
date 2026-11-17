"""
config.py - one place for paths and settings shared by every stage.

Why a config file?  If a path or setting changes, you change it here once
instead of hunting through five scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_DIR is the folder this file lives in (disease_prediction/).
# Using __file__ means the scripts work no matter which folder you run them from.
PROJECT_DIR = Path(__file__).resolve().parent

# The raw CSVs sit one level up, in "ML Project/". We read them in place and
# never modify them.
DATA_DIR = PROJECT_DIR.parent
TRAIN_CSV = DATA_DIR / "Training.csv"
TEST_CSV = DATA_DIR / "Testing.csv"

PROCESSED_DIR = PROJECT_DIR / "data"          # cleaned train/test splits (Stage 2)
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
VIS_DIR = PROJECT_DIR / "visualizations"

# ---------------------------------------------------------------------------
# Dataset facts (confirmed by inspecting the files, not assumed)
# ---------------------------------------------------------------------------
TARGET_COL = "prognosis"   # the only text column; holds the disease name

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # fixed seed so every run gives the same split/results
TEST_SIZE = 0.25           # share of unique patterns held out for the final test

# Cross-validation. After de-duplication the smallest disease has only 5 unique
# patterns, so ~3-4 land in training. 5 folds would leave some folds without a
# single example of that disease, so we use 3 folds and repeat 5 times with
# different shuffles to get a stable average (15 fits per model).
CV_FOLDS = 3
CV_REPEATS = 5

# "Missing symptom" robustness test. Real patients rarely report every symptom.
# Because the clean data is so easy, we also score models after randomly hiding
# this many of each validation/test patient's symptoms (always keeping >= 1).
SYMPTOMS_TO_HIDE = 2

# Chart colours (a single blue for bars, white->blue for heatmaps)
BAR_COLOR = "#2a78d6"
HEATMAP_LIGHT = "#fcfcfb"
HEATMAP_DARK = "#0d3a73"


# Final bundle produced by Stage 8 and used by prediction.py / app.py
FINAL_MODEL_PATH = MODELS_DIR / "disease_model.joblib"
CANDIDATES_PATH = MODELS_DIR / "candidate_models.joblib"


def ensure_dirs() -> None:
    """Create the output folders if they don't exist yet."""
    for d in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, VIS_DIR):
        d.mkdir(parents=True, exist_ok=True)
