"""Central configuration: paths, seeds and default hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CATEGORY = "Video_Games"
HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

SEED = 42
POS_RATING = 4.0          # rating >= 4 counts as a positive interaction
N_USERS_SAMPLE = 30000    # random user subsample (keeps CPU training short)
MIN_USER_TRAIN = 3        # min positive train interactions per kept user
MIN_ITEM_TRAIN = 5        # min positive train interactions per kept item

KS = (5, 10, 20)

# Shared model / training defaults
EMB_DIM = 64
N_LAYERS = 3
LR = 5e-3
REG = 1e-3
BATCH = 4096
MAX_EPOCHS = 60
PATIENCE = 6
