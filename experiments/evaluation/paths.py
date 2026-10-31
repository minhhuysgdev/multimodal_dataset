"""Central paths for D1/D2/D3 evaluation pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Sources
D1_CLEAN = ROOT / "data/raw/hasaki/d1_dataset.json"
D2_VERIFIED = ROOT / "data/d2_dataset.json"
D2_FULL_VERIFIED_SOURCE = ROOT / "data/d2_dataset.json"
D3_TEMPLATE = ROOT / "data/gold/d3_dataset.json"
D3_GOLD = ROOT / "data/gold/d3_dataset.json"

# Train sets
D1_TRAIN = ROOT / "data/datasets/d1_annotator_train.json"
D2_TRAIN = ROOT / "data/datasets/d2_verified_train.json"

# Meta
D2_BUILD_META = ROOT / "data/datasets/d2_build_meta.json"
D3_SAMPLING_META = ROOT / "data/gold/d3_sampling_meta.json"
TRAIN_META = ROOT / "data/datasets/dataset_train_meta.json"

# Results
RESULTS_DIR = ROOT / "experiments/evaluation/results"
