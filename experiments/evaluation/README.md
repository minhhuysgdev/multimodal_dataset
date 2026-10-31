# Evaluation Pipeline (D1 / D2 / D3)

Pipeline đánh giá tách khỏi annotator notebook.

## Dataset pipeline

```bash
# Step 0: Full verify ALL ~2011 samples
#   Colab: open notebooks/run_full_reverify_colab.ipynb
#   Local: python scripts/run_full_reverify.py
#   Output: data/raw/hasaki/hasaki_labelled_full_verified.json

# Step 1: Build D2 from full verify (preferred)
python scripts/build_d2_verified_dataset.py --mode full

# OR legacy partial (1603 clean + 99 reverified only — NOT full verify)
# python scripts/build_d2_verified_dataset.py --mode partial

# Step 2: Sample D3 human gold template (300)
python scripts/sample_d3_human_gold.py --n 300 --seed 42

# Step 3: Human annotate CSV → save as data/gold/d3_human_gold.json

# Step 4: Build train sets (exclude D3)
python scripts/build_d1_d2_train.py --d3-file data/gold/d3_human_gold.json
```

## Evaluation notebook

Open [`notebooks/evaluation_d1_d2_d3.ipynb`](../../notebooks/evaluation_d1_d2_d3.ipynb) for:
- Table 1: dataset statistics
- Table 2: label quality (D1/D2 vs D3)
- Table 3: downstream TF-IDF+SVM (+ optional transformers)

## CLI scripts

| Script | Purpose |
|--------|---------|
| `evaluate_labels.py` | D1/D2 vs human gold metrics |
| `train_tfidf_svm.py` | TF-IDF + LinearSVC baseline |
| `train_transformer.py` | PhoBERT-base / XLM-R-base |

Paths: see `experiments/evaluation/paths.py`

## Expected counts

| Dataset | N |
|---------|--:|
| D2 verified | 1702 |
| D3 gold | 300 |
| D1_train / D2_train | ~1402 |
