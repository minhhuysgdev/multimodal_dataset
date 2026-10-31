# Phase 2: Retrieval Baseline Evaluation

Đánh giá hiệu quả retrieval của các phương pháp trước khi dùng cho annotation pipeline.

## Baselines

| ID | Method | Description |
|----|--------|-------------|
| R1 | Regex/Token | Token overlap, inverted index |
| R2 | Dense Embedding | Sentence-Transformers cosine similarity |
| R3 | Hybrid | Union of R1 + R2 |
| R4 | TechniqueRAG-inspired | R3 → GPT-4o rerank → top-k exemplars |

## Metrics

- **Recall@k** (k=1,3,5,10): Có candidate đúng L3 trong top-k không?
- **MRR**: Mean Reciprocal Rank của candidate đầu tiên đúng
- **Taxonomy Coverage**: Số lượng unique L3 intents trong top-k

## Usage

### R1-R3 (Training-free)

```bash
cd /Users/huynguyen/Documents/research/mutli\ intentclassification\ data
python experiments/phase2_retrieval/evaluate_r1_r2_r3.py
```

Output: `experiments/phase2_retrieval/results/`
- `R1_regex_results.json`
- `R2_embedding_results.json`
- `R3_hybrid_results.json`
- `retrieval_summary.json`

### R4 (Requires OpenAI API)

```bash
export OPENAI_API_KEY="your-key"
python experiments/phase2_retrieval/evaluate_r4_reranking.py
```

Output: `experiments/phase2_retrieval/results/R4_techniquerag_results.json`

## Data Sources

- **Corpus**: `data/datasets/d1_annotator_train.json` (1454 samples, D3 excluded)
- **Test queries**: `data/gold/d3_human_gold.json` (300 samples)
- **Taxonomy**: `data/unified_intents.csv`

## Notes

- All retrievers exclude D3 sample_ids from corpus to prevent leakage
- R2 uses `paraphrase-multilingual-MiniLM-L12-v2` (configurable)
- R4 reranking prompt follows TechniqueRAG methodology adapted for hierarchical Vietnamese intents
