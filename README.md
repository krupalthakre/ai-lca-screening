# AI/ML-assisted rapid Life Cycle Assessment (LCA) screening

Term thesis (B.Tech/Dual, IIT Kharagpur) — *Integration of AI/ML in Life Cycle Assessment*.

**Question:** Can machine learning give a fast, explainable, uncertainty-aware estimate of a
product's environmental footprint *before* a full LCA is done — and can NLP automate the
tedious step of mapping product descriptions to LCA database entries?

| Module | What it does | Method |
|---|---|---|
| **A — Surrogate impact model** | predicts 6 impact indicators from pre-LCA information | XGBoost on metadata + recipe ingredients + name words; SHAP; Mondrian conformal 90 % intervals |
| **B — Text-to-dataset mapping** | maps free text (EN/FR) to the closest Agribalyse datasets | multilingual-E5 sentence embeddings (benchmarked against TF-IDF, MiniLM, LaBSE, E5-base) |

## Data
Agribalyse 3.1 (ADEME, France) — open licence, downloaded from data.gouv.fr / data.ademe.fr:
synthesis (2,451 products × 16 EF 3.0 indicators), detail per ingredient, detail per life-cycle stage.

## Reproduce
```bash
pip install pandas numpy scikit-learn xgboost shap sentence-transformers streamlit matplotlib joblib
python src/prepare_data.py     # clean -> data/processed/
python src/embed.py            # E5 embeddings of names
python src/experiments.py      # E1-E4 + final model -> results/, models/
python src/matching_eval.py    # E5 text-mapping benchmark
python src/ood_threshold.py    # out-of-scope similarity threshold
python src/figures.py          # figures/ + SHAP
streamlit run app/app.py       # demo (multi-page: estimate, recipe, insights, about)
```

## Folder map
- `data/raw` original CSVs · `data/processed` clean tables + embeddings
- `src/` all code · `results/` metric tables · `figures/` plots · `models/` trained model
- `app/app.py` Streamlit demo · `thesis/` report and slides
