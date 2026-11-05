# Results summary (source of truth for thesis + slides)

All numbers are out-of-fold (5-fold CV, seed 42). Target = log10(kg CO2-eq / kg product).
Data: Agribalyse 3.1, n = 2,451 products, 1,094 with recipes, 286 distinct ingredients.
Final model: XGBoost, 635 features (META + 125 ingredients + 421 name words).

## E1 — Model comparison (climate change)  `e1_model_comparison.csv`, `f3`
| Method | R² log (random) | within 2× | R² log (unseen sub-group) | within 2× |
|---|---|---|---|---|
| Global median | −0.01 | 44.0 % | −0.03 | 42.2 % |
| Sub-group median (proxy) | 0.62 | 77.4 % | 0.33 | 54.2 % |
| Ridge | 0.81 | 89.2 % | 0.57 | 69.0 % |
| Random Forest | 0.83 | 91.0 % | 0.58 | 69.9 % |
| **XGBoost (final)** | **0.84** | **91.1 %** | **0.62** | **72.4 %** |
| kNN on E5 embedding | 0.71 | 82.7 % | 0.25 | 56.3 % |
| XGBoost + kNN feature | 0.83 | 90.7 % | 0.56 | 69.7 % |
XGBoost: MdAPE 19.1 %, MAE 1.49 kg, Spearman 0.92 (proxy 31.8 %, 2.80 kg, 0.82).

## E2 — Ablation (XGBoost)  `e2_ablation.csv`, `f5`
Random: META 0.70 → +ING 0.75 → +TXT 0.84 → +KNN 0.83. Unseen: 0.48 → 0.52 → 0.62 → 0.56.

## E3 — Six impact categories (random CV, R² log proxy → XGBoost)  `e3_multi_impact.csv`, `f8`
Climate 0.62→0.84 · EF score 0.64→0.84 · Land use 0.58→0.78 (n=2417) · Water 0.27→0.64 (n=2450) ·
Marine eutro 0.59→0.78 · Particulates 0.70→0.86.

## E4 — Mondrian conformal 90 %  `e4_*.csv`, `f10`
Random: coverage 91.1 %, ×/÷ 2.0. Unseen sub-group: 87.2 %, ×/÷ 3.3; failures: dairy 46 %, culinary aids 71 %, sugary 72 %.

## E5 — Mapping (module B)  `e5_matching.csv`, `f9`
E5-small: T2 terse query top-1 75.0 / top-5 94.9 / borrow<2× 96.8; T1 FR→EN 47.1 / 66.0. LaBSE best on T1 (59.7 / 76.8).
MiniLM ≈ TF-IDF cross-lingually (28 %). TF-IDF+E5 hybrid worse than E5 alone.

## SHAP  `shap_importance.csv`, `shap_direction.csv`, `f6`, `f7`
Top: group Meat/eggs/fish, delivery Chilled (proxy for animal products, NOT causal), F&V group, Beverages, butter, n_ingredients, beef.
Effects: lamb ×2.35, minced beef ×2.12, beef ×2.01, veal ×1.69, chocolate ×1.46; chicken ×0.79, turkey ×0.74, liver ×0.67.

## Largest errors (good defence material)
Tap water (unique); air-freighted mango/beans (only 2 air-freight products → effect unlearnable);
cocoa powder, coffee, cocoa butter (tropical crops inside low-impact groups).

## App examples
Chicken nuggets: 4.80 (90 % 2.41–9.56), closest dataset 3.87. "car tyre": sim 0.83 → out-of-domain warning
(model still outputs 6.02 behind it → interval ≠ domain check).

## E6 — Out-of-scope threshold  `e6_ood_threshold.csv` (src/ood_threshold.py)
Threshold 0.85 (E5-small): flags 20/20 out-of-domain probes (non-food goods + paneer/idli/dhokla) and 0.04 % of
2,451 in-domain terse queries. Max OOD similarity 0.848 (petrol); in-domain 1st percentile 0.873.
Caveat: cross-lingual queries score lower (French "yaourt nature" correct match at 0.81) → app suggests English.
