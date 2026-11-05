"""Step 3 - All experiments for the thesis.

E1  Model comparison (baselines vs ML) under two validation schemes:
      - random 5-fold CV               : "a similar product is already in the database"
      - leave-subgroup-out (Group 5-fold): "a product from a category never seen before"
E2  Feature ablation (META -> +ING -> +TXT -> +KNN)
E3  Generalisation to 6 impact categories (not only climate change)
E4  Conformal prediction intervals (90 %) - coverage and width, per food group
Outputs: results/*.csv, results/oof_*.csv, models/*.joblib
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from features import FeatureBuilder, knn_predict, load

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MOD = ROOT / "models"
RES.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)
SEED = 42
FINAL = "XGBoost"  # chosen after E1/E2: kNN stacking did not help (see e2_ablation.csv)
ALPHA = 0.10  # 90 % intervals

df, ING, EMB, EMB_FR = load()


def xgb():
    return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                        colsample_bytree=0.5, min_child_weight=2, reg_lambda=1.0,
                        random_state=SEED, n_jobs=-1, verbosity=0)


def metrics(y_log, p_log):
    y, p = 10 ** y_log, 10 ** p_log
    ratio = np.abs(y_log - p_log)
    return {
        "R2_log": r2_score(y_log, p_log),
        "MAE_kg": np.mean(np.abs(y - p)),
        "MdAPE_%": 100 * np.median(np.abs(y - p) / y),
        "within_2x_%": 100 * np.mean(ratio < np.log10(2)),
        "within_25%_%": 100 * np.mean(np.abs(y - p) / y < 0.25),
        "Spearman": spearmanr(y, p).correlation,
    }


def splits(scheme):
    if scheme == "random":
        return list(KFold(5, shuffle=True, random_state=SEED).split(df))
    return list(GroupKFold(5).split(df, groups=df["subgroup_fr"]))


def knn_feature(tr, te, y):
    """kNN feature: LOO on train rows (no self-match), train->test for test rows."""
    k_tr, _, _ = knn_predict(EMB[tr], EMB[tr], y[tr], exclude_self=True)
    k_te, _, _ = knn_predict(EMB[te], EMB[tr], y[tr])
    return k_tr, k_te


def fit_predict(method, tr, te, y, blocks=("META", "ING", "TXT")):
    d_tr, d_te = df.iloc[tr], df.iloc[te]
    if method == "Global median":
        return np.full(len(te), np.median(y[tr]))
    if method == "Sub-group median (proxy)":
        # what a practitioner does without data: borrow the category average
        s = pd.Series(y[tr]).groupby(d_tr.subgroup_fr.values).median()
        g = pd.Series(y[tr]).groupby(d_tr.group.values).median()
        p = d_te.subgroup_fr.map(s)
        p = p.fillna(d_te.group.map(g)).fillna(np.median(y[tr]))
        return p.values
    if method == "kNN-embedding (B)":
        return knn_predict(EMB[te], EMB[tr], y[tr])[0]

    use_knn = "KNN" in blocks or method == "Hybrid XGB + kNN"
    fb = FeatureBuilder(blocks=tuple(b for b in blocks if b != "KNN")).fit(d_tr, ING)
    Xtr, Xte = fb.transform(d_tr, ING), fb.transform(d_te, ING)
    if use_knn:
        k_tr, k_te = knn_feature(tr, te, y)
        Xtr, Xte = np.c_[Xtr, k_tr], np.c_[Xte, k_te]
    if method == "Ridge":
        m = Ridge(alpha=3.0)
    elif method == "Random Forest":
        m = RandomForestRegressor(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=SEED)
    else:  # XGBoost / Hybrid
        m = xgb()
    m.fit(Xtr, y[tr])
    return m.predict(Xte)


def run_cv(method, scheme, target="log_gwp", blocks=("META", "ING", "TXT")):
    y = np.log10(df[target].clip(lower=1e-12).values) if target != "log_gwp" else df.log_gwp.values
    oof = np.zeros(len(df))
    for tr, te in splits(scheme):
        oof[te] = fit_predict(method, tr, te, y, blocks)
    return y, oof


METHODS = ["Global median", "Sub-group median (proxy)", "Ridge", "Random Forest",
           "XGBoost", "kNN-embedding (B)", "Hybrid XGB + kNN"]


def e1_model_comparison():
    rows, oofs = [], {}
    for scheme in ["random", "leave-subgroup-out"]:
        for m in METHODS:
            y, p = run_cv(m, scheme)
            rows.append({"scheme": scheme, "method": m, **metrics(y, p)})
            oofs[f"{scheme}|{m}"] = p
            print(f"[E1] {scheme:20s} {m:26s} " + " ".join(f"{k}={v:.3f}" for k, v in rows[-1].items() if k not in ("scheme", "method")))
    pd.DataFrame(rows).to_csv(RES / "e1_model_comparison.csv", index=False)
    o = pd.DataFrame(oofs)
    o.insert(0, "y_log", df.log_gwp.values)
    o.insert(0, "code", df.code.values)
    o.to_csv(RES / "oof_predictions.csv", index=False)


def e2_ablation():
    rows = []
    configs = {"META": ("META",), "META+ING": ("META", "ING"), "META+ING+TXT": ("META", "ING", "TXT"),
               "META+ING+TXT+KNN": ("META", "ING", "TXT", "KNN")}
    for scheme in ["random", "leave-subgroup-out"]:
        for name, b in configs.items():
            y, p = run_cv("XGBoost", scheme, blocks=b)
            rows.append({"scheme": scheme, "features": name, **metrics(y, p)})
            print(f"[E2] {scheme:20s} {name:18s} R2={rows[-1]['R2_log']:.3f} w2x={rows[-1]['within_2x_%']:.1f}")
    pd.DataFrame(rows).to_csv(RES / "e2_ablation.csv", index=False)


IMPACT_LABELS = {
    "gwp_kgco2e": "Climate change (kg CO2e/kg)",
    "ef_score": "EF single score (mPt/kg)",
    "land_use": "Land use (Pt/kg)",
    "water_use": "Water use (m3 world-eq/kg)",
    "marine_eutro": "Marine eutrophication (kg N-eq/kg)",
    "particulate": "Particulate matter (disease inc./kg)",
}


def e3_multi_impact():
    """Log-scale models need positive targets: rows with <=0 impact (e.g. 34 land-use
    credits) are excluded for that impact only, and the count is reported."""
    global df, EMB
    full_df, full_emb = df, EMB
    rows = []
    for t, lab in IMPACT_LABELS.items():
        mask = (full_df[t] > 0).values
        df, EMB = full_df[mask].reset_index(drop=True), full_emb[mask]
        for m in ["Sub-group median (proxy)", FINAL]:
            y, p = run_cv(m, "random", target=t)
            rows.append({"impact": lab, "method": m, "n": int(mask.sum()), **metrics(y, p)})
            print(f"[E3] {lab:40s} {m:26s} n={mask.sum()} R2={rows[-1]['R2_log']:.3f} w2x={rows[-1]['within_2x_%']:.1f}")
    df, EMB = full_df, full_emb
    pd.DataFrame(rows).to_csv(RES / "e3_multi_impact.csv", index=False)


def conformal_q(resid_abs, groups, alpha=ALPHA):
    """Mondrian split-conformal: one quantile per food group (+ global fallback)."""
    def q(r):
        n = len(r)
        lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(r, lvl))
    qs = {g: q(resid_abs[groups == g]) for g in np.unique(groups) if (groups == g).sum() >= 20}
    qs["__global__"] = q(resid_abs)
    return qs


def e4_conformal():
    """Nested: inner CV on the training fold gives calibration residuals, outer fold tests coverage."""
    y = df.log_gwp.values
    rows, per = [], []
    for scheme in ["random", "leave-subgroup-out"]:
        lo, hi, pred = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        for tr, te in splits(scheme):
            inner = np.zeros(len(tr))
            sub = df.iloc[tr].reset_index(drop=True)
            in_splits = (KFold(5, shuffle=True, random_state=SEED).split(sub) if scheme == "random"
                         else GroupKFold(5).split(sub, groups=sub.subgroup_fr))
            for itr, ite in in_splits:
                inner[ite] = fit_predict(FINAL, tr[itr], tr[ite], y)
            qs = conformal_q(np.abs(y[tr] - inner), df.group.values[tr])
            p = fit_predict(FINAL, tr, te, y)
            qv = np.array([qs.get(g, qs["__global__"]) for g in df.group.values[te]])
            pred[te], lo[te], hi[te] = p, p - qv, p + qv
        cover = (y >= lo) & (y <= hi)
        factor = 10 ** ((hi - lo) / 2)
        rows.append({"scheme": scheme, "target_coverage_%": 100 * (1 - ALPHA),
                     "empirical_coverage_%": 100 * cover.mean(), "median_factor_x": np.median(factor)})
        for g in sorted(df.group.unique()):
            mk = df.group.values == g
            per.append({"scheme": scheme, "group": g, "n": int(mk.sum()),
                        "coverage_%": 100 * cover[mk].mean(), "interval_factor_x": float(np.median(factor[mk]))})
        print(f"[E4] {scheme}: coverage={100*cover.mean():.1f}%  median +/- factor={np.median(factor):.2f}x")
    pd.DataFrame(rows).to_csv(RES / "e4_conformal_summary.csv", index=False)
    pd.DataFrame(per).to_csv(RES / "e4_conformal_by_group.csv", index=False)


def final_model():
    """Train the deployable XGBoost model on all data + conformal calibration from 5-fold OOF."""
    y = df.log_gwp.values
    _, oof = run_cv(FINAL, "random")
    qs = conformal_q(np.abs(y - oof), df.group.values)
    fb = FeatureBuilder().fit(df, ING)
    X = fb.transform(df, ING)
    m = xgb().fit(X, y)
    joblib.dump({"model": m, "fb": fb, "feature_names": fb.feature_names, "conformal_q": qs}, MOD / "xgb_gwp.joblib")
    np.save(MOD / "X_full.npy", X)
    json.dump(qs, open(RES / "conformal_quantiles.json", "w"), indent=1)
    print("[final] saved, features:", X.shape)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["e1", "e2", "e3", "e4", "final"]
    for t in todo:
        {"e1": e1_model_comparison, "e2": e2_ablation, "e3": e3_multi_impact,
         "e4": e4_conformal, "final": final_model}[t]()
