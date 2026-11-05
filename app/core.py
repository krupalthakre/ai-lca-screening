"""Shared logic for the screening app: resources, mapping (Module B), estimation (Module A),
and the explanation helpers used by several pages."""
import os
import sys
from pathlib import Path


import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from embed import encode  # noqa: E402
from features import load  # noqa: E402
from subgroups_en import SUBGROUP_EN  # noqa: E402
from ingredients_en import INGREDIENT_EN  # noqa: E402

UP, DOWN, NEUTRAL = "#eb6834", "#1baf7a", "#2a78d6"
SIM_OK = 0.85      # calibrated in src/ood_threshold.py: catches 20/20 out-of-domain probes,
                   # flags 0.04 % of in-domain queries (results/e6_ood_threshold.csv)
PLANT_SWAPS = {"plats végétariens", "substituts de viande", "substituts de charcuterie", "légumineuses"}
SIM_GOOD = 0.90
STAGES = [("agri", "Farming"), ("processing", "Processing"), ("packaging", "Packaging"),
          ("transport", "Transport"), ("retail", "Shop"), ("consumer", "At home")]


@st.cache_resource(show_spinner="Loading the food database, AI model and language model (first start ~40 s)…")
def resources():
    df, ing, emb, _ = load()
    b = joblib.load(ROOT / "models" / "xgb_gwp.joblib")
    explainer = shap.TreeExplainer(b["model"])
    encode(["warm-up"], "query")
    return df, ing, emb, b, explainer


_LAZY = {"df", "ING", "EMB", "B", "EXPLAINER", "FB", "MODEL", "QS", "FNAMES", "BASE"}


def _state():
    """Load heavy resources on first use (thread-safe via st.cache_resource), never at import time.
    Loading at import caused an ImportError when two browser sessions started simultaneously."""
    df_, ing, emb, b, ex = resources()
    g = globals()
    if "FNAMES" not in g:
        g.update(df=df_, ING=ing, EMB=emb, B=b, EXPLAINER=ex, FB=b["fb"], MODEL=b["model"],
                 QS=b["conformal_q"], FNAMES=b["feature_names"],
                 BASE=float(np.atleast_1d(ex.expected_value)[0]))  # mean prediction (log10)
    return g


def __getattr__(name):  # PEP 562: `from core import df` triggers lazy loading
    if name in _LAZY:
        return _state()[name]
    raise AttributeError(name)


@st.cache_data(max_entries=500, show_spinner=False)
def match(text: str, k: int = 5):
    """Module B: cosine similarity between the query and all 2,451 dataset names (E5-small)."""
    _state()
    s = encode([text], "query")[0] @ EMB.T
    top = np.argsort(-s)[:k]
    return top, s[top]


def meta_from_row(r):
    return {c: r[c] for c in ["group", "subgroup_fr", "delivery", "prep", "packaging_src", "season_code", "air_code"]}


def ingredients_of(code):
    _state()
    return [i for i in ING.loc[ING.code == str(code), "ingredient_fr"].tolist() if i in FB.ing_vocab]


def estimate(text, meta, ingredients):
    """Module A: XGBoost estimate (log10), conformal half-width q, SHAP values, feature vector."""
    _state()
    row = pd.DataFrame([{**meta, "code": "__query__", "name_en": text,
                         "n_ingredients": len(ingredients), "is_composite": int(len(ingredients) > 0)}])
    ing = pd.DataFrame({"code": ["__query__"] * len(ingredients), "ingredient_fr": ingredients})
    X = FB.transform(row, ing)
    p = float(MODEL.predict(X)[0])
    q = QS.get(meta["group"], QS["__global__"])
    sv = EXPLAINER.shap_values(X)[0]
    return {"log": p, "q": q, "est": 10 ** p, "lo": 10 ** (p - q), "hi": 10 ** (p + q), "shap": sv, "x": X[0]}


# ---------------------------------------------------------------- explanation helpers
def pretty(n):
    if n.startswith("subgroup_fr="):
        return "Sub-group: " + SUBGROUP_EN.get(n[12:], n[12:])
    m = {"group=": "Food group: ", "delivery=": "Storage: ", "prep=": "Preparation: ",
         "packaging_src=": "Packaging data: ", "ing:": "Ingredient: ", "word:": "Word in name: "}
    if n.startswith("ing:"):
        return "Ingredient: " + INGREDIENT_EN.get(n[4:], n[4:])
    for k, v in m.items():
        if n.startswith(k):
            return v + n[len(k):].replace("PACK AGB", "specific").replace("PACK PROXY", "proxy")
    return {"n_ingredients": "Number of ingredients", "is_composite": "Is a recipe (composite)",
            "season_code": "Seasonality", "air_code": "Air-freighted"}.get(n, n)


def contributions(r, top=8):
    """Active features (present / non-zero) sorted by |SHAP|; remaining effects grouped as 'Other'."""
    _state()
    d = pd.DataFrame({"feature": [pretty(n) for n in FNAMES], "value": r["x"], "shap": r["shap"]})
    act = d[d.value != 0].reindex(d[d.value != 0].shap.abs().sort_values(ascending=False).index)
    shown = act.head(top).copy()
    other = r["shap"].sum() - shown.shap.sum()  # includes effects of absent features
    if abs(other) > 1e-6:
        shown = pd.concat([shown, pd.DataFrame([{"feature": "All other features", "value": 0, "shap": other}])])
    shown["factor"] = 10 ** shown.shap
    return shown.reset_index(drop=True)


def waterfall(r, top=7):
    """Waterfall in kg CO2e/kg: database average -> each driver (multiplicative) -> estimate."""
    _state()
    c = contributions(r, top)
    rows, cur = [], BASE
    rows.append({"step": "Typical food in database", "start": 0, "end": 10 ** BASE, "kind": "base",
                 "label": f"{10 ** BASE:.2f}"})
    for _, s in c.iterrows():
        nxt = cur + s.shap
        rows.append({"step": s.feature, "start": 10 ** cur, "end": 10 ** nxt,
                     "kind": "up" if s.shap > 0 else "down", "label": f"×{10 ** s.shap:.2f}"})
        cur = nxt
    rows.append({"step": "Estimate for your product", "start": 0, "end": 10 ** cur, "kind": "total",
                 "label": f"{10 ** cur:.2f}"})
    d = pd.DataFrame(rows)
    d["order"] = range(len(d))
    color = alt.Color("kind:N", scale=alt.Scale(domain=["base", "up", "down", "total"],
                                                range=["#9aa5a0", UP, DOWN, "#1a7f5a"]), legend=None)
    y = alt.Y("step:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=280, labelOverlap=False))
    bars = alt.Chart(d).mark_bar(cornerRadius=3, height=18).encode(
        y=y, x=alt.X("start:Q", title="kg CO₂-eq per kg"), x2="end:Q", color=color,
        tooltip=[alt.Tooltip("step:N", title="Step"), alt.Tooltip("label:N", title="Effect"),
                 alt.Tooltip("end:Q", title="Running value", format=".2f")])
    text = alt.Chart(d).mark_text(align="left", dx=4, fontSize=11, color="#3c4a43").encode(
        y=y, x=alt.X("xmax:Q"), text="label:N").transform_calculate(xmax="max(datum.start, datum.end)")
    return (bars + text).properties(height=30 * len(d) + 10)


def why_sentence(r):
    c = contributions(r, 10)
    c = c[c.feature != "All other features"]
    up = c[c.shap > 0.02].head(2)    # ignore effects smaller than about ×1.05
    dn = c[c.shap < -0.02].head(2)

    def one(f, v):
        if ": " in f:
            kind, val = f.split(": ", 1)
            return f"**{val}** ({kind.lower()}, ×{10 ** v:.2f})"
        return f"**{f.lower()}** (×{10 ** v:.2f})"
    fmt = lambda s: ", ".join(one(f, v) for f, v in zip(s.feature, s.shap))
    parts = []
    if len(up):
        parts.append(f"the estimate is pushed **up** mainly by {fmt(up)}")
    if len(dn):
        parts.append(f"pulled **down** by {fmt(dn)}")
    return "; ".join(parts) + "." if parts else "No single feature dominates this estimate."


def confidence(sim, q):
    width = 10 ** q
    if sim >= SIM_GOOD and width <= 2.2:
        return "High", "green", "Very close match in the database and a narrow uncertainty range."
    if sim >= SIM_OK:
        return "Medium", "orange", "Reasonable match; the true value could be about ×/÷ %.1f the estimate." % width
    return "Low", "red", "No close match — the product is probably outside what the model knows."


def level(est):
    """Plain-language band using database quartiles (25% / 50% / 75%: 1.37 / 2.78 / 6.57)."""
    _state()
    q1, q2, q3 = np.percentile(df.gwp_kgco2e, [25, 50, 75])
    if est < q1:
        return "Low", "green"
    if est < q2:
        return "Below average", "blue"
    if est < q3:
        return "Above average", "orange"
    return "High", "red"


def stage_chart(row):
    vals = [(lab, max(float(row[f"gwp_stage_{k}"]), 0)) for k, lab in STAGES]
    d = pd.DataFrame(vals, columns=["stage", "kg"])
    tot = d.kg.sum()
    d["share"] = d.kg / tot if tot > 0 else 0
    return alt.Chart(d).mark_bar(height=26).encode(
        x=alt.X("sum(kg):Q", title="kg CO₂-eq per kg", stack="zero"),
        color=alt.Color("stage:N", sort=[s for _, s in STAGES],
                        scale=alt.Scale(domain=[s for _, s in STAGES],
                                        range=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]),
                        legend=alt.Legend(orient="top", title=None, columns=3, symbolType="square")),
        order=alt.Order("stage_order:Q"),
        tooltip=["stage:N", alt.Tooltip("kg:Q", format=".2f"), alt.Tooltip("share:Q", format=".0%")]
    ).transform_calculate(stage_order="indexof(%s, datum.stage)" % [s for _, s in STAGES]).properties(height=130)


def alternatives(text, est, n=4):
    """Lower-impact options that can plausibly replace the product: same sub-group as the best match
    (plus plant-based substitutes for meat), at least 30 % lower, ranked by similarity to the query."""
    _state()
    top, sims = match(text, 1)
    if sims[0] < SIM_OK:
        return df.iloc[[]]
    bi = int(top[0])
    best = df.iloc[bi]
    lower = df.gwp_kgco2e < 0.7 * est
    # same-type swaps: same sub-group, ranked by similarity to the matched product (product-to-product)
    same = df[lower & (df.subgroup_fr == best.subgroup_fr) & (df.index != bi)]
    same = same.assign(similarity=EMB[same.index] @ EMB[bi]).sort_values("similarity", ascending=False)
    parts = [same.head(n if best.group != "Meat, eggs, fish" else 2)]
    if best.group == "Meat, eggs, fish":  # plus plant-based substitutes, ranked by the user's wording
        plant = df[lower & df.subgroup_fr.isin(PLANT_SWAPS)]
        plant = plant.assign(similarity=encode([text], "query")[0] @ EMB[plant.index].T)
        parts.append(plant.sort_values("similarity", ascending=False).head(2))
    return pd.concat(parts).drop_duplicates("name_en").head(n)


def group_position(est, group):
    _state()
    g = df.loc[df.group == group, "gwp_kgco2e"]
    return (g < est).mean() * 100, g.median()
