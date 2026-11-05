"""Feature construction shared by experiments and the demo app.

Feature blocks (used for the ablation study):
  META : food group, sub-group, delivery/storage, consumer preparation,
         packaging data source, season & air-freight flags, recipe size
  ING  : multi-hot recipe ingredients (composite foods only; 0 otherwise)
  TXT  : bag-of-words from the English product name (interpretable in SHAP)
  KNN  : similarity-weighted footprint of the k most similar *other* products
         by sentence embedding (module B output, used as a stacked feature)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"

CAT_COLS = ["group", "subgroup_fr", "delivery", "prep", "packaging_src"]
NUM_COLS = ["season_code", "air_code", "n_ingredients", "is_composite"]
STOP = {"with", "and", "in", "of", "or", "for", "the", "a", "to", "without", "from", "type", "e", "g"}


def load():
    df = pd.read_csv(P / "products.csv")
    ing = pd.read_csv(P / "ingredients.csv", dtype={"code": str})
    df["code"] = df["code"].astype(str)
    emb_en = np.load(P / "emb_name_en.npy")
    emb_fr = np.load(P / "emb_name_fr.npy")
    return df, ing, emb_en, emb_fr


class FeatureBuilder:
    """Fit on training rows only (vocabularies), then transform any rows."""

    def __init__(self, blocks=("META", "ING", "TXT"), min_ing=5, min_word=5):
        self.blocks = blocks
        self.min_ing = min_ing
        self.min_word = min_word

    def fit(self, df, ing):
        self.cat_levels = {c: sorted(df[c].dropna().unique()) for c in CAT_COLS}
        tr_ing = ing[ing.code.isin(df.code)]
        vc = tr_ing.ingredient_fr.value_counts()
        self.ing_vocab = sorted(vc[vc >= self.min_ing].index)
        self.cv = CountVectorizer(binary=True, min_df=self.min_word, stop_words=list(STOP),
                                  token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        self.cv.fit(df["name_en"].str.lower())
        return self

    def transform(self, df, ing):
        parts, names = [], []
        if "META" in self.blocks:
            for c in CAT_COLS:
                for lv in self.cat_levels[c]:
                    parts.append((df[c] == lv).astype(np.float32).values[:, None])
                    names.append(f"{c}={lv}")
            for c in NUM_COLS:
                parts.append(df[c].astype(np.float32).values[:, None])
                names.append(c)
        if "ING" in self.blocks:
            idx = {code: i for i, code in enumerate(df.code)}
            M = np.zeros((len(df), len(self.ing_vocab)), np.float32)
            col = {g: j for j, g in enumerate(self.ing_vocab)}
            sub = ing[ing.code.isin(idx) & ing.ingredient_fr.isin(col)]
            for code, g in zip(sub.code, sub.ingredient_fr):
                M[idx[code], col[g]] = 1
            parts.append(M)
            names += [f"ing:{g}" for g in self.ing_vocab]
        if "TXT" in self.blocks:
            parts.append(self.cv.transform(df["name_en"].str.lower()).toarray().astype(np.float32))
            names += [f"word:{w}" for w in self.cv.get_feature_names_out()]
        X = np.hstack(parts) if parts else np.zeros((len(df), 0), np.float32)
        self.feature_names = names
        return X


def knn_predict(E_query, E_ref, y_ref, k=10, exclude_self=False, T=0.01):
    """Softmax(similarity / T)-weighted mean of y over the k nearest reference rows (cosine).
    k=10, T=0.01 (E5's training temperature) chosen on a small grid, see thesis Sec. 4."""
    S = E_query @ E_ref.T
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, nn, axis=1)
    w = np.exp((s - s.max(1, keepdims=True)) / T)
    pred = (w * y_ref[nn]).sum(1) / w.sum(1)
    return pred, nn, s
