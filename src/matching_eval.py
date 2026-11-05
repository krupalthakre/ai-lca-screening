"""Step 4 - Module B evaluation: mapping free-text product descriptions to LCA datasets.

In practice an LCA analyst must map each item of a bill of materials / recipe
("frozen breaded chicken") to the right database process - slow manual work.
We test automatic mapping in two realistic settings:

  T1 cross-lingual : query = French product name, corpus = English LCI names
  T2 short query   : query = first 3 words of the English name (terse user input)
Methods: lexical TF-IDF (character 3-5 grams), four sentence encoders, and a
hybrid (lexical + E5) score.
Metrics: top-1 / top-5 accuracy, and the climate-impact error if the analyst
simply uses the top-1 match's footprint ("match-and-borrow").
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from embed import MODEL_NAME, encode
from features import load

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

df, _, EMB_EN, EMB_FR = load()
names = df.name_en.values
y = df.log_gwp.values
ENCODERS = {
    "MiniLM-L12 multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "LaBSE": "sentence-transformers/LaBSE",
    "E5-base multilingual": "intfloat/multilingual-e5-base",
    "E5-small multilingual (chosen)": MODEL_NAME,
}


def topk_eval(S, task, method):
    order = np.argsort(-S, axis=1)[:, :5]
    hit = names[order] == names[:, None]  # identical English names count as correct
    top1 = order[:, 0]
    err = np.abs(y[top1] - y)
    return {"task": task, "method": method,
            "top1_acc_%": 100 * hit[:, 0].mean(), "top5_acc_%": 100 * hit.any(1).mean(),
            "borrowed_footprint_within_2x_%": 100 * np.mean(err < np.log10(2)),
            "same_subgroup_top1_%": 100 * np.mean(df.subgroup_fr.values[top1] == df.subgroup_fr.values)}


def lexical(q, corpus):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(list(corpus) + list(q))
    return (v.transform(q) @ v.transform(corpus).T).toarray()


def main():
    corpus = pd.Series(names).str.lower()
    short = pd.Series(names).str.replace(",", " ").str.split().str[:3].str.join(" ").str.lower()
    queries = {"T1 FR->EN": df.name_fr.tolist(), "T2 short EN query": short.tolist()}
    rows = []
    for task, q in queries.items():
        L = lexical(pd.Series(q).str.lower(), corpus)
        rows.append(topk_eval(L, task, "TF-IDF char n-gram"))
        for lab, mn in ENCODERS.items():
            S = encode(q, "query", mn) @ encode(names.tolist(), "passage", mn).T
            rows.append(topk_eval(S, task, lab))
            if mn == MODEL_NAME:
                rows.append(topk_eval(0.5 * L + 0.5 * S, task, "Hybrid (TF-IDF + E5-small)"))
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(RES / "e5_matching.csv", index=False)
    print(out.round(1).to_string(index=False))

    # Qualitative examples for slides (T1, chosen encoder)
    S = EMB_FR @ EMB_EN.T
    rng = np.random.default_rng(7)
    ex = []
    for i in rng.choice(len(df), 10, replace=False):
        j = int(np.argmax(S[i]))
        ex.append({"query_fr": df.name_fr[i], "matched_en": names[j], "true_en": names[i],
                   "cos": round(float(S[i, j]), 3), "true_gwp": round(10 ** y[i], 2), "matched_gwp": round(10 ** y[j], 2)})
    pd.DataFrame(ex).to_csv(RES / "e5_matching_examples.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
