"""Step 2 - Sentence embeddings of product names (module B).

Main model: intfloat/multilingual-e5-small (Wang et al., 2024), 384-dim, 100+ languages.
E5 models expect a prefix: "passage: " for database entries, "query: " for user input.
Chosen after comparing 4 encoders (see matching_eval.py / results/e5_matching.csv).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = {"intfloat/multilingual-e5-small": ("query: ", "passage: "),
          "intfloat/multilingual-e5-base": ("query: ", "passage: ")}


@lru_cache(maxsize=4)
def get_model(name=MODEL_NAME):
    """Use the locally cached model if present (fast, offline); otherwise download it once
    from the Hugging Face Hub (e.g. on a fresh cloud server)."""
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(name, local_files_only=True)
    except Exception:
        return SentenceTransformer(name)


def encode(texts, kind="query", name=MODEL_NAME):
    q, p = PREFIX.get(name, ("", ""))
    pre = q if kind == "query" else p
    return get_model(name).encode([pre + t for t in texts], batch_size=64,
                                  normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def main():
    df = pd.read_csv(P / "products.csv")
    np.save(P / "emb_name_en.npy", encode(df.name_en.tolist(), "passage"))  # database side
    np.save(P / "emb_name_fr.npy", encode(df.name_fr.tolist(), "query"))    # query side (T1)
    print("saved embeddings with", MODEL_NAME)


if __name__ == "__main__":
    main()
