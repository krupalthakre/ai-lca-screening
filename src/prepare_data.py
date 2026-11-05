"""Step 1 - Clean Agribalyse 3.1 and build a modelling table.

Inputs  : data/raw/agribalyse31_*.csv  (ADEME, open licence, data.gouv.fr)
Outputs : data/processed/products.csv     one row per product, clean English columns
          data/processed/ingredients.csv  long table (product code, ingredient name)

Design rule (important for the thesis defence): NO feature may be computed from
LCA results. Stage-wise impacts and per-ingredient impacts sum to the target, so
using them would be target leakage. We only use information a practitioner has
BEFORE doing an LCA: product name, food category, storage/delivery mode,
packaging data source, consumer preparation, seasonality/air-freight flags and
(for composite foods) the list of recipe ingredients.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_EN = {
    "viandes, œufs, poissons": "Meat, eggs, fish",
    "fruits, légumes, légumineuses et oléagineux": "Fruits, vegetables, legumes, nuts",
    "produits céréaliers": "Cereal products",
    "entrées et plats composés": "Starters and mixed dishes",
    "lait et produits laitiers": "Milk and dairy",
    "aides culinaires et ingrédients divers": "Culinary aids and misc.",
    "boissons": "Beverages",
    "matières grasses": "Fats and oils",
    "produits sucrés": "Sugary products",
    "aliments infantiles": "Baby food",
    "glaces et sorbets": "Ice cream and sorbets",
}
DELIVERY_EN = {
    "Glacé": "Chilled",
    "Ambiant (long)": "Ambient (long shelf life)",
    "Ambiant (moyen)": "Ambient (medium shelf life)",
    "Ambiant (court)": "Ambient (short shelf life)",
    "Congelé": "Frozen",
}
PREP_EN = {
    "Pas de préparation": "No preparation",  # not "None": pandas reads that as NaN
    "Micro-onde": "Microwave",
    "Four": "Oven",
    "Cuisson à l'eau": "Boiling",
    "Poêle": "Pan-frying",
    "Réfrigéré chez le consommateur": "Refrigerated at home",
}

IMPACTS = {
    "Changement climatique": "gwp_kgco2e",            # kg CO2-eq / kg product
    "Score unique EF": "ef_score",                    # mPt / kg (EF 3.0 single score)
    "Utilisation du sol": "land_use",                 # Pt / kg
    "Épuisement des ressources eau": "water_use",     # m3 world-eq / kg
    "Eutrophisation marine": "marine_eutro",          # kg N-eq / kg
    "Particules fines": "particulate",                # disease incidence / kg
}


def load_products() -> pd.DataFrame:
    df = pd.read_csv(RAW / "agribalyse31_synthese.csv", low_memory=False, encoding="utf-8")
    df = df.dropna(subset=["Code AGB"]).copy()  # file has ~12.5k empty trailing rows
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame({
        "code": df["Code AGB"].astype(str).str.strip(),
        "name_fr": df["Nom du Produit en Français"].str.strip(),
        "name_en": df["LCI Name"].str.strip(),
        "group_fr": df["Groupe d'aliment"].str.strip(),
        "subgroup_fr": df["Sous-groupe d'aliment"].str.strip(),
        "delivery_fr": df["Livraison"].str.strip(),
        "packaging_src": df["Approche emballage"].str.strip(),
        "prep_fr": df["Préparation"].str.strip(),
        "season_code": df["code saison"].astype(float),
        "air_code": df["code avion"].astype(float),
        "dqr": df["DQR"].astype(float),
    })
    out["group"] = out["group_fr"].map(GROUP_EN)
    out["delivery"] = out["delivery_fr"].map(DELIVERY_EN)
    out["prep"] = out["prep_fr"].map(PREP_EN)
    for fr, en in IMPACTS.items():
        out[en] = df[fr].astype(float).values

    # Stage-wise climate impact - kept ONLY for descriptive analysis (never a feature)
    st = pd.read_csv(RAW / "agribalyse31_detail-par-etape.csv", low_memory=False, encoding="utf-8")
    st = st.dropna(subset=["Code AGB"]).copy()
    st["code"] = st["Code AGB"].astype(str).str.strip()
    stages = {"Agriculture": "agri", "Transformation": "processing", "Emballage": "packaging",
              "Transport": "transport", "Supermarché et distribution": "retail",
              "Consommation": "consumer"}
    cols = {f"Changement climatique - {k}": f"gwp_stage_{v}" for k, v in stages.items()}
    st = st[["code", *cols]].rename(columns=cols).drop_duplicates("code")
    out = out.drop_duplicates("code").merge(st, on="code", how="left")

    assert out[list(IMPACTS.values())].notna().all().all()
    assert out["group"].notna().all() and out["delivery"].notna().all() and out["prep"].notna().all()
    return out


def load_ingredients() -> pd.DataFrame:
    ing = pd.read_csv(RAW / "agribalyse31_detail-par-ingredient.csv", low_memory=False, encoding="utf-8")
    ing = ing.dropna(subset=["Ciqual  AGB"]).copy()
    ing["code"] = ing["Ciqual  AGB"].astype(str).str.strip()
    ing["ingredient_fr"] = ing["Ingredients"].str.strip()
    ing = ing[ing["ingredient_fr"] != "Autres étapes"]  # "other steps" = non-ingredient bucket
    return ing[["code", "ingredient_fr"]].drop_duplicates()


def main():
    prod = load_products()
    ing = load_ingredients()
    n_ing = ing.groupby("code").size().rename("n_ingredients")
    prod = prod.merge(n_ing, on="code", how="left")
    prod["n_ingredients"] = prod["n_ingredients"].fillna(0).astype(int)
    prod["is_composite"] = (prod["n_ingredients"] > 0).astype(int)
    prod["log_gwp"] = np.log10(prod["gwp_kgco2e"])

    prod.to_csv(OUT / "products.csv", index=False, encoding="utf-8")
    ing.to_csv(OUT / "ingredients.csv", index=False, encoding="utf-8")
    print(f"products: {len(prod)}  composite: {prod.is_composite.sum()}  "
          f"ingredient links: {len(ing)}  unique ingredients: {ing.ingredient_fr.nunique()}")
    print(prod.groupby("group")["gwp_kgco2e"].describe()[["count", "50%", "max"]].round(2))


if __name__ == "__main__":
    main()
