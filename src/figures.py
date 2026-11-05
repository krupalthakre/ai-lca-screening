"""Step 5 - All thesis/slide figures (PNG, 200 dpi) + SHAP explainability.

Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green);
single-hue charts use blue only. Grid and axes kept recessive, no dual axes.
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from subgroups_en import SUBGROUP_EN

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, MOD = ROOT / "results", ROOT / "figures", ROOT / "models"
FIG.mkdir(exist_ok=True)
P = ROOT / "data" / "processed"

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


def short(m):
    return {"Sub-group median (proxy)": "Category average\n(proxy)", "Global median": "Global\nmedian",
            "kNN-embedding (B)": "kNN on text\nembedding (B)", "Hybrid XGB + kNN": "XGBoost +\nkNN feature",
            "Random Forest": "Random\nForest"}.get(m, m)


df = pd.read_csv(P / "products.csv")
FR_EN = {"Bœuf haché": "minced beef", "Abats de bœuf": "beef offal", "Viande de bœuf sans os": "boneless beef",
         "Courgette": "courgette", "Sucre blanc": "white sugar", "Farine de blé": "wheat flour", "Beurre": "butter", "Fromage frais crémeux": "cream cheese", "Poudre de lait": "milk powder",
         "Fromage": "cheese", "Agneau": "lamb", "Porc": "pork", "Poulet": "chicken", "Tomate": "tomato",
         "Oignons": "onions", "Carotte": "carrot", "Lait demi-écrémé": "semi-skimmed milk", "Œuf de poule": "hen egg"}


def f1_distribution():
    order = df.groupby("group").gwp_kgco2e.median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.6))
    data = [df.loc[df.group == g, "gwp_kgco2e"] for g in order]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=C[0], markeredgewidth=0, alpha=.45),
                    medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=INK2),
                    capprops=dict(color=INK2), boxprops=dict(edgecolor=C[0]))
    for b in bp["boxes"]:
        b.set_facecolor("#cde2fb")
    ax.set_yticks(range(1, len(order) + 1), [f"{g} (n={len(d)})" for g, d in zip(order, data)])
    ax.set_xscale("log")
    ax.set_xlabel("Climate change impact (kg CO₂-eq per kg product, log scale)")
    ax.set_title("Footprints span five orders of magnitude", loc="left")
    save(fig, "f1_gwp_by_group")


def f2_stages():
    st = ["agri", "processing", "packaging", "transport", "retail", "consumer"]
    lab = ["Agriculture", "Processing", "Packaging", "Transport", "Retail", "Consumer use"]
    s = df.groupby("group")[[f"gwp_stage_{x}" for x in st]].sum().clip(lower=0)
    s = s.div(s.sum(1), axis=0) * 100
    s = s.sort_values("gwp_stage_agri")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    left = np.zeros(len(s))
    for i, (col, l) in enumerate(zip(s.columns, lab)):
        ax.barh(s.index, s[col], left=left, color=C[i], label=l, edgecolor=SURF, linewidth=1.5, height=0.7)
        left += s[col].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total climate impact (%)")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.4, -0.13), fontsize=8.5, handlelength=1)
    ax.set_title("Agriculture dominates the footprint of most food groups", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "f2_stage_contribution")


def f3_model_comparison():
    r = pd.read_csv(RES / "e1_model_comparison.csv")
    methods = r.method.unique()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for ax, metric, lab in [(axes[0], "R2_log", "R² (log scale)"), (axes[1], "within_2x_%", "Predictions within factor 2 (%)")]:
        x = np.arange(len(methods))
        for k, (sch, nm) in enumerate([("random", "Random 5-fold CV"), ("leave-subgroup-out", "Unseen sub-group (grouped CV)")]):
            v = r[r.scheme == sch].set_index("method").loc[methods, metric]
            b = ax.bar(x + (k - .5) * .38, v, .36, color=C[k], label=nm, edgecolor=SURF, linewidth=1)
            for xi, vi in zip(x + (k - .5) * .38, v):
                if metric == "R2_log" and vi < 0.05:
                    continue  # global median ~0: skip label to avoid overlap
                ax.text(xi, max(vi, 0) + (0.01 if metric == "R2_log" else 1), f"{vi:.2f}" if metric == "R2_log" else f"{vi:.0f}",
                        ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, [short(m) for m in methods], fontsize=7.5)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        if metric == "R2_log":
            ax.set_ylim(min(-0.1, r[metric].min() - .05), 1)
            ax.axhline(0, color=INK2, lw=.8)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("ML beats the category-average proxy under both validation schemes", x=0.01, ha="left", fontweight="bold")
    save(fig, "f3_model_comparison")


def f4_pred_vs_actual():
    o = pd.read_csv(RES / "oof_predictions.csv")
    y, p = 10 ** o.y_log, 10 ** o["random|XGBoost"]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    lim = [3e-4, 80]
    xs = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 50)
    ax.fill_between(xs, xs / 2, xs * 2, color="#cde2fb", alpha=.6, lw=0, label="within factor 2")
    ax.plot(xs, xs, color=INK2, lw=1, ls="--", label="perfect prediction")
    ax.scatter(y, p, s=9, color=C[0], alpha=.55, lw=0)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="Agribalyse value (kg CO₂-eq/kg)", ylabel="Predicted (kg CO₂-eq/kg)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("XGBoost, out-of-fold predictions (n=2451)", loc="left", fontsize=11)
    save(fig, "f4_pred_vs_actual")


def f5_ablation():
    r = pd.read_csv(RES / "e2_ablation.csv")
    feats = r.features.unique()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(feats))
    for k, (sch, nm) in enumerate([("random", "Random 5-fold CV"), ("leave-subgroup-out", "Unseen sub-group")]):
        v = r[r.scheme == sch].set_index("features").loc[feats, "R2_log"]
        ax.plot(x, v, marker="o", color=C[k], lw=2, ms=8, label=nm)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.2f}", (xi, vi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["Metadata\nonly", "+ Ingredients", "+ Name words", "+ kNN text\nembedding"])
    ax.set_ylabel("R² (log scale)")
    ax.set_ylim(0.4, 0.9)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Each information layer adds accuracy; product name is the biggest gain", loc="left", fontsize=11)
    save(fig, "f5_ablation")


def f6_shap():
    b = joblib.load(MOD / "xgb_gwp.joblib")
    X = np.load(MOD / "X_full.npy")
    names = [("sub-group: " + SUBGROUP_EN.get(n[12:], n[12:])) if n.startswith("subgroup_fr=") else n for n in b["feature_names"]]
    names = [n.replace("subgroup_fr=", "subgroup: ").replace("group=", "group: ").replace("delivery=", "delivery: ")
             .replace("prep=", "prep: ").replace("ing:", "ingredient: ").replace("word:", "name word: ")
             .replace("knn_similar_products", "kNN footprint of similar products") for n in names]
    names = [n if not n.startswith("ingredient: ") else "ingredient: " + FR_EN.get(n[12:], n[12:]) for n in names]
    ex = shap.TreeExplainer(b["model"])
    sv = ex.shap_values(X)
    np.save(MOD / "shap_values.npy", sv)
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    imp.to_csv(RES / "shap_importance.csv", header=["mean_abs_shap_log10"])
    top = imp.head(18)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.barh(top.index, top.values, color=C[0], height=.65)
    ax.set_xlabel("Mean |SHAP| (effect on log₁₀ kg CO₂-eq/kg)")
    ax.set_title("What drives the predicted footprint (SHAP)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "f6_shap_importance")

    # direction plot for interpretable binary features: mean SHAP when feature present
    rows = []
    for j, n in enumerate(names):
        mk = X[:, j] > 0
        if n.startswith(("name word", "ingredient", "delivery", "prep")) and mk.sum() >= 15:
            rows.append((n, sv[mk, j].mean(), mk.sum()))
    d = pd.DataFrame(rows, columns=["f", "eff", "n"]).sort_values("eff")
    d = pd.concat([d.head(9), d.tail(9)])
    d["factor"] = 10 ** d.eff
    d["f"] = d.f.replace(FR_EN, regex=True)
    d.to_csv(RES / "shap_direction.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.barh(d.f, d.factor - 1, left=1, color=[C[2] if v < 0 else C[1] for v in d.eff], height=.65)
    ax.axvline(1, color=INK2, lw=.8)
    ax.set_xscale("log")
    tk = [0.5, 0.7, 1, 1.5, 2, 2.5]
    ax.set_xticks(tk, [f"{t:g}×" for t in tk])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Average multiplicative effect on predicted footprint when present (×)")
    ax.set_title("Words & ingredients that push the footprint up or down", loc="left", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "f7_shap_direction")


def f8_multi_impact():
    r = pd.read_csv(RES / "e3_multi_impact.csv")
    imps = r.impact.unique()
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(imps))
    for k, (m, nm) in enumerate([("Sub-group median (proxy)", "Category average (proxy)"), ("XGBoost", "ML model (XGBoost)")]):
        v = r[r.method == m].set_index("impact").loc[imps, "R2_log"]
        ax.barh(y + (k - .5) * .38, v, .36, color=C[k], label=nm)
        for yi, vi in zip(y + (k - .5) * .38, v):
            ax.text(vi + .01, yi, f"{vi:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [i.split(" (")[0] for i in imps])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("R² (log scale), random 5-fold CV")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_title("The approach generalises beyond carbon to other impact categories", loc="left", fontsize=11)
    save(fig, "f8_multi_impact")


def f9_matching():
    r = pd.read_csv(RES / "e5_matching.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, task in zip(axes, r.task.unique()):
        s = r[r.task == task]
        y = np.arange(len(s))
        ax.barh(y + .19, s["top5_acc_%"], .36, color=C[0], label="Correct in top 5")
        ax.barh(y - .19, s["top1_acc_%"], .36, color=C[1], label="Correct at rank 1")
        ax.set_yticks(y, s.method, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_title({"T1 FR->EN": "T1: French name → English database", "T2 short EN query": "T2: 3-word query → full database name"}[task],
                     loc="left", fontsize=10.5)
        ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Module B: automatic mapping of product descriptions to LCA datasets", x=.01, ha="left", fontweight="bold")
    save(fig, "f9_matching")


def f10_conformal():
    r = pd.read_csv(RES / "e4_conformal_by_group.csv")
    r = r[r.scheme == "random"].sort_values("interval_factor_x")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    axes[0].barh(r.group, r["coverage_%"], color=C[0], height=.6)
    axes[0].axvline(90, color=C[1], lw=1.5, ls="--")
    axes[0].text(90.5, -0.6, "90 % target", color=INK2, fontsize=8)
    axes[0].set_xlim(60, 100)
    axes[0].set_xlabel("Empirical coverage of 90 % interval (%)")
    axes[1].barh(r.group, r.interval_factor_x - 1, left=1, color=C[2], height=.6)
    axes[1].set_xlim(1, None)
    axes[1].set_xlabel("Interval half-width (× / ÷ factor)")
    for a in axes:
        a.grid(axis="y", visible=False)
    fig.suptitle("Calibrated uncertainty: honest intervals, wider where data are heterogeneous", x=.01, ha="left", fontweight="bold")
    save(fig, "f10_conformal")


if __name__ == "__main__":
    import sys
    fns = {k: v for k, v in globals().items() if k[:1] == "f" and k[1:2].isdigit()}
    for name, fn in fns.items():
        if len(sys.argv) == 1 or any(name.startswith(a) for a in sys.argv[1:]):
            fn()
