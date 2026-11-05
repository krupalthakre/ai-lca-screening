import altair as alt
import pandas as pd
import streamlit as st

from core import RES, UP, DOWN, NEUTRAL


@st.cache_data
def tables():
    return {n: pd.read_csv(RES / f"{n}.csv") for n in
            ["e1_model_comparison", "e2_ablation", "e3_multi_impact", "e4_conformal_summary",
             "e4_conformal_by_group", "e5_matching", "shap_importance", "shap_direction"]}


T = tables()
st.title("How the model thinks", icon=":material/psychology:")
st.caption("What the model learned from 2,451 expert LCAs, and how well it performs when tested on products it never saw.")

e1 = T["e1_model_comparison"].set_index(["scheme", "method"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Within ×2 of expert value", f"{e1.loc[('random', 'XGBoost'), 'within_2x_%']:.0f} %",
          f"{e1.loc[('random', 'XGBoost'), 'within_2x_%'] - e1.loc[('random', 'Sub-group median (proxy)'), 'within_2x_%']:+.0f} pts vs category average",
          border=True)
c2.metric("Accuracy on unseen categories (R²)", f"{e1.loc[('leave-subgroup-out', 'XGBoost'), 'R2_log']:.2f}",
          f"{e1.loc[('leave-subgroup-out', 'XGBoost'), 'R2_log'] - e1.loc[('leave-subgroup-out', 'Sub-group median (proxy)'), 'R2_log']:+.2f} vs category average",
          border=True)
cov = T["e4_conformal_summary"].set_index("scheme").loc["random", "empirical_coverage_%"]
c3.metric("Range contains true value", f"{cov:.0f} %", "target 90 %", delta_color="off", border=True)
m5 = T["e5_matching"]
top5 = m5[(m5.task == "T2 short EN query") & m5.method.str.contains("chosen")]["top5_acc_%"].iloc[0]
c4.metric("Right product in top 5 matches", f"{top5:.0f} %", border=True)

st.subheader("What drives footprints overall", icon=":material/leaderboard:")
left, right = st.columns(2, gap="large")
with left:
    imp = T["shap_importance"].rename(columns={"Unnamed: 0": "feature", "mean_abs_shap_log10": "imp"}).head(12)
    imp["feature"] = imp.feature.str.replace("name word: ", "word: ").str.replace("n_ingredients", "number of ingredients")
    st.altair_chart(alt.Chart(imp).mark_bar(color=NEUTRAL, cornerRadiusEnd=3).encode(
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("imp:Q", title="Average influence (mean |SHAP|)"),
        tooltip=["feature", alt.Tooltip("imp:Q", format=".3f")]).properties(height=470),
        alt="Global feature importance")
    st.caption("Most important overall: the food group, storage mode and words like *beef* or *chocolate*. "
               "*Chilled* ranks high because chilled foods are mostly fresh meat and dairy — a correlation, not a cause.")
with right:
    d = T["shap_direction"].copy()
    d["dir"] = d.eff.map(lambda v: "raises" if v > 0 else "lowers")
    d["one"] = 1.0
    d["f"] = d.f.str.replace("name word: ", "word: ")
    st.altair_chart(alt.Chart(d).mark_bar(cornerRadius=3).encode(
        y=alt.Y("f:N", sort=alt.SortField("factor", order="descending"), title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        x=alt.X("factor:Q", title="Average effect when present (×, log scale)",
                scale=alt.Scale(type="log", domain=[0.6, 2.6])), x2="one:Q",
        color=alt.Color("dir:N", scale=alt.Scale(domain=["raises", "lowers"], range=[UP, DOWN]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["f", alt.Tooltip("factor:Q", format=".2f"), alt.Tooltip("n:Q", title="products")]).properties(height=470),
        alt="Directional effects of words and ingredients")
    st.caption("Lamb ≈ ×2.4, beef ≈ ×2.0 push estimates up; chicken, turkey and pork pull down relative to "
               "other meats — the same ranking found in the food-LCA literature.")

st.subheader("Is it better than the usual shortcut?", icon=":material/compare_arrows:")
scheme = st.segmented_control("Test type", ["Similar products exist", "Category never seen"],
                              default="Similar products exist", required=True,
                              help="'Category never seen' hides a whole sub-group (e.g. all cheeses) during training.")
key = "random" if scheme == "Similar products exist" else "leave-subgroup-out"
e = T["e1_model_comparison"]
e = e[e.scheme == key].replace({"Sub-group median (proxy)": "Category average (today)", "Hybrid XGB + kNN": "XGBoost + kNN",
                                "kNN-embedding (B)": "kNN on text"})
e["is_final"] = e.method.eq("XGBoost").map({True: "Final model", False: "Other"})
st.altair_chart(alt.Chart(e).mark_bar(cornerRadiusEnd=3).encode(
    y=alt.Y("method:N", sort="-x", title=None),
    x=alt.X("within_2x_%:Q", title="% of products estimated within ×2 of the expert value", scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("is_final:N", scale=alt.Scale(domain=["Final model", "Other"], range=["#1a7f5a", "#b9c7bf"]), legend=None),
    tooltip=["method", alt.Tooltip("within_2x_%:Q", format=".1f"), alt.Tooltip("R2_log:Q", format=".2f")]).properties(height=260),
    alt="Model comparison")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Works beyond carbon", icon=":material/water_drop:")
    m = T["e3_multi_impact"].copy()
    m["impact"] = m.impact.str.split(" \\(").str[0]
    m["method"] = m.method.replace({"Sub-group median (proxy)": "Category average", "XGBoost": "ML model"})
    st.altair_chart(alt.Chart(m).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("impact:N", title=None), yOffset="method:N",
        x=alt.X("R2_log:Q", title="Accuracy (R², log scale)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("method:N", scale=alt.Scale(domain=["Category average", "ML model"], range=["#b9c7bf", "#1a7f5a"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["impact", "method", alt.Tooltip("R2_log:Q", format=".2f")]).properties(height=320),
        alt="Accuracy for six impact categories")
with right:
    st.subheader("Are the ranges honest?", icon=":material/target:")
    g = T["e4_conformal_by_group"]
    g = g[g.scheme == "random"]
    st.altair_chart((alt.Chart(g).mark_circle(color=NEUTRAL, size=160, opacity=1).encode(
        y=alt.Y("group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        x=alt.X("coverage_%:Q", title="% of true values inside the 90 % range", scale=alt.Scale(domain=[80, 100])),
        tooltip=["group", alt.Tooltip("coverage_%:Q", format=".1f"), alt.Tooltip("interval_factor_x:Q", format=".2f")])
        + alt.Chart(pd.DataFrame({"x": [90]})).mark_rule(color=UP, strokeDash=[4, 3]).encode(x="x:Q")).properties(height=360),
        alt="Interval coverage by food group")
    st.caption("Dashed line = 90 % target. Every food group is close to it. Honest caveat: for *entirely new* dairy "
               "sub-groups coverage drops to about 46 %, so ranges for unusual new products can be too narrow.")

with st.expander("All result tables", icon=":material/table:"):
    for n, t in T.items():
        st.markdown(f"**{n}**")
        st.dataframe(t.round(3), hide_index=True, alt=n)
