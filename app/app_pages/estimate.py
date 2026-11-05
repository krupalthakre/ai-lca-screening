import altair as alt
import pandas as pd
import streamlit as st

from core import (SIM_OK, SUBGROUP_EN, INGREDIENT_EN, FB, df, match, meta_from_row, ingredients_of, estimate,
                  waterfall, why_sentence, confidence, level, stage_chart, alternatives,
                  group_position, contributions)

EXAMPLES = {
    "Frozen chicken nuggets": "frozen breaded chicken nuggets",
    "Hamburger": "hamburger",
    "Dark chocolate": "dark chocolate bar 70%",
    "Oat drink": "oat drink",
    "Lentil soup": "lentil soup",
    "Fromage de chèvre (FR)": "fromage de chèvre",
    "Car tyre (out of scope)": "car tyre",
}


def use_example():
    pick = st.session_state.get("example")
    if pick:
        st.session_state.q = EXAMPLES[pick]
        st.session_state.example = None


st.title("Estimate a product's carbon footprint", icon=":material/eco:")
st.caption("Type any food product. The AI finds similar products in an expert LCA database, "
           "estimates the footprint, tells you how sure it is, and explains why.")

if "q" not in st.session_state:
    st.session_state.q = "frozen breaded chicken nuggets"
text = st.text_input("Product description (English or French)", key="q", bind="query-params",
                     placeholder="e.g. frozen breaded chicken nuggets", icon=":material/search:")
st.pills("Try an example", list(EXAMPLES), key="example", on_change=use_example)

if not text or not text.strip():
    st.info("Type a product above or pick an example to get started.", icon=":material/arrow_upward:")
    st.stop()

top, sims = match(text.strip(), 5)
best = df.iloc[top[0]]

# ------------------------------------------------------------------ product details (editable)
lv = FB.cat_levels
with st.expander("Product details used by the model — pre-filled from the closest match, edit if you know better",
                 icon=":material/tune:"):
    a, b, c = st.columns(3)
    meta = {
        "group": a.selectbox("Food group", lv["group"], lv["group"].index(best.group)),
        "subgroup_fr": b.selectbox("Sub-group", lv["subgroup_fr"], lv["subgroup_fr"].index(best.subgroup_fr),
                                   format_func=lambda x: SUBGROUP_EN.get(x, x)),
        "delivery": c.selectbox("Storage", lv["delivery"], lv["delivery"].index(best.delivery)),
        "prep": a.selectbox("Preparation at home", lv["prep"], lv["prep"].index(best.prep)),
        "packaging_src": b.selectbox("Packaging data", lv["packaging_src"], lv["packaging_src"].index(best.packaging_src),
                                     format_func=lambda x: {"PACK AGB": "Product-specific",
                                                            "PACK PROXY": "Generic proxy"}.get(x, x)),
        "season_code": float(best.season_code),
        "air_code": float(c.toggle("Air-freighted", bool(best.air_code),
                                   help="Very rare in the training data (2 products) — the model cannot learn this "
                                        "effect well, so treat air-freighted estimates with caution.")),
    }
    ingredients = st.multiselect("Recipe ingredients (for mixed foods)", FB.ing_vocab, ingredients_of(best.code),
                                 format_func=lambda x: INGREDIENT_EN.get(x, x))
    st.caption("Tip: change *Storage* or remove an ingredient and watch the estimate and explanation update.")

r = estimate(text.strip(), meta, ingredients)
conf, conf_color, conf_text = confidence(float(sims[0]), r["q"])
lev, lev_color = level(r["est"])

# ------------------------------------------------------------------ headline result
if sims[0] < SIM_OK:
    st.error(f"The closest match (*{best.name_en}*) has a similarity of only {sims[0]:.2f} (in-scope ≥ {SIM_OK}). "
             "This product is probably **outside what the model knows** (French food products), so the number "
             "below is not meaningful. If you typed in French or another language, try English — "
             "cross-language matches score lower.", icon=":material/block:", title="Probably out of scope")

with st.container(border=True):
    st.markdown(f"#### :material/insights: Result for *{text.strip()}*")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated footprint", f"{r['est']:.2f}", help="kg CO₂-equivalent per kg of product, "
              "from farm to fork (farming, processing, packaging, transport, shop, home).")
    m2.metric("Likely range (90 %)", f"{r['lo']:.1f} – {r['hi']:.1f}",
              help="In testing, the true value fell inside this range 91 % of the time.")
    m3.metric("Closest expert value", f"{best.gwp_kgco2e:.2f}",
              help=f"Full LCA value of the most similar database product: {best.name_en}")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"Impact level: {lev}", color=lev_color, icon=":material/speed:")
        st.badge(f"Confidence: {conf}", color=conf_color, icon=":material/verified:")
        st.caption(conf_text + " All values in kg CO₂-eq per kg of product.")

    pct_all = (df.gwp_kgco2e < r["est"]).mean() * 100
    pct_g, med_g = group_position(r["est"], meta["group"])
    st.progress(min(int(pct_all), 100),
                text=f"Higher than **{pct_all:.0f} %** of all foods · higher than **{pct_g:.0f} %** of "
                     f"*{meta['group']}* (group median {med_g:.2f})")

# ------------------------------------------------------------------ explanation
left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader("Why this number?", icon=":material/lightbulb:")
    st.markdown(f"Starting from a typical food in the database, {why_sentence(r)}")
    st.altair_chart(waterfall(r), alt="Waterfall chart from the database average to the estimate")
    st.caption("Read top to bottom: each bar multiplies the running value. Orange raises the footprint, "
               "green lowers it. Explanations are SHAP values from the trained model — they describe what "
               "the **model** learned, not proven causes (e.g. *chilled* mostly signals fresh meat or dairy).")

with right:
    st.subheader("Closest products in the database", icon=":material/database:")
    st.dataframe(pd.DataFrame({
        "Product": df.name_en.values[top],
        "kg CO₂-eq/kg": df.gwp_kgco2e.values[top],
        "Match": sims}), hide_index=True,
        column_config={"Product": st.column_config.TextColumn(width=190),
                       "kg CO₂-eq/kg": st.column_config.NumberColumn(format="%.2f", width="small"),
                       "Match": st.column_config.ProgressColumn(min_value=0.7, max_value=1.0, format="%.2f",
                                                                help="Similarity of meaning (≥ 0.85 = in scope)")},
        alt="Five most similar database products")

    st.subheader("Where the closest product's footprint comes from", icon=":material/stacked_bar_chart:")
    st.altair_chart(stage_chart(best), alt="Life-cycle stage breakdown of the closest product")
    st.caption(f"Expert LCA breakdown for *{best.name_en}* (descriptive only — the model never sees these numbers).")

    alts = alternatives(text.strip(), r["est"])
    st.subheader("Lower-impact similar options", icon=":material/swap_horiz:")
    if len(alts):
        for _, a_ in alts.iterrows():
            st.markdown(f"- **{a_.name_en}** — {a_.gwp_kgco2e:.2f} kg CO₂-eq/kg "
                        f"(:green[−{(1 - a_.gwp_kgco2e / r['est']):.0%}])")
    else:
        st.caption("No clearly lower-impact similar product found in the database.")

with st.expander("Detailed driver table", icon=":material/table_view:"):
    c = contributions(r, 12)
    st.dataframe(c[["feature", "factor"]].rename(columns={"feature": "Feature", "factor": "Effect (×)"}),
                 hide_index=True, column_config={"Effect (×)": st.column_config.NumberColumn(format="×%.2f")},
                 alt="SHAP effects of active features")
