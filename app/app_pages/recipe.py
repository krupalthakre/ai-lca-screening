import altair as alt
import pandas as pd
import streamlit as st

from core import SIM_OK, UP, df, match, meta_from_row, ingredients_of, estimate, alternatives

st.title("Screen a recipe or product portfolio", icon=":material/receipt_long:")
st.caption("List items with their mass. Each line is matched to the database, estimated, and ranked — "
           "so you can see the **hotspot** that deserves attention first.")

EXAMPLE = pd.DataFrame({
    "item": ["beef mince 15% fat", "parboiled rice, cooked", "tomato sauce with onions", "cheddar cheese", "olive oil"],
    "mass_g": [125, 150, 80, 30, 10]})

with st.form("recipe"):
    bom = st.data_editor(EXAMPLE, num_rows="dynamic", hide_index=True, alt="Recipe items and masses",
                         column_config={"item": st.column_config.TextColumn("Item", required=True, width="large"),
                                        "mass_g": st.column_config.NumberColumn("Mass (g)", min_value=0, step=5,
                                                                                required=True)})
    go = st.form_submit_button("Screen recipe", type="primary", icon=":material/play_arrow:")

if go:
    out = []
    with st.spinner("Matching and estimating each item…"):
        for it, g in zip(bom["item"], bom["mass_g"]):
            if not isinstance(it, str) or not it.strip() or pd.isna(g):
                continue
            t, s = match(it.strip(), 1)
            row = df.iloc[t[0]]
            r = estimate(it.strip(), meta_from_row(row), ingredients_of(row.code))
            kg = g / 1000
            out.append({"Item": it, "Matched dataset": row.name_en, "Match": float(s[0]),
                        "kg CO₂e per kg": r["est"], "Mass (g)": g, "kg CO₂e": r["est"] * kg,
                        "Low": r["lo"] * kg, "High": r["hi"] * kg})
    st.session_state.recipe_results = pd.DataFrame(out).sort_values("kg CO₂e", ascending=False)

res = st.session_state.get("recipe_results")
if res is None or res.empty:
    st.info("Edit the table above (or keep the example) and press **Screen recipe**.", icon=":material/info:")
    st.stop()

total = res["kg CO₂e"].sum()
res = res.assign(**{"Share": res["kg CO₂e"] / total})
hot = res.iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Total footprint", f"{total:.2f} kg CO₂-eq", border=True,
          help="Sum of item estimates. Item ranges are not added up, because their errors are not independent.")
c2.metric("Total mass", f"{res['Mass (g)'].sum():.0f} g", border=True)
c3.metric("Biggest hotspot", hot["Item"], f"{hot['Share']:.0%} of total", delta_color="off", border=True)

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Where the footprint comes from", icon=":material/donut_large:")
    chart = alt.Chart(res).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=alt.Y("Item:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("kg CO₂e:Q", title="kg CO₂-eq in this recipe"),
        color=alt.condition(alt.datum.Item == hot["Item"], alt.value(UP), alt.value("#2a78d6")),
        tooltip=["Item", "Matched dataset", alt.Tooltip("kg CO₂e:Q", format=".3f"), alt.Tooltip("Share:Q", format=".0%")])
    st.altair_chart(chart.properties(height=40 * len(res) + 20), alt="Footprint contribution of each recipe item")

    alts = alternatives(hot["Item"], hot["kg CO₂e per kg"], 3)
    with st.container(border=True):
        st.markdown(f":material/tips_and_updates: **Hotspot: {hot['Item']}** contributes "
                    f"**{hot['Share']:.0%}** of the total.")
        if len(alts):
            st.markdown("Similar lower-impact options to consider:")
            for _, a in alts.iterrows():
                saving = (hot["kg CO₂e per kg"] - a.gwp_kgco2e) * hot["Mass (g)"] / 1000
                st.markdown(f"- {a.name_en} — {a.gwp_kgco2e:.2f} kg/kg (saves ≈ {saving:.2f} kg CO₂-eq in this recipe)")

with right:
    st.subheader("Item details", icon=":material/table_view:")
    st.dataframe(res, hide_index=True, alt="Per-item screening results",
                 column_order=["Item", "Matched dataset", "Match", "Mass (g)", "kg CO₂e per kg", "kg CO₂e", "Share"],
                 column_config={
                     "Match": st.column_config.ProgressColumn(min_value=0.7, max_value=1.0, format="%.2f"),
                     "kg CO₂e per kg": st.column_config.NumberColumn(format="%.2f"),
                     "kg CO₂e": st.column_config.NumberColumn(format="%.3f"),
                     "Share": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    low = res[res.Match < SIM_OK]
    if len(low):
        st.warning("Weak match for: " + ", ".join(low.Item) + ". Rephrase or check these lines manually.",
                   icon=":material/help:")
    st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                       file_name="recipe_screening.csv", mime="text/csv", icon=":material/download:")
