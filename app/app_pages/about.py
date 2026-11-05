import streamlit as st

st.title("About this tool", icon=":material/info:")
st.markdown("A full **Life Cycle Assessment (LCA)** of a product takes weeks. This prototype gives a "
            "**fast first estimate** of a food's carbon footprint — with an honest uncertainty range and an "
            "explanation — so you can decide which products deserve a full LCA.")

st.subheader("How it works", icon=":material/route:")
cols = st.columns(4)
steps = [
    (":material/search:", "1 · Describe", "Type a product name in English or French."),
    (":material/database:", "2 · Match", "A language model (E5) finds the most similar of 2,451 expert-assessed foods."),
    (":material/model_training:", "3 · Estimate", "An XGBoost model predicts the footprint from category, storage, "
                                                 "ingredients and name words."),
    (":material/lightbulb:", "4 · Explain", "You get a likely range, a confidence level and the drivers behind the number."),
]
for col, (icon, head, body) in zip(cols, steps):
    with col.container(border=True, height="stretch"):
        st.markdown(f"### {icon}")
        st.markdown(f"**{head}**")
        st.caption(body)

st.subheader("How to read the results", icon=":material/menu_book:")
with st.expander("What does “kg CO₂-eq per kg” mean?", icon=":material/co2:"):
    st.markdown("All greenhouse gases (CO₂, methane, nitrous oxide…) converted to the warming effect of CO₂, "
                "per kilogram of food, from the farm to your plate. Example: 2.0 means producing and eating 1 kg "
                "of this food warms the climate like 2 kg of CO₂.")
with st.expander("What is the “likely range”?", icon=":material/straighten:"):
    st.markdown("A 90 % prediction interval built with *conformal prediction*. When tested on products the "
                "model had not seen, the true value fell inside the range 91 % of the time. Wider range = less certain.")
with st.expander("What do the orange and green bars mean?", icon=":material/bar_chart:"):
    st.markdown("They are **SHAP** explanations. Starting from the average food, each feature multiplies the "
                "estimate up (orange) or down (green). They show what the model learned — useful for understanding, "
                "but they are correlations, not proof of cause.")
with st.expander("What do “confidence” and “match” mean?", icon=":material/verified:"):
    st.markdown("**Match** is how close in meaning your description is to a database product (0–1). Below **0.85** "
                "the product is probably outside what the model knows and you get a warning (French or other non-English queries score a little lower, so English works best). **Confidence** combines "
                "the match with the width of the likely range.")

st.subheader("Data and method", icon=":material/science:")
st.markdown("""
- **Data:** AGRIBALYSE 3.1 by ADEME (French Agency for Ecological Transition), open licence — 2,451 food products
  with full LCA results (EF 3.0 method).
- **No cheating:** the model only uses information known *before* an LCA. Stage-by-stage results are never inputs.
- **Validation:** tested on random products *and* on whole food categories hidden during training.
- **Result:** 91 % of estimates within ×2 of the expert value, versus 77 % for the usual category-average shortcut.
""")

st.subheader("Limitations", icon=":material/report:")
st.warning("""
- French food system only — numbers do not transfer directly to India or other countries.
- Rare but important factors (e.g. **air freight**, exotic origins) are poorly learned.
- Only which ingredients are present is known, not their amounts.
- For screening and prioritisation — **not** for product claims, eco-labels or ISO-compliant comparisons.
""", icon=":material/warning:", title="Use responsibly")
