"""AI-assisted rapid LCA screening tool - entry point.

Run from the project root:  streamlit run app/app.py
Pages live in app/app_pages/; shared logic in app/core.py.
"""
import streamlit as st

st.set_page_config(page_title="Rapid LCA screening", page_icon=":material/eco:", layout="wide")

pages = {
    "Screen": [
        st.Page("app_pages/estimate.py", title="Estimate a product", icon=":material/search:", default=True),
        st.Page("app_pages/recipe.py", title="Recipe & portfolio", icon=":material/receipt_long:"),
    ],
    "Understand": [
        st.Page("app_pages/insights.py", title="How the model thinks", icon=":material/psychology:"),
        st.Page("app_pages/about.py", title="About & how to use", icon=":material/info:"),
    ],
}
page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.caption("**Rapid LCA screening with AI/ML**  \nTerm thesis prototype · IIT Kharagpur")
    st.caption("Data: AGRIBALYSE 3.1 (ADEME) · 2,451 French food products · "
               "climate impact per kg, farm to fork")
    st.caption(":material/warning: Screening estimates only — not a replacement for a full ISO 14040/44 LCA.")

page.run()
