"""Streamlit demo.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.recommend import Recommender

st.set_page_config(page_title="Retail Product Recommender (LightGCN)", layout="wide")


@st.cache_resource
def get_rec() -> Recommender:
    return Recommender()


def show(df: pd.DataFrame):
    out = df.copy()
    if "price" in out:
        out["price"] = out["price"].map(lambda p: "-" if pd.isna(p) else f"${p:,.2f}")
    st.dataframe(out, hide_index=True, width="stretch")


rec = get_rec()
st.title("Retail Product Recommendation using LightGCN")
st.caption("Amazon Reviews 2023 - Video Games (subsample).  Graph neural network over the customer-product purchase graph.")
k = st.sidebar.slider("How many recommendations?", 3, 30, 10)
st.sidebar.write(f"{len(rec.users):,} customers - {len(rec.items):,} products - {len(rec.train):,} training edges")

tab1, tab2 = st.tabs(["Customer ID", "Build a profile (cold start)"])

with tab1:
    # users with a few purchases make a more readable demo
    if "uid" not in st.session_state:
        st.session_state.uid = rec.users.user_id[0]
    if st.button("Pick a random customer"):
        st.session_state.uid = rec.users.user_id.sample(1).iloc[0]
    uid = st.text_input("Customer ID (Amazon user_id)", key="uid").strip()
    if uid:
        try:
            left, right = st.columns(2)
            with left:
                st.subheader("Past purchases (liked items)")
                show(rec.history(uid))
            with right:
                st.subheader("Recommended for this customer")
                show(rec.recommend(uid, k))
        except KeyError:
            st.error("Unknown customer ID. Click 'Pick a random customer' for a valid one.")

with tab2:
    st.write("Search for products you like, add them, and get recommendations - no account needed.")
    q = st.text_input("Search product title (e.g. 'zelda', 'fifa', 'minecraft')")
    pool = rec.items.assign(pop=rec.items.index.map(rec.pop).fillna(0)).sort_values("pop", ascending=False)
    hits = pool[pool.title.str.contains(q, case=False, regex=False)].head(50) if q else pool.head(50)
    label = {a: f"{t[:80]}  [{a}]" for a, t in zip(hits.asin, hits.title)}
    if "picked" not in st.session_state:
        st.session_state.picked = []
    options = list(dict.fromkeys(list(st.session_state.picked) + list(label)))
    titles = dict(zip(rec.items.asin, rec.items.title))
    picked = st.multiselect("Selected products", options, default=st.session_state.picked,
                            format_func=lambda a: f"{titles[a][:80]}  [{a}]")
    st.session_state.picked = picked
    if picked:
        left, right = st.columns(2)
        with left:
            st.subheader("Your profile")
            show(rec.items[rec.items.asin.isin(picked)][["asin", "title", "category", "price"]])
        with right:
            st.subheader("Recommended")
            show(rec.recommend_from_profile(picked, k))
