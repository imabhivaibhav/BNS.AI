# appy_with_sec.py

import streamlit as st
from datetime import datetime
import nltk

# Import backend functions
from backend import (
    load_sections,
    load_model,
    embed_sections,
    find_matching_sections,
    run_ai_mode
)

nltk.download('punkt', quiet=True)

st.set_page_config(page_title="WAL.AI", layout="centered", initial_sidebar_state="collapsed")

# -----------------------------
# Load using backend
# -----------------------------
@st.cache_data
def cached_sections():
    return load_sections()
sections_data = cached_sections()

@st.cache_resource
def cached_model():
    return load_model()
model = cached_model()

@st.cache_data
def cached_embeddings():
    return embed_sections(sections_data, model)
section_embeddings = cached_embeddings()

# -----------------------------
# UI HEADER (same as original)
# -----------------------------
today = datetime.now().strftime("%A, %B %d, %Y")
st.markdown(f"""
<div style="width:100%; display:flex; justify-content:center;">
    <div style="text-align:center; font-size:20px; padding:15px; border-radius:10px;">
        Welcome to <b>WAL.AI</b> — your intelligent legal advisor.<br>
        {today}.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#28a745; font-size:140px;'>WAL.AI</h1>", unsafe_allow_html=True)

# -----------------------------
# INPUT AREA (same)
# -----------------------------
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    user_case = st.text_area(
        "Enter your case description or question:",
        placeholder="E.g., 'A person killed someone' or 'Punishment for theft'",
        height=40
    )

    mode_col, spacer, btn_col = st.columns([5, 2, 1])
    with mode_col:
        mode = st.radio("", ["Find Matching Sections", "Ask AI"], horizontal=True)
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button("➜")

# -----------------------------
# MAIN LOGIC (calls backend)
# -----------------------------
if submit and user_case.strip():
    query = user_case.strip()

    # --- SEARCH MODE ---
    if mode == "Find Matching Sections":
        with st.spinner("Finding relevant sections..."):
            results = find_matching_sections(query, sections_data, model, section_embeddings)

        if not results:
            st.warning("No matching sections found.")
        else:
            st.markdown("<h3 style='text-align:center;'>Relevant Section(s):</h3>", unsafe_allow_html=True)
            for idx, score in results:
                sec = sections_data[idx]
                with st.expander(f"Section {sec['Section']}: {sec['Title']}"):
                    st.write(sec["Description"])
                    st.write(f"**Punishment:** {sec.get('Punishment', '')}")
                    st.caption(f"Relevance score: {score:.3f}")

    # --- AI MODE ---
    else:
        with st.spinner("Analyzing and generating response..."):
            ai_answer, retrieved, cases = run_ai_mode(query, sections_data, model, section_embeddings)

        st.success(ai_answer)

        st.markdown("### Referenced Sections")
        for sec, score in retrieved:
            with st.expander(f"Section {sec['Section']}: {sec['Title']}"):
                st.write(sec["Description"])
                st.caption(f"Relevance: {score:.3f}")

        st.markdown("### Cases in History")
        if cases:
            for case in cases:
                with st.expander(case.get("title", "Case")):
                    st.write(case.get("snippet"))
                    st.markdown(f"[Read More]({case.get('link')})")
        else:
            st.info("No cases found.")
