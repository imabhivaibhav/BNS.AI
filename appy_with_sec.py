# frontend.py

import streamlit as st
from datetime import datetime

from backend import (
    load_sections,
    load_model,
    embed_sections,
    find_matching_sections,
    run_ai_mode,
)


# ----------------------------------------------------
# Streamlit UI CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="WAL.AI", layout="centered", initial_sidebar_state="collapsed")


# ----------------------------------------------------
# Cached loading for speed
# ----------------------------------------------------
@st.cache_data
def cached_sections():
    return load_sections()

@st.cache_resource
def cached_model():
    return load_model()

@st.cache_data
def cached_embeddings(sections, model):
    return embed_sections(sections, model)


sections_data = cached_sections()
model = cached_model()
section_embeddings = cached_embeddings(sections_data, model)


# ----------------------------------------------------
# UI HEADER
# ----------------------------------------------------
today = datetime.now().strftime("%A, %B %d, %Y")

st.markdown(f"""
<div style="width:100%; text-align:center; font-size:20px; padding:15px;">
    Welcome to <b>WAL.AI</b> — Your Intelligent Legal Assistant<br>
    {today}
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#28a745; font-size:120px;'>WAL.AI</h1>", unsafe_allow_html=True)


# ----------------------------------------------------
# USER INPUT
# ----------------------------------------------------
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    user_case = st.text_area(
        "Enter your case description or legal question:",
        placeholder="Example: 'Punishment for theft' or 'A person killed someone'",
        height=40
    )

    mode_col, gap, button_col = st.columns([5, 2, 2])

    with mode_col:
        mode = st.radio(
            "",
            ["Find Matching Sections", "Ask AI"],
            horizontal=True
        )

    with button_col:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button("Search")


# ----------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------
if submit and user_case.strip():
    query = user_case.strip()

    # --------------------------
    # Mode: Matching Sections
    # --------------------------
    if mode == "Find Matching Sections":
        with st.spinner("Analyzing..."):
            results = find_matching_sections(query, sections_data, model, section_embeddings)

        if not results:
            st.warning("No matching sections found.")
        else:
            st.markdown("### Relevant Sections")
            for idx, score in results:
                sec = sections_data[idx]
                with st.expander(f"Section {sec.get('Section')}: {sec.get('Title')}"):
                    st.write(sec.get("Description"))
                    st.write(f"**Punishment:** {sec.get('Punishment')}")
                    st.caption(f"Relevance Score: {score:.3f}")

    # --------------------------
    # Mode: AI Legal Answer
    # --------------------------
    else:
        with st.spinner("Generating AI response..."):
            ai_answer, retrieved, cases = run_ai_mode(query, sections_data, model, section_embeddings)

        st.success(ai_answer)

        st.markdown("### Referenced Sections")
        for sec, score in retrieved:
            with st.expander(f"Section {sec.get('Section')}: {sec.get('Title')}"):
                st.write(sec.get('Description'))
                st.caption(f"Score: {score:.3f}")

        st.markdown("### Case History")
        if cases:
            for case in cases:
                with st.expander(case.get("title", "Case")):
                    st.write(case.get("snippet", ""))
                    st.markdown(f"[Read More]({case.get('link', '#')})")
        else:
            st.info("No historical cases found.")
