# app.py
import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from datetime import datetime

from ai_mode import retrieve_top_sections, generate_ai_answer
from web_search import search_cases

# -----------------------------
# Setup
# -----------------------------
nltk.download('punkt', quiet=True)
st.set_page_config(page_title="WAL.AI", layout="centered", initial_sidebar_state="collapsed")

# Load dataset
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# Load model for semantic search
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# Embed sections
@st.cache_data
def embed_sections(sections):
    texts = [
        f"Section {sec.get('Section', '')}: {sec.get('Title', '')}. {sec.get('Description', '')}"
        for sec in sections
    ]
    return model.encode(texts, convert_to_tensor=True)

section_embeddings = embed_sections(sections_data)

# -----------------------------
# UI Header
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
# Input Section
# -----------------------------
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    user_case = st.text_area(
        "Enter your case description or question:",
        placeholder="E.g., 'A person killed someone' or 'What is the punishment for theft under BNS?'",
        height=40,
        key="user_input"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("➜")

# -----------------------------
# Main Logic (AI + Web Search)
# -----------------------------
if submit and user_case.strip():
    query = user_case.strip()
    
    with st.spinner("Analyzing and generating response..."):
        # AI Answer
        retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
        ai_answer = generate_ai_answer(query, retrieved)
        
        # Web Search Cases
        cases = search_cases(query, max_results=5)

    with col2:
        # AI Result
        st.success(ai_answer)
        st.markdown("<h4>Referenced Sections:</h4>", unsafe_allow_html=True)
        for sec, score in retrieved:
            with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                st.write(sec.get('Description', ''))
                st.caption(f"Relevance score: {score:.3f}")

        # Web Cases Result
        st.markdown("<h4>Cases in History:</h4>", unsafe_allow_html=True)
        if cases:
            for case in cases:
                with st.expander(case.get("title", "No title")):
                    st.write(case.get("snippet", "No snippet"))
                    st.markdown(f"[Link]({case.get('link', '#')})")
        else:
            st.warning("No cases found from web search.")
