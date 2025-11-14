# app.py
import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
import nltk

from ai_mode import retrieve_top_sections, generate_ai_answer
from web_search import search_cases

nltk.download('punkt', quiet=True)

st.set_page_config(page_title="WAL.AI", layout="centered")

# Load dataset
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# Load model
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

# UI
today = datetime.now().strftime("%A, %B %d, %Y")
st.markdown(f"<div style='text-align:center'>Welcome to <b>WAL.AI</b> — {today}</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; color:#28a745; font-size:120px;'>WAL.AI</h1>", unsafe_allow_html=True)

# Input
user_case = st.text_area("Enter your case or question:", height=60, placeholder="Describe your case or ask a question")

mode = st.radio("", ["Find Matching Sections", "Ask AI"], horizontal=True)
submit = st.button("➜")

if submit and user_case.strip():
    query = user_case.strip()

    if mode == "Find Matching Sections":
        # Semantic search
        subqueries = re.split(r",| and | or ", query)
        subqueries = [q.strip() for q in subqueries if q.strip()]
        matched = {}
        for sq in subqueries:
            sq_emb = model.encode(sq, convert_to_tensor=True)
            sims = util.cos_sim(sq_emb, section_embeddings)[0]
            top_k = min(10, len(sims))
            top_idx = torch.argsort(sims, descending=True)[:top_k]
            top_scores = sims[top_idx]
            threshold = max(0.45, float(torch.median(top_scores)) - 0.05)
            for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
                if score >= threshold:
                    matched[idx] = max(matched.get(idx, 0), float(score))

        if matched:
            sorted_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10]
            indices, scores = zip(*sorted_matched)
        else:
            indices, scores = [], []

        if not indices:
            st.warning("No matching sections found.")
        else:
            st.markdown("<h3>Relevant Sections:</h3>", unsafe_allow_html=True)
            for idx, score in zip(indices, scores):
                sec = sections_data[idx]
                with st.expander(f"Section {sec['Section']}: {sec['Title']}"):
                    st.markdown(f"**Description:** {sec['Description']}")
                    st.markdown(f"**Punishment:** {sec.get('Punishment', 'N/A')}")
                    st.caption(f"Relevance score: {score:.3f}")

    elif mode == "Ask AI":
        with st.spinner("Generating AI response..."):
            retrieved_sections = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
            retrieved_cases = search_cases(query, max_results=5)
            ai_answer = generate_ai_answer(query, retrieved_sections, retrieved_cases)

        st.success(ai_answer)

        if retrieved_sections:
            st.markdown("<h4>Referenced Sections:</h4>", unsafe_allow_html=True)
            for sec, score in retrieved_sections:
                with st.expander(f"Section {sec['Section']}: {sec['Title']}"):
                    st.write(sec['Description'])
                    st.caption(f"Relevance score: {score:.3f}")

        if retrieved_cases:
            st.markdown("<h4>Cases in History:</h4>", unsafe_allow_html=True)
            for c in retrieved_cases:
                with st.expander(f"{c['title']}"):
                    st.write(c['snippet'])
                    st.markdown(f"[Link]({c['link']})")
