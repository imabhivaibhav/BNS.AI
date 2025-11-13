import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from datetime import datetime

from ai_mode import retrieve_top_sections, generate_ai_answer
from wal_ai_history import search_history_ui

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
        👋 Welcome to <b>WAL.AI</b> — your intelligent legal advisor.<br>
        {today}.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#28a745; font-size:140px;'>WAL.AI</h1>", unsafe_allow_html=True)

# -----------------------------
# Input Section (Centered)
# -----------------------------
st.markdown("""
<style>
.input-button-container {
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: center;
    margin-top: 20px;
}
.text-input {
    width: 500px;
    min-height: 40px;
    max-height: 300px;
    resize: none;
    padding: 8px;
    font-size: 16px;
}
.circular-btn {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #28a745;
    color: white;
    border: none;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 20px;
}
.rotate {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="input-button-container">', unsafe_allow_html=True)

# Input box
user_case = st.text_area(
    "",
    placeholder="Type your case or question here...",
    height=40,
    key="user_input",
    label_visibility="collapsed"
)

# Hidden Streamlit button for submit logic
submit = st.button("hidden_submit", key="submit_trigger", label_visibility="collapsed")

# Circular search button (visual)
btn_placeholder = st.empty()
btn_html = '<button class="circular-btn" id="search-btn">➜</button>'
btn_placeholder.markdown(btn_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Mode selector
mode = st.radio(
    "",
    ["Find Matching Sections", "Ask AI"],
    horizontal=True,
    key="mode_inline"
)

search_history_ui()

# -----------------------------
# Rotate button while processing
# -----------------------------
if submit and user_case.strip():
    st.components.v1.html("""
    <script>
    const btn = window.parent.document.getElementById('search-btn');
    if(btn){ btn.classList.add('rotate'); }
    </script>
    """, height=0)

# -----------------------------
# Main Logic
# -----------------------------
if submit and user_case.strip():
    query = user_case.strip()

    # --- SEARCH MODE ---
    if mode == "Find Matching Sections":
        with st.spinner("Finding relevant sections..."):
            section_numbers = re.findall(r"\d+", query)
            subqueries = re.split(r",| and | or ", query)
            subqueries = [q.strip() for q in subqueries if q.strip()]

            matched = {}
            for i, s in enumerate(sections_data):
                sec_num = "".join(re.findall(r"\d+", s.get("Section", "")))
                if any(num == sec_num for num in section_numbers):
                    matched[i] = 1.0

            if subqueries and (not section_numbers or len(subqueries) > len(section_numbers)):
                for sq in subqueries:
                    sq_emb = model.encode(sq, convert_to_tensor=True)
                    sims = util.cos_sim(sq_emb, section_embeddings)[0]

                    top_k = min(10, len(sims))
                    top_idx = torch.argsort(sims, descending=True)[:top_k]
                    top_scores = sims[top_idx]
                    median_score = float(torch.median(top_scores))
                    threshold = max(0.45, median_score - 0.05)

                    for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
                        if score >= threshold:
                            matched[idx] = max(matched.get(idx, 0), float(score))

            if matched:
                sorted_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10]
                indices, scores = zip(*sorted_matched)
            else:
                indices, scores = [], []

        if not indices:
            st.warning("No matching sections found. Try describing your case differently.")
        else:
            st.markdown("<h3 style='text-align:center;'>Relevant Section(s):</h3>", unsafe_allow_html=True)
            for idx, score in zip(indices, scores):
                sec = sections_data[idx]
                with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                    st.markdown(f"**Description:** {sec.get('Description', '')}")
                    st.markdown(f"**Punishment:** {sec.get('Punishment', '')}")
                    st.caption(f"Relevance score: {score:.3f}")

    # --- AI MODE ---
    elif mode == "Ask AI":
        with st.spinner("Analyzing and generating response..."):
            retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
            ai_answer = generate_ai_answer(query, retrieved)

        st.success(ai_answer)
        st.markdown("<h4>Referenced Sections:</h4>", unsafe_allow_html=True)
        for sec, score in retrieved:
            with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                st.write(sec.get('Description', ''))
                st.caption(f"Relevance score: {score:.3f}")
