import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from datetime import datetime

from ai_mode import retrieve_top_sections, generate_ai_answer

# -----------------------------
# Setup
# -----------------------------
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="WAL.AI", layout="centered", initial_sidebar_state="collapsed")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# -----------------------------
# Load model for semantic search
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# -----------------------------
# Embed sections
# -----------------------------
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
# Initialize session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Chat display container
# -----------------------------
chat_container = st.container()

# -----------------------------
# Input section at the bottom
# -----------------------------
# Automatically clear input after submission
default_input = ""  

user_input = st.text_area(
    "Type your case/question here...",
    placeholder="E.g., 'A person killed someone' or 'What is the punishment for theft under BNS?'",
    height=80,
    key="chat_input",
    value=default_input
)

mode = st.radio(
    "Mode:",
    ["Find Matching Sections", "Ask AI"],
    horizontal=True,
    key="chat_mode"
)

submit = st.button("➜ Send", key="chat_submit")

# -----------------------------
# Handle user input
# -----------------------------
if submit and user_input.strip():
    query = user_input.strip()
    entry = {"query": query, "mode": mode, "response": None}

    # --- Find Matching Sections ---
    if mode == "Find Matching Sections":
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

        entry["response"] = {"indices": indices, "scores": scores}

    # --- AI Mode ---
    elif mode == "Ask AI":
        retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
        ai_answer = generate_ai_answer(query, retrieved)
        entry["response"] = {"ai_answer": ai_answer, "retrieved": retrieved}

    # Add entry to chat history
    st.session_state.chat_history.append(entry)

# -----------------------------
# Render chat history
# -----------------------------
with chat_container:
    for entry in st.session_state.chat_history:
        st.markdown(f"**Q:** {entry['query']}")
        
        if entry["mode"] == "Find Matching Sections":
            res = entry["response"]
            if res and res["indices"]:
                for idx, score in zip(res["indices"], res["scores"]):
                    sec = sections_data[idx]
                    st.markdown(f"- **Section {sec.get('Section','')}: {sec.get('Title','')}**")
                    st.markdown(f"  - Description: {sec.get('Description','')}")
                    st.markdown(f"  - Punishment: {sec.get('Punishment','')}")
                    st.caption(f"Relevance score: {score:.3f}")
            else:
                st.write("No matching sections found.")
        
        elif entry["mode"] == "Ask AI":
            res = entry["response"]
            if res:
                st.success(res["ai_answer"])
                st.markdown("**Referenced Sections:**")
                for sec, score in res["retrieved"]:
                    st.markdown(f"- **Section {sec.get('Section','')}: {sec.get('Title','')}**")
                    st.markdown(f"  - Description: {sec.get('Description','')}")
                    st.caption(f"Relevance score: {score:.3f}")
        st.markdown("---")
