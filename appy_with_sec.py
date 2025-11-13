# wal_ai_chat.py
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

st.set_page_config(page_title="WAL.AI", layout="wide", initial_sidebar_state="collapsed")

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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# UI Header
# -----------------------------
today = datetime.now().strftime("%A, %B %d, %Y")
st.markdown(f"""
<div style="width:100%; text-align:center; padding:15px; border-radius:10px;">
    👋 Welcome to <b>WAL.AI</b> — your intelligent legal advisor.<br>
    {today}.
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#28a745; font-size:120px;'>WAL.AI</h1>", unsafe_allow_html=True)

# -----------------------------
# Chat Display Area
# -----------------------------
chat_container = st.container()

for entry in st.session_state.chat_history:
    query = entry["query"]
    response = entry["response"]
    mode = entry["mode"]

    # Display question
    chat_container.markdown(f"""
    <div style="width:80%; margin:auto; padding:10px; background-color:#f0f2f6; border-radius:8px; margin-top:10px;">
        <b>You:</b> {query}
    </div>
    """, unsafe_allow_html=True)

    if response:
        # Display answer
        chat_container.markdown(f"""
        <div style="width:80%; margin:auto; padding:15px; background-color:#ffffff; border-radius:8px; margin-top:5px; border:1px solid #ddd;">
            <b>WAL.AI ({mode}):</b><br>{response}
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Input Section (Fixed Bottom)
# -----------------------------
with st.container():
    st.markdown("<br><br>")  # spacing

    with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([8, 1])
    with cols[0]:
        user_input = st.text_area(
            "",
            placeholder="Type your case/question here...",
            key="chat_input",
            label_visibility="collapsed",
            height=60
        )
    with cols[1]:
        submit = st.form_submit_button("➜")

    mode = st.radio(
        "Mode:",
        ["Find Matching Sections", "Ask AI"],
        horizontal=True,
        key="chat_mode"
    )

# -----------------------------
# Main Logic
# -----------------------------
if submit and user_input.strip():
    query = user_input.strip()
    entry = {"query": query, "mode": mode, "response": ""}

    # --- SEARCH MODE ---
    if mode == "Find Matching Sections":
        matched_sections = []

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
            response_text = ""
            for idx, score in sorted_matched:
                sec = sections_data[idx]
                response_text += f"**Section {sec.get('Section', '')}: {sec.get('Title', '')}**\n"
                response_text += f"{sec.get('Description', '')}\nPunishment: {sec.get('Punishment', '')}\nRelevance: {score:.3f}\n\n"
            entry["response"] = response_text
        else:
            entry["response"] = "No matching sections found. Try describing your case differently."

    # --- AI MODE ---
    elif mode == "Ask AI":
        retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
        ai_answer = generate_ai_answer(query, retrieved)

        response_text = ai_answer + "\n\nReferenced Sections:\n"
        for sec, score in retrieved:
            response_text += f"Section {sec.get('Section', '')}: {sec.get('Title', '')} - {sec.get('Description', '')} (Relevance: {score:.3f})\n\n"

        entry["response"] = response_text

    st.session_state.chat_history.append(entry)
    chat_container = st.container()
for entry in st.session_state.chat_history:
    chat_container.markdown(f"""
    <div style="width:80%; margin:auto; padding:10px; background-color:#f0f2f6; border-radius:8px; margin-top:10px;">
        <b>You:</b> {entry['query']}
    </div>
    """, unsafe_allow_html=True)
    if entry["response"]:
        chat_container.markdown(f"""
        <div style="width:80%; margin:auto; padding:15px; background-color:#ffffff; border-radius:8px; margin-top:5px; border:1px solid #ddd;">
            <b>WAL.AI ({entry['mode']}):</b><br>{entry['response']}
        </div>
        """, unsafe_allow_html=True)
