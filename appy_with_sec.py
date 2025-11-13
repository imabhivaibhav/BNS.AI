import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from datetime import datetime
import requests
import os

# -----------------------------
# AI Mode functions
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_ID = "MiniMaxAI/MiniMax-M2:novita"  # Example HF chat model

API_URL = "https://api-inference.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_chat(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
    return response.json()

def generate_ai_answer(question, retrieved_sections):
    # Combine retrieved sections into context
    context = "\n\n".join([f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections])
    prompt = f"You are an expert Indian legal assistant. Based on the following Bhartiya Nyay Sanhita (BNS) sections, answer clearly:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 350,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        result = query_hf_chat(payload)
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "⚠️ Unexpected response from model."
    except Exception as e:
        return f"⚠️ AI generation error: {str(e)}"

def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

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

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# Embed sections
@st.cache_data
def embed_sections(sections):
    texts = [f"Section {sec.get('Section', '')}: {sec.get('Title', '')}. {sec.get('Description', '')}" for sec in sections]
    return model.encode(texts, convert_to_tensor=True)

section_embeddings = embed_sections(sections_data)

# -----------------------------
# UI Header
# -----------------------------
today = datetime.now().strftime("%A, %B %d, %Y")
st.markdown(f"""
<div style="width:100%; display:flex; justify-content:center;">
    <div style="text-align:center; font-size:20px; padding:15px; border-radius:10px;">
        👋 Welcome to <b>WAL.AI</b> — your intelligent legal advisor.<br>{today}.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#28a745; font-size:140px;'>WAL.AI</h1>", unsafe_allow_html=True)

# -----------------------------
# Chat-style Input with inline mode selector
# -----------------------------
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    container = st.container()
    with container:
        # Mode selector
        mode = st.radio("", ["Find Matching Sections", "Ask AI"], horizontal=True, label_visibility="collapsed")

        # Text input with arrow button
        user_case = st.text_area("Type your case or question here...", key="input_box", height=120)
        submit = st.button("➡️")

# -----------------------------
# Main Logic
# -----------------------------
if submit and user_case.strip():
    query = user_case.strip()

    # --- SEARCH MODE (hybrid: exact number + semantic) ---
    if mode == "Find Matching Sections":
        with st.spinner("Finding relevant sections..."):
            section_numbers = re.findall(r"\d+", query)
            matched_indices = []

            # First, exact section number match
            if section_numbers:
                for i, s in enumerate(sections_data):
                    sec_num = "".join(re.findall(r"\d+", s.get("Section", "")))
                    if sec_num in section_numbers:
                        matched_indices.append(i)

            # If no number or no matches, do semantic search
            if not matched_indices:
                retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=10)
                matched_indices = [sections_data.index(sec) for sec, _ in retrieved]

        # Display matched sections
        with col2:
            if not matched_indices:
                st.warning("No matching sections found. Try describing your case differently.")
            else:
                st.markdown("<h3 style='text-align:center;'>Relevant Section(s):</h3>", unsafe_allow_html=True)
                for idx in matched_indices:
                    sec = sections_data[idx]
                    with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                        st.markdown(f"**Description:** {sec.get('Description', '')}")
                        st.markdown(f"**Punishment:** {sec.get('Punishment', '')}")

    # --- AI MODE ---
    elif mode == "Ask AI":
        with st.spinner("Analyzing and generating response..."):
            retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
            ai_answer = generate_ai_answer(query, retrieved)

        # Display AI response and referenced sections
        with col2:
            st.markdown("<h3 style='text-align:center;'>AI Response:</h3>", unsafe_allow_html=True)
            st.success(ai_answer)

            st.markdown("<h4>Referenced Sections:</h4>", unsafe_allow_html=True)
            for sec, score in retrieved:
                with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                    st.write(sec.get('Description', ''))
                    st.caption(f"Relevance score: {score:.3f}")


