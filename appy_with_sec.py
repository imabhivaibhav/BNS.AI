import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from datetime import datetime
import requests
from nltk.tokenize import sent_tokenize

# -----------------------------
# Setup
# -----------------------------
nltk.download('punkt', quiet=True)

st.set_page_config(
    page_title="WAL.AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Optional CSS for styling
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# -----------------------------
# Load model
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
# Helper: Retrieve Top Sections
# -----------------------------
def retrieve_top_sections(query, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, section_embeddings)[0]
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    results = [(sections_data[i], float(sims[i])) for i in top_indices]
    return results

# -----------------------------
# Helper: Generate AI Answer
# -----------------------------
HF_TOKEN = "your_huggingface_token_here"  # Replace or use environment variable

def generate_ai_answer(question, retrieved_sections):
    # Combine sections for context
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = (
        f"You are an expert Indian legal assistant. Based on the following Bhartiya Nyay Sanhita (BNS) sections, "
        f"answer the user's question clearly and concisely:\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )

    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 350}},
    )

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"].split("Answer:")[-1].strip()
        except Exception:
            return response.json()[0]["generated_text"]
    else:
        return f"⚠️ AI generation failed: {response.status_code} - {response.text}"

# -----------------------------
# UI Header
# -----------------------------
today = datetime.now().strftime("%A, %B %d, %Y")
st.markdown(f"""
<div class="welcome-box">
    👋 Welcome to <b>WAL.AI</b> — your intelligent legal advisor.<br>
    {today}.
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#28a745; font-size:140px;'>WAL.AI</h1>", unsafe_allow_html=True)

# -----------------------------
# User Input
# -----------------------------
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    mode = st.radio(
        "Choose Mode:",
        ["Find Matching Sections", "Ask AI"],
        horizontal=True
    )

    user_case = st.text_area(
        "Enter your case description or question:",
        placeholder="E.g., 'A person killed someone' or 'What is the punishment for theft under BNS?'",
        height=190
    )
    submit = st.button("Submit")

# -----------------------------
# MAIN LOGIC
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
            # Direct number match
            for i, s in enumerate(sections_data):
                sec_num = "".join(re.findall(r"\d+", s.get("Section", "")))
                if any(num == sec_num for num in section_numbers):
                    matched[i] = 1.0

            # Semantic match
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

        with col2:
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

    # --- ASK AI MODE ---
    elif mode == "Ask AI":
        with st.spinner("Analyzing and generating response..."):
            retrieved = retrieve_top_sections(query, top_k=4)
            ai_answer = generate_ai_answer(query, retrieved)

        with col2:
            st.markdown("<h3 style='text-align:center;'>AI Response:</h3>", unsafe_allow_html=True)
            st.success(ai_answer)

            st.markdown("<h4>Referenced Sections:</h4>", unsafe_allow_html=True)
            for sec, score in retrieved:
                with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                    st.write(sec.get('Description', ''))
                    st.caption(f"Relevance score: {score:.3f}")
