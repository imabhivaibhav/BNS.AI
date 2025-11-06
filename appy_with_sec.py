import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from datetime import datetime
from nltk.tokenize import sent_tokenize

# --------------------------------------------
# Setup
# --------------------------------------------
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="WAL.AI", page_icon="⚖️", layout="wide")

# --------------------------------------------
# Load dataset
# --------------------------------------------
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# --------------------------------------------
# Load model
# --------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# --------------------------------------------
# Embed sections
# --------------------------------------------
@st.cache_data
def embed_sections(sections):
    texts = [
        f"Section {sec.get('Section', '')}: {sec.get('Title', '')}. {sec.get('Description', '')}"
        for sec in sections
    ]
    return model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

section_embeddings = embed_sections(sections_data)

# --------------------------------------------
# UI
# --------------------------------------------
today = datetime.now().strftime("%A, %B %d, %Y")
st.markdown(
    f"""
    <div style="
        padding:20px;
        border-radius:12px;
        color:#666666;
        font-size:20px;
        font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        text-align:center;
    ">
        👋 Welcome to <b>WAL.AI</b>! {today}.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='
        text-align: center;
        color: #28a745;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        font-size: 120px;
        font-weight: bold;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    '>
        WAL.AI
    </h1>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
# Input
# --------------------------------------------
user_case = st.text_area(
    "Enter your case description or section numbers:",
    placeholder="E.g., 'Section 2, 3 and 4' or 'Kidnapping and murder case involving ransom'"
)

# --------------------------------------------
# Search Logic
# --------------------------------------------
if st.button("Find Matching Sections") and user_case.strip():
    query = user_case.lower().strip()

    with st.spinner("🔍 Analyzing your input and finding matching sections..."):

        # --- 1️⃣ Extract multiple section numbers (handles commas, 'and', etc.)
        section_numbers = re.findall(r"\d+", query)

        # --- 2️⃣ Split query into smaller parts for multi-topic (handles commas, and/or)
        subqueries = re.split(r",| and | or ", query)
        subqueries = [q.strip() for q in subqueries if q.strip()]

        matched = {}

        # --- 3️⃣ Direct number-based matches
        for i, s in enumerate(sections_data):
            sec_num = "".join(re.findall(r"\d+", s.get("Section", "")))
            if any(num == sec_num for num in section_numbers):
                matched[i] = 1.0  # Perfect direct match

        # --- 4️⃣ Semantic matches for multi-phrase / keyword queries
        if subqueries and (not section_numbers or len(subqueries) > len(section_numbers)):
            for sq in subqueries:
                # Encode each subquery (handles things like 'kidnapping', 'murder', etc.)
                sq_emb = model.encode(sq, convert_to_tensor=True)
                sims = util.cos_sim(sq_emb, section_embeddings)[0]

                # Top results
                top_k = min(10, len(sims))
                top_idx = torch.argsort(sims, descending=True)[:top_k]
                top_scores = sims[top_idx]

                # Dynamic threshold
                median_score = float(torch.median(top_scores))
                threshold = max(0.45, median_score - 0.05)

                for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
                    if score >= threshold:
                        matched[idx] = max(matched.get(idx, 0), float(score))

        # --- 5️⃣ Sort by relevance
        if matched:
            sorted_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10]
            indices, scores = zip(*sorted_matched)
        else:
            indices, scores = [], []

    # --------------------------------------------
    # Display Results
    # --------------------------------------------
    if not indices:
        st.warning("No matching sections found. Try describing your case differently.")
    else:
        st.subheader("📘 Relevant Section(s):")
        for idx, score in zip(indices, scores):
            sec = sections_data[idx]
            with st.expander(f"⚖️ Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                st.markdown(f"**Description:** {sec.get('Description', '')}")
                st.markdown(f"**Punishment:** {sec.get('Punishment', '')}")
                st.caption(f"Relevance score: {score:.3f}")

