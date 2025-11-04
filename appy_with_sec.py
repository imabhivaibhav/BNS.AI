import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="WAL.AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Load dataset (cached)
# -----------------------
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.sidebar.write("⚖️ **WAL.AI Legal Search**")
st.sidebar.info("Enter your case description or section number to find relevant legal sections.")

st.write("### Welcome to WAL.AI")
st.write("Describe your case below and find the matching law sections instantly.")

st.write("---")

# -----------------------
# Embed all section descriptions (cached)
# -----------------------
@st.cache_data
def embed_sections(sections):
    texts = [sec["Description"] for sec in sections]
    return model.encode(texts, convert_to_tensor=True)

model = load_model()
section_embeddings = embed_sections(sections_data)

# -----------------------
# User input
# -----------------------
user_case = st.text_area("Enter your case detail:", height=150)

if st.button("Find Matching Sections") and user_case.strip():
    query = user_case.lower().strip()
    number = "".join(re.findall(r"\d+", query))

    # 1️⃣ Direct section match
    direct = [
        i for i, s in enumerate(sections_data)
        if number and number == "".join(re.findall(r"\d+", s.get("Section", "")))
    ]

    # 2️⃣ Semantic search if no direct match
    if direct:
        indices = direct
        scores = [1.0] * len(indices)
    else:
        user_emb = model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(user_emb, section_embeddings)[0]

        # Top 10 and threshold filter
        top_k = 10
        best = torch.topk(sims, k=min(top_k, len(sims)))
        indices = best.indices.tolist()
        scores = best.values.tolist()
        threshold = max(0.55, float(torch.median(best.values)) + 0.03)
        filtered = [(i, s) for i, s in zip(indices, scores) if s >= threshold]
        if filtered:
            indices, scores = zip(*filtered)
        else:
            indices, scores = [], []

    # 3️⃣ Display results nicely
    if not indices:
        st.warning("No matching sections found. Try describing your case in more detail or use a valid section number.")
    else:
        st.subheader(f"Matched Sections ({len(indices)})")
        for idx, score in zip(indices, scores):
            sec = sections_data[idx]
            with st.container():
                st.markdown(
                    f"""
                    <div style="padding:15px; border-radius:10px; background-color:#f2f2f7; margin-bottom:10px;">
                        <h4 style="color:#0d6efd;">Section {sec.get('Section', '')}: {sec.get('Title', '')}</h4>
                        <p><b>Punishment:</b> {sec.get('Punishment', '')}</p>
                        <p><b>Description:</b> {sec.get('Description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
