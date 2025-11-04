import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------
# Load dataset (cached)
# -----------------------
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        sections_data = json.load(f)
    return sections_data

sections_data = load_sections()

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.write("Loading model, please wait...")
model = load_model()

# -----------------------
# Embed all section descriptions (cached)
# -----------------------
@st.cache_data
def embed_sections(sections):
    texts = [sec["Description"] for sec in sections]
    return model.encode(texts, convert_to_tensor=True)

section_embeddings = embed_sections(sections_data)

# -----------------------
# Streamlit UI
# -----------------------
st.title("WAL.AI")
user_case = st.text_area("Enter your case detail below...")

if st.button("Find Matching Sections") and user_case.strip():
    query = user_case.lower().strip()
    number = "".join(re.findall(r"\d+", query))

    # 1️⃣ Direct section match
    direct = [
        i for i, s in enumerate(sections_data)
        if number and number == "".join(re.findall(r"\d+", s.get("Section", "")))
    ]

    # 2️⃣ If direct match found, skip semantic search
    if direct:
        indices = direct
        scores = [1.0] * len(indices)
    else:
        # Semantic search
        user_emb = model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(user_emb, section_embeddings)[0]

        # Sort and get top 10, then filter by similarity threshold
        top_k = 10
        best = torch.topk(sims, k=min(top_k, len(sims)))
        indices = best.indices.tolist()
        scores = best.values.tolist()

        # Dynamic threshold – keep only strong matches
        threshold = max(0.55, float(torch.median(best.values)) + 0.03)
        filtered = [(i, s) for i, s in zip(indices, scores) if s >= threshold]
        if filtered:
            indices, scores = zip(*filtered)
        else:
            indices, scores = [], []

    # 3️⃣ Show Results
    if not indices:
        st.warning("No matching sections found. Try describing your case in more detail or use a valid section number.")
    else:
        st.subheader("Matched Sections:")
        for idx, score in zip(indices, scores):
            sec = sections_data[idx]
            st.write("---")
            st.write(f"**Section:** {sec.get('Section', '')}")
            st.write(f"**Title:** {sec.get('Title', '')}")
            st.write(f"**Punishment:** {sec.get('Punishment', '')}")
            st.write(f"**Description:** {sec.get('Description', '')}")




