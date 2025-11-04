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
        return json.load(f)

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
st.set_page_config(page_title="WAL.AI", layout="wide")
st.title("⚖️ WAL.AI - Legal Section Finder")
st.markdown("Enter your case details below. WAL.AI will find relevant legal sections.")

user_case = st.text_area("Case details", height=150)

def display_section_card(sec, score=None):
    """Display a section with a card-like style."""
    html_score = f"<p style='color:green; font-weight:bold;'>Similarity: {score:.2f}</p>" if score else ""
    st.markdown(
        f"""
        <div style="
            padding: 15px; 
            border-radius: 10px; 
            background-color: #f9f9f9; 
            box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        ">
            <h3 style='color:#2F4F4F'>{sec.get('Section', '')} - {sec.get('Title', '')}</h3>
            {html_score}
            <p><strong>Punishment:</strong> {sec.get('Punishment', 'N/A')}</p>
            <p>{sec.get('Description', '')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    # 3️⃣ Display results
    if not indices:
        st.warning("No matching sections found. Try describing your case in more detail or use a valid section number.")
    else:
        st.subheader("Matched Sections:")
        for idx, score in zip(indices, scores):
            sec = sections_data[idx]
            display_section_card(sec, score)
