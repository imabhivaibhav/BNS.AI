
import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch



# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

sections_data = load_sections()

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.write("Welcome")
model = load_model()

# -----------------------
# Embed sections
# -----------------------
@st.cache_data
def embed_sections(sections):
    texts = [
        f"Section {sec.get('Section', '')}: {sec.get('Title', '')}. {sec.get('Description', '')}"
        for sec in sections
    ]
    return model.encode(texts, convert_to_tensor=True)

section_embeddings = embed_sections(sections_data)

# -----------------------
# Streamlit UI
# -----------------------
st.title("WAL.AI")
user_case = st.text_area("Enter your case description or section number:")

if st.button("Find Matching Sections") and user_case.strip():
    query = user_case.lower().strip()
    number = "".join(re.findall(r"\d+", query))

    # 1️⃣ Direct section match
    direct = [
        i for i, s in enumerate(sections_data)
        if number and number == "".join(re.findall(r"\d+", s.get("Section", "")))
    ]

    if direct:
        indices = direct
        scores = [1.0] * len(indices)
    else:
        # 2️⃣ Multi-sentence / multi-phrase semantic search
        sentences = sent_tokenize(user_case)
        matched = {}

        for sent in sentences:
            sent_emb = model.encode(sent, convert_to_tensor=True)
            sims = util.cos_sim(sent_emb, section_embeddings)[0]

            # top 10 for this phrase
            top_k = min(10, len(sims))
            top_idx = torch.argsort(sims, descending=True)[:top_k]
            top_scores = sims[top_idx]

            # dynamic threshold per phrase
            median_score = float(torch.median(top_scores))
            threshold = max(0.45, median_score - 0.05)

            for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
                if score >= threshold:
                    if idx not in matched or score > matched[idx]:
                        matched[idx] = score

        # sort by relevance
        if matched:
            sorted_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:7]
            indices, scores = zip(*sorted_matched)
        else:
            indices, scores = [], []

    # 3️⃣ Show results
    if not indices:
        st.warning("No matching sections found. Try describing your case differently.")
    else:
        st.subheader("Relevant Section(s):")
        for idx, score in zip(indices, scores):
            sec = sections_data[idx]
            st.write("---")
            st.write(f"**Section:** {sec.get('Section', '')}")
            st.write(f"**Title:** {sec.get('Title', '')}")
            st.write(f"**Punishment:** {sec.get('Punishment', '')}")
            st.write(f"**Description:** {sec.get('Description', '')}")
            st.caption(f"Relevance: {score:.3f}")

