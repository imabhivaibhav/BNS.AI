import json
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.tokenize import sent_tokenize  # if multi-sentence search is used

# -----------------------
# Page config and CSS
# -----------------------
st.set_page_config(page_title="WAL.AI", layout="centered")

# Inject custom CSS for ChatGPT-like theme
st.markdown(
    """
    <style>
    /* Background and font */
    body, .stApp {
        background-color: #202123;
        color: #ECECF1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Center all content */
    .css-1d391kg {  /* Streamlit main content wrapper */
        max-width: 700px;
        margin: auto;
    }

    /* Title style */
    h1 {
        text-align: center;
        color: #00BFFF;
        font-weight: bold;
        font-size: 2.5rem;
    }

    /* Text area / input box */
    .stTextArea>div>div>textarea {
        background-color: #2a2b2f;
        color: #ECECF1;
        border-radius: 8px;
        padding: 10px;
    }

    /* Button style */
    div.stButton > button {
        background-color: #00BFFF;
        color: #202123;
        font-weight: bold;
        border-radius: 8px;
        height: 40px;
        width: 250px;
        display: block;
        margin: 10px auto;
    }

    /* Output card style */
    .stMarkdown, .stWrite {
        background-color: #2a2b2f;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Warning style */
    .stWarning {
        background-color: #ff4b5c;
        color: #fff;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }

    /* Captions */
    .stCaption {
        color: #aaaaaa;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Load model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.write(f"Welcome! Today's date: {st.session_state.get('today', st.experimental_get_query_params())}")
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

    # Direct section match
    direct = [
        i for i, s in enumerate(sections_data)
        if number and number == "".join(re.findall(r"\d+", s.get("Section", "")))
    ]

    if direct:
        indices = direct
        scores = [1.0] * len(indices)
    else:
        # Semantic search
        sentences = sent_tokenize(user_case)
        matched = {}

        for sent in sentences:
            sent_emb = model.encode(sent, convert_to_tensor=True)
            sims = util.cos_sim(sent_emb, section_embeddings)[0]

            top_k = min(10, len(sims))
            top_idx = torch.argsort(sims, descending=True)[:top_k]
            top_scores = sims[top_idx]

            median_score = float(torch.median(top_scores))
            threshold = max(0.45, median_score - 0.05)

            for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
                if score >= threshold:
                    if idx not in matched or score > matched[idx]:
                        matched[idx] = score

        if matched:
            sorted_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:7]
            indices, scores = zip(*sorted_matched)
        else:
            indices, scores = [], []

    # Show results
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
