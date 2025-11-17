# backend.py

import json
import re
import torch
import nltk
from sentence_transformers import SentenceTransformer, util

from ai_mode import retrieve_top_sections, generate_ai_answer
from web_search import search_cases

nltk.download('punkt', quiet=True)

# -----------------------------
# Load dataset
# -----------------------------
def load_sections():
    with open("laws_sections.json", "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Load SentenceTransformer model
# -----------------------------
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

# -----------------------------
# Embed all sections
# -----------------------------
def embed_sections(sections, model):
    texts = [
        f"Section {sec.get('Section', '')}: {sec.get('Title', '')}. {sec.get('Description', '')}"
        for sec in sections
    ]
    return model.encode(texts, convert_to_tensor=True)

# -----------------------------
# SEARCH MODE BACKEND LOGIC
# -----------------------------
def find_matching_sections(query, sections_data, model, section_embeddings):
    section_numbers = re.findall(r"\d+", query)
    subqueries = re.split(r",| and | or ", query)
    subqueries = [q.strip() for q in subqueries if q.strip()]

    matched = {}

    # Direct match by numbers
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
        return sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10]

    return []

# -----------------------------
# AI MODE BACKEND LOGIC
# -----------------------------
def run_ai_mode(query, sections_data, model, section_embeddings):
    retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
    ai_answer = generate_ai_answer(query, retrieved)
    cases = search_cases(query, max_results=5)
    return ai_answer, retrieved, cases
