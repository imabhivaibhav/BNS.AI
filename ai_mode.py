# ai_mode.py
import os
import streamlit as st
import torch
from google import genai

# -----------------------------
# CONFIGURATION
# -----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# Retrieve top sections based on semantic similarity
# -----------------------------
def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    """
    Retrieves top-k sections most semantically similar to the query.
    """
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate AI answer using Gemini API
# -----------------------------
def generate_ai_answer(question, retrieved_sections):
    """
    Generates an AI-based answer using context from retrieved legal sections via Gemini API.
    Ensures the model answers ONLY based on retrieved sections.
    """
    if len(retrieved_sections) == 0:
        return "❌ I cannot answer that as it is outside the provided legal sections."

    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = f"""
You are an expert Indian legal assistant.
You can only answer questions based on the Bhartiya Nyay Sanhita (BNS) sections provided in the context.
If a question is outside these sections, respond with:
'❌ I cannot answer that as it is outside the provided legal sections.'

Context:
{context}

Question: {question}
Answer concisely based only on the context.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini AI generation error: {str(e)}"
