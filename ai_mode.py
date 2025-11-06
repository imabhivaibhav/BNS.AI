# ai_mode.py
import torch
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import requests

# -----------------------------
# Get HF_TOKEN from Streamlit secrets
# -----------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set in Streamlit secrets. Add it under Settings -> Secrets.")

# -----------------------------
# Retrieve top sections based on semantic similarity
# -----------------------------
def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, section_embeddings)[0]
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate AI answer using Hugging Face Inference API
# -----------------------------
def generate_ai_answer(question, retrieved_sections, model_id="meta-llama/Llama-3.1-8B-Instruct"):
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = (
        f"You are an expert Indian legal assistant. Based on the following Bhartiya Nyay Sanhita (BNS) sections, "
        f"answer the user's question clearly and concisely:\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )

    url = f"https://router.huggingface.co/hf-inference{model_id}"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 350}}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            # Some HF models return a list, some dict
            if isinstance(data, list):
                return data[0].get("generated_text", "").split("Answer:")[-1].strip()
            elif isinstance(data, dict):
                return data.get("generated_text", "").split("Answer:")[-1].strip()
            else:
                return "⚠️ Unexpected AI response format."
        elif response.status_code == 401:
            return "⚠️ Invalid Hugging Face credentials."
        else:
            return f"⚠️ AI generation failed ({response.status_code}): {response.text}"
    except Exception as e:
        return f"⚠️ AI generation error: {str(e)}"



