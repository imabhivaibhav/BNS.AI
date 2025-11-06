# ai_mode.py
import torch
import requests
import streamlit as st

HF_TOKEN = st.secrets["HF_TOKEN"]  # your Hugging Face token
MODEL_ID = "deepseek-ai/DeepSeek-OCR"
# -----------------------------
# Retrieve top sections based on semantic similarity
# -----------------------------
def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate AI answer using Hugging Face Inference Router API
# -----------------------------
def generate_ai_answer(question, retrieved_sections):
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = (
        f"You are an expert Indian legal assistant. Based on the following Bhartiya Nyay Sanhita (BNS) sections, "
        f"answer the user's question clearly and concisely:\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 350}
    }

    try:
        response = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                return result[0].get("generated_text", "").split("Answer:")[-1].strip()
            elif isinstance(result, dict):
                return result.get("generated_text", "").split("Answer:")[-1].strip()
            else:
                return "⚠️ Unexpected response format from model."
        elif response.status_code == 401:
            return "⚠️ Invalid Hugging Face credentials. Check your HF_TOKEN."
        elif response.status_code == 404:
            return f"⚠️ Model '{MODEL_ID}' not found or not available for inference."
        else:
            return f"⚠️ AI generation failed ({response.status_code}): {response.text}"
    except Exception as e:
        return f"⚠️ AI generation error: {str(e)}"








