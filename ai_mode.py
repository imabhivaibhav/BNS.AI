import os
import requests
import torch
import streamlit as st

# -----------------------------
# CONFIGURATION
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_ID = "moonshotai/Kimi-K2-Instruct-0905:groq"  # Updated model

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# -----------------------------
# Retrieve top sections (unchanged)
# -----------------------------
def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate AI answer
# -----------------------------
def generate_ai_answer(question, retrieved_sections):
    if len(retrieved_sections) == 0:
        return "❌ I cannot answer that as it is outside the provided legal sections."

    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    system_message = {
        "role": "system",
        "content": (
            "You are an expert Indian legal assistant. "
            "You can ONLY answer questions using the Bhartiya Nyay Sanhita (BNS) sections provided in the context. "
            "You MUST NOT invent or reference any section not in the context. "
            "If a question is outside the context, your ONLY reply must be exactly: "
            "'❌ I cannot answer that as it is outside the provided legal sections.' "
            "Do NOT include any section numbers, titles, or content not in the provided context."
        )
    }

    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely based only on the context."
    }

    payload = {
        "model": MODEL_ID,
        "messages": [system_message, user_message],
        "temperature": 0.7,
        "max_new_tokens": 350,
        "top_p": 0.9
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=90)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        elif response.status_code == 401:
            return "⚠️ Invalid Hugging Face token."
        elif response.status_code == 404:
            return f"⚠️ Model '{MODEL_ID}' not found."
        else:
            return f"⚠️ AI generation failed ({response.status_code}): {response.text}"
    except Exception as e:
        return f"⚠️ AI generation error: {str(e)}"

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    sections = [
        {"Section": "101", "Title": "Murder", "Description": "Whoever causes death with intention..."},
        {"Section": "102", "Title": "Culpable Homicide", "Description": "Whoever causes death without intention..."}
    ]

    # Fake embeddings for demo
    model = None  # Normally a SentenceTransformer model
    section_embeddings = torch.rand((2, 768))

    query1 = "What is the punishment for murder under BNS?"
    retrieved_sections1 = [(sections[0], 0.95), (sections[1], 0.85)]
    answer1 = generate_ai_answer(query1, retrieved_sections1)
    print("\nAI Answer (legal question):\n", answer1)

    query2 = "Who won the 2020 Olympics?"
    retrieved_sections2 = []
    answer2 = generate_ai_answer(query2, retrieved_sections2)
    print("\nAI Answer (outside dataset):\n", answer2)
