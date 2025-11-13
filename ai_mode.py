# ai_mode.py
import os
import requests
import streamlit as st
import torch

# -----------------------------
# CONFIGURATION
# -----------------------------
# Retrieve your Hugging Face API token from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

# Use a model that supports chat via HF Router
# Example: "gpt2" is for testing; replace with an actual HF Router chat model
MODEL_ID = "MiniMaxAI/MiniMax-M2:novita"

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
# Generate AI answer using Hugging Face Router API
# -----------------------------
def generate_ai_answer(question, retrieved_sections):
    """
    Generates an AI-based answer using context from retrieved legal sections.
    """
    # Combine retrieved sections into a single context string
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    # Construct the chat message
    system_message = {
        "role": "system",
        "content": (
        "You are an expert Indian legal assistant. "
        "You can only answer questions based on the Bhartiya Nyay Sanhita (BNS) sections provided in the context. "
        "If a question is outside these sections, respond with: '❌ I cannot answer that as it is outside the provided legal sections.'"
        )
    }


    user_message = {
        "role": "user",
        "content": f"Based on the following Bhartiya Nyay Sanhita (BNS) sections:\n{context}\n\nQuestion: {question}\nAnswer concisely."
    }

    payload = {
        "model": MODEL_ID,
        "messages": [system_message, user_message],
        "temperature": 0.7,
        "max_new_tokens": 350,
        "top_p": 0.9
    }

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        elif response.status_code == 401:
            return "⚠️ Invalid Hugging Face token. Check your HF_TOKEN in Streamlit secrets."
        elif response.status_code == 404:
            return f"⚠️ Model '{MODEL_ID}' not found or unavailable for inference."
        else:
            return f"⚠️ AI generation failed ({response.status_code}): {response.text}"
    except requests.exceptions.Timeout:
        return "⚠️ The model request timed out. Please try again later."
    except Exception as e:
        return f"⚠️ AI generation error: {str(e)}"


# -----------------------------
# Example (for testing locally)
# -----------------------------
if __name__ == "__main__":
    sections = [
        {"Section": "101", "Title": "Murder", "Description": "Whoever causes death with intention..."},
        {"Section": "102", "Title": "Culpable Homicide", "Description": "Whoever causes death without intention..."}
    ]

    # Fake embeddings for demo
    model = None  # Normally a SentenceTransformer model
    section_embeddings = torch.rand((2, 768))

    query = "What is the punishment for murder under BNS?"
    retrieved_sections = [(sections[0], 0.95), (sections[1], 0.85)]

    answer = generate_ai_answer(query, retrieved_sections)
    print("\nAI Answer:\n", answer)

