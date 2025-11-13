# ai_mode.py
import os
import requests
import streamlit as st
import torch

# -----------------------------
# CONFIGURATION
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
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
    Ensures the model answers ONLY based on retrieved sections.
    """
    if len(retrieved_sections) == 0:
        # No relevant sections found
        return "❌ I cannot answer that as it is outside the provided legal sections."

    # Combine retrieved sections into a single context string
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    system_message = {
    "role": "system",
    "content": (
        "You are an expert Indian legal assistant. "
        "You can ONLY answer questions using the Bhartiya Nyay Sanhita (BNS) sections provided in the context. "
        "You MUST NOT invent or reference any section that is NOT in the context. "
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

    # Legal question (within context)
    query1 = "What is the punishment for murder under BNS?"
    retrieved_sections1 = [(sections[0], 0.95), (sections[1], 0.85)]
    answer1 = generate_ai_answer(query1, retrieved_sections1)
    print("\nAI Answer (legal question):\n", answer1)

    # Out-of-topic question
    query2 = "Who won the 2020 Olympics?"
    retrieved_sections2 = []  # No relevant sections
    answer2 = generate_ai_answer(query2, retrieved_sections2)
    print("\nAI Answer (outside legal dataset):\n", answer2)

    # Out-of-topic but trying to hallucinate a section
    query3 = "Explain criminal law about theft in India."
    retrieved_sections3 = []  # No relevant sections
    answer3 = generate_ai_answer(query3, retrieved_sections3)
    print("\nAI Answer (hallucination test):\n", answer3)

