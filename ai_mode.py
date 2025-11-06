# ai_mode.py
import torch
import requests

HF_TOKEN = "hf_your_valid_token_here"  # Replace with your token or use env variable

# -----------------------------
# Retrieve top sections based on semantic similarity
# -----------------------------
def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate AI answer using Hugging Face Inference API
# -----------------------------
def generate_ai_answer(question, retrieved_sections):
    # Combine sections as context
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = (
        f"You are an expert Indian legal assistant. Based on the following Bhartiya Nyay Sanhita (BNS) sections, "
        f"answer the user's question clearly and concisely:\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 350}},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()[0]["generated_text"].split("Answer:")[-1].strip()
        elif response.status_code == 401:
            return "⚠️ Invalid Hugging Face credentials. Please check your HF_TOKEN."
        else:
            return f"⚠️ AI generation failed ({response.status_code}): {response.text}"
    except Exception as e:
        return f"⚠️ AI generation error: {str(e)}"
