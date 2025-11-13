# ai_mode.py
import torch
import requests
import streamlit as st

# -----------------------------
# CONFIGURATION
# -----------------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # smaller, stable model

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
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    try:
        # UPDATED ENDPOINT
        response = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            try:
                if isinstance(result, list):
                    return result[0].get("generated_text", str(result)).strip()
                elif isinstance(result, dict):
                    return result.get("generated_text", str(result)).strip()
                else:
                    return str(result)
            except Exception:
                return str(result)

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

    model = None  # Normally a SentenceTransformer model
    section_embeddings = torch.rand((2, 768))

    query = "What is the punishment for murder under BNS?"
    retrieved_sections = [(sections[0], 0.95), (sections[1], 0.85)]

    answer = generate_ai_answer(query, retrieved_sections)
    print("\nAI Answer:\n", answer)

