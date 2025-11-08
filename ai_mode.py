# ai_mode.py
import torch
import requests
import streamlit as st

# -----------------------------
# CONFIGURATION
# -----------------------------
# Retrieve your Hugging Face API token from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

# Use a proper text-generation model (not OCR)
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct"
# Other good options:
# MODEL_ID = "meta-llama/Llama-3-8b-instruct"
# MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"


# -----------------------------
# Retrieve top sections based on semantic similarity
# -----------------------------
def retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=5):
    """
    Retrieves top-k sections most semantically similar to the query.

    Args:
        query (str): The user's question or input.
        sections_data (list): List of dicts containing section info.
        model: SentenceTransformer or embedding model.
        section_embeddings (torch.Tensor): Precomputed embeddings for all sections.
        top_k (int): Number of sections to retrieve.

    Returns:
        list of tuples: [(section_dict, similarity_score), ...]
    """
    query_emb = model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]


# -----------------------------
# Generate AI answer using Hugging Face Inference API
# -----------------------------
def generate_ai_answer(question, retrieved_sections):
    """
    Generates an AI-based answer using context from retrieved legal sections.

    Args:
        question (str): User's query.
        retrieved_sections (list): List of (section_dict, score) tuples.

    Returns:
        str: AI-generated answer or an error message.
    """
    # Combine retrieved sections into a single context string
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    # Construct the instruction prompt
    prompt = (
        f"You are an expert Indian legal assistant. Based on the following Bhartiya Nyay Sanhita (BNS) sections, "
        f"answer the user's question clearly and concisely:\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Prepare request to Hugging Face Inference API
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 350,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    try:
        # Send POST request
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_ID}",
            headers=headers,
            json=payload,
            timeout=90
        )

        # Handle API responses
        if response.status_code == 200:
            result = response.json()

            # Debug: Uncomment to inspect response
            # st.write("DEBUG Response:", result)

            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].strip()
            else:
                return "⚠️ Unexpected response format from model."

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
    # Dummy data for testing
    sections = [
        {"Section": "101", "Title": "Murder", "Description": "Whoever causes death with intention..."},
        {"Section": "102", "Title": "Culpable Homicide", "Description": "Whoever causes death without intention..."}
    ]

    # Fake embeddings (for demo)
    model = None  # Normally a SentenceTransformer model
    section_embeddings = torch.rand((2, 768))

    query = "What is the punishment for murder under BNS?"
    # Mock retrieval (skipping embedding model for demo)
    retrieved_sections = [(sections[0], 0.95), (sections[1], 0.85)]

    answer = generate_ai_answer(query, retrieved_sections)
    print("\nAI Answer:\n", answer)
