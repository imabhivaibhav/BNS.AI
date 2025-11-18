# ai_mode_local.py
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# CONFIGURATION
# -----------------------------
# Path to your saved GPT-2 model in Google Drive
MODEL_PATH = "https://drive.google.com/drive/folders/1cDgpRqGolfLMLRtASsH4GsslfFEx-kHM?usp=sharing"  # update if different

# Load tokenizer and model from Drive
@st.cache_resource(show_spinner=True)
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(MODEL_PATH)

# -----------------------------
# Retrieve top sections (same as before)
# -----------------------------
def retrieve_top_sections(query, sections_data, section_embeddings, top_k=5):
    """
    Retrieves top-k sections most semantically similar to the query.
    """
    query_emb = section_embeddings.model.encode(query, convert_to_tensor=True)  # if using sentence-transformers
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate AI answer locally
# -----------------------------
def generate_ai_answer_local(question, retrieved_sections, max_length=350):
    """
    Generates AI answer using local GPT-2 based on retrieved sections.
    """
    if len(retrieved_sections) == 0:
        return "❌ I cannot answer that as it is outside the provided legal sections."

    # Combine all retrieved sections as context
    context_text = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer concisely based only on the context."

    # Encode input and generate output
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    # Decode the generated text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from answer if present
    answer = answer.replace(prompt, "").strip()
    return answer

# -----------------------------
# Example Streamlit usage
# -----------------------------
st.title("📜 Legal AI Assistant (Local GPT-2)")

question = st.text_input("Enter your legal question:")

if st.button("Get Answer"):
    # For demonstration, assume `sections_data` and `section_embeddings` exist
    # retrieved_sections = retrieve_top_sections(question, sections_data, section_embeddings)
    retrieved_sections = []  # Replace with real retrieval logic
    answer = generate_ai_answer_local(question, retrieved_sections)
    st.write("**Answer:**")
    st.write(answer)
