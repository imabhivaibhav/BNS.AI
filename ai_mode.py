import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Load GPT-2 from Google Drive path
# -----------------------------
# The Drive path should be mounted via Colab or accessible via PyDrive
MODEL_PATH = "/content/drive/MyDrive/gpt2_model"  # Update with your Drive path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(device)
    return tokenizer, model

# -----------------------------
# Semantic search
# -----------------------------
def retrieve_top_sections(query, sections_data, semantic_model, section_embeddings, top_k=5):
    query_emb = semantic_model.encode(query, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), section_embeddings)
    top_indices = torch.argsort(sims, descending=True)[:top_k]
    return [(sections_data[i], float(sims[i])) for i in top_indices]

# -----------------------------
# Generate answer using local GPT-2
# -----------------------------
def generate_ai_answer(question, retrieved_sections, tokenizer, model, max_tokens=150):
    if len(retrieved_sections) == 0:
        return "❌ I cannot answer that as it is outside the provided legal sections."

    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely based only on the context."

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()
