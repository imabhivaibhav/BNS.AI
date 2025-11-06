import os
import requests

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

def generate_ai_answer(question, retrieved_sections):
    context = "\n\n".join(
        [f"Section {s['Section']}: {s['Title']}\n{s['Description']}" for s, _ in retrieved_sections]
    )
    prompt = f"Answer based on sections:\n{context}\nQuestion: {question}\nAnswer:"

    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.1-8B-Instruct",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 350}},
        timeout=30
    )

    if response.status_code == 200:
        return response.json()[0]["generated_text"].split("Answer:")[-1].strip()
    else:
        return f"⚠️ AI generation failed ({response.status_code}): {response.text}"
