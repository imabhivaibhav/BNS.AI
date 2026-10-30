# backend.py  (pure logic - UI se alag)
import json
import re
import torch
from sentence_transformers import SentenceTransformer, util

from ai_mode import retrieve_top_sections, generate_ai_answer, is_smalltalk, MIN_SCORE
from web_search import search_cases


def load_sections(path="laws_sections.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    return SentenceTransformer("all-mpnet-base-v2")


def embed_sections(sections, model):
    texts = [f"Section {s.get('Section', '')}: {s.get('Title', '')}. {s.get('Description', '')}" for s in sections]
    return model.encode(texts, convert_to_tensor=True)


def find_matching_sections(query, sections_data, model, section_embeddings, limit=10):
    """Returns [(section_dict, score)]. Section number likhne par exact match milta hai."""
    nums = re.findall(r"\d+", query)
    parts = [p.strip() for p in re.split(r",| and | or | aur ", query) if p.strip()]
    matched = {}

    for i, s in enumerate(sections_data):
        if nums and "".join(re.findall(r"\d+", s.get("Section", ""))) in nums:
            matched[i] = 1.0

    if parts and (not nums or len(parts) > len(nums)):
        for p in parts:
            sims = util.cos_sim(model.encode(p, convert_to_tensor=True), section_embeddings)[0]
            top = torch.argsort(sims, descending=True)[:min(10, len(sims))]
            scores = sims[top]
            thr = max(0.45, float(torch.median(scores)) - 0.05)
            for idx, sc in zip(top.tolist(), scores.tolist()):
                if sc >= thr:
                    matched[idx] = max(matched.get(idx, 0), sc)

    ranked = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [(sections_data[i], sc) for i, sc in ranked]


# ---------- Ask AI (with conversation memory) ----------
def _merge(*lists, k=5):
    best = {}
    for lst in lists:
        for s, sc in lst:
            key = s.get("Section")
            if key not in best or sc > best[key][1]:
                best[key] = (s, sc)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)[:k]


def _by_number(query, sections_data):
    """'BNS 103' / 'section 103' / 'धारा 103' -> that exact section."""
    if not re.search(r"section|sec\b|bns|dhara|धारा", query, re.I):
        return []
    nums = re.findall(r"\d+", query)
    return [(s, 1.0) for s in sections_data if "".join(re.findall(r"\d+", s.get("Section", ""))) in nums]


def run_ai_mode(query, sections_data, model, section_embeddings, history=None, language_hint=""):
    """history = [{"q", "a", "sections"}] of earlier Ask-AI turns in this thread."""
    history = history or []

    # Greetings / thanks / "who are you": just talk - no sections, no web search.
    if is_smalltalk(query):
        return {"answer": generate_ai_answer(query, [], history, language_hint, smalltalk=True),
                "sections": [], "cases": []}

    # Short follow-ups ("and the punishment?") borrow the previous topic.
    follow_up = bool(history) and len(query.split()) <= 7 and not re.search(r"\d", query)
    lists = [_by_number(query, sections_data),
             retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)]
    if follow_up:
        prev = history[-1]
        lists.append(retrieve_top_sections(f"{prev['q']}. {query}", sections_data, model, section_embeddings, top_k=4))
        lists.append(list(prev.get("sections", []))[:2])
    retrieved = [x for x in _merge(*lists, k=5) if x[1] >= MIN_SCORE]

    answer = generate_ai_answer(query, retrieved, history, language_hint)
    in_scope = not answer.startswith(("❌", "⚠️"))
    topic = f"{history[-1]['q']} {query}" if follow_up else query
    hint = retrieved[0][0].get("Title", "") if (retrieved and in_scope) else ""
    cases = search_cases(topic, max_results=5, hint=hint) if in_scope else []
    return {"answer": answer, "sections": retrieved, "cases": cases}
