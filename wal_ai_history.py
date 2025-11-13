# wal_ai_history.py
import streamlit as st
import re
import torch
from sentence_transformers import util
from datetime import datetime
from wal_ai import sections_data, model, section_embeddings
from ai_mode import retrieve_top_sections, generate_ai_answer

def search_history_ui():
    """
    Handles search history, input box, and displaying results.
    This can be called from wal_ai.py
    """
    # Initialize session state
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    # Shrink input box if history exists
    height = 100 if st.session_state.search_history else 180

    user_case = st.text_area(
        "Enter your case description or question:",
        placeholder="E.g., 'A person killed someone' or 'What is the punishment for theft under BNS?'",
        height=height,
        key="history_input"
    )

    mode_col, spacer_col, btn_col = st.columns([5, 2, 1])
    with mode_col:
        mode = st.radio(
            "",
            ["Find Matching Sections", "Ask AI"],
            horizontal=True,
            key="history_mode"
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button("➜ Search", key="history_submit")

    # Handle query submission
    if submit and user_case.strip():
        query = user_case.strip()
        st.session_state.search_history.append({"query": query, "mode": mode, "results": None})

        # --- SEARCH MODE ---
        if mode == "Find Matching Sections":
            section_numbers = re.findall(r"\d+", query)
            subqueries = re.split(r",| and | or ", query)
            subqueries = [q.strip() for q in subqueries if q.strip()]

            matched = {}
            for i, s in enumerate(sections_data):
                sec_num = "".join(re.findall(r"\d+", s.get("Section", "")))
                if any(num == sec_num for num in section_numbers):
                    matched[i] = 1.0

            if subqueries and (not section_numbers or len(subqueries) > len(section_numbers)):
                for sq in subqueries:
                    sq_emb = model.encode(sq, convert_to_tensor=True)
                    sims = util.cos_sim(sq_emb, section_embeddings)[0]
                    top_k = min(10, len(sims))
                    top_idx = torch.argsort(sims, descending=True)[:top_k]
                    top_scores = sims[top_idx]
                    median_score = float(torch.median(top_scores))
                    threshold = max(0.45, median_score - 0.05)
                    for idx, score in zip(top_idx.tolist(), top_scores.tolist()):
                        if score >= threshold:
                            matched[idx] = max(matched.get(idx, 0), float(score))

            if matched:
                sorted_matched = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10]
                indices, scores = zip(*sorted_matched)
            else:
                indices, scores = [], []

            st.session_state.search_history[-1]["results"] = {"indices": indices, "scores": scores}

        # --- AI MODE ---
        elif mode == "Ask AI":
            retrieved = retrieve_top_sections(query, sections_data, model, section_embeddings, top_k=4)
            ai_answer = generate_ai_answer(query, retrieved)
            st.session_state.search_history[-1]["results"] = {"ai_answer": ai_answer, "retrieved": retrieved}

    # -----------------------------
    # Display Search History
    # -----------------------------
    if st.session_state.search_history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Search History</h3>", unsafe_allow_html=True)

        for entry in reversed(st.session_state.search_history):
            q, m, res = entry["query"], entry["mode"], entry["results"]
            with st.expander(f"{m} → {q}"):
                if m == "Find Matching Sections":
                    if res and res["indices"]:
                        for idx, score in zip(res["indices"], res["scores"]):
                            sec = sections_data[idx]
                            st.markdown(f"**Section {sec.get('Section', '')}: {sec.get('Title', '')}**")
                            st.markdown(f"Description: {sec.get('Description', '')}")
                            st.markdown(f"Punishment: {sec.get('Punishment', '')}")
                            st.caption(f"Relevance score: {score:.3f}")
                    else:
                        st.write("No matching sections found.")
                elif m == "Ask AI":
                    if res:
                        st.success(res["ai_answer"])
                        st.markdown("<h4>Referenced Sections:</h4>", unsafe_allow_html=True)
                        for sec, score in res["retrieved"]:
                            with st.expander(f"Section {sec.get('Section', '')}: {sec.get('Title', '')}"):
                                st.write(sec.get('Description', ''))
                                st.caption(f"Relevance score: {score:.3f}")
