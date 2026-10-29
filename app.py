# app.py  ->  streamlit run app.py
import streamlit as st

import backend

st.set_page_config(page_title="WAL.AI - BNS Legal Guide", page_icon="⚖️", layout="centered")

with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def get_sections():
    return backend.load_sections()


@st.cache_resource(show_spinner="Loading language model...")
def get_model():
    return backend.load_model()


@st.cache_resource(show_spinner="Indexing BNS sections...")
def get_embeddings():
    return backend.embed_sections(get_sections(), get_model())


sections, model, embeds = get_sections(), get_model(), get_embeddings()

if "chat" not in st.session_state:
    st.session_state.chat = []

st.markdown("""
<div class="masthead">
  <h1>WAL.AI</h1>
  <p>Describe a situation or enter a BNS section number. Get the relevant sections, the punishment, and related court cases.</p>
</div>""", unsafe_allow_html=True)

left, right = st.columns([4, 1])
with left:
    mode = st.radio("Mode", ["Ask AI", "Find Sections"], horizontal=True, label_visibility="collapsed",
                    help="Ask AI: a plain-language answer with related cases. Find Sections: matching sections only.")
with right:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()


def render_sections(items, show_punishment=True):
    for sec, score in items:
        with st.expander(f"Section {sec.get('Section', '')} - {sec.get('Title', '')}"):
            st.write(sec.get("Description", ""))
            if show_punishment and sec.get("Punishment"):
                st.markdown(f"<div class='punish'><b>Punishment:</b> {sec['Punishment']}</div>", unsafe_allow_html=True)
            st.progress(min(max(score, 0.0), 1.0), text=f"Match {score:.0%}")


def render_cases(cases):
    for c in cases:
        snip = (c["snippet"][:220] + "...") if len(c["snippet"]) > 220 else c["snippet"]
        st.markdown(
            f"<div class='case'><a href='{c['link']}' target='_blank'>{c['title']}</a>"
            f"<div class='src'>{c['source']}</div><div class='snip'>{snip}</div></div>",
            unsafe_allow_html=True)


def render_entry(e):
    if e["mode"] == "Find Sections":
        if e["sections"]:
            st.markdown("<div class='sec-head'>Matching sections</div>", unsafe_allow_html=True)
            render_sections(e["sections"])
        else:
            st.warning("No matching sections found. Try describing the case in different words.")
        return
    st.markdown(e["answer"])
    used = [x for x in e["sections"] if x[1] >= 0.28]
    if used:
        st.markdown("<div class='sec-head'>Sections referenced</div>", unsafe_allow_html=True)
        render_sections(used, show_punishment=False)
    if e["cases"]:
        st.markdown("<div class='sec-head'>Related court cases</div>", unsafe_allow_html=True)
        render_cases(e["cases"])


for e in st.session_state.chat:
    with st.chat_message("user"):
        st.write(e["query"])
    with st.chat_message("assistant", avatar="⚖️"):
        render_entry(e)

if not st.session_state.chat:
    st.markdown("**Try one of these**")
    examples = ["Someone stole my phone on the bus", "What is BNS 103?", "Threatening someone over the phone"]
    for col, ex in zip(st.columns(len(examples)), examples):
        col.button(ex, key=ex, use_container_width=True, on_click=lambda t=ex: st.session_state.update(pending=t))

prompt = st.chat_input("Describe your case or ask a question...") or st.session_state.pop("pending", None)

if prompt and prompt.strip():
    query = prompt.strip()
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant", avatar="⚖️"):
        entry = {"query": query, "mode": mode}
        if mode == "Find Sections":
            with st.spinner("Searching sections..."):
                entry["sections"] = backend.find_matching_sections(query, sections, model, embeds)
        else:
            history = [(e["query"], e["answer"]) for e in st.session_state.chat if e["mode"] == "Ask AI"]
            with st.spinner("Preparing your answer..."):
                entry.update(backend.run_ai_mode(query, sections, model, embeds, history))
        render_entry(entry)
    st.session_state.chat.append(entry)

st.markdown("<div class='disclaimer'>WAL.AI gives general information, not legal advice. "
            "Consult a lawyer for important matters.</div>", unsafe_allow_html=True)
