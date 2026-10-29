# app.py  ->  streamlit run app.py
from datetime import datetime
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

# ---------- Masthead ----------
st.markdown(f"""
<div class="masthead">
  <h1>WAL.AI</h1>
  <p>Apni situation ya BNS section number likhiye. Hum relevant dhara, saza aur milte-julte court cases dikhayenge.</p>
  <div class="date">{datetime.now():%A, %d %B %Y}</div>
</div>""", unsafe_allow_html=True)

top_l, top_r = st.columns([4, 1])
with top_l:
    mode = st.radio("Mode", ["Ask AI", "Find Sections"], horizontal=True, label_visibility="collapsed",
                    help="Ask AI: samjha hua jawab + cases. Find Sections: sirf matching dhara.")
with top_r:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()


# ---------- Renderers ----------
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
            st.warning("Koi matching section nahi mila. Case ko doosre shabdon me likhkar dekhein.")
        return
    st.markdown(e["answer"])
    if e["sections"]:
        with st.expander("Sections used for this answer"):
            render_sections([x for x in e["sections"] if x[1] >= 0.28], show_punishment=False)
    if e["cases"]:
        st.markdown("<div class='sec-head' style='margin-top:.6rem'>Related court cases</div>", unsafe_allow_html=True)
        render_cases(e["cases"])


# ---------- History ----------
for e in st.session_state.chat:
    with st.chat_message("user"):
        st.write(e["query"])
    with st.chat_message("assistant", avatar="⚖️"):
        render_entry(e)

# ---------- Empty state: example prompts ----------
if not st.session_state.chat:
    st.markdown("**Try one of these**")
    examples = ["Someone stole my phone on the bus", "BNS 103 kya hai?", "Threatening someone over phone, punishment?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        col.button(ex, key=ex, use_container_width=True, on_click=lambda t=ex: st.session_state.update(pending=t))

prompt = st.chat_input("Apna case ya sawal likhiye...") or st.session_state.pop("pending", None)

if prompt and prompt.strip():
    query = prompt.strip()
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant", avatar="⚖️"):
        entry = {"query": query, "mode": mode}
        if mode == "Find Sections":
            with st.spinner("Sections dhoondh rahe hain..."):
                entry["sections"] = backend.find_matching_sections(query, sections, model, embeds)
        else:
            history = [(e["query"], e["answer"]) for e in st.session_state.chat if e["mode"] == "Ask AI"]
            with st.spinner("Jawab taiyar ho raha hai..."):
                entry.update(backend.run_ai_mode(query, sections, model, embeds, history))
        render_entry(entry)
    st.session_state.chat.append(entry)

st.markdown("<div class='disclaimer'>WAL.AI general jaankari deta hai, kanooni salah nahi. "
            "Zaroori maamlon me vakeel se milein.</div>", unsafe_allow_html=True)
