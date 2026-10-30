# app.py  ->  streamlit run app.py
import os
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

import backend

ICON = "favicon.png" if os.path.exists("favicon.png") else "⚖️"
st.set_page_config(page_title="WAL.AI", page_icon=ICON, layout="centered", initial_sidebar_state="auto")

# ---------- Settings data ----------
THEMES = ["System", "Light", "Dark"]
ACCENTS = {"Blue": ("#4285F4", "#9B72CB"), "Green": ("#1E8E3E", "#34A853"), "Purple": ("#8B5CF6", "#C084FC"),
           "Orange": ("#F28B30", "#F5B14C"), "Rose": ("#E5484D", "#F08FA0")}
FONTS = {"Inter": "Inter", "Poppins": "Poppins", "Roboto": "Roboto",
         "Merriweather (serif)": "Merriweather", "Fira Code (mono)": "Fira Code"}
SIZES = {"Small": 14, "Medium": 16, "Large": 18}
LANGS = {"Auto": "", "English": "Reply in English.", "हिन्दी": "Reply in Hindi using Devanagari script.",
         "Hinglish": "Reply in Hinglish: Hindi written in Roman script."}
LIGHT = dict(bg="#F6F8FC", surface="#FFFFFF", surface2="#EDF1F8", text="#1B1F27", muted="#667085", link="#1A56DB",
             border="#DDE3EE", shadow="0 4px 18px rgba(16,24,40,.06)")
DARK = dict(bg="#15171C", surface="#222630", surface2="#2F3441", text="#EEF1F6", muted="#A3ABBB", link="#8AB4F8",
            border="#434958", shadow="0 4px 22px rgba(0,0,0,.35)")

T = {
    "en": dict(hello="Hello there", sub="How can I help with the law today?", ask="Ask AI", find="Find Sections",
               new="New chat", inp="Ask WAL.AI about a case or a BNS section", settings="Settings", theme="Theme",
               accent="Accent colour", font="Font style", size="Text size", lang="Primary language",
               think="Thinking...", search="Searching sections...", matching="Matching sections",
               refs="Sections referenced", cases="Related court cases",
               nomatch="No matching sections found. Try describing the case in different words.",
               disc="WAL.AI gives general information, not legal advice. Consult a lawyer for important matters.",
               saved="Settings are saved in the page URL.",
               ex=["📱 Someone stole my phone on the bus", "📖 What is BNS section 103?", "📞 Threatening someone over a phone call"]),
    "hi": dict(hello="नमस्ते", sub="आज मैं कानून से जुड़ी क्या मदद कर सकता हूँ?", ask="AI से पूछें", find="धाराएँ खोजें",
               new="नई चैट", inp="अपना केस या BNS धारा पूछें", settings="सेटिंग्स", theme="थीम",
               accent="रंग", font="फ़ॉन्ट", size="टेक्स्ट साइज़", lang="प्रमुख भाषा",
               think="सोच रहा हूँ...", search="धाराएँ खोज रहा हूँ...", matching="मिलती-जुलती धाराएँ",
               refs="संदर्भित धाराएँ", cases="संबंधित अदालती मामले",
               nomatch="कोई धारा नहीं मिली। केस को दूसरे शब्दों में लिखकर देखें।",
               disc="WAL.AI सामान्य जानकारी देता है, कानूनी सलाह नहीं। ज़रूरी मामलों में वकील से मिलें।",
               saved="सेटिंग्स पेज के URL में सेव होती हैं।",
               ex=["📱 किसी ने बस में मेरा फोन चुरा लिया", "📖 BNS धारा 103 क्या है?", "📞 फोन पर धमकी देना"]),
}


def pick(key, options, default):
    v = st.query_params.get(key, default)
    return v if v in options else default


def theme_css(theme, accent, font, px):
    a1, a2 = ACCENTS[accent]
    fam = FONTS[font]
    vb = lambda d: "".join(f"--{k}:{v};" for k, v in d.items())
    base = f"--accent:{a1};--accent2:{a2};--font:'{fam}';--fs:{px}px;"
    if theme == "Dark":
        root = f":root{{{base}{vb(DARK)}color-scheme:dark}}"
    elif theme == "Light":
        root = f":root{{{base}{vb(LIGHT)}color-scheme:light}}"
    else:
        root = f":root{{{base}{vb(LIGHT)}}}@media (prefers-color-scheme: dark){{:root{{{vb(DARK)}color-scheme:dark}}}}"
    imp = (f"@import url('https://fonts.googleapis.com/css2?family={fam.replace(' ', '+')}:wght@400;500;600"
           "&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');")
    return imp + root


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
st.session_state.setdefault("chat", [])

t = T["hi" if st.session_state.get("lang", pick("lang", LANGS, "Auto")) == "हिन्दी" else "en"]

# ---------- Sidebar (collapsible with the arrow at its top) ----------
with st.sidebar:
    st.markdown("<div class='brandrow'><div class='logo-w lg'>W</div><span class='brandname'>WAL.AI</span></div>",
                unsafe_allow_html=True)
    if st.button(t["new"], use_container_width=True, icon=":material/add:"):
        st.session_state.chat = []
        st.rerun()
    st.divider()
    st.markdown(f"<div class='side-label'>{t['settings']}</div>", unsafe_allow_html=True)
    theme = st.radio(t["theme"], THEMES, index=THEMES.index(pick("theme", THEMES, "System")), key="theme", horizontal=True)
    accent = st.selectbox(t["accent"], list(ACCENTS), index=list(ACCENTS).index(pick("accent", ACCENTS, "Blue")), key="accent")
    font = st.selectbox(t["font"], list(FONTS), index=list(FONTS).index(pick("font", FONTS, "Inter")), key="font")
    size = st.select_slider(t["size"], list(SIZES), value=pick("size", SIZES, "Medium"), key="size")
    lang = st.selectbox(t["lang"], list(LANGS), index=list(LANGS).index(pick("lang", LANGS, "Auto")), key="lang")
    st.caption(t["saved"])
st.query_params.update(theme=theme, accent=accent, font=font, size=size, lang=lang)

with open("style.css", encoding="utf-8") as f:
    # ONE <style> block only: a second tag on the same line makes markdown print the CSS as text
    st.markdown(f"<style>{theme_css(theme, accent, font, SIZES[size])}\n{f.read()}</style>",
                unsafe_allow_html=True)

# ---------- Top bar ----------
st.markdown("<div class='brandrow'><div class='logo-w'>W</div><span class='brandname grad'>WAL.AI</span></div>",
            unsafe_allow_html=True)
mode = st.segmented_control("Mode", ["Ask AI", "Find Sections"], default="Ask AI", label_visibility="collapsed",
                            format_func=lambda k: t["ask"] if k == "Ask AI" else t["find"]) or "Ask AI"


# ---------- Renderers ----------
def render_sections(items, show_punishment=True):
    for sec, score in items:
        with st.expander(f"Section {sec.get('Section', '')} - {sec.get('Title', '')}"):
            st.write(sec.get("Description", ""))
            if show_punishment and sec.get("Punishment"):
                st.markdown(f"<div class='punish'><b>Punishment:</b> {sec['Punishment']}</div>", unsafe_allow_html=True)
            st.caption(f"Match {score:.0%}")


def render_cases(cases):
    for c in cases:
        snip = (c["snippet"][:220] + "...") if len(c["snippet"]) > 220 else c["snippet"]
        st.markdown(f"<div class='case'><a href='{c['link']}' target='_blank'>{c['title']}</a>"
                    f"<div class='src'>{c['source']}</div><div class='snip'>{snip}</div></div>", unsafe_allow_html=True)


def render_entry(e):
    if e["mode"] == "Find Sections":
        if e["sections"]:
            st.markdown(f"<div class='sec-head'>{t['matching']}</div>", unsafe_allow_html=True)
            render_sections(e["sections"])
        else:
            st.warning(t["nomatch"])
        return
    st.markdown(e["answer"])
    if e["sections"] and not e["answer"].startswith(("❌", "⚠️")):
        st.markdown(f"<div class='sec-head'>{t['refs']}</div>", unsafe_allow_html=True)
        render_sections(e["sections"], show_punishment=False)
    if e["cases"]:
        st.markdown(f"<div class='sec-head'>{t['cases']}</div>", unsafe_allow_html=True)
        render_cases(e["cases"])


# ---------- Chat ----------
if not st.session_state.chat:
    st.markdown(f"<div class='hero'><div class='badge'>Powered by Belicit</div><div class='hello grad'>{t['hello']}</div><div class='sub'>{t['sub']}</div></div>",
                unsafe_allow_html=True)
    with st.container(key="chips"):
        for col, ex in zip(st.columns(3), t["ex"]):
            col.button(ex, key=f"chip_{ex}", use_container_width=True,
                       on_click=lambda q=ex: st.session_state.update(pending=q))

for e in st.session_state.chat:
    with st.chat_message("user"):
        st.write(e["query"])
    with st.chat_message("assistant", avatar=ICON):
        render_entry(e)

prompt = st.chat_input(t["inp"]) or st.session_state.pop("pending", None)

if prompt and prompt.strip():
    query = prompt.strip()
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant", avatar=ICON):
        entry = {"query": query, "mode": mode}
        if mode == "Find Sections":
            with st.spinner(t["search"]):
                entry["sections"] = backend.find_matching_sections(query, sections, model, embeds)
        else:
            history = [(e["query"], e["answer"]) for e in st.session_state.chat if e["mode"] == "Ask AI"]
            with st.spinner(t["think"]):
                entry.update(backend.run_ai_mode(query, sections, model, embeds, history, LANGS[lang]))
        render_entry(entry)
    st.session_state.chat.append(entry)


# ---------- Footer ----------
PRIVACY = ("**Privacy**\n\nWAL.AI itself does not save your chats to a database; they stay in your browser session. "
           "To answer, your question is sent to Google Gemini and to a web search service. "
           "Please avoid entering personal or sensitive details.\n\nQuestions: wal@belicit.com")
TERMS = ("**Terms of Use**\n\nWAL.AI is provided \"as is\" for general information only. Answers may be incomplete or "
         "out of date. Belicit is not liable for decisions made based on its output.")
LEGAL = ("**Disclaimer**\n\nWAL.AI is not a law firm and does not provide legal advice. For any real legal matter, "
         "please consult a qualified lawyer.")


def clock_html(theme):
    pal = lambda d: f"--t:{d['text']};--m:{d['muted']}"
    if theme == "Dark":
        css = f":root{{{pal(DARK)}}}"
    elif theme == "Light":
        css = f":root{{{pal(LIGHT)}}}"
    else:
        css = f":root{{{pal(LIGHT)}}}@media(prefers-color-scheme:dark){{:root{{{pal(DARK)}}}}}"
    return (f"<style>{css} body{{margin:0;font-family:system-ui,sans-serif;text-align:center;font-size:14px;color:var(--m)}}"
            "b{color:var(--t);font-weight:600}</style>"
            "<div>🕒 <b id='c'></b> &nbsp;·&nbsp; 📍 <b>India</b></div>"
            "<script>const f=()=>{document.getElementById('c').textContent=new Date().toLocaleString('en-IN',"
            "{timeZone:'Asia/Kolkata',weekday:'short',day:'2-digit',month:'short',year:'numeric',hour:'2-digit',"
            "minute:'2-digit',second:'2-digit'})+' IST'};f();setInterval(f,1000)</script>")


st.divider()
components.html(clock_html(theme), height=34)
p1, p2, p3 = st.columns(3)
with p1.popover("Privacy", use_container_width=True):
    st.markdown(PRIVACY)
with p2.popover("Terms", use_container_width=True):
    st.markdown(TERMS)
with p3.popover("Disclaimer", use_container_width=True):
    st.markdown(LEGAL)
st.markdown(f"""<div class='footer'>
<div>Support: <a href='mailto:wal@belicit.com'>wal@belicit.com</a></div>
<div>© {datetime.now().year} Belicit. All rights reserved.</div>
<div>{t['disc']}</div></div>""", unsafe_allow_html=True)
