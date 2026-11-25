import { useEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import { redeemFromUrl, describeUser, logout, LOGIN_URL } from './auth';
import { ask, find } from './api';

const uid = () => Math.random().toString(36).slice(2, 10);
const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
const store = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage full */ } };

const LANGS = ['Auto', 'English', 'Hindi', 'Hinglish'];
const LANG_LABEL = { Hindi: 'हिन्दी' };
const SIZES = { Small: 14, Medium: 16, Large: 18 };
const STARTERS = ['Someone stole my phone on the bus', 'What is BNS section 103?', 'Threatening someone over a phone call'];
const DEFAULTS = { theme: 'system', lang: 'Auto', size: 'Medium' };

export default function App() {
  const [auth, setAuth] = useState({ loading: true });
  useEffect(() => { redeemFromUrl().then((r) => setAuth({ loading: false, ...r })); }, []);
  if (auth.loading) return <div className="center muted">Loading…</div>;
  if (!auth.user) return <Login error={auth.error} />;
  return <Shell user={auth.user} />;
}

function Login({ error }) {
  return (
    <div className="center">
      <div className="login">
        <img src="/favicon.png" alt="" width="56" height="56" />
        <h1 className="serif">Log in to WAL.AI</h1>
        <p className="muted">Use your Belicit account to ask questions about the Bharatiya Nyaya Sanhita.</p>
        {error && <p className="err">{error}</p>}
        <a className="btn primary" href={LOGIN_URL}>Log in with Belicit</a>
      </div>
    </div>
  );
}

function Shell({ user }) {
  const me = describeUser(user);
  const chatKey = `wal_chats_${me.id || me.display}`;
  const [settings, setSettings] = useState({ ...DEFAULTS, ...load('wal_settings', {}) });
  const [convos, setConvos] = useState(() => load(chatKey, []));
  const [activeId, setActiveId] = useState(null);
  const [mode, setMode] = useState('ask');
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [nav, setNav] = useState(false);
  const [menu, setMenu] = useState(false);
  const [modal, setModal] = useState(null);
  const endRef = useRef(null);

  const active = convos.find((c) => c.id === activeId);
  const entries = active?.entries || [];

  useEffect(() => {
    const root = document.documentElement;
    if (settings.theme === 'system') delete root.dataset.theme; else root.dataset.theme = settings.theme;
    root.style.setProperty('--fs', `${SIZES[settings.size]}px`);
    store('wal_settings', settings);
  }, [settings]);

  useEffect(() => {
    if (busy) return;
    store(chatKey, convos.slice(0, 50).map((c) => ({ ...c, entries: c.entries.map((e) => ({ ...e, pending: false })) })));
  }, [convos, busy, chatKey]);

  useEffect(() => { if (busy) endRef.current?.scrollIntoView({ block: 'end' }); }, [entries, busy]);

  async function send(text) {
    const query = text.trim();
    if (!query || busy) return;
    setInput('');
    setBusy(true);
    const entry = { id: uid(), query, mode, answer: '', sections: [], cases: [], pending: true };
    const history = entries.filter((e) => e.mode === 'ask' && !e.error)
      .map((e) => ({ q: e.query, a: e.answer, sections: e.sections.map((s) => s.n) }));
    let cid = activeId;
    if (!cid) {
      cid = uid();
      setActiveId(cid);
      setConvos((c) => [{ id: cid, title: query.slice(0, 48), updated: Date.now(), entries: [entry] }, ...c]);
    } else {
      setConvos((c) => c.map((x) => (x.id === cid ? { ...x, updated: Date.now(), entries: [...x.entries, entry] } : x)));
    }
    const patch = (p) => setConvos((c) => c.map((x) => (x.id !== cid ? x : {
      ...x, entries: x.entries.map((e) => (e.id === entry.id ? { ...e, ...(typeof p === 'function' ? p(e) : p) } : e)),
    })));
    try {
      if (mode === 'find') {
        patch({ sections: (await find(query)).sections });
      } else {
        await ask({ query, history, language: settings.lang }, (m) => {
          if (m.type === 'meta') patch({ sections: m.sections });
          else if (m.type === 'text') patch((e) => ({ answer: e.answer + m.t }));
          else if (m.type === 'cases') patch({ cases: m.cases });
          else if (m.type === 'error') throw new Error(m.message);
        });
      }
      patch({ pending: false });
    } catch (err) {
      patch({ pending: false, error: err.message });
    } finally {
      setBusy(false);
    }
  }

  const newChat = () => { setActiveId(null); setNav(false); setInput(''); };
  const hour = new Date().getHours();
  const hello = settings.lang === 'Hindi' ? 'नमस्ते' : hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
  const composer = <Composer {...{ input, setInput, send, busy, mode, setMode }} />;

  return (
    <div className={`app ${nav ? 'nav-open' : ''}`}>
      <aside className="side">
        <div className="brand"><img src="/favicon.png" alt="" width="30" height="30" /><span className="serif">WAL.AI</span></div>
        <button className="btn newchat" onClick={newChat}>+ New chat</button>
        <div className="list">
          {convos.length > 0 && <div className="label">Recent</div>}
          {convos.map((c) => (
            <div key={c.id} className={`item ${c.id === activeId ? 'on' : ''}`}>
              <button className="title" onClick={() => { setActiveId(c.id); setNav(false); }}>{c.title}</button>
              <button className="del" aria-label="Delete chat" onClick={() => { setConvos((all) => all.filter((x) => x.id !== c.id)); if (c.id === activeId) setActiveId(null); }}>×</button>
            </div>
          ))}
        </div>
        <div className="me">
          {menu && (
            <div className="menu" onMouseLeave={() => setMenu(false)}>
              <button onClick={() => { setModal('account'); setMenu(false); }}>Account details</button>
              <button onClick={() => { setModal('settings'); setMenu(false); }}>Settings</button>
              <button onClick={() => { logout(); location.reload(); }}>Log out</button>
            </div>
          )}
          <button className="who" onClick={() => setMenu((m) => !m)}>
            <span className="avatar">{me.display[0]?.toUpperCase()}</span>
            <span className="names"><b>{me.username ? `@${me.username}` : me.display}</b>{me.email && <small>{me.email}</small>}</span>
          </button>
        </div>
      </aside>
      {nav && <div className="scrim" onClick={() => setNav(false)} />}

      <main className="main">
        <header className="top">
          <button className="icon" aria-label="Open menu" onClick={() => setNav(true)}>☰</button>
          <span className="serif">{active?.title || 'WAL.AI'}</span>
        </header>
        {entries.length === 0 ? (
          <div className="empty">
            <div className="hero"><img src="/favicon.png" alt="" width="40" height="40" /><h1 className="serif">{hello}, {me.display}</h1></div>
            <p className="muted">Ask about a situation or a BNS section.</p>
            {composer}
            <div className="starters">{STARTERS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}</div>
          </div>
        ) : (
          <>
            <div className="thread"><div className="col">
              {entries.map((e) => <Entry key={e.id} e={e} />)}
              <div ref={endRef} />
            </div></div>
            <div className="dock">{composer}</div>
          </>
        )}
      </main>

      {modal === 'settings' && <SettingsModal settings={settings} setSettings={setSettings} close={() => setModal(null)} />}
      {modal === 'account' && <AccountModal me={me} close={() => setModal(null)} />}
    </div>
  );
}

function Composer({ input, setInput, send, busy, mode, setMode }) {
  const ref = useRef(null);
  const grow = (el) => { el.style.height = 'auto'; el.style.height = `${Math.min(el.scrollHeight, 200)}px`; };
  useEffect(() => { if (ref.current && !input) ref.current.style.height = 'auto'; }, [input]);
  return (
    <div className="composer">
      <textarea ref={ref} rows={1} value={input} placeholder={mode === 'ask' ? 'Ask WAL about a case or a BNS section' : 'Type a section number or describe the offence'}
        onChange={(ev) => { setInput(ev.target.value); grow(ev.target); }}
        onKeyDown={(ev) => { if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); send(input); } }} />
      <div className="bar">
        <div className="seg">
          <button className={mode === 'ask' ? 'on' : ''} onClick={() => setMode('ask')}>Ask AI</button>
          <button className={mode === 'find' ? 'on' : ''} onClick={() => setMode('find')}>Find sections</button>
        </div>
        <button className="send" disabled={busy || !input.trim()} aria-label="Send" onClick={() => send(input)}>↑</button>
      </div>
      <div className="fine">WAL gives general information, not legal advice. Consult a lawyer for important matters.</div>
    </div>
  );
}

function Entry({ e }) {
  const outOfScope = e.answer.startsWith('❌');
  return (
    <>
      <div className="u"><div className="bubble">{e.query}</div></div>
      <div className="a">
        <img src="/favicon.png" alt="" width="28" height="28" />
        <div className="body">
          {e.mode === 'ask' && (e.answer ? <div className="md"><Markdown>{e.answer}</Markdown></div> : e.pending && <div className="dots"><i /><i /><i /></div>)}
          {e.mode === 'find' && e.pending && <div className="dots"><i /><i /><i /></div>}
          {e.mode === 'find' && !e.pending && !e.error && !e.sections.length && <p className="muted">No matching sections found. Try describing the case in different words.</p>}
          {e.error && <p className="err">{e.error}</p>}
          {e.sections.length > 0 && !outOfScope && (
            <>
              <div className="label">{e.mode === 'find' ? 'Matching sections' : 'Sections referenced'}</div>
              {e.sections.map((s) => (
                <details key={s.n} className="card">
                  <summary>Section {s.n}: {s.title}</summary>
                  <p className="desc">{s.desc}</p>
                  {s.pun && <p className="pun"><b>Punishment:</b> {s.pun}</p>}
                  <small className="muted">Match {Math.round(s.score * 100)}%</small>
                </details>
              ))}
            </>
          )}
          {e.cases?.length > 0 && (
            <>
              <div className="label">Related court cases</div>
              {e.cases.map((c) => (
                <a key={c.link} className="card case" href={c.link} target="_blank" rel="noreferrer">
                  <b>{c.title}</b><small className="muted">{c.source}</small><span>{c.snippet}</span>
                </a>
              ))}
            </>
          )}
        </div>
      </div>
    </>
  );
}

function Modal({ title, close, children }) {
  return (
    <div className="scrim on" onClick={close}>
      <div className="modal" role="dialog" aria-label={title} onClick={(ev) => ev.stopPropagation()}>
        <div className="mh"><h2 className="serif">{title}</h2><button className="icon" aria-label="Close" onClick={close}>×</button></div>
        {children}
      </div>
    </div>
  );
}

function Choice({ label, value, options, onChange, names = {} }) {
  return (
    <div className="row"><span>{label}</span>
      <div className="seg">{options.map((o) => <button key={o} className={value === o ? 'on' : ''} onClick={() => onChange(o)}>{names[o] || o}</button>)}</div>
    </div>
  );
}

function SettingsModal({ settings, setSettings, close }) {
  const set = (k) => (v) => setSettings((s) => ({ ...s, [k]: v }));
  return (
    <Modal title="Settings" close={close}>
      <Choice label="Theme" value={settings.theme} options={['system', 'light', 'dark']} names={{ system: 'System', light: 'Light', dark: 'Dark' }} onChange={set('theme')} />
      <Choice label="Reply language" value={settings.lang} options={LANGS} names={LANG_LABEL} onChange={set('lang')} />
      <Choice label="Text size" value={settings.size} options={Object.keys(SIZES)} onChange={set('size')} />
      <p className="fine left"><b>Privacy.</b> Chats are stored only in this browser. To answer, your question is sent to Google Gemini{' '}
        (and to a web search service for related cases). Avoid entering personal or sensitive details. Questions: wal@belicit.com</p>
      <p className="fine left">WAL is provided as is for general information. It is not a law firm and does not give legal advice. © {new Date().getFullYear()} Belicit.</p>
    </Modal>
  );
}

function AccountModal({ me, close }) {
  const rows = [['Username', me.username && `@${me.username}`], ['Name', me.name], ['Email', me.email],
    ['Member since', me.created && new Date(me.created).toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' })],
    ['User ID', me.id]].filter((r) => r[1]);
  return (
    <Modal title="Account" close={close}>
      <div className="acct"><span className="avatar big">{me.display[0]?.toUpperCase()}</span><div><b>{me.display}</b><div className="muted">Signed in with Belicit</div></div></div>
      {rows.map(([k, v]) => <div className="row" key={k}><span className="muted">{k}</span><span className="val">{v}</span></div>)}
      <a className="btn" href="https://www.belicit.com" target="_blank" rel="noreferrer">Manage your Belicit profile</a>
    </Modal>
  );
}
