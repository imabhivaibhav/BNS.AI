// Builds worker/_data/sections.json (+ embeddings.json) from data/laws_sections.json
// Runs automatically as part of `npm run build` (Cloudflare Pages build). Needs GEMINI_API_KEY as a build variable.
import fs from 'node:fs';

const raw = JSON.parse(fs.readFileSync('data/laws_sections.json', 'utf8'));
const clean = (t) => String(t || '').replace(/[ \t]+\n/g, '\n').trim();

const sections = raw.map((s) => {
  const n = String(s.Section || '').replace(/^Section\s*/i, '').trim();
  const pun = [...new Set((s.Punishment || [])
    .map((p) => [p.type, p.duration && `(${p.duration})`, p.classification && `[${p.classification}]`].filter(Boolean).join(' '))
    .filter((t) => t && t !== 'Unspecified'))].join('; ');
  return { n, num: n.replace(/\D/g, ''), title: clean(String(s.Title || '').replace(/^\s*\d+\s*\n/, '')), desc: clean(s.Description), pun };
});

fs.mkdirSync('worker/_data', { recursive: true });
fs.writeFileSync('worker/_data/sections.json', JSON.stringify(sections));
console.log(`sections.json written (${sections.length} sections)`);
if (process.argv.includes('--sections-only')) process.exit(0);

// Skip the API calls when an up-to-date index is already committed (delete embeddings.json to force a rebuild)
try {
  const current = JSON.parse(fs.readFileSync('worker/_data/embeddings.json', 'utf8'));
  if (current.length === sections.length) { console.log('embeddings.json is up to date, skipping.'); process.exit(0); }
} catch { /* missing or unreadable: rebuild */ }

const key = process.env.GEMINI_API_KEY;
if (!key) { console.warn('GEMINI_API_KEY is not set for the build, skipping embeddings. Add it as a build variable.'); process.exit(0); }

const MODEL = 'gemini-embedding-001';
const DIM = 256;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const unit = (v) => { const n = Math.hypot(...v) || 1; return v.map((a) => +(a / n).toFixed(5)); };

async function batch(texts) {
  for (let attempt = 1; attempt <= 6; attempt++) {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:batchEmbedContents?key=${key}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        requests: texts.map((t) => ({
          model: `models/${MODEL}`, content: { parts: [{ text: t }] },
          taskType: 'RETRIEVAL_DOCUMENT', outputDimensionality: DIM,
        })),
      }),
    });
    if (res.ok) return (await res.json()).embeddings.map((e) => unit(e.values));
    if (res.status === 429 || res.status >= 500) { await sleep(5000 * attempt); continue; }
    throw new Error(await res.text());
  }
  throw new Error('Gave up after repeated rate limits');
}

const texts = sections.map((s) => `Section ${s.n}: ${s.title}. ${s.desc}`.slice(0, 2000));
const vecs = [];
for (let i = 0; i < texts.length; i += 50) {
  vecs.push(...(await batch(texts.slice(i, i + 50))));
  console.log(`embedded ${vecs.length}/${texts.length}`);
  await sleep(1500);
}
fs.writeFileSync('worker/_data/embeddings.json', JSON.stringify(vecs));
console.log('embeddings.json written');
