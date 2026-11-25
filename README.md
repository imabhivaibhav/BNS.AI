# WAL.AI

BNS (Bharatiya Nyaya Sanhita) assistant. React + Vite frontend, Cloudflare Pages Functions backend, Gemini for answers and search, Belicit (Supabase) login.

## Deploy (GitHub upload + Cloudflare Pages, no local commands)

1. Upload the contents of this folder to your GitHub repo (keep the folder structure) and delete the old Streamlit files:
   `app.py`, `appy_with_sec.py`, `ai_mode.py`, `backend.py`, `belicit_auth.py`, `web_search.py`, `style.css`, `requirements.txt`, `config.toml`, `devcontainer.json`.
2. Cloudflare Pages > connect the repo. Build command: `npm run build`. Output directory: `dist`.
3. Settings > Build > Variables and secrets (used while building):
   `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `GEMINI_API_KEY`
4. Settings > Variables and secrets (used at runtime, add as Secrets):
   `GEMINI_API_KEY` and optionally `TAVILY_API_KEY` (related court cases), `GEMINI_MODEL`
5. Retry the deployment. The build creates the section index from `data/laws_sections.json` with Gemini, then builds the site.
6. Point `wal.belicit.com` at the Pages project.

To rebuild the index after editing the sections data, delete `functions/_data/embeddings.json` (leave `[]` in it) and redeploy.

## Local development (optional)

```
cp .dev.vars.example .dev.vars   # GEMINI_API_KEY
cp .env.example .env             # Supabase values
GEMINI_API_KEY=your_key npm run build
npm run dev:api                  # terminal 1
npm run dev                      # terminal 2
```
