# WAL.AI

BNS (Bharatiya Nyaya Sanhita) assistant. React + Vite frontend, one Cloudflare Worker (API + static site), Gemini for answers and search, Belicit (Supabase) login.

## Deploy (GitHub upload + Cloudflare Workers Builds)

1. Delete the old files from the repo: the whole `functions` folder, `app.py`, `appy_with_sec.py`, `ai_mode.py`, `backend.py`, `belicit_auth.py`, `web_search.py`, `style.css`, `requirements.txt`, `config.toml`, `devcontainer.json`. Then upload this project.
2. Cloudflare project settings: build command `npm run build`, deploy command `npx wrangler deploy`.
3. Settings > Build > Variables and secrets (used while building):
   `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `GEMINI_API_KEY`
4. Settings > Variables and secrets (used at runtime, add as Secrets):
   `GEMINI_API_KEY`, optionally `TAVILY_API_KEY` (related court cases) and `GEMINI_MODEL`
5. `name` in `wrangler.toml` must equal your Worker's name.
6. Redeploy. The build creates the section index (`worker/_data/embeddings.json`) with Gemini, then builds the site.

To rebuild the index after editing the sections data, put `[]` in `worker/_data/embeddings.json` and redeploy.

## Local development (optional)

```
cp .dev.vars.example .dev.vars   # GEMINI_API_KEY
cp .env.example .env             # Supabase values
GEMINI_API_KEY=your_key npm run build
npm run dev:api                  # terminal 1
npm run dev                      # terminal 2
```
