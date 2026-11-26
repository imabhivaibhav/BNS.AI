# WAL.AI

BNS (Bharatiya Nyaya Sanhita) assistant. React + Vite frontend, Cloudflare Pages Function for the API, Gemini for answers and search, Belicit (Supabase) login.

## Deploy (GitHub upload + Cloudflare Pages)

1. Cloudflare dashboard > Workers & Pages > Create > Pages > Connect to Git, pick this repo.
2. Build command `npm run build`, output directory `dist`. No deploy command.
3. Settings > Variables and Secrets: add `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `GEMINI_API_KEY` (type Text, so the build can read it too), optionally `TAVILY_API_KEY` and `GEMINI_MODEL`.
4. Redeploy. The build creates the section index (`functions/_data/embeddings.json`) with Gemini, then builds the site.

To rebuild the index after editing the sections data, put `[]` in `functions/_data/embeddings.json` and redeploy.
