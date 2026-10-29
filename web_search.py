# web_search.py
from urllib.parse import urlparse

try:
    from ddgs import DDGS
except ImportError:  # purana package
    from duckduckgo_search import DDGS

TRUSTED = {
    "indiankanoon.org": 3, "main.sci.gov.in": 3, "sci.gov.in": 3,
    "livelaw.in": 2, "barandbench.com": 2, "scobserver.in": 2, "verdictum.in": 2,
}
BLOCKED = ("quora.com", "pinterest.", "facebook.com", "youtube.com", "reddit.com")


def _domain(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def search_cases(query: str, max_results: int = 5, hint: str = ""):
    """India-focused case search. `hint` = top matched BNS section title (better query)."""
    topic = f"{hint} {query}".strip() if hint and hint.lower() not in query.lower() else query
    queries = [
        f"{topic} Supreme Court judgment site:indiankanoon.org",
        f"{topic} landmark judgment India Supreme Court High Court",
    ]
    seen, found = set(), []
    for q in queries:
        try:
            with DDGS() as d:
                rows = d.text(q, region="in-en", safesearch="moderate", max_results=max_results * 2)
                for r in rows or []:
                    link = r.get("href", "")
                    dom = _domain(link)
                    if not link or link in seen or any(b in dom for b in BLOCKED):
                        continue
                    seen.add(link)
                    found.append({
                        "title": r.get("title", "").strip() or "Untitled",
                        "snippet": r.get("body", "").strip(),
                        "link": link,
                        "source": dom,
                        "rank": max((v for k, v in TRUSTED.items() if dom.endswith(k)), default=0),
                    })
        except Exception as e:
            print(f"Web search error: {e}")
    found.sort(key=lambda x: x["rank"], reverse=True)  # stable: trusted sources pehle
    return found[:max_results]
