# web_search.py
from ddgs import DDGS

def search_cases(query: str, max_results: int = 5):
    """
    Search DuckDuckGo for Supreme Court / High Court cases.
    Returns list of dicts with title, snippet, link.
    """
    search_query = f"{query} Supreme Court case OR High Court landmark case"
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "link": r.get("href", "")
                })
    except Exception as e:
        print(f"⚠️ Web search error: {str(e)}")

    if not results:
        print("⚠️ No cases found. Try simplifying your query.")

    return results

# Test
if __name__ == "__main__":
    cases = search_cases("murder punishment", max_results=3)
    for c in cases:
        print(c)
