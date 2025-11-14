# web_search.py
from duckduckgo_search import DDGS

def search_cases(query: str, max_results: int = 5):
    """
    Search DuckDuckGo for Supreme Court / High Court cases.
    Returns list of dictionaries with title, snippet, and link.
    """
    search_query = f"{query} Supreme Court case OR High Court landmark case"
    results = []

    try:
        with DDGS() as ddg:
            for r in ddg.text(search_query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "link": r.get("href", "")
                })
    except Exception as e:
        print(f"⚠️ Web search error: {str(e)}")

    return results

# Example test
if __name__ == "__main__":
    cases = search_cases("murder punishment", max_results=3)
    for c in cases:
        print(c)
