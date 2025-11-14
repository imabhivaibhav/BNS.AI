# web_search.py

from duckduckgo_search import DDGS

def search_cases(query: str, max_results: int = 5):
    """
    Search DuckDuckGo for Supreme Court / High Court cases related to a query.

    Args:
        query (str): User query, e.g., 'murder punishment'.
        max_results (int): Max number of case results to return.

    Returns:
        List[dict]: Each dict contains 'title', 'snippet', 'link'.
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


# -----------------------------
# Example usage (for testing)
# -----------------------------
if __name__ == "__main__":
    query = "murder punishment"
    cases = search_cases(query, max_results=3)
    for i, c in enumerate(cases, start=1):
        print(f"{i}. {c['title']}")
        print(f"   {c['snippet']}")
        print(f"   Link: {c['link']}\n")
