# Web scraping for given URLs
#
# Uses Tavily's extract API rather than requests+bs4 — plain HTTP requests
# get blocked or return near-empty HTML on JS-heavy/login-walled sites
# (LinkedIn, Instagram, Handshake all fall in that category); Tavily
# handles that far more robustly. Even so, login-walled sites will still
# fail or return only generic page chrome — this degrades gracefully
# rather than crashing ingestion when that happens.
from functools import lru_cache  # to cache the single Tavily client instance

from langchain_core.documents import Document
from tavily import TavilyClient

from src import config

MIN_USEFUL_CONTENT_LENGTH = 200  # shorter than this is almost certainly boilerplate/nav chrome, not real content


@lru_cache(maxsize=1)  # shared with the live web-search fallback in rag_agent/nodes/retrieval.py
def get_tavily_client() -> TavilyClient | None:
    if not config.TAVILY_API_KEY:  # not configured — callers should handle None
        return None
    return TavilyClient(api_key=config.TAVILY_API_KEY)


def load_websites() -> list[Document]:
    """Fetches config.WEBSITE_URLS via Tavily and returns one Document per
    URL that actually yielded substantive content. Returns an empty list
    (rather than raising) if no URLs or API key are configured."""
    client = get_tavily_client()
    if not config.WEBSITE_URLS or client is None:  # nothing configured — skip this source cleanly
        return []

    documents: list[Document] = []

    try:
        response = client.extract(urls=config.WEBSITE_URLS, extract_depth="advanced", format="text")
    except Exception as e:  # network error, bad key, etc. — degrade gracefully rather than crash ingestion
        print(f"website_scraper: Tavily extract call failed: {e}")
        return []

    for failure in response.get("failed_results", []):  # sites that outright refused to be fetched
        print(f"website_scraper: failed to fetch {failure['url']}: {failure['error']}")

    for result in response.get("results", []):  # sites that did respond
        text = result.get("raw_content", "")
        if len(text) < MIN_USEFUL_CONTENT_LENGTH:  # likely just nav/boilerplate (e.g. a JS-rendered profile page)
            print(f"website_scraper: skipping {result['url']}, content too short to be useful ({len(text)} chars)")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": result["url"], "source_type": "website"},
            )
        )

    return documents
