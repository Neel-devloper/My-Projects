"""Simple command-line search utility using Google Custom Search API.

This script allows users to search for general links, images, or videos. If
no special suffix is provided, it will search Wikipedia for the query and
scrape the resulting page for its title and paragraphs.

Usage examples:
    python search engine.py
    # then type your query, optionally ending with /images, /videos or /links

API credentials must be supplied through the ``GOOGLE_API_KEY`` and
``SEARCH_ENGINE_ID`` environment variables.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"


def google_search(query: str, search_type: Optional[str] = None) -> List[str]:
    """Return a list of result links from Google Custom Search.

    Parameters
    ----------
    query:
        The search query to execute.
    search_type:
        Optional type of search ("image" for image results).
    """

    params = {"q": query, "key": API_KEY, "cx": SEARCH_ENGINE_ID}
    if search_type == "image":
        params["searchType"] = "image"

    response = requests.get(GOOGLE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return [item.get("link", "") for item in data.get("items", [])]


def display_links(links: List[str]) -> None:
    """Print each link on a new line."""

    if not links:
        print("No results found.")
        return
    for link in links:
        print(link)
        print()


def scrape_wikipedia(query: str) -> None:
    """Search Wikipedia for the query and print the page content."""

    links = google_search(f"{query} wikipedia")
    if not links:
        print("No results found.")
        return

    response = requests.get(links[0], timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("h1")
    if title_tag:
        print(f"Title: {title_tag.text}")

    for i, paragraph in enumerate(soup.find_all("p"), start=1):
        text = paragraph.get_text(strip=True)
        if text:
            print(f"Paragraph {i}: {text}")


def main() -> None:
    print("Add '/images' to search for images at the end of your search")
    print("Add '/links' to search for links at the end of your search")
    print("Add '/videos' to search for videos at the end of your search\n")

    query = input("Search Anything: ").strip()

    if query.endswith("/images"):
        clean_query = query[:-len("/images")].strip()
        links = google_search(clean_query, search_type="image")
        display_links(links)
    elif query.endswith("/videos"):
        clean_query = query[:-len("/videos")].strip() + " youtube"
        links = google_search(clean_query)
        display_links(links)
    elif query.endswith("/links"):
        clean_query = query[:-len("/links")].strip()
        links = google_search(clean_query)
        display_links(links)
    else:
        scrape_wikipedia(query)


if __name__ == "__main__":
    if not API_KEY or not SEARCH_ENGINE_ID:
        print(
            "Error: API credentials missing. Set GOOGLE_API_KEY and "
            "SEARCH_ENGINE_ID environment variables."
        )
        sys.exit(1)
    main()

