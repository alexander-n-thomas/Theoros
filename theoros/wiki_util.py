import sys

import wikipediaapi
from wikipediaapi import WikiSearchSort, SearchSort, WikiSearchProp, WikiSearchInfo, WikiSearchWhat, WikiSearchQiProfile

def search_function(
    query: str,
    sort: WikiSearchSort = SearchSort.RELEVANCE,
    prop: list[WikiSearchProp] | None = None,
    info: list[WikiSearchInfo] | None = None,
    what: WikiSearchWhat | None = None,
    qi_profile: WikiSearchQiProfile | None = None
) -> wikipediaapi.SearchResults:
    """Search Wikipedia with either enum or string parameters."""
    wiki = wikipediaapi.Wikipedia('Theoros/1.0')
    return wiki.search(query, sort=sort, prop=prop, info=info, what=what, qi_profile=qi_profile)

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='Theoros (alex@thirdhelix.ai)',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def search_wikipedia(query: str) -> str:
    print(f"Searching Wikipedia for: {query}", file=sys.stderr)
    results = search_function(query)
    title = [result for result in results.pages]
    if not title:
        return "No article found"
    title = title[0]
    page = wiki_wiki.page(title)
    return page.summary
