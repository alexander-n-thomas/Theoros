import textwrap

from pydantic_ai import Agent

from theoros import model
from theoros.wiki_util import search_wikipedia

movie_expert = Agent(
    model=model,
    instructions=textwrap.dedent("""
    You are an expert on movies and cinematic history. You are known for
    your encyclopedic knowledge of movies. If there is something you
    don't know off the top of your head, you know where to look it up
    in Wikipedia.""")
)

movie_expert.tool_plain(search_wikipedia)

@movie_expert.instructions
def add_query() -> str:
    return (f"Answer the given query, using an article retrieved from "
            "Wikipedia.\n"
            "If no article is found, try and answer directly.\n"
            "Respond with only the answer to the query")

