
import textwrap

from pydantic_ai import Agent


from theoros import model
from theoros.graph_like_search import WikidataClient

theoros_agent = Agent(
    model=model,
    instructions=textwrap.dedent("""
    You are Theoros, an expert in all things movies. You have access to a powerful movie database and can answer any 
    question about movies, directors, cinematographers, awards, and more. If you don't know the answer off the top of 
    your head, you can use other agents to look it up. 
    Always provide a concise and accurate answer to the user's query.""")
)

wikidata_agent = WikidataClient()

@theoros_agent.tool_plain
async def basic_movie_query(query: str) -> str:
    """Use this tool to answer basic movie-related questions using your internal knowledge."""
    print(f"Received basic movie query: {query}")
    from theoros.movie_expert import movie_expert
    result = await movie_expert.run(query)
    return result.response.text

@theoros_agent.tool_plain
async def graph_like_query(query: str) -> str:
    """Use this tool for more complex queries that require structured reasoning or multiple steps."""
    print(f"Received graph-like query: {query}")
    from theoros.graph_like_search import graph_search_agent
    result = await graph_search_agent.run(query, deps=wikidata_agent)
    return result.response.text

@theoros_agent.tool_plain
async def plot_query(query: str) -> str:
    """Use this tool to recommend movies based off their plots."""
    print(f"Received plot query: {query}")
    from theoros.plot_search import plot_search_agent
    result = await plot_search_agent.run(query)
    print(result.response.text)
    return result.response.text

@theoros_agent.tool_plain
async def pitch_query(query: str) -> str:
    """Use this tool to generate a movie pitch based off a list of bullet points."""
    print(f"Received pitch query: {query}")
    from theoros.pitch_builder import pitch_agent
    result = await pitch_agent.run(query)
    return result.response.text

@theoros_agent.tool_plain
async def pitch_review(query: str) -> str:
    """Use this tool to review a movie pitch and provide feedback."""
    print(f"Received pitch review query: {query}")
    from theoros.pitch_reviewer import pitch_review_agent
    result = await pitch_review_agent.run(query)
    return result.response.text