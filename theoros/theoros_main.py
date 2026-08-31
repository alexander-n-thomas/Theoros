
import textwrap

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import Sampler, ParentBased, TraceIdRatioBased
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from theoros import model
from theoros.graph_like_search import WikidataClient

resource = Resource.create(attributes={
    "service.name": "theoros"
})
quality_resource = Resource.create(attributes={
    "service.name": "theoros-quality"
})
safety_resource = Resource.create(attributes={
    "service.name": "theoros"
})
quality_sampler = ParentBased(root=TraceIdRatioBased(rate=0.10))
safety_sampler = ParentBased(root=TraceIdRatioBased(rate=0.10))
provider = TracerProvider(resource=resource)
quality_provider = TracerProvider(resource=quality_resource, sampler=quality_sampler)
safety_provider = TracerProvider(resource=safety_resource, sampler=safety_sampler)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://192.168.4.52:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

theoros_agent = Agent(
    model=model,
    instructions=textwrap.dedent("""
    You are Theoros, an expert in all things movies. You have access to a powerful movie database and can answer any 
    question about movies, directors, cinematographers, awards, and more. If you don't know the answer off the top of 
    your head, you can use other agents to look it up. 
    Always provide a concise and accurate answer to the user's query."""),
    capabilities=[Instrumentation()]
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