from typing import List

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation

from theoros import model


class WikidataTriple(BaseModel):
    subject: str = Field(description="The variable name (e.g., '?movie' or '?dp')")
    predicate: str = Field(
        description="The Wikidata property ID starting with 'wdt:', e.g., 'wdt:P57' (directed by) or 'wdt:P166' (award received)")
    obj: str = Field(description="The variable name OR the exact Wikidata Q-ID, e.g., '?dp' or 'wd:Q222956'")


class SparqlQuerySchema(BaseModel):
    """Structured plan containing the exact properties and items needed to build a valid SPARQL string."""
    target_variable: list[str] = Field(description="The variables to return, e.g., '?movie'")
    triples: List[WikidataTriple] = Field(
        description="The logical triples defining the graph traversal query conditions.")


class WikidataClient:
    """Helper client to interact with Wikidata APIs."""

    def __init__(self):
        self.sparql_url = "https://query.wikidata.org/sparql"
        self.search_url = "https://www.wikidata.org/w/api.php"
        self.headers = {"User-Agent": "Theoros/1.0 (alex@thirdhelix.com)"}

    def entity_search(self, search_text: str) -> str:
        """Search text and return a summary of matches with their Q-IDs."""
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json", "formatversion": "2",
            "ns0": "1", "ns120": "1", # Limit to items and properties
            "search": search_text
        }
        res = httpx.get(self.search_url, params=params, headers=self.headers)
        res = res.json()
        results = res.get("search", [])
        return "\n".join([f"ID: {r['id']} | Label: {r['label']} ({r.get('description', '')})" for r in results[:3]])

    def sparql_query(self, query: str) -> dict:
        """Execute a SPARQL query and return the JSON results."""
        response = httpx.get(
            self.sparql_url,
            params={"query": query, "format": "json"},
            headers=self.headers
        )
        if response.status_code != 200:
            raise ValueError(f"Error executing SPARQL query: {response.text}")
        return response.json()


graph_search_agent = Agent(
    model,
    deps_type=WikidataClient,
    system_prompt=(
        "You are an elite Semantic Web Agent translating natural language into Wikidata SPARQL logic.\n"
        "Step 1: ALWAYS use the lookup tool to verify the Q-IDs or P-IDs for entities/properties if you are unsure.\n"
        "Step: 2: Break the query down into a series of triples that represent the graph traversal needed to answer the question.\n"
        "Step 3: Generate the exact triples mapping the user's graph question.\n"
        "Common knowledge properties for reference:\n"
        "- P31: instance of\n"
        "- P11424: film (Q-ID)\n"
        "- P166: award received\n\n"
        "If the graph query fails try to answer the question directly using your internal knowledge, but always try "
        "to use the graph search for complex queries that require multiple steps or specific details that are "
        "unlikely to be in your training data."
    ),
    capabilities=[Instrumentation()]
)


@graph_search_agent.tool
def lookup_wikidata_id(ctx: RunContext[WikidataClient], search_term: str) -> str:
    """Use this tool first to discover or verify exact Wikidata Q-IDs or P-IDs for roles, movies, or awards."""
    print("Searching Wikidata for:", search_term)
    lookup_results = ctx.deps.entity_search(search_term)
    print("Lookup results:\n", lookup_results)
    return lookup_results


@graph_search_agent.tool
def execute_wikidata_sparql(ctx: RunContext[WikidataClient], query_plan: SparqlQuerySchema) -> str:
    """
    Executes a structured SPARQL plan against Wikidata.
    Pass the variables and verified P/Q numbers generated in the plan.
    """
    select_clauses = []

    for var in query_plan.target_variable:
        select_clauses.append(var)

    select_block = " ".join(select_clauses)

    where_clauses = []
    for t in query_plan.triples:
        where_clauses.append(f"  {t.subject} {t.predicate} {t.obj} .")

    where_block = "\n".join(where_clauses)

    sparql_query = f"""
    SELECT DISTINCT {select_block} 
    WHERE {{
        {where_block}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 100
    """

    print(f"\n--- [Generated SPARQL] ---\n{sparql_query}\n")

    data = ctx.deps.sparql_query(query=sparql_query)

    bindings = data.get("results", {}).get("bindings", [])

    output = []
    for b in bindings:
        row = []
        for var in b:
            if "value" in b[var] and var.endswith("Label"):
                row.append(f"{var}: {b[var]['value']}")
        output.append(", ".join(row))

    for v in output:
        print(v)

    return "\n".join(output) if output \
        else "No results found matching that specific logic."
