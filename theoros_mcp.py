from mcp.server.fastmcp import FastMCP

from theoros.theoros_main import theoros_agent

server = FastMCP('Pydantic AI Server')

@server.tool()
async def theoros(query: str) -> str:
    """
    The MCP tool for theoros - an agent to assist with various related to films.

    Theoros offers the following functionalities
    - Film information retrieval
    - Film recommendation
    - Plot writing and review

    :param query: The query to ask theoros
    :return: The response from theoros
    """
    r = await theoros_agent.run(query)
    return r.output

def main():
    server.run()

if __name__ == '__main__':
    main()