import textwrap

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation

from theoros import model, all_models
from theoros.wiki_util import search_wikipedia


plot_search_agent_prompt_based = Agent(
    model,
    system_prompt=textwrap.dedent("""
        Imagine three different experts are answering this question.\n"
        1. The Movie Buff: This expert has an encyclopedic knowledge of movies and 
           can provide detailed information about movie plots, characters, and 
           storylines.
        2. The Film Critic: This expert has a deep understanding of film analysis 
           and can provide insights into the themes, motifs, and narrative 
           structures of movies.
        3. The Storyteller: This expert has a talent for summarizing and retelling 
           stories in a compelling way, capturing the essence of the plot while 
           keeping it engaging and concise.
        When answering the user's query, you will first generate three separate 
        answers from each of the experts. After that, you will synthesize these 
        three perspectives into a single, comprehensive answer that combines the 
        detailed knowledge of the Movie Buff, the analytical insights of the Film 
        Critic, and the engaging storytelling of the Storyteller. Always provide a 
        final answer that is concise, accurate, and captures the essence of the 
        plot in an engaging way.
        The experts should follow these steps
        1. Each expert should create a Wikipedia query to find a relevant movie 
           plot summary. If their search returns no results, they should create a 
           new query and try again.
        2. Each expert should provide a summary of the plot based on the Wikipedia 
           article.
        3. The final answer should rank the three films, combine them to create a 
           watchlist of the three films, and provide a final recommendation on  
           which ones to watch based on the user's query.
        Include the movies and reasoning for each expert in the final 
       answer, and provide a clear and concise summary of the plot for 
       each movie.
       Format the output as follows:
       Movie Buff:
       Movie Title: <title>
       Plot Summary: <summary>
       Reasoning: <reasoning>

       Film Critic:
       Movie Title: <title>
       Plot Summary: <summary>
       Reasoning: <reasoning>

       Storyteller:
       Movie Title: <title>
       Plot Summary: <summary>
       Reasoning: <reasoning>

       Final Recommendation:
       Watchlist: <movie1>, <movie2>, <movie3>
       Recommended Movie: <movie>
    """),
    capabilities=[Instrumentation()]
)

plot_search_agent_prompt_based.tool_plain(search_wikipedia)

plot_search_agent_manager = Agent(
    model,
    system_prompt=textwrap.dedent("""
        You are an expert in movie plots and storytelling. Your task is to take a 
        short list of bullet points and construct a coherent and engaging plot 
        summary that captures the essence of the story.
        You should rely on the expertise of three different experts to provide a comprehensive answer:
        1. The Movie Buff: This expert has an encyclopedic knowledge of movies and 
        can provide detailed information about movie plots, characters, and storylines.
        2. The Film Critic: This expert has a deep understanding of film analysis and c
        an provide insights into the themes, motifs, and narrative structures of movies.
        3. The Storyteller: This expert has a talent for summarizing and retelling stories 
        in a compelling way, capturing the essence of the plot while keeping it engaging and concise.
        When answering the user's query, you will first generate three separate answers from each of 
        the experts. After that, you will synthesize these three perspectives into a single, 
        comprehensive answer that combines the detailed knowledge of the Movie Buff, the analytical 
        insights of the Film Critic, and the engaging storytelling of the Storyteller. Always provide 
        a final answer that is concise
    """),
    capabilities=[Instrumentation()]
)

plot_search_agent_movie_buff = Agent(
    all_models[0],
    system_prompt=textwrap.dedent("""
        You are an expert who has an encyclopedic knowledge of movies and can 
        provide detailed information about movie plots, characters, and 
        storylines. Your task is to take a query and construct a 
        recommendation that is coherent and engaging plot summary of 
        movies matching the query. As the movie buff, you should explain 
        what makes people love the movie, and why it is a must-watch.
    """),
    capabilities=[Instrumentation()]
)

plot_search_agent_movie_buff.tool_plain(search_wikipedia)

plot_search_agent_film_critic = Agent(
    all_models[1],
    system_prompt=textwrap.dedent("""
        You are an expert who has has a deep understanding of film analysis 
        and can provide insights into the themes, motifs, and narrative 
        structures of movies. Your task is to take a query and construct a 
        recommendation that is coherent and engaging plot summary of 
        movies matching the query. As the film critic, you should explain 
        why the movie is critically acclaimed, and what makes it a must-watch.
    """),
    capabilities=[Instrumentation()]
)

plot_search_agent_film_critic.tool_plain(search_wikipedia)

plot_search_agent_storyteller = Agent(
    all_models[2],
    system_prompt=textwrap.dedent("""
        You are an expert who has has a talent for summarizing and retelling 
        stories in a compelling way, capturing the essence of the plot while 
        keeping it engaging and concise. Your task is to take a query and 
        construct a recommendation that is coherent and engaging plot summary of 
        movies matching the query. As the storyteller, you should explain why 
        this story is compelling, and what makes it a must-watch.
    """),
    capabilities=[Instrumentation()]
)

plot_search_agent_storyteller.tool_plain(search_wikipedia)

@plot_search_agent_manager.tool_plain
async def movie_buff_plot_search(query: str) -> str:
    """Use this tool to recommend movies based off their plots, using the expertise of a movie buff."""
    result = await plot_search_agent_movie_buff.run(query)
    print(f"Movie Buff Result: {result.all_messages()[-1].parts[0].content}")
    return result.all_messages()[-1].parts[0].content


@plot_search_agent_manager.tool_plain
async def film_critic_plot_search(query: str) -> str:
    """Use this tool to recommend movies based off their plots, using the expertise of a film critic."""
    result = await plot_search_agent_film_critic.run(query)
    print(f"Film Critic Result: {result.all_messages()[-1].parts[0].content}")
    return result.all_messages()[-1].parts[0].content


@plot_search_agent_manager.tool_plain
async def storyteller_plot_search(query: str) -> str:
    """Use this tool to recommend movies based off their plots, using the expertise of a storyteller."""
    result = await plot_search_agent_storyteller.run(query)
    print(f"Storyteller Result: {result.all_messages()[-1].parts[0].content}")
    return result.all_messages()[-1].parts[0].content


# plot_search_agent = plot_search_agent_prompt_based
plot_search_agent = plot_search_agent_manager