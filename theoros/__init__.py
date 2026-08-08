import os

import dotenv

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

dotenv.load_dotenv()

deepseekr18b = OpenAIChatModel(
    model_name="deepseek-r1:8b",
    provider=LiteLLMProvider(
        api_key="ollama",
        api_base="http://192.168.4.52:11434/v1"
    )
)

gpt54mini = OpenAIChatModel(
    "gpt-5.4-mini",
    provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
)


llama4scout = OpenAIChatModel(
    "meta-llama/llama-4-scout",
    provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])
)


all_models = [deepseekr18b, gpt54mini, llama4scout]
model = gpt54mini

__all__ = ["model", "all_models", "deepseekr18b", "gpt54mini"]