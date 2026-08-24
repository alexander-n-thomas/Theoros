import os

import dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.providers.openai import OpenAIProvider

dotenv.load_dotenv()

deepseekr18b = OpenAIChatModel(
    model_name="deepseek-r1:8b",
    provider=LiteLLMProvider(
        api_key="ollama",
        api_base="localhost:11434/v1"
    )
)

gpt54mini = OpenAIChatModel(
    "gpt-5.4-mini",
    provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
)


qwen3_8b = OpenAIChatModel(
    model_name="qwen3:8b",
    provider=LiteLLMProvider(
        api_key="ollama",
        api_base="localhost:11434/v1"
    )
)


all_models = [deepseekr18b, gpt54mini, qwen3_8b]
model = gpt54mini

__all__ = ["model", "all_models", "deepseekr18b", "gpt54mini", "qwen3_8b"]