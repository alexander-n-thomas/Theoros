"""
Fast API Server for Theoros (OpenAI Compatible)
Theoros: A movie recommendation system
"""
import time
from typing import Optional, List, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pydantic_ai import AgentRunResult

from theoros.guardrails import PromptScanner, OutputScanner
from theoros.theoros_main import theoros_agent

app = FastAPI(title="Theoros: A movie recommendation system")
prompt_scanner = PromptScanner()
output_scanner = OutputScanner()
chat_history = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"theoros-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    user_input = ""
    for message in request.messages:
        if message.role == "user":
            user_input += message.content + "\n"
    sanitized_prompt, results_valid, results_score = \
        prompt_scanner.scan_input(user_input)
    for scanner_name, scanner_result in results_valid.items():
        if not scanner_result:
            raise Exception(f"Input scanner fail: '{scanner_name}', {results_score[scanner_name]}")

    result: AgentRunResult[Any] | None = None
    if theoros_agent.model == request.model:
        result = theoros_agent.run_sync(sanitized_prompt, message_history=chat_history)
    else:
        with theoros_agent.override(model=request.model):
            result = theoros_agent.run_sync(sanitized_prompt, message_history=chat_history)

    if result is None:
        raise Exception("Agent run result is None")

    chat_history.extend(result.all_messages())

    sanitized_output, results_valid, results_score = output_scanner.scan_output(sanitized_prompt, result.output)
    for scanner_name, scanner_result in results_valid.items():
        if not scanner_result:
            raise Exception(f"Output scanner fail: '{scanner_name}', {results_score[scanner_name]}")

    response = ChatCompletionResponse(
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=sanitized_output),
            finish_reason="stop"
        )],
        usage=ChatCompletionUsage(
            completion_tokens=result.usage.output_tokens,
            prompt_tokens=result.usage.input_tokens,
            total_tokens=result.usage.total_tokens
        )
    )
    return response