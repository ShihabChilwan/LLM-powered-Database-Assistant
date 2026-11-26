"""
Helpers that build fake model replies, so tests can script what "the model" says.
"""

from openai.types.chat import ChatCompletionMessage

from app.llm.client import CallStats, ChatResult


def tool_call_message(calls: list[tuple[str, str]]) -> ChatCompletionMessage:
    """Builds an assistant reply that asks for tools.

    Input: (tool name, JSON arguments) pairs, e.g. [("run_sql_query", '{"sql": "SELECT 1"}')].
    """
    tool_calls = []
    for index, (name, arguments) in enumerate(calls):
        tool_calls.append(
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": None, "tool_calls": tool_calls}
    )


def text_message(text: str) -> ChatCompletionMessage:
    """Builds an assistant reply with a final answer and no tool calls."""
    return ChatCompletionMessage(role="assistant", content=text)


def chat_result(message: ChatCompletionMessage, total_tokens: int = 15) -> ChatResult:
    """Wraps a fake reply the way client.chat() does, with made-up token counts."""
    return ChatResult(
        message=message,
        stats=CallStats(
            prompt_tokens=10,
            completion_tokens=total_tokens - 10,
            total_tokens=total_tokens,
        ),
    )
