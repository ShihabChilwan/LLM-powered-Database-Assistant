"""
The only place in the app that talks to the model provider.

Every other module calls chat() or embed() from here and never uses the
openai SDK directly. This makes the model a configuration choice: switching
to another model, or another OpenAI-compatible provider, means changing
.env, not code.

OpenRouter's API is OpenAI-compatible, so we use the official openai
package and only point it at a different base URL.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from app import config


@dataclass
class CallStats:
    """Token usage, cost, latency and error of one model call.

    A dataclass is roughly a Java record: Python generates __init__,
    __repr__ and __eq__ from the annotated fields below.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None  # in OpenRouter credits; None if not returned
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        """Returns the stats as a plain dict, ready to attach to a step or print as JSON."""
        return {
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens,
            },
            "cost": self.cost,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }


@dataclass
class ChatResult:
    """What chat() returns: the assistant message (None if the call failed) and its stats."""

    message: ChatCompletionMessage | None  # None when the call failed
    stats: CallStats = field(default_factory=CallStats)


# Created on first use, not at import time, so modules (and tests) can import
# this file without an API key being set.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Returns the shared OpenAI-compatible client, creating it on the first call.

    Raises RuntimeError if OPENROUTER_API_KEY is missing.
    """
    # 'global' lets this function assign to the module-level variable;
    # without it, Python would create a new local variable instead.
    global _client
    if _client is None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(
            api_key=config.OPENROUTER_API_KEY, base_url=config.LLM_BASE_URL
        )
    return _client


def _stats_from_usage(usage: Any, latency_s: float) -> CallStats:
    """Builds CallStats from a response's 'usage' object.

    Input: the usage object (may be None) and the measured latency.
    Output: a CallStats with tokens and, if OpenRouter sent it, cost.
    """
    stats = CallStats(latency_s=latency_s)
    if usage is None:
        return stats
    # getattr(obj, name, default) reads an attribute that might not exist.
    # 'cost' and 'completion_tokens' are not in every response type
    # (for example, embedding responses have no completion tokens).
    stats.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    stats.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    stats.total_tokens = getattr(usage, "total_tokens", 0) or 0
    stats.cost = getattr(usage, "cost", None)
    return stats


def chat(messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
    """Sends a conversation to LLM_MODEL and returns the assistant's reply.

    Input: messages in OpenAI chat format ({"role": ..., "content": ...}),
           and optional tool definitions in OpenAI 'tools' format.
    Output: ChatResult with the reply message and its CallStats.
            Errors are not raised: they are recorded in stats.error and
            message is None, so the caller can log them as a step.
    """
    request: dict = {"model": config.LLM_MODEL, "messages": messages}
    if tools:
        request["tools"] = tools

    start = time.perf_counter()
    try:
        # **request unpacks the dict into keyword arguments: model=..., messages=..., tools=...
        response = _get_client().chat.completions.create(**request)
    # Deliberately broad: any failure (network, auth, rate limit, bad request) becomes
    # a recorded error instead of crashing the loop.
    except Exception as error:  # noqa: BLE001
        latency_s = time.perf_counter() - start
        return ChatResult(
            message=None, stats=CallStats(latency_s=latency_s, error=str(error))
        )
    latency_s = time.perf_counter() - start

    stats = _stats_from_usage(response.usage, latency_s)
    return ChatResult(message=response.choices[0].message, stats=stats)


def embed(texts: list[str]) -> tuple[list[list[float]], CallStats]:
    """Turns each text into a vector using EMBEDDING_MODEL.

    Input: a list of texts.
    Output: (vectors, stats). vectors[i] is the embedding of texts[i].
    Unlike chat(), errors are raised: without vectors there is nothing
    sensible to return.
    """
    start = time.perf_counter()
    response = _get_client().embeddings.create(
        model=config.EMBEDDING_MODEL, input=texts
    )
    latency_s = time.perf_counter() - start

    # Sort by index so the output order is guaranteed to match the input order.
    items = sorted(response.data, key=lambda item: item.index)
    vectors = []
    for item in items:
        vectors.append(item.embedding)
    return vectors, _stats_from_usage(response.usage, latency_s)
