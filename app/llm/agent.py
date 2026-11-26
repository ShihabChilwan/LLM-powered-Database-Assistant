"""
The agent: one bounded loop in which the model chooses between two tools,
run_sql_query (database) and search_documents (policy documents).

How the loop works:
  1. Send the system instructions, the question and both tool definitions.
  2. If the reply contains tool_calls, run each one, send the results back as
     "tool" messages, and ask again.
  3. If the reply has no tool_calls, it is the final answer.
  4. Stop after MAX_TURNS model calls in any case.

All model calls go through app.llm.client.chat(); the tools live in
app/llm/tools.py and the instructions in app/llm/prompts.py.

Run from the project root with:  python -m app.llm.agent "your question"
"""

import json
import sys
import time
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessage

from app import config
from app.llm import client, prompts, tools
from app.rag.citations import CitationCheck, check_citations

MAX_TURNS_MESSAGE = "Reached the maximum number of turns without a final answer."

# Short labels for the UI and reports, e.g. "SQL + Documents".
TOOL_LABELS = {"run_sql_query": "SQL", "search_documents": "Documents"}


@dataclass
class AgentResult:
    """What answer_question() returns.

    steps: one dict per tool call:
        {"turn", "tool", "input", "result", "error", "tokens", "latency_s"}
      - tokens: usage of the model call that asked for this tool. Tool calls
        made in the same turn share it, so do not add up step tokens; use
        total_tokens (or model_calls) instead.
      - latency_s: how long the tool itself took to run.
      - tool_call_id (extra): links the step to the model's tool call.
    model_calls: one stats dict per model call, including the final answer and
        any failed call.
    citation_check: the answer's citations, split into verified (retrieved in
        this run) and unverified (see app/rag/citations.py).
    """

    answer: str
    steps: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_s: float = 0.0
    model_calls: list[dict] = field(default_factory=list)
    citation_check: CitationCheck = field(default_factory=CitationCheck)

    def tools_label(self) -> str:
        """Returns the tools used as a short label, e.g. "SQL + Documents", or "None"."""
        labels = []
        for name in self.tools_used:
            labels.append(TOOL_LABELS.get(name, name))
        return " + ".join(labels) if labels else "None"


def assistant_message_to_dict(message: ChatCompletionMessage) -> dict:
    """Converts the SDK's reply object into a plain message dict.

    Output: {"role": "assistant", "content": ..., "tool_calls": [...]}, which can be
            appended to the conversation and sent back on the next call.
    """
    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": tool_calls,
    }


def tool_result_message(
    tool_call_id: str, result: dict | None, error: str | None
) -> dict:
    """Builds the "tool" message that returns one tool's result (or error) to the model."""
    payload = result if error is None else {"error": error}
    # default=str turns values json cannot encode (for example bytes) into text.
    content = json.dumps(payload, default=str, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def run_tool_calls(
    turn: int, message: ChatCompletionMessage, call_stats: dict
) -> list[dict]:
    """Runs every tool call in one model reply.

    Input: the turn number, the reply, and the stats of the model call that made it.
    Output: one step dict per tool call (see AgentResult).
    """
    steps = []
    for tool_call in message.tool_calls:
        start = time.perf_counter()
        tool_input, result, error = tools.execute_tool(
            tool_call.function.name, tool_call.function.arguments
        )
        step = {
            "turn": turn,
            "tool": tool_call.function.name,
            "input": tool_input,
            "result": result,
            "error": error,
            "tokens": call_stats["tokens"],
            "latency_s": round(time.perf_counter() - start, 3),
            "tool_call_id": tool_call.id,  # links the step to its "tool" message
        }
        steps.append(step)
    return steps


def finish(
    answer: str, steps: list[dict], model_calls: list[dict], start: float
) -> AgentResult:
    """Builds the AgentResult, adding up tokens and listing the tools used in order."""
    total_tokens = 0
    for call in model_calls:
        total_tokens += call["tokens"]["total"]
    tools_used: list[str] = []
    for step in steps:
        # Only real tools count; a made-up tool name is an error step, not a tool used.
        if step["tool"] in tools.TOOL_FUNCTIONS and step["tool"] not in tools_used:
            tools_used.append(step["tool"])
    return AgentResult(
        answer=answer,
        steps=steps,
        tools_used=tools_used,
        total_tokens=total_tokens,
        total_latency_s=round(time.perf_counter() - start, 3),
        model_calls=model_calls,
        citation_check=check_citations(answer, steps),
    )


def answer_question(question: str, max_turns: int = config.MAX_TURNS) -> AgentResult:
    """Answers a question with the two-tool loop.

    Input: the question; max_turns limits the number of model calls.
    Output: an AgentResult with the answer, the steps, the tools used, and totals.
    """
    start = time.perf_counter()
    messages = prompts.build_messages(question)
    steps: list[dict] = []
    model_calls: list[dict] = []

    for turn in range(1, max_turns + 1):
        reply = client.chat(messages, tools=tools.TOOLS)
        call_stats = reply.stats.to_dict()
        # {"turn": turn, **call_stats} builds a new dict: "turn" plus every key of call_stats.
        model_calls.append({"turn": turn, **call_stats})

        if reply.message is None:
            answer = f"The model call failed: {reply.stats.error}"
            return finish(answer, steps, model_calls, start)
        if not reply.message.tool_calls:
            # No tool requested: this reply is the final answer.
            return finish(reply.message.content or "", steps, model_calls, start)

        messages.append(assistant_message_to_dict(reply.message))
        for step in run_tool_calls(turn, reply.message, call_stats):
            steps.append(step)
            messages.append(
                tool_result_message(step["tool_call_id"], step["result"], step["error"])
            )

    return finish(MAX_TURNS_MESSAGE, steps, model_calls, start)


def print_result(result: AgentResult) -> None:
    """Prints the tool calls, the answer and the totals of one run."""
    for step in result.steps:
        outcome = step["error"] or summarize_result(step["tool"], step["result"])
        print(f"  turn {step['turn']}: {step['tool']}({step['input']!r}) -> {outcome}")
    print(f"Tools used: {result.tools_label()}")
    print(f"Answer:\n{result.answer}")
    for citation in result.citation_check.verified:
        print(f"  verified citation:   {citation.text()}")
    for citation in result.citation_check.unverified:
        print(f"  UNVERIFIED citation: {citation.text()}")
    print(
        f"Totals: {result.total_tokens} tokens, {len(result.model_calls)} model calls, "
        f"{result.total_latency_s}s"
    )


def summarize_result(tool: str, result: dict) -> str:
    """Returns a one-line summary of a tool result, for printing."""
    if tool == "search_documents":
        ids = []
        for passage in result["passages"]:
            ids.append(f"{passage['chunk_id']} ({passage['score']})")
        return ", ".join(ids) if ids else "no passages"
    suffix = ", truncated" if result["truncated"] else ""
    return f"{result['total_rows']} row(s){suffix}"


if __name__ == "__main__":
    # sys.argv is the command line, like main(String[] args); [1:] skips the script name.
    question_text = (
        " ".join(sys.argv[1:]) or "How many orders did customer VINET place?"
    )
    print(f"Question: {question_text}")
    print_result(answer_question(question_text))
