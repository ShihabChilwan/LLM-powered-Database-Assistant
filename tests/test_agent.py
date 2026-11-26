"""Tests for the agent loop (app/llm/agent.py) with a scripted fake model."""

import pytest

from app import config
from app.llm import agent, client, tools
from app.llm.client import CallStats, ChatResult
from app.rag.retriever import RetrievedChunk
from tests.fakes import chat_result, text_message, tool_call_message


class ScriptedModel:
    """A fake client.chat that returns prepared replies in order and remembers each request."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    # The parameter must be called `tools`, because the agent passes it as tools=...
    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Same signature as client.chat; pops the next prepared reply."""
        self.requests.append(
            list(messages)
        )  # a copy: the agent keeps appending to its list
        return self.replies.pop(0)


def fake_search(query: str) -> list[RetrievedChunk]:
    """Stands in for retriever.search: always returns one passage."""
    return [
        RetrievedChunk(
            "returns_and_refunds.md#04",
            "returns_and_refunds.md",
            "3.1 Deadlines",
            "48 hours.",
            0.6,
        )
    ]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Returns a function that installs a ScriptedModel with the given replies."""

    def install(replies: list) -> ScriptedModel:
        model = ScriptedModel(replies)
        monkeypatch.setattr(client, "chat", model.chat)
        monkeypatch.setattr(tools.retriever, "search", fake_search)
        return model

    return install


def test_tool_calls_are_executed_and_final_answer_returned(test_db, scripted) -> None:
    model = scripted(
        [
            chat_result(
                tool_call_message(
                    [
                        (
                            "run_sql_query",
                            '{"sql": "SELECT COUNT(*) AS n FROM orders"}',
                        ),
                        ("search_documents", '{"query": "damage deadline"}'),
                    ]
                ),
                total_tokens=20,
            ),
            chat_result(
                text_message(
                    "1 order. [source: returns_and_refunds.md › 3.1 Deadlines]"
                ),
                30,
            ),
        ]
    )

    result = agent.answer_question("How many orders, and what is the deadline?")

    assert result.answer.startswith("1 order.")
    assert result.tools_used == ["run_sql_query", "search_documents"]
    assert result.tools_label() == "SQL + Documents"
    assert result.total_tokens == 50
    assert len(result.model_calls) == 2

    sql_step, search_step = result.steps
    assert sql_step["turn"] == 1 and sql_step["error"] is None
    assert sql_step["result"]["rows"] == [
        {"n": 1}
    ]  # really ran against the test database
    assert (
        search_step["result"]["passages"][0]["citation"]
        == "[source: returns_and_refunds.md › 3.1 Deadlines]"
    )

    # The second request sent the tool results back to the model.
    roles = []
    for message in model.requests[1]:
        roles.append(message["role"])
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assert result.citation_check.all_verified()


def test_tool_error_is_sent_back_and_loop_continues(test_db, scripted) -> None:
    scripted(
        [
            chat_result(
                tool_call_message([("run_sql_query", '{"sql": "DROP TABLE stock"}')])
            ),
            chat_result(text_message("I cannot change data.")),
        ]
    )

    result = agent.answer_question("Delete the stock table")

    assert result.steps[0]["error"] == "Write operations are not allowed"
    assert result.answer == "I cannot change data."


def test_loop_stops_at_max_turns(test_db, scripted) -> None:
    endless = []
    for _turn in range(config.MAX_TURNS + 5):
        endless.append(
            chat_result(tool_call_message([("run_sql_query", '{"sql": "SELECT 1"}')]))
        )
    model = scripted(endless)

    result = agent.answer_question("Keep going")

    assert result.answer == agent.MAX_TURNS_MESSAGE
    assert len(model.requests) == config.MAX_TURNS
    assert len(result.model_calls) == config.MAX_TURNS
    assert len(result.steps) == config.MAX_TURNS


def test_failed_model_call_is_reported(scripted) -> None:
    scripted([ChatResult(message=None, stats=CallStats(error="401 Unauthorized"))])

    result = agent.answer_question("Anything")

    assert result.answer == "The model call failed: 401 Unauthorized"
    assert result.model_calls[0]["error"] == "401 Unauthorized"
