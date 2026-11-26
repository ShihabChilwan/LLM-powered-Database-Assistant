"""
Temporary adapter for the Streamlit UI.

The question-answering loop now lives in app/llm/agent.py (two tools: SQL and
documents). The UI still expects the older (answer, steps, model_calls) shape
with SQL steps only, so this module converts the agent's result to that shape.
It is removed when the UI switches to the agent directly (Part B6).
"""

from app import config
from app.llm.agent import answer_question


def answer_question_manual_loop(
    question: str, max_turns: int = config.MAX_TURNS, verbose: bool = False
) -> tuple[str, list[dict], list[dict]]:
    """Answers a question with the agent and returns the old result shape.

    Input: the question; max_turns limits the number of model calls;
           verbose is accepted for compatibility and ignored.
    Output: (answer, sql_steps, model_calls). sql_steps holds only the
            run_sql_query steps, as {"turn", "sql", "result", "error", "tokens",
            "latency_s"}; document searches are not shown until Part B6.
    """
    result = answer_question(question, max_turns=max_turns)
    sql_steps = []
    for step in result.steps:
        if step["tool"] != "run_sql_query":
            continue
        sql_steps.append(
            {
                "turn": step["turn"],
                "sql": step["input"],
                "result": step["result"],
                "error": step["error"],
                "tokens": step["tokens"],
                "latency_s": step["latency_s"],
            }
        )
    return result.answer, sql_steps, result.model_calls
