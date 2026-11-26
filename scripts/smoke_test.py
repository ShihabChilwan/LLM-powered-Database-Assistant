"""
Smoke test: does the configured model (LLM_MODEL) call tools reliably?

Asks 3 SQL questions through the real text-to-SQL loop and prints, for each:
the SQL the model ran, its answer, token usage, cost and latency. For
comparison it also prints a reference value computed directly with SQL.

This calls the real API (it needs OPENROUTER_API_KEY in .env) and costs a
fraction of a cent. It is not part of the automated tests.

Run from the project root with:  python scripts/smoke_test.py
Exit code 0 means every question used the tool and got an answer; 1 means not.
"""

import sqlite3
import sys
from pathlib import Path

# Running a file directly puts only its own folder (scripts/) on the import path.
# Add the project root so `import app...` works. sys.path is Python's classpath.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # must come after the sys.path change
from app.llm.agent import MAX_TURNS_MESSAGE, answer_question

# Each question comes with a reference query, so you can check the model's answer by eye.
QUESTIONS = [
    {
        "question": "How many orders are there in total?",
        "reference_sql": "SELECT COUNT(*) FROM orders",
    },
    {
        "question": "How many units of Alice Mutton were sold in July 2016?",
        "reference_sql": "SELECT units_sold FROM stock WHERE product = 'Alice Mutton' AND month = '2016-07'",
    },
    {
        "question": "What was the total price of order 10248, and which shipper delivered it?",
        "reference_sql": "SELECT total_price, shipper_name FROM orders WHERE order_id = '10248'",
    },
]


def reference_value(sql: str) -> list:
    """Runs a reference query directly (no model) and returns its rows as tuples."""
    connection = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def print_model_calls(model_calls: list[dict]) -> None:
    """Prints one line per model call: tokens, cost, latency and error if any."""
    for call in model_calls:
        tokens = call["tokens"]
        line = (
            f"  model call {call['turn']}: {tokens['prompt']} in + {tokens['completion']} out"
            f" = {tokens['total']} tokens, cost={call['cost']}, {call['latency_s']}s"
        )
        if call["error"]:
            line += f", ERROR: {call['error']}"
        print(line)


def run_one(item: dict) -> bool:
    """Asks one question and prints the trace.

    Input: one entry of QUESTIONS.
    Output: True if the model called the SQL tool at least once and gave a final answer.
    """
    print("=" * 70)
    print(f"Q: {item['question']}")
    result = answer_question(item["question"])

    sql_steps = []
    for step in result.steps:
        if step["error"]:
            outcome = f"ERROR: {step['error']}"
        elif step["tool"] == "run_sql_query":
            outcome = f"{step['result']['total_rows']} row(s)"
        else:
            outcome = f"{len(step['result']['passages'])} passage(s)"
        print(
            f"  tool call (turn {step['turn']}): {step['tool']}: {step['input']}  -> {outcome}"
        )
        if step["tool"] == "run_sql_query":
            sql_steps.append(step)
    print_model_calls(result.model_calls)
    print(f"A: {result.answer}")
    print(f"Reference: {reference_value(item['reference_sql'])}")

    model_failed = any(call["error"] for call in result.model_calls)
    used_tool = len(sql_steps) > 0
    finished = result.answer != MAX_TURNS_MESSAGE and not model_failed
    if model_failed:
        print("  !! A model call failed (see ERROR above), so tool use was not tested.")
    elif not used_tool:
        print("  !! The model answered WITHOUT calling the SQL tool.")
    return used_tool and finished


def main() -> int:
    """Runs all questions and prints a summary. Returns the process exit code."""
    print(f"Model: {config.LLM_MODEL} via {config.LLM_BASE_URL}")
    passed = 0
    for item in QUESTIONS:
        if run_one(item):
            passed += 1
    print("=" * 70)
    print(
        f"{passed}/{len(QUESTIONS)} questions used the SQL tool and produced an answer."
    )
    return 0 if passed == len(QUESTIONS) else 1


# This block runs only when the file is executed, not when it is imported
# (similar to Java's public static void main).
if __name__ == "__main__":
    sys.exit(main())
