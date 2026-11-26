"""
Text-to-SQL: the model answers a question by running read-only SQL queries
in a bounded loop.

How the loop works:
  1. Send the question, the schema and the tool definition to the model.
  2. If the reply contains tool_calls, run each SQL query, send the results
     back as "tool" messages, and ask again.
  3. If the reply has no tool_calls, it is the final answer.
  4. Stop after MAX_TURNS model calls in any case.

All model calls go through app.llm.client.chat(), never the SDK directly.
"""

import json
import sqlite3

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)

from app import config
from app.llm import client

SCHEMA = """
customers(customer_id TEXT PRIMARY KEY, company_name TEXT, contact_name TEXT, address TEXT, city TEXT, postal_code TEXT, country TEXT, phone TEXT, fax TEXT, ship_region TEXT)
orders(order_id TEXT PRIMARY KEY, order_date TEXT, customer_id TEXT REFERENCES customers, shipped_date TEXT, employee_name TEXT, shipper_name TEXT, ship_address TEXT, ship_city TEXT, ship_country TEXT, total_price REAL)
order_items(order_id TEXT REFERENCES orders, product_id TEXT, product_name TEXT, quantity INTEGER, unit_price REAL, line_total REAL)
stock(category TEXT, product TEXT, month TEXT, units_sold INTEGER, units_in_stock INTEGER, unit_price REAL)

Important notes about the stock table:
- It has ONE ROW PER PRODUCT PER MONTH (a monthly snapshot), not one row per product overall.
- units_in_stock and unit_price are POINT-IN-TIME values for that month, like a bank balance - NEVER sum them across months, that produces a meaningless number.
- units_sold is a per-month flow value - summing it across months IS meaningful (e.g. total units sold over a period).
- For questions about "current" or "latest" stock, filter to the most recent month for that product, for example:
  WHERE month = (SELECT MAX(month) FROM stock WHERE product = 'X')
"""

# Tool definition in the OpenAI "tools" format: a name, a description the
# model reads to decide when to call it, and a JSON Schema for its arguments.
SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_probe_query",
        "description": (
            "Runs a read-only SELECT query against the database and returns the resulting rows. "
            "Use this to explore the actual data (e.g. check what months exist for a specific "
            "product) before committing to a final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single SQLite SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    },
}

MAX_TURNS_MESSAGE = "Reached max turns without a final answer."


def build_initial_messages(question: str) -> list[dict]:
    """Builds the first message of the conversation.

    Input: the user's question.
    Output: a one-element message list with the schema, tool hint and question.
    """
    prompt = f"""
You are answering questions against this SQLite schema:

{SCHEMA}

You have a tool, run_probe_query, to run SELECT queries and inspect real data before
answering. Use it to check things like which months exist for a specific product before
doing any date arithmetic - don't guess.

Question: "{question}"

Give a final natural-language answer using only data you've actually queried.
"""
    return [{"role": "user", "content": prompt}]


def assistant_message_to_dict(message: ChatCompletionMessage) -> dict:
    """Converts the SDK's reply object into a plain message dict.

    Input: an openai ChatCompletionMessage.
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


def execute_tool_call(
    tool_call: ChatCompletionMessageFunctionToolCall,
) -> tuple[str, list, str | None]:
    """Runs one tool call requested by the model.

    Input: one entry of message.tool_calls.
    Output: (sql, rows, error). error is None on success; on failure rows is []
            and error holds the message that will be shown to the model.
    """
    sql = ""
    try:
        if tool_call.function.name != SQL_TOOL["function"]["name"]:
            raise ValueError(f"Unknown tool: {tool_call.function.name}")
        # Arguments arrive as a JSON string, e.g. '{"sql": "SELECT ..."}'.
        arguments = json.loads(tool_call.function.arguments or "{}")
        sql = arguments.get("sql", "")
        return sql, run_sql_readonly(sql), None
    # Deliberately broad: any failure (bad JSON, bad SQL, blocked write) is sent back
    # to the model as an error so it can correct its query on the next turn.
    except Exception as error:  # noqa: BLE001
        return sql, [], str(error)


def tool_result_message(tool_call_id: str, rows: list, error: str | None) -> dict:
    """Builds the "tool" message that returns a result to the model.

    Input: the id of the tool call being answered, its rows, and its error (or None).
    Output: a message dict whose content is JSON text.
    """
    payload = {"result": rows} if error is None else {"error": error}
    # default=str turns values json cannot encode (for example bytes) into text.
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, default=str),
    }


def answer_question_manual_loop(
    question: str, max_turns: int = config.MAX_TURNS, verbose: bool = True
) -> tuple[str, list[dict], list[dict]]:
    """Runs the bounded text-to-SQL loop.

    Input: the question; max_turns limits the number of model calls;
           verbose=False suppresses console printing (e.g. from a UI).
    Output: (answer, steps, model_calls)
      - steps: one dict per SQL tool call:
          {"turn", "sql", "result", "error", "tokens", "cost", "latency_s"}
        tokens/cost/latency_s belong to the model call that asked for this query,
        so steps from the same turn share them (do not add them up).
      - model_calls: one stats dict per model call, including the final answer
        and failed calls. Use these for totals.
    """
    messages = build_initial_messages(question)
    steps: list[dict] = []
    model_calls: list[dict] = []

    for turn in range(1, max_turns + 1):
        reply = client.chat(messages, tools=[SQL_TOOL])
        call_stats = reply.stats.to_dict()
        # {"turn": turn, **call_stats} builds a new dict: "turn" plus every key of call_stats.
        model_calls.append({"turn": turn, **call_stats})

        if reply.message is None:
            return f"The model call failed: {reply.stats.error}", steps, model_calls
        if not reply.message.tool_calls:
            # No tool requested: this reply is the final answer.
            return reply.message.content or "", steps, model_calls

        messages.append(assistant_message_to_dict(reply.message))
        for tool_call in reply.message.tool_calls:
            sql, rows, error = execute_tool_call(tool_call)
            if verbose:
                # {sql!r} prints the value with quotes, like repr(), so empty SQL is visible.
                print(f"Turn {turn}: model called {tool_call.function.name}({sql!r})")
                print(f"  -> result: {rows if error is None else error}")
            step = {
                "turn": turn,
                "sql": sql,
                "result": rows,
                "error": error,
                "tokens": call_stats["tokens"],
                "cost": call_stats["cost"],
                "latency_s": call_stats["latency_s"],
            }
            steps.append(step)
            messages.append(tool_result_message(tool_call.id, rows, error))

    return MAX_TURNS_MESSAGE, steps, model_calls


def run_sql_readonly(sql: str) -> list[dict]:
    """Runs a SELECT query on a read-only connection and returns the rows as dicts.

    Input: one SQL statement.
    Output: a list of {column: value} dicts.
    Raises ValueError for statements that start with a write keyword.
    (Unchanged in Part A0; Part A tightens this guard and caps the rows.)
    """
    normalized = sql.strip().upper()
    dangerous = (
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "ATTACH",
        "PRAGMA",
        "REPLACE",
        "TRUNCATE",
    )
    if any(normalized.startswith(keyword) for keyword in dangerous):
        raise ValueError("Write operations are not allowed")

    connection = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()
    return rows


# Run from the project root with:  python -m app.llm.text_to_sql
if __name__ == "__main__":
    # question = "How many units of 'Alice Mutton' were sold 3 months back and what is the stock of 'Chang' a month back?"
    # question = "give me the entire order history of 10248"
    question = "give me the entire order history of 'VINET' company"

    print("\n=== Agentic (manual loop, visible steps) ===")
    answer, _steps, _model_calls = answer_question_manual_loop(question)
    print(answer)
