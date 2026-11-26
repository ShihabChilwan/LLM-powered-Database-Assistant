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

# An f-string: {config.MAX_SQL_ROWS} below is replaced by its value.
SCHEMA = f"""
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

Important notes about the orders table:
- Some orders have NO shipper_name and NO shipped_date (NULL): they appear only in purchase
  orders, with no shipping order. NULL means "not recorded", not "not shipped" - say so
  instead of guessing, and remember that WHERE shipper_name = '...' skips these orders.

Query results:
- At most {config.MAX_SQL_ROWS} rows are returned. If the result says "truncated": true, only the first
  rows are included and "total_rows" gives the real count: narrow the query (WHERE, GROUP BY,
  COUNT, LIMIT) instead of drawing conclusions from a partial list.
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
) -> tuple[str, dict | None, str | None]:
    """Runs one tool call requested by the model.

    Input: one entry of message.tool_calls.
    Output: (sql, result, error). On success result is the dict from
            run_sql_readonly() and error is None; on failure result is None
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
        return sql, None, str(error)


def tool_result_message(
    tool_call_id: str, result: dict | None, error: str | None
) -> dict:
    """Builds the "tool" message that returns a result to the model.

    Input: the id of the tool call being answered, its result, and its error (or None).
    Output: a message dict whose content is JSON text, either
            {"rows": [...], "truncated": ..., "total_rows": ...} or {"error": "..."}.
    """
    payload = result if error is None else {"error": error}
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
        result is {"rows", "truncated", "total_rows"}, or None if error is set.
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
            sql, result, error = execute_tool_call(tool_call)
            if verbose:
                # {sql!r} prints the value with quotes, like repr(), so empty SQL is visible.
                print(f"Turn {turn}: model called {tool_call.function.name}({sql!r})")
                print(f"  -> result: {result if error is None else error}")
            step = {
                "turn": turn,
                "sql": sql,
                "result": result,
                "error": error,
                "tokens": call_stats["tokens"],
                "cost": call_stats["cost"],
                "latency_s": call_stats["latency_s"],
            }
            steps.append(step)
            messages.append(tool_result_message(tool_call.id, result, error))

    return MAX_TURNS_MESSAGE, steps, model_calls


# Statements that change data or the schema. Checked only as the first word.
WRITE_KEYWORDS = (
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


def reject_obvious_writes(sql: str) -> None:
    """Fast first filter: raises ValueError if the statement starts with a write keyword.

    This is NOT the real protection. It only looks at the first word, so for example
    "WITH x AS (SELECT 1) DELETE FROM stock" gets past it. The actual guarantee is the
    read-only connection (mode=ro) in run_sql_readonly(): SQLite itself refuses every
    write on it. This check just gives the model a clearer error message, sooner.
    """
    normalized = sql.strip().upper()
    for keyword in WRITE_KEYWORDS:
        if normalized.startswith(keyword):
            raise ValueError("Write operations are not allowed")


def fetch_capped(cursor: sqlite3.Cursor, max_rows: int) -> dict:
    """Reads at most max_rows rows from a cursor, and counts the rest without keeping them.

    Input: a cursor that has executed a query (rows are sqlite3.Row), and the cap.
    Output: {"rows": [...], "truncated": bool, "total_rows": int}.
    """
    rows = []
    for row in cursor.fetchmany(max_rows):
        rows.append(dict(row))
    # Count the remaining rows one by one, so a huge result never sits in memory.
    remaining = 0
    for _row in cursor:
        remaining += 1
    return {
        "rows": rows,
        "truncated": remaining > 0,
        "total_rows": len(rows) + remaining,
    }


def run_sql_readonly(sql: str, max_rows: int = config.MAX_SQL_ROWS) -> dict:
    """Runs one SQL statement on a read-only connection, returning at most max_rows rows.

    Input: one SQL statement, and the row cap.
    Output: {"rows": [{column: value}, ...], "truncated": bool, "total_rows": int}.
            "truncated" is True when rows were cut off, so the model knows to narrow
            its query; "total_rows" is the full count either way.
    Raises ValueError for an obvious write, sqlite3.Error for invalid SQL or any write.
    """
    reject_obvious_writes(sql)

    # mode=ro is the real write guard: SQLite opens the file read-only and refuses
    # every INSERT, UPDATE, DELETE, DROP ..., however the statement is phrased.
    connection = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return fetch_capped(connection.execute(sql), max_rows)
    finally:
        connection.close()


# Run from the project root with:  python -m app.llm.text_to_sql
if __name__ == "__main__":
    # question = "How many units of 'Alice Mutton' were sold 3 months back and what is the stock of 'Chang' a month back?"
    # question = "give me the entire order history of 10248"
    question = "give me the entire order history of 'VINET' company"

    print("\n=== Agentic (manual loop, visible steps) ===")
    answer, _steps, _model_calls = answer_question_manual_loop(question)
    print(answer)
