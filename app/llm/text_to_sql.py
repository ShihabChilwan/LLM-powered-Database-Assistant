import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
from google.genai import types

load_dotenv()

client = genai.Client(http_options=HttpOptions(api_version="v1"))

DB_PATH = Path(__file__).resolve().parents[1] / "database" / "processed.db"

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

# ---------- Approach: Bounded exploratory tool-use ----------

def run_probe_query(sql: str) -> list:
    """Runs a read-only SELECT query against the database and returns the resulting rows.
    Use this to explore the actual data (e.g. check what months exist for a specific product)
    before committing to a final answer."""
    return run_sql_readonly(sql)

def answer_question_manual_loop(question, max_turns=10, verbose=True):
    """Runs the bounded text-to-SQL loop.

    Always returns (answer, steps) where steps is a list of dicts:
        {"turn": int, "sql": str, "result": list, "error": str | None}
    so any caller (this file's CLI, the Streamlit UI, evaluate.py) can
    inspect what was actually queried instead of re-implementing the loop.
    Set verbose=False to suppress console printing (e.g. from a UI).
    """
    steps = []
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=f"""
You are answering questions against this SQLite schema:

{SCHEMA}

You have a tool, run_probe_query, to run SELECT queries and inspect real data before
answering. Use it to check things like which months exist for a specific product before
doing any date arithmetic - don't guess.

Question: "{question}"

Give a final natural-language answer using only data you've actually queried.
""")])
    ]

    for turn in range(max_turns):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[run_probe_query],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        function_calls = response.function_calls

        if not function_calls:
            return response.text, steps   # model gave a final answer, no more tool calls requested

        contents.append(response.candidates[0].content)

        response_parts = []
        for call in function_calls:
            sql = call.args.get("sql", "")
            if verbose:
                print(f"Turn {turn + 1}: model called {call.name}({call.args})")

            try:
                result = run_probe_query(**call.args)
                error = None
            except Exception as e:
                result = []
                error = str(e)

            if verbose:
                print(f"  -> result: {result if error is None else error}")

            steps.append({"turn": turn + 1, "sql": sql, "result": result, "error": error})

            response_parts.append(types.Part.from_function_response(
                name=call.name,
                response={"result": result} if error is None else {"error": error},
            ))

        contents.append(types.Content(role="tool", parts=response_parts))

    return "Reached max turns without a final answer.", steps


def run_sql_readonly(sql):
    normalized = sql.strip().upper()
    dangerous = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "PRAGMA", "REPLACE", "TRUNCATE")
    if any(normalized.startswith(kw) for kw in dangerous):
        raise ValueError("Write operations are not allowed")
    
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()
    return rows

if __name__ == "__main__":
    # question = "How many units of 'Alice Mutton' were sold 3 months back and what is the stock of 'Chang' a month back?"
    # question = "give me the entire order history of 10248"
    question = "give me the entire order history of 'VINET' company"

    print("\n=== Agentic (manual loop, visible steps) ===")
    answer, steps = answer_question_manual_loop(question)
    print(answer)

