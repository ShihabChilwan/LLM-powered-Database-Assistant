"""
Database QA and Report Assistant - Streamlit UI

Two features, wired to your existing modules:
  1. Ask a question -> answered by the agent (app/llm/agent.py), which uses SQL
     for data, the policy documents for rules, or both. Every tool call is shown:
     SQL with its rows, document searches with the passages they retrieved.
  2. Generate an Order Report -> reuses dbjson.py + order_report.py to
     produce a downloadable .xlsx for a given order_id.

Run from the project root with:  streamlit run app/streamlit_app.py
Uses app/database/processed.db, built by app/database/db.py.
"""

import io
import sqlite3
import sys
from pathlib import Path


import streamlit as st

# `streamlit run app/streamlit_app.py` only puts app/ on the import path, which is
# what lets `reporting...` below resolve. The newer modules import each other as
# `app.<module>` (so they also work with `python -m app...` and from scripts/), which
# needs the project root on the path too. sys.path is Python's classpath.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.llm.agent import TOOL_LABELS, AgentResult, answer_question
from reporting.dbjson import get_order_report_data
from reporting.order_report import generate_order_report

DB_PATH = config.DB_PATH

st.set_page_config(page_title=" Database QA and Report Assistant", layout="wide")


# ---------- Shared resources ----------


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    """Opens one shared, read-only connection to the database.

    The UI only reads, so mode=ro makes any accidental write fail instead of
    changing the data. @st.cache_resource makes Streamlit create it once and
    reuse it on every rerun (a decorator wraps the function, like an annotation
    that adds behaviour). check_same_thread=False is needed because Streamlit
    may call it from different threads.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_stats(conn):
    stats = {}
    for table in ("customers", "orders", "order_items", "stock"):
        try:
            stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            stats[table] = "—"
    return stats


# ---------- Rendering the agent's answer and trace ----------


def render_answer(result: AgentResult) -> None:
    """Shows which tools were used, the answer, the citation check and the totals."""
    st.markdown(f"**Tools used:** {result.tools_label()}")
    st.markdown(result.answer)

    check = result.citation_check
    if not check.all_verified():
        lines = []
        for citation in check.unverified:
            lines.append(f"- `{citation.text()}`")
        st.warning(
            "These citations do not match any passage retrieved for this answer, "
            "so treat the statements next to them with care:\n" + "\n".join(lines)
        )
    elif check.verified:
        st.caption(f"All {len(check.verified)} citation(s) match retrieved passages.")

    st.caption(
        f"{result.total_tokens} tokens · {len(result.model_calls)} model call(s) · "
        f"{len(result.steps)} tool call(s) · {result.total_latency_s:.1f}s"
    )


def render_step_header(step: dict) -> None:
    """Shows one step's turn, tool, model-call tokens and tool latency."""
    label = TOOL_LABELS.get(step["tool"], step["tool"])
    st.markdown(
        f"**Turn {step['turn']} · {label}** · {step['tokens']['total']} tokens "
        f"(model call that asked for it) · tool ran in {step['latency_s']:.2f}s"
    )


def render_sql_step(step: dict) -> None:
    """Shows the SQL the model ran and the rows it got back."""
    st.code(step["input"], language="sql")
    result = step["result"]
    if result["truncated"]:
        st.warning(
            f"Showing the first {len(result['rows'])} of {result['total_rows']} rows "
            "(the model saw the same)."
        )
    st.dataframe(result["rows"], width="stretch")


def render_document_step(step: dict) -> None:
    """Shows the search text and every passage retrieved, with source, section and score."""
    st.markdown(f"Search: *{step['input']}*")
    passages = step["result"]["passages"]
    if not passages:
        st.info(
            "No passage scored above the minimum, so the documents do not cover this."
        )
    for passage in passages:
        # A bordered container groups each passage (expanders cannot be nested).
        with st.container(border=True):
            st.markdown(
                f"`{passage['source']}` › {passage['section']} · score **{passage['score']:.3f}**"
            )
            st.text(passage["text"])


def render_trace(result: AgentResult) -> None:
    """Shows every tool call in order, then one row per model call."""
    with st.expander("Show reasoning trace (tool calls, results, tokens)"):
        for step in result.steps:
            render_step_header(step)
            if step["error"]:
                st.code(step["input"] or "(no input)")
                st.error(step["error"])
            elif step["tool"] == "run_sql_query":
                render_sql_step(step)
            else:
                render_document_step(step)
        st.markdown("**Model calls**")
        st.dataframe(model_call_rows(result.model_calls), width="stretch")


def model_call_rows(model_calls: list[dict]) -> list[dict]:
    """Flattens the model-call stats into table rows (one per call)."""
    rows = []
    for call in model_calls:
        rows.append(
            {
                "turn": call["turn"],
                "prompt tokens": call["tokens"]["prompt"],
                "completion tokens": call["tokens"]["completion"],
                "total tokens": call["tokens"]["total"],
                "latency (s)": call["latency_s"],
                "error": call["error"] or "",
            }
        )
    return rows


# ---------- UI ----------

st.title("📄Database QA and Report Assistant")
st.caption(
    "Ask questions over the ingested order/invoice/stock data, or generate a consolidated Order Report."
)

conn = get_connection()

with st.sidebar:
    st.subheader("Database overview")
    stats = db_stats(conn)
    st.metric("Customers", stats["customers"])
    st.metric("Orders", stats["orders"])
    st.metric("Order line items", stats["order_items"])
    st.metric("Stock snapshot rows", stats["stock"])
    st.caption(f"Source: `{DB_PATH}`")

tab_ask, tab_report = st.tabs(["💬 Ask a question", "📄 Generate Order Report"])

# --- Tab 1: questions answered by the agent (SQL, documents, or both) ---
with tab_ask:
    st.write(
        "Examples: "
        '*"How many orders did customer VINET place?"* (SQL), '
        '*"How long do I have to report damaged cheese?"* (documents), '
        '*"Order 10248 arrived damaged. Is the customer eligible for a refund?"* (both)'
    )

    question = st.text_input("Your question", key="question_input")
    ask_clicked = st.button("Ask", type="primary", disabled=not question)

    if ask_clicked:
        with st.spinner("Thinking..."):
            agent_result = answer_question(question)
        render_answer(agent_result)
        if agent_result.steps or agent_result.model_calls:
            render_trace(agent_result)

# --- Tab 2: Order Report generation ---
with tab_report:
    st.write(
        "Pulls the header, line items, and computed total for one order across "
        "invoice, purchase order, and shipping order data, and builds a downloadable Excel report."
    )

    order_id = st.text_input("Order ID", value="10248", key="order_id_input")
    generate_clicked = st.button(
        "Generate report", type="primary", disabled=not order_id
    )

    if generate_clicked:
        try:
            report = get_order_report_data(conn, order_id.strip())
            if report is None:
                st.error(f"No order found with ID '{order_id}'.")
            else:
                buffer = io.BytesIO()
                generate_order_report(report, buffer)
                buffer.seek(0)

                st.success(f"Report generated for order {order_id}.")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Header**")
                    st.json({k: v for k, v in report.items() if k != "line_items"})
                with col2:
                    st.markdown("**Line items**")
                    st.dataframe(report["line_items"], width="stretch")

                st.download_button(
                    label="⬇️ Download Order_Report.xlsx",
                    data=buffer,
                    file_name=f"order_{order_id}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Failed to generate report: {e}")
