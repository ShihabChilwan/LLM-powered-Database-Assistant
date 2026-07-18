"""
Database QA and Report Assistant - Streamlit UI

Two features, wired to your existing modules:
  1. Ask a question -> routed through the text-to-SQL loop (dbcrawler.py),
     with every tool call (generated SQL + results) shown transparently.
  2. Generate an Order Report -> reuses dbjson.py + order_report.py to
     produce a downloadable .xlsx for a given order_id.

Run from the project root with:  streamlit run app/streamlit_app.py
Uses app/database/processed.db, built by app/database/db.py.
"""

import io
import sqlite3
import time
from pathlib import Path


import streamlit as st

from llm.text_to_sql import answer_question_manual_loop
from reporting.dbjson import get_order_report_data
from reporting.order_report import generate_order_report

DB_PATH = Path(__file__).resolve().parent / "database" / "processed.db"

st.set_page_config(page_title=" Database QA and Report Assistant", layout="wide")


# ---------- Shared resources ----------

@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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


# ---------- UI ----------

st.title("📄Database QA and Report Assistant")
st.caption("Ask questions over the ingested order/invoice/stock data, or generate a consolidated Order Report.")

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

# --- Tab 1: text-to-SQL ---
with tab_ask:
    st.write("Examples: *\"How many units of Alice Mutton were sold in July 2016?\"*, "
             "*\"Give me the entire order history of 10248\"*, "
             "*\"Which orders were shipped more than a week after ordering?\"*")

    question = st.text_input("Your question", key="question_input")
    ask_clicked = st.button("Ask", type="primary", disabled=not question)

    if ask_clicked:
        with st.spinner("Thinking..."):
            start = time.time()
            answer, steps = answer_question_manual_loop(question, verbose=False)
            elapsed = time.time() - start

        st.success(answer)
        st.caption(f"Answered in {elapsed:.1f}s using {len(steps)} SQL call(s).")

        if steps:
            with st.expander("Show reasoning trace (generated SQL + results)"):
                for step in steps:
                    st.markdown(f"**Turn {step['turn']}**")
                    st.code(step["sql"], language="sql")
                    if step["error"]:
                        st.error(step["error"])
                    else:
                        st.dataframe(step["result"], use_container_width=True)

# --- Tab 2: Order Report generation ---
with tab_report:
    st.write("Pulls the header, line items, and computed total for one order across "
             "invoice, purchase order, and shipping order data, and builds a downloadable Excel report.")

    order_id = st.text_input("Order ID", value="10248", key="order_id_input")
    generate_clicked = st.button("Generate report", type="primary", disabled=not order_id)

    if generate_clicked:
        try:
            report = get_order_report_data(conn, order_id)
            if report.get("order_id") is None:
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
                    st.dataframe(report["line_items"], use_container_width=True)

                st.download_button(
                    label="⬇️ Download Order_Report.xlsx",
                    data=buffer,
                    file_name=f"order_{order_id}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Failed to generate report: {e}")




