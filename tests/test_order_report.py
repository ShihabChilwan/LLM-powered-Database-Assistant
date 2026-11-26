"""Tests for the order report data (app/reporting/dbjson.py)."""

import sqlite3
from pathlib import Path

from app.reporting.dbjson import get_order_report_data


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Opens the test database the way the Streamlit app does: read-only, rows as sqlite3.Row."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def test_missing_order_returns_none_instead_of_crashing(test_db: Path) -> None:
    connection = open_readonly(test_db)
    try:
        assert get_order_report_data(connection, "99999") is None
    finally:
        connection.close()


def test_existing_order_has_header_lines_and_total(test_db: Path) -> None:
    connection = open_readonly(test_db)
    try:
        report = get_order_report_data(connection, "10248")
    finally:
        connection.close()

    assert report["company_name"] == "Vins et alcools Chevalier"
    assert report["shipper_name"] == "Federal Shipping"
    assert len(report["line_items"]) == 1
    assert report["order_total"] == 168.0
