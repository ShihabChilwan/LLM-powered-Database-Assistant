"""Tests for the read-only SQL tool (app/llm/tools.py)."""

import sqlite3
from pathlib import Path

import pytest

from app import config
from app.llm import tools


def count_stock_rows(db_path: Path) -> int:
    """Counts rows directly, bypassing the tool, to prove nothing was deleted."""
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0]
    finally:
        connection.close()


# parametrize runs the same test once per value, like a JUnit @ParameterizedTest.
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM stock",
        "DROP TABLE stock",
        "  insert into stock VALUES ('x', 'y', 'z', 1, 1, 1)",
        "UPDATE stock SET units_sold = 0",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA journal_mode = DELETE",
    ],
)
def test_keyword_check_rejects_writes(sql: str) -> None:
    with pytest.raises(ValueError, match="Write operations are not allowed"):
        tools.reject_obvious_writes(sql)


def test_write_that_passes_keyword_check_still_fails_on_readonly_connection(
    test_db: Path,
) -> None:
    sql = "WITH x AS (SELECT 1) DELETE FROM stock"
    tools.reject_obvious_writes(sql)  # the first-word filter lets this through...

    # ...but mode=ro makes SQLite itself refuse the write.
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        tools.run_sql_query(sql)
    assert count_stock_rows(test_db) == 250


def test_result_is_capped_at_max_sql_rows(test_db: Path) -> None:
    result = tools.run_sql_query("SELECT * FROM stock")

    assert len(result["rows"]) == config.MAX_SQL_ROWS
    assert result["truncated"] is True
    assert result["total_rows"] == 250


def test_small_result_is_not_truncated(test_db: Path) -> None:
    result = tools.run_sql_query("SELECT COUNT(*) AS n FROM stock")

    assert result == {"rows": [{"n": 250}], "truncated": False, "total_rows": 1}


def test_execute_tool_returns_errors_instead_of_raising(test_db: Path) -> None:
    _input, result, error = tools.execute_tool(
        "run_sql_query", '{"sql": "SELECT * FROM missing"}'
    )
    assert result is None
    assert "no such table" in error

    _input, result, error = tools.execute_tool("made_up_tool", "{}")
    assert result is None
    assert "Unknown tool" in error
