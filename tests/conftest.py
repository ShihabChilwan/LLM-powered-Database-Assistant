"""
Shared test fixtures. pytest loads this file automatically before the tests.

A fixture is a function that prepares something a test needs; a test asks for
it by naming it as a parameter (pytest passes it in, a bit like dependency
injection). `monkeypatch` is pytest's built-in fixture for temporarily
replacing attributes; every replacement is undone after the test.

Guarantees for every test:
  - no network: creating the real OpenRouter client raises an error, so a test
    that forgot to mock client.chat or client.embed fails loudly instead of
    calling the API
  - no API key is needed or read
  - no test touches app/database/processed.db or app/rag/chroma_db/
"""

import sqlite3
import uuid
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from app import config
from app.llm import client


@pytest.fixture(autouse=True)  # autouse = applied to every test without being asked for
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes any real API call fail, and removes the API key."""

    def refuse_network() -> None:
        raise RuntimeError(
            "Tests must not call the network: mock client.chat / client.embed"
        )

    monkeypatch.setattr(client, "_get_client", refuse_network)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")


# ---------- A small SQLite database instead of processed.db ----------


@pytest.fixture
def test_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Creates a small database in a temporary folder and points config.DB_PATH at it.

    tmp_path is a built-in fixture: a fresh, empty folder for this test only.
    Contents: 250 stock rows (more than MAX_SQL_ROWS), and one order with one line.
    """
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE stock (category TEXT, product TEXT, month TEXT,
                            units_sold INTEGER, units_in_stock INTEGER, unit_price REAL);
        CREATE TABLE customers (customer_id TEXT PRIMARY KEY, company_name TEXT, contact_name TEXT);
        CREATE TABLE orders (order_id TEXT PRIMARY KEY, order_date TEXT, customer_id TEXT,
                             shipped_date TEXT, employee_name TEXT, shipper_name TEXT,
                             ship_address TEXT, ship_city TEXT, ship_country TEXT, total_price REAL);
        CREATE TABLE order_items (order_id TEXT, product_id TEXT, product_name TEXT,
                                  quantity INTEGER, unit_price REAL, line_total REAL);
        INSERT INTO customers VALUES ('VINET', 'Vins et alcools Chevalier', 'Paul Henriot');
        INSERT INTO orders VALUES ('10248', '2016-07-04', 'VINET', '2016-07-16', 'Steven Buchanan',
                                   'Federal Shipping', '59 rue de l''Abbaye', 'Reims', 'France', 168.0);
        INSERT INTO order_items VALUES ('10248', '11', 'Queso Cabrales', 12, 14.0, 168.0);
        """
    )
    for index in range(250):
        connection.execute(
            "INSERT INTO stock VALUES ('Beverages', ?, '2016-07', 1, 10, 2.5)",
            (f"Product {index}",),
        )
    connection.commit()
    connection.close()
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return db_path


# ---------- An in-memory ChromaDB collection ----------

# One in-memory Chroma client for the whole test session: Chroma refuses to create
# a second in-process client with different settings. Telemetry is switched off so
# Chroma itself makes no network calls either.
_CHROMA = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


@pytest.fixture
def chroma_client() -> chromadb.ClientAPI:
    """Returns the shared in-memory ChromaDB client (nothing is written to disk)."""
    return _CHROMA


@pytest.fixture
def empty_collection(chroma_client: chromadb.ClientAPI):
    """A new, empty cosine collection with a unique name, deleted after the test.

    `yield` hands the collection to the test; the code after it runs once the
    test has finished (like a finally block).
    """
    name = f"test_{uuid.uuid4().hex[:12]}"
    collection = chroma_client.create_collection(
        name=name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    yield collection
    chroma_client.delete_collection(name)
