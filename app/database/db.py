"""
Builds app/database/processed.db from the PDFs in CompanyDocuments/.

Run from the project root with:  python app/database/db.py

Nothing happens on import: all work is in main(), so other modules (and tests)
can import the upsert functions without touching the database.
"""

import glob
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from parsers.invoice_parser import parse_invoice
from parsers.purchase_order_parser import parse_purchase_order
from parsers.shipping_order_parser import parse_shipping_order
from parsers.stock_report_parser import parse_stock_report_category

DB_PATH = Path(__file__).resolve().parent / "processed.db"
DOCS_ROOT = Path(__file__).resolve().parents[2] / "CompanyDocuments"


def create_schema(conn: sqlite3.Connection) -> None:
    """Creates the four tables if they do not exist yet."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        company_name TEXT,
        contact_name TEXT,
        address TEXT,
        city TEXT,
        postal_code TEXT,
        country TEXT,
        phone TEXT,
        fax TEXT,
        ship_region TEXT
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        order_date TEXT,
        customer_id TEXT REFERENCES customers(customer_id),
        shipped_date TEXT,
        employee_name TEXT,
        shipper_name TEXT,
        ship_address TEXT,
        ship_city TEXT,
        ship_country TEXT,
        total_price REAL
    );

    CREATE TABLE IF NOT EXISTS order_items (
        order_id TEXT REFERENCES orders(order_id),
        product_id TEXT,
        product_name TEXT,
        quantity INTEGER,
        unit_price REAL,
        line_total REAL,
        PRIMARY KEY (order_id, product_id)
    );

    CREATE TABLE IF NOT EXISTS stock (
        category TEXT,
        product TEXT,
        month TEXT,
        units_sold INTEGER,
        units_in_stock INTEGER,
        unit_price REAL,
        PRIMARY KEY (category, product, month)
    );
    """)
    conn.commit()


def upsert_customer(conn: sqlite3.Connection, fields: dict) -> None:
    """Inserts a customer, or fills in missing columns of an existing one.

    Does not commit: the caller commits once per document.
    """
    conn.execute(
        """
        INSERT INTO customers (customer_id, contact_name, company_name, address, city, postal_code, country, phone, fax, ship_region)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(customer_id) DO UPDATE SET
            contact_name = COALESCE(excluded.contact_name, customers.contact_name),
            company_name = COALESCE(excluded.company_name, customers.company_name),
            address = COALESCE(excluded.address, customers.address),
            city = COALESCE(excluded.city, customers.city),
            postal_code = COALESCE(excluded.postal_code, customers.postal_code),
            country = COALESCE(excluded.country, customers.country),
            phone = COALESCE(excluded.phone, customers.phone),
            fax = COALESCE(excluded.fax, customers.fax),
            ship_region = COALESCE(excluded.ship_region, customers.ship_region)
    """,
        (
            fields.get("customer_id"),
            fields.get("contact_name"),
            fields.get("company_name"),
            fields.get("address"),
            fields.get("city"),
            fields.get("postal_code"),
            fields.get("country"),
            fields.get("phone"),
            fields.get("fax"),
            fields.get("ship_region"),
        ),
    )


def upsert_order(conn: sqlite3.Connection, fields: dict) -> None:
    """Inserts an order, or fills in missing columns of an existing one.

    COALESCE keeps the old value when the new document lacks a field, so an
    invoice and a shipping order for the same order complete each other.
    Does not commit: the caller commits once per document.
    """
    conn.execute(
        """
        INSERT INTO orders (order_id, order_date, customer_id, shipped_date, employee_name, shipper_name, ship_address, ship_city, ship_country, total_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_id) DO UPDATE SET
            order_date = COALESCE(excluded.order_date, orders.order_date),
            customer_id = COALESCE(excluded.customer_id, orders.customer_id),
            shipped_date = COALESCE(excluded.shipped_date, orders.shipped_date),
            employee_name = COALESCE(excluded.employee_name, orders.employee_name),
            shipper_name = COALESCE(excluded.shipper_name, orders.shipper_name),
            ship_address = COALESCE(excluded.ship_address, orders.ship_address),
            ship_city = COALESCE(excluded.ship_city, orders.ship_city),
            ship_country = COALESCE(excluded.ship_country, orders.ship_country),
            total_price = COALESCE(excluded.total_price, orders.total_price)
    """,
        (
            fields.get("order_id"),
            fields.get("order_date"),
            fields.get("customer_id"),
            fields.get("shipped_date"),
            fields.get("employee_name"),
            fields.get("shipper_name"),
            fields.get("ship_address"),
            fields.get("ship_city"),
            fields.get("ship_country"),
            fields.get("total_price"),
        ),
    )


def upsert_order_item(conn: sqlite3.Connection, order_id: str, item: dict) -> None:
    """Inserts or replaces one order line and computes its line_total.

    Does not commit: the caller commits once per document.
    """
    line_total = item["quantity"] * item["unit_price"]
    conn.execute(
        """
        INSERT INTO order_items (order_id, product_id, product_name, quantity, unit_price, line_total)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_id, product_id) DO UPDATE SET
            product_name = excluded.product_name,
            quantity = excluded.quantity,
            unit_price = excluded.unit_price,
            line_total = excluded.line_total
    """,
        (
            order_id,
            item["product_id"],
            item["product_name"],
            item["quantity"],
            item["unit_price"],
            line_total,
        ),
    )


def upsert_stock(conn: sqlite3.Connection, row: dict) -> None:
    """Inserts or replaces one product's monthly stock snapshot.

    Does not commit: the caller commits once per document.
    """
    conn.execute(
        """
        INSERT INTO stock (category, product, month, units_sold, units_in_stock, unit_price)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(category, product, month) DO UPDATE SET
            units_sold = excluded.units_sold,
            units_in_stock = excluded.units_in_stock,
            unit_price = excluded.unit_price
    """,
        (
            row["category"],
            row["product"],
            row["month"],
            row["units_sold"],
            row["units_in_stock"],
            row["unit_price"],
        ),
    )


# ---------- One loader per document type ----------
# Each loader parses one PDF and writes its rows. Each takes (conn, path) so
# ingest_folder() can treat them all the same way.


def load_invoice(conn: sqlite3.Connection, path: str) -> None:
    """Parses one invoice PDF and stores its customer, order and line items."""
    result = parse_invoice(path)
    upsert_customer(conn, result)
    upsert_order(conn, result)
    for item in result["line_items"]:
        upsert_order_item(conn, result["order_id"], item)


def load_purchase_order(conn: sqlite3.Connection, path: str) -> None:
    """Parses one purchase order PDF and stores its order and line items."""
    result = parse_purchase_order(path)
    upsert_order(conn, result)
    for item in result["line_items"]:
        upsert_order_item(conn, result["order_id"], item)


def load_shipping_order(conn: sqlite3.Connection, path: str) -> None:
    """Parses one shipping order PDF and stores its customer and order."""
    result = parse_shipping_order(path)
    upsert_customer(conn, result)
    upsert_order(conn, result)


def load_stock_report(conn: sqlite3.Connection, path: str) -> None:
    """Parses one monthly stock report PDF and stores one row per product."""
    for row in parse_stock_report_category(path):
        upsert_stock(conn, row)


# (folder pattern under CompanyDocuments/, loader). Order matters: invoices first,
# then purchase and shipping orders fill in the columns invoices do not have.
SOURCES: list[tuple[str, Callable[[sqlite3.Connection, str], None]]] = [
    ("invoices/*.pdf", load_invoice),
    ("PurchaseOrders/*.pdf", load_purchase_order),
    ("Shipping orders/*.pdf", load_shipping_order),
    ("Inventory Report/monthly-Category/monthly-Category/*.pdf", load_stock_report),
]


def ingest_folder(
    conn: sqlite3.Connection,
    pattern: str,
    loader: Callable[[sqlite3.Connection, str], None],
) -> tuple[int, int]:
    """Runs loader on every PDF matching pattern, one transaction per document.

    Input: an open connection, a glob pattern relative to DOCS_ROOT, and a loader.
    Output: (files_parsed, files_failed).
    """
    parsed = 0
    failed = 0
    for path in sorted(glob.glob(str(DOCS_ROOT / pattern))):
        try:
            # `with conn:` is a transaction, not a resource block: it commits when the
            # block ends normally and rolls back if it raises. It does NOT close the
            # connection. So a document that fails halfway leaves no partial rows.
            with conn:
                loader(conn, path)
            parsed += 1
        # Deliberately broad: one unreadable PDF must not stop the whole build.
        except Exception as error:  # noqa: BLE001
            print(f"FAILED: {path} - {error}")
            failed += 1
    return parsed, failed


def main(db_path: Path = DB_PATH) -> None:
    """Builds (or updates) the database from every PDF and prints a summary.

    Input: where to write the database (a parameter so a test can use a temp file).
    """
    conn = sqlite3.connect(db_path)
    try:
        # Kept from the original script: stock is rebuilt from scratch on every run.
        # The other tables are upserted, so re-running never duplicates rows.
        conn.execute("DROP TABLE IF EXISTS stock")
        create_schema(conn)
        total_parsed = 0
        total_failed = 0
        for pattern, loader in SOURCES:
            parsed, failed = ingest_folder(conn, pattern, loader)
            print(f"{pattern}: {parsed} parsed, {failed} failed")
            total_parsed += parsed
            total_failed += failed
        print(
            f"Done: {total_parsed} files parsed, {total_failed} files failed -> {db_path}"
        )
    finally:
        conn.close()


# Runs only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
