import sqlite3
import glob
import sys
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

conn = sqlite3.connect(DB_PATH)
conn.execute("DROP TABLE IF EXISTS stock")
def create_schema(conn):
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


def upsert_customer(conn, fields):
    conn.execute("""
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
    """, (
        fields.get("customer_id"), fields.get("contact_name"), fields.get("company_name"),
        fields.get("address"), fields.get("city"), fields.get("postal_code"),
        fields.get("country"), fields.get("phone"), fields.get("fax"), fields.get("ship_region"),
    ))
    conn.commit()


def upsert_order(conn, fields):
    conn.execute("""
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
    """, (
        fields.get("order_id"), fields.get("order_date"), fields.get("customer_id"),
        fields.get("shipped_date"), fields.get("employee_name"), fields.get("shipper_name"),
        fields.get("ship_address"), fields.get("ship_city"), fields.get("ship_country"),
        fields.get("total_price"),
    ))
    conn.commit()


def upsert_order_item(conn, order_id, item):
    line_total = item["quantity"] * item["unit_price"]
    conn.execute("""
        INSERT INTO order_items (order_id, product_id, product_name, quantity, unit_price, line_total)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(order_id, product_id) DO UPDATE SET
            product_name = excluded.product_name,
            quantity = excluded.quantity,
            unit_price = excluded.unit_price,
            line_total = excluded.line_total
    """, (order_id, item["product_id"], item["product_name"], item["quantity"], item["unit_price"], line_total))
    conn.commit()


def upsert_stock(conn, row):
    conn.execute("""
        INSERT INTO stock (category, product, month, units_sold, units_in_stock, unit_price)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(category, product, month) DO UPDATE SET
            units_sold = excluded.units_sold,
            units_in_stock = excluded.units_in_stock,
            unit_price = excluded.unit_price
    """, (row["category"], row["product"], row["month"], row["units_sold"], row["units_in_stock"], row["unit_price"]))
    conn.commit()


create_schema(conn)

for path in glob.glob(str(DOCS_ROOT / "invoices" / "*.pdf")):
    try:
        result = parse_invoice(path)
        upsert_customer(conn, result)
        upsert_order(conn, result)
        for item in result["line_items"]:
            upsert_order_item(conn, result["order_id"], item)
    except Exception as e:
        print(f"FAILED: {path} — {e}")

for path in glob.glob(str(DOCS_ROOT / "PurchaseOrders" / "*.pdf")):
    try:
        result = parse_purchase_order(path)
        upsert_order(conn, result)
        for item in result["line_items"]:
            upsert_order_item(conn, result["order_id"], item)
    except Exception as e:
        print(f"FAILED: {path} — {e}")

for path in glob.glob(str(DOCS_ROOT / "Shipping orders" / "*.pdf")):
    try:
        result = parse_shipping_order(path)
        upsert_customer(conn, result)
        upsert_order(conn, result)
    except Exception as e:
        print(f"FAILED: {path} — {e}")

for path in glob.glob(str(DOCS_ROOT / "Inventory Report" / "monthly-Category" / "monthly-Category" / "*.pdf")):
    try:
        for row in parse_stock_report_category(path):
            upsert_stock(conn, row)
    except Exception as e:
        print(f"FAILED: {path} — {e}")

