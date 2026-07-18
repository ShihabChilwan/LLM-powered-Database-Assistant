import sqlite3
from pathlib import Path

def get_order_report_data(conn, order_id):
    header_row = conn.execute("""
        SELECT o.order_id, o.order_date, o.customer_id, o.shipped_date, o.employee_name, o.shipper_name,
               o.ship_address, o.ship_city, o.ship_country, o.total_price,
               c.company_name, c.contact_name
        FROM orders o
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_id = ?
    """, (order_id,)).fetchone()

    report = dict(header_row)

    line_items = conn.execute("""
        SELECT product_id, product_name, quantity, unit_price, line_total
        FROM order_items
        WHERE order_id = ?
        ORDER BY product_id
    """, (order_id,)).fetchall()

    report["line_items"] = [dict(item) for item in line_items]
    report["order_total"] = sum(item["line_total"] for item in report["line_items"])

    return report

if __name__ == "__main__":
    conn = sqlite3.connect(Path(__file__).resolve().parents[1] / "database" / "processed.db")
    conn.row_factory = sqlite3.Row
    result = get_order_report_data(conn, "10248")
    import json
    print(json.dumps(result, indent=2))


