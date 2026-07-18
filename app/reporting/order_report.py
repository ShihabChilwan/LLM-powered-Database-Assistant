import sqlite3
from pathlib import Path
from reporting.dbjson import get_order_report_data
from openpyxl import Workbook


def generate_order_report(report, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Order Report"

    # --- header fields ---
    ws.append(["Order ID", report["order_id"]])
    ws.append(["Order Date", report["order_date"]])
    ws.append(["Customer Company", report["company_name"]])
    ws.append(["Customer Contact", report["contact_name"]])
    ws.append(["Customer ID", report["customer_id"]])
    ws.append(["Ship Address", report["ship_address"]])
    ws.append(["Ship City", report["ship_city"]])
    ws.append(["Ship Country", report["ship_country"]])
    ws.append(["Employee", report["employee_name"]])
    ws.append(["Shipper", report["shipper_name"]])
    ws.append(["Shipped Date", report["shipped_date"]])

    ws.append([])   

    # --- line items table ---
    ws.append(["Product ID", "Product Name", "Quantity", "Unit Price", "Line Total"])
    for item in report["line_items"]:
        ws.append([item["product_id"], item["product_name"], item["quantity"], item["unit_price"], item["line_total"]])

    ws.append([])
    ws.append(["Order Total", report["order_total"]])

    wb.save(output_path)

if __name__ == "__main__":
    conn = sqlite3.connect(Path(__file__).resolve().parents[1] / "database" / "processed.db")
    conn.row_factory = sqlite3.Row
    result = get_order_report_data(conn, "10248")
    generate_order_report(result, "order_10248_report.xlsx")



