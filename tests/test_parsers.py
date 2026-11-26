"""One test per PDF type, each on a real file from CompanyDocuments/ (order 10248 and July 2016)."""

from app import config
from app.parsers.invoice_parser import parse_invoice
from app.parsers.purchase_order_parser import parse_purchase_order
from app.parsers.shipping_order_parser import parse_shipping_order
from app.parsers.stock_report_parser import parse_stock_report_category

DOCS = config.PROJECT_ROOT / "CompanyDocuments"


def test_invoice_parser() -> None:
    result = parse_invoice(str(DOCS / "invoices" / "invoice_10248.pdf"))

    assert result["order_id"] == "10248"
    assert result["customer_id"] == "VINET"
    assert result["order_date"] == "2016-07-04"
    assert result["total_price"] == 440.0
    assert len(result["line_items"]) == 3
    assert result["line_items"][0] == {
        "product_id": "11",
        "product_name": "Queso Cabrales",
        "quantity": 12,
        "unit_price": 14.0,
    }


def test_purchase_order_parser() -> None:
    result = parse_purchase_order(
        str(DOCS / "PurchaseOrders" / "purchase_orders_10248.pdf")
    )

    assert result["order_id"] == "10248"
    assert result["contact_name"] == "Paul Henriot"
    product_names = []
    for item in result["line_items"]:
        product_names.append(item["product_name"])
    assert product_names == [
        "Queso Cabrales",
        "Singaporean Hokkien Fried Mee",
        "Mozzarella di Giovanni",
    ]


def test_shipping_order_parser() -> None:
    result = parse_shipping_order(str(DOCS / "Shipping orders" / "order_10248.pdf"))

    assert result["order_id"] == "10248"
    assert result["shipper_name"] == "Federal Shipping"
    assert result["shipped_date"] == "2016-07-16"
    assert result["ship_country"] == "France"
    assert result["company_name"] == "Vins et alcools Chevalier"


def test_stock_report_parser() -> None:
    path = (
        DOCS
        / "Inventory Report"
        / "monthly-Category"
        / "monthly-Category"
        / "StockReport_2016-07_1.pdf"
    )
    rows = parse_stock_report_category(str(path))

    assert len(rows) == 6
    assert rows[0] == {
        "category": "Beverages",
        "product": "Chang",
        "units_sold": 105,
        "units_in_stock": 17,
        "unit_price": 19.0,
        "month": "2016-07",
    }
