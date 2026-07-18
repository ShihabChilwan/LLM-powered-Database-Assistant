import re
import pdfplumber

LINE_PATTERN = re.compile(r"^(.+?)\s+(\d+)\s+(\d+)\s+([\d.]+)$")

def parse_stock_report_category(path):
    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()

    category = None
    month = None
    rows = []
    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("Stock Report for"):
            month = line.replace("Stock Report for", "").strip()
            continue

        match = LINE_PATTERN.match(line)
        if match:
            product, units_sold, units_in_stock, unit_price = match.groups()
            rows.append({
                "category": category,
                "product": product,
                "units_sold": int(units_sold),
                "units_in_stock": int(units_in_stock),
                "unit_price": float(unit_price),
                "month": month,
            })
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip().lower() == "category":
                category = value.strip()

    return rows

if __name__=="__main__":
    result = parse_stock_report_category("CompanyDocuments/Inventory Report/monthly-Category/monthly-Category/StockReport_2016-08_1.pdf")
    import json
    print(json.dumps(result,indent=2))