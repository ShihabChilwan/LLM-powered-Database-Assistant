import pdfplumber

def parse_shipping_order(path):
    with pdfplumber.open(path) as pdf:
        lines = []
        for page in pdf.pages:
            lines.extend(page.extract_text().split("\n"))

    header = {}
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        if key.strip() == "Products":
            break   # everything from here on (products, total price) is out of scope

        value = value.strip()
        if not value:
            continue   # other section-header lines (e.g. "Shipping Details:") - nothing to store

        key_clean = key.strip().lower().replace(" ", "_")
        header[key_clean] = value

    header["company_name"] = header.pop("ship_name")
    header.pop("customer_name", None)
    return header

if __name__=="__main__":
    result = parse_shipping_order("CompanyDocuments/Shipping orders/order_10248.pdf")
    import json
    print(json.dumps(result,indent=2))