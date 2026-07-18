import pdfplumber

def parse_invoice(path):
    with pdfplumber.open(path) as pdf:
        page0 = pdf.pages[0]

        # --- Region 1: loose header lines (before "Customer Details:") ---
        header = {}
        for line in page0.extract_text().split("\n"):
            line = line.strip()
            if line == "Customer Details:":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                header[key] = value.strip()

        # --- Region 2: Customer Details table -> label/value rows ---
        page0_tables = page0.extract_tables()
        for label, value in page0_tables[0]:
            key = label.rstrip(":").strip().lower().replace(" ", "_")
            header[key] = value.strip()

        # --- Region 3: Product Details table -> line items + total ---
        product_table_page0 = page0_tables[1]
        col_names = [c.strip().lower().replace(" ", "_") for c in product_table_page0[0]]

        product_rows = product_table_page0[1:]
        for page in pdf.pages[1:]:
            product_rows.extend(page.extract_tables()[0])
        line_items = []
        total_price = None
        for row in product_rows:
            if row[2] == "TotalPrice":
                total_price = float(row[3])
                continue
            item = dict(zip(col_names, row))
            item["quantity"] = int(item["quantity"])
            item["unit_price"] = float(item["unit_price"])
            line_items.append(item)

        header["line_items"] = line_items
        header["total_price"] = total_price
        return header

if __name__ == "__main__":
    result = parse_invoice("CompanyDocuments/invoices/invoice_11077.pdf")
    import json
    print(json.dumps(result, indent=2))
    computed_total = sum(item["quantity"] * item["unit_price"] for item in result["line_items"])
    assert abs(computed_total - result["total_price"]) < 0.01, f"Mismatch: {computed_total} vs {result['total_price']}"