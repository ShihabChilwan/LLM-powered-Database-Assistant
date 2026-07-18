import pdfplumber

def parse_purchase_order(path):
    with pdfplumber.open(path) as pdf:
        page0 = pdf.pages[0]
        page0_tables = page0.extract_tables()

        # --- Region 1: header table (1 header row + 1 data row) ---
        header_table = page0_tables[0]
        header_cols = [c.strip().lower().replace(" ", "_") for c in header_table[0]]
        header = dict(zip(header_cols, header_table[1]))

        header["contact_name"] = header.pop("customer_name")

        # --- Region 2: Product table, possibly spanning multiple pages ---
        product_table_page0 = page0_tables[1]
        col_names = [c.rstrip(":").strip().lower().replace(" ", "_") for c in product_table_page0[0]]

        product_rows = product_table_page0[1:]
        for page in pdf.pages[1:]:
            product_rows.extend(page.extract_tables()[0])

        line_items = []
        for row in product_rows:
            item = dict(zip(col_names, row))
            item["product_name"] = item.pop("product")
            item["quantity"] = int(item["quantity"])
            item["unit_price"] = float(item["unit_price"])
            line_items.append(item)

        header["line_items"] = line_items
        return header

if __name__=="__main__":
    result = parse_purchase_order("CompanyDocuments/PurchaseOrders/purchase_orders_11077.pdf")
    import json
    print(json.dumps(result,indent=2))