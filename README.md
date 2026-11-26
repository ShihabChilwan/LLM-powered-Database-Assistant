# LLM-powered Database Assistant

Ask plain-English questions about a wholesale food distributor's orders,
customers and stock. Structured company PDFs are parsed into SQLite, and an
LLM answers by running read-only SQL queries, showing every query it ran.

> This README is completed in a later step. For now it records the known
> limitations of the data and the code.

## Known limitations

- **Some orders have no shipper.** 21 of the 830 orders have no
  `shipper_name` and no `shipped_date`. They appear only in purchase orders,
  with no matching shipping order. The data is left as it is: `NULL` means
  "not recorded", not "not shipped". The schema notes given to the model say
  this, so it reports the gap instead of guessing, and a filter such as
  `WHERE shipper_name = 'Speedy Express'` silently skips these orders.
- **SQL results are capped** at `MAX_SQL_ROWS` (200) rows per query. The
  model is told when a result was cut off (`"truncated": true`, plus
  `"total_rows"`) and is asked to narrow the query.
- **The write keyword check is only a first filter.** It looks at the first
  word of a statement, so `WITH x AS (SELECT 1) DELETE FROM stock` passes it.
  The real protection is that every connection the app uses to answer
  questions is opened read-only (`mode=ro`), so SQLite refuses any write.
