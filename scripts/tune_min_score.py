"""
Helps choose MIN_SCORE: prints the retrieval scores of labelled test queries.

Each query is labelled:
  relevant    - the policy documents answer it
  borderline  - related topic, but the documents do NOT answer it
  irrelevant  - nothing to do with the documents

A good MIN_SCORE keeps the top chunk of every relevant query and drops most
borderline and all irrelevant ones. Retrieval alone cannot separate borderline
queries perfectly (they share words with the policies), so the agent's
"answer only from the documents" instruction is the second line of defence.

Needs the index (python -m app.rag.ingest) and OPENROUTER_API_KEY.
Run from the project root with:  python scripts/tune_min_score.py
"""

import sys
from pathlib import Path

# Make `import app...` work when this file is run directly (sys.path is Python's classpath).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # must come after the sys.path change
from app.rag.retriever import search

QUERIES = [
    ("relevant", "How long do I have to report damaged cheese?"),
    ("relevant", "What is the late payment fee?"),
    ("relevant", "Which shipper is fastest to Europe?"),
    ("relevant", "What temperature should seafood be stored at?"),
    ("relevant", "When do we reorder beverages?"),
    ("relevant", "Can I return unopened products I no longer need?"),
    ("relevant", "Do you give discounts on big orders?"),
    ("relevant", "Who pays import duties on a shipment to Brazil?"),
    ("relevant", "My invoice looks wrong, what can I do?"),
    ("relevant", "What happens if my account is two months overdue?"),
    ("relevant", "When is the supplier portal offline?"),
    (
        "relevant",
        "Order arrived three days late with United Package, any compensation?",
    ),
    # Mixed and terse questions: the order details dilute the policy part, so they score lower.
    ("relevant", "Order 10248 arrived damaged. Is the customer eligible for a refund?"),
    (
        "relevant",
        "Customer VINET's cheese arrived warm yesterday. What are their options?",
    ),
    ("relevant", "Chang is down to 15 units in stock. Should we reorder it?"),
    ("relevant", "refund damaged dairy"),
    ("relevant", "What does the supplier portal notice say?"),
    ("borderline", "How many vacation days do employees get?"),
    ("borderline", "Are your products certified organic?"),
    ("borderline", "What is the warranty on refrigeration equipment?"),
    ("borderline", "What is your customer data privacy policy?"),
    ("borderline", "What is the minimum order value for customers?"),
    ("borderline", "Is there a loyalty points programme for customers?"),
    ("borderline", "How many orders did customer VINET place?"),
    ("borderline", "What was the total price of order 10248?"),
    ("irrelevant", "What is the capital of France?"),
    ("irrelevant", "How do I bake sourdough bread?"),
    ("irrelevant", "Who won the 2018 football World Cup?"),
    ("irrelevant", "Write a Python function that sorts a list."),
    ("irrelevant", "Explain quantum entanglement in simple terms."),
    ("irrelevant", "What will the weather be like tomorrow?"),
]


def main() -> None:
    """Prints one row per query: label, top score, the 2nd to 4th scores, top chunk."""
    print(
        f"Current MIN_SCORE = {config.MIN_SCORE}; embedding model = {config.EMBEDDING_MODEL}"
    )
    print(f"{'label':<11} {'top':>5}  {'next scores':<17}  top chunk  |  query")
    for label, query in QUERIES:
        # min_score=-1.0 keeps every chunk, so we see the raw scores.
        chunks = search(query, k=config.TOP_K, min_score=-1.0)
        next_scores = []
        for chunk in chunks[1:]:
            next_scores.append(f"{chunk.score:.2f}")
        top = chunks[0]
        print(
            f"{label:<11} {top.score:5.3f}  {' '.join(next_scores):<17}  "
            f"{top.chunk_id}  |  {query}"
        )


if __name__ == "__main__":
    main()
