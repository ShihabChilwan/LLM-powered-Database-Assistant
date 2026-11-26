"""
Runs every evaluation question through the same agent the app uses and scores it.

Scored separately (see eval/scoring.py):
  tool_choice, answer_facts, citations, refusal, injection
Also recorded per question: tokens, latency, cost, tool calls and errors.

Outputs: a terminal summary, eval/results.json and eval/results.md.

Needs the database, the document index (python -m app.rag.ingest) and
OPENROUTER_API_KEY. Run from the project root with:
    python eval/evaluate.py              # one run
    python eval/evaluate.py --runs 2     # two runs (LLM output varies)
    python eval/evaluate.py --ids mix-02 inj-01
"""

import argparse
import json
import sys
from pathlib import Path

# Make `import app...` and `import eval...` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # must come after the sys.path change
from app.llm.agent import AgentResult, answer_question
from eval import scoring

EVAL_DIR = PROJECT_ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_JSON = EVAL_DIR / "results.json"
RESULTS_MD = EVAL_DIR / "results.md"

SCORERS = ["tool_choice", "answer_facts", "citations", "refusal", "injection"]
TYPES = ["sql", "documents", "mixed", "unanswerable", "injection"]


def load_questions(ids: list[str] | None) -> list[dict]:
    """Reads eval/questions.json, optionally keeping only the given ids."""
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    if not ids:
        return questions
    selected = []
    for question in questions:
        if question["id"] in ids:
            selected.append(question)
    return selected


def run_details(result: AgentResult) -> dict:
    """Extracts what we record about one run: retrieved chunks, citations, errors, cost."""
    retrieved_ids = []
    errors = []
    for step in result.steps:
        if step["error"]:
            errors.append(f"{step['tool']}: {step['error']}")
        elif step["tool"] == "search_documents":
            for passage in step["result"]["passages"]:
                retrieved_ids.append(passage["chunk_id"])
    cost = 0.0
    for call in result.model_calls:
        cost += call["cost"] or 0.0  # `or 0.0` turns a missing (None) cost into 0
        if call["error"]:
            errors.append(f"model: {call['error']}")
    verified_sources = []
    for citation in result.citation_check.verified:
        verified_sources.append(citation.source)
    unverified = []
    for citation in result.citation_check.unverified:
        unverified.append(citation.text())
    return {
        "retrieved_ids": retrieved_ids,
        "errors": errors,
        "cost": round(cost, 6),
        "verified_sources": sorted(set(verified_sources)),
        "unverified_citations": unverified,
    }


def score_question(question: dict, result: AgentResult, details: dict) -> dict:
    """Runs all five scorers. Output: {scorer name: {"passed": ..., "reason": ...}}."""
    sql_facts = []
    if question.get("expected_sql"):
        sql_facts = scoring.facts_from_sql(question["expected_sql"])
    scores = {
        "tool_choice": scoring.score_tool_choice(question, result.tools_used),
        "answer_facts": scoring.score_answer_facts(question, result.answer, sql_facts),
        "citations": scoring.score_citations(
            question, details["verified_sources"], details["unverified_citations"]
        ),
        "refusal": scoring.score_refusal(question, result.answer),
        "injection": scoring.score_injection(
            question, result.answer, details["retrieved_ids"]
        ),
    }
    as_dicts = {}
    for name, score in scores.items():
        as_dicts[name] = {"passed": score.passed, "reason": score.reason}
    return as_dicts


def evaluate_question(question: dict) -> dict:
    """Asks one question through the agent and returns its full record."""
    result = answer_question(question["question"])
    details = run_details(result)
    scores = score_question(question, result, details)
    failures = []
    for name in SCORERS:
        if scores[name]["passed"] is False:
            failures.append(scores[name]["reason"])
    tool_inputs = []
    for step in result.steps:
        tool_inputs.append({"tool": step["tool"], "input": step["input"]})
    return {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "answer": result.answer,
        "tools_used": result.tools_used,
        "tool_calls": tool_inputs,
        "model_calls": len(result.model_calls),
        "tokens": result.total_tokens,
        "latency_s": result.total_latency_s,
        **details,  # adds retrieved_ids, errors, cost, verified_sources, unverified_citations
        "scores": scores,
        "passed": not failures,
        "failures": failures,
    }


def summarize(records: list[dict]) -> dict:
    """Builds per-scorer pass rates and per-type averages for one run."""
    scorer_rates = {}
    for name in SCORERS:
        applicable = 0
        passed = 0
        for record in records:
            outcome = record["scores"][name]["passed"]
            if outcome is not None:
                applicable += 1
                passed += 1 if outcome else 0
        scorer_rates[name] = {"passed": passed, "applicable": applicable}
    by_type = {}
    for question_type in TYPES + ["all"]:
        group = []
        for record in records:
            if question_type in ("all", record["type"]):
                group.append(record)
        if group:
            by_type[question_type] = averages(group)
    return {"scorers": scorer_rates, "by_type": by_type}


def averages(group: list[dict]) -> dict:
    """Average tokens, latency, cost and tool calls for a group of records, plus totals."""
    count = len(group)
    tokens = sum(record["tokens"] for record in group)
    latency = sum(record["latency_s"] for record in group)
    cost = sum(record["cost"] for record in group)
    tool_calls = sum(len(record["tool_calls"]) for record in group)
    errors = sum(len(record["errors"]) for record in group)
    passed = sum(1 for record in group if record["passed"])
    return {
        "questions": count,
        "passed": passed,
        "avg_tokens": round(tokens / count),
        "avg_latency_s": round(latency / count, 1),
        "avg_cost": round(cost / count, 6),
        "avg_tool_calls": round(tool_calls / count, 1),
        "total_tokens": tokens,
        "total_cost": round(cost, 6),
        "errors": errors,
    }


def print_summary(run_number: int, summary: dict, records: list[dict]) -> None:
    """Prints the per-scorer pass rates, per-type averages and failures of one run."""
    print(f"\n=== Run {run_number} ===")
    for name, rate in summary["scorers"].items():
        print(f"  {name:<13} {rate['passed']}/{rate['applicable']}")
    for question_type, stats in summary["by_type"].items():
        print(
            f"  {question_type:<13} {stats['passed']}/{stats['questions']} passed, "
            f"avg {stats['avg_tokens']} tokens, {stats['avg_latency_s']}s, "
            f"{stats['avg_cost']:.6f} credits"
        )
    for record in records:
        if not record["passed"]:
            print(f"  FAIL {record['id']}: {' | '.join(record['failures'])}")


def mark(score: dict) -> str:
    """Returns the table symbol for one scorer outcome."""
    if score["passed"] is None:
        return "–"
    return "✅" if score["passed"] else "❌"


def markdown_for_run(run_number: int, summary: dict, records: list[dict]) -> list[str]:
    """Builds the Markdown section for one run: pass rates, averages, one row per question."""
    lines = [f"## Run {run_number}", "", "| Scorer | Passed |", "|---|---|"]
    for name, rate in summary["scorers"].items():
        lines.append(f"| {name} | {rate['passed']}/{rate['applicable']} |")
    lines += [
        "",
        "| Type | Passed | Avg tokens | Avg latency (s) | Avg cost (credits) | Avg tool calls |",
        "|---|---|---|---|---|---|",
    ]
    for question_type, stats in summary["by_type"].items():
        lines.append(
            f"| {question_type} | {stats['passed']}/{stats['questions']} | {stats['avg_tokens']} | "
            f"{stats['avg_latency_s']} | {stats['avg_cost']:.6f} | {stats['avg_tool_calls']} |"
        )
    lines += [
        "",
        "| ID | Type | Tools | Tool | Facts | Cites | Refusal | Injection | Tokens | Latency (s) | Failure reasons |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        scores = record["scores"]
        reasons = "; ".join(record["failures"]).replace("|", "/") or "–"
        lines.append(
            f"| {record['id']} | {record['type']} | {', '.join(record['tools_used']) or 'none'} | "
            f"{mark(scores['tool_choice'])} | {mark(scores['answer_facts'])} | "
            f"{mark(scores['citations'])} | {mark(scores['refusal'])} | "
            f"{mark(scores['injection'])} | {record['tokens']} | {record['latency_s']} | {reasons} |"
        )
    lines.append("")
    return lines


def write_outputs(runs: list[dict]) -> None:
    """Writes eval/results.json (everything) and eval/results.md (the tables)."""
    settings = {
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "min_score": config.MIN_SCORE,
        "top_k": config.TOP_K,
        "max_turns": config.MAX_TURNS,
    }
    RESULTS_JSON.write_text(
        json.dumps({"settings": settings, "runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# Evaluation results", ""]
    for key, value in settings.items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "✅ passed · ❌ failed · – does not apply to this question", ""]
    for run in runs:
        lines += markdown_for_run(run["run"], run["summary"], run["records"])
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Parses the command line, runs the evaluation and writes the outputs."""
    # argparse reads command-line options, e.g. --runs 2 (like a small CLI library).
    parser = argparse.ArgumentParser(description="Evaluate the assistant.")
    parser.add_argument(
        "--runs", type=int, default=1, help="how many times to run every question"
    )
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    options = parser.parse_args()

    questions = load_questions(options.ids)
    runs = []
    for run_number in range(1, options.runs + 1):
        records = []
        for question in questions:
            print(f"run {run_number}: {question['id']} ...", flush=True)
            records.append(evaluate_question(question))
        summary = summarize(records)
        print_summary(run_number, summary, records)
        runs.append({"run": run_number, "summary": summary, "records": records})
    write_outputs(runs)
    print(f"\nWrote {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    main()
