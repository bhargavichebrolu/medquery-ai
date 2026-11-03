"""Grounded answers: retrieve, build the context, generate (PRD 3.3).

`answer_question` runs the whole flow for one question, then checks the provider's citations
against the retrieved chunks (`llm.citations`, PRD 3.4).

The review command answers the research's 10 evaluation questions for reading alongside the
archived Llama answers. It uses the mock provider unless told otherwise, and refuses to call a
paid provider without --allow-paid-calls:

    uv run python -m medquery_api.answering review [--provider claude --allow-paid-calls]
        [--model MPNet] [--top-k 5] [--out PATH]
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from medquery_api import config
from medquery_api.embeddings.query import QueryEmbedder
from medquery_api.knowledge_base import KnowledgeBase
from medquery_api.llm import LLMProvider, create_provider
from medquery_api.llm.base import GenerationResult
from medquery_api.llm.citations import (
    CitationCheck,
    Grounding,
    check_citations,
    grounding_status,
)
from medquery_api.llm.context import AnswerContext, build_context
from medquery_api.pricing import CostEstimate, estimate_cost
from medquery_api.retrieval import Retrieval, retrieve

EVALUATION_FILE = (
    config.REPO_ROOT / "Healthcare_RAG" / "results" / "formal_rag_evaluation_data.json"
)


@dataclass(frozen=True)
class Answer:
    retrieval: Retrieval
    context: AnswerContext
    result: GenerationResult  # exactly as the provider returned it
    citations: CitationCheck  # the answer with only valid citations, plus what was dropped
    grounding: Grounding
    cost: CostEstimate
    total_ms: float  # retrieval, context building, generation and citation checks together


def answer_question(
    question: str,
    embedding_model: str,
    top_k: int,
    *,
    knowledge_base: KnowledgeBase,
    embedder: QueryEmbedder,
    provider: LLMProvider,
) -> Answer:
    started = time.perf_counter()
    retrieval = retrieve(
        question, embedding_model, top_k, knowledge_base=knowledge_base, embedder=embedder
    )
    context = build_context(retrieval, knowledge_base)
    result = provider.generate(context.request)
    citations = check_citations(result.parts, context.request.documents)
    total_ms = (time.perf_counter() - started) * 1000
    return Answer(
        retrieval=retrieval,
        context=context,
        result=result,
        citations=citations,
        grounding=grounding_status(result, citations),
        cost=estimate_cost(result.provider, result.model, result.usage),
        total_ms=total_ms,
    )


# --- The 10-question review command ---


def load_evaluation_questions(path: Path = EVALUATION_FILE) -> list[dict]:
    """The research's evaluation questions with its archived (historical) Llama answers."""
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    return [
        {
            "question_id": record["question_id"],
            "question": record["question"],
            "relevant_sources": record["relevant_sources"],
            "historical_answer": record["answer"],
            "historical_generator": record["generator_model"],
        }
        for record in records
    ]


def review(
    questions: list[dict],
    embedding_model: str,
    top_k: int,
    *,
    knowledge_base: KnowledgeBase,
    embedder: QueryEmbedder,
    provider: LLMProvider,
) -> list[dict]:
    rows = []
    for item in questions:
        answer = answer_question(
            item["question"],
            embedding_model,
            top_k,
            knowledge_base=knowledge_base,
            embedder=embedder,
            provider=provider,
        )
        result, check = answer.result, answer.citations
        rows.append(
            {
                **item,
                "answer": result.text,
                "grounding": answer.grounding,
                "unsupported_sentences": [
                    result.text[s.start : s.end] for s in check.sentence_list if not s.supported
                ],
                "citations": [asdict(c) for part in check.parts for c in part.citations],
                "dropped_citations": [asdict(d) for d in check.dropped],
                "citation_check": {
                    "total": check.total,
                    "valid": check.valid,
                    "dropped_by_reason": check.dropped_by_reason,
                    "validity_rate": check.validity_rate,
                    "sentences": check.sentences,
                    "supported_sentences": check.supported_sentences,
                    "grounding_coverage": check.grounding_coverage,
                },
                "sources": [asdict(s) for s in answer.context.sources],
                "provider": result.provider,
                "model": result.model,
                "stop_reason": result.stop_reason,
                "refused": result.refused,
                "usage": asdict(result.usage),
                "estimated_cost_usd": answer.cost.usd,
                "generation_ms": round(result.latency_ms, 1),
                "total_ms": round(answer.total_ms, 1),
            }
        )
    return rows


def _refuse_archive_path(path: Path) -> Path:
    resolved = path.resolve()
    archive = (config.REPO_ROOT / "Healthcare_RAG").resolve()
    if resolved == archive or archive in resolved.parents:
        raise SystemExit(f"Refusing to write inside the research archive: {path}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m medquery_api.answering")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("review", help="answer the 10 research evaluation questions")
    command.add_argument("--provider", default="mock", choices=["mock", "claude"])
    command.add_argument("--allow-paid-calls", action="store_true", help="required for claude")
    command.add_argument("--model", default="MPNet", help="embedding model (research used MPNet)")
    command.add_argument("--top-k", type=int, default=5)
    command.add_argument(
        "--out", type=Path, help="write the answers as JSON (not in Healthcare_RAG)"
    )
    args = parser.parse_args(argv)

    if args.provider != "mock" and not args.allow_paid_calls:
        print(
            f"The {args.provider} provider costs money per question. Re-run with "
            "--allow-paid-calls once the cost is approved.",
            file=sys.stderr,
        )
        return 2
    out = _refuse_archive_path(args.out) if args.out else None

    from medquery_api.knowledge_base import load_knowledge_base

    rows = review(
        load_evaluation_questions(),
        args.model,
        args.top_k,
        knowledge_base=load_knowledge_base(),
        embedder=QueryEmbedder(),
        provider=create_provider(args.provider),
    )
    for row in rows:
        cost, check = row["estimated_cost_usd"], row["citation_check"]
        coverage = check["grounding_coverage"]
        print(
            f"Q{row['question_id']:>2} {row['stop_reason']:<9} {row['grounding']:<19} "
            f"citations {check['valid']}/{check['total']} valid "
            f"coverage {'n/a' if coverage is None else f'{coverage:.0%}'} "
            f"cost {'unknown' if cost is None else f'${cost:.4f}'} "
            f"total {row['total_ms']:.0f} ms | {row['answer'][:70]!r}"
        )
    known = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
    print(f"{len(rows)} questions; estimated cost ${sum(known):.4f} (provider {args.provider})")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Answers written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
