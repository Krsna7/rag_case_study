#!/usr/bin/env python3
"""CLI wrapper around the RAG pipeline.

Usage
-----
python -m scripts.run_pipeline <path/to/10K.pdf> "What are the main risk factors?"

The script runs the full eight‑stage pipeline and prints the final (validated) answer
on stdout.  All engine models and hyper‑parameters are configured through the usual
environment variables (see README.md).
"""

import argparse
import logging
import pathlib
import sys
from typing import Optional

from rag_pipeline import RAGPipeline

logger = logging.getLogger("run_pipeline")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RAG pipeline on a single PDF and question",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf",
        type=pathlib.Path,
        help="Path to the 10‑K (or any) PDF to analyse",
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Natural‑language question to ask about the document.  If omitted, an interactive prompt is opened.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=4000,
        help="Maximum tokens per chunk before truncation/batching (passed to the pipeline)",
    )
    return parser


def interactive_question() -> str:
    """Read a multi‑line question from stdin until EOF / empty line."""
    print("Enter your question (end with an empty line):", file=sys.stderr)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.pdf.exists():
        parser.error(f"PDF not found: {args.pdf}")

    question: Optional[str] = args.query or interactive_question()
    if not question:
        parser.error("No question provided")

    # Run pipeline --------------------------------------------------------
    pipe = RAGPipeline(str(args.pdf), max_tokens_per_chunk=args.max_chunk_tokens)

    pipe.ingest_and_chunk()
    pipe.build_indices()
    answer = pipe.answer(question)

    print("\n=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
