"""
Core RAG Pipeline implementation (refactored).
------------------------------------------------
* Uses Engine classes from sub‑modules instead of legacy free functions
* Keeps pipeline orchestration thin and maintainable
"""

import os
import sys
import io
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv

# ── Local modules ────────────────────────────────────────────────────────────
from .ingestion import (
    extract_pdf_text,
    preprocess_text,
    split_into_sections,
    create_parent_child_splits,
)
from .retrieval import RetrievalEngine
from .rerank import RerankEngine
from .generation import AnswerGenerator
from .validation import AnswerValidator

# ── Config & logging ─────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rag_pipeline.log")],
)
logger = logging.getLogger("10K_RAG")

# Ensure UTF‑8 console (Windows / some IDEs)
if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Pipeline class ───────────────────────────────────────────────────────────
class RAGPipeline:
    """End‑to‑end orchestration wrapper around the specialised engine classes."""

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(self, pdf_path: str, max_tokens_per_chunk: int = 4000):
        self.pdf_path = pdf_path
        self.max_tokens_per_chunk = max_tokens_per_chunk

        # Storage for artefacts produced along the stages
        self.document_chunks: List[Dict[str, Any]] = []
        self.parent_chunks:   List[Dict[str, Any]] = []
        self.child_chunks:    List[Dict[str, Any]] = []
        self.parent_to_children: Dict[str, List[str]] = {}

        # ----------------------------------------------------------------
        # Models & engines (lazy‑loaded inside each class where possible)
        # ----------------------------------------------------------------
        embedding_model   = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        cross_encoder     = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        chat_model        = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        entailment_model  = os.getenv("ENTAILMENT_MODEL", "roberta-large-mnli")

        self.retrieval_engine = RetrievalEngine(embedding_model)
        self.rerank_engine    = RerankEngine(cross_encoder, embedding_model)
        self.generator        = AnswerGenerator(chat_model)
        self.validator        = AnswerValidator(entailment_model, chat_model)

    # ---------------------------------------------------------------------
    # STAGE 1‑3: Ingestion & Chunking
    # ---------------------------------------------------------------------
    def ingest_and_chunk(self) -> None:
        """Read PDF → clean text → hierarchical chunks."""
        logger.info("Stage 1-3 : Ingestion & pre-processing")

        text_by_page = extract_pdf_text(self.pdf_path)
        full_text = preprocess_text(text_by_page)
        sections = split_into_sections(full_text)

        # `create_parent_child_splits` now returns five values:
        (
            self.parent_chunks,
            self.child_chunks,
            self.parent_to_children,
            self.document_chunks,      # <- all chunks (parents+children)
            _text_chunks,              # <- raw texts (unused here)
        ) = create_parent_child_splits(sections)

        logger.info(
            "Chunking complete – %d parents, %d children",
            len(self.parent_chunks),
            len(self.child_chunks),
        )

    # ---------------------------------------------------------------------
    # STAGE 4: Index build
    # ---------------------------------------------------------------------
    def build_indices(self):
        texts = [c["text"] for c in self.document_chunks]
        self.retrieval_engine.set_document_chunks(self.document_chunks)
        self.retrieval_engine.build_indices(texts, self.max_tokens_per_chunk)

    # ---------------------------------------------------------------------
    # STAGE 5: Retrieval
    # ---------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 60):
        return self.retrieval_engine.retrieve(query, k)

    # ---------------------------------------------------------------------
    # STAGE 6: Rerank & assemble context
    # ---------------------------------------------------------------------
    def rerank_and_assemble(self, query: str, retrieval_results):
        return self.rerank_engine.rerank_and_assemble(
            query,
            retrieval_results,
            parent_chunks=self.parent_chunks,
            get_embeddings_fn=self.retrieval_engine._get_embeddings,  # MMR needs embeddings fn
        )

    # ---------------------------------------------------------------------
    # STAGE 7‑8: Answer generation + validation
    # ---------------------------------------------------------------------
    def answer(self, query: str) -> str:
        """Single‑shot helper: retrieve → rerank → generate → validate."""
        retrieval_results = self.retrieve(query)
        context_chunks    = self.rerank_and_assemble(query, retrieval_results)

        answer, _iters = self.generator.self_rag_iteration(query, context_chunks)

        # Entailment & consistency checks
        if not self.validator.verify_entailment(query, answer, context_chunks):
            answer = self.validator.revise_answer(query, answer, context_chunks)

        answer = self.validator.check_consistency_and_truncate(answer)
        return answer


# ── Convenience CLI entrypoint (used by scripts/run_pipeline.py) ───────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG pipeline on a 10‑K PDF")
    parser.add_argument("pdf", help="Path to the 10‑K PDF file")
    parser.add_argument("query", help="User question")
    args = parser.parse_args()

    rag = RAGPipeline(args.pdf)
    rag.ingest_and_chunk()
    rag.build_indices()
    print(rag.answer(args.query))