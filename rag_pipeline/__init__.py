"""
RAG Pipeline for 10-K Report Analysis
====================================

A modular Retrieval-Augmented Generation pipeline for analyzing 10-K reports,
following an 8-stage architecture with environment variables and logging.

Modules:
    - pipeline: Core RAGPipeline class and main interface
    - ingestion: PDF extraction and document chunking
    - retrieval: FAISS and Whoosh indexing and search
    - rerank: Cross-encoder reranking and MMR for diversity
    - generation: Self-RAG and answer generation with OpenAI
    - validation: Post-generation validation and consistency checks
"""

from importlib import metadata

from .pipeline import RAGPipeline    

__all__ = ["RAGPipeline"]
__version__ = metadata.version(__name__)