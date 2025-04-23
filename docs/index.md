# RAG Pipeline for 10-K Report Analysis

## Architecture Overview

This RAG pipeline consists of 8 stages:

1. **Document Ingestion**: Process PDFs into hierarchical parent-child chunks
2. **Query Classification**: Determine intent and translate if needed
3. **Query Reconstruction**: Enhance query with synonyms and simpler sub-questions
4. **Index Building**: Create FAISS (dense) and Whoosh (BM25) indices
5. **Retrieval**: Hybrid search with RRF fusion
6. **Reranking**: Cross-encoder reranking with parent chunk promotion and MMR
7. **Generation**: Self-RAG with LLM for initial answer and critique
8. **Validation**: Post-generation entailment checking and consistency verification

## Module Structure

The pipeline is separated into modules for better organization:

- `pipeline.py`: Core RAGPipeline class orchestrating the workflow
- `ingestion.py`: PDF extraction, preprocessing, and chunking
- `retrieval.py`: FAISS and Whoosh indexing and search
- `rerank.py`: Cross-encoder reranking and MMR diversification
- `generation.py`: LLM-based answer generation with self-RAG
- `validation.py`: Entailment checks and consistency validation

## Usage Examples

```python
# Basic usage
from rag_pipeline.pipeline import RAGPipeline

pipeline = RAGPipeline("annual_report.pdf")
result = pipeline.run_pipeline("What was the revenue growth in 2023?")
print(result["answer"])

# Advanced: Custom chunking parameters
pipeline = RAGPipeline("annual_report.pdf", max_tokens_per_chunk=2048)

# Access individual pipeline stages
pipeline.ingest_document()
pipeline.build_indices()
context = pipeline.retrieve("revenue growth")
```

## Environment Configuration

The pipeline uses these environment variables (typically in a `.env` file):

- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and completions
- `INTENT_MODEL`: Model for query intent classification
- `TRANSLATION_MODEL`: Model for query translation
- `EMBEDDING_MODEL`: Model for vector embeddings
- `CROSS_ENCODER_MODEL`: Model for reranking
- `ENTAILMENT_MODEL`: Model for factual validation
- `CHAT_MODEL`: OpenAI model for answer generation


