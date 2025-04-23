# RAG Pipeline for 10-K Report Analysis

A modular Retrieval-Augmented Generation (RAG) pipeline for analyzing 10-K financial reports.

## Features

- **8-Stage Pipeline**: From document ingestion to validated responses
- **Hybrid Retrieval**: Combining dense (FAISS) and sparse (BM25) retrieval methods
- **Parent-Child Chunking**: Hierarchical document chunking strategy
- **Cross-Encoder Reranking**: Improved context relevance with modern reranking
- **MMR Diversification**: Maximize marginal relevance to ensure diverse context
- **Self-RAG**: Self-critique and refinement for accurate answers
- **Post-Generation Validation**: Entailment checking for factual consistency

## Installation

```bash
# Clone the repository
git clone 
cd rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from rag_pipeline.pipeline import RAGPipeline

# Initialize the pipeline with a PDF document
pipeline = RAGPipeline("path/to/10k_report.pdf")

# Ask a question
result = pipeline.run_pipeline("What was the company's revenue in 2023?")

# Print the answer
print(result["answer"])
```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY = "your_api_key"
INTENT_MODEL = "facebook/bart-large-mnli"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en" 
EMBEDDING_MODEL = "text-embedding-ada-002"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENTAILMENT_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
OPENAI_VERSION = "0.28.0"
CHAT_MODEL = "gpt-4o-mini"
```

## Docker Support

To run the pipeline in a containerized environment:

```bash
docker build -t rag-pipeline .
docker run -it --env-file .env rag-pipeline python scripts/run_pipeline.py --pdf /data/report.pdf
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
