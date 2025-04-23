"""
Retrieval module for RAG Pipeline.
Handles vector search (FAISS) and sparse retrieval (Whoosh).
"""

import logging
import numpy as np
import faiss
import os
import tempfile
from typing import List, Dict, Any
import openai
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh import scoring
from whoosh.qparser import QueryParser

logger = logging.getLogger("10K_RAG")

class RetrievalEngine:
    def __init__(self, embedding_model_name: str):
        """Initialize retrieval components"""
        self.embedding_model_name = embedding_model_name
        self.faiss_index = None
        self.whoosh_index = None
        self.embeddings = []
        self.document_chunks = []

    def set_document_chunks(self, document_chunks: List[Dict[str, Any]]):
        """Set document chunks to be indexed"""
        self.document_chunks = document_chunks
    
    def build_indices(self, text_chunks: List[str], max_tokens_per_chunk: int = 4000):
        """Build both FAISS (dense) and Whoosh (sparse) indices."""
        logger.info("Building retrieval indices")
        self._build_faiss_index(text_chunks, max_tokens_per_chunk)
        self._build_whoosh_index()
        logger.info("Indices built successfully")

    def retrieve(self, query: str, k: int = 60) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using multiple methods"""
        logger.info("Stage 5: Retrieval")
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0].reshape(1, -1)
        
        # Dense retrieval with FAISS
        dense_results = self._faiss_search(query_embedding, k)
        
        # BM25 retrieval with Whoosh
        bm25_results = self._whoosh_search(query, k)
        
        # Combine results with RRF fusion
        combined_results = self._rrf_fusion(dense_results, bm25_results)
        
        logger.info(f"Retrieved {len(combined_results)} chunks after fusion")
        return combined_results

    def _get_embeddings(self, texts: List[str], max_tokens_per_chunk: int = 4000) -> np.ndarray:
        """
        Embed text chunks with two safeguards:
          1) Truncate any chunk to max_tokens_per_chunk tokens.
          2) Dynamically batch truncated chunks so no call exceeds max_tokens_per_call.
        """
        embeddings: List[List[float]] = []

        # 1) Truncate each chunk to a safe size
        truncated_texts = []
        for txt in texts:
            tokens = txt.split()
            if len(tokens) > max_tokens_per_chunk:
                # Only keep the first max_tokens_per_chunk tokens
                tokens = tokens[:max_tokens_per_chunk]
            truncated_texts.append(" ".join(tokens))

        # 2) Dynamically batch under the context limit
        max_tokens_per_call = 7000   # headroom under 8192
        batch: List[str] = []
        token_sum = 0

        def _embed_batch(batch_texts: List[str]) -> List[List[float]]:
            """Attempt one multi‑text call, fallback to per‑text on failure."""
            try:
                resp = openai.Embedding.create(
                    model=self.embedding_model_name,
                    input=batch_texts
                )
                return [item["embedding"] for item in resp["data"]]
            except Exception as e:
                logger.warning(f"Batch embed failed ({len(batch_texts)} texts): {e}")
                results = []
                for t in batch_texts:
                    try:
                        single = openai.Embedding.create(
                            model=self.embedding_model_name,
                            input=[t]
                        )
                        results.append(single["data"][0]["embedding"])
                    except Exception as err:
                        logger.error(f"Single embed failed: {err}. Zero‑vector used.")
                        results.append([0.0] * 1536)
                return results

        # Build and flush dynamic batches
        for txt in truncated_texts:
            approx = len(txt.split())
            if batch and (token_sum + approx > max_tokens_per_call):
                embeddings.extend(_embed_batch(batch))
                batch, token_sum = [], 0

            batch.append(txt)
            token_sum += approx

        if batch:
            embeddings.extend(_embed_batch(batch))

        return np.array(embeddings, dtype=np.float32)

    def _build_faiss_index(self, texts: List[str], max_tokens_per_chunk: int):
        """Compute embeddings (with truncation + batching) and build FAISS HNSW."""
        if not texts:
            logger.error("No document chunks to index")
            return

        try:
            # Now respects per‑chunk truncation
            self.embeddings = self._get_embeddings(texts, max_tokens_per_chunk)

            d = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexHNSWFlat(d, 32)
            self.faiss_index.hnsw.efConstruction = 250
            self.faiss_index.hnsw.efSearch = 100
            self.faiss_index.add(self.embeddings)

            logger.info(f"FAISS index: {self.faiss_index.ntotal} vectors, dim={d}")
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")

    def _build_whoosh_index(self):
        """Build a Whoosh BM25 inverted index for keyword search."""
        if not self.document_chunks:
            logger.error("No document chunks to index")
            return

        schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
        index_dir = tempfile.mkdtemp()
        try:
            ix = create_in(index_dir, schema)
            writer = ix.writer()
            for i, chunk in enumerate(self.document_chunks):
                writer.add_document(id=str(i), content=chunk["text"])
            writer.commit()
            self.whoosh_index = ix
            logger.info(f"Whoosh index built with {len(self.document_chunks)} docs")
        except Exception as e:
            logger.error(f"Error building Whoosh index: {e}")

    def _faiss_search(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search with FAISS index"""
        try:
            # Search index
            distances, indices = self.faiss_index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.document_chunks):  # Valid index
                    chunk = self.document_chunks[idx]
                    results.append({
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to score
                        "source": "dense",
                        "parent_id": chunk.get("parent_id", chunk["id"]),
                        "section": chunk["section"],
                        "page": chunk["page"]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return []
    
    def _whoosh_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search with Whoosh using BM25F scoring."""
        try:
            # Configure BM25F scorer
            bm25f = scoring.BM25F(K1=1.5, B=0.75)
            # Open a searcher with BM25F weighting
            with self.whoosh_index.searcher(weighting=bm25f) as searcher:
                # Parse the user's query
                parser = QueryParser("content", self.whoosh_index.schema)
                q = parser.parse(query)
                # Execute the search
                results = searcher.search(q, limit=k)
                
                whoosh_results = []
                for hit in results:
                    idx = int(hit["id"])
                    if 0 <= idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]
                        whoosh_results.append({
                            "chunk_id": chunk["id"],
                            "text": chunk["text"],
                            "score": hit.score,
                            "source": "bm25",
                            "parent_id": chunk.get("parent_id", chunk["id"]),
                            "section": chunk["section"],
                            "page": chunk["page"]
                        })
                return whoosh_results

        except Exception as e:
            logger.error(f"Whoosh search error: {e}")
            return []
    
    def _rrf_fusion(self, dense_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """Combine results with Reciprocal Rank Fusion"""
        # Combine all results
        all_results = dense_results + bm25_results
        
        # Group by chunk ID
        chunk_groups = {}
        for result in all_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in chunk_groups:
                chunk_groups[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "sources": [],
                    "ranks": {},
                    "parent_id": result["parent_id"],
                    "section": result["section"],
                    "page": result["page"]
                }
            chunk_groups[chunk_id]["sources"].append(result["source"])
        
        # Assign ranks within each method
        for method in ["dense", "bm25"]:
            method_results = [r for r in all_results if r["source"] == method]
            method_results.sort(key=lambda x: x["score"], reverse=True)
            
            for rank, result in enumerate(method_results):
                chunk_id = result["chunk_id"]
                chunk_groups[chunk_id]["ranks"][method] = rank + 1
        
        # Calculate RRF score
        alpha = 60  # RRF parameter
        
        for chunk_id, chunk in chunk_groups.items():
            rrf_score = 0
            for method, rank in chunk["ranks"].items():
                rrf_score += 1.0 / (alpha + rank)
            chunk["rrf_score"] = rrf_score
        
        # Sort by RRF score
        fused_results = list(chunk_groups.values())
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Limit to top k
        return fused_results[:k]
