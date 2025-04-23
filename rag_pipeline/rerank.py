"""
Reranking module for RAG Pipeline.
Handles cross-encoder reranking, parent promotion, and MMR diversification.
"""

import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("10K_RAG")

class RerankEngine:
    def __init__(self, cross_encoder_model_name: str, embedding_model_name: str):
        """Initialize reranking components"""
        self.cross_encoder_model_name = cross_encoder_model_name
        self.embedding_model_name = embedding_model_name
        self.cross_encoder = None
        self.ce_tokenizer = None
        self.ce_model = None
        
    def rerank_and_assemble(self, query: str, retrieval_results: List[Dict[str, Any]], 
                           parent_chunks: List[Dict[str, Any]], 
                           get_embeddings_fn) -> List[Dict[str, Any]]:
        """
        Perform reranking and context assembly:
        1) Cross-encoder reranking
        2) Parent chunk promotion
        3) MMR diversification
        """
        logger.info("Stage 6: Reranking and context assembly")
        
        # 1) Cross-encoder reranking
        reranked = self._cross_encoder_rerank(query, retrieval_results)
        
        # 2) Promote parent chunks
        with_parents = self._promote_parent_chunks(reranked, parent_chunks)
        
        # 3) Apply MMR for diversity
        final_context = self._apply_mmr(query, with_parents, get_embeddings_fn)
        
        logger.info(f"Final context assembled with {len(final_context)} chunks")
        return final_context

    def _load_cross_encoder(self):
        """Lazy-load the cross-encoder as a text-classification pipeline"""
        if self.cross_encoder is None:
            logger.info(f"Loading cross-encoder model: {self.cross_encoder_model_name}")
            # Sentence-level relevance can be modeled as a sequence-classification task
            self.ce_tokenizer = AutoTokenizer.from_pretrained(self.cross_encoder_model_name)
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(self.cross_encoder_model_name)
            self.cross_encoder = pipeline(
                "text-classification",
                model=self.ce_model,
                tokenizer=self.ce_tokenizer,
                return_all_scores=False,
                truncation=True,         # ← ensure truncation
                max_length=512           # ← model's limit
            )
            logger.info("Cross-encoder model loaded successfully")
        return self.cross_encoder

    def _cross_encoder_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank top-30 passages using the cross-encoder for semantic similarity."""
        try:
            # take up to 30 candidates
            to_rerank = results[:min(30, len(results))]
            ce = self._load_cross_encoder()
            # prepare batch inputs
            inputs = [{"text": query, "text_pair": r["text"]} for r in to_rerank]
            
            scores: List[float] = []
            batch_size = 16
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i : i + batch_size]
                out = ce(batch, truncation=True, max_length=512)  # returns list of {"label":..., "score":...}
                scores.extend([o["score"] for o in out])
            
            # assign and sort
            for idx, r in enumerate(to_rerank):
                r["ce_score"] = scores[idx]
            to_rerank.sort(key=lambda x: x["ce_score"], reverse=True)
            
            return to_rerank[:8]
        
        except Exception as e:
            logger.error(f"Cross-encoder reranking error: {e}")
            # fallback: preserve original ranking, seed ce_score from original score
            fallback = results[:8]
            for r in fallback:
                r["ce_score"] = r.get("rrf_score", 0.0)

            return fallback

    def _promote_parent_chunks(self, results: List[Dict[str, Any]], parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure that whenever a child chunk is chosen, its parent appears too."""
        promoted = list(results)
        included = set()
        
        # record already-included parents
        for r in results:
            included.add(r["chunk_id"] if r["chunk_id"].startswith("p") else r["parent_id"])
        
        # add missing parents
        for r in results:
            if r["chunk_id"].startswith("c") and r["parent_id"] not in included:
                pid = r["parent_id"]
                parent = next((p for p in parent_chunks if p["id"] == pid), None)
                if parent:
                    promoted.append({
                        "chunk_id": pid,
                        "text": parent["text"],
                        "ce_score": r["ce_score"] * 0.95,  # slightly lower
                        "parent_id": pid,
                        "section": parent["section"],
                        "page": parent["page"]
                    })
                    included.add(pid)
        
        # final sort by ce_score
        promoted.sort(key=lambda x: x["ce_score"], reverse=True)
        return promoted
    
    def _apply_mmr(self, query: str, results: List[Dict[str, Any]], get_embeddings_fn) -> List[Dict[str, Any]]:
        """Apply Maximal Marginal Relevance for diversity"""
        if len(results) <= 1:
            return results
        
        # Get query embedding
        query_embedding = get_embeddings_fn([query])[0]
        
        # Get embeddings for results
        result_texts = [r["text"] for r in results]
        result_embeddings = get_embeddings_fn(result_texts)
        
        # MMR parameters
        lambda_param = 0.5  # Diversity parameter
        selected = []
        remaining = list(range(len(results)))
        
        # Initialize similarity matrices
        query_sim = cosine_similarity([query_embedding], result_embeddings)[0]
        doc_sim = cosine_similarity(result_embeddings)
        
        # First document is the most similar to the query
        first_idx = np.argmax(query_sim)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Select documents using MMR
        while remaining and len(selected) < 8:
            mmr_scores = []
            
            for doc_idx in remaining:
                # Relevance term
                relevance = query_sim[doc_idx]
                
                # Diversity term (maximum similarity to any selected document)
                if selected:
                    diversity = max(doc_sim[doc_idx][sel_idx] for sel_idx in selected)
                else:
                    diversity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((doc_idx, mmr_score))
            
            # Select document with highest MMR score
            selected_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(selected_idx)
            remaining.remove(selected_idx)
        
        # Assemble final results in order of selection
        mmr_results = [results[idx] for idx in selected]
        return mmr_results

    def format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """Format context passages for LLM consumption"""
        formatted = []
        
        for i, c in enumerate(context):
            section_info = f"Section: {c['section']}" if 'section' in c else ""
            page_info = f"Page: {c['page']}" if 'page' in c else ""
            metadata = f"{section_info} {page_info}".strip()
            
            formatted.append(f"[{i+1}] {metadata}\n{c['text']}\n")
        
        return "\n".join(formatted)
