"""
Generation module for the RAG Pipeline.
Contains Self-RAG functionality to generate high-quality answers from retrieved contexts.
"""

import logging
import openai
import os
import re
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("10K_RAG.generation")

class AnswerGenerator:
    """
    Manages answer generation with self-critique and refinement capabilities.
    Implements the Self-RAG paradigm for improved accuracy and factuality.
    """
    
    def __init__(self, chat_model_name: str = None):
        """
        Initialize the generator with model configuration.
        
        Args:
            chat_model_name: Name of the OpenAI model to use for generation
        """
        self.chat_model_name = chat_model_name or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        logger.info(f"Initialized AnswerGenerator with model: {self.chat_model_name}")
    
    def format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context passages for LLM consumption
        
        Args:
            context: List of context dictionaries with text and metadata
            
        Returns:
            Formatted context string with section and page information
        """
        formatted = []
        
        for i, c in enumerate(context):
            section_info = f"Section: {c['section']}" if 'section' in c else ""
            page_info = f"Page: {c['page']}" if 'page' in c else ""
            metadata = f"{section_info} {page_info}".strip()
            
            formatted.append(f"[{i+1}] {metadata}\n{c['text']}\n")
        
        return "\n".join(formatted)
    
    def generate_answer(self, query: str, context: str, previous_answer: str = "") -> str:
        """
        Generate answer using LLM
        
        Args:
            query: The user's question
            context: Formatted context string
            previous_answer: Optional previous answer for refinement
            
        Returns:
            Generated answer based on context
        """
        try:
            system_prompt = """You are a financial analyst assistant that helps extract information from 10-K reports. 
                                Your goal is to provide accurate, concise answers based ONLY on the given context.
                                If the information is not available in the context, say so clearly - do NOT make up information.
                                Be precise and quote specific numbers, metrics, and facts directly from the context when appropriate.
                                Include page and section references when possible."""
            
            user_prompt = f"Question: {query}\n\nContext:\n{context}"
            
            if previous_answer:
                user_prompt += f"\n\nPrevious draft answer: {previous_answer}\n\nPlease revise and improve the answer based on the context provided."
            
            response = openai.ChatCompletion.create(
                model=self.chat_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}")
            return "I'm unable to generate an answer at this time due to a technical issue."
    
    def self_critique(self, query: str, answer: str, context: str) -> str: 
        """
        Self-critique the generated answer
        
        Args:
            query: The user's question
            answer: Generated answer to critique
            context: Formatted context string
            
        Returns:
            Critique of the answer with potential issues highlighted
        """
        try:
            system_prompt = """You are a critical evaluator of financial analysis answers.
                                Your task is to identify any missing evidence or factual errors in the answer relative to the context provided.
                                If you identify missing critical information, mark it with <missing_evidence>specific information needed</missing_evidence> tags.
                                If the answer includes information not supported by the context, note these as unsupported claims.
                                Be specific about what evidence is missing or what claims are unsupported."""
            
            user_prompt = f"Question: {query}\n\nAnswer to evaluate:\n{answer}\n\nContext:\n{context}\n\nProvide your critique:"
            
            response = openai.ChatCompletion.create(
                model=self.chat_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            critique = response.choices[0].message.content.strip()
            return critique
            
        except Exception as e:
            logger.error(f"Self-critique error: {str(e)}")
            return "No critical issues identified."
    
    def merge_contexts(self, original_context: List[Dict[str, Any]], new_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge original and new contexts, avoiding duplicates
        
        Args:
            original_context: Initial context list
            new_context: Additional context to merge
            
        Returns:
            Merged context without duplicates
        """
        seen_ids = set(c["chunk_id"] for c in original_context)
        merged = list(original_context)
        
        for chunk in new_context:
            if chunk["chunk_id"] not in seen_ids:
                merged.append(chunk)
                seen_ids.add(chunk["chunk_id"])

        return merged
    
    def self_rag_iteration(self, query: str, context: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        Generate an answer with a single LLM pass and perform self-critique.
        
        Args:
            query: The user's question
            context: List of context dictionaries
            
        Returns:
            Tuple of (final answer, number of iterations)
        """
        logger.info("Stage 7: Self-RAG iteration")

        # Format the chosen context passages for the LLM
        formatted_context = self.format_context_for_llm(context)

        # 1) Generate the initial answer
        answer = self.generate_answer(query, formatted_context)
        iterations = 1

        # 2) Perform self-critique to log any missing evidence, but do not loop
        critique = self.self_critique(query, answer, formatted_context)
        if "<missing_evidence>" in critique:
            logger.info("Self-critique detected missing evidence; halting further refinement.")

        return answer, iterations
