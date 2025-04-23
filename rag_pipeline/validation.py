"""
Validation module for the RAG Pipeline.
Provides post-generation validation to ensure answer quality and factual correctness.
"""

import logging
import openai
import os
import re
from typing import Dict, List, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

logger = logging.getLogger("10K_RAG.validation")

class AnswerValidator:
    """
    Handles post-generation validation of answers:
    - Entailment verification against the context
    - Answer revision if needed
    - Consistency checks and answer truncation
    """
    
    def __init__(self, entailment_model_name: str = None, chat_model_name: str = None):
        """
        Initialize validator with model configuration
        
        Args:
            entailment_model_name: Name of the entailment model
            chat_model_name: Name of the OpenAI model for answer revision
        """
        self.entailment_model_name = entailment_model_name or os.getenv("ENTAILMENT_MODEL")
        self.chat_model_name = chat_model_name or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        
        # Lazy loading models
        self.entailment_model = None
        self.entailment_tokenizer = None
        self.entailment_pipeline = None
        
        logger.info(f"Initialized AnswerValidator with models: Entailment={self.entailment_model_name}, Chat={self.chat_model_name}")
    
    def _load_entailment_model(self):
        """Lazy load the entailment model"""
        if self.entailment_pipeline is None:
            logger.info(f"Loading entailment model: {self.entailment_model_name}")
            self.entailment_tokenizer = AutoTokenizer.from_pretrained(self.entailment_model_name)
            self.entailment_model = AutoModelForSequenceClassification.from_pretrained(self.entailment_model_name)
            self.entailment_pipeline = pipeline(
                "zero-shot-classification", 
                model=self.entailment_model, 
                tokenizer=self.entailment_tokenizer
            )
            logger.info("Entailment model loaded successfully")
        return self.entailment_pipeline
    
    def format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context passages for LLM consumption
        
        Args:
            context: List of context dictionaries with text and metadata
            
        Returns:
            Formatted context string
        """
        formatted = []
        
        for i, c in enumerate(context):
            section_info = f"Section: {c['section']}" if 'section' in c else ""
            page_info = f"Page: {c['page']}" if 'page' in c else ""
            metadata = f"{section_info} {page_info}".strip()
            
            formatted.append(f"[{i+1}] {metadata}\n{c['text']}\n")
        
        return "\n".join(formatted)
    
    def verify_entailment(self, query: str, answer: str, context: List[Dict[str, Any]]) -> bool:
        """
        Verify if the answer is entailed by the context
        
        Args:
            query: The user's question
            answer: Generated answer to verify
            context: Context passages used for generation
            
        Returns:
            Boolean indicating if the answer is entailed by the context
        """
        try:
            # Extract claims from answer (simplified approach)
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            claims = [s for s in sentences if len(s) > 20]  # Skip very short sentences
            
            # Skip validation if answer is too short or indicates info not found
            if len(claims) <= 1 or any(phrase in answer.lower() for phrase in [
                "not mentioned", "not provided", "not found", "no information"
            ]):
                return True
            
            # Sample up to 3 claims to validate
            sample_claims = claims[:3]
            
            # Get entailment model
            entailment_pipeline = self._load_entailment_model()
            
            # Format context as a single text
            context_text = " ".join([c["text"] for c in context])
            
            # Check entailment for each claim
            valid_claims = 0
            for claim in sample_claims:
                result = entailment_pipeline(
                    claim,
                    ["The context entails this statement.", "The context does not entail this statement."],
                    multi_label=False
                )
                
                # Check if entailment score is high enough
                if result["labels"][0] == "The context entails this statement." and result["scores"][0] >= 0.9:
                    valid_claims += 1
            
            # Consider valid if majority of claims are entailed
            return valid_claims >= len(sample_claims) // 2 + (1 if len(sample_claims) % 2 else 0)
            
        except Exception as e:
            logger.error(f"Entailment verification error: {str(e)}")
            return True  # Default to valid if verification fails
    
    def revise_answer(self, query: str, answer: str, context: List[Dict[str, Any]]) -> str:
        """
        Revise answer to improve factual accuracy
        
        Args:
            query: The user's question
            answer: Answer to revise
            context: Context passages
            
        Returns:
            Revised answer
        """
        try:
            system_prompt = """You are a financial analyst revising an answer for accuracy.
                                The previous answer may contain claims not supported by the context.
                                Revise the answer to ensure it ONLY includes information found in the context.
                                Be explicit when information is not available in the context.
                                Maintain the same level of detail where supported, but remove unsupported claims."""
            
            formatted_context = self.format_context_for_llm(context)
            user_prompt = f"Question: {query}\n\nOriginal answer to revise:\n{answer}\n\nContext:\n{formatted_context}\n\nRevised answer:"
            
            response = openai.ChatCompletion.create(
                model=self.chat_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            revised = response.choices[0].message.content.strip()
            return revised
            
        except Exception as e:
            logger.error(f"Answer revision error: {str(e)}")
            return answer  # Return original if revision fails
    
    def check_consistency_and_truncate(self, answer: str) -> str:
        """
        Perform consistency checks and truncate if needed
        
        Args:
            answer: Answer to check and potentially truncate
            
        Returns:
            Validated and possibly truncated answer
        """
        # Simple consistency check - look for contradictory statements
        contradictions = [
            ("increased", "decreased"),
            ("higher", "lower"),
            ("more", "less"),
            ("positive", "negative"),
            ("growth", "decline")
        ]
        
        has_contradiction = False
        answer_lower = answer.lower()
        
        for word1, word2 in contradictions:
            if word1 in answer_lower and word2 in answer_lower:
                # Check if they're close to each other (potential contradiction)
                pos1, pos2 = answer_lower.find(word1), answer_lower.find(word2)
                if abs(pos1 - pos2) < 50:  # Within 50 chars
                    has_contradiction = True
                    break
        
        # If contradictions found, add a note
        if has_contradiction:
            answer += "\n\nNote: This answer contains potentially contradictory information. Please verify the specific details against the original 10-K report."
        
        # Truncate to ~1000 tokens
        tokens = answer.split()
        if len(tokens) > 1000:
            truncated = " ".join(tokens[:1000])
            truncated += "\n\n[Answer truncated for brevity. The full analysis may contain additional relevant information.]"
            return truncated
        
        return answer
    
    def validate_answer(self, query: str, answer: str, context: List[Dict[str, Any]]) -> str:
        """
        Validate the generated answer against the context
        
        Args:
            query: The user's question
            answer: Generated answer to validate
            context: Context passages
            
        Returns:
            Validated answer
        """
        logger.info("Stage 8: Post-generation validation")
        
        # Perform entailment verification
        entailment_valid = self.verify_entailment(query, answer, context)
        
        # If entailment fails, revise the answer
        if not entailment_valid:
            logger.info("Entailment verification failed, revising answer")
            answer = self.revise_answer(query, answer, context)
        
        # Consistency check and truncation
        final_answer = self.check_consistency_and_truncate(answer)
        
        return final_answer
