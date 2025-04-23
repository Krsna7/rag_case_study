"""
Document ingestion and processing module for the RAG pipeline.
Handles PDF extraction, text preprocessing, and document chunking.
"""

import logging
import re
from typing import Dict, List, Any, Tuple
import PyPDF2

logger = logging.getLogger("10K_RAG")

def extract_pdf_text(pdf_path: str) -> Dict[int, str]:
    """Extract text from PDF with page numbers"""
    text_by_page = {}
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_by_page[page_num + 1] = page.extract_text()
        
        logger.info(f"Successfully extracted text from {len(text_by_page)} pages")
        return text_by_page
    
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise

def preprocess_text(text_by_page: Dict[int, str]) -> str:
    """Preprocess the extracted text, strip boilerplate, etc."""
    processed_text = ""
    
    for page_num, text in sorted(text_by_page.items()):
        # Remove headers, footers, and page numbers (common boilerplate)
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            # Skip empty lines or lines that are just page numbers
            if not line.strip() or re.match(r'^\s*\d+\s*$', line):
                continue
            
            # Skip common footer patterns
            if re.search(r'(confidential|all rights reserved|©\d{4})', line, re.IGNORECASE):
                continue
            
            filtered_lines.append(line)
        
        processed_page = '\n'.join(filtered_lines)
        processed_text += f"\n--- Page {page_num} ---\n{processed_page}"
    
    logger.info("Text preprocessing complete")
    return processed_text

def split_into_sections(text: str) -> List[Dict[str, Any]]:
    """Split text into sections based on ALL-CAPS headings"""
    sections = []
    current_section = {"heading": "DOCUMENT_START", "content": "", "page": 1}
    current_page = 1
    
    lines = text.split('\n')
    
    # Regex for ALL-CAPS headings
    heading_pattern = r"^([A-Z][A-Z\s,&\-]{3,})$"
    page_pattern = r"^--- Page (\d+) ---$"
    
    for line in lines:
        # Check if this is a page marker
        page_match = re.match(page_pattern, line)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        
        # Check if this is a heading
        heading_match = re.match(heading_pattern, line)
        if heading_match:
            # If we already have content in the current section, save it
            if current_section["content"].strip():
                sections.append(current_section)
            
            # Start a new section
            current_section = {
                "heading": heading_match.group(1).strip(),
                "content": "",
                "page": current_page
            }
        else:
            # Add content to current section
            current_section["content"] += line + "\n"
    
    # Add the last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    logger.info(f"Split document into {len(sections)} sections")
    return sections

def create_parent_child_splits(sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]], List[Dict[str, Any]], List[str]]:
    """Create parent-child splits from sections"""
    min_tokens = 128
    max_tokens = 512
    stride = 64
    
    parent_chunks = []
    child_chunks = []
    parent_to_children = {}
    
    parent_id = 0
    child_id = 0
    
    for section in sections:
        # Create parent chunk
        parent_text = f"{section['heading']}\n{section['content']}"
        parent = {
            "id": f"p{parent_id}",
            "text": parent_text,
            "section": section['heading'],
            "page": section['page'],
            "tokens": len(parent_text.split())  # Token count
        }
        parent_chunks.append(parent)
        
        # Initialize children list for this parent
        parent_to_children[parent["id"]] = []
        
        # Create overlapping child chunks
        tokens = parent_text.split()
        start = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            
            # Ensure chunk meets minimum token requirement
            if end - start >= min_tokens or end == len(tokens):
                chunk_text = ' '.join(tokens[start:end])
                child = {
                    "id": f"c{child_id}",
                    "text": chunk_text,
                    "parent_id": parent["id"],
                    "section": section['heading'],
                    "page": section['page'],
                    "tokens": end - start
                }
                child_chunks.append(child)
                parent_to_children[parent["id"]].append(child["id"])
                child_id += 1
            
            # Move start pointer for next chunk with overlap
            start += stride
        
        parent_id += 1
    
    # Combine all chunks for indexing
    document_chunks = parent_chunks + child_chunks
    text_chunks = [chunk["text"] for chunk in document_chunks]
    
    logger.info(f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks")
    
    return parent_chunks, child_chunks, parent_to_children, document_chunks, text_chunks
