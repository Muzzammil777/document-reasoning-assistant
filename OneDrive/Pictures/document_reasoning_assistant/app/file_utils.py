import fitz
import docx
import os
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Extract text from various document formats"""
    
    def __init__(self):
        self.supported_formats = {".pdf", ".docx", ".txt"}
    
    def extract_text(self, file_path: str) -> Dict[str, any]:
        """Extract text from document based on file extension"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        try:
            if file_ext == ".pdf":
                return self._extract_pdf(file_path)
            elif file_ext == ".docx":
                return self._extract_docx(file_path)
            elif file_ext == ".txt":
                return self._extract_txt(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def _extract_pdf(self, file_path: str) -> Dict[str, any]:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(file_path)
        text_content = []
        metadata = {
            "total_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "file_type": "pdf"
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            if page_text.strip():  # Only add non-empty pages
                text_content.append({
                    "page_number": page_num + 1,
                    "text": page_text
                })
        
        doc.close()
        
        # Combine all text
        full_text = "\n\n".join([page["text"] for page in text_content])
        
        return {
            "text": full_text,
            "pages": text_content,
            "metadata": metadata
        }
    
    def _extract_docx(self, file_path: str) -> Dict[str, any]:
        """Extract text from DOCX using python-docx"""
        doc = docx.Document(file_path)
        
        paragraphs = []
        full_text_parts = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                paragraphs.append({
                    "paragraph_number": i + 1,
                    "text": paragraph.text
                })
                full_text_parts.append(paragraph.text)
        
        full_text = "\n\n".join(full_text_parts)
        
        # Extract basic metadata
        core_props = doc.core_properties
        metadata = {
            "title": core_props.title or "",
            "author": core_props.author or "",
            "total_paragraphs": len(paragraphs),
            "file_type": "docx"
        }
        
        return {
            "text": full_text,
            "paragraphs": paragraphs,
            "metadata": metadata
        }
    
    def _extract_txt(self, file_path: str) -> Dict[str, any]:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        metadata = {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "file_type": "txt"
        }
        
        return {
            "text": content,
            "lines": lines,
            "metadata": metadata
        }
    
    def validate_document(self, file_path: str) -> bool:
        """Validate if document can be processed"""
        try:
            result = self.extract_text(file_path)
            return len(result["text"].strip()) > 0
        except Exception as e:
            logger.error(f"Document validation failed: {str(e)}")
            return False
