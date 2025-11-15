import PyPDF2
import re
import os
from typing import Optional


class SECDataLoader:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from locally downloaded PDF file"""
        try:
            if not os.path.exists(pdf_path):
                print(f"ERROR: PDF file does not exist: {pdf_path}")
                return ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                total_pages = len(pdf_reader.pages)
                
                print(f"Reading PDF with {total_pages} pages...")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    # Progress indicator
                    if (page_num + 1) % 50 == 0:
                        print(f"  Processed {page_num + 1}/{total_pages} pages...")
                
                print(f"✅ Successfully extracted text from {total_pages} pages")
                print(f"   Total text length: {len(text):,} characters")
                
                return text
                
        except Exception as e:
            print(f"ERROR: Failed to extract text from PDF: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        print("Cleaning extracted text...")
        original_length = len(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers (common patterns in SEC filings)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'-\s*\d+\s*-', '', text)  # Page numbers like "- 23 -"
        
        # Remove form headers
        text = re.sub(r'UNITED STATES SECURITIES AND EXCHANGE COMMISSION', '', text, flags=re.IGNORECASE)
        text = re.sub(r'FORM 10-K', '', text, flags=re.IGNORECASE)
        
        # Clean up multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove trailing/leading whitespace
        text = text.strip()
        
        print(f"✅ Text cleaned: {original_length:,} → {len(text):,} characters")
        
        return text
    
    def get_document_info(self, pdf_path: str) -> dict:
        """Get basic information about the PDF document"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = {
                    "file_path": pdf_path,
                    "file_size_mb": round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
                    "total_pages": len(pdf_reader.pages),
                    "has_metadata": bool(pdf_reader.metadata),
                    "metadata": pdf_reader.metadata or {}
                }
                
                return info
                
        except Exception as e:
            return {"error": str(e)}
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """Validate if PDF is readable and not corrupted"""
        try:
            if not os.path.exists(pdf_path):
                print(f"ERROR: PDF file does not exist: {pdf_path}")
                return False
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Basic checks
                if len(pdf_reader.pages) == 0:
                    print("ERROR: PDF has no pages")
                    return False
                
                # Try to extract text from first page
                first_page_text = pdf_reader.pages[0].extract_text()
                if not first_page_text or len(first_page_text.strip()) < 10:
                    print("WARNING: PDF appears to be scanned or has no extractable text")
                    return False
                
                print(f"✅ PDF validation successful: {len(pdf_reader.pages)} pages")
                return True
                
        except Exception as e:
            print(f"ERROR: PDF validation failed: {e}")
            return False