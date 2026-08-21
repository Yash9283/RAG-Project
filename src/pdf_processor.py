import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pypdf import PdfReader

@dataclass
class DocumentPage:
    file_name: str
    file_path: str
    page_number: int
    total_pages: int
    quarter: str
    text: str

class PDFProcessor:
    """
    Extracts text page-by-page from quarterly financial PDF reports,
    attaching exact file name, page number, and quarter metadata.
    """
    
    @staticmethod
    def extract_quarter_from_filename(filename: str) -> str:
        """
        Infer quarter identifier (e.g. Q1 FY25, Q2 FY25) from filename.
        """
        match = re.search(r'(Q[1-4][_ -]?FY?\d{2,4})', filename, re.IGNORECASE)
        if match:
            return match.group(1).upper().replace('_', ' ').replace('-', ' ')
        
        # Secondary fallback pattern like Q1_2025 or Q1 2025
        match2 = re.search(r'(Q[1-4][_ -]?\d{4})', filename, re.IGNORECASE)
        if match2:
            return match2.group(1).upper().replace('_', ' ').replace('-', ' ')
            
        return "Unknown Quarter"

    def process_pdf(self, pdf_path: str) -> List[DocumentPage]:
        """
        Processes a single PDF file and returns a list of DocumentPage objects.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        file_name = path.name
        quarter = self.extract_quarter_from_filename(file_name)
        
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        pages_data: List[DocumentPage] = []
        
        for idx, page in enumerate(reader.pages):
            page_number = idx + 1
            extracted_text = page.extract_text() or ""
            # Basic cleanup of null chars or excessive whitespace
            clean_text = extracted_text.replace('\x00', '').strip()
            
            pages_data.append(
                DocumentPage(
                    file_name=file_name,
                    file_path=str(path.resolve()),
                    page_number=page_number,
                    total_pages=total_pages,
                    quarter=quarter,
                    text=clean_text
                )
            )
            
        return pages_data

    def process_directory(self, dir_path: str) -> List[DocumentPage]:
        """
        Processes all PDF files in a given directory.
        """
        directory = Path(dir_path)
        if not directory.exists():
            return []
            
        all_pages: List[DocumentPage] = []
        pdf_files = sorted(list(directory.glob("*.pdf")))
        for pdf_file in pdf_files:
            pages = self.process_pdf(str(pdf_file))
            all_pages.extend(pages)
            
        return all_pages
