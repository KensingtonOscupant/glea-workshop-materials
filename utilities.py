"""Preprocessing helpers, lifted from the hv-praesenzen project.

process_pdf is the original verbatim, with one addition: an lru_cache, because the
replayed evaluations in Layer 3 run over the same ~100 PDFs three times and pdfplumber,
not the (replayed) LLM call, is what makes a run take a minute.
"""

import csv
import os
from functools import lru_cache
from pathlib import Path

import pdfplumber
import weave

GOLD_DIR = Path(__file__).resolve().parent / "data" / "gold"


def gold_split(split: str) -> list[dict]:
    """The corrected `split` from data/gold, shaped for the scorer.

    label_value is None where the report has no value to extract, which is what lets the
    scorer credit a -2 answer; label_present carries that distinction explicitly.
    """
    with open(GOLD_DIR / f"250802_{split}_set.csv") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "file_path": r["file_path"],
            "label_value": float(r["label_value"]) if r["label_value"] else None,
            "label_present": r["label_present"] == "True",
        }
        for r in rows
    ]


@weave.op()
@lru_cache(maxsize=None)
def process_pdf(file_path):
    # Handle case where file_path might be a dictionary
    if isinstance(file_path, dict):
        if 'file_path' not in file_path:
            return [], "Input dictionary missing 'file_path' key"
        actual_path = file_path['file_path']
    else:
        actual_path = file_path
    
    # Validate input type
    if not isinstance(actual_path, (str, Path)):
        return [], f"Invalid file path type: {type(actual_path)}. Expected string or Path object"
    
    # Convert to Path object for easier handling
    path_obj = Path(actual_path)
    
    # Check if file path exists
    if not path_obj.exists():
        return [], f"File does not exist: {actual_path}"
    
    # Check if path is actually a file (not a directory)
    if not path_obj.is_file():
        return [], f"Path is not a file: {actual_path}"
    
    # Check file extension
    if path_obj.suffix.lower() != '.pdf':
        return [], f"File is not a PDF (extension: {path_obj.suffix}): {actual_path}"
    
    # Check if file is empty
    if path_obj.stat().st_size == 0:
        return [], f"PDF file is empty: {actual_path}"
    
    # Check file permissions
    if not os.access(actual_path, os.R_OK):
        return [], f"No read permission for file: {actual_path}"
    
    try:
        with pdfplumber.open(actual_path) as pdf:
            # Check if PDF has any pages
            if len(pdf.pages) == 0:
                return [], f"PDF file contains no pages: {actual_path}"
            
            pages = []
            failed_pages = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text is not None and page_text.strip() != "":
                        pages.append(page_text)
                    # Note: We don't treat empty pages as errors, just skip them
                except Exception as page_error:
                    failed_pages.append(f"Page {page_num}: {str(page_error)}")
                    continue
            
            # If we couldn't extract text from any pages
            if len(pages) == 0 and len(failed_pages) > 0:
                return [], f"Failed to extract text from all pages in {actual_path}. Errors: {'; '.join(failed_pages)}"
            elif len(pages) == 0:
                return [], f"No text content found in PDF: {actual_path}"
            
            return pages, None
            
    except FileNotFoundError:
        return [], f"File not found during processing: {actual_path}"
    except PermissionError:
        return [], f"Permission denied when accessing file: {actual_path}"
    except pdfplumber.exceptions.PDFSyntaxError:
        return [], f"Invalid or corrupted PDF file: {actual_path}"
    except pdfplumber.exceptions.PasswordProtected:
        return [], f"PDF file is password protected: {actual_path}"
    except MemoryError:
        return [], f"Insufficient memory to process large PDF file: {actual_path}"
    except OSError as os_error:
        return [], f"Operating system error when accessing file {actual_path}: {str(os_error)}"
    except Exception as e:
        # Catch any other unexpected errors
        error_type = type(e).__name__
        return [], f"Unexpected error processing PDF {actual_path} ({error_type}): {str(e)}"
