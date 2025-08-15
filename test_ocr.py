#!/usr/bin/env python3
"""
Test script for OCR functionality
"""

import os
import sys
import json
from pathlib import Path

# Set TESSDATA_PREFIX environment variable
project_root = Path(__file__).parent
tesseract_path = project_root / "tesseract"
os.environ['TESSDATA_PREFIX'] = str(tesseract_path / "tessdata")

# Add project root to Python path
sys.path.insert(0, str(project_root))

from utils.image_processor import extract_text_from_pdf, check_tesseract_installation
from config.settings import load_configuration
from utils.logger_util import get_application_logger


def test_ocr():
    """Test OCR functionality"""
    print("YOKOGAWA OCR - Tesseract OCR Test")
    print("=" * 60)
    
    # Check Tesseract installation
    print("\n1. Checking Tesseract installation...")
    tesseract_info = check_tesseract_installation()
    print(f"   Installed: {tesseract_info['installed']}")
    print(f"   Path: {tesseract_info['path']}")
    print(f"   Version: {tesseract_info['version']}")
    print(f"   Languages: {', '.join(tesseract_info['languages']) if tesseract_info['languages'] else 'None detected'}")
    
    if not tesseract_info['installed']:
        print("\n[ERROR] Tesseract is not installed!")
        return
    
    # Find test PDF files
    print("\n2. Finding test PDF files...")
    test_pdf_dir = project_root / "data" / "raw"
    pdf_files = list(test_pdf_dir.glob("*.pdf"))[:2]  # Get first 2 PDFs
    
    if not pdf_files:
        print("   No PDF files found in data/raw directory")
        return
    
    print(f"   Found {len(pdf_files)} PDF files")
    
    # Test OCR on each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n3.{i} Testing OCR on: {pdf_path.name}")
        print("   " + "-" * 50)
        
        try:
            # Extract text from PDF
            results = extract_text_from_pdf(str(pdf_path))
            
            # Display results for each page
            for page_result in results:
                page_num = page_result.get('page_number', 0)
                text = page_result.get('text', '')
                confidence = page_result.get('confidence', 0)
                words = page_result.get('words', [])
                lines = page_result.get('lines', [])
                stats = page_result.get('statistics', {})
                
                print(f"\n   Page {page_num}:")
                print(f"   - Text length: {len(text)} characters")
                print(f"   - Words found: {len(words)}")
                print(f"   - Lines found: {len(lines)}")
                print(f"   - Average confidence: {confidence:.2f}")
                
                # Show first few lines of extracted text
                if lines:
                    print(f"\n   First 5 lines of text:")
                    for j, line in enumerate(lines[:5], 1):
                        print(f"   {j}. {line[:80]}{'...' if len(line) > 80 else ''}")
                
                # Show statistics
                if stats:
                    print(f"\n   Statistics:")
                    print(f"   - Total characters: {stats.get('total_characters', 0)}")
                    print(f"   - Total words: {stats.get('total_words', 0)}")
                    print(f"   - Total lines: {stats.get('total_lines', 0)}")
        
        except Exception as e:
            print(f"   [ERROR] OCR failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("OCR test completed!")


if __name__ == "__main__":
    test_ocr()