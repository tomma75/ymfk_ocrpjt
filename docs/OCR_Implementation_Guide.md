# YOKOGAWA OCR Implementation Guide

## Overview

The OCR functionality has been successfully implemented in the YMFK_OCR system using Tesseract OCR engine. This guide describes the implementation details and usage.

## Implementation Details

### 1. OCR Components Added

#### Classes:
- **OCREngine** (Enum): Defines supported OCR engines (Tesseract, EasyOCR, PyTesseract)
- **OCRProcessor**: Main OCR processing class that handles text extraction

#### Key Methods:
- `extract_text()`: Extract text from images
- `extract_from_region()`: Extract text from specific image regions
- `batch_extract()`: Process multiple images in batch

### 2. OCR Configuration

The OCR functionality is configured through `ImageProcessingOptions`:

```python
# OCR Options
enable_ocr: bool = False
ocr_engine: OCREngine = OCREngine.TESSERACT
ocr_language: str = "eng+kor"  # English + Korean
ocr_config: str = "--psm 3"  # Page segmentation mode
ocr_confidence_threshold: float = 0.5
```

### 3. Tesseract Installation

Tesseract is installed at: `YMFK_OCR/tesseract/`
- Version: 5.5.0.20241111
- Languages: Over 100 languages including English (eng) and Korean (kor)
- Data files: Located in `tesseract/tessdata/`

## Usage Examples

### 1. Extract Text from Image

```python
from utils.image_processor import extract_text_from_image

# Extract text from a single image
result = extract_text_from_image(
    "path/to/image.png",
    language="eng+kor",  # English and Korean
    config="--psm 3"     # Automatic page segmentation
)

print(f"Extracted text: {result['text']}")
print(f"Confidence: {result['confidence']}")
print(f"Words found: {len(result['words'])}")
```

### 2. Extract Text from PDF

```python
from utils.image_processor import extract_text_from_pdf

# Extract text from all pages of a PDF
results = extract_text_from_pdf(
    "path/to/document.pdf",
    language="eng+kor"
)

for page_result in results:
    print(f"Page {page_result['page_number']}:")
    print(f"Text: {page_result['text'][:200]}...")
    print(f"Confidence: {page_result['confidence']:.2f}")
```

### 3. Batch Processing

```python
from utils.image_processor import batch_ocr_extract

# Process multiple files
files = ["file1.pdf", "file2.png", "file3.jpg"]
results = batch_ocr_extract(files, language="eng+kor")

for result in results:
    if result.get('success', False):
        print(f"{result['source_file']}: {len(result['text'])} characters extracted")
    else:
        print(f"{result['source_file']}: Failed - {result['error']}")
```

### 4. Check Tesseract Installation

```python
from utils.image_processor import check_tesseract_installation

info = check_tesseract_installation()
print(f"Installed: {info['installed']}")
print(f"Version: {info['version']}")
print(f"Languages: {', '.join(info['languages'][:5])}...")
```

## Integration with Image Processing Pipeline

The OCR functionality is fully integrated with the image processing pipeline:

1. **PDF Processing**: Automatically performs OCR on each page when `enable_ocr=True`
2. **Image Enhancement**: OCR is performed after image enhancement for better results
3. **Statistics**: OCR results include detailed statistics (word count, line count, confidence)

## OCR Result Structure

```python
{
    "text": str,              # Extracted text
    "confidence": float,      # Average confidence (0-100)
    "words": [                # Word-level details
        {
            "text": str,
            "confidence": float,
            "bbox": {         # Bounding box
                "x": int,
                "y": int,
                "width": int,
                "height": int
            }
        }
    ],
    "lines": [str],          # Text organized by lines
    "statistics": {
        "total_words": int,
        "total_lines": int,
        "total_characters": int,
        "average_confidence": float
    },
    "metadata": {
        "ocr_engine": str,
        "language": str,
        "config": str,
        "timestamp": str
    }
}
```

## Performance Considerations

1. **Language Selection**: Use only required languages (e.g., "eng" instead of "eng+kor") for faster processing
2. **Page Segmentation Mode**: 
   - PSM 3: Fully automatic page segmentation (default)
   - PSM 6: Uniform block of text
   - PSM 11: Sparse text
3. **Image Quality**: Pre-process images with enhancement options for better OCR results
4. **Batch Processing**: Use batch methods for processing multiple files efficiently

## Error Handling

The OCR system includes comprehensive error handling:
- Missing Tesseract installation detection
- Graceful fallback when pytesseract is not available
- Detailed error messages in results
- Automatic TESSDATA_PREFIX configuration

## Testing

Run the OCR test script:

```bash
python test_ocr.py
```

This will:
1. Check Tesseract installation
2. Find test PDF files
3. Perform OCR on each page
4. Display extracted text and statistics

## Troubleshooting

1. **"Tesseract not found" error**:
   - Ensure tesseract.exe exists in YMFK_OCR/tesseract/
   - Check TESSDATA_PREFIX environment variable

2. **Language not available**:
   - Verify language data file exists in tesseract/tessdata/
   - Use correct language code (e.g., "kor" not "korean")

3. **Low confidence scores**:
   - Enable image enhancement options
   - Adjust page segmentation mode
   - Ensure good image quality (300 DPI recommended)

## Future Enhancements

1. Support for additional OCR engines (EasyOCR)
2. Parallel processing for multiple pages
3. Custom training data for domain-specific text
4. Real-time OCR progress tracking
5. OCR result caching for repeated processing