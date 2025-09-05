import json
import os
from pathlib import Path
import re

def create_standard_label(png_filename):
    """Create a standard label template for each PNG file"""
    
    # Parse filename
    parts = png_filename.split('_page_')
    if len(parts) != 2:
        return None
    
    base_name = parts[0]
    page_num = int(parts[1])
    pdf_name = base_name + ".pdf"
    
    # Create standard template based on typical document structure
    label_data = {
        "filename": pdf_name,
        "filepath": pdf_name,
        "pageNumber": page_num,
        "class": "purchase_order",
        "bboxes": [
            {
                "x": 2163,
                "y": 100,
                "width": 208,
                "height": 42,
                "label": "Order number",
                "text": "0000000000"  # Placeholder
            },
            {
                "x": 618,
                "y": 1380,
                "width": 350,
                "height": 43,
                "label": "Case mark",
                "text": "YMG KOFU"  # Placeholder
            },
            {
                "x": 158,
                "y": 1380,
                "width": 169,
                "height": 43,
                "label": "Shipping line",
                "text": "C0000000"  # Placeholder
            },
            {
                "x": 103,
                "y": 1700,
                "width": 256,
                "height": 42,
                "label": "Part number",
                "text": "A0000AA-00"  # Placeholder
            },
            {
                "x": 593,
                "y": 1810,
                "width": 196,
                "height": 52,
                "label": "Shipping date",
                "text": "01-01-2024"  # Placeholder
            },
            {
                "x": 1049,
                "y": 1810,
                "width": 178,
                "height": 58,
                "label": "Quantity",
                "text": "1.000 ST"  # Placeholder
            },
            {
                "x": 1689,
                "y": 1810,
                "width": 142,
                "height": 56,
                "label": "Unit price",
                "text": "0.0000"  # Placeholder
            },
            {
                "x": 1291,
                "y": 2115,
                "width": 180,
                "height": 46,
                "label": "Net amount (total)",
                "text": "Total 0.00"  # Placeholder
            }
        ],
        "items": [
            {
                "group_id": "header_00001",
                "type": "header",
                "y_position": 100,
                "labels": [
                    {
                        "label": "Order number",
                        "text": "0000000000",
                        "bbox": [2163, 100, 208, 42]
                    }
                ]
            },
            {
                "group_id": "header_00002",
                "type": "header",
                "y_position": 1380,
                "labels": [
                    {
                        "label": "Case mark",
                        "text": "YMG KOFU",
                        "bbox": [618, 1380, 350, 43]
                    },
                    {
                        "label": "Shipping line",
                        "text": "C0000000",
                        "bbox": [158, 1380, 169, 43]
                    }
                ]
            },
            {
                "group_id": "item_00003",
                "type": "item",
                "y_position": 1700,
                "labels": [
                    {
                        "label": "Part number",
                        "text": "A0000AA-00",
                        "bbox": [103, 1700, 256, 42]
                    },
                    {
                        "label": "Shipping date",
                        "text": "01-01-2024",
                        "bbox": [593, 1810, 196, 52]
                    },
                    {
                        "label": "Quantity",
                        "text": "1.000 ST",
                        "bbox": [1049, 1810, 178, 58]
                    },
                    {
                        "label": "Unit price",
                        "text": "0.0000",
                        "bbox": [1689, 1810, 142, 56]
                    }
                ]
            },
            {
                "group_id": "summary_00004",
                "type": "summary",
                "y_position": 2115,
                "labels": [
                    {
                        "label": "Net amount (total)",
                        "text": "Total 0.00",
                        "bbox": [1291, 2115, 180, 46]
                    }
                ]
            }
        ],
        "total_groups": 4,
        "group_summary": {
            "total_items": 1,
            "total_headers": 2,
            "total_summaries": 1,
            "item_numbers": ["unknown"]
        }
    }
    
    return label_data

def main():
    base_path = Path("D:/8.간접업무자동화/1. PO시트입력자동화/YMFK_OCR/data/processed")
    images_path = base_path / "images"
    labels_path = base_path / "labels"
    
    # Get all PNG files
    png_files = list(images_path.glob("*.png"))
    print(f"Total PNG files: {len(png_files)}")
    
    # Get existing labels
    existing_labels = set()
    for label_file in labels_path.glob("*_label.json"):
        label_name = label_file.stem.replace("_label", "")
        label_name = re.sub(r'_page(\d{3})', r'_page_\1', label_name)
        existing_labels.add(label_name)
    
    print(f"Existing labels: {len(existing_labels)}")
    
    # Process missing labels
    created_count = 0
    skipped_count = 0
    error_count = 0
    
    for png_file in png_files:
        png_name = png_file.stem
        
        # Skip if label exists
        if png_name in existing_labels:
            skipped_count += 1
            continue
        
        try:
            # Create label data
            label_data = create_standard_label(png_name)
            
            if label_data:
                # Create label filename
                label_filename = png_name.replace("_page_", "_page") + "_label.json"
                label_path = labels_path / label_filename
                
                # Save JSON
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, indent=2, ensure_ascii=False)
                
                created_count += 1
                if created_count % 50 == 0:
                    print(f"Progress: {created_count} labels created")
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"Error processing {png_name}: {e}")
    
    print(f"\nCompleted!")
    print(f"Labels created: {created_count}")
    print(f"Skipped (existing): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"\nNote: Created labels contain placeholder text and need manual review.")

if __name__ == "__main__":
    main()