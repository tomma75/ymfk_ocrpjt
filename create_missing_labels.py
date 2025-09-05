import json
import os
import re
from pathlib import Path
import cv2
import pytesseract
from typing import Dict, List, Tuple, Optional
import numpy as np

# Set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:\8.간접업무자동화\1. PO시트입력자동화\tesseract\tesseract.exe'

class LabelCreator:
    def __init__(self):
        self.base_path = Path("D:/8.간접업무자동화/1. PO시트입력자동화/YMFK_OCR/data/processed")
        self.images_path = self.base_path / "images"
        self.labels_path = self.base_path / "labels"
        
        # Standard Y coordinates and patterns
        self.y_coordinates = {
            "order_number": (70, 110),
            "case_mark": (350, 550),
            "shipping_line": (480, 550),
            "item_number": (640, 700),
            "part_number": (680, 740),
            "delivery_date": (680, 740),
            "quantity": (680, 740),
            "unit_price": (680, 740),
            "total": (780, 850)
        }
        
    def is_shipping_line(self, text: str) -> bool:
        """Check if text matches shipping line pattern (8-digit alphanumeric)"""
        # Pattern: Letter followed by 7 digits
        pattern1 = r'^[A-Z]\d{7}$'
        # Alternative pattern: 8 alphanumeric with at least one digit
        pattern2 = r'^[A-Z0-9]{8}$'
        
        text = text.strip()
        return bool(re.match(pattern1, text) or 
                   (re.match(pattern2, text) and any(c.isdigit() for c in text)))
    
    def is_case_mark(self, text: str) -> bool:
        """Check if text contains case mark indicators"""
        case_mark_keywords = [
            "YMG", "KOFU", "ISHIKAWA", "KOSUGE", "TOKYO",
            "EXT:", "LOC:", "K5", "K8", "P/C", "731-"
        ]
        return any(keyword in text.upper() for keyword in case_mark_keywords)
    
    def extract_text_from_image(self, image_path: Path) -> List[Dict]:
        """Extract text and coordinates from image using OCR"""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error reading image: {image_path}")
            return []
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get better OCR results
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        
        extracted_texts = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    extracted_texts.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    })
        
        return extracted_texts
    
    def classify_text(self, text: str, y_coord: int) -> Optional[str]:
        """Classify text based on content and position"""
        text = text.strip()
        
        # Order number pattern (10 digits)
        if re.match(r'^\d{10}$', text) and 70 <= y_coord <= 110:
            return "Order number"
        
        # Date pattern
        if re.match(r'\d{1,2}-\d{1,2}-\d{4}', text):
            return "Delivery date"
        
        # Quantity pattern
        if re.match(r'[\d,]+\.\d{3}\s*ST', text):
            return "Quantity"
        
        # Price pattern
        if re.match(r'^\d+\.\d{2,4}$', text):
            if y_coord < 750:
                return "Unit price"
            else:
                return "Net amount (total)"
        
        # Item number pattern
        if re.match(r'^\d{5}$', text) and 640 <= y_coord <= 700:
            return "Item number"
        
        # Shipping line check
        if self.is_shipping_line(text):
            return "Shipping line"
        
        # Case mark check
        if self.is_case_mark(text):
            return "Case mark"
        
        # Part number (alphanumeric codes)
        if re.match(r'^[A-Z]\d{4}[A-Z]{2}(-\d{2})?$', text):
            return "Part number"
        
        return None
    
    def create_label_json(self, image_path: Path) -> Dict:
        """Create label JSON for a single image"""
        # Extract filename info
        filename = image_path.stem  # Remove .png
        parts = filename.split('_page_')
        base_name = parts[0]
        page_num = int(parts[1]) if len(parts) > 1 else 1
        
        pdf_name = base_name + ".pdf"
        
        # Extract text from image
        ocr_results = self.extract_text_from_image(image_path)
        
        # Create base structure
        label_data = {
            "filename": pdf_name,
            "filepath": pdf_name,
            "pageNumber": page_num,
            "class": "purchase_order",
            "bboxes": [],
            "items": [],
            "total_groups": 0,
            "group_summary": {
                "total_items": 0,
                "total_headers": 0,
                "total_summaries": 0,
                "item_numbers": []
            }
        }
        
        # Process OCR results and classify
        classified_items = []
        for item in ocr_results:
            label = self.classify_text(item['text'], item['y'])
            if label:
                bbox_item = {
                    "x": item['x'],
                    "y": item['y'],
                    "width": item['width'],
                    "height": item['height'],
                    "label": label,
                    "text": item['text']
                }
                label_data["bboxes"].append(bbox_item)
                classified_items.append((label, bbox_item))
        
        # Group items by Y position
        self.group_items(label_data, classified_items)
        
        return label_data
    
    def group_items(self, label_data: Dict, classified_items: List):
        """Group items by their Y position"""
        if not classified_items:
            return
        
        # Sort by Y position
        classified_items.sort(key=lambda x: x[1]['y'])
        
        header_count = 0
        item_count = 0
        summary_count = 0
        current_group = None
        current_y = -1
        
        for label_type, bbox in classified_items:
            # Determine if we need a new group (Y difference > 20px)
            if current_y == -1 or abs(bbox['y'] - current_y) > 20:
                # Save previous group if exists
                if current_group:
                    label_data["items"].append(current_group)
                
                # Create new group
                if label_type in ["Order number", "Case mark", "Shipping line"]:
                    header_count += 1
                    group_type = "header"
                    group_id = f"header_{header_count:05d}"
                elif label_type == "Net amount (total)":
                    summary_count += 1
                    group_type = "summary"
                    group_id = f"summary_{summary_count:05d}"
                else:
                    item_count += 1
                    group_type = "item"
                    group_id = f"item_{item_count:05d}"
                    
                    # Add item number to summary
                    if label_type == "Item number":
                        label_data["group_summary"]["item_numbers"].append(bbox['text'])
                
                current_group = {
                    "group_id": group_id,
                    "type": group_type,
                    "y_position": bbox['y'],
                    "labels": []
                }
                current_y = bbox['y']
            
            # Add label to current group
            current_group["labels"].append({
                "label": label_type,
                "text": bbox['text'],
                "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
            })
        
        # Add last group
        if current_group:
            label_data["items"].append(current_group)
        
        # Update summary
        label_data["total_groups"] = len(label_data["items"])
        label_data["group_summary"]["total_items"] = item_count
        label_data["group_summary"]["total_headers"] = header_count
        label_data["group_summary"]["total_summaries"] = summary_count
    
    def process_missing_labels(self):
        """Process all images without labels"""
        # Get all PNG files
        png_files = set(p.stem for p in self.images_path.glob("*.png"))
        
        # Get existing labels (remove _label suffix and fix page numbering)
        existing_labels = set()
        for label_file in self.labels_path.glob("*_label.json"):
            label_name = label_file.stem.replace("_label", "")
            # Fix page numbering format
            label_name = re.sub(r'_page(\d{3})', r'_page_\1', label_name)
            existing_labels.add(label_name)
        
        # Find missing labels
        missing = png_files - existing_labels
        
        print(f"Found {len(missing)} images without labels")
        
        # Process each missing image
        for i, image_name in enumerate(missing, 1):
            if i % 10 == 0:
                print(f"Processing {i}/{len(missing)}...")
            
            image_path = self.images_path / f"{image_name}.png"
            
            # Create label
            try:
                label_data = self.create_label_json(image_path)
                
                # Fix the label filename format (pageXXX instead of page_XXX)
                label_filename = image_name.replace("_page_", "_page") + "_label.json"
                label_path = self.labels_path / label_filename
                
                # Save label
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue
        
        print("Label creation completed!")

if __name__ == "__main__":
    creator = LabelCreator()
    creator.process_missing_labels()