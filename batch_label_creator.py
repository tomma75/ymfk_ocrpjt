import json
import os
import re
from pathlib import Path
import cv2
import pytesseract
from typing import Dict, List, Tuple, Optional
import numpy as np

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'D:\8.간접업무자동화\1. PO시트입력자동화\tesseract\tesseract.exe'

class PurchaseOrderLabelCreator:
    """Purchase Order 및 Delivery Sheet 라벨 자동 생성 클래스"""
    
    def __init__(self):
        self.base_path = Path("D:/8.간접업무자동화/1. PO시트입력자동화/YMFK_OCR/data/processed")
        self.images_path = self.base_path / "images"
        self.labels_path = self.base_path / "labels"
        
        # 이미지 크기 표준
        self.IMAGE_WIDTH = 2481
        self.IMAGE_HEIGHT = 3510
        
        # Y 좌표 표준 범위 (라벨링 스크립트 기준)
        self.Y_RANGES = {
            "order_number": (70, 105),
            "case_mark": (1300, 1500),
            "shipping_line": (1300, 1500),
            "item_number": (1680, 1730),
            "part_number": (1690, 1730),
            "shipping_date": (1790, 1820),
            "quantity": (1795, 1820),
            "unit_price": (1795, 1820),
            "total": (2100, 2135)
        }
        
    def is_shipping_line(self, text: str) -> bool:
        """8자리 영숫자 패턴 확인 (Shipping line)"""
        text = text.strip().upper()
        # Pattern: [A-Z][0-9]{7} 또는 유사한 8자리 영숫자
        pattern1 = r'^[A-Z]\d{7}$'
        pattern2 = r'^[A-Z0-9]{8}$'
        
        if re.match(pattern1, text):
            return True
        if re.match(pattern2, text) and any(c.isdigit() for c in text):
            # 최소 하나 이상의 숫자가 포함된 8자리
            return True
        return False
    
    def is_case_mark(self, text: str) -> bool:
        """Case mark 패턴 확인 (주소/위치 정보)"""
        text = text.upper()
        case_keywords = [
            "YMG", "KOFU", "ISHIKAWA", "KOSUGE", "TOKYO",
            "EXT:", "LOC:", "K5", "K8", "K70", "P/C", "731-",
            "SHI", "YAMANASHI"
        ]
        
        # 키워드 포함 여부
        has_keyword = any(kw in text for kw in case_keywords)
        # 여러 단어로 구성 또는 특수 문자 포함
        is_complex = len(text.split()) > 1 or "-" in text or "/" in text or ":" in text
        
        return has_keyword or is_complex
    
    def extract_text_regions(self, image_path: Path) -> List[Dict]:
        """이미지에서 텍스트 영역 추출"""
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 이진화 처리
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # OCR 수행 - 상세 데이터 추출
        try:
            data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, lang='eng')
        except Exception as e:
            print(f"OCR 오류: {e}")
            return []
        
        # 텍스트 영역 추출
        n_boxes = len(data['text'])
        regions = []
        
        for i in range(n_boxes):
            # 신뢰도 30 이상인 텍스트만 추출
            if int(data['conf'][i]) > 30:
                text = data['text'][i].strip()
                if text and len(text) > 0:
                    regions.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': data['conf'][i]
                    })
        
        return regions
    
    def classify_and_create_bbox(self, text: str, x: int, y: int, width: int, height: int) -> Optional[Dict]:
        """텍스트를 분류하고 bbox 생성"""
        text = text.strip()
        
        # Order number: 10자리 숫자
        if re.match(r'^\d{10}$', text) and self.Y_RANGES["order_number"][0] <= y <= self.Y_RANGES["order_number"][1]:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Order number",
                "text": text
            }
        
        # Item number: 5자리 숫자 (00001, 00010 등)
        if re.match(r'^\d{5}$', text) and 1600 <= y <= 1800:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Item number",
                "text": text
            }
        
        # Date patterns (Shipping date or Delivery date)
        if re.match(r'\d{1,2}-\d{1,2}-\d{4}', text):
            label = "Delivery date" if y > 1750 else "Shipping date"
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": label,
                "text": text
            }
        
        # Quantity: X.XXX ST 형식
        if re.match(r'[\d,]+\.\d{3}\s*ST?', text):
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Quantity",
                "text": text
            }
        
        # Unit price: 숫자.숫자 형식
        if re.match(r'^\d+\.\d{2,4}$', text) and 1700 <= y <= 1900:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Unit price",
                "text": text
            }
        
        # Total amount
        if ('Total' in text or re.match(r'^\d+\.\d{2}$', text)) and y >= 2000:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Net amount (total)",
                "text": text
            }
        
        # Shipping line: 8자리 영숫자
        if self.is_shipping_line(text) and 1200 <= y <= 1600:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Shipping line",
                "text": text
            }
        
        # Case mark: 위치/주소 정보
        if self.is_case_mark(text) and 1200 <= y <= 1600:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Case mark",
                "text": text
            }
        
        # Part number: 영숫자 조합 (A1234BC-01 형식)
        if re.match(r'^[A-Z]\d{4}[A-Z]{2}(-\d{2})?$', text) and 1650 <= y <= 1800:
            return {
                "x": x, "y": y, "width": width, "height": height,
                "label": "Part number",
                "text": text
            }
        
        return None
    
    def create_groups(self, bboxes: List[Dict]) -> Tuple[List[Dict], Dict]:
        """bbox를 그룹으로 구성"""
        if not bboxes:
            return [], {"total_items": 0, "total_headers": 0, "total_summaries": 0, "item_numbers": []}
        
        # Y 좌표로 정렬
        bboxes.sort(key=lambda x: x['y'])
        
        groups = []
        header_count = 0
        item_count = 0
        summary_count = 0
        item_numbers = []
        
        current_group = None
        current_y = -1
        
        for bbox in bboxes:
            # 새 그룹이 필요한지 확인 (Y 차이 > 20px)
            if current_y == -1 or abs(bbox['y'] - current_y) > 20:
                # 이전 그룹 저장
                if current_group:
                    groups.append(current_group)
                
                # 그룹 타입 결정
                if bbox['label'] in ["Order number", "Case mark", "Shipping line"]:
                    header_count += 1
                    group_type = "header"
                    group_id = f"header_{header_count:05d}"
                elif bbox['label'] == "Net amount (total)":
                    summary_count += 1
                    group_type = "summary"
                    group_id = f"summary_{summary_count:05d}"
                else:
                    item_count += 1
                    group_type = "item"
                    group_id = f"item_{item_count:05d}"
                    
                    if bbox['label'] == "Item number":
                        item_numbers.append(bbox['text'])
                
                current_group = {
                    "group_id": group_id,
                    "type": group_type,
                    "y_position": bbox['y'],
                    "labels": []
                }
                current_y = bbox['y']
            
            # 현재 그룹에 라벨 추가
            current_group["labels"].append({
                "label": bbox['label'],
                "text": bbox['text'],
                "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
            })
        
        # 마지막 그룹 추가
        if current_group:
            groups.append(current_group)
        
        summary = {
            "total_items": item_count,
            "total_headers": header_count,
            "total_summaries": summary_count,
            "item_numbers": item_numbers if item_numbers else ["unknown"]
        }
        
        return groups, summary
    
    def create_label_json(self, image_path: Path) -> Optional[Dict]:
        """단일 이미지에 대한 라벨 JSON 생성"""
        # 파일명 파싱
        filename = image_path.stem
        parts = filename.split('_page_')
        if len(parts) != 2:
            print(f"잘못된 파일명 형식: {filename}")
            return None
        
        base_name = parts[0]
        page_num = int(parts[1])
        pdf_name = base_name + ".pdf"
        
        # 텍스트 영역 추출
        regions = self.extract_text_regions(image_path)
        
        # bbox 생성
        bboxes = []
        for region in regions:
            bbox = self.classify_and_create_bbox(
                region['text'], 
                region['x'], 
                region['y'],
                region['width'],
                region['height']
            )
            if bbox:
                bboxes.append(bbox)
        
        # 그룹 생성
        groups, group_summary = self.create_groups(bboxes)
        
        # 최종 JSON 구조
        label_data = {
            "filename": pdf_name,
            "filepath": pdf_name,
            "pageNumber": page_num,
            "class": "purchase_order",
            "bboxes": bboxes,
            "items": groups,
            "total_groups": len(groups),
            "group_summary": group_summary
        }
        
        return label_data
    
    def process_all_missing_labels(self):
        """모든 누락된 라벨 처리"""
        # PNG 파일 목록
        png_files = list(self.images_path.glob("*.png"))
        print(f"Total PNG files: {len(png_files)}")
        
        # Check existing labels
        existing_labels = set()
        for label_file in self.labels_path.glob("*_label.json"):
            # Convert pageXXX format to page_XXX for comparison
            label_name = label_file.stem.replace("_label", "")
            label_name = re.sub(r'_page(\d{3})', r'_page_\1', label_name)
            existing_labels.add(label_name)
        
        print(f"Existing labels: {len(existing_labels)}")
        
        # Files to process
        processed_count = 0
        error_count = 0
        
        for png_file in png_files:
            # Check page_XXX format
            png_name = png_file.stem
            
            # Check if label already exists
            if png_name in existing_labels:
                continue
            
            # Create label
            try:
                print(f"Processing: {png_name}")
                label_data = self.create_label_json(png_file)
                
                if label_data:
                    # 파일명 형식 조정 (page_XXX -> pageXXX)
                    label_filename = png_name.replace("_page_", "_page") + "_label.json"
                    label_path = self.labels_path / label_filename
                    
                    # JSON 저장
                    with open(label_path, 'w', encoding='utf-8') as f:
                        json.dump(label_data, f, indent=2, ensure_ascii=False)
                    
                    processed_count += 1
                    print(f"  [OK] Label created: {label_filename}")
                else:
                    error_count += 1
                    print(f"  [FAIL] Label creation failed")
                    
            except Exception as e:
                error_count += 1
                print(f"  [ERROR] Exception: {e}")
            
            # Progress update
            if processed_count % 10 == 0:
                print(f"\nProgress: {processed_count} files processed\n")
        
        print(f"\nCompleted!")
        print(f"Labels created: {processed_count}")
        print(f"Errors: {error_count}")

if __name__ == "__main__":
    creator = PurchaseOrderLabelCreator()
    creator.process_all_missing_labels()