import json
import os
import re
from pathlib import Path
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\tesseract\tesseract.exe'

class LabelGenerator:
    def __init__(self):
        self.image_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\processed\images")
        self.annotation_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\annotations")
        self.annotation_dir.mkdir(exist_ok=True)
        
    def is_shipping_line(self, text):
        """8자리 영숫자 패턴 확인 - Shipping line 판별"""
        text = text.strip()
        # 패턴 1: 대문자로 시작, 숫자 7자리
        pattern1 = r'^[A-Z]\d{7}$'
        # 패턴 2: 영문자+숫자 조합 8자리
        pattern2 = r'^[A-Z0-9]{8}$'
        
        return bool(re.match(pattern1, text) or 
                   (re.match(pattern2, text) and any(c.isdigit() for c in text)))
    
    def is_case_mark(self, text):
        """Case mark 특징 확인"""
        case_mark_keywords = [
            "KOFU", "ISHIKAWA", "KOSUGE", "TOKYO", "YMG",
            "EXT:", "LOC:", "K5", "K8", "P/C", "731-"
        ]
        
        has_location_info = any(keyword in text for keyword in case_mark_keywords)
        is_multi_word = len(text.split()) > 1 or "-" in text or "/" in text
        
        return has_location_info or is_multi_word
    
    def extract_text_with_coordinates(self, image_path):
        """이미지에서 텍스트와 좌표 추출"""
        # 절대 경로 사용
        full_path = str(image_path).replace('\\', '/')
        img = cv2.imread(full_path)
        
        if img is None:
            print(f"Warning: Could not read image {full_path}")
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # OCR 수행
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, 
                                        config='--psm 6', lang='eng+kor')
        
        results = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # 신뢰도 30 이상
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    results.append({
                        'text': text,
                        'bbox': [x, y, w, h],
                        'conf': data['conf'][i]
                    })
        
        return results
    
    def classify_and_group_text(self, ocr_results):
        """OCR 결과를 라벨별로 분류하고 그룹화"""
        items = []
        group_counter = 1
        
        # Order number 찾기 (Y: 70-110 범위)
        order_numbers = [r for r in ocr_results if 70 <= r['bbox'][1] <= 110 and r['bbox'][0] > 2000]
        if order_numbers:
            items.append({
                "group_id": f"item_{group_counter:05d}",
                "labels": [{
                    "label": "Order number",
                    "text": order_numbers[0]['text'],
                    "bbox": order_numbers[0]['bbox']
                }]
            })
            group_counter += 1
        
        # Case mark와 Shipping line 찾기 (Y: 1200-1600 범위)
        case_shipping_texts = [r for r in ocr_results if 1200 <= r['bbox'][1] <= 1600]
        
        case_marks = []
        shipping_lines = []
        
        for text_data in case_shipping_texts:
            text = text_data['text']
            
            # 긴 텍스트를 공백으로 분리하여 검사
            parts = text.split()
            
            for i, part in enumerate(parts):
                if self.is_shipping_line(part):
                    # Shipping line 발견
                    shipping_lines.append({
                        "label": "Shipping line",
                        "text": part,
                        "bbox": text_data['bbox']  # 실제로는 부분 bbox 계산 필요
                    })
                    
                    # 나머지 텍스트가 있으면 Case mark로
                    remaining = ' '.join(parts[i+1:])
                    if remaining and self.is_case_mark(remaining):
                        case_marks.append({
                            "label": "Case mark", 
                            "text": remaining,
                            "bbox": text_data['bbox']  # 실제로는 부분 bbox 계산 필요
                        })
                elif self.is_case_mark(text):
                    case_marks.append({
                        "label": "Case mark",
                        "text": text,
                        "bbox": text_data['bbox']
                    })
        
        # Case mark와 Shipping line 그룹화
        if case_marks or shipping_lines:
            group_labels = []
            if shipping_lines:
                group_labels.append(shipping_lines[0])
            if case_marks:
                group_labels.append(case_marks[0])
                
            if group_labels:
                items.append({
                    "group_id": f"item_{group_counter:05d}",
                    "labels": group_labels
                })
                group_counter += 1
        
        # Item number와 Part number 찾기 (Y: 1680-1730)
        item_part_texts = [r for r in ocr_results if 1680 <= r['bbox'][1] <= 1730]
        
        item_numbers = []
        part_numbers = []
        
        for text_data in item_part_texts:
            text = text_data['text']
            # Item number는 보통 숫자
            if text.isdigit() and len(text) <= 3:
                item_numbers.append({
                    "label": "Item number",
                    "text": text,
                    "bbox": text_data['bbox']
                })
            # Part number는 영숫자 조합 (A로 시작하는 경우가 많음)
            elif re.match(r'^[A-Z]\d+', text):
                # SHISAKU 접두사 제거
                clean_text = re.sub(r'^SHISAKU-\d+_', '', text)
                part_numbers.append({
                    "label": "Part number",
                    "text": clean_text,
                    "bbox": text_data['bbox']
                })
        
        if item_numbers or part_numbers:
            group_labels = []
            if item_numbers:
                group_labels.append(item_numbers[0])
            if part_numbers:
                group_labels.append(part_numbers[0])
                
            if group_labels:
                items.append({
                    "group_id": f"item_{group_counter:05d}",
                    "labels": group_labels
                })
                group_counter += 1
        
        # Shipping date, Quantity, Unit price 찾기 (Y: 1790-1820)
        date_qty_price_texts = [r for r in ocr_results if 1790 <= r['bbox'][1] <= 1820]
        
        dates = []
        quantities = []
        prices = []
        
        for text_data in date_qty_price_texts:
            text = text_data['text']
            
            # 날짜 패턴 (DD-MM-YYYY)
            if re.match(r'\d{2}-\d{2}-\d{4}', text):
                dates.append({
                    "label": "Shipping date",
                    "text": text,
                    "bbox": text_data['bbox']
                })
            # 수량 패턴 (숫자 + ST)
            elif 'ST' in text and '.' in text:
                quantities.append({
                    "label": "Quantity",
                    "text": text,
                    "bbox": text_data['bbox']
                })
            # 가격 패턴 (소수점 포함 숫자)
            elif re.match(r'^\d+\.\d+$', text):
                prices.append({
                    "label": "Unit price",
                    "text": text,
                    "bbox": text_data['bbox']
                })
        
        if dates or quantities or prices:
            group_labels = []
            if dates:
                group_labels.append(dates[0])
            if quantities:
                group_labels.append(quantities[0])
            if prices:
                group_labels.append(prices[0])
                
            if group_labels:
                items.append({
                    "group_id": f"item_{group_counter:05d}",
                    "labels": group_labels
                })
                group_counter += 1
        
        # Total 찾기 (Y: 2100-2135)
        total_texts = [r for r in ocr_results if 2100 <= r['bbox'][1] <= 2135]
        
        for text_data in total_texts:
            if re.match(r'^\d+\.\d+$', text_data['text']):
                items.append({
                    "group_id": f"item_{group_counter:05d}",
                    "labels": [{
                        "label": "Net amount (total)",
                        "text": text_data['text'],
                        "bbox": text_data['bbox']
                    }]
                })
                group_counter += 1
                break
        
        return items
    
    def create_json_for_image(self, image_path):
        """이미지에 대한 JSON 생성"""
        try:
            # OCR 수행
            ocr_results = self.extract_text_with_coordinates(image_path)
            
            # 텍스트 분류 및 그룹화
            items = self.classify_and_group_text(ocr_results)
            
            # JSON 구조 생성
            json_data = {
                "file_name": image_path.name,
                "class": "purchase_order",
                "items": items
            }
            
            return json_data
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # 기본 템플릿 반환
            return {
                "file_name": image_path.name,
                "class": "purchase_order",
                "items": []
            }
    
    def generate_all_missing_jsons(self):
        """모든 PNG 파일에 대해 누락된 JSON 생성"""
        png_files = list(self.image_dir.glob("*.png"))
        
        print(f"Found {len(png_files)} PNG files")
        
        generated_count = 0
        for i, png_file in enumerate(png_files, 1):
            json_file = self.annotation_dir / f"{png_file.stem}.json"
            
            if not json_file.exists():
                print(f"[{i}/{len(png_files)}] Generating JSON for {png_file.name}...")
                
                json_data = self.create_json_for_image(png_file)
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                generated_count += 1
            else:
                print(f"[{i}/{len(png_files)}] JSON already exists for {png_file.name}")
        
        print(f"\nCompleted! Generated {generated_count} JSON files")
        return generated_count

if __name__ == "__main__":
    generator = LabelGenerator()
    generator.generate_all_missing_jsons()