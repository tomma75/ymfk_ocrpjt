import json
import os
from pathlib import Path
import random

class BatchJSONGenerator:
    def __init__(self):
        self.image_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\processed\images")
        self.annotation_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\annotations")
        self.annotation_dir.mkdir(exist_ok=True)
        
        # 샘플 데이터 템플릿
        self.order_numbers = ["4512221369", "4512221370", "4512221371", "4512221372", "4512221373"]
        self.shipping_lines = ["C5800002", "D4700001", "A1234567", "B9876543", "E3456789"]
        self.case_marks = [
            "YMG KOFU P/C K8-2 KOSUGE 731-48176",
            "ISHIKAWA EXT:731-48177LOC:K70-4",
            "YMG KOFU-SHI K5",
            "YMG KOFU K5-1 40KI 731-48229",
            "TOKYO WAREHOUSE LOC:A-23"
        ]
        self.part_numbers = [
            "A1511JD-09", "A6128HD-01", "A6075VD-11", "F9138YA", "A1126EB",
            "B2234CD-02", "C3345EF-03", "D4456GH-04", "E5567IJ-05"
        ]
        self.quantities = ["1.000", "2.000", "5.000", "8.000", "10.000", "30.000", "46.000", "100.000"]
        self.prices = ["0.0700", "0.5000", "1.7300", "12.3000", "46.7500", "96.7000", "141.9000"]
        
    def generate_template_json(self, png_file):
        """PNG 파일에 대한 템플릿 JSON 생성"""
        
        # 랜덤 데이터 선택
        order_num = random.choice(self.order_numbers)
        shipping_line = random.choice(self.shipping_lines)
        case_mark = random.choice(self.case_marks)
        part_number = random.choice(self.part_numbers)
        quantity = random.choice(self.quantities)
        price = random.choice(self.prices)
        
        # Total 계산
        total = float(quantity) * float(price)
        total_str = f"{total:.2f}"
        
        # 날짜 생성 (파일명에서 추출 시도)
        date_parts = png_file.stem.split('_')
        if len(date_parts) > 2:
            date_str = "10-22-2024"  # 기본값
        else:
            date_str = "10-22-2024"
        
        # JSON 구조 생성
        json_data = {
            "file_name": png_file.name,
            "class": "purchase_order",
            "items": [
                {
                    "group_id": "item_00001",
                    "labels": [
                        {
                            "label": "Order number",
                            "text": order_num,
                            "bbox": [2163, 100, 220, 42]
                        }
                    ]
                },
                {
                    "group_id": "item_00002",
                    "labels": [
                        {
                            "label": "Shipping line",
                            "text": shipping_line,
                            "bbox": [158, 1311, 169, 50]
                        },
                        {
                            "label": "Case mark",
                            "text": case_mark,
                            "bbox": [323, 1308, 578, 43]
                        }
                    ]
                },
                {
                    "group_id": "item_00003",
                    "labels": [
                        {
                            "label": "Item number",
                            "text": "00010",
                            "bbox": [139, 1700, 108, 42]
                        },
                        {
                            "label": "Part number",
                            "text": part_number,
                            "bbox": [447, 1700, 256, 42]
                        }
                    ]
                },
                {
                    "group_id": "item_00004",
                    "labels": [
                        {
                            "label": "Delivery date",
                            "text": date_str,
                            "bbox": [823, 1800, 217, 42]
                        },
                        {
                            "label": "Quantity",
                            "text": f"{quantity} ST",
                            "bbox": [1244, 1800, 170, 42]
                        },
                        {
                            "label": "Unit price",
                            "text": price,
                            "bbox": [1723, 1800, 115, 42]
                        }
                    ]
                },
                {
                    "group_id": "item_00005",
                    "labels": [
                        {
                            "label": "Net amount (total)",
                            "text": total_str,
                            "bbox": [2171, 2110, 83, 58]
                        }
                    ]
                }
            ]
        }
        
        return json_data
    
    def generate_all_jsons(self):
        """모든 PNG 파일에 대해 JSON 생성"""
        png_files = list(self.image_dir.glob("*.png"))
        
        print(f"Found {len(png_files)} PNG files")
        
        generated_count = 0
        skipped_count = 0
        
        for i, png_file in enumerate(png_files, 1):
            json_file = self.annotation_dir / f"{png_file.stem}.json"
            
            if json_file.exists():
                print(f"[{i}/{len(png_files)}] Skipping {png_file.name} - JSON already exists")
                skipped_count += 1
            else:
                print(f"[{i}/{len(png_files)}] Generating JSON for {png_file.name}...")
                
                json_data = self.generate_template_json(png_file)
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                generated_count += 1
                
                # 진행 상황 표시
                if generated_count % 10 == 0:
                    print(f"  Progress: Generated {generated_count} JSON files so far...")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total PNG files: {len(png_files)}")
        print(f"Generated: {generated_count} JSON files")
        print(f"Skipped (already exists): {skipped_count} files")
        print(f"Annotations saved to: {self.annotation_dir}")
        
        return generated_count

if __name__ == "__main__":
    generator = BatchJSONGenerator()
    generator.generate_all_jsons()