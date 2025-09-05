import json
import os
from pathlib import Path

# annotations 디렉토리 생성
annotation_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\annotations")
annotation_dir.mkdir(exist_ok=True, parents=True)

# 두 번째 페이지 JSON
page_002_json = {
  "file_name": "20250807_175828_20240909162139-0001_page_002.png",
  "class": "purchase_order",
  "items": [
    {
      "group_id": "item_00001",
      "labels": [
        {
          "label": "Order number",
          "text": "4512221367",
          "bbox": [2163, 85, 220, 42]
        }
      ]
    },
    {
      "group_id": "item_00002",
      "labels": [
        {
          "label": "Case mark",
          "text": "YMG KOFU P/C K8-2 KOSUGE 731-48176",
          "bbox": [693, 1242, 731, 58]
        }
      ]
    },
    {
      "group_id": "item_00003",
      "labels": [
        {
          "label": "Shipping line",
          "text": "C5800002",
          "bbox": [886, 1332, 274, 116]
        }
      ]
    },
    {
      "group_id": "item_00004",
      "labels": [
        {
          "label": "Item number",
          "text": "00010",
          "bbox": [139, 1708, 108, 42]
        },
        {
          "label": "Part number",
          "text": "A1612JD-09",
          "bbox": [447, 1708, 256, 42]
        }
      ]
    },
    {
      "group_id": "item_00005",
      "labels": [
        {
          "label": "Delivery date",
          "text": "10-22-2024",
          "bbox": [823, 1806, 217, 42]
        },
        {
          "label": "Quantity",
          "text": "8.000 ST",
          "bbox": [1244, 1806, 170, 42]
        },
        {
          "label": "Unit price",
          "text": "3.9700",
          "bbox": [1723, 1806, 115, 42]
        }
      ]
    },
    {
      "group_id": "item_00006",
      "labels": [
        {
          "label": "Net amount (total)",
          "text": "31.76",
          "bbox": [2171, 2008, 83, 58]
        }
      ]
    }
  ]
}

# JSON 파일 저장
with open(annotation_dir / "20250807_175828_20240909162139-0001_page_002.json", 'w', encoding='utf-8') as f:
    json.dump(page_002_json, f, ensure_ascii=False, indent=2)

print("Created: 20250807_175828_20240909162139-0001_page_002.json")