#!/usr/bin/env python3
"""
기존 라벨 JSON 파일을 그룹 기반 형식으로 변환하는 스크립트
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

def group_labels_by_row(bboxes: List[Dict], y_threshold: int = 50) -> List[Dict]:
    """
    Y좌표가 비슷한 라벨들을 같은 행(그룹)으로 묶기
    
    Args:
        bboxes: bbox 데이터 리스트
        y_threshold: 같은 행으로 판단할 Y좌표 차이 임계값
    
    Returns:
        그룹화된 아이템 리스트
    """
    if not bboxes:
        return []
    
    # Y좌표로 정렬
    sorted_bboxes = sorted(bboxes, key=lambda x: x.get('y', 0))
    
    groups = []
    current_group = []
    current_group_y = None
    group_id = 1
    
    for bbox in sorted_bboxes:
        bbox_y = bbox.get('y', 0)
        
        # Part number를 포함하는 라벨을 기준으로 그룹 시작
        if 'part number' in bbox.get('label', '').lower():
            # 이전 그룹이 있으면 저장
            if current_group:
                process_group(current_group, groups, group_id)
                group_id += 1
            
            # 새 그룹 시작
            current_group = [bbox]
            current_group_y = bbox_y
        elif current_group_y is not None and abs(bbox_y - current_group_y) <= y_threshold:
            # 같은 행의 라벨 추가
            current_group.append(bbox)
        elif current_group:
            # 그룹에 속하지만 Y좌표가 약간 다른 경우 (Shipping date, Quantity 등)
            # label이 관련 필드인지 확인
            related_labels = ['shipping date', 'quantity', 'unit price', 'fixed in usd']
            if any(label in bbox.get('label', '').lower() for label in related_labels):
                current_group.append(bbox)
        elif not current_group:
            # 그룹이 없는 독립적인 라벨 (Order number, Case mark 등)
            independent_group = {
                'group_id': f'header_{group_id:05d}',
                'type': 'header',
                'y_position': bbox_y,
                'labels': [{
                    'label': bbox.get('label', ''),
                    'text': bbox.get('text', ''),
                    'bbox': [bbox.get('x', 0), bbox.get('y', 0), 
                            bbox.get('width', 0), bbox.get('height', 0)]
                }]
            }
            groups.append(independent_group)
            group_id += 1
    
    # 마지막 그룹 처리
    if current_group:
        process_group(current_group, groups, group_id)
    
    # Net amount는 별도 그룹으로 처리
    for bbox in sorted_bboxes:
        if 'net amount' in bbox.get('label', '').lower():
            groups.append({
                'group_id': f'summary_{group_id:05d}',
                'type': 'summary',
                'y_position': bbox.get('y', 0),
                'labels': [{
                    'label': bbox.get('label', ''),
                    'text': bbox.get('text', ''),
                    'bbox': [bbox.get('x', 0), bbox.get('y', 0), 
                            bbox.get('width', 0), bbox.get('height', 0)]
                }]
            })
            break
    
    return groups

def process_group(group_bboxes: List[Dict], groups: List[Dict], group_id: int):
    """그룹 처리 및 정보 생성"""
    if not group_bboxes:
        return
    
    # X좌표로 정렬
    group_bboxes.sort(key=lambda x: x.get('x', 0))
    
    # 그룹 정보 생성
    group_info = {
        'group_id': f'item_{group_id:05d}',
        'type': 'item',
        'y_position': group_bboxes[0].get('y', 0),
        'labels': []
    }
    
    # 아이템 번호 찾기
    item_number = None
    for label in group_bboxes:
        if 'part number' in label.get('label', '').lower():
            # Part number에서 아이템 번호 추출 (예: "000010 A1612JD-09" -> "000010")
            text = label.get('text', '')
            parts = text.split()
            if parts and parts[0].isdigit():
                item_number = parts[0]
            break
    
    if item_number:
        group_info['item_number'] = item_number
    
    # 라벨 추가
    for label in group_bboxes:
        group_info['labels'].append({
            'label': label.get('label', ''),
            'text': label.get('text', ''),
            'bbox': [label.get('x', 0), label.get('y', 0), 
                    label.get('width', 0), label.get('height', 0)]
        })
    
    groups.append(group_info)

def convert_json_file(file_path: Path) -> None:
    """JSON 파일을 그룹 기반 형식으로 변환"""
    print(f"Processing: {file_path.name}")
    
    try:
        # 기존 JSON 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # bboxes가 있는 경우만 처리
        if 'bboxes' in data:
            # 그룹화 수행
            grouped_items = group_labels_by_row(data['bboxes'])
            
            # 새로운 형식으로 데이터 구조 변경
            data['items'] = grouped_items
            data['total_groups'] = len(grouped_items)
            
            # 그룹 정보 요약 추가
            item_groups = [g for g in grouped_items if g.get('type') == 'item']
            header_groups = [g for g in grouped_items if g.get('type') == 'header']
            summary_groups = [g for g in grouped_items if g.get('type') == 'summary']
            
            data['group_summary'] = {
                'total_items': len(item_groups),
                'total_headers': len(header_groups),
                'total_summaries': len(summary_groups),
                'item_numbers': [g.get('item_number', 'unknown') for g in item_groups]
            }
            
            # 백업 파일 생성
            backup_path = file_path.with_suffix('.json.backup')
            os.rename(file_path, backup_path)
            print(f"  - Backup created: {backup_path.name}")
            
            # 변환된 데이터 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  - Converted successfully!")
            print(f"  - Found {len(item_groups)} item groups")
            
        else:
            print(f"  - Skipped (no bboxes found)")
            
    except Exception as e:
        print(f"  - Error: {str(e)}")

def main():
    """메인 실행 함수"""
    # labels 폴더 경로
    labels_dir = Path(r"D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR\data\processed\labels")
    
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # 모든 JSON 파일 찾기
    json_files = list(labels_dir.glob("*.json"))
    
    # 백업 파일은 제외
    json_files = [f for f in json_files if not f.name.endswith('.backup')]
    
    print(f"Found {len(json_files)} JSON files to convert")
    print("=" * 50)
    
    # 각 파일 변환
    for json_file in json_files:
        convert_json_file(json_file)
        print("-" * 30)
    
    print("=" * 50)
    print("Conversion completed!")
    print("Original files have been backed up with .backup extension")

if __name__ == "__main__":
    main()