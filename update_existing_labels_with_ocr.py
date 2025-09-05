#!/usr/bin/env python3
"""
기존 라벨 파일들에 OCR 데이터를 추가하는 스크립트
"""

import os
import sys
from pathlib import Path
import json
# tqdm이 없으면 간단한 진행 표시 사용
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        total = len(list(iterable))
        for i, item in enumerate(iterable):
            if desc:
                print(f"\r{desc}: {i+1}/{total}", end='')
            yield item
        if desc:
            print()  # 줄바꿈

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from services.ocr_integration_service import OCRIntegrationService
from config import settings

def update_all_labels_with_ocr():
    """모든 라벨 파일에 OCR 데이터 추가"""
    
    # OCR 통합 서비스 초기화
    ocr_service = OCRIntegrationService()
    
    # 라벨 디렉토리
    label_dir = Path(settings.PROCESSED_DATA_DIR) / "labels"
    image_dir = Path(settings.PROCESSED_DATA_DIR) / "images"
    
    # 모든 라벨 파일 찾기
    label_files = list(label_dir.glob("*_label.json"))
    
    print(f"총 {len(label_files)}개의 라벨 파일을 처리합니다.")
    
    updated_count = 0
    error_count = 0
    
    # 각 라벨 파일 처리
    for label_path in tqdm(label_files, desc="라벨 파일 처리중"):
        try:
            # 라벨 데이터 로드
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 이미 OCR 데이터가 있는지 확인
            has_ocr = False
            if 'bboxes' in label_data:
                for bbox in label_data['bboxes']:
                    if 'ocr_original' in bbox and bbox['ocr_original']:
                        has_ocr = True
                        break
            
            if has_ocr:
                print(f"\n{label_path.name}은 이미 OCR 데이터가 있습니다. 건너뜁니다.")
                continue
            
            # 해당하는 이미지 파일 찾기
            # 라벨 파일명에서 이미지 파일명 추출
            # 예: 20250811_174451_20241129171650-0001_page001_label.json
            # -> 20250811_174451_20241129171650-0001_page_001.png
            
            label_stem = label_path.stem.replace('_label', '')
            
            # 페이지 번호 형식 변환 (page001 -> page_001)
            if '_page' in label_stem:
                parts = label_stem.split('_page')
                page_num = parts[1]
                image_filename = f"{parts[0]}_page_{page_num}.png"
            else:
                image_filename = f"{label_stem}.png"
            
            image_path = image_dir / image_filename
            
            if not image_path.exists():
                # 다른 형식 시도
                image_filename = f"{label_stem}.png"
                image_path = image_dir / image_filename
                
                if not image_path.exists():
                    print(f"\n경고: {label_path.name}에 대한 이미지 파일을 찾을 수 없습니다.")
                    error_count += 1
                    continue
            
            # OCR 수행 및 라벨 업데이트
            print(f"\n처리중: {label_path.name}")
            updated_label_data = ocr_service.process_label_with_ocr(str(image_path), label_data)
            
            # 저장
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(updated_label_data, f, ensure_ascii=False, indent=2)
            
            updated_count += 1
            print(f"완료: {label_path.name}")
            
        except Exception as e:
            print(f"\n오류 발생 ({label_path.name}): {str(e)}")
            error_count += 1
            continue
    
    print(f"\n처리 완료!")
    print(f"- 업데이트된 파일: {updated_count}개")
    print(f"- 오류 발생: {error_count}개")
    print(f"- 건너뛴 파일 (이미 OCR 있음): {len(label_files) - updated_count - error_count}개")

def update_single_label_with_ocr(label_filename: str):
    """특정 라벨 파일에만 OCR 데이터 추가"""
    
    # OCR 통합 서비스 초기화
    ocr_service = OCRIntegrationService()
    
    # 파일 경로
    label_path = Path(settings.PROCESSED_DATA_DIR) / "labels" / label_filename
    
    if not label_path.exists():
        print(f"오류: {label_filename} 파일을 찾을 수 없습니다.")
        return
    
    # 라벨 데이터 로드
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    # 이미지 파일 찾기
    label_stem = label_path.stem.replace('_label', '')
    
    # 페이지 번호 형식 변환
    if '_page' in label_stem:
        parts = label_stem.split('_page')
        page_num = parts[1]
        image_filename = f"{parts[0]}_page_{page_num}.png"
    else:
        image_filename = f"{label_stem}.png"
    
    image_path = Path(settings.PROCESSED_DATA_DIR) / "images" / image_filename
    
    if not image_path.exists():
        print(f"오류: 이미지 파일 {image_filename}을 찾을 수 없습니다.")
        return
    
    # OCR 수행
    print(f"OCR 수행중: {label_filename}")
    updated_label_data = ocr_service.process_label_with_ocr(str(image_path), label_data)
    
    # 저장
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(updated_label_data, f, ensure_ascii=False, indent=2)
    
    print(f"완료: {label_filename}")
    
    # OCR 결과 요약 출력
    ocr_count = 0
    if 'bboxes' in updated_label_data:
        for bbox in updated_label_data['bboxes']:
            if bbox.get('ocr_original'):
                ocr_count += 1
                print(f"  - {bbox['label']}: '{bbox['ocr_original']}' -> '{bbox['text']}'")
    
    print(f"총 {ocr_count}개의 bbox에 OCR이 추가되었습니다.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='기존 라벨 파일에 OCR 데이터 추가')
    parser.add_argument('--file', type=str, help='특정 라벨 파일만 처리 (예: 20250811_174451_20241129171650-0001_page002_label.json)')
    parser.add_argument('--all', action='store_true', help='모든 라벨 파일 처리')
    
    args = parser.parse_args()
    
    if args.file:
        update_single_label_with_ocr(args.file)
    elif args.all:
        update_all_labels_with_ocr()
    else:
        print("사용법:")
        print("  python update_existing_labels_with_ocr.py --file <label_filename>")
        print("  python update_existing_labels_with_ocr.py --all")
        print("\n예시:")
        print("  python update_existing_labels_with_ocr.py --file 20250811_174451_20241129171650-0001_page002_label.json")
        print("  python update_existing_labels_with_ocr.py --all")