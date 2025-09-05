#!/usr/bin/env python3
"""
Shipping date를 Delivery date로 변경하는 스크립트
기존에 저장된 모든 라벨 데이터에서 Shipping date를 Delivery date로 변경합니다.
"""

import json
import os
from pathlib import Path

def update_label_files(data_dir):
    """라벨 파일들에서 Shipping date를 Delivery date로 변경"""
    
    # 처리할 디렉토리들
    directories = [
        Path(data_dir) / 'labels',
        Path(data_dir) / 'annotations', 
        Path(data_dir) / 'processed' / 'labels',
        Path(data_dir) / 'processed' / 'labels_v2',
        Path(data_dir) / 'models'
    ]
    
    total_files = 0
    updated_files = 0
    
    for directory in directories:
        if not directory.exists():
            print(f"디렉토리가 존재하지 않습니다: {directory}")
            continue
            
        print(f"\n처리 중: {directory}")
        
        # JSON 파일들 찾기
        json_files = list(directory.glob('*.json'))
        
        for json_file in json_files:
            total_files += 1
            file_updated = False
            
            try:
                # 파일 읽기
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 재귀적으로 Shipping date를 Delivery date로 변경
                def update_shipping_date(obj):
                    nonlocal file_updated
                    
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if key == 'label' and value == 'Shipping date':
                                obj[key] = 'Delivery date'
                                file_updated = True
                                print(f"  - {json_file.name}: 'Shipping date' → 'Delivery date'")
                            elif isinstance(value, (dict, list)):
                                update_shipping_date(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            update_shipping_date(item)
                
                update_shipping_date(data)
                
                # 변경사항이 있으면 파일 저장
                if file_updated:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    updated_files += 1
                    
            except json.JSONDecodeError:
                print(f"  ❌ JSON 파싱 오류: {json_file.name}")
            except Exception as e:
                print(f"  ❌ 오류 발생 ({json_file.name}): {e}")
    
    print(f"\n===== 처리 완료 =====")
    print(f"전체 파일 수: {total_files}")
    print(f"업데이트된 파일 수: {updated_files}")

def update_config_files():
    """설정 파일에서 Shipping date 제거"""
    
    config_file = Path('config/settings.json')
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # label_types에서 Shipping date 제거
            if 'label_types' in config:
                original_count = len(config['label_types'])
                config['label_types'] = [
                    label for label in config['label_types'] 
                    if label != 'Shipping date'
                ]
                
                # Delivery date가 없으면 추가
                if 'Delivery date' not in config['label_types']:
                    # Shipping line 다음에 Delivery date 추가
                    try:
                        shipping_line_idx = config['label_types'].index('Shipping line')
                        config['label_types'].insert(shipping_line_idx + 1, 'Delivery date')
                    except ValueError:
                        config['label_types'].append('Delivery date')
                
                if len(config['label_types']) != original_count:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    print(f"✅ config/settings.json 업데이트 완료")
                    
        except Exception as e:
            print(f"❌ config/settings.json 업데이트 실패: {e}")

if __name__ == "__main__":
    # 데이터 디렉토리 설정
    data_dir = 'data'
    
    print("=" * 50)
    print("Shipping date → Delivery date 변환 스크립트")
    print("=" * 50)
    
    # 라벨 파일들 업데이트
    update_label_files(data_dir)
    
    # 설정 파일 업데이트
    print("\n설정 파일 업데이트 중...")
    update_config_files()
    
    print("\n✅ 모든 변환 작업이 완료되었습니다!")