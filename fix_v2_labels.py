#!/usr/bin/env python3
"""
v2 라벨 파일 수정 스크립트

기존 v1 라벨에서 제대로 된 v2 파일을 생성
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent))

from services.advanced_model_service import LayoutAnalyzer, RelationshipAnalyzer, Entity
from config.settings import ApplicationConfig
from utils.logger_util import get_application_logger


def fix_v2_file(v1_file: Path, v2_file: Path):
    """v1 파일을 기반으로 v2 파일 수정"""
    logger = get_application_logger("fix_v2")
    
    # v1 데이터 로드
    with open(v1_file, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)
    
    # v2 데이터 로드 (있으면)
    if v2_file.exists():
        with open(v2_file, 'r', encoding='utf-8') as f:
            v2_data = json.load(f)
    else:
        # 새로 생성
        v2_data = create_v2_template(v1_data)
    
    # 분석기 초기화
    layout_analyzer = LayoutAnalyzer()
    relationship_analyzer = RelationshipAnalyzer()
    
    # 엔티티 생성
    entities = []
    entity_counter = 0
    
    # v1의 bboxes를 올바르게 엔티티로 변환
    for bbox_data in v1_data.get('bboxes', []):
        entity = Entity(
            entity_id=f"ent_{entity_counter:03d}",
            bbox={
                'x': bbox_data['x'],
                'y': bbox_data['y'],
                'width': bbox_data['width'],
                'height': bbox_data['height']
            },
            text=bbox_data.get('text', ''),
            label=bbox_data.get('label', 'unknown'),
            confidence=0.9  # 수동 라벨링이므로 높은 신뢰도
        )
        entities.append(entity)
        entity_counter += 1
    
    # 레이아웃 분석
    page_info = v2_data.get('page_info', {'width': 2480, 'height': 3508})
    layout = layout_analyzer.analyze_layout(entities, page_info)
    
    # 관계 분석
    relationships = relationship_analyzer.analyze_relationships(entities)
    
    # v2 데이터 업데이트
    v2_data['entities'] = []
    for entity in entities:
        entity_data = {
            "entity_id": entity.entity_id,
            "bbox": entity.bbox,
            "text": {
                "value": entity.text,
                "confidence": 0.95,  # OCR 신뢰도
                "alternatives": []
            },
            "label": {
                "primary": entity.label,
                "confidence": entity.confidence,
                "alternatives": []
            },
            "features": {
                "position": {
                    "x_normalized": entity.bbox['x'] / page_info['width'],
                    "y_normalized": entity.bbox['y'] / page_info['height'],
                    "region": _get_region(entity, layout),
                    "quadrant": _get_quadrant(entity, page_info)
                },
                "text_properties": _analyze_text_properties(entity.text),
                "visual_properties": {
                    "area": entity.bbox['width'] * entity.bbox['height'],
                    "aspect_ratio": entity.bbox['width'] / entity.bbox['height'] if entity.bbox['height'] > 0 else 1
                },
                "semantic_properties": _analyze_semantic_properties(entity.text, entity.label)
            },
            "relationships": relationships.get(entity.entity_id, []),
            "context": {},
            "validation": {
                "is_valid": True,
                "validation_rules": []
            },
            "ocr_results": {
                "tesseract_raw": bbox_data.get('ocr_original', entity.text),
                "tesseract_confidence": bbox_data.get('ocr_confidence', 0.9),
                "corrected_value": entity.text,
                "was_corrected": bbox_data.get('was_corrected', False)
            }
        }
        v2_data['entities'].append(entity_data)
    
    # 레이아웃 정보 업데이트
    v2_data['layout_analysis'] = layout
    
    # 그룹 정보 업데이트
    v2_data['groups'] = _create_groups(v1_data, entities)
    
    # 품질 메트릭 업데이트
    v2_data['quality_metrics'] = {
        'overall_confidence': 0.9,
        'completeness': len(entities) / 20,  # 예상 필드 수 대비
        'consistency': 0.95,
        'ocr_quality': {
            'average_confidence': 0.9,
            'low_confidence_entities': [],
            'failed_regions': []
        }
    }
    
    # 저장
    with open(v2_file, 'w', encoding='utf-8') as f:
        json.dump(v2_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Fixed v2 file: {v2_file}")
    return v2_data


def create_v2_template(v1_data):
    """v2 템플릿 생성"""
    return {
        "annotation_version": "2.0",
        "document_metadata": {
            "filename": v1_data.get('filename', ''),
            "filepath": v1_data.get('filepath', ''),
            "pageNumber": v1_data.get('pageNumber', 1),
            "document_type": "purchase_order",
            "document_class": v1_data.get('class', 'purchase_order'),
            "template_id": "yokogawa_po_template_v1",
            "language": "en",
            "ocr_engine": "tesseract_with_learning",
            "ocr_confidence": 0.9
        },
        "page_info": {
            "width": 2480,
            "height": 3508,
            "dpi": 300,
            "orientation": "portrait",
            "background_color": "white",
            "has_watermark": False
        },
        "layout_analysis": {},
        "entities": [],
        "groups": [],
        "templates": {
            "detected_template": "yokogawa_po_template",
            "template_confidence": 0.85,
            "expected_fields": []
        },
        "quality_metrics": {},
        "processing_info": {
            "processed_at": datetime.now().isoformat(),
            "processing_time_ms": 0,
            "annotator": "fixed_from_v1",
            "review_status": "completed",
            "version": 2
        }
    }


def _get_region(entity: Entity, layout: dict) -> str:
    """엔티티가 속한 영역 반환"""
    for region in layout.get('regions', []):
        if entity.entity_id in region.get('entities', []):
            return region['type']
    return 'body'


def _get_quadrant(entity: Entity, page_info: dict) -> str:
    """엔티티가 속한 사분면 반환"""
    x_mid = page_info['width'] / 2
    y_mid = page_info['height'] / 2
    
    if entity.x_center < x_mid:
        if entity.y_center < y_mid:
            return 'top_left'
        else:
            return 'bottom_left'
    else:
        if entity.y_center < y_mid:
            return 'top_right'
        else:
            return 'bottom_right'


def _analyze_text_properties(text: str) -> dict:
    """텍스트 속성 분석"""
    import re
    
    return {
        'length': len(text),
        'is_numeric': text.replace('.', '').replace(',', '').isdigit(),
        'is_date': bool(re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', text)),
        'is_uppercase': text.isupper(),
        'has_special_chars': bool(re.search(r'[!@#$%^&*()_+=]', text)),
        'word_count': len(text.split())
    }


def _analyze_semantic_properties(text: str, label: str) -> dict:
    """의미적 속성 분석"""
    properties = {
        'is_identifier': label in ['Order number', 'Item number', 'Part number', 'Shipping line'],
        'is_quantity': label in ['Quantity', 'Unit price', 'Net amount', 'Total'],
        'is_date': label in ['Date', 'Delivery date'],
        'is_descriptive': label in ['Case mark', 'Description', 'Branch']
    }
    return properties


def _create_groups(v1_data: dict, entities: List[Entity]) -> list:
    """그룹 정보 생성"""
    groups = []
    group_map = {}
    
    # v1의 items 정보를 사용하여 그룹 생성
    for item in v1_data.get('items', []):
        group_id = item.get('group_id', '-')
        if group_id != '-' and group_id not in group_map:
            group_type = 'item_line' if 'ITEM' in group_id else 'general'
            group = {
                'group_id': group_id,
                'group_type': group_type,
                'y_position': item.get('y_position', 0),
                'entities': [],
                'properties': {},
                'validation': {}
            }
            groups.append(group)
            group_map[group_id] = group
    
    # 엔티티를 그룹에 할당
    for i, bbox in enumerate(v1_data.get('bboxes', [])):
        group_id = bbox.get('group_id', '-')
        if group_id in group_map and i < len(entities):
            group_map[group_id]['entities'].append(entities[i].entity_id)
    
    return groups


def main():
    """메인 함수"""
    logger = get_application_logger("fix_v2")
    logger.info("Starting v2 label fix process...")
    
    # 디렉토리 설정
    labels_dir = Path("data/processed/labels")
    labels_v2_dir = Path("data/processed/labels_v2")
    labels_v2_dir.mkdir(exist_ok=True)
    
    # 모든 v1 파일 처리
    v1_files = list(labels_dir.glob("*_label.json"))
    logger.info(f"Found {len(v1_files)} v1 label files")
    
    fixed_count = 0
    for v1_file in v1_files:
        try:
            # v2 파일명 생성
            v2_filename = v1_file.stem + "_v2.json"
            v2_file = labels_v2_dir / v2_filename
            
            # 수정
            fix_v2_file(v1_file, v2_file)
            fixed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to fix {v1_file}: {e}")
    
    logger.info(f"Fixed {fixed_count}/{len(v1_files)} v2 files")
    print(f"\n✅ v2 파일 수정 완료: {fixed_count}개")


if __name__ == "__main__":
    main()