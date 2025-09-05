#!/usr/bin/env python3
"""
고급 모델 학습 스크립트

v2 라벨 데이터를 사용하여 95% 정확도를 목표로 학습
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from config.settings import ApplicationConfig
from services.advanced_model_service import AdvancedModelService
from utils.logger_util import get_application_logger


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return get_application_logger("advanced_training")


def convert_v1_to_v2(v1_file: Path) -> Dict:
    """v1 형식을 v2 형식으로 변환"""
    with open(v1_file, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)
    
    # v2 형식으로 변환
    v2_data = {
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
            "ocr_confidence": 0.0
        },
        "page_info": {
            "width": 2480,
            "height": 3508,
            "dpi": 300,
            "orientation": "portrait",
            "background_color": "white",
            "has_watermark": False
        },
        "layout_analysis": {
            "layout_pattern": "standard",
            "layout_confidence": 0.8,
            "regions": [],
            "columns": []
        },
        "entities": [],
        "groups": [],
        "templates": {
            "detected_template": "yokogawa_po_template",
            "template_confidence": 0.85,
            "expected_fields": []
        },
        "quality_metrics": {
            "overall_confidence": 0.8,
            "completeness": 0.0,
            "consistency": 0.0,
            "ocr_quality": {
                "average_confidence": 0.8,
                "low_confidence_entities": [],
                "failed_regions": []
            }
        },
        "processing_info": {
            "processed_at": datetime.now().isoformat(),
            "processing_time_ms": 0,
            "annotator": "converted_from_v1",
            "review_status": "completed",
            "version": 1
        }
    }
    
    # 엔티티 변환
    entity_counter = 0
    groups = {}
    
    for item in v1_data.get('items', []):
        group_id = item.get('group_id', '-')
        
        for label_info in item.get('labels', []):
            entity_id = f"ent_{entity_counter:03d}"
            entity_counter += 1
            
            # bbox 변환
            bbox = {
                'x': label_info['bbox'][0],
                'y': label_info['bbox'][1],
                'width': label_info['bbox'][2],
                'height': label_info['bbox'][3]
            }
            
            # 엔티티 생성
            entity = {
                "entity_id": entity_id,
                "bbox": bbox,
                "text": {
                    "value": label_info.get('text', ''),
                    "confidence": label_info.get('ocr_confidence', 0.0),
                    "alternatives": []
                },
                "label": {
                    "primary": label_info.get('label', 'unknown'),
                    "confidence": 0.9,
                    "alternatives": []
                },
                "features": {
                    "position": {
                        "x_normalized": bbox['x'] / 2480,
                        "y_normalized": bbox['y'] / 3508,
                        "region": "",
                        "quadrant": ""
                    },
                    "text_properties": {},
                    "visual_properties": {},
                    "semantic_properties": {}
                },
                "relationships": [],
                "context": {},
                "validation": {
                    "is_valid": True,
                    "validation_rules": []
                },
                "ocr_results": {
                    "tesseract_raw": label_info.get('ocr_original', ''),
                    "tesseract_confidence": label_info.get('ocr_confidence', 0.0),
                    "corrected_value": label_info.get('text', ''),
                    "was_corrected": label_info.get('was_corrected', False)
                }
            }
            
            v2_data['entities'].append(entity)
            
            # 그룹 정보 수집
            if group_id != '-':
                if group_id not in groups:
                    groups[group_id] = {
                        'group_id': group_id,
                        'group_type': 'item_line' if 'ITEM' in group_id else 'general',
                        'y_position': item.get('y_position', bbox['y']),
                        'entities': [],
                        'properties': {},
                        'validation': {}
                    }
                groups[group_id]['entities'].append(entity_id)
    
    # 그룹 추가
    v2_data['groups'] = list(groups.values())
    
    # 품질 메트릭 업데이트
    if v2_data['entities']:
        avg_confidence = sum(e['text']['confidence'] for e in v2_data['entities']) / len(v2_data['entities'])
        v2_data['quality_metrics']['overall_confidence'] = avg_confidence
        v2_data['quality_metrics']['ocr_quality']['average_confidence'] = avg_confidence
    
    return v2_data


def prepare_training_data(logger):
    """학습 데이터 준비"""
    logger.info("Preparing training data...")
    
    # v2 라벨 디렉토리
    labels_v2_dir = Path("data/processed/labels_v2")
    v2_files = list(labels_v2_dir.glob("*.json"))
    
    logger.info(f"Found {len(v2_files)} v2 label files")
    
    # v1 라벨 중 v2가 없는 것들 변환
    labels_v1_dir = Path("data/processed/labels")
    v1_files = list(labels_v1_dir.glob("*.json"))
    
    converted_count = 0
    for v1_file in v1_files:
        # 대응하는 v2 파일 이름
        v2_filename = v1_file.stem + "_v2.json"
        v2_path = labels_v2_dir / v2_filename
        
        if not v2_path.exists():
            try:
                logger.info(f"Converting {v1_file.name} to v2 format...")
                v2_data = convert_v1_to_v2(v1_file)
                
                with open(v2_path, 'w', encoding='utf-8') as f:
                    json.dump(v2_data, f, ensure_ascii=False, indent=2)
                
                converted_count += 1
            except Exception as e:
                logger.error(f"Failed to convert {v1_file}: {e}")
    
    logger.info(f"Converted {converted_count} v1 files to v2 format")
    
    # 전체 v2 파일 목록
    all_v2_files = list(labels_v2_dir.glob("*.json"))
    logger.info(f"Total {len(all_v2_files)} v2 label files available for training")
    
    return all_v2_files


def analyze_training_data(v2_files, logger):
    """학습 데이터 분석"""
    logger.info("Analyzing training data...")
    
    total_entities = 0
    label_counts = {}
    template_counts = {}
    
    for v2_file in v2_files:
        try:
            with open(v2_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 엔티티 수 계산
            entities = data.get('entities', [])
            total_entities += len(entities)
            
            # 라벨 분포
            for entity in entities:
                label = entity.get('label', {}).get('primary', 'unknown')
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # 템플릿 분포
            template = data.get('templates', {}).get('detected_template', 'unknown')
            template_counts[template] = template_counts.get(template, 0) + 1
            
        except Exception as e:
            logger.error(f"Error analyzing {v2_file}: {e}")
    
    logger.info(f"\nData Analysis:")
    logger.info(f"- Total files: {len(v2_files)}")
    logger.info(f"- Total entities: {total_entities}")
    logger.info(f"- Average entities per file: {total_entities / len(v2_files):.1f}")
    
    logger.info(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {label}: {count}")
    
    logger.info(f"\nTemplate distribution:")
    for template, count in sorted(template_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {template}: {count}")
    
    return {
        'total_entities': total_entities,
        'label_counts': label_counts,
        'template_counts': template_counts
    }


def train_model(v2_files, logger):
    """모델 학습"""
    logger.info("\n=== Starting Advanced Model Training ===")
    
    # 설정 및 서비스 초기화
    config = ApplicationConfig()
    service = AdvancedModelService(config, logger)
    
    if not service.initialize():
        logger.error("Failed to initialize Advanced Model Service")
        return False
    
    # 학습 실행
    try:
        result = service.train_from_v2_labels(v2_files)
        
        if result['status'] == 'success':
            logger.info("\n✅ Training completed successfully!")
            logger.info(f"- Accuracy: {result['accuracy']:.2%}")
            logger.info(f"- Total samples: {result['total_samples']}")
            logger.info(f"- Cross-validation scores: {result['cross_val_scores']}")
            
            logger.info("\nLabel distribution in training:")
            for label, count in result['label_distribution'].items():
                logger.info(f"  - {label}: {count}")
            
            return True
        else:
            logger.error(f"Training failed: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False


def test_predictions(logger):
    """예측 테스트"""
    logger.info("\n=== Testing Predictions ===")
    
    # 테스트할 파일 선택
    test_image = Path("data/processed/images/20250811_174447_20241108174601-0001_page_006.png")
    
    if not test_image.exists():
        logger.warning("Test image not found")
        return
    
    # 더미 OCR 결과 (실제로는 OCR 서비스에서 가져와야 함)
    ocr_results = [{
        'page': 1,
        'words': [
            {'text': '4512365062', 'bbox': {'x': 2167, 'y': 54, 'width': 208, 'height': 42}},
            {'text': 'Purchase', 'bbox': {'x': 1034, 'y': 20, 'width': 200, 'height': 50}},
            {'text': 'Order', 'bbox': {'x': 1250, 'y': 20, 'width': 150, 'height': 50}},
            {'text': '11-08-2024', 'bbox': {'x': 2167, 'y': 100, 'width': 200, 'height': 40}},
            {'text': 'C5800002', 'bbox': {'x': 1956, 'y': 1067, 'width': 247, 'height': 131}},
            {'text': 'YMG', 'bbox': {'x': 1538, 'y': 1276, 'width': 100, 'height': 40}},
            {'text': 'KOFU', 'bbox': {'x': 1650, 'y': 1276, 'width': 120, 'height': 40}},
            {'text': 'K5-1', 'bbox': {'x': 1780, 'y': 1276, 'width': 100, 'height': 40}},
            {'text': '00010', 'bbox': {'x': 126, 'y': 1688, 'width': 110, 'height': 52}},
            {'text': 'W9117RD-B', 'bbox': {'x': 273, 'y': 1686, 'width': 250, 'height': 45}},
            {'text': '12-13-2024', 'bbox': {'x': 595, 'y': 1792, 'width': 200, 'height': 50}},
            {'text': '2.000', 'bbox': {'x': 1050, 'y': 1790, 'width': 100, 'height': 50}},
            {'text': 'ST', 'bbox': {'x': 1160, 'y': 1790, 'width': 50, 'height': 50}},
            {'text': '138.9900', 'bbox': {'x': 1700, 'y': 1795, 'width': 150, 'height': 50}},
            {'text': 'Total', 'bbox': {'x': 1900, 'y': 2300, 'width': 100, 'height': 40}},
            {'text': '277.98', 'bbox': {'x': 2100, 'y': 2300, 'width': 150, 'height': 40}}
        ]
    }]
    
    # 예측 실행
    config = ApplicationConfig()
    service = AdvancedModelService(config, logger)
    service.initialize()
    
    result = service.predict_with_confidence(str(test_image), ocr_results)
    
    logger.info(f"\nPrediction Results:")
    logger.info(f"- Overall confidence: {result.get('overall_confidence', 0):.2%}")
    logger.info(f"- Template detected: {result.get('template', 'None')}")
    
    logger.info(f"\nPredicted entities:")
    for entity in result.get('entities', []):
        logger.info(f"  {entity['text']} => {entity['label']} (confidence: {entity['confidence']:.2%})")


def main():
    """메인 함수"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("Advanced Model Training Script")
    logger.info("Target Accuracy: 95%")
    logger.info("=" * 80)
    
    # 1. 데이터 준비
    v2_files = prepare_training_data(logger)
    
    if len(v2_files) < 10:
        logger.error(f"Insufficient training data. Need at least 10 files, found {len(v2_files)}")
        return
    
    # 2. 데이터 분석
    stats = analyze_training_data(v2_files, logger)
    
    # 3. 모델 학습
    if train_model(v2_files, logger):
        # 4. 예측 테스트
        test_predictions(logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training script completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()