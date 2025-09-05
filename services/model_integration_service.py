#!/usr/bin/env python3
"""
모델 통합 서비스

기존 모델과 고급 모델을 통합하여 최적의 결과를 제공
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import numpy as np
from config.settings import ApplicationConfig
from core.base_classes import BaseService
from services.model_service import ModelService
from services.advanced_model_service import AdvancedModelService
from services.hybrid_ocr_labeler import HybridOCRLabeler


class ModelIntegrationService(BaseService):
    """모델 통합 서비스"""
    
    def __init__(self, config: ApplicationConfig, logger: logging.Logger):
        super().__init__(config, logger)
        
        # 모델 서비스 초기화
        self.basic_model = ModelService(config, logger)
        self.advanced_model = AdvancedModelService(config, logger)
        self.hybrid_model = HybridOCRLabeler(config, logger)
        
        # 모델 활성화 상태
        self.use_advanced = True  # 기본적으로 고급 모델 사용
        self.use_hybrid = True  # 하이브리드 모델 우선 사용
        
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self._logger.info("Initializing Model Integration Service")
            
            # 기본 모델 초기화
            basic_init = self.basic_model.initialize()
            
            # 고급 모델 초기화
            advanced_init = self.advanced_model.initialize()
            
            # 하이브리드 모델 초기화
            hybrid_init = self.hybrid_model.initialize()
            
            if not basic_init and not advanced_init and not hybrid_init:
                self._logger.error("All models failed to initialize")
                self._is_initialized = False
                return False
            
            self._logger.info(f"Model status - Basic: {basic_init}, Advanced: {advanced_init}, Hybrid: {hybrid_init}")
            self._is_initialized = True
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Model Integration Service: {e}")
            self._is_initialized = False
            return False
    
    def train_models(self) -> Dict[str, Any]:
        """모든 모델 학습"""
        results = {}
        
        # v1 라벨로 기본 모델 학습
        self._logger.info("Training basic model with v1 labels...")
        labels_dir = Path(self._config.processed_data_directory) / 'labels'
        v1_annotations = []
        
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.json'):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        v1_annotations.append(json.load(f))
                except Exception as e:
                    self._logger.error(f"Error loading {label_file}: {e}")
        
        if v1_annotations:
            results['basic_model'] = self.basic_model.train_from_annotations(v1_annotations)
        
        # v2 라벨로 고급 모델 학습
        self._logger.info("Training advanced model with v2 labels...")
        labels_v2_dir = Path(self._config.processed_data_directory) / 'labels_v2'
        v2_files = list(labels_v2_dir.glob('*.json')) if labels_v2_dir.exists() else []
        
        if v2_files:
            results['advanced_model'] = self.advanced_model.train_from_v2_labels(v2_files)
            
            # 하이브리드 모델도 v2 라벨로 학습
            self._logger.info("Training hybrid model with v2 labels...")
            v2_data = []
            for v2_file in v2_files:
                try:
                    with open(v2_file, 'r', encoding='utf-8') as f:
                        v2_data.append(json.load(f))
                except Exception as e:
                    self._logger.error(f"Error loading {v2_file}: {e}")
            
            if v2_data:
                results['hybrid_model'] = self.hybrid_model.train(v2_data)
        
        return results
    
    def predict_labels(self, image_path: str, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """통합 라벨 예측"""
        try:
            # 하이브리드 모델 우선 시도
            if self.use_hybrid and self.hybrid_model:
                self._logger.info("Using hybrid model for prediction")
                
                # OCR 결과를 엔티티 형식으로 변환
                entities = []
                for i, ocr_item in enumerate(ocr_results):
                    # OCR 결과 구조에 맞게 bbox 정보 추출
                    bbox = {}
                    if 'bbox' in ocr_item:
                        bbox = ocr_item['bbox']
                    elif all(key in ocr_item for key in ['x', 'y', 'width', 'height']):
                        bbox = {
                            'x': ocr_item['x'],
                            'y': ocr_item['y'],
                            'width': ocr_item['width'],
                            'height': ocr_item['height']
                        }
                    elif all(key in ocr_item for key in ['left', 'top', 'width', 'height']):
                        bbox = {
                            'x': ocr_item['left'],
                            'y': ocr_item['top'],
                            'width': ocr_item['width'],
                            'height': ocr_item['height']
                        }
                    else:
                        # 기본값 설정
                        bbox = {
                            'x': ocr_item.get('left', 0),
                            'y': ocr_item.get('top', 0),
                            'width': ocr_item.get('width', 100),
                            'height': ocr_item.get('height', 30)
                        }
                    
                    entity = {
                        'entity_id': f"ent_{i:03d}",
                        'bbox': bbox,
                        'text': ocr_item.get('text', '')
                    }
                    entities.append(entity)
                
                # 하이브리드 모델로 예측
                predictions = self.hybrid_model.predict(entities)
                
                if predictions:
                    # 예측 결과를 v2 형식으로 변환
                    v2_entities = []
                    for pred in predictions:
                        v2_entity = {
                            'entity_id': pred['entity_id'],
                            'bbox': pred['bbox'],
                            'text': {'value': pred['text'], 'confidence': pred['confidence']},
                            'label': {
                                'primary': pred['predicted_label'],
                                'confidence': pred['confidence'],
                                'alternatives': pred.get('alternatives', [])
                            },
                            'model_confidences': pred.get('model_confidences', {})
                        }
                        v2_entities.append(v2_entity)
                    
                    return {
                        'model_used': 'hybrid',
                        'predictions': v2_entities,
                        'confidence': np.mean([p['confidence'] for p in predictions]),
                        'model_details': {
                            'rf': np.mean([p['model_confidences'].get('rf', 0) for p in predictions]),
                            'xgb': np.mean([p['model_confidences'].get('xgb', 0) for p in predictions]),
                            'lgbm': np.mean([p['model_confidences'].get('lgbm', 0) for p in predictions]),
                            'crf': np.mean([p['model_confidences'].get('crf', 0) for p in predictions])
                        }
                    }
            
            # 고급 모델 시도
            if self.use_advanced and hasattr(self.advanced_model, 'ensemble_model') and self.advanced_model.ensemble_model:
                self._logger.info("Using advanced model for prediction")
                result = self.advanced_model.predict_with_confidence(image_path, ocr_results)
                
                # 고급 모델 결과가 유효한 경우
                if result.get('entities') and result.get('overall_confidence', 0) > 0.5:
                    return {
                        'model_used': 'advanced',
                        'predictions': result['entities'],
                        'confidence': result['overall_confidence'],
                        'layout': result.get('layout', {}),
                        'template': result.get('template')
                    }
            
            # 기본 모델로 폴백
            self._logger.info("Using basic model for prediction")
            basic_results = self.basic_model.predict_labels(ocr_results, image_path)
            
            # 기본 모델 결과를 고급 형식으로 변환
            predictions = []
            for page_result in basic_results:
                for word in page_result.get('words', []):
                    if word.get('predicted_label'):
                        predictions.append({
                            'entity_id': f"ent_{len(predictions):03d}",
                            'bbox': word.get('bbox', {}),
                            'text': word.get('text', ''),
                            'label': word['predicted_label'],
                            'confidence': word.get('prediction_confidence', 0.5),
                            'template_match': False
                        })
            
            return {
                'model_used': 'basic',
                'predictions': predictions,
                'confidence': sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0,
                'layout': {},
                'template': None
            }
            
        except Exception as e:
            self._logger.error(f"Prediction failed: {e}")
            return {
                'model_used': 'none',
                'predictions': [],
                'confidence': 0,
                'error': str(e)
            }
    
    def suggest_labels(self, image_path: str, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """라벨 제안 (웹 인터페이스용)"""
        try:
            # 통합 예측 실행
            result = self.predict_labels(image_path, ocr_results)
            
            self._logger.info(f"Prediction result: model_used={result.get('model_used')}, predictions_count={len(result.get('predictions', []))}")
            
            # 중요한 라벨들만 필터링 (빠른라벨선택 항목들)
            important_labels = [
                'Order number', 'Shipping line', 'Case mark', 
                'Item number', 'Part number', 'Delivery date',
                'Quantity', 'Unit price', 'Net amount (total)'
            ]
            
            # 제안 형식으로 변환
            suggestions = []
            for i, pred in enumerate(result.get('predictions', [])):
                # 디버그: 각 예측의 라벨과 신뢰도 확인
                if i < 5:  # 처음 5개만 로그
                    self._logger.debug(f"Prediction {i}: label={pred.get('label')}, confidence={pred.get('confidence')}")
                
                # 중요 라벨이거나 신뢰도가 높은 경우 포함 (조건 완화)
                if (pred.get('label', 'unknown') in important_labels and pred.get('confidence', 0) > 0.3) or \
                   pred.get('confidence', 0) > 0.7:
                    
                    # 페이지 번호 추출 (OCR 결과에서)
                    page_num = 1  # 기본값
                    if ocr_results and len(ocr_results) > 0:
                        page_num = ocr_results[0].get('page', 1)
                    
                    suggestion = {
                        'page': page_num,
                        'x': pred['bbox'].get('x', 0),
                        'y': pred['bbox'].get('y', 0),
                        'width': pred['bbox'].get('width', 100),
                        'height': pred['bbox'].get('height', 30),
                        'text': pred.get('text', ''),
                        'label': pred.get('label', 'unknown'),
                        'confidence': pred.get('confidence', 0.0),
                        'is_suggestion': True,
                        'model_used': result.get('model_used', 'unknown'),
                        'group_id': pred.get('group_id', '-')
                    }
                    suggestions.append(suggestion)
            
            # 중복 제거 (같은 라벨과 비슷한 위치는 하나만)
            unique_suggestions = []
            seen_positions = set()
            
            for s in suggestions:
                # 위치 기반 중복 체크 (20픽셀 이내는 같은 위치로 간주)
                pos_key = f"{s['label']}_{s['x']//20}_{s['y']//20}"
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    unique_suggestions.append(s)
            
            suggestions = unique_suggestions
            
            # 신뢰도로 정렬
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 라벨별로 최대 2개씩만 유지
            label_counts = {}
            filtered_suggestions = []
            for s in suggestions:
                label = s['label']
                if label not in label_counts:
                    label_counts[label] = 0
                if label_counts[label] < 2:  # 각 라벨 유형당 최대 2개
                    filtered_suggestions.append(s)
                    label_counts[label] += 1
            
            suggestions = filtered_suggestions
            
            # 그룹핑 (기본 모델의 그룹핑 로직 활용)
            if result.get('model_used') == 'basic' and hasattr(self.basic_model, '_group_suggestions'):
                suggestions = self.basic_model._group_suggestions(suggestions)
            
            # hybrid 모델이 빈 결과를 반환하면 basic 모델로 재시도
            if len(suggestions) == 0 and result.get('model_used') == 'hybrid':
                self._logger.info("Hybrid model returned no suggestions, falling back to basic model")
                # 기본 모델로 직접 호출
                basic_results = self.basic_model.predict_labels(ocr_results, image_path)
                for page_idx, page_result in enumerate(basic_results):
                    current_page = page_result.get('page', page_idx + 1)
                    for word in page_result.get('words', []):
                        if word.get('predicted_label') and word.get('prediction_confidence', 0) > 0.3:
                            # bbox 정보 추출 (다양한 형식 지원)
                            bbox = word.get('bbox', {})
                            x = bbox.get('x', word.get('left', 0))
                            y = bbox.get('y', word.get('top', 0))
                            width = bbox.get('width', word.get('width', 100))
                            height = bbox.get('height', word.get('height', 30))
                            
                            suggestions.append({
                                'page': current_page,
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height,
                                'text': word.get('text', ''),
                                'label': word['predicted_label'],
                                'confidence': word.get('prediction_confidence', 0.5),
                                'is_suggestion': True,
                                'model_used': 'basic_fallback',
                                'group_id': word.get('group_id', '-')
                            })
                # 중복 제거
                unique_fallback = []
                seen_fallback = set()
                for s in suggestions:
                    pos_key = f"{s['label']}_{s['x']//20}_{s['y']//20}"
                    if pos_key not in seen_fallback:
                        seen_fallback.add(pos_key)
                        unique_fallback.append(s)
                
                # 신뢰도로 정렬
                unique_fallback.sort(key=lambda x: x['confidence'], reverse=True)
                
                # 중요 라벨 우선, 각 라벨당 최대 2개
                label_counts = {}
                filtered = []
                for s in unique_fallback:
                    if s['label'] in important_labels:
                        if s['label'] not in label_counts:
                            label_counts[s['label']] = 0
                        if label_counts[s['label']] < 2 and len(filtered) < 10:
                            filtered.append(s)
                            label_counts[s['label']] += 1
                suggestions = filtered
            
            self._logger.info(f"Generated {len(suggestions)} suggestions using {result.get('model_used')} model")
            return suggestions[:10]  # 최대 10개 제안
            
        except Exception as e:
            self._logger.error(f"Label suggestion failed: {e}")
            return []
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """통합 모델 통계"""
        stats = {
            'basic_model': self.basic_model.get_model_statistics(),
            'advanced_model': {
                'is_trained': hasattr(self.advanced_model, 'ensemble_model') and self.advanced_model.ensemble_model is not None,
                'training_stats': self.advanced_model.training_stats if hasattr(self.advanced_model, 'training_stats') else {}
            },
            'active_model': 'advanced' if self.use_advanced else 'basic'
        }
        
        return stats
    
    def set_active_model(self, model_type: str) -> bool:
        """활성 모델 설정"""
        if model_type == 'advanced':
            self.use_advanced = True
            self._logger.info("Switched to advanced model")
            return True
        elif model_type == 'basic':
            self.use_advanced = False
            self._logger.info("Switched to basic model")
            return True
        else:
            self._logger.error(f"Unknown model type: {model_type}")
            return False
    
    def reset_models(self) -> Dict[str, Any]:
        """모든 모델 초기화"""
        results = {}
        
        # 기본 모델 초기화
        basic_result = self.basic_model.reset_model()
        results['basic_model'] = basic_result
        
        # 고급 모델 초기화
        try:
            # 고급 모델 파일 삭제
            models_path = self.advanced_model.models_path
            if models_path.exists():
                for file in models_path.glob('*.pkl'):
                    file.unlink()
            
            # 모델 인스턴스 초기화
            self.advanced_model.ensemble_model = None
            self.advanced_model.text_vectorizer = None
            self.advanced_model.position_scaler = None
            self.advanced_model.training_stats = {
                'total_samples': 0,
                'accuracy': 0.0,
                'label_distribution': {},
                'template_distribution': {},
                'last_training_time': None,
                'model_version': '2.0.0'
            }
            
            results['advanced_model'] = {'status': 'success', 'message': 'Advanced model reset'}
        except Exception as e:
            results['advanced_model'] = {'status': 'error', 'message': str(e)}
        
        # 전체 결과 상태 결정
        if basic_result.get('status') == 'success' and results['advanced_model'].get('status') == 'success':
            results['status'] = 'success'
            results['message'] = 'All models reset successfully'
            results['backup_location'] = basic_result.get('backup_location')
            results['backed_up_files'] = basic_result.get('backed_up_files', [])
            results['removed_reports'] = basic_result.get('removed_reports', 0)
        else:
            results['status'] = 'partial_success'
            results['message'] = 'Some models reset successfully'
        
        return results
    
    def cleanup(self) -> None:
        """서비스 정리"""
        self._logger.info("Cleaning up Model Integration Service")
        self.basic_model.cleanup()
        self.advanced_model.cleanup()
        super().cleanup()
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            basic_health = self.basic_model.health_check()
            advanced_health = self.advanced_model.health_check()
            
            return basic_health or advanced_health
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False