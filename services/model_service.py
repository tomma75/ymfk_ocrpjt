#!/usr/bin/env python3
"""
YOKOGAWA OCR 모델 관리 서비스

자동 라벨 제안을 위한 모델 학습 및 예측 서비스
라벨링된 데이터를 활용하여 점진적으로 학습하고 예측 성능을 개선

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-01-07
"""

import json
import pickle
from pathlib import Path
try:
    import numpy as np
except ImportError:
    np = None
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    # scikit-learn이 설치되지 않은 경우 더미 클래스 사용
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            self.vocabulary_ = {}
        def fit_transform(self, X):
            return [[0] * len(X)]
        def transform(self, X):
            return [[0] * len(X)]
    
    class RandomForestClassifier:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            self.classes_ = list(set(y)) if y else []
        def predict(self, X):
            return [self.classes_[0] if self.classes_ else 'unknown'] * len(X)
        def predict_proba(self, X):
            return [[1.0] + [0.0] * (len(self.classes_) - 1)] * len(X)

from config.settings import ApplicationConfig
from core.base_classes import BaseService
import logging
from models.annotation_model import AnnotationModel, BoundingBox


class ModelService(BaseService):
    """모델 학습 및 예측 서비스"""
    
    def __init__(self, config: ApplicationConfig, logger: logging.Logger):
        """
        ModelService 초기화
        
        Args:
            config: 애플리케이션 설정
            logger: 로거 인스턴스
        """
        super().__init__(config, logger)
        self.model_directory = Path(config.model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장 경로
        self.label_classifier_path = self.model_directory / "label_classifier.pkl"
        self.text_vectorizer_path = self.model_directory / "text_vectorizer.pkl"
        self.bbox_predictor_path = self.model_directory / "bbox_predictor.pkl"
        
        # 모델 인스턴스
        self.label_classifier = None
        self.text_vectorizer = None
        self.bbox_predictor = None
        
        # 학습 데이터 통계
        self.training_stats = {
            'total_samples': 0,
            'label_distribution': {},
            'last_training_time': None,
            'model_version': '0.1.0'
        }
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self._logger.info("Initializing ModelService")
            
            # 기존 모델 로드 시도
            self._load_models()
            
            # 모델이 없으면 새로 생성
            if self.label_classifier is None:
                self._logger.info("Creating new models")
                self.text_vectorizer = TfidfVectorizer(max_features=1000)
                self.label_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                
            self._logger.info("ModelService initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize ModelService: {e}")
            return False
    
    def _load_models(self) -> None:
        """저장된 모델 로드"""
        try:
            if self.label_classifier_path.exists():
                with open(self.label_classifier_path, 'rb') as f:
                    self.label_classifier = pickle.load(f)
                self._logger.info("Label classifier loaded")
                
            if self.text_vectorizer_path.exists():
                with open(self.text_vectorizer_path, 'rb') as f:
                    self.text_vectorizer = pickle.load(f)
                self._logger.info("Text vectorizer loaded")
                
            # 학습 통계 로드
            stats_path = self.model_directory / "training_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.training_stats = json.load(f)
                    
        except Exception as e:
            self._logger.error(f"Error loading models: {e}")
    
    def _save_models(self) -> None:
        """모델 저장"""
        try:
            if self.label_classifier is not None:
                with open(self.label_classifier_path, 'wb') as f:
                    pickle.dump(self.label_classifier, f)
                self._logger.info("Label classifier saved")
                
            if self.text_vectorizer is not None:
                with open(self.text_vectorizer_path, 'wb') as f:
                    pickle.dump(self.text_vectorizer, f)
                self._logger.info("Text vectorizer saved")
                
            # 학습 통계 저장
            stats_path = self.model_directory / "training_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self._logger.error(f"Error saving models: {e}")
    
    def train_from_annotations(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        라벨링된 데이터로부터 모델 학습
        
        Args:
            annotations: 라벨링된 어노테이션 데이터 리스트
            
        Returns:
            학습 결과 통계
        """
        try:
            self._logger.info(f"Starting training with {len(annotations)} annotations")
            
            # 학습 데이터 준비
            texts = []
            labels = []
            
            for ann in annotations:
                if 'bboxes' in ann and ann['bboxes']:
                    for bbox in ann['bboxes']:
                        if bbox.get('text') and bbox.get('label'):
                            texts.append(bbox['text'])
                            labels.append(bbox['label'])
            
            if len(texts) < 10:
                self._logger.warning("Not enough training data")
                return {
                    'status': 'insufficient_data',
                    'message': 'At least 10 labeled samples required',
                    'sample_count': len(texts)
                }
            
            # 텍스트 벡터화
            if hasattr(self.text_vectorizer, 'vocabulary_'):
                # 기존 vocabulary에 추가
                X = self.text_vectorizer.transform(texts)
            else:
                # 새로 학습
                X = self.text_vectorizer.fit_transform(texts)
            
            # 라벨 분류기 학습
            self.label_classifier.fit(X, labels)
            
            # 학습 통계 업데이트
            self.training_stats['total_samples'] = len(texts)
            self.training_stats['last_training_time'] = datetime.now().isoformat()
            
            # 라벨 분포 계산
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            self.training_stats['label_distribution'] = label_counts
            
            # 모델 저장
            self._save_models()
            
            self._logger.info("Training completed successfully")
            return {
                'status': 'success',
                'total_samples': len(texts),
                'unique_labels': len(set(labels)),
                'label_distribution': label_counts
            }
            
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_labels(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OCR 결과에 대한 라벨 예측
        
        Args:
            ocr_results: OCR 결과 데이터
            
        Returns:
            예측된 라벨이 포함된 결과
        """
        try:
            if self.label_classifier is None or not hasattr(self.text_vectorizer, 'vocabulary_'):
                self._logger.warning("Model not trained yet")
                return ocr_results
            
            predictions = []
            
            for page_result in ocr_results:
                if 'words' in page_result:
                    for word in page_result['words']:
                        if word.get('text'):
                            try:
                                # 텍스트 벡터화
                                X = self.text_vectorizer.transform([word['text']])
                                
                                # 라벨 예측
                                predicted_label = self.label_classifier.predict(X)[0]
                                
                                # 예측 확률 계산
                                probabilities = self.label_classifier.predict_proba(X)[0]
                                max_prob = max(probabilities)
                                
                                # 예측 결과 추가
                                word['predicted_label'] = predicted_label
                                word['prediction_confidence'] = float(max_prob)
                                
                            except Exception as e:
                                self._logger.error(f"Prediction error for word: {e}")
                                word['predicted_label'] = None
                                word['prediction_confidence'] = 0.0
            
            return ocr_results
            
        except Exception as e:
            self._logger.error(f"Label prediction failed: {e}")
            return ocr_results
    
    def suggest_bboxes_with_labels(self, image_path: str, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        이미지에서 bbox와 라벨을 함께 제안
        
        Args:
            image_path: 이미지 경로
            ocr_results: OCR 결과
            
        Returns:
            제안된 bbox와 라벨 리스트
        """
        try:
            # OCR 결과에 라벨 예측 추가
            predicted_results = self.predict_labels(ocr_results)
            
            suggestions = []
            
            for page_idx, page_result in enumerate(predicted_results):
                if 'words' in page_result:
                    for word in page_result['words']:
                        if word.get('bbox') and word.get('predicted_label'):
                            suggestion = {
                                'page': page_idx + 1,
                                'x': word['bbox'].get('x', 0),
                                'y': word['bbox'].get('y', 0),
                                'width': word['bbox'].get('width', 100),
                                'height': word['bbox'].get('height', 30),
                                'text': word.get('text', ''),
                                'label': word['predicted_label'],
                                'confidence': word.get('prediction_confidence', 0.0),
                                'is_suggestion': True
                            }
                            suggestions.append(suggestion)
            
            # 신뢰도 기준으로 정렬
            suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return suggestions
            
        except Exception as e:
            self._logger.error(f"Bbox suggestion failed: {e}")
            return []
    
    def update_model_incrementally(self, new_annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        새로운 라벨링 데이터로 모델 점진적 업데이트
        
        Args:
            new_annotation: 새로 라벨링된 데이터
            
        Returns:
            업데이트 결과
        """
        try:
            # 기존 라벨링 데이터 로드
            labels_dir = Path(self._config.processed_data_directory) / 'labels'
            all_annotations = []
            
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.json'):
                    with open(label_file, 'r', encoding='utf-8') as f:
                        all_annotations.append(json.load(f))
            
            # 새 데이터 추가
            all_annotations.append(new_annotation)
            
            # 모델 재학습
            result = self.train_from_annotations(all_annotations)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Incremental update failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """모델 통계 정보 반환"""
        return {
            'is_trained': self.label_classifier is not None,
            'training_stats': self.training_stats,
            'model_files': {
                'classifier': self.label_classifier_path.exists(),
                'vectorizer': self.text_vectorizer_path.exists()
            }
        }
    
    def cleanup(self) -> None:
        """서비스 정리"""
        self._logger.info("Cleaning up ModelService")
        self._save_models()
        super().cleanup()
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            if not self._is_initialized:
                return False
            
            # 모델 디렉터리 접근 가능 여부 확인
            if not self.model_directory.exists():
                return False
            
            # 모델 파일 존재 여부 확인 (선택적)
            if self.label_classifier is not None:
                # 모델이 로드되어 있으면 정상
                return True
            
            # 모델이 없어도 서비스는 정상 (처음 실행 시)
            return True
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False