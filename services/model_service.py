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
import shutil
from pathlib import Path
try:
    import numpy as np
except ImportError:
    np = None
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
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
    
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X

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
        self.position_scaler_path = self.model_directory / "position_scaler.pkl"
        self.training_stats_path = self.model_directory / "training_stats.json"
        self.layout_patterns_path = self.model_directory / "layout_patterns.json"
        
        # 모델 인스턴스
        self.label_classifier = None
        self.text_vectorizer = None
        self.bbox_predictor = None
        self.position_scaler = None
        
        # 레이아웃 패턴 저장
        self.layout_patterns = {}
        self.document_classes = {}
        
        # 학습 데이터 통계
        self.training_stats = {
            'total_samples': 0,
            'label_distribution': {},
            'last_training_time': None,
            'model_version': '0.2.0',
            'layout_patterns': [],
            'document_types': []
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
                
            # 위치 스케일러 로드
            scaler_path = self.model_directory / "position_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.position_scaler = pickle.load(f)
                self._logger.info("Position scaler loaded")
                
            # 레이아웃 패턴 로드
            patterns_path = self.model_directory / "layout_patterns.json"
            if patterns_path.exists():
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    self.layout_patterns = json.load(f)
                self._logger.info("Layout patterns loaded")
                
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
                
            if self.position_scaler is not None:
                scaler_path = self.model_directory / "position_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.position_scaler, f)
                    
            # 레이아웃 패턴 저장
            patterns_path = self.model_directory / "layout_patterns.json"
            with open(patterns_path, 'w', encoding='utf-8') as f:
                json.dump(self.layout_patterns, f, ensure_ascii=False, indent=2)
                
            # 학습 통계 저장
            stats_path = self.model_directory / "training_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self._logger.error(f"Error saving models: {e}")
    
    def _save_training_stats(self) -> None:
        """학습 통계만 저장"""
        try:
            with open(self.training_stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_stats, f, ensure_ascii=False, indent=2)
            self._logger.info("Training stats saved")
        except Exception as e:
            self._logger.error(f"Error saving training stats: {e}")
    
    def _extract_features(self, bbox: Dict[str, Any], page_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        바운딩 박스에서 특징 추출
        
        Args:
            bbox: 바운딩 박스 정보
            page_info: 페이지 정보 (너비, 높이 등)
            
        Returns:
            추출된 특징 딕셔너리
        """
        text = bbox.get('text', '')
        x = bbox.get('x', 0)
        y = bbox.get('y', 0)
        width = bbox.get('width', 100)
        height = bbox.get('height', 30)
        
        # 페이지 크기 (기본값)
        page_width = page_info.get('width', 2480)
        page_height = page_info.get('height', 3508)
        
        # 텍스트 특징
        text_features = {
            'is_numeric': text.replace('.', '').replace(',', '').isdigit(),
            'has_dash': '-' in text,
            'has_colon': ':' in text,
            'text_length': len(text),
            'is_date_format': bool(re.match(r'\d{1,2}-\d{1,2}-\d{4}', text)),
            'is_order_number': bool(re.match(r'^\d{10}$', text)),
            'is_item_number': bool(re.match(r'^\d{5}$', text)),
            'contains_total': 'total' in text.lower(),
            'is_currency': bool(re.match(r'^\d+\.\d{2,4}$', text)),
        }
        
        # 위치 특징 (정규화)
        position_features = {
            'x_normalized': x / page_width,
            'y_normalized': y / page_height,
            'width_normalized': width / page_width,
            'height_normalized': height / page_height,
            'x_center': (x + width/2) / page_width,
            'y_center': (y + height/2) / page_height,
            'is_top_area': y < page_height * 0.1,
            'is_bottom_area': y > page_height * 0.8,
            'is_left_area': x < page_width * 0.3,
            'is_right_area': x > page_width * 0.7,
            'is_center_area': 0.3 < x/page_width < 0.7,
        }
        
        return {**text_features, **position_features}
    
    def _detect_layout_pattern(self, annotations: List[Dict[str, Any]]) -> str:
        """
        문서의 레이아웃 패턴 감지
        
        Args:
            annotations: 라벨링된 데이터
            
        Returns:
            패턴 식별자
        """
        # Shipping line 위치로 패턴 구분
        shipping_line_positions = []
        
        for ann in annotations:
            if 'bboxes' in ann:
                for bbox in ann['bboxes']:
                    if bbox.get('label') == 'Shipping line':
                        shipping_line_positions.append(bbox.get('x', 0))
        
        if not shipping_line_positions:
            return 'unknown'
        
        avg_x = sum(shipping_line_positions) / len(shipping_line_positions)
        
        # 패턴 분류
        if avg_x < 500:
            return 'pattern_A'  # 왼쪽 배치
        elif avg_x > 1800:
            return 'pattern_B'  # 오른쪽 배치
        else:
            return 'pattern_C'  # 중앙 배치
    
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
            features_list = []
            labels = []
            texts = []
            
            # 레이아웃 패턴 감지
            layout_patterns = {}
            
            for ann in annotations:
                if 'bboxes' in ann and ann['bboxes']:
                    # 페이지 정보 (기본값 사용)
                    page_info = {
                        'width': 2480,
                        'height': 3508
                    }
                    
                    # 문서 클래스 저장
                    doc_class = ann.get('class', '')
                    filename = ann.get('filename', '')
                    if filename:
                        self.document_classes[filename] = doc_class
                    
                    # 패턴 감지
                    pattern = self._detect_layout_pattern([ann])
                    if filename:
                        layout_patterns[filename] = pattern
                    
                    for bbox in ann['bboxes']:
                        if bbox.get('text') and bbox.get('label'):
                            # 특징 추출
                            features = self._extract_features(bbox, page_info)
                            features_list.append(features)
                            labels.append(bbox['label'])
                            texts.append(bbox['text'])
            
            if len(features_list) < 10:
                self._logger.warning("Not enough training data")
                return {
                    'status': 'insufficient_data',
                    'message': 'At least 10 labeled samples required',
                    'sample_count': len(features_list)
                }
            
            # 특징 벡터 준비
            if np is None:
                self._logger.error("NumPy not available")
                return {'status': 'error', 'message': 'NumPy required for advanced features'}
            
            # 수치 특징 추출
            numeric_features = []
            for feat in features_list:
                numeric_feat = [
                    float(feat.get('x_normalized', 0)),
                    float(feat.get('y_normalized', 0)),
                    float(feat.get('width_normalized', 0)),
                    float(feat.get('height_normalized', 0)),
                    float(feat.get('x_center', 0)),
                    float(feat.get('y_center', 0)),
                    float(feat.get('is_numeric', 0)),
                    float(feat.get('is_date_format', 0)),
                    float(feat.get('is_order_number', 0)),
                    float(feat.get('is_item_number', 0)),
                    float(feat.get('contains_total', 0)),
                    float(feat.get('is_currency', 0)),
                    float(feat.get('text_length', 0)),
                    float(feat.get('is_top_area', 0)),
                    float(feat.get('is_bottom_area', 0)),
                    float(feat.get('is_right_area', 0)),
                ]
                numeric_features.append(numeric_feat)
            
            X_numeric = np.array(numeric_features)
            
            # 텍스트 특징
            if not hasattr(self.text_vectorizer, 'vocabulary_') or len(self.text_vectorizer.vocabulary_) == 0:
                # 새로 학습
                X_text = self.text_vectorizer.fit_transform(texts)
            else:
                # 기존 vocabulary 사용
                X_text = self.text_vectorizer.transform(texts)
            
            # 위치 특징 스케일링
            if self.position_scaler is None:
                self.position_scaler = StandardScaler()
                X_numeric_scaled = self.position_scaler.fit_transform(X_numeric)
            else:
                X_numeric_scaled = self.position_scaler.transform(X_numeric)
            
            # 특징 결합
            if hasattr(X_text, 'toarray'):
                X_text_array = X_text.toarray()
            else:
                X_text_array = X_text
                
            X_combined = np.hstack([X_numeric_scaled, X_text_array])
            
            # 라벨 분류기 학습
            self.label_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.label_classifier.fit(X_combined, labels)
            
            # 레이아웃 패턴 저장
            self.layout_patterns = layout_patterns
            
            # 학습 통계 업데이트
            self.training_stats['total_samples'] = len(labels)
            self.training_stats['last_training_time'] = datetime.now().isoformat()
            self.training_stats['layout_patterns'] = list(set(layout_patterns.values()))
            self.training_stats['document_types'] = list(set(self.document_classes.values()))
            
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
                'total_samples': len(labels),
                'unique_labels': len(set(labels)),
                'label_distribution': label_counts,
                'layout_patterns': list(set(layout_patterns.values())),
                'document_types': list(set(self.document_classes.values()))
            }
            
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_labels(self, ocr_results: List[Dict[str, Any]], image_path: str = None) -> List[Dict[str, Any]]:
        """
        OCR 결과에 대한 라벨 예측
        
        Args:
            ocr_results: OCR 결과 데이터
            image_path: 이미지 경로 (레이아웃 패턴 추정용)
            
        Returns:
            예측된 라벨이 포함된 결과
        """
        try:
            if self.label_classifier is None or not hasattr(self.text_vectorizer, 'vocabulary_'):
                self._logger.warning("Model not trained yet")
                return ocr_results
            
            # 페이지 정보
            page_info = {
                'width': 2480,
                'height': 3508
            }
            
            # 레이아웃 패턴 추정 (파일명 기반)
            estimated_pattern = 'unknown'
            if image_path:
                filename = Path(image_path).stem
                for saved_filename, pattern in self.layout_patterns.items():
                    if filename in saved_filename:
                        estimated_pattern = pattern
                        break
            
            predictions = []
            
            for page_idx, page_result in enumerate(ocr_results):
                if 'words' in page_result:
                    for word in page_result['words']:
                        if word.get('text') and word.get('bbox'):
                            try:
                                # bbox 정보 구성
                                bbox = {
                                    'text': word['text'],
                                    'x': word['bbox'].get('x', 0),
                                    'y': word['bbox'].get('y', 0),
                                    'width': word['bbox'].get('width', 100),
                                    'height': word['bbox'].get('height', 30)
                                }
                                
                                # 특징 추출
                                features = self._extract_features(bbox, page_info)
                                
                                # 수치 특징
                                numeric_feat = [
                                    float(features.get('x_normalized', 0)),
                                    float(features.get('y_normalized', 0)),
                                    float(features.get('width_normalized', 0)),
                                    float(features.get('height_normalized', 0)),
                                    float(features.get('x_center', 0)),
                                    float(features.get('y_center', 0)),
                                    float(features.get('is_numeric', 0)),
                                    float(features.get('is_date_format', 0)),
                                    float(features.get('is_order_number', 0)),
                                    float(features.get('is_item_number', 0)),
                                    float(features.get('contains_total', 0)),
                                    float(features.get('is_currency', 0)),
                                    float(features.get('text_length', 0)),
                                    float(features.get('is_top_area', 0)),
                                    float(features.get('is_bottom_area', 0)),
                                    float(features.get('is_right_area', 0)),
                                ]
                                
                                if np is not None:
                                    X_numeric = np.array([numeric_feat])
                                    
                                    # 텍스트 특징
                                    X_text = self.text_vectorizer.transform([word['text']])
                                    
                                    # 위치 특징 스케일링
                                    if self.position_scaler:
                                        X_numeric_scaled = self.position_scaler.transform(X_numeric)
                                    else:
                                        X_numeric_scaled = X_numeric
                                    
                                    # 특징 결합
                                    if hasattr(X_text, 'toarray'):
                                        X_text_array = X_text.toarray()
                                    else:
                                        X_text_array = X_text
                                    
                                    X_combined = np.hstack([X_numeric_scaled, X_text_array])
                                    
                                    # 라벨 예측
                                    predicted_label = self.label_classifier.predict(X_combined)[0]
                                    
                                    # 예측 확률 계산
                                    probabilities = self.label_classifier.predict_proba(X_combined)[0]
                                    max_prob = max(probabilities)
                                    
                                    # 규칙 기반 보정 (텍스트 추가)
                                    features['text'] = word.get('text', '')
                                    predicted_label = self._apply_rules(predicted_label, features, estimated_pattern)
                                    
                                    # 예측 결과 추가
                                    word['predicted_label'] = predicted_label
                                    word['prediction_confidence'] = float(max_prob)
                                else:
                                    # NumPy 없을 때 간단한 예측
                                    word['predicted_label'] = 'unknown'
                                    word['prediction_confidence'] = 0.0
                                
                            except Exception as e:
                                self._logger.error(f"Prediction error for word: {e}")
                                word['predicted_label'] = None
                                word['prediction_confidence'] = 0.0
            
            return ocr_results
            
        except Exception as e:
            self._logger.error(f"Label prediction failed: {e}")
            return ocr_results
    
    def _apply_rules(self, predicted_label: str, features: Dict[str, Any], pattern: str) -> str:
        """
        규칙 기반 라벨 보정
        
        Args:
            predicted_label: 예측된 라벨
            features: 특징 정보
            pattern: 레이아웃 패턴
            
        Returns:
            보정된 라벨
        """
        # 텍스트 추출 (특징에서 텍스트 정보가 있다고 가정)
        text = features.get('text', '')
        
        # Order number는 항상 오른쪽 상단
        if features.get('is_order_number') and features.get('is_top_area') and features.get('is_right_area'):
            return 'Order number'
        
        # Total은 항상 오른쪽 하단
        if features.get('contains_total') and features.get('is_bottom_area') and features.get('is_right_area'):
            return 'Net amount (total)'
        
        # Shipping line 패턴 (C로 시작하고 숫자가 따라오는 경우)
        if re.match(r'^[A-Z]\d{7}', text) and features.get('is_right_area'):
            return 'Shipping line'
        
        # Case mark 패턴 (YMG, KOFU 등의 키워드 포함)
        case_mark_keywords = ['YMG', 'KOFU', 'AOKI', 'K5-']
        if any(keyword in text.upper() for keyword in case_mark_keywords):
            return 'Case mark'
        
        # 날짜 형식
        if features.get('is_date_format'):
            if features.get('y_normalized', 0) > 0.4:  # 중간 이하
                return 'Delivery date'
        
        # 5자리 숫자는 보통 Item number
        if features.get('is_item_number') and features.get('is_left_area'):
            return 'Item number'
        
        # Part number 패턴 (알파벳+숫자 조합)
        if re.match(r'^[A-Z]\d{4}[A-Z]{2}-[A-Z]$', text):
            return 'Part number'
        
        return predicted_label
    
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
            predicted_results = self.predict_labels(ocr_results, image_path)
            
            suggestions = []
            
            for page_idx, page_result in enumerate(predicted_results):
                # 페이지 번호를 OCR 결과에서 가져오거나 기본값 사용
                page_num = page_result.get('page', page_idx + 1)
                
                if 'words' in page_result:
                    for word in page_result['words']:
                        if word.get('bbox') and word.get('predicted_label'):
                            # 신뢰도가 낮은 예측은 제외 (임계값을 높임)
                            if word.get('prediction_confidence', 0) < 0.5:
                                continue
                                
                            # 불필요한 라벨 필터링
                            predicted_label = word.get('predicted_label', '')
                            if predicted_label in ['unknown', 'None', '']:
                                continue
                                
                            suggestion = {
                                'page': page_num,
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
            
            # 그룹핑 로직 추가
            suggestions = self._group_suggestions(suggestions)
            
            # 중요 라벨 필터링
            important_labels = [
                'Order number', 'Shipping line', 'Case mark', 
                'Item number', 'Part number', 'Delivery date',
                'Quantity', 'Unit price', 'Net amount (total)'
            ]
            
            # 중요 라벨만 유지하거나 신뢰도가 매우 높은 경우만 유지
            filtered_suggestions = []
            for suggestion in suggestions:
                if suggestion.get('label') in important_labels:
                    filtered_suggestions.append(suggestion)
                elif suggestion.get('confidence', 0) > 0.8:
                    filtered_suggestions.append(suggestion)
            
            # 신뢰도 기준으로 정렬
            filtered_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 최대 30개로 제한
            return filtered_suggestions[:30]
            
        except Exception as e:
            self._logger.error(f"Bbox suggestion failed: {e}")
            return []
    
    def _group_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        제안된 bbox들을 그룹으로 묶기
        
        Args:
            suggestions: 제안된 bbox 리스트
            
        Returns:
            그룹 정보가 추가된 bbox 리스트
        """
        if not suggestions:
            return suggestions
        
        # 페이지별로 그룹화
        pages = {}
        for suggestion in suggestions:
            page = suggestion.get('page', 1)
            if page not in pages:
                pages[page] = []
            pages[page].append(suggestion)
        
        # 각 페이지별로 그룹핑
        for page, page_suggestions in pages.items():
            # Y좌표로 정렬
            page_suggestions.sort(key=lambda x: (x.get('y', 0), x.get('x', 0)))
            
            # 같은 행의 bbox들을 그룹화
            y_threshold = 30  # 같은 행으로 판단할 Y좌표 차이
            group_counter = 1
            rows = []
            current_row = []
            last_y = None
            
            # 행 단위로 그룹화
            for suggestion in page_suggestions:
                current_y = suggestion.get('y', 0)
                
                if last_y is None or abs(current_y - last_y) <= y_threshold:
                    current_row.append(suggestion)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [suggestion]
                
                last_y = current_y
            
            if current_row:
                rows.append(current_row)
            
            # 각 행에 그룹 ID 할당
            for row in rows:
                # Item number가 있는 행은 해당 번호로 그룹화
                item_number = None
                for item in row:
                    if item.get('label') == 'Item number':
                        item_text = item.get('text', '').strip()
                        if item_text.isdigit():
                            item_number = item_text
                            break
                
                # 그룹 ID 할당
                if item_number:
                    # 5자리 숫자로 패딩 (예: 10 -> 00010)
                    padded_number = item_number.zfill(5)
                    group_id = f'ITEM_{padded_number}'
                else:
                    # 라벨 종류에 따라 그룹화
                    has_order = any(item.get('label') == 'Order number' for item in row)
                    has_shipping = any('shipping' in item.get('label', '').lower() for item in row)
                    has_amount = any('amount' in item.get('label', '').lower() or 
                                   'price' in item.get('label', '').lower() for item in row)
                    
                    if has_order:
                        group_id = 'ORDER_INFO'
                    elif has_shipping:
                        group_id = 'SHIPPING_INFO'
                    elif has_amount:
                        group_id = f'AMOUNT_{group_counter}'
                        group_counter += 1
                    else:
                        group_id = f'GROUP_{group_counter}'
                        group_counter += 1
                
                # 행의 모든 항목에 같은 그룹 ID 할당
                for item in row:
                    item['group_id'] = group_id
        
        return suggestions
    
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
    
    def reset_model(self) -> Dict[str, Any]:
        """모델 및 학습 데이터 초기화"""
        try:
            self._logger.info("Starting model reset...")
            
            # 백업 디렉터리 생성 시도
            backup_enabled = False
            backup_dir = None
            backed_up = []
            
            try:
                backup_dir = self.model_directory / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_enabled = True
            except (PermissionError, OSError) as e:
                self._logger.warning(f"Cannot create backup directory: {e}. Proceeding without backup.")
            
            # 기존 모델 파일 처리
            model_files = [
                self.label_classifier_path,
                self.text_vectorizer_path,
                self.position_scaler_path,
                self.training_stats_path,
                self.layout_patterns_path
            ]
            
            for file_path in model_files:
                if file_path.exists():
                    try:
                        # 백업이 가능한 경우에만 백업
                        if backup_enabled and backup_dir:
                            backup_path = backup_dir / file_path.name
                            shutil.copy2(file_path, backup_path)
                            backed_up.append(file_path.name)
                        
                        # 권한 문제 방지를 위해 안전하게 삭제
                        try:
                            file_path.unlink()
                        except (PermissionError, OSError):
                            # 권한 문제 시 파일 내용만 초기화
                            try:
                                with open(file_path, 'wb') as f:
                                    f.truncate(0)
                            except:
                                pass  # 파일 접근 불가시 무시
                    except Exception as e:
                        self._logger.warning(f"Could not process {file_path.name}: {e}")
                        # 파일 처리 실패는 무시하고 계속
            
            # OCR 보정 데이터 백업 및 초기화
            ocr_corrections_dir = self.model_directory / 'ocr_corrections'
            learning_reports = []
            removed_reports = 0
            
            if ocr_corrections_dir.exists() and backup_enabled and backup_dir:
                try:
                    # 백업
                    ocr_backup_dir = backup_dir / 'ocr_corrections'
                    ocr_backup_dir.mkdir(exist_ok=True)
                    
                    for file in ocr_corrections_dir.iterdir():
                        if file.is_file() and file.name not in ['accuracy_metrics.json', 'correction_history.json', 'learning_report_*.json']:
                            # 기타 파일들은 백업만
                            shutil.copy2(file, ocr_backup_dir / file.name)
                        elif file.name in ['accuracy_metrics.json', 'correction_history.json']:
                            # 중요 파일은 백업 후 삭제
                            try:
                                shutil.copy2(file, ocr_backup_dir / file.name)
                                try:
                                    file.unlink()
                                except PermissionError:
                                    # 권한 문제 시 파일 내용만 초기화
                                    with open(file, 'w') as f:
                                        json.dump({}, f)
                                backed_up.append(f'ocr_corrections/{file.name}')
                            except Exception as e:
                                self._logger.warning(f"Could not backup/remove {file.name}: {e}")
                    
                    # 모든 learning_report 파일 백업 후 삭제
                    learning_reports = list(ocr_corrections_dir.glob('learning_report_*.json'))
                    removed_reports = len(learning_reports)
                    for report in learning_reports:
                        try:
                            # 백업
                            shutil.copy2(report, ocr_backup_dir / report.name)
                            # 삭제
                            try:
                                report.unlink()
                            except PermissionError:
                                # 권한 문제 시 파일 내용만 초기화
                                with open(report, 'w') as f:
                                    json.dump({}, f)
                        except Exception as e:
                            self._logger.warning(f"Could not backup/remove {report.name}: {e}")
                except Exception as e:
                    self._logger.warning(f"Could not process OCR corrections: {e}")
            
            # 메모리에서 모델 초기화
            self.label_classifier = None
            self.text_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.position_scaler = None
            self.training_stats = {
                'accuracy': 0.0,
                'training_samples': 0,
                'label_count': 0,
                'label_distribution': {},
                'last_trained': None
            }
            self.document_classes = {}
            
            # 초기화된 통계 저장
            self._save_training_stats()
            
            return {
                'status': 'success',
                'message': 'Model reset completed' if backup_enabled else 'Model reset completed (without backup)',
                'backup_location': str(backup_dir) if backup_dir else None,
                'backed_up_files': backed_up,
                'removed_reports': removed_reports
            }
            
        except PermissionError as e:
            self._logger.warning(f"Permission error during reset, using alternative method: {str(e)}")
            # 권한 문제가 있을 때 대체 방법 사용
            try:
                # 모델 인스턴스만 초기화
                self.label_classifier = None
                self.text_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                self.position_scaler = None
                self.training_stats = {
                    'accuracy': 0.0,
                    'training_samples': 0,
                    'label_count': 0,
                    'label_distribution': {},
                    'last_trained': None
                }
                self.layout_patterns = {}
                self.document_classes = {}
                
                return {
                    'status': 'success',
                    'message': 'Model reset completed (in-memory only due to permission restrictions)',
                    'backed_up_files': [],
                    'removed_reports': 0
                }
            except Exception as inner_e:
                return {
                    'status': 'error',
                    'message': f'Permission error: {str(e)}, Alternative failed: {str(inner_e)}'
                }
        except Exception as e:
            self._logger.error(f"Model reset failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
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