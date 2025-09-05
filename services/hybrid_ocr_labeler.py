#!/usr/bin/env python3
"""
YOKOGAWA OCR Hybrid Labeler
하이브리드 모델을 사용한 고급 OCR 라벨링 시스템

XGBoost, LightGBM, CRF를 결합한 앙상블 모델로
95% 이상의 정확도를 목표로 함

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-08-22
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
import re
from collections import defaultdict
import logging

# 기본 ML 라이브러리
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# 새로 추가된 모델들
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sklearn_crfsuite
from sklearn_crfsuite import CRF

import joblib

from config.settings import ApplicationConfig
from core.base_classes import BaseService


@dataclass
class Entity:
    """엔티티 클래스"""
    entity_id: str
    bbox: Dict[str, float]
    text: str
    label: Optional[str] = None
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def x_center(self) -> float:
        return self.bbox['x'] + self.bbox['width'] / 2
    
    @property
    def y_center(self) -> float:
        return self.bbox['y'] + self.bbox['height'] / 2
    
    @property
    def area(self) -> float:
        return self.bbox['width'] * self.bbox['height']


class FeatureExtractor:
    """특징 추출기"""
    
    def __init__(self, page_width: int = 2480, page_height: int = 3508):
        self.page_width = page_width
        self.page_height = page_height
        self.text_vectorizer = TfidfVectorizer(max_features=100, analyzer='char', ngram_range=(1, 3))
        self.position_scaler = StandardScaler()
        
    def extract_position_features(self, entity: Entity) -> Dict[str, float]:
        """위치 기반 특징 추출"""
        features = {
            'x_normalized': entity.bbox['x'] / self.page_width,
            'y_normalized': entity.bbox['y'] / self.page_height,
            'x_center_normalized': entity.x_center / self.page_width,
            'y_center_normalized': entity.y_center / self.page_height,
            'width_normalized': entity.bbox['width'] / self.page_width,
            'height_normalized': entity.bbox['height'] / self.page_height,
            'area_normalized': entity.area / (self.page_width * self.page_height),
            'aspect_ratio': entity.bbox['width'] / max(entity.bbox['height'], 1),
        }
        
        # 사분면 정보
        if entity.x_center < self.page_width / 2:
            if entity.y_center < self.page_height / 2:
                features['quadrant'] = 1  # top-left
            else:
                features['quadrant'] = 3  # bottom-left
        else:
            if entity.y_center < self.page_height / 2:
                features['quadrant'] = 2  # top-right
            else:
                features['quadrant'] = 4  # bottom-right
                
        # 영역 정보 (header, body, footer)
        if entity.y_center < self.page_height * 0.15:
            features['region'] = 1  # header
        elif entity.y_center > self.page_height * 0.85:
            features['region'] = 3  # footer
        else:
            features['region'] = 2  # body
            
        return features
    
    def extract_text_features(self, entity: Entity) -> Dict[str, Any]:
        """텍스트 기반 특징 추출"""
        text = entity.text
        features = {
            'text_length': len(text),
            'is_numeric': text.isdigit(),
            'is_alpha': text.isalpha(),
            'is_alnum': text.isalnum(),
            'has_digits': any(c.isdigit() for c in text),
            'has_letters': any(c.isalpha() for c in text),
            'has_special': any(not c.isalnum() and not c.isspace() for c in text),
            'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
            'upper_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
            'space_count': text.count(' '),
            'dash_count': text.count('-'),
            'dot_count': text.count('.'),
            'comma_count': text.count(','),
        }
        
        # 패턴 매칭
        patterns = {
            'date_pattern': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'order_number': r'\d{10}',
            'currency': r'[$¥€£]\s*[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'[\+\d]?[\d\s\-\(\)]+',
            'item_number': r'^\d{5}$',
            'part_number': r'^[A-Z0-9\-]+$'
        }
        
        for pattern_name, pattern in patterns.items():
            features[f'matches_{pattern_name}'] = bool(re.match(pattern, text))
            
        return features
    
    def extract_relational_features(self, entity: Entity, all_entities: List[Entity]) -> Dict[str, Any]:
        """관계성 특징 추출"""
        features = {}
        
        # 같은 행의 엔티티들
        same_row_threshold = 20  # 픽셀
        same_row_entities = [
            e for e in all_entities 
            if e.entity_id != entity.entity_id and 
            abs(e.y_center - entity.y_center) < same_row_threshold
        ]
        
        features['same_row_count'] = len(same_row_entities)
        features['is_leftmost'] = all(entity.x_center < e.x_center for e in same_row_entities) if same_row_entities else True
        features['is_rightmost'] = all(entity.x_center > e.x_center for e in same_row_entities) if same_row_entities else True
        
        # 같은 열의 엔티티들
        same_col_threshold = 50  # 픽셀
        same_col_entities = [
            e for e in all_entities
            if e.entity_id != entity.entity_id and
            abs(e.x_center - entity.x_center) < same_col_threshold
        ]
        
        features['same_col_count'] = len(same_col_entities)
        features['is_topmost'] = all(entity.y_center < e.y_center for e in same_col_entities) if same_col_entities else True
        features['is_bottommost'] = all(entity.y_center > e.y_center for e in same_col_entities) if same_col_entities else True
        
        # 가장 가까운 이웃들
        if all_entities:
            distances = [
                np.sqrt((entity.x_center - e.x_center)**2 + (entity.y_center - e.y_center)**2)
                for e in all_entities if e.entity_id != entity.entity_id
            ]
            if distances:
                features['min_distance'] = min(distances)
                features['avg_distance'] = np.mean(distances)
                features['nearest_count_100px'] = sum(d < 100 for d in distances)
                features['nearest_count_200px'] = sum(d < 200 for d in distances)
        
        # 좌우 이웃 정보
        left_neighbors = [e for e in same_row_entities if e.x_center < entity.x_center]
        right_neighbors = [e for e in same_row_entities if e.x_center > entity.x_center]
        
        if left_neighbors:
            nearest_left = min(left_neighbors, key=lambda e: entity.x_center - e.x_center)
            features['left_distance'] = entity.bbox['x'] - (nearest_left.bbox['x'] + nearest_left.bbox['width'])
            features['has_left_neighbor'] = True
        else:
            features['left_distance'] = entity.bbox['x']  # 페이지 왼쪽 끝까지의 거리
            features['has_left_neighbor'] = False
            
        if right_neighbors:
            nearest_right = min(right_neighbors, key=lambda e: e.x_center - entity.x_center)
            features['right_distance'] = nearest_right.bbox['x'] - (entity.bbox['x'] + entity.bbox['width'])
            features['has_right_neighbor'] = True
        else:
            features['right_distance'] = self.page_width - (entity.bbox['x'] + entity.bbox['width'])
            features['has_right_neighbor'] = False
            
        return features
    
    def extract_all_features(self, entity: Entity, all_entities: List[Entity]) -> np.ndarray:
        """모든 특징 추출 및 벡터화"""
        # 위치 특징
        position_features = self.extract_position_features(entity)
        
        # 텍스트 특징
        text_features = self.extract_text_features(entity)
        
        # 관계성 특징
        relational_features = self.extract_relational_features(entity, all_entities)
        
        # 모든 특징 결합
        all_features = {**position_features, **text_features, **relational_features}
        
        # 특징 벡터로 변환
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector, all_features


class HybridOCRLabeler(BaseService):
    """하이브리드 OCR 라벨러"""
    
    def __init__(self, config: ApplicationConfig, logger: logging.Logger):
        """
        HybridOCRLabeler 초기화
        
        Args:
            config: 애플리케이션 설정
            logger: 로거 인스턴스
        """
        super().__init__(config, logger)
        self.model_directory = Path(config.model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장 경로
        self.model_paths = {
            'rf': self.model_directory / "hybrid_rf_model.pkl",
            'xgb': self.model_directory / "hybrid_xgb_model.pkl",
            'lgbm': self.model_directory / "hybrid_lgbm_model.pkl",
            'crf': self.model_directory / "hybrid_crf_model.pkl",
            'feature_extractor': self.model_directory / "hybrid_feature_extractor.pkl",
            'label_encoder': self.model_directory / "hybrid_label_encoder.pkl",
            'training_stats': self.model_directory / "hybrid_training_stats.json"
        }
        
        # 모델 인스턴스
        self.rf_classifier = None
        self.xgb_classifier = None
        self.lgbm_classifier = None
        self.crf_model = None
        self.feature_extractor = None
        self.label_encoder = {}
        self.label_decoder = {}
        
        # 학습 통계
        self.training_stats = {
            'total_samples': 0,
            'label_distribution': {},
            'model_performances': {},
            'last_training': None,
            'training_history': []
        }
        
        # 앙상블 가중치
        self.ensemble_weights = {
            'rf': 0.2,
            'xgb': 0.3,
            'lgbm': 0.3,
            'crf': 0.2
        }
        
        self.logger.info("HybridOCRLabeler initialized")
        
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self.feature_extractor = FeatureExtractor()
            
            # 모델 로드 시도
            if self._load_models():
                self.logger.info("Existing models loaded successfully")
            else:
                self.logger.info("Creating new models")
                self._create_new_models()
                
            self.logger.info("HybridOCRLabeler initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize HybridOCRLabeler: {e}")
            return False
    
    def _create_new_models(self):
        """새 모델 생성"""
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.xgb_classifier = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42,
            n_jobs=-1
        )
        
        self.lgbm_classifier = LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.crf_model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
    def _load_models(self) -> bool:
        """저장된 모델 로드"""
        try:
            if not all(path.exists() for path in self.model_paths.values() if path.suffix == '.pkl'):
                return False
                
            self.rf_classifier = joblib.load(self.model_paths['rf'])
            self.xgb_classifier = joblib.load(self.model_paths['xgb'])
            self.lgbm_classifier = joblib.load(self.model_paths['lgbm'])
            self.crf_model = joblib.load(self.model_paths['crf'])
            self.feature_extractor = joblib.load(self.model_paths['feature_extractor'])
            self.label_encoder = joblib.load(self.model_paths['label_encoder'])
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
            
            # 학습 통계 로드
            if self.model_paths['training_stats'].exists():
                with open(self.model_paths['training_stats'], 'r') as f:
                    self.training_stats = json.load(f)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def _save_models(self):
        """모델 저장"""
        try:
            joblib.dump(self.rf_classifier, self.model_paths['rf'])
            joblib.dump(self.xgb_classifier, self.model_paths['xgb'])
            joblib.dump(self.lgbm_classifier, self.model_paths['lgbm'])
            joblib.dump(self.crf_model, self.model_paths['crf'])
            joblib.dump(self.feature_extractor, self.model_paths['feature_extractor'])
            joblib.dump(self.label_encoder, self.model_paths['label_encoder'])
            
            # 학습 통계 저장
            with open(self.model_paths['training_stats'], 'w') as f:
                json.dump(self.training_stats, f, indent=2, default=str)
                
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            training_data: 학습 데이터 리스트
                [{'entities': [...], 'document_metadata': {...}}, ...]
        
        Returns:
            학습 결과 통계
        """
        try:
            self.logger.info(f"Starting training with {len(training_data)} documents")
            
            # 특징과 라벨 추출
            X_all = []
            y_all = []
            sequences = []  # CRF를 위한 시퀀스 데이터
            
            for doc in training_data:
                entities = [Entity(
                    entity_id=e.get('entity_id', f"ent_{i}"),
                    bbox=e['bbox'],
                    text=e['text']['value'],
                    label=e['label']['primary']
                ) for i, e in enumerate(doc['entities'])]
                
                # 문서별 특징 추출
                doc_features = []
                doc_labels = []
                
                for entity in entities:
                    feature_vector, feature_dict = self.feature_extractor.extract_all_features(entity, entities)
                    X_all.append(feature_vector)
                    y_all.append(entity.label)
                    
                    # CRF를 위한 특징 (딕셔너리 형태)
                    doc_features.append(feature_dict)
                    doc_labels.append(entity.label)
                
                sequences.append((doc_features, doc_labels))
            
            # 라벨 인코딩
            unique_labels = list(set(y_all))
            self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
            
            y_encoded = [self.label_encoder[label] for label in y_all]
            
            # 특징 배열 변환
            X_array = np.array(X_all)
            
            # 각 모델 학습
            self.logger.info("Training Random Forest...")
            self.rf_classifier.fit(X_array, y_encoded)
            rf_score = cross_val_score(self.rf_classifier, X_array, y_encoded, cv=3).mean()
            self.logger.info(f"RF Cross-validation score: {rf_score:.3f}")
            
            self.logger.info("Training XGBoost...")
            self.xgb_classifier.fit(X_array, y_encoded)
            xgb_score = cross_val_score(self.xgb_classifier, X_array, y_encoded, cv=3).mean()
            self.logger.info(f"XGBoost Cross-validation score: {xgb_score:.3f}")
            
            self.logger.info("Training LightGBM...")
            self.lgbm_classifier.fit(X_array, y_encoded)
            lgbm_score = cross_val_score(self.lgbm_classifier, X_array, y_encoded, cv=3).mean()
            self.logger.info(f"LightGBM Cross-validation score: {lgbm_score:.3f}")
            
            # CRF 학습 (시퀀스 데이터 사용)
            self.logger.info("Training CRF...")
            X_crf = [doc_features for doc_features, _ in sequences]
            y_crf = [doc_labels for _, doc_labels in sequences]
            
            # CRF 특징을 딕셔너리 리스트로 변환
            X_crf_formatted = []
            for doc_features in X_crf:
                doc_formatted = []
                for features in doc_features:
                    # 특징을 문자열 키로 변환 (CRF 요구사항)
                    formatted_features = {str(k): v for k, v in features.items()}
                    doc_formatted.append(formatted_features)
                X_crf_formatted.append(doc_formatted)
            
            self.crf_model.fit(X_crf_formatted, y_crf)
            
            # 학습 통계 업데이트
            self.training_stats['total_samples'] = len(X_all)
            self.training_stats['label_distribution'] = {
                label: y_all.count(label) for label in unique_labels
            }
            self.training_stats['model_performances'] = {
                'rf': rf_score,
                'xgb': xgb_score,
                'lgbm': lgbm_score
            }
            self.training_stats['last_training'] = datetime.now().isoformat()
            
            # 모델 저장
            self._save_models()
            
            self.logger.info("Training completed successfully")
            return self.training_stats
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, entities: List[Dict[str, Any]], document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        라벨 예측
        
        Args:
            entities: 예측할 엔티티 리스트
            document_metadata: 문서 메타데이터
        
        Returns:
            예측 결과 리스트
        """
        try:
            if not entities:
                return []
            
            # 모델이 학습되지 않았으면 빈 결과 반환 (자동 학습 방지)
            if (not hasattr(self, 'rf_classifier') or self.rf_classifier is None or 
                not hasattr(self, 'label_encoder') or not self.label_encoder):
                self.logger.warning("Models are not trained yet. Returning empty predictions.")
                return []
            
            # Entity 객체로 변환
            entity_objects = []
            for i, e in enumerate(entities):
                # text 처리: dict인 경우 value 추출
                text_value = e.get('text', '')
                if isinstance(text_value, dict):
                    text_value = text_value.get('value', '')
                
                entity_objects.append(Entity(
                    entity_id=e.get('entity_id', f"ent_{i}"),
                    bbox=e['bbox'],
                    text=text_value
                ))
            
            # 특징 추출
            X_features = []
            feature_dicts = []
            
            for entity in entity_objects:
                feature_vector, feature_dict = self.feature_extractor.extract_all_features(entity, entity_objects)
                X_features.append(feature_vector)
                feature_dicts.append(feature_dict)
            
            X_array = np.array(X_features)
            
            # 각 모델로 예측
            predictions = {}
            
            # Random Forest
            try:
                rf_proba = self.rf_classifier.predict_proba(X_array)
                predictions['rf'] = rf_proba
            except ValueError as e:
                self.logger.warning(f"RF prediction failed: {e}. Using fallback.")
                # 피처 개수가 맞지 않으면 기본값 사용
                num_classes = len(self.label_decoder)
                predictions['rf'] = np.ones((len(X_array), num_classes)) / num_classes
            
            # XGBoost
            try:
                xgb_proba = self.xgb_classifier.predict_proba(X_array)
                predictions['xgb'] = xgb_proba
            except (ValueError, Exception) as e:
                self.logger.warning(f"XGB prediction failed: {e}. Using fallback.")
                num_classes = len(self.label_decoder)
                predictions['xgb'] = np.ones((len(X_array), num_classes)) / num_classes
            
            # LightGBM
            try:
                lgbm_proba = self.lgbm_classifier.predict_proba(X_array)
                predictions['lgbm'] = lgbm_proba
            except (ValueError, Exception) as e:
                self.logger.warning(f"LGBM prediction failed: {e}. Using fallback.")
                num_classes = len(self.label_decoder)
                predictions['lgbm'] = np.ones((len(X_array), num_classes)) / num_classes
            
            # CRF (문서 전체를 시퀀스로 처리)
            X_crf = [{str(k): v for k, v in features.items()} for features in feature_dicts]
            crf_predictions = self.crf_model.predict([X_crf])[0]
            
            # CRF 예측을 확률로 변환 (간단한 방법)
            crf_proba = np.zeros((len(entities), len(self.label_encoder)))
            for i, pred_label in enumerate(crf_predictions):
                if pred_label in self.label_encoder:
                    label_idx = self.label_encoder[pred_label]
                    crf_proba[i, label_idx] = 1.0
            predictions['crf'] = crf_proba
            
            # 앙상블 (가중 평균)
            ensemble_proba = (
                self.ensemble_weights['rf'] * predictions['rf'] +
                self.ensemble_weights['xgb'] * predictions['xgb'] +
                self.ensemble_weights['lgbm'] * predictions['lgbm'] +
                self.ensemble_weights['crf'] * predictions['crf']
            )
            
            # 최종 예측
            results = []
            for i, entity in enumerate(entities):
                pred_idx = np.argmax(ensemble_proba[i])
                pred_label = self.label_decoder[pred_idx]
                confidence = float(ensemble_proba[i, pred_idx])
                
                # 대체 라벨들
                sorted_indices = np.argsort(ensemble_proba[i])[::-1]
                alternatives = []
                for idx in sorted_indices[1:4]:  # 상위 3개 대체 라벨
                    if ensemble_proba[i, idx] > 0.1:
                        alternatives.append({
                            'label': self.label_decoder[idx],
                            'confidence': float(ensemble_proba[i, idx])
                        })
                
                # text 처리: dict인 경우 value 추출
                text_value = entity.get('text', '')
                if isinstance(text_value, dict):
                    text_value = text_value.get('value', '')
                
                result = {
                    'entity_id': entity.get('entity_id', f"ent_{i}"),
                    'bbox': entity['bbox'],
                    'text': text_value,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'alternatives': alternatives,
                    'model_confidences': {
                        'rf': float(predictions['rf'][i, pred_idx]),
                        'xgb': float(predictions['xgb'][i, pred_idx]),
                        'lgbm': float(predictions['lgbm'][i, pred_idx]),
                        'crf': float(predictions['crf'][i, pred_idx])
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return []
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            test_data: 테스트 데이터
        
        Returns:
            평가 결과
        """
        try:
            all_true = []
            all_pred = []
            
            for doc in test_data:
                entities = doc['entities']
                
                # 실제 라벨
                true_labels = [e['label']['primary'] for e in entities]
                all_true.extend(true_labels)
                
                # 예측
                predictions = self.predict(entities, doc.get('document_metadata'))
                pred_labels = [p['predicted_label'] for p in predictions]
                all_pred.extend(pred_labels)
            
            # 평가 지표 계산
            accuracy = accuracy_score(all_true, all_pred)
            report = classification_report(all_true, all_pred, output_dict=True)
            
            evaluation_results = {
                'accuracy': accuracy,
                'classification_report': report,
                'total_samples': len(all_true),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Evaluation completed - Accuracy: {accuracy:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {}
    
    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """
        앙상블 가중치 업데이트
        
        Args:
            new_weights: 새로운 가중치 {'rf': 0.2, 'xgb': 0.3, ...}
        """
        if sum(new_weights.values()) != 1.0:
            # 정규화
            total = sum(new_weights.values())
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        self.ensemble_weights = new_weights
        self.logger.info(f"Ensemble weights updated: {self.ensemble_weights}")
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """특징 중요도 반환"""
        importance = {}
        
        if hasattr(self.rf_classifier, 'feature_importances_'):
            importance['rf'] = self.rf_classifier.feature_importances_.tolist()
        
        if hasattr(self.xgb_classifier, 'feature_importances_'):
            importance['xgb'] = self.xgb_classifier.feature_importances_.tolist()
        
        if hasattr(self.lgbm_classifier, 'feature_importances_'):
            importance['lgbm'] = self.lgbm_classifier.feature_importances_.tolist()
        
        return importance
    
    def cleanup(self) -> bool:
        """리소스 정리"""
        try:
            # 모델 저장
            self._save_models()
            self.logger.info("HybridOCRLabeler cleanup completed")
            return True
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """서비스 상태 확인"""
        return {
            'status': 'healthy' if self.rf_classifier is not None else 'not_initialized',
            'models_loaded': {
                'rf': self.rf_classifier is not None,
                'xgb': self.xgb_classifier is not None,
                'lgbm': self.lgbm_classifier is not None,
                'crf': self.crf_model is not None
            },
            'total_samples': self.training_stats.get('total_samples', 0),
            'last_training': self.training_stats.get('last_training'),
            'ensemble_weights': self.ensemble_weights
        }