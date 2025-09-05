#!/usr/bin/env python3
"""
YOKOGAWA OCR 고급 모델 서비스

95% 정확도를 목표로 하는 향상된 라벨 예측 시스템
- v2 라벨 형식 활용
- 문서 구조 이해
- 컨텍스트 기반 학습
- 템플릿 매칭

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-08-17
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib

from config.settings import ApplicationConfig
from core.base_classes import BaseService


@dataclass
class Entity:
    """엔티티 클래스"""
    entity_id: str
    bbox: Dict[str, float]
    text: str
    label: str
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    
    @property
    def x_center(self) -> float:
        return self.bbox['x'] + self.bbox['width'] / 2
    
    @property
    def y_center(self) -> float:
        return self.bbox['y'] + self.bbox['height'] / 2


@dataclass
class DocumentTemplate:
    """문서 템플릿 클래스"""
    template_id: str
    document_type: str
    layout_pattern: str
    expected_fields: List[str]
    field_positions: Dict[str, Dict[str, float]]  # 필드별 예상 위치
    field_patterns: Dict[str, str]  # 필드별 텍스트 패턴
    validation_rules: Dict[str, Any]


class LayoutAnalyzer:
    """문서 레이아웃 분석기"""
    
    def __init__(self):
        self.regions = []
        self.columns = []
        self.tables = []
        
    def analyze_layout(self, entities: List[Entity], page_info: Dict[str, Any]) -> Dict[str, Any]:
        """레이아웃 분석"""
        # 영역 감지
        regions = self._detect_regions(entities, page_info)
        
        # 컬럼 감지
        columns = self._detect_columns(entities, page_info)
        
        # 테이블 감지
        tables = self._detect_tables(entities, page_info)
        
        # 레이아웃 패턴 결정
        pattern = self._determine_pattern(regions, columns, tables)
        
        return {
            'regions': regions,
            'columns': columns,
            'tables': tables,
            'pattern': pattern,
            'confidence': self._calculate_confidence(regions, columns, tables)
        }
    
    def _detect_regions(self, entities: List[Entity], page_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """영역 감지 (헤더, 본문, 푸터 등)"""
        page_height = page_info.get('height', 3508)
        
        regions = []
        
        # 상단 영역 (0-15%)
        header_entities = [e for e in entities if e.y_center < page_height * 0.15]
        if header_entities:
            regions.append({
                'type': 'header',
                'bounds': {'y_min': 0, 'y_max': page_height * 0.15},
                'entities': [e.entity_id for e in header_entities]
            })
        
        # 하단 영역 (85-100%)
        footer_entities = [e for e in entities if e.y_center > page_height * 0.85]
        if footer_entities:
            regions.append({
                'type': 'footer',
                'bounds': {'y_min': page_height * 0.85, 'y_max': page_height},
                'entities': [e.entity_id for e in footer_entities]
            })
        
        # 본문 영역
        body_entities = [e for e in entities if page_height * 0.15 <= e.y_center <= page_height * 0.85]
        if body_entities:
            regions.append({
                'type': 'body',
                'bounds': {'y_min': page_height * 0.15, 'y_max': page_height * 0.85},
                'entities': [e.entity_id for e in body_entities]
            })
        
        return regions
    
    def _detect_columns(self, entities: List[Entity], page_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """컬럼 구조 감지"""
        if not entities:
            return []
        
        # X 좌표로 정렬
        sorted_entities = sorted(entities, key=lambda e: e.x_center)
        
        # 클러스터링으로 컬럼 감지
        columns = []
        current_column = []
        threshold = page_info.get('width', 2480) * 0.05  # 5% 너비를 임계값으로
        
        for entity in sorted_entities:
            if not current_column:
                current_column.append(entity)
            else:
                # 이전 엔티티와의 X 거리 확인
                prev_x = current_column[-1].x_center
                if abs(entity.x_center - prev_x) < threshold:
                    current_column.append(entity)
                else:
                    # 새로운 컬럼 시작
                    columns.append({
                        'entities': [e.entity_id for e in current_column],
                        'x_center': sum(e.x_center for e in current_column) / len(current_column)
                    })
                    current_column = [entity]
        
        if current_column:
            columns.append({
                'entities': [e.entity_id for e in current_column],
                'x_center': sum(e.x_center for e in current_column) / len(current_column)
            })
        
        return columns
    
    def _detect_tables(self, entities: List[Entity], page_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """테이블 구조 감지"""
        # Y 좌표로 그룹화 (같은 행)
        rows = defaultdict(list)
        row_threshold = 30  # 같은 행으로 간주할 Y 차이
        
        for entity in entities:
            # 가장 가까운 행 찾기
            matched_row = None
            for row_y in rows.keys():
                if abs(entity.y_center - row_y) < row_threshold:
                    matched_row = row_y
                    break
            
            if matched_row:
                rows[matched_row].append(entity)
            else:
                rows[entity.y_center] = [entity]
        
        # 테이블 후보 찾기 (3개 이상의 엔티티가 있는 행)
        table_rows = []
        for row_y, row_entities in rows.items():
            if len(row_entities) >= 3:
                table_rows.append({
                    'y': row_y,
                    'entities': sorted(row_entities, key=lambda e: e.x_center)
                })
        
        # 연속된 행들을 테이블로 그룹화
        tables = []
        if table_rows:
            table_rows.sort(key=lambda r: r['y'])
            current_table = [table_rows[0]]
            
            for i in range(1, len(table_rows)):
                if table_rows[i]['y'] - table_rows[i-1]['y'] < row_threshold * 3:
                    current_table.append(table_rows[i])
                else:
                    if len(current_table) >= 2:  # 최소 2행 이상
                        tables.append({
                            'rows': current_table,
                            'row_count': len(current_table),
                            'column_count': max(len(r['entities']) for r in current_table)
                        })
                    current_table = [table_rows[i]]
            
            if len(current_table) >= 2:
                tables.append({
                    'rows': current_table,
                    'row_count': len(current_table),
                    'column_count': max(len(r['entities']) for r in current_table)
                })
        
        return tables
    
    def _determine_pattern(self, regions: List[Dict], columns: List[Dict], tables: List[Dict]) -> str:
        """레이아웃 패턴 결정"""
        if tables:
            return 'table_based'
        elif len(columns) >= 2:
            return 'multi_column'
        else:
            return 'single_column'
    
    def _calculate_confidence(self, regions: List[Dict], columns: List[Dict], tables: List[Dict]) -> float:
        """레이아웃 분석 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        if regions:
            confidence += 0.2
        if columns:
            confidence += 0.2
        if tables:
            confidence += 0.1
            
        return min(confidence, 1.0)


class RelationshipAnalyzer:
    """엔티티 간 관계 분석기"""
    
    def analyze_relationships(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """엔티티 간 관계 분석"""
        relationships = defaultdict(list)
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # 공간적 관계
                    spatial_rel = self._get_spatial_relationship(entity1, entity2)
                    if spatial_rel:
                        relationships[entity1.entity_id].append({
                            'target': entity2.entity_id,
                            'type': spatial_rel,
                            'confidence': 0.9
                        })
                    
                    # 의미적 관계
                    semantic_rel = self._get_semantic_relationship(entity1, entity2)
                    if semantic_rel:
                        relationships[entity1.entity_id].append({
                            'target': entity2.entity_id,
                            'type': semantic_rel,
                            'confidence': 0.8
                        })
        
        return dict(relationships)
    
    def _get_spatial_relationship(self, e1: Entity, e2: Entity) -> Optional[str]:
        """공간적 관계 판단"""
        # 수평 정렬 확인
        if abs(e1.y_center - e2.y_center) < 30:
            if e1.x_center < e2.x_center:
                return 'left_of'
            else:
                return 'right_of'
        
        # 수직 정렬 확인
        if abs(e1.x_center - e2.x_center) < 50:
            if e1.y_center < e2.y_center:
                return 'above'
            else:
                return 'below'
        
        return None
    
    def _get_semantic_relationship(self, e1: Entity, e2: Entity) -> Optional[str]:
        """의미적 관계 판단"""
        # Item number와 관련 필드
        if e1.label == 'Item number' and e2.label in ['Part number', 'Quantity', 'Unit price']:
            if abs(e1.y_center - e2.y_center) < 50:  # 같은 행
                return 'item_detail'
        
        # 라벨과 값의 관계
        if ':' in e1.text and abs(e1.y_center - e2.y_center) < 30:
            if e1.x_center < e2.x_center:
                return 'label_value'
        
        return None


class AdvancedModelService(BaseService):
    """고급 모델 서비스"""
    
    def __init__(self, config: ApplicationConfig, logger: logging.Logger):
        super().__init__(config, logger)
        self.model_directory = Path(config.model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        # 서비스 컴포넌트
        self.layout_analyzer = LayoutAnalyzer()
        self.relationship_analyzer = RelationshipAnalyzer()
        
        # 모델 저장 경로
        self.models_path = self.model_directory / "advanced_models"
        self.models_path.mkdir(exist_ok=True)
        
        # 모델 인스턴스
        self.label_classifier = None
        self.text_vectorizer = None
        self.position_scaler = None
        self.ensemble_model = None
        
        # 템플릿 저장소
        self.templates: Dict[str, DocumentTemplate] = {}
        self.load_templates()
        
        # 학습 통계
        self.training_stats = {
            'total_samples': 0,
            'accuracy': 0.0,
            'label_distribution': {},
            'template_distribution': {},
            'last_training_time': None,
            'model_version': '2.0.0'
        }
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self._logger.info("Initializing Advanced Model Service")
            
            # 기존 모델 로드 시도
            self._load_models()
            
            # 모델이 없으면 새로 생성
            if self.label_classifier is None:
                self._logger.info("Creating new advanced models")
                self.text_vectorizer = TfidfVectorizer(
                    max_features=2000,
                    ngram_range=(1, 3),
                    min_df=1  # 단일 문서도 처리 가능하도록 수정
                )
                self.position_scaler = StandardScaler()
                
                # 앙상블 모델 사용
                self.ensemble_model = {
                    'rf': RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42),
                    'gb': GradientBoostingClassifier(n_estimators=200, random_state=42)
                }
            
            self._logger.info("Advanced Model Service initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Advanced Model Service: {e}")
            return False
    
    def load_templates(self):
        """문서 템플릿 로드"""
        # Purchase Order 템플릿
        self.templates['yokogawa_po_v1'] = DocumentTemplate(
            template_id='yokogawa_po_v1',
            document_type='purchase_order',
            layout_pattern='standard',
            expected_fields=[
                'Order number', 'Date', 'Vendor', 'Ship to',
                'Item number', 'Part number', 'Description',
                'Quantity', 'Unit price', 'Net amount',
                'Shipping line', 'Case mark', 'Total'
            ],
            field_positions={
                'Order number': {'x_norm': 0.85, 'y_norm': 0.05, 'tolerance': 0.1},
                'Date': {'x_norm': 0.85, 'y_norm': 0.08, 'tolerance': 0.1},
                'Total': {'x_norm': 0.7, 'y_norm': 0.9, 'tolerance': 0.15}
            },
            field_patterns={
                'Order number': r'^\d{10}$',
                'Date': r'\d{1,2}-\d{1,2}-\d{4}',
                'Item number': r'^\d{5}$',
                'Shipping line': r'^[A-Z]\d{7}$',
                'Case mark': r'.*YMG.*KOFU.*'
            },
            validation_rules={
                'required_fields': ['Order number', 'Date', 'Total'],
                'sum_validation': True  # 아이템 합계 = 총액
            }
        )
    
    def _load_models(self):
        """모델 로드"""
        try:
            model_file = self.models_path / "advanced_model.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.ensemble_model = saved_data['ensemble_model']
                    self.text_vectorizer = saved_data['text_vectorizer']
                    self.position_scaler = saved_data['position_scaler']
                    self.training_stats = saved_data['training_stats']
                self._logger.info("Advanced models loaded successfully")
        except Exception as e:
            self._logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """모델 저장"""
        try:
            model_file = self.models_path / "advanced_model.pkl"
            save_data = {
                'ensemble_model': self.ensemble_model,
                'text_vectorizer': self.text_vectorizer,
                'position_scaler': self.position_scaler,
                'training_stats': self.training_stats,
                'version': '2.0.0'
            }
            with open(model_file, 'wb') as f:
                pickle.dump(save_data, f)
            self._logger.info("Advanced models saved successfully")
        except Exception as e:
            self._logger.error(f"Error saving models: {e}")
    
    def train_from_v2_labels(self, label_files: List[Path]) -> Dict[str, Any]:
        """v2 라벨 파일로부터 학습"""
        self._logger.info(f"Starting training with {len(label_files)} v2 label files")
        
        # 데이터 로드 및 전처리
        all_entities = []
        all_features = []
        all_labels = []
        template_counts = defaultdict(int)
        
        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # v2 형식 확인
                if data.get('annotation_version') != '2.0':
                    continue
                
                # 엔티티 추출
                entities = []
                for ent_data in data.get('entities', []):
                    entity = Entity(
                        entity_id=ent_data['entity_id'],
                        bbox=ent_data['bbox'],
                        text=ent_data['text']['value'],
                        label=ent_data['label']['primary'],
                        confidence=ent_data['label']['confidence'],
                        features=ent_data.get('features', {}),
                        relationships=ent_data.get('relationships', [])
                    )
                    entities.append(entity)
                
                # 레이아웃 분석
                page_info = data.get('page_info', {})
                layout = self.layout_analyzer.analyze_layout(entities, page_info)
                
                # 관계 분석
                relationships = self.relationship_analyzer.analyze_relationships(entities)
                
                # 특징 추출
                for entity in entities:
                    features = self._extract_advanced_features(
                        entity, entities, layout, relationships, data
                    )
                    all_features.append(features)
                    all_labels.append(entity.label)
                    all_entities.append(entity)
                
                # 템플릿 통계
                template_id = data.get('templates', {}).get('detected_template', 'unknown')
                template_counts[template_id] += 1
                
            except Exception as e:
                self._logger.error(f"Error processing {label_file}: {e}")
                continue
        
        if len(all_features) < 20:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 20 samples for training',
                'sample_count': len(all_features)
            }
        
        # 특징 벡터 준비
        X_numeric, X_text = self._prepare_feature_vectors(all_entities, all_features)
        y = np.array(all_labels)
        
        # 모델 학습
        self._train_ensemble_model(X_numeric, X_text, y)
        
        # 교차 검증
        scores = self._evaluate_model(X_numeric, X_text, y)
        
        # 통계 업데이트
        self.training_stats.update({
            'total_samples': len(all_labels),
            'accuracy': np.mean(scores),
            'label_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'template_distribution': dict(template_counts),
            'last_training_time': datetime.now().isoformat()
        })
        
        # 모델 저장
        self._save_models()
        
        return {
            'status': 'success',
            'total_samples': len(all_labels),
            'accuracy': np.mean(scores),
            'cross_val_scores': scores.tolist(),
            'label_distribution': self.training_stats['label_distribution'],
            'template_distribution': self.training_stats['template_distribution']
        }
    
    def _extract_advanced_features(
        self, 
        entity: Entity, 
        all_entities: List[Entity],
        layout: Dict[str, Any],
        relationships: Dict[str, List[str]],
        document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """고급 특징 추출"""
        features = {}
        
        # 1. 기본 텍스트 특징
        text = entity.text
        features.update({
            'text_length': len(text),
            'is_numeric': text.replace('.', '').replace(',', '').isdigit(),
            'has_dash': '-' in text,
            'has_colon': ':' in text,
            'is_uppercase': text.isupper(),
            'word_count': len(text.split()),
            'has_special_chars': bool(re.search(r'[!@#$%^&*()_+=]', text)),
        })
        
        # 2. 패턴 매칭 특징
        features.update({
            'is_date_format': bool(re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', text)),
            'is_order_number': bool(re.match(r'^\d{10}$', text)),
            'is_item_number': bool(re.match(r'^\d{5}$', text)),
            'is_currency': bool(re.match(r'^\d+\.\d{2,4}$', text)),
            'is_shipping_line': bool(re.match(r'^[A-Z]\d{7}$', text)),
            'contains_total': 'total' in text.lower(),
            'contains_item': 'item' in text.lower(),
        })
        
        # 3. 위치 특징 (정규화)
        page_width = document_data.get('page_info', {}).get('width', 2480)
        page_height = document_data.get('page_info', {}).get('height', 3508)
        
        features.update({
            'x_normalized': entity.bbox['x'] / page_width,
            'y_normalized': entity.bbox['y'] / page_height,
            'width_normalized': entity.bbox['width'] / page_width,
            'height_normalized': entity.bbox['height'] / page_height,
            'x_center_norm': entity.x_center / page_width,
            'y_center_norm': entity.y_center / page_height,
            'area_normalized': (entity.bbox['width'] * entity.bbox['height']) / (page_width * page_height),
        })
        
        # 4. 영역 특징
        features.update({
            'in_header': entity.y_center < page_height * 0.15,
            'in_footer': entity.y_center > page_height * 0.85,
            'in_left_margin': entity.x_center < page_width * 0.2,
            'in_right_margin': entity.x_center > page_width * 0.8,
            'in_center': 0.3 < entity.x_center / page_width < 0.7,
        })
        
        # 5. 레이아웃 컨텍스트 특징
        features['layout_pattern'] = layout.get('pattern', 'unknown')
        features['in_table'] = any(
            entity.entity_id in row['entities'] 
            for table in layout.get('tables', [])
            for row in table.get('rows', [])
            for e in row.get('entities', [])
        )
        
        # 6. 관계 특징
        entity_relationships = relationships.get(entity.entity_id, [])
        features['num_relationships'] = len(entity_relationships)
        features['has_left_neighbor'] = any(r['type'] == 'left_of' for r in entity_relationships)
        features['has_right_neighbor'] = any(r['type'] == 'right_of' for r in entity_relationships)
        features['has_above_neighbor'] = any(r['type'] == 'above' for r in entity_relationships)
        features['has_below_neighbor'] = any(r['type'] == 'below' for r in entity_relationships)
        
        # 7. 주변 엔티티 컨텍스트
        nearby_entities = self._get_nearby_entities(entity, all_entities, threshold=100)
        features['nearby_entity_count'] = len(nearby_entities)
        
        # 같은 행의 엔티티 수
        same_row_entities = [e for e in all_entities if abs(e.y_center - entity.y_center) < 30]
        features['same_row_count'] = len(same_row_entities)
        
        # 8. OCR 신뢰도 특징
        ocr_results = entity.features.get('ocr_results', {})
        features['ocr_confidence'] = ocr_results.get('tesseract_confidence', 0)
        features['was_corrected'] = ocr_results.get('was_corrected', False)
        
        return features
    
    def _get_nearby_entities(self, entity: Entity, all_entities: List[Entity], threshold: float) -> List[Entity]:
        """주변 엔티티 찾기"""
        nearby = []
        for other in all_entities:
            if other.entity_id != entity.entity_id:
                distance = np.sqrt(
                    (entity.x_center - other.x_center) ** 2 + 
                    (entity.y_center - other.y_center) ** 2
                )
                if distance < threshold:
                    nearby.append(other)
        return nearby
    
    def _prepare_feature_vectors(self, entities: List[Entity], features: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """특징 벡터 준비"""
        # 수치 특징
        numeric_features = []
        text_features = []
        
        for i, feat in enumerate(features):
            # 수치 특징 추출
            numeric_feat = [
                feat.get('text_length', 0),
                float(feat.get('is_numeric', 0)),
                float(feat.get('has_dash', 0)),
                float(feat.get('has_colon', 0)),
                float(feat.get('is_uppercase', 0)),
                feat.get('word_count', 0),
                float(feat.get('has_special_chars', 0)),
                float(feat.get('is_date_format', 0)),
                float(feat.get('is_order_number', 0)),
                float(feat.get('is_item_number', 0)),
                float(feat.get('is_currency', 0)),
                float(feat.get('is_shipping_line', 0)),
                float(feat.get('contains_total', 0)),
                float(feat.get('contains_item', 0)),
                feat.get('x_normalized', 0),
                feat.get('y_normalized', 0),
                feat.get('width_normalized', 0),
                feat.get('height_normalized', 0),
                feat.get('x_center_norm', 0),
                feat.get('y_center_norm', 0),
                feat.get('area_normalized', 0),
                float(feat.get('in_header', 0)),
                float(feat.get('in_footer', 0)),
                float(feat.get('in_left_margin', 0)),
                float(feat.get('in_right_margin', 0)),
                float(feat.get('in_center', 0)),
                float(feat.get('in_table', 0)),
                feat.get('num_relationships', 0),
                float(feat.get('has_left_neighbor', 0)),
                float(feat.get('has_right_neighbor', 0)),
                float(feat.get('has_above_neighbor', 0)),
                float(feat.get('has_below_neighbor', 0)),
                feat.get('nearby_entity_count', 0),
                feat.get('same_row_count', 0),
                feat.get('ocr_confidence', 0),
                float(feat.get('was_corrected', 0))
            ]
            numeric_features.append(numeric_feat)
            
            # 텍스트 특징
            text_features.append(entities[i].text)
        
        # 변환
        X_numeric = np.array(numeric_features)
        
        # 텍스트 벡터화
        if not hasattr(self.text_vectorizer, 'vocabulary_'):
            X_text = self.text_vectorizer.fit_transform(text_features)
        else:
            X_text = self.text_vectorizer.transform(text_features)
        
        # 스케일링
        if not hasattr(self.position_scaler, 'mean_'):
            X_numeric = self.position_scaler.fit_transform(X_numeric)
        else:
            X_numeric = self.position_scaler.transform(X_numeric)
        
        return X_numeric, X_text
    
    def _train_ensemble_model(self, X_numeric: np.ndarray, X_text: Any, y: np.ndarray):
        """앙상블 모델 학습"""
        # 특징 결합
        X_combined = np.hstack([X_numeric, X_text.toarray()])
        
        # 각 모델 학습
        for name, model in self.ensemble_model.items():
            self._logger.info(f"Training {name} model...")
            model.fit(X_combined, y)
        
        self._logger.info("Ensemble model training completed")
    
    def _evaluate_model(self, X_numeric: np.ndarray, X_text: Any, y: np.ndarray) -> np.ndarray:
        """모델 평가"""
        X_combined = np.hstack([X_numeric, X_text.toarray()])
        
        # 교차 검증
        scores = []
        for name, model in self.ensemble_model.items():
            cv_scores = cross_val_score(model, X_combined, y, cv=5, scoring='accuracy')
            scores.extend(cv_scores)
            self._logger.info(f"{name} CV scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return np.array(scores)
    
    def predict_with_confidence(self, image_path: str, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """신뢰도와 함께 예측"""
        try:
            # OCR 결과를 엔티티로 변환
            entities = self._convert_ocr_to_entities(ocr_results)
            
            if not entities:
                return {'entities': [], 'confidence': 0.0}
            
            # 레이아웃 분석
            page_info = {'width': 2480, 'height': 3508}  # 기본값
            layout = self.layout_analyzer.analyze_layout(entities, page_info)
            
            # 관계 분석
            relationships = self.relationship_analyzer.analyze_relationships(entities)
            
            # 템플릿 매칭
            matched_template = self._match_template(entities, layout)
            
            # 각 엔티티에 대한 예측
            predictions = []
            for entity in entities:
                features = self._extract_advanced_features(
                    entity, entities, layout, relationships, 
                    {'page_info': page_info, 'templates': {'detected_template': matched_template}}
                )
                
                # 특징 벡터 준비
                X_numeric, X_text = self._prepare_feature_vectors([entity], [features])
                X_combined = np.hstack([X_numeric, X_text.toarray()])
                
                # 앙상블 예측
                predictions_all = []
                confidences_all = []
                
                for name, model in self.ensemble_model.items():
                    pred = model.predict(X_combined)[0]
                    pred_proba = model.predict_proba(X_combined)[0]
                    predictions_all.append(pred)
                    confidences_all.append(max(pred_proba))
                
                # 다수결 투표
                from collections import Counter
                label_counts = Counter(predictions_all)
                final_label = label_counts.most_common(1)[0][0]
                
                # 평균 신뢰도
                avg_confidence = np.mean(confidences_all)
                
                # 템플릿 기반 보정
                if matched_template and matched_template in self.templates:
                    final_label = self._apply_template_rules(
                        entity, final_label, self.templates[matched_template]
                    )
                
                entity.label = final_label
                entity.confidence = avg_confidence
                
                predictions.append({
                    'entity_id': entity.entity_id,
                    'bbox': entity.bbox,
                    'text': entity.text,
                    'label': final_label,
                    'confidence': avg_confidence,
                    'template_match': matched_template is not None
                })
            
            # 후처리 및 검증
            predictions = self._post_process_predictions(predictions, layout, matched_template)
            
            return {
                'entities': predictions,
                'layout': layout,
                'template': matched_template,
                'overall_confidence': np.mean([p['confidence'] for p in predictions])
            }
            
        except Exception as e:
            self._logger.error(f"Prediction failed: {e}")
            return {'entities': [], 'confidence': 0.0, 'error': str(e)}
    
    def _convert_ocr_to_entities(self, ocr_results: List[Dict[str, Any]]) -> List[Entity]:
        """OCR 결과를 엔티티로 변환"""
        entities = []
        entity_counter = 0
        
        for page_result in ocr_results:
            for word in page_result.get('words', []):
                if word.get('text') and word.get('bbox'):
                    entity = Entity(
                        entity_id=f"ent_{entity_counter:03d}",
                        bbox={
                            'x': word['bbox'].get('x', 0),
                            'y': word['bbox'].get('y', 0),
                            'width': word['bbox'].get('width', 100),
                            'height': word['bbox'].get('height', 30)
                        },
                        text=word['text'],
                        label='unknown',
                        confidence=0.0
                    )
                    entities.append(entity)
                    entity_counter += 1
        
        return entities
    
    def _match_template(self, entities: List[Entity], layout: Dict[str, Any]) -> Optional[str]:
        """템플릿 매칭"""
        best_match = None
        best_score = 0.0
        
        for template_id, template in self.templates.items():
            score = 0.0
            
            # 레이아웃 패턴 매칭
            if layout.get('pattern') == template.layout_pattern:
                score += 0.3
            
            # 필드 패턴 매칭
            for field, pattern in template.field_patterns.items():
                for entity in entities:
                    if re.match(pattern, entity.text):
                        score += 0.1
                        break
            
            # 위치 매칭
            for field, position in template.field_positions.items():
                for entity in entities:
                    x_norm = entity.x_center / 2480  # 기본 너비
                    y_norm = entity.y_center / 3508  # 기본 높이
                    
                    x_diff = abs(x_norm - position['x_norm'])
                    y_diff = abs(y_norm - position['y_norm'])
                    
                    if x_diff < position['tolerance'] and y_diff < position['tolerance']:
                        score += 0.2
                        break
            
            if score > best_score:
                best_score = score
                best_match = template_id
        
        return best_match if best_score > 0.5 else None
    
    def _apply_template_rules(self, entity: Entity, predicted_label: str, template: DocumentTemplate) -> str:
        """템플릿 기반 규칙 적용"""
        # 위치 기반 보정
        for field, position in template.field_positions.items():
            x_norm = entity.x_center / 2480
            y_norm = entity.y_center / 3508
            
            x_diff = abs(x_norm - position['x_norm'])
            y_diff = abs(y_norm - position['y_norm'])
            
            if x_diff < position['tolerance'] and y_diff < position['tolerance']:
                # 패턴도 확인
                if field in template.field_patterns:
                    if re.match(template.field_patterns[field], entity.text):
                        return field
        
        return predicted_label
    
    def _post_process_predictions(
        self, 
        predictions: List[Dict[str, Any]], 
        layout: Dict[str, Any],
        template_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """예측 후처리"""
        # 1. 중복 제거
        seen = set()
        unique_predictions = []
        for pred in predictions:
            key = (pred['text'], pred['label'])
            if key not in seen:
                seen.add(key)
                unique_predictions.append(pred)
        
        # 2. 그룹핑
        predictions = self._group_predictions(unique_predictions, layout)
        
        # 3. 검증 규칙 적용
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            predictions = self._apply_validation_rules(predictions, template)
        
        # 4. 신뢰도 기반 필터링
        min_confidence = 0.3
        predictions = [p for p in predictions if p['confidence'] >= min_confidence]
        
        return predictions
    
    def _group_predictions(self, predictions: List[Dict[str, Any]], layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """예측 그룹핑"""
        # 아이템 번호 기준으로 그룹화
        groups = defaultdict(list)
        
        for pred in predictions:
            if pred['label'] == 'Item number':
                group_id = f"ITEM_{pred['text']}"
            else:
                # Y 좌표로 그룹 찾기
                group_id = None
                for existing_group, members in groups.items():
                    if any(abs(pred['bbox']['y'] - m['bbox']['y']) < 50 for m in members):
                        group_id = existing_group
                        break
                
                if not group_id:
                    group_id = f"GROUP_{len(groups) + 1}"
            
            pred['group_id'] = group_id
            groups[group_id].append(pred)
        
        return predictions
    
    def _apply_validation_rules(self, predictions: List[Dict[str, Any]], template: DocumentTemplate) -> List[Dict[str, Any]]:
        """검증 규칙 적용"""
        # 필수 필드 확인
        required_fields = template.validation_rules.get('required_fields', [])
        found_fields = {p['label'] for p in predictions}
        
        # 누락된 필수 필드에 대한 경고
        missing_fields = set(required_fields) - found_fields
        if missing_fields:
            self._logger.warning(f"Missing required fields: {missing_fields}")
        
        # 합계 검증
        if template.validation_rules.get('sum_validation'):
            self._validate_sum(predictions)
        
        return predictions
    
    def _validate_sum(self, predictions: List[Dict[str, Any]]):
        """합계 검증"""
        # 아이템별 금액 계산
        item_totals = []
        
        for group_id, group_preds in self._group_by_id(predictions).items():
            if group_id.startswith('ITEM_'):
                quantity = None
                unit_price = None
                
                for pred in group_preds:
                    if pred['label'] == 'Quantity':
                        try:
                            quantity = float(pred['text'].split()[0])
                        except:
                            pass
                    elif pred['label'] == 'Unit price':
                        try:
                            unit_price = float(pred['text'].replace(',', ''))
                        except:
                            pass
                
                if quantity and unit_price:
                    item_totals.append(quantity * unit_price)
        
        # 총액과 비교
        total_pred = next((p for p in predictions if p['label'] == 'Net amount (total)'), None)
        if total_pred and item_totals:
            try:
                total_value = float(re.search(r'[\d,]+\.?\d*', total_pred['text']).group().replace(',', ''))
                calculated_total = sum(item_totals)
                
                if abs(total_value - calculated_total) > 0.01:
                    self._logger.warning(
                        f"Sum validation failed: Total={total_value}, Calculated={calculated_total}"
                    )
            except:
                pass
    
    def _group_by_id(self, predictions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """그룹 ID별로 예측 그룹화"""
        groups = defaultdict(list)
        for pred in predictions:
            groups[pred.get('group_id', 'unknown')].append(pred)
        return dict(groups)
    
    def cleanup(self) -> None:
        """서비스 정리"""
        self._logger.info("Cleaning up Advanced Model Service")
        self._save_models()
        super().cleanup()
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            if not self._is_initialized:
                return False
            
            if not self.models_path.exists():
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False