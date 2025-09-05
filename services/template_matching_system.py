#!/usr/bin/env python3
"""
템플릿 매칭 시스템 (3주차)

문서 템플릿을 학습하고 매칭하여 예측 정확도를 향상시키는 시스템
- 문서 유형별 템플릿 자동 생성
- 템플릿 기반 필드 예측
- 동적 템플릿 업데이트
- 유사도 기반 매칭

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-08-22
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re
from datetime import datetime
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import hashlib


@dataclass
class FieldTemplate:
    """필드 템플릿"""
    field_name: str  # 필드 이름 (예: "Order number")
    expected_position: Dict[str, float]  # 정규화된 위치 (x, y, width, height)
    position_variance: Dict[str, float]  # 위치 분산
    text_patterns: List[str]  # 텍스트 패턴들
    pattern_confidence: float  # 패턴 신뢰도
    occurrence_count: int  # 출현 횟수
    required: bool = True  # 필수 필드 여부
    validation_rules: List[str] = field(default_factory=list)  # 검증 규칙
    related_fields: List[str] = field(default_factory=list)  # 관련 필드들


@dataclass
class DocumentTemplate:
    """문서 템플릿"""
    template_id: str  # 템플릿 ID
    document_type: str  # 문서 유형
    template_name: str  # 템플릿 이름
    fields: Dict[str, FieldTemplate]  # 필드 템플릿들
    field_sequence: List[str]  # 필드 출현 순서
    layout_pattern: str  # 레이아웃 패턴
    sample_count: int  # 학습에 사용된 샘플 수
    confidence_threshold: float = 0.7  # 매칭 임계값
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 구조적 정보
    table_regions: List[Dict[str, Any]] = field(default_factory=list)  # 테이블 영역
    header_region: Optional[Dict[str, float]] = None  # 헤더 영역
    footer_region: Optional[Dict[str, float]] = None  # 푸터 영역
    
    # 관계 정보
    field_relationships: Dict[str, List[str]] = field(default_factory=dict)  # 필드 간 관계
    row_templates: List[List[str]] = field(default_factory=list)  # 행 템플릿 (테이블용)


class TemplateMatchingSystem:
    """템플릿 매칭 시스템"""
    
    def __init__(self, model_directory: Path = Path("models/templates")):
        self.model_directory = model_directory
        self.model_directory.mkdir(parents=True, exist_ok=True)
        
        # 템플릿 저장소
        self.templates: Dict[str, DocumentTemplate] = {}
        
        # 템플릿 클러스터 (유사한 템플릿 그룹화)
        self.template_clusters: Dict[str, List[str]] = {}
        
        # 필드 통계
        self.field_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_count': 0,
            'positions': [],
            'text_samples': [],
            'co_occurrences': defaultdict(int)
        })
        
        # 템플릿 매칭 캐시
        self.matching_cache: Dict[str, str] = {}
        
        # 로드 기존 템플릿
        self._load_templates()
    
    def learn_template_from_documents(self, labeled_documents: List[Dict[str, Any]]) -> List[DocumentTemplate]:
        """
        라벨링된 문서들로부터 템플릿 학습
        
        Args:
            labeled_documents: v2 형식의 라벨링된 문서들
            
        Returns:
            생성된 템플릿 리스트
        """
        # 문서 유형별로 그룹화
        doc_groups = defaultdict(list)
        for doc in labeled_documents:
            doc_type = doc.get('document_metadata', {}).get('document_type', 'unknown')
            doc_groups[doc_type].append(doc)
        
        new_templates = []
        
        # 각 문서 유형별로 템플릿 생성
        for doc_type, docs in doc_groups.items():
            print(f"Learning template for document type: {doc_type} ({len(docs)} documents)")
            
            # 레이아웃 패턴별로 다시 그룹화
            layout_groups = self._group_by_layout(docs)
            
            for layout_pattern, layout_docs in layout_groups.items():
                template = self._create_template_from_group(
                    layout_docs, 
                    doc_type, 
                    layout_pattern
                )
                
                if template:
                    # 템플릿 ID 생성
                    template.template_id = self._generate_template_id(template)
                    
                    # 저장
                    self.templates[template.template_id] = template
                    new_templates.append(template)
                    
                    print(f"  Created template: {template.template_name} "
                          f"({len(template.fields)} fields, {template.sample_count} samples)")
        
        # 템플릿 클러스터링
        self._cluster_templates()
        
        # 저장
        self._save_templates()
        
        return new_templates
    
    def _group_by_layout(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """레이아웃 패턴별로 문서 그룹화"""
        layout_groups = defaultdict(list)
        
        for doc in documents:
            # 레이아웃 시그니처 생성
            signature = self._get_layout_signature(doc)
            layout_groups[signature].append(doc)
        
        return layout_groups
    
    def _get_layout_signature(self, document: Dict[str, Any]) -> str:
        """문서의 레이아웃 시그니처 생성"""
        entities = document.get('entities', [])
        
        if not entities:
            return 'empty'
        
        # 필드 집합과 대략적인 위치로 시그니처 생성
        field_positions = []
        
        for entity in entities:
            label = entity.get('label', {}).get('primary', '')
            if label:
                bbox = entity['bbox']
                # 위치를 그리드로 양자화 (10x10 그리드)
                x_grid = int(bbox['x'] * 10 / 2480)
                y_grid = int(bbox['y'] * 10 / 3508)
                field_positions.append(f"{label}:{x_grid},{y_grid}")
        
        # 정렬하여 일관된 시그니처 생성
        field_positions.sort()
        signature = '|'.join(field_positions[:20])  # 상위 20개 필드만 사용
        
        # 해시로 축약
        return hashlib.md5(signature.encode()).hexdigest()[:8]
    
    def _create_template_from_group(self, documents: List[Dict], doc_type: str, layout_pattern: str) -> Optional[DocumentTemplate]:
        """문서 그룹으로부터 템플릿 생성"""
        if len(documents) < 2:  # 최소 2개 이상의 문서 필요
            return None
        
        # 필드별 정보 수집
        field_data = defaultdict(lambda: {
            'positions': [],
            'texts': [],
            'count': 0
        })
        
        # 모든 문서에서 필드 정보 수집
        for doc in documents:
            page_info = doc.get('page_info', {})
            page_width = page_info.get('width', 2480)
            page_height = page_info.get('height', 3508)
            
            for entity in doc.get('entities', []):
                label = entity.get('label', {}).get('primary', '')
                if not label:
                    continue
                
                bbox = entity['bbox']
                # 정규화된 위치
                norm_pos = {
                    'x': bbox['x'] / page_width,
                    'y': bbox['y'] / page_height,
                    'width': bbox['width'] / page_width,
                    'height': bbox['height'] / page_height
                }
                
                field_data[label]['positions'].append(norm_pos)
                field_data[label]['texts'].append(entity.get('text', {}).get('value', ''))
                field_data[label]['count'] += 1
        
        # 필드 템플릿 생성
        fields = {}
        field_sequence = []
        
        for field_name, data in field_data.items():
            if data['count'] < len(documents) * 0.5:  # 50% 이상의 문서에 나타나는 필드만
                continue
            
            # 평균 위치 계산
            positions = data['positions']
            avg_pos = {
                'x': np.mean([p['x'] for p in positions]),
                'y': np.mean([p['y'] for p in positions]),
                'width': np.mean([p['width'] for p in positions]),
                'height': np.mean([p['height'] for p in positions])
            }
            
            # 위치 분산 계산
            var_pos = {
                'x': np.std([p['x'] for p in positions]),
                'y': np.std([p['y'] for p in positions]),
                'width': np.std([p['width'] for p in positions]),
                'height': np.std([p['height'] for p in positions])
            }
            
            # 텍스트 패턴 추출
            text_patterns = self._extract_text_patterns(data['texts'])
            
            # 필드 템플릿 생성
            field_template = FieldTemplate(
                field_name=field_name,
                expected_position=avg_pos,
                position_variance=var_pos,
                text_patterns=text_patterns,
                pattern_confidence=data['count'] / len(documents),
                occurrence_count=data['count'],
                required=(data['count'] >= len(documents) * 0.8)  # 80% 이상 출현시 필수
            )
            
            fields[field_name] = field_template
            field_sequence.append(field_name)
        
        if not fields:
            return None
        
        # 필드 순서 정렬 (위치 기반)
        field_sequence.sort(key=lambda f: (fields[f].expected_position['y'], fields[f].expected_position['x']))
        
        # 테이블 영역 탐지
        table_regions = self._detect_table_regions(documents)
        
        # 헤더/푸터 영역 탐지
        header_region, footer_region = self._detect_header_footer_regions(documents)
        
        # 필드 관계 분석
        field_relationships = self._analyze_field_relationships(documents)
        
        # 행 템플릿 추출 (테이블용)
        row_templates = self._extract_row_templates(documents)
        
        # 문서 템플릿 생성
        template = DocumentTemplate(
            template_id='',  # 나중에 생성
            document_type=doc_type,
            template_name=f"{doc_type}_{layout_pattern}",
            fields=fields,
            field_sequence=field_sequence,
            layout_pattern=layout_pattern,
            sample_count=len(documents),
            table_regions=table_regions,
            header_region=header_region,
            footer_region=footer_region,
            field_relationships=field_relationships,
            row_templates=row_templates
        )
        
        return template
    
    def _extract_text_patterns(self, texts: List[str]) -> List[str]:
        """텍스트 샘플에서 패턴 추출"""
        patterns = []
        
        if not texts:
            return patterns
        
        # 공통 패턴 찾기
        # 1. 정규식 패턴
        regex_patterns = {
            'numeric': r'^\d+$',
            'alphanumeric': r'^[A-Z0-9]+$',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'currency': r'[$¥€£]\s*[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        }
        
        for pattern_name, regex in regex_patterns.items():
            matches = sum(1 for text in texts if re.match(regex, text))
            if matches > len(texts) * 0.5:  # 50% 이상 매칭
                patterns.append(pattern_name)
        
        # 2. 길이 패턴
        lengths = [len(text) for text in texts if text]
        if lengths:
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            patterns.append(f"length:{avg_length:.0f}±{std_length:.1f}")
        
        # 3. 접두사/접미사 패턴
        if len(texts) > 3:
            # 공통 접두사
            prefix = self._find_common_prefix(texts)
            if prefix and len(prefix) > 2:
                patterns.append(f"prefix:{prefix}")
            
            # 공통 접미사
            suffix = self._find_common_suffix(texts)
            if suffix and len(suffix) > 2:
                patterns.append(f"suffix:{suffix}")
        
        return patterns
    
    def _find_common_prefix(self, strings: List[str]) -> str:
        """문자열 리스트의 공통 접두사 찾기"""
        if not strings:
            return ""
        
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
    
    def _find_common_suffix(self, strings: List[str]) -> str:
        """문자열 리스트의 공통 접미사 찾기"""
        if not strings:
            return ""
        
        suffix = strings[0]
        for s in strings[1:]:
            while not s.endswith(suffix):
                suffix = suffix[1:]
                if not suffix:
                    return ""
        return suffix
    
    def _detect_table_regions(self, documents: List[Dict]) -> List[Dict[str, Any]]:
        """테이블 영역 탐지"""
        table_regions = []
        
        # 간단한 휴리스틱: 규칙적인 행/열 구조 찾기
        for doc in documents[:3]:  # 처음 3개 문서만 확인
            entities = doc.get('entities', [])
            
            # y 좌표로 그룹화 (행)
            rows = defaultdict(list)
            for entity in entities:
                y_center = entity['bbox']['y'] + entity['bbox']['height'] / 2
                y_bin = int(y_center / 50)  # 50픽셀 단위로 그룹화
                rows[y_bin].append(entity)
            
            # 연속된 행 중 비슷한 개수의 엔티티를 가진 것들 찾기
            consecutive_rows = []
            prev_bin = None
            prev_count = 0
            
            for y_bin in sorted(rows.keys()):
                count = len(rows[y_bin])
                
                if prev_bin is not None and y_bin == prev_bin + 1 and abs(count - prev_count) <= 1:
                    if not consecutive_rows:
                        consecutive_rows.append(rows[prev_bin])
                    consecutive_rows.append(rows[y_bin])
                elif len(consecutive_rows) >= 3:  # 3행 이상
                    # 테이블 영역으로 판단
                    all_entities = [e for row in consecutive_rows for e in row]
                    if all_entities:
                        min_x = min(e['bbox']['x'] for e in all_entities)
                        max_x = max(e['bbox']['x'] + e['bbox']['width'] for e in all_entities)
                        min_y = min(e['bbox']['y'] for e in all_entities)
                        max_y = max(e['bbox']['y'] + e['bbox']['height'] for e in all_entities)
                        
                        table_regions.append({
                            'x': min_x,
                            'y': min_y,
                            'width': max_x - min_x,
                            'height': max_y - min_y,
                            'row_count': len(consecutive_rows),
                            'avg_col_count': np.mean([len(row) for row in consecutive_rows])
                        })
                    consecutive_rows = []
                else:
                    consecutive_rows = []
                
                prev_bin = y_bin
                prev_count = count
        
        return table_regions
    
    def _detect_header_footer_regions(self, documents: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """헤더/푸터 영역 탐지"""
        header_region = None
        footer_region = None
        
        for doc in documents[:3]:  # 처음 3개 문서만 확인
            page_height = doc.get('page_info', {}).get('height', 3508)
            entities = doc.get('entities', [])
            
            if not entities:
                continue
            
            # 상단 15% 영역의 엔티티들
            header_entities = [e for e in entities if e['bbox']['y'] < page_height * 0.15]
            if header_entities:
                min_x = min(e['bbox']['x'] for e in header_entities)
                max_x = max(e['bbox']['x'] + e['bbox']['width'] for e in header_entities)
                min_y = min(e['bbox']['y'] for e in header_entities)
                max_y = max(e['bbox']['y'] + e['bbox']['height'] for e in header_entities)
                
                header_region = {
                    'x': min_x,
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y
                }
            
            # 하단 15% 영역의 엔티티들
            footer_entities = [e for e in entities if e['bbox']['y'] > page_height * 0.85]
            if footer_entities:
                min_x = min(e['bbox']['x'] for e in footer_entities)
                max_x = max(e['bbox']['x'] + e['bbox']['width'] for e in footer_entities)
                min_y = min(e['bbox']['y'] for e in footer_entities)
                max_y = max(e['bbox']['y'] + e['bbox']['height'] for e in footer_entities)
                
                footer_region = {
                    'x': min_x,
                    'y': min_y,
                    'width': max_x - min_x,
                    'height': max_y - min_y
                }
            
            break  # 첫 번째 유효한 문서만 사용
        
        return header_region, footer_region
    
    def _analyze_field_relationships(self, documents: List[Dict]) -> Dict[str, List[str]]:
        """필드 간 관계 분석"""
        relationships = defaultdict(set)
        
        for doc in documents:
            entities = doc.get('entities', [])
            
            # 같은 행에 있는 필드들 찾기
            rows = defaultdict(list)
            for entity in entities:
                label = entity.get('label', {}).get('primary', '')
                if label:
                    y_center = entity['bbox']['y'] + entity['bbox']['height'] / 2
                    y_bin = int(y_center / 30)  # 30픽셀 단위로 그룹화
                    rows[y_bin].append(label)
            
            # 같은 행의 필드들은 서로 관계가 있다고 판단
            for row_labels in rows.values():
                if len(row_labels) > 1:
                    for i, label1 in enumerate(row_labels):
                        for label2 in row_labels[i+1:]:
                            relationships[label1].add(label2)
                            relationships[label2].add(label1)
        
        # set을 list로 변환
        return {k: list(v) for k, v in relationships.items()}
    
    def _extract_row_templates(self, documents: List[Dict]) -> List[List[str]]:
        """행 템플릿 추출 (테이블용)"""
        row_patterns = defaultdict(int)
        
        for doc in documents:
            entities = doc.get('entities', [])
            
            # y 좌표로 그룹화
            rows = defaultdict(list)
            for entity in entities:
                label = entity.get('label', {}).get('primary', '')
                if label:
                    y_center = entity['bbox']['y'] + entity['bbox']['height'] / 2
                    y_bin = int(y_center / 30)
                    rows[y_bin].append((entity['bbox']['x'], label))
            
            # 각 행의 라벨 시퀀스 추출
            for row_data in rows.values():
                if len(row_data) > 1:
                    # x 좌표로 정렬
                    row_data.sort(key=lambda x: x[0])
                    label_sequence = tuple(item[1] for item in row_data)
                    row_patterns[label_sequence] += 1
        
        # 빈도가 높은 패턴들을 행 템플릿으로
        row_templates = []
        for pattern, count in row_patterns.items():
            if count >= len(documents) * 0.3:  # 30% 이상 출현
                row_templates.append(list(pattern))
        
        return row_templates
    
    def _generate_template_id(self, template: DocumentTemplate) -> str:
        """템플릿 ID 생성"""
        # 템플릿의 특징을 기반으로 유니크한 ID 생성
        features = f"{template.document_type}_{template.layout_pattern}_{len(template.fields)}"
        field_names = ''.join(sorted(template.fields.keys()))
        
        return hashlib.md5(f"{features}_{field_names}".encode()).hexdigest()[:12]
    
    def match_template(self, document: Dict[str, Any]) -> Tuple[Optional[DocumentTemplate], float]:
        """
        문서에 가장 적합한 템플릿 매칭
        
        Args:
            document: 매칭할 문서
            
        Returns:
            (매칭된 템플릿, 매칭 점수)
        """
        # 캐시 확인
        doc_hash = self._get_document_hash(document)
        if doc_hash in self.matching_cache:
            template_id = self.matching_cache[doc_hash]
            return self.templates.get(template_id), 1.0
        
        best_template = None
        best_score = 0.0
        
        # 문서 유형으로 1차 필터링
        doc_type = document.get('document_metadata', {}).get('document_type', 'unknown')
        candidate_templates = [t for t in self.templates.values() 
                              if t.document_type == doc_type or doc_type == 'unknown']
        
        if not candidate_templates:
            candidate_templates = list(self.templates.values())
        
        # 각 템플릿과 매칭
        for template in candidate_templates:
            score = self._calculate_matching_score(document, template)
            
            if score > best_score:
                best_score = score
                best_template = template
        
        # 임계값 이상인 경우만 반환
        if best_template and best_score >= best_template.confidence_threshold:
            # 캐시 업데이트
            self.matching_cache[doc_hash] = best_template.template_id
            return best_template, best_score
        
        return None, 0.0
    
    def _get_document_hash(self, document: Dict[str, Any]) -> str:
        """문서 해시 생성"""
        entities = document.get('entities', [])
        entity_info = []
        
        for entity in entities[:20]:  # 상위 20개만
            label = entity.get('label', {}).get('primary', '')
            bbox = entity['bbox']
            entity_info.append(f"{label}:{bbox['x']},{bbox['y']}")
        
        return hashlib.md5('|'.join(entity_info).encode()).hexdigest()
    
    def _calculate_matching_score(self, document: Dict[str, Any], template: DocumentTemplate) -> float:
        """템플릿 매칭 점수 계산"""
        scores = []
        
        # 1. 필드 매칭 점수
        field_score = self._calculate_field_matching_score(document, template)
        scores.append(field_score * 0.4)  # 40% 가중치
        
        # 2. 위치 매칭 점수
        position_score = self._calculate_position_matching_score(document, template)
        scores.append(position_score * 0.3)  # 30% 가중치
        
        # 3. 구조 매칭 점수
        structure_score = self._calculate_structure_matching_score(document, template)
        scores.append(structure_score * 0.2)  # 20% 가중치
        
        # 4. 패턴 매칭 점수
        pattern_score = self._calculate_pattern_matching_score(document, template)
        scores.append(pattern_score * 0.1)  # 10% 가중치
        
        return sum(scores)
    
    def _calculate_field_matching_score(self, document: Dict[str, Any], template: DocumentTemplate) -> float:
        """필드 매칭 점수 계산"""
        doc_fields = set()
        for entity in document.get('entities', []):
            label = entity.get('label', {}).get('primary', '')
            if label:
                doc_fields.add(label)
        
        template_fields = set(template.fields.keys())
        required_fields = set(f for f, ft in template.fields.items() if ft.required)
        
        # 필수 필드 매칭률
        required_match = len(required_fields & doc_fields) / max(len(required_fields), 1)
        
        # 전체 필드 매칭률
        total_match = len(template_fields & doc_fields) / max(len(template_fields), 1)
        
        # 추가 필드 페널티
        extra_fields = len(doc_fields - template_fields)
        penalty = max(0, 1 - extra_fields * 0.05)  # 추가 필드당 5% 감점
        
        return (required_match * 0.6 + total_match * 0.4) * penalty
    
    def _calculate_position_matching_score(self, document: Dict[str, Any], template: DocumentTemplate) -> float:
        """위치 매칭 점수 계산"""
        position_scores = []
        page_info = document.get('page_info', {})
        page_width = page_info.get('width', 2480)
        page_height = page_info.get('height', 3508)
        
        for entity in document.get('entities', []):
            label = entity.get('label', {}).get('primary', '')
            
            if label in template.fields:
                field_template = template.fields[label]
                
                # 정규화된 위치
                bbox = entity['bbox']
                norm_pos = {
                    'x': bbox['x'] / page_width,
                    'y': bbox['y'] / page_height,
                    'width': bbox['width'] / page_width,
                    'height': bbox['height'] / page_height
                }
                
                # 예상 위치와의 거리
                expected = field_template.expected_position
                variance = field_template.position_variance
                
                # 각 차원별 점수 (분산 고려)
                x_diff = abs(norm_pos['x'] - expected['x'])
                y_diff = abs(norm_pos['y'] - expected['y'])
                
                x_score = max(0, 1 - x_diff / max(variance['x'] * 3, 0.1))
                y_score = max(0, 1 - y_diff / max(variance['y'] * 3, 0.1))
                
                position_scores.append((x_score + y_score) / 2)
        
        return np.mean(position_scores) if position_scores else 0.0
    
    def _calculate_structure_matching_score(self, document: Dict[str, Any], template: DocumentTemplate) -> float:
        """구조 매칭 점수 계산"""
        scores = []
        
        # 테이블 구조 매칭
        if template.table_regions:
            doc_tables = self._detect_table_regions([document])
            if doc_tables:
                # 테이블 개수 비교
                count_score = min(len(doc_tables), len(template.table_regions)) / max(len(template.table_regions), 1)
                scores.append(count_score)
        
        # 행 템플릿 매칭
        if template.row_templates:
            doc_rows = self._extract_row_templates([document])
            if doc_rows:
                # 공통 행 패턴 찾기
                common_patterns = 0
                for doc_row in doc_rows:
                    for template_row in template.row_templates:
                        if doc_row == template_row:
                            common_patterns += 1
                            break
                
                row_score = common_patterns / max(len(template.row_templates), 1)
                scores.append(row_score)
        
        return np.mean(scores) if scores else 0.5  # 구조 정보가 없으면 중립 점수
    
    def _calculate_pattern_matching_score(self, document: Dict[str, Any], template: DocumentTemplate) -> float:
        """패턴 매칭 점수 계산"""
        pattern_scores = []
        
        for entity in document.get('entities', []):
            label = entity.get('label', {}).get('primary', '')
            text = entity.get('text', {}).get('value', '')
            
            if label in template.fields and text:
                field_template = template.fields[label]
                
                # 텍스트 패턴 매칭
                if field_template.text_patterns:
                    pattern_match = False
                    
                    for pattern in field_template.text_patterns:
                        if pattern.startswith('length:'):
                            # 길이 패턴
                            expected_length = float(pattern.split(':')[1].split('±')[0])
                            if abs(len(text) - expected_length) < 5:
                                pattern_match = True
                                break
                        elif pattern == 'numeric' and text.isdigit():
                            pattern_match = True
                            break
                        elif pattern == 'alphanumeric' and text.isalnum():
                            pattern_match = True
                            break
                    
                    pattern_scores.append(1.0 if pattern_match else 0.0)
        
        return np.mean(pattern_scores) if pattern_scores else 0.5
    
    def predict_with_template(self, document: Dict[str, Any], template: Optional[DocumentTemplate] = None) -> Dict[str, Any]:
        """
        템플릿 기반 예측
        
        Args:
            document: 예측할 문서
            template: 사용할 템플릿 (None이면 자동 매칭)
            
        Returns:
            예측 결과
        """
        # 템플릿 매칭
        if template is None:
            template, match_score = self.match_template(document)
            if template is None:
                return {
                    'success': False,
                    'message': 'No matching template found',
                    'predictions': []
                }
        else:
            match_score = 1.0
        
        predictions = []
        
        # 각 엔티티에 대해 예측
        for entity in document.get('entities', []):
            bbox = entity['bbox']
            text = entity.get('text', {}).get('value', '')
            
            # 가장 가능성 높은 필드 찾기
            best_field = None
            best_score = 0.0
            
            for field_name, field_template in template.fields.items():
                # 위치 기반 점수
                position_score = self._calculate_entity_position_score(entity, field_template, document)
                
                # 텍스트 패턴 점수
                pattern_score = self._calculate_entity_pattern_score(text, field_template)
                
                # 종합 점수
                total_score = position_score * 0.6 + pattern_score * 0.4
                
                if total_score > best_score:
                    best_score = total_score
                    best_field = field_name
            
            if best_field and best_score > 0.5:
                predictions.append({
                    'entity_id': entity.get('entity_id', ''),
                    'bbox': bbox,
                    'text': text,
                    'predicted_label': best_field,
                    'confidence': best_score * match_score,
                    'template_match': template.template_name
                })
        
        # 누락된 필수 필드 확인
        predicted_fields = set(p['predicted_label'] for p in predictions)
        missing_required = []
        
        for field_name, field_template in template.fields.items():
            if field_template.required and field_name not in predicted_fields:
                missing_required.append({
                    'field': field_name,
                    'expected_position': field_template.expected_position
                })
        
        return {
            'success': True,
            'template': template.template_name,
            'template_confidence': match_score,
            'predictions': predictions,
            'missing_required_fields': missing_required
        }
    
    def _calculate_entity_position_score(self, entity: Dict, field_template: FieldTemplate, document: Dict) -> float:
        """엔티티의 위치 점수 계산"""
        page_info = document.get('page_info', {})
        page_width = page_info.get('width', 2480)
        page_height = page_info.get('height', 3508)
        
        bbox = entity['bbox']
        norm_pos = {
            'x': bbox['x'] / page_width,
            'y': bbox['y'] / page_height
        }
        
        expected = field_template.expected_position
        variance = field_template.position_variance
        
        # 유클리드 거리
        distance = np.sqrt((norm_pos['x'] - expected['x'])**2 + (norm_pos['y'] - expected['y'])**2)
        
        # 분산을 고려한 점수
        max_variance = max(variance['x'], variance['y'], 0.01)
        score = max(0, 1 - distance / (max_variance * 5))
        
        return score
    
    def _calculate_entity_pattern_score(self, text: str, field_template: FieldTemplate) -> float:
        """엔티티의 패턴 점수 계산"""
        if not text or not field_template.text_patterns:
            return 0.5  # 중립 점수
        
        scores = []
        
        for pattern in field_template.text_patterns:
            if pattern == 'numeric' and text.isdigit():
                scores.append(1.0)
            elif pattern == 'alphanumeric' and text.isalnum():
                scores.append(1.0)
            elif pattern.startswith('length:'):
                expected_length = float(pattern.split(':')[1].split('±')[0])
                length_diff = abs(len(text) - expected_length)
                scores.append(max(0, 1 - length_diff / 10))
            elif pattern.startswith('prefix:'):
                prefix = pattern.split(':')[1]
                scores.append(1.0 if text.startswith(prefix) else 0.0)
            elif pattern.startswith('suffix:'):
                suffix = pattern.split(':')[1]
                scores.append(1.0 if text.endswith(suffix) else 0.0)
        
        return np.mean(scores) if scores else 0.5
    
    def update_template(self, template_id: str, new_documents: List[Dict[str, Any]]):
        """기존 템플릿 업데이트"""
        if template_id not in self.templates:
            return
        
        template = self.templates[template_id]
        
        # 새 문서들로부터 정보 수집
        for doc in new_documents:
            for entity in doc.get('entities', []):
                label = entity.get('label', {}).get('primary', '')
                
                if label in template.fields:
                    # 기존 필드 업데이트
                    field_template = template.fields[label]
                    
                    # 위치 업데이트 (이동 평균)
                    page_info = doc.get('page_info', {})
                    page_width = page_info.get('width', 2480)
                    page_height = page_info.get('height', 3508)
                    
                    bbox = entity['bbox']
                    norm_pos = {
                        'x': bbox['x'] / page_width,
                        'y': bbox['y'] / page_height,
                        'width': bbox['width'] / page_width,
                        'height': bbox['height'] / page_height
                    }
                    
                    # 이동 평균 업데이트
                    alpha = 0.1  # 학습률
                    for key in ['x', 'y', 'width', 'height']:
                        field_template.expected_position[key] = (
                            (1 - alpha) * field_template.expected_position[key] + 
                            alpha * norm_pos[key]
                        )
                    
                    field_template.occurrence_count += 1
                
                elif label and self._should_add_field(label, template):
                    # 새 필드 추가 고려
                    pass  # 구현 생략
        
        template.sample_count += len(new_documents)
        template.updated_at = datetime.now().isoformat()
        
        # 저장
        self._save_templates()
    
    def _should_add_field(self, field_name: str, template: DocumentTemplate) -> bool:
        """새 필드를 템플릿에 추가할지 결정"""
        # 간단한 휴리스틱: 이미 많은 필드가 있으면 추가하지 않음
        return len(template.fields) < 50
    
    def _cluster_templates(self):
        """유사한 템플릿 클러스터링"""
        if len(self.templates) < 2:
            return
        
        # 템플릿 특징 벡터 생성
        template_ids = list(self.templates.keys())
        features = []
        
        for tid in template_ids:
            template = self.templates[tid]
            # 필드 집합을 특징으로 사용
            field_vector = [1 if f in template.fields else 0 
                          for f in self._get_all_field_names()]
            features.append(field_vector)
        
        if not features or not features[0]:
            return
        
        # K-means 클러스터링
        n_clusters = min(5, len(template_ids))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # 클러스터 저장
            self.template_clusters = defaultdict(list)
            for tid, cluster in zip(template_ids, clusters):
                self.template_clusters[f"cluster_{cluster}"].append(tid)
    
    def _get_all_field_names(self) -> List[str]:
        """모든 템플릿의 필드 이름 수집"""
        all_fields = set()
        for template in self.templates.values():
            all_fields.update(template.fields.keys())
        return sorted(list(all_fields))
    
    def _save_templates(self):
        """템플릿 저장"""
        save_path = self.model_directory / "templates.pkl"
        
        data = {
            'templates': self.templates,
            'clusters': self.template_clusters,
            'field_statistics': dict(self.field_statistics)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_templates(self):
        """템플릿 로드"""
        load_path = self.model_directory / "templates.pkl"
        
        if load_path.exists():
            try:
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.templates = data.get('templates', {})
                self.template_clusters = data.get('clusters', {})
                self.field_statistics = defaultdict(
                    lambda: {'total_count': 0, 'positions': [], 'text_samples': [], 'co_occurrences': defaultdict(int)},
                    data.get('field_statistics', {})
                )
                
                print(f"Loaded {len(self.templates)} templates")
            except Exception as e:
                print(f"Error loading templates: {e}")
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """템플릿 통계 반환"""
        stats = {
            'total_templates': len(self.templates),
            'document_types': defaultdict(int),
            'field_frequencies': defaultdict(int),
            'template_details': []
        }
        
        for template in self.templates.values():
            stats['document_types'][template.document_type] += 1
            
            for field_name in template.fields:
                stats['field_frequencies'][field_name] += 1
            
            stats['template_details'].append({
                'id': template.template_id,
                'name': template.template_name,
                'type': template.document_type,
                'fields': len(template.fields),
                'samples': template.sample_count,
                'created': template.created_at,
                'updated': template.updated_at
            })
        
        return stats