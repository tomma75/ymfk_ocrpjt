#!/usr/bin/env python3
"""
관계성 특징 추출기 (2주차 개선)

엔티티 간의 복잡한 관계를 학습하기 위한 고급 특징 추출
- 공간적 관계 그래프
- 의미적 관계 추론
- 이미지 기반 시각적 단서
- 템플릿 패턴 학습

작성자: YOKOGAWA OCR 개발팀  
작성일: 2025-08-22
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import re
from collections import defaultdict
import cv2
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import networkx as nx

# 이미지 처리 관련
try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None


@dataclass
class SpatialRelation:
    """공간적 관계 정보"""
    source_id: str
    target_id: str
    relation_type: str  # above, below, left, right, same_row, same_column
    distance: float
    angle: float  # 방향 각도
    overlap_ratio: float  # 영역 겹침 비율
    alignment_score: float  # 정렬 점수


@dataclass 
class SemanticRelation:
    """의미적 관계 정보"""
    source_id: str
    target_id: str
    relation_type: str  # key-value, header-item, group-member
    confidence: float
    evidence: List[str]  # 관계 근거


class EnhancedRelationalFeatureExtractor:
    """강화된 관계성 특징 추출기"""
    
    def __init__(self, page_width: int = 2480, page_height: int = 3508):
        self.page_width = page_width
        self.page_height = page_height
        
        # 관계 그래프
        self.spatial_graph = nx.Graph()
        self.semantic_graph = nx.DiGraph()
        
        # 클러스터링 파라미터
        self.row_threshold = 20  # 같은 행 판단 기준 (픽셀)
        self.col_threshold = 50  # 같은 열 판단 기준 (픽셀)
        
        # 패턴 학습용 저장소
        self.learned_patterns = {
            'spatial_patterns': {},
            'label_sequences': {},
            'group_templates': {}
        }
        
    def extract_enhanced_features(self, entities: List[Dict], image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        강화된 관계성 특징 추출
        
        Args:
            entities: 엔티티 리스트
            image_path: 원본 이미지 경로 (선택)
            
        Returns:
            관계성 특징 딕셔너리
        """
        features = {
            'spatial_relations': self._extract_spatial_relations(entities),
            'semantic_relations': self._extract_semantic_relations(entities),
            'structural_groups': self._detect_structural_groups(entities),
            'layout_patterns': self._analyze_layout_patterns(entities),
            'visual_cues': {}
        }
        
        # 이미지 기반 특징 추출
        if image_path and Path(image_path).exists():
            features['visual_cues'] = self._extract_visual_cues(entities, image_path)
        
        # 그래프 기반 특징
        features['graph_features'] = self._extract_graph_features(entities)
        
        return features
    
    def _extract_spatial_relations(self, entities: List[Dict]) -> List[SpatialRelation]:
        """공간적 관계 추출"""
        relations = []
        
        for i, entity1 in enumerate(entities):
            bbox1 = entity1['bbox']
            x1_center = bbox1['x'] + bbox1['width'] / 2
            y1_center = bbox1['y'] + bbox1['height'] / 2
            
            for j, entity2 in enumerate(entities):
                if i >= j:  # 중복 방지
                    continue
                    
                bbox2 = entity2['bbox']
                x2_center = bbox2['x'] + bbox2['width'] / 2
                y2_center = bbox2['y'] + bbox2['height'] / 2
                
                # 거리 계산
                distance = np.sqrt((x2_center - x1_center)**2 + (y2_center - y1_center)**2)
                
                # 각도 계산 (라디안)
                angle = np.arctan2(y2_center - y1_center, x2_center - x1_center)
                
                # 관계 타입 결정
                relation_type = self._determine_spatial_relation(bbox1, bbox2)
                
                # 겹침 비율 계산
                overlap_ratio = self._calculate_overlap_ratio(bbox1, bbox2)
                
                # 정렬 점수 계산
                alignment_score = self._calculate_alignment_score(bbox1, bbox2)
                
                relation = SpatialRelation(
                    source_id=entity1.get('entity_id', f'ent_{i}'),
                    target_id=entity2.get('entity_id', f'ent_{j}'),
                    relation_type=relation_type,
                    distance=distance,
                    angle=angle,
                    overlap_ratio=overlap_ratio,
                    alignment_score=alignment_score
                )
                
                relations.append(relation)
                
                # 그래프에 추가
                self.spatial_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    weight=1/max(distance, 1),  # 거리의 역수를 가중치로
                    relation=relation
                )
        
        return relations
    
    def _determine_spatial_relation(self, bbox1: Dict, bbox2: Dict) -> str:
        """두 바운딩박스 간의 공간적 관계 결정"""
        x1_center = bbox1['x'] + bbox1['width'] / 2
        y1_center = bbox1['y'] + bbox1['height'] / 2
        x2_center = bbox2['x'] + bbox2['width'] / 2
        y2_center = bbox2['y'] + bbox2['height'] / 2
        
        # 같은 행 체크
        if abs(y1_center - y2_center) < self.row_threshold:
            if x1_center < x2_center:
                return 'same_row_left'
            else:
                return 'same_row_right'
        
        # 같은 열 체크
        if abs(x1_center - x2_center) < self.col_threshold:
            if y1_center < y2_center:
                return 'same_column_above'
            else:
                return 'same_column_below'
        
        # 대각선 관계
        if x1_center < x2_center and y1_center < y2_center:
            return 'diagonal_top_left'
        elif x1_center > x2_center and y1_center < y2_center:
            return 'diagonal_top_right'
        elif x1_center < x2_center and y1_center > y2_center:
            return 'diagonal_bottom_left'
        else:
            return 'diagonal_bottom_right'
    
    def _calculate_overlap_ratio(self, bbox1: Dict, bbox2: Dict) -> float:
        """두 바운딩박스의 겹침 비율 계산"""
        x1_min, y1_min = bbox1['x'], bbox1['y']
        x1_max = x1_min + bbox1['width']
        y1_max = y1_min + bbox1['height']
        
        x2_min, y2_min = bbox2['x'], bbox2['y']
        x2_max = x2_min + bbox2['width']
        y2_max = y2_min + bbox2['height']
        
        # 겹치는 영역 계산
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        overlap_area = x_overlap * y_overlap
        
        # 전체 영역
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        
        # IoU (Intersection over Union)
        union_area = area1 + area2 - overlap_area
        
        return overlap_area / max(union_area, 1)
    
    def _calculate_alignment_score(self, bbox1: Dict, bbox2: Dict) -> float:
        """정렬 점수 계산 (0~1)"""
        scores = []
        
        # 상단 정렬
        top_diff = abs(bbox1['y'] - bbox2['y'])
        scores.append(1 - min(top_diff / 100, 1))
        
        # 하단 정렬
        bottom1 = bbox1['y'] + bbox1['height']
        bottom2 = bbox2['y'] + bbox2['height']
        bottom_diff = abs(bottom1 - bottom2)
        scores.append(1 - min(bottom_diff / 100, 1))
        
        # 좌측 정렬
        left_diff = abs(bbox1['x'] - bbox2['x'])
        scores.append(1 - min(left_diff / 100, 1))
        
        # 우측 정렬
        right1 = bbox1['x'] + bbox1['width']
        right2 = bbox2['x'] + bbox2['width']
        right_diff = abs(right1 - right2)
        scores.append(1 - min(right_diff / 100, 1))
        
        # 최대 정렬 점수 반환
        return max(scores)
    
    def _extract_semantic_relations(self, entities: List[Dict]) -> List[SemanticRelation]:
        """의미적 관계 추출"""
        relations = []
        
        # 키-값 쌍 탐지
        key_value_pairs = self._detect_key_value_pairs(entities)
        for kv in key_value_pairs:
            relations.append(kv)
        
        # 헤더-아이템 관계 탐지
        header_item_relations = self._detect_header_item_relations(entities)
        for hi in header_item_relations:
            relations.append(hi)
        
        # 그룹 멤버 관계 탐지
        group_relations = self._detect_group_relations(entities)
        for gr in group_relations:
            relations.append(gr)
        
        return relations
    
    def _detect_key_value_pairs(self, entities: List[Dict]) -> List[SemanticRelation]:
        """키-값 쌍 탐지"""
        relations = []
        
        # 일반적인 키 패턴
        key_patterns = [
            r'.*number$', r'.*date$', r'.*name$', r'.*address$',
            r'.*code$', r'.*id$', r'.*total$', r'.*amount$'
        ]
        
        for i, entity1 in enumerate(entities):
            label1 = entity1.get('label', {}).get('primary', '')
            if not label1:
                continue
                
            # 키 패턴 매칭
            is_key = any(re.match(pattern, label1.lower()) for pattern in key_patterns)
            
            if is_key:
                # 가장 가까운 오른쪽 엔티티 찾기
                best_candidate = None
                min_distance = float('inf')
                
                bbox1 = entity1['bbox']
                x1_right = bbox1['x'] + bbox1['width']
                y1_center = bbox1['y'] + bbox1['height'] / 2
                
                for j, entity2 in enumerate(entities):
                    if i == j:
                        continue
                        
                    bbox2 = entity2['bbox']
                    x2_left = bbox2['x']
                    y2_center = bbox2['y'] + bbox2['height'] / 2
                    
                    # 같은 행에 있고 오른쪽에 있는지 체크
                    if abs(y1_center - y2_center) < self.row_threshold and x2_left > x1_right:
                        distance = x2_left - x1_right
                        if distance < min_distance:
                            min_distance = distance
                            best_candidate = entity2
                
                if best_candidate and min_distance < 200:  # 200픽셀 이내
                    relation = SemanticRelation(
                        source_id=entity1.get('entity_id', f'ent_{i}'),
                        target_id=best_candidate.get('entity_id', ''),
                        relation_type='key_value',
                        confidence=0.8,
                        evidence=[f'Key pattern match: {label1}', f'Distance: {min_distance:.1f}px']
                    )
                    relations.append(relation)
        
        return relations
    
    def _detect_header_item_relations(self, entities: List[Dict]) -> List[SemanticRelation]:
        """헤더-아이템 관계 탐지"""
        relations = []
        
        # 열 기반 그룹핑
        columns = self._group_by_columns(entities)
        
        for col_entities in columns:
            if len(col_entities) < 2:
                continue
                
            # 최상단 엔티티를 헤더로 가정
            sorted_entities = sorted(col_entities, key=lambda e: e['bbox']['y'])
            header = sorted_entities[0]
            
            # 헤더 아래 엔티티들을 아이템으로
            for item in sorted_entities[1:]:
                relation = SemanticRelation(
                    source_id=header.get('entity_id', ''),
                    target_id=item.get('entity_id', ''),
                    relation_type='header_item',
                    confidence=0.7,
                    evidence=['Column alignment', f"Column position: {header['bbox']['x']}"]
                )
                relations.append(relation)
        
        return relations
    
    def _detect_group_relations(self, entities: List[Dict]) -> List[SemanticRelation]:
        """그룹 관계 탐지"""
        relations = []
        
        # 행 기반 그룹핑
        rows = self._group_by_rows(entities)
        
        for row_entities in rows:
            if len(row_entities) < 2:
                continue
            
            # Item number를 찾아서 그룹 리더로 설정
            leader = None
            for entity in row_entities:
                label = entity.get('label', {}).get('primary', '')
                if 'item' in label.lower() and 'number' in label.lower():
                    leader = entity
                    break
            
            if leader:
                for member in row_entities:
                    if member != leader:
                        relation = SemanticRelation(
                            source_id=leader.get('entity_id', ''),
                            target_id=member.get('entity_id', ''),
                            relation_type='group_member',
                            confidence=0.75,
                            evidence=['Same row', f"Y position: {leader['bbox']['y']}"]
                        )
                        relations.append(relation)
        
        return relations
    
    def _detect_structural_groups(self, entities: List[Dict]) -> Dict[str, List]:
        """구조적 그룹 탐지"""
        groups = {
            'rows': self._group_by_rows(entities),
            'columns': self._group_by_columns(entities),
            'tables': self._detect_tables(entities),
            'clusters': self._detect_clusters(entities)
        }
        
        return groups
    
    def _group_by_rows(self, entities: List[Dict]) -> List[List[Dict]]:
        """행별로 그룹핑"""
        if not entities:
            return []
        
        # y 좌표로 정렬
        sorted_entities = sorted(entities, key=lambda e: e['bbox']['y'])
        
        rows = []
        current_row = [sorted_entities[0]]
        current_y = sorted_entities[0]['bbox']['y'] + sorted_entities[0]['bbox']['height'] / 2
        
        for entity in sorted_entities[1:]:
            y_center = entity['bbox']['y'] + entity['bbox']['height'] / 2
            
            if abs(y_center - current_y) < self.row_threshold:
                current_row.append(entity)
            else:
                if current_row:
                    # x 좌표로 정렬
                    current_row.sort(key=lambda e: e['bbox']['x'])
                    rows.append(current_row)
                current_row = [entity]
                current_y = y_center
        
        if current_row:
            current_row.sort(key=lambda e: e['bbox']['x'])
            rows.append(current_row)
        
        return rows
    
    def _group_by_columns(self, entities: List[Dict]) -> List[List[Dict]]:
        """열별로 그룹핑"""
        if not entities:
            return []
        
        # x 좌표로 정렬
        sorted_entities = sorted(entities, key=lambda e: e['bbox']['x'])
        
        columns = []
        current_col = [sorted_entities[0]]
        current_x = sorted_entities[0]['bbox']['x'] + sorted_entities[0]['bbox']['width'] / 2
        
        for entity in sorted_entities[1:]:
            x_center = entity['bbox']['x'] + entity['bbox']['width'] / 2
            
            if abs(x_center - current_x) < self.col_threshold:
                current_col.append(entity)
            else:
                if current_col:
                    # y 좌표로 정렬
                    current_col.sort(key=lambda e: e['bbox']['y'])
                    columns.append(current_col)
                current_col = [entity]
                current_x = x_center
        
        if current_col:
            current_col.sort(key=lambda e: e['bbox']['y'])
            columns.append(current_col)
        
        return columns
    
    def _detect_tables(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """테이블 구조 탐지"""
        tables = []
        
        # 행과 열이 규칙적으로 정렬된 영역 찾기
        rows = self._group_by_rows(entities)
        
        # 연속된 행들 중 같은 수의 엔티티를 가진 것들을 테이블로 판단
        current_table = []
        current_col_count = 0
        
        for row in rows:
            col_count = len(row)
            
            if col_count > 1:  # 최소 2개 이상의 열
                if current_col_count == 0 or abs(col_count - current_col_count) <= 1:
                    current_table.append(row)
                    current_col_count = col_count
                else:
                    if len(current_table) >= 2:  # 최소 2행 이상
                        tables.append({
                            'rows': current_table,
                            'row_count': len(current_table),
                            'col_count': current_col_count
                        })
                    current_table = [row]
                    current_col_count = col_count
        
        if len(current_table) >= 2:
            tables.append({
                'rows': current_table,
                'row_count': len(current_table),
                'col_count': current_col_count
            })
        
        return tables
    
    def _detect_clusters(self, entities: List[Dict]) -> List[List[Dict]]:
        """DBSCAN을 사용한 클러스터링"""
        if len(entities) < 2:
            return [entities]
        
        # 중심점 좌표 추출
        centers = []
        for entity in entities:
            bbox = entity['bbox']
            x_center = bbox['x'] + bbox['width'] / 2
            y_center = bbox['y'] + bbox['height'] / 2
            centers.append([x_center, y_center])
        
        centers = np.array(centers)
        
        # DBSCAN 클러스터링
        clustering = DBSCAN(eps=100, min_samples=2).fit(centers)
        
        # 클러스터별로 그룹화
        clusters = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label != -1:  # 노이즈가 아닌 경우
                clusters[label].append(entities[i])
        
        return list(clusters.values())
    
    def _analyze_layout_patterns(self, entities: List[Dict]) -> Dict[str, Any]:
        """레이아웃 패턴 분석"""
        patterns = {}
        
        # 헤더 영역 탐지
        header_entities = [e for e in entities if e['bbox']['y'] < self.page_height * 0.15]
        patterns['header_count'] = len(header_entities)
        
        # 본문 영역 탐지
        body_entities = [e for e in entities if 
                        self.page_height * 0.15 <= e['bbox']['y'] <= self.page_height * 0.85]
        patterns['body_count'] = len(body_entities)
        
        # 푸터 영역 탐지
        footer_entities = [e for e in entities if e['bbox']['y'] > self.page_height * 0.85]
        patterns['footer_count'] = len(footer_entities)
        
        # 밀도 계산
        if entities:
            all_y = [e['bbox']['y'] for e in entities]
            patterns['vertical_spread'] = max(all_y) - min(all_y)
            
            all_x = [e['bbox']['x'] for e in entities]
            patterns['horizontal_spread'] = max(all_x) - min(all_x)
            
            # 엔티티 간 평균 거리
            if len(entities) > 1:
                centers = np.array([[e['bbox']['x'] + e['bbox']['width']/2, 
                                   e['bbox']['y'] + e['bbox']['height']/2] 
                                  for e in entities])
                distances = cdist(centers, centers)
                np.fill_diagonal(distances, np.inf)
                patterns['avg_min_distance'] = np.mean(np.min(distances, axis=1))
        
        return patterns
    
    def _extract_visual_cues(self, entities: List[Dict], image_path: str) -> Dict[str, Any]:
        """이미지 기반 시각적 단서 추출"""
        visual_cues = {}
        
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                return visual_cues
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 선 검출 (테이블 구조 파악용)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                # 수평선과 수직선 분류
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                    
                    if angle < np.pi / 6:  # 수평선 (30도 이내)
                        horizontal_lines.append((y1 + y2) / 2)
                    elif angle > np.pi / 3:  # 수직선 (60도 이상)
                        vertical_lines.append((x1 + x2) / 2)
                
                visual_cues['horizontal_lines'] = len(horizontal_lines)
                visual_cues['vertical_lines'] = len(vertical_lines)
                visual_cues['has_table_structure'] = len(horizontal_lines) > 3 and len(vertical_lines) > 3
            
            # 각 엔티티 영역의 시각적 특징
            entity_visual_features = []
            
            for entity in entities:
                bbox = entity['bbox']
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                
                # 경계 체크
                if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                    continue
                
                roi = gray[y:y+h, x:x+w]
                
                if roi.size > 0:
                    features = {
                        'entity_id': entity.get('entity_id', ''),
                        'mean_intensity': np.mean(roi),
                        'std_intensity': np.std(roi),
                        'contrast': np.max(roi) - np.min(roi)
                    }
                    
                    # 폰트 크기 추정 (연결 요소 분석)
                    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        heights = [cv2.boundingRect(c)[3] for c in contours]
                        features['estimated_font_size'] = np.median(heights) if heights else 0
                    
                    entity_visual_features.append(features)
            
            visual_cues['entity_features'] = entity_visual_features
            
        except Exception as e:
            print(f"Error extracting visual cues: {e}")
        
        return visual_cues
    
    def _extract_graph_features(self, entities: List[Dict]) -> Dict[str, Any]:
        """그래프 기반 특징 추출"""
        graph_features = {}
        
        if self.spatial_graph.number_of_nodes() > 0:
            # 그래프 통계
            graph_features['num_nodes'] = self.spatial_graph.number_of_nodes()
            graph_features['num_edges'] = self.spatial_graph.number_of_edges()
            graph_features['density'] = nx.density(self.spatial_graph)
            
            # 중심성 측정
            if self.spatial_graph.number_of_nodes() > 1:
                try:
                    degree_centrality = nx.degree_centrality(self.spatial_graph)
                    graph_features['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
                    
                    # 가장 중심적인 노드들
                    top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                    graph_features['top_central_nodes'] = top_central
                    
                    # 클러스터링 계수
                    graph_features['avg_clustering'] = nx.average_clustering(self.spatial_graph)
                    
                    # 연결 요소
                    components = list(nx.connected_components(self.spatial_graph))
                    graph_features['num_components'] = len(components)
                    graph_features['largest_component_size'] = max(len(c) for c in components)
                    
                except Exception as e:
                    print(f"Error in graph analysis: {e}")
        
        return graph_features
    
    def learn_from_labeled_data(self, labeled_data: List[Dict]):
        """라벨링된 데이터로부터 패턴 학습"""
        for doc in labeled_data:
            entities = doc.get('entities', [])
            
            if not entities:
                continue
            
            # 공간적 패턴 학습
            rows = self._group_by_rows(entities)
            for row in rows:
                if len(row) > 1:
                    # 행 내 라벨 시퀀스 저장
                    labels = [e.get('label', {}).get('primary', '') for e in row]
                    label_seq = tuple(labels)
                    
                    if label_seq not in self.learned_patterns['label_sequences']:
                        self.learned_patterns['label_sequences'][label_seq] = 0
                    self.learned_patterns['label_sequences'][label_seq] += 1
            
            # 그룹 템플릿 학습
            for entity in entities:
                label = entity.get('label', {}).get('primary', '')
                if label:
                    bbox = entity['bbox']
                    normalized_pos = {
                        'x_norm': bbox['x'] / self.page_width,
                        'y_norm': bbox['y'] / self.page_height,
                        'w_norm': bbox['width'] / self.page_width,
                        'h_norm': bbox['height'] / self.page_height
                    }
                    
                    if label not in self.learned_patterns['group_templates']:
                        self.learned_patterns['group_templates'][label] = []
                    self.learned_patterns['group_templates'][label].append(normalized_pos)
    
    def predict_relations(self, entities: List[Dict]) -> Dict[str, Any]:
        """학습된 패턴을 기반으로 관계 예측"""
        predictions = {
            'predicted_groups': [],
            'predicted_sequences': [],
            'confidence_scores': {}
        }
        
        # 행별 라벨 시퀀스 예측
        rows = self._group_by_rows(entities)
        for row in rows:
            if len(row) > 1:
                # 가장 유사한 학습된 시퀀스 찾기
                best_match = None
                best_score = 0
                
                for learned_seq, count in self.learned_patterns['label_sequences'].items():
                    if len(learned_seq) == len(row):
                        # 위치 기반 매칭 점수
                        score = count  # 빈도를 점수로 사용
                        if score > best_score:
                            best_score = score
                            best_match = learned_seq
                
                if best_match:
                    predictions['predicted_sequences'].append({
                        'row_entities': [e.get('entity_id', '') for e in row],
                        'predicted_labels': best_match,
                        'confidence': min(best_score / 10, 1.0)  # 정규화
                    })
        
        return predictions


class RelationalModelIntegration:
    """관계성 모델 통합 클래스"""
    
    def __init__(self, hybrid_model):
        self.hybrid_model = hybrid_model
        self.feature_extractor = EnhancedRelationalFeatureExtractor()
        
    def enhance_predictions_with_relations(self, entities: List[Dict], image_path: Optional[str] = None) -> List[Dict]:
        """관계성 정보를 활용한 예측 개선"""
        
        # 관계성 특징 추출
        relational_features = self.feature_extractor.extract_enhanced_features(entities, image_path)
        
        # 기본 예측
        base_predictions = self.hybrid_model.predict(entities)
        
        # 관계성 기반 보정
        enhanced_predictions = []
        
        for i, pred in enumerate(base_predictions):
            enhanced_pred = pred.copy()
            
            # 같은 행의 다른 예측 참고
            row_context = self._get_row_context(i, base_predictions, relational_features['structural_groups']['rows'])
            
            # 신뢰도 조정
            if row_context:
                # 같은 행에 Item number가 있으면 다른 엔티티들의 신뢰도 상승
                if any('Item number' in p.get('predicted_label', '') for p in row_context):
                    if 'Part number' in pred['predicted_label'] or 'Quantity' in pred['predicted_label']:
                        enhanced_pred['confidence'] = min(pred['confidence'] * 1.2, 1.0)
                        enhanced_pred['relation_boost'] = True
            
            # 키-값 관계 활용
            kv_relations = [r for r in relational_features['semantic_relations'] 
                          if r.relation_type == 'key_value' and r.target_id == pred['entity_id']]
            
            if kv_relations:
                enhanced_pred['is_value_of'] = kv_relations[0].source_id
                enhanced_pred['confidence'] = min(pred['confidence'] * 1.1, 1.0)
            
            enhanced_predictions.append(enhanced_pred)
        
        return enhanced_predictions
    
    def _get_row_context(self, entity_idx: int, predictions: List[Dict], rows: List[List[Dict]]) -> List[Dict]:
        """같은 행의 예측 결과 가져오기"""
        target_id = predictions[entity_idx].get('entity_id', '')
        
        for row in rows:
            row_ids = [e.get('entity_id', '') for e in row]
            if target_id in row_ids:
                return [p for p in predictions if p.get('entity_id', '') in row_ids]
        
        return []
    
    def train_with_relations(self, training_data: List[Dict]):
        """관계성 정보를 포함한 학습"""
        
        # 관계성 패턴 학습
        self.feature_extractor.learn_from_labeled_data(training_data)
        
        # 확장된 특징으로 모델 학습
        enhanced_training_data = []
        
        for doc in training_data:
            entities = doc.get('entities', [])
            
            # 관계성 특징 추출
            relational_features = self.feature_extractor.extract_enhanced_features(entities)
            
            # 각 엔티티에 관계성 특징 추가
            for entity in entities:
                entity['relational_features'] = self._get_entity_relational_features(
                    entity, 
                    relational_features
                )
            
            enhanced_doc = doc.copy()
            enhanced_doc['entities'] = entities
            enhanced_doc['document_relations'] = relational_features
            
            enhanced_training_data.append(enhanced_doc)
        
        # 하이브리드 모델 학습
        return self.hybrid_model.train(enhanced_training_data)
    
    def _get_entity_relational_features(self, entity: Dict, relational_features: Dict) -> Dict:
        """엔티티별 관계성 특징 추출"""
        entity_id = entity.get('entity_id', '')
        
        features = {
            'spatial_relations': [],
            'semantic_relations': [],
            'group_membership': [],
            'graph_centrality': 0
        }
        
        # 공간적 관계
        for relation in relational_features.get('spatial_relations', []):
            if relation.source_id == entity_id or relation.target_id == entity_id:
                features['spatial_relations'].append({
                    'type': relation.relation_type,
                    'distance': relation.distance,
                    'alignment': relation.alignment_score
                })
        
        # 의미적 관계
        for relation in relational_features.get('semantic_relations', []):
            if relation.source_id == entity_id or relation.target_id == entity_id:
                features['semantic_relations'].append({
                    'type': relation.relation_type,
                    'confidence': relation.confidence
                })
        
        # 그래프 중심성
        graph_features = relational_features.get('graph_features', {})
        centrality_scores = graph_features.get('top_central_nodes', [])
        for node_id, score in centrality_scores:
            if node_id == entity_id:
                features['graph_centrality'] = score
                break
        
        return features