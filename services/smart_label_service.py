#!/usr/bin/env python3
"""
스마트 라벨 추천 서비스

이미지의 레이아웃과 기존 라벨링 데이터를 학습하여
새로운 이미지에 대해 라벨 위치와 타입을 자동으로 추천합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-09-02
버전: 1.0.0
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN

from core.base_classes import BaseService
from config.settings import ApplicationConfig
from core.exceptions import ServiceError, ProcessingError


class SmartLabelService(BaseService):
    """
    스마트 라벨 추천 서비스
    
    기존 라벨링 데이터를 학습하여 새로운 이미지에 대해
    자동으로 라벨 위치와 타입을 추천합니다.
    """
    
    def __init__(
        self,
        config: ApplicationConfig,
        logger: logging.Logger
    ):
        """
        스마트 라벨 서비스 초기화
        
        Args:
            config: 애플리케이션 설정
            logger: 로거 인스턴스
        """
        super().__init__(config, logger)
        
        # 모델 저장 경로
        self.model_path = Path(config.model_directory) / "smart_label"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터 디렉토리
        self.data_dir = Path(config.processed_data_directory) / "smart_labels"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 모델
        self.position_predictor = None  # 위치 예측 모델
        self.label_classifier = None    # 라벨 타입 분류 모델
        self.pattern_database = {}      # 문서 패턴 데이터베이스
        
        # 설정
        self.confidence_threshold = 0.6
        self.min_overlap_ratio = 0.3
        
        # 모델 로드
        self._load_models()
    
    def _load_models(self) -> None:
        """저장된 모델 로드"""
        try:
            # 위치 예측 모델
            position_model_file = self.model_path / "position_predictor.pkl"
            if position_model_file.exists():
                with open(position_model_file, 'rb') as f:
                    self.position_predictor = pickle.load(f)
                self.logger.info("위치 예측 모델 로드 완료")
            
            # 라벨 분류 모델
            label_model_file = self.model_path / "label_classifier.pkl"
            if label_model_file.exists():
                with open(label_model_file, 'rb') as f:
                    self.label_classifier = pickle.load(f)
                self.logger.info("라벨 분류 모델 로드 완료")
            
            # 패턴 데이터베이스
            pattern_file = self.model_path / "pattern_database.json"
            if pattern_file.exists():
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    self.pattern_database = json.load(f)
                self.logger.info(f"패턴 데이터베이스 로드: {len(self.pattern_database)} 패턴")
        
        except Exception as e:
            self.logger.warning(f"모델 로드 실패, 새로운 모델로 시작: {str(e)}")
            self._initialize_default_models()
    
    def _initialize_default_models(self) -> None:
        """기본 모델 초기화"""
        # 랜덤 포레스트 기반 모델
        self.position_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.label_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # 기본 패턴 데이터베이스
        self.pattern_database = {
            "purchase_order": {
                "common_labels": [
                    {"label": "Order number", "typical_position": {"top": 0.1, "left": 0.1}},
                    {"label": "Delivery date", "typical_position": {"top": 0.15, "left": 0.1}},
                    {"label": "Item number", "typical_position": {"top": 0.3, "left": 0.1}},
                    {"label": "Quantity", "typical_position": {"top": 0.3, "left": 0.5}},
                    {"label": "Unit price", "typical_position": {"top": 0.3, "left": 0.6}},
                    {"label": "Net amount (total)", "typical_position": {"top": 0.8, "left": 0.7}}
                ]
            },
            "invoice": {
                "common_labels": [
                    {"label": "Invoice number", "typical_position": {"top": 0.1, "left": 0.1}},
                    {"label": "Invoice date", "typical_position": {"top": 0.15, "left": 0.1}},
                    {"label": "Total amount", "typical_position": {"top": 0.85, "left": 0.7}}
                ]
            }
        }
        
        self.logger.info("기본 스마트 라벨 모델 초기화 완료")
    
    def predict_labels(
        self,
        image_path: str,
        document_type: Optional[str] = None,
        existing_labels: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        이미지에 대한 라벨 예측
        
        Args:
            image_path: 이미지 파일 경로
            document_type: 문서 타입 (purchase_order, invoice 등)
            existing_labels: 기존 라벨 정보 (있는 경우)
            
        Returns:
            예측된 라벨 리스트
        """
        try:
            self.logger.info(f"라벨 예측 시작: {image_path}")
            
            # 이미지 정보 추출
            image_features = self._extract_image_features(image_path)
            image_features['image_path'] = image_path
            
            # 문서 타입 추론 (제공되지 않은 경우)
            if not document_type:
                document_type = self._infer_document_type(image_features)
            
            # 패턴 기반 예측
            pattern_predictions = self._get_pattern_based_predictions(
                document_type,
                image_features
            )
            
            # ML 모델 기반 예측 (학습된 경우)
            ml_predictions = []
            if self._is_model_trained():
                ml_predictions = self._get_ml_based_predictions(image_features)
            
            # 예측 결과 병합 및 최적화
            predictions = self._merge_predictions(
                pattern_predictions,
                ml_predictions,
                existing_labels
            )
            
            # 신뢰도 기반 필터링
            filtered_predictions = [
                p for p in predictions
                if p.get('confidence', 0) >= self.confidence_threshold
            ]
            
            self.logger.info(f"예측 완료: {len(filtered_predictions)}개 라벨")
            return filtered_predictions
            
        except Exception as e:
            self.logger.error(f"라벨 예측 실패: {str(e)}")
            raise ProcessingError(f"라벨 예측 실패: {str(e)}")
    
    def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """이미지에서 특징 추출"""
        features = {
            "path": image_path,
            "filename": Path(image_path).name,
            "size": None,
            "text_regions": [],
            "layout": None
        }
        
        # 이미지 크기 정보
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                features["size"] = {"width": img.width, "height": img.height}
        except Exception as e:
            self.logger.warning(f"이미지 크기 정보 추출 실패: {str(e)}")
        
        # OCR로 텍스트 영역 감지 (간단한 구현)
        # 실제로는 더 복잡한 레이아웃 분석 필요
        features["text_regions"] = self._detect_text_regions(image_path)
        
        return features
    
    def _detect_text_regions(self, image_path: str) -> List[Dict]:
        """텍스트 영역 감지 (간단한 구현)"""
        # 실제로는 OCR 엔진을 사용하여 텍스트 영역 감지
        # 여기서는 더미 데이터 반환
        return [
            {"x": 50, "y": 50, "width": 200, "height": 30, "text": "Order Number"},
            {"x": 50, "y": 100, "width": 150, "height": 30, "text": "Date"},
            {"x": 50, "y": 200, "width": 500, "height": 200, "text": "Items"}
        ]
    
    def _infer_document_type(self, features: Dict) -> str:
        """문서 타입 추론"""
        # 파일명이나 텍스트 내용을 기반으로 문서 타입 추론
        filename = features.get("filename", "").lower()
        
        if "po" in filename or "purchase" in filename:
            return "purchase_order"
        elif "invoice" in filename:
            return "invoice"
        elif "receipt" in filename:
            return "receipt"
        
        return "other"
    
    def _get_pattern_based_predictions(
        self,
        document_type: str,
        features: Dict
    ) -> List[Dict]:
        """패턴 기반 예측"""
        predictions = []
        
        # 문서 타입에 해당하는 패턴 가져오기
        patterns = self.pattern_database.get(document_type, {})
        common_labels = patterns.get("common_labels", [])
        
        image_size = features.get("size", {"width": 1000, "height": 1000})
        
        for label_pattern in common_labels:
            # 상대 위치를 절대 위치로 변환
            position = label_pattern.get("typical_position", {})
            x = int(position.get("left", 0.5) * image_size["width"])
            y = int(position.get("top", 0.5) * image_size["height"])
            
            # 기본 크기 설정
            width = 150
            height = 30
            
            predictions.append({
                "label": label_pattern["label"],
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                },
                "confidence": 0.7,  # 패턴 기반 기본 신뢰도
                "source": "pattern"
            })
        
        return predictions
    
    def _is_model_trained(self) -> bool:
        """모델이 학습되었는지 확인"""
        try:
            if self.position_predictor and hasattr(self.position_predictor, 'n_features_in_'):
                return self.position_predictor.n_features_in_ > 0
        except:
            pass
        return False
    
    def _get_ml_based_predictions(self, features: Dict) -> List[Dict]:
        """ML 모델 기반 예측 (하이브리드 모델 사용)"""
        predictions = []
        
        try:
            # HybridOCRLabeler 사용하여 예측
            from .hybrid_ocr_labeler import HybridOCRLabeler
            
            # 하이브리드 라벨러가 없으면 초기화
            if not hasattr(self, 'hybrid_labeler'):
                self.hybrid_labeler = HybridOCRLabeler(
                    config=self.config, 
                    logger=self.logger
                )
                if self.hybrid_labeler.initialize():
                    self.logger.info("하이브리드 라벨러 초기화 성공")
                else:
                    self.logger.warning("하이브리드 라벨러 초기화 실패")
                    return []
            
            # 이미지에서 OCR 엔티티 추출
            image_path = features.get('image_path')
            if image_path and Path(image_path).exists():
                # OCR을 통해 텍스트 박스들 추출 (간단한 더미 엔티티 생성)
                entities = self._extract_ocr_entities(image_path)
                
                if entities:
                    # 하이브리드 모델로 라벨 예측
                    hybrid_predictions = self.hybrid_labeler.predict(entities)
                    
                    for i, pred in enumerate(hybrid_predictions):
                        if i < len(entities):
                            predictions.append({
                                "label": pred.get('predicted_label', ''),
                                "bbox": entities[i].get('bbox', {}),
                                "confidence": pred.get('confidence', 0.0),
                                "source": "hybrid_ml"
                            })
                    
                    self.logger.info(f"하이브리드 모델로 {len(predictions)}개 라벨 예측")
                
        except Exception as e:
            self.logger.warning(f"하이브리드 모델 예측 실패: {e}")
        
        return predictions
        
    def _extract_ocr_entities(self, image_path: str) -> List[Dict]:
        """이미지에서 OCR 엔티티 추출"""
        try:
            # 기존 어노테이션 파일에서 엔티티 추출
            image_name = Path(image_path).stem
            annotation_files = list(Path('data/annotations').glob(f"*{image_name}*.json"))
            
            if annotation_files:
                # 가장 최근 어노테이션 파일 사용
                annotation_file = annotation_files[0]
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)
                
                entities = []
                # 어노테이션 형식 확인 후 엔티티 추출
                if 'items' in annotation_data:
                    # items/labels 형식
                    for item in annotation_data.get('items', []):
                        for label_info in item.get('labels', []):
                            bbox_list = label_info.get('bbox', [])
                            if len(bbox_list) >= 4:
                                bbox_dict = {
                                    'x': bbox_list[0], 
                                    'y': bbox_list[1], 
                                    'width': bbox_list[2], 
                                    'height': bbox_list[3]
                                }
                            else:
                                bbox_dict = {'x': 0, 'y': 0, 'width': 100, 'height': 30}
                                
                            entities.append({
                                'entity_id': f"entity_{len(entities)}",
                                'bbox': bbox_dict,
                                'text': label_info.get('text', '')
                            })
                elif 'pages' in annotation_data:
                    # pages/regions 형식 
                    for page in annotation_data.get('pages', []):
                        for region in page.get('regions', []):
                            entities.append({
                                'entity_id': region.get('id', f"region_{len(entities)}"),
                                'bbox': region.get('bbox', {}),
                                'text': region.get('text', '')
                            })
                
                self.logger.info(f"어노테이션에서 {len(entities)}개 엔티티 추출")
                return entities
            
            # 어노테이션이 없으면 기본 더미 엔티티 생성
            return [
                {'entity_id': 'dummy_1', 'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 30}, 'text': 'Order Number'},
                {'entity_id': 'dummy_2', 'bbox': {'x': 100, 'y': 200, 'width': 150, 'height': 30}, 'text': 'Date'},
            ]
            
        except Exception as e:
            self.logger.warning(f"엔티티 추출 실패: {e}")
            return []

    def _load_pattern_database(self):
        """패턴 데이터베이스 로드"""
        try:
            pattern_file = self.model_path / "pattern_database.json"
            if pattern_file.exists():
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    self.pattern_database = json.load(f)
                self.logger.info(f"패턴 데이터베이스 로드: {len(self.pattern_database)} 패턴")
            else:
                self.logger.info("패턴 데이터베이스 파일이 없어 기본값 사용")
        except Exception as e:
            self.logger.warning(f"패턴 데이터베이스 로드 실패: {e}")
    
    def _merge_predictions(
        self,
        pattern_predictions: List[Dict],
        ml_predictions: List[Dict],
        existing_labels: Optional[List[Dict]]
    ) -> List[Dict]:
        """예측 결과 병합"""
        merged = []
        
        # 패턴 기반 예측 추가
        for pred in pattern_predictions:
            pred["id"] = f"smart_{len(merged)}"
            merged.append(pred)
        
        # ML 기반 예측 추가 (중복 제거)
        for ml_pred in ml_predictions:
            is_duplicate = False
            for existing in merged:
                if self._is_overlap(ml_pred["bbox"], existing["bbox"]):
                    # 신뢰도가 더 높은 것으로 대체
                    if ml_pred.get("confidence", 0) > existing.get("confidence", 0):
                        existing.update(ml_pred)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                ml_pred["id"] = f"smart_{len(merged)}"
                merged.append(ml_pred)
        
        # 기존 라벨과 중복되는 예측 제거
        if existing_labels:
            filtered = []
            for pred in merged:
                is_duplicate = False
                for existing in existing_labels:
                    if self._is_overlap(pred["bbox"], existing.get("bbox", {})):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered.append(pred)
            merged = filtered
        
        return merged
    
    def _is_overlap(self, bbox1: Dict, bbox2: Dict) -> bool:
        """두 바운딩 박스가 겹치는지 확인"""
        if not bbox1 or not bbox2:
            return False
        
        # 박스 좌표 계산
        x1_min = bbox1.get("x", 0)
        y1_min = bbox1.get("y", 0)
        x1_max = x1_min + bbox1.get("width", 0)
        y1_max = y1_min + bbox1.get("height", 0)
        
        x2_min = bbox2.get("x", 0)
        y2_min = bbox2.get("y", 0)
        x2_max = x2_min + bbox2.get("width", 0)
        y2_max = y2_min + bbox2.get("height", 0)
        
        # 겹침 영역 계산
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = x_overlap * y_overlap
        
        # 각 박스의 면적
        area1 = bbox1.get("width", 0) * bbox1.get("height", 0)
        area2 = bbox2.get("width", 0) * bbox2.get("height", 0)
        
        # 겹침 비율 계산 (IoU)
        if area1 + area2 - overlap_area > 0:
            iou = overlap_area / (area1 + area2 - overlap_area)
            return iou > self.min_overlap_ratio
        
        return False
    
    def train_from_labels(self, training_data: List[Dict]) -> Dict[str, Any]:
        """
        라벨링 데이터로부터 모델 학습
        
        Args:
            training_data: 학습 데이터 리스트
            
        Returns:
            학습 결과 정보
        """
        try:
            self.logger.info(f"스마트 라벨 모델 학습 시작: {len(training_data)}개 샘플")
            
            if len(training_data) < 10:
                return {
                    "status": "insufficient_data",
                    "message": "최소 10개 이상의 학습 샘플이 필요합니다",
                    "samples_count": len(training_data)
                }
            
            # 패턴 데이터베이스 업데이트
            self._update_pattern_database(training_data)
            
            # ML 모델 학습 (충분한 데이터가 있는 경우)
            if len(training_data) >= 50:
                self._train_ml_models(training_data)
            
            # 모델 저장
            self._save_models()
            
            result = {
                "status": "success",
                "samples_count": len(training_data),
                "patterns_count": sum(
                    len(p.get("common_labels", []))
                    for p in self.pattern_database.values()
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"모델 학습 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"모델 학습 실패: {str(e)}")
            raise ServiceError(f"모델 학습 실패: {str(e)}")
    
    def _update_pattern_database(self, training_data: List[Dict]) -> None:
        """패턴 데이터베이스 업데이트"""
        # 문서 타입별로 그룹화
        grouped_data = {}
        for sample in training_data:
            doc_type = sample.get("document_type", "other")
            if doc_type not in grouped_data:
                grouped_data[doc_type] = []
            grouped_data[doc_type].append(sample)
        
        # 각 문서 타입별로 패턴 분석
        for doc_type, samples in grouped_data.items():
            if doc_type not in self.pattern_database:
                self.pattern_database[doc_type] = {"common_labels": []}
            
            # 라벨별 위치 통계 계산
            label_positions = {}
            for sample in samples:
                for bbox in sample.get("bboxes", []):
                    label = bbox.get("label")
                    if label:
                        if label not in label_positions:
                            label_positions[label] = []
                        
                        # 상대 위치 계산
                        image_size = sample.get("image_size", {"width": 1000, "height": 1000})
                        rel_x = bbox.get("x", 0) / image_size["width"]
                        rel_y = bbox.get("y", 0) / image_size["height"]
                        label_positions[label].append({"x": rel_x, "y": rel_y})
            
            # 평균 위치 계산
            common_labels = []
            for label, positions in label_positions.items():
                if len(positions) >= 3:  # 최소 3번 이상 나타난 라벨만
                    avg_x = np.mean([p["x"] for p in positions])
                    avg_y = np.mean([p["y"] for p in positions])
                    
                    common_labels.append({
                        "label": label,
                        "typical_position": {"left": avg_x, "top": avg_y},
                        "frequency": len(positions)
                    })
            
            # 빈도순으로 정렬
            common_labels.sort(key=lambda x: x["frequency"], reverse=True)
            self.pattern_database[doc_type]["common_labels"] = common_labels[:20]  # 상위 20개만
    
    def _train_ml_models(self, training_data: List[Dict]) -> None:
        """ML 모델 학습"""
        # 실제로는 더 복잡한 특징 추출과 학습이 필요
        # 여기서는 간단한 구현
        self.logger.info("ML 모델 학습 생략 (추가 구현 필요)")
    
    def _save_models(self) -> None:
        """모델 저장"""
        try:
            # 패턴 데이터베이스 저장
            pattern_file = self.model_path / "pattern_database.json"
            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(self.pattern_database, f, ensure_ascii=False, indent=2)
            
            # ML 모델 저장 (학습된 경우)
            if self._is_model_trained():
                position_model_file = self.model_path / "position_predictor.pkl"
                with open(position_model_file, 'wb') as f:
                    pickle.dump(self.position_predictor, f)
                
                label_model_file = self.model_path / "label_classifier.pkl"
                with open(label_model_file, 'wb') as f:
                    pickle.dump(self.label_classifier, f)
            
            self.logger.info("모델 저장 완료")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """스마트 라벨 통계 정보 반환"""
        stats = {
            "patterns_count": len(self.pattern_database),
            "total_label_types": sum(
                len(p.get("common_labels", []))
                for p in self.pattern_database.values()
            ),
            "confidence_threshold": self.confidence_threshold,
            "model_trained": self._is_model_trained()
        }
        
        # 각 문서 타입별 라벨 수
        stats["document_types"] = {}
        for doc_type, patterns in self.pattern_database.items():
            stats["document_types"][doc_type] = len(patterns.get("common_labels", []))
        
        return stats
    
    def update_from_feedback(self, filename: str, prediction: Dict[str, Any], accepted: bool) -> None:
        """
        사용자 피드백으로부터 학습
        
        Args:
            filename: 파일명
            prediction: 예측 결과
            accepted: 사용자 수락 여부
        """
        feedback_entry = {
            'filename': filename,
            'prediction': prediction,
            'accepted': accepted,
            'timestamp': datetime.now().isoformat()
        }
        
        # 피드백 데이터 저장
        feedback_file = self.data_dir / 'feedback.json'
        feedback_data = []
        
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        
        feedback_data.append(feedback_entry)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        # 수락된 예측을 패턴 DB에 추가
        if accepted and prediction:
            doc_type = prediction.get('document_type', 'purchase_order')
            
            if doc_type not in self.pattern_database:
                self.pattern_database[doc_type] = {
                    'common_labels': [],
                    'layout_patterns': []
                }
            
            # 라벨 타입 추가
            label_type = prediction.get('label_type')
            if label_type and label_type not in self.pattern_database[doc_type]['common_labels']:
                self.pattern_database[doc_type]['common_labels'].append(label_type)
            
            # 레이아웃 패턴 추가
            layout = {
                'x': prediction.get('x', 0),
                'y': prediction.get('y', 0),
                'width': prediction.get('width', 0),
                'height': prediction.get('height', 0),
                'label': label_type,
                'confidence': prediction.get('confidence', 0.5)
            }
            self.pattern_database[doc_type]['layout_patterns'].append(layout)
            
            self._save_pattern_database()
        
        self.logger.info(f"Feedback updated for {filename}: {'Accepted' if accepted else 'Rejected'}")
    
    def train_model(self, use_existing_data: bool = True) -> Dict[str, Any]:
        """
        스마트 라벨 모델 학습
        
        Args:
            use_existing_data: 기존 데이터 사용 여부
            
        Returns:
            학습 결과 정보
        """
        try:
            training_samples = 0
            
            if use_existing_data:
                # 기존 라벨 데이터에서 학습
                labels_dir = self.data_dir.parent / 'labels'
                if labels_dir.exists():
                    for label_file in labels_dir.glob('*.json'):
                        with open(label_file, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            
                        # 학습 데이터 추출
                        if 'bboxes' in label_data:
                            for bbox in label_data['bboxes']:
                                if 'label' in bbox:
                                    training_samples += 1
            
            # 피드백 데이터에서도 학습
            feedback_file = self.data_dir / 'feedback.json'
            if feedback_file.exists():
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    training_samples += len([f for f in feedback_data if f.get('accepted')])
            
            return {
                'status': 'success',
                'training_samples': training_samples,
                'model_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train model: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self.logger.info("Initializing SmartLabelService")
            # 데이터 디렉토리 확인
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # 패턴 데이터베이스 로드
            self._load_pattern_database()
            
            self.logger.info("SmartLabelService initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize SmartLabelService: {e}")
            return False
    
    def cleanup(self) -> None:
        """서비스 정리"""
        try:
            self.logger.info("Cleaning up SmartLabelService")
            # 패턴 데이터베이스 저장
            self._save_pattern_database()
            # 모델 저장
            self._save_models()
            self.logger.info("SmartLabelService cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during SmartLabelService cleanup: {e}")
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 패턴 데이터베이스 확인
            has_patterns = bool(self.pattern_database)
            
            # 디렉토리 접근 가능 여부 확인
            can_access_dir = self.data_dir.exists()
            
            return has_patterns and can_access_dir
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False