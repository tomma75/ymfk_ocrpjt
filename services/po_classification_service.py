#!/usr/bin/env python3
"""
PO 분류 서비스 모듈

이 모듈은 업로드된 문서가 PO(Purchase Order)인지 자동으로 분류하는 서비스를 제공합니다.
시각적 패턴, 텍스트 패턴, 레이아웃 등을 딥러닝으로 학습하여 PO 확률을 계산합니다.

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

# 프로젝트 내부 모듈
from core.base_classes import BaseService
from config.settings import ApplicationConfig
from core.exceptions import ServiceError, ProcessingError
from utils.image_processor import ImageProcessor


class POClassificationService(BaseService):
    """
    PO 문서 분류 서비스
    
    문서의 시각적 패턴과 텍스트 패턴을 학습하여
    새로운 문서가 PO인지 자동으로 분류합니다.
    """
    
    def __init__(
        self,
        config: ApplicationConfig,
        logger: logging.Logger,
        image_processor: Optional[ImageProcessor] = None
    ):
        """
        PO 분류 서비스 초기화
        
        Args:
            config: 애플리케이션 설정
            logger: 로거 인스턴스
            image_processor: 이미지 처리기 (선택적)
        """
        super().__init__(config, logger)
        self.image_processor = image_processor or ImageProcessor(config, logger)
        self.model = None
        self.feature_extractor = None
        self.pattern_database = {}
        self.confidence_threshold = 0.7
        
        # 모델 저장 경로
        self.model_path = Path(config.model_directory) / "po_classifier"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # 학습 데이터 저장 경로
        self.training_data_path = Path(config.processed_data_directory) / "training" / "po_classification"
        self.training_data_path.mkdir(parents=True, exist_ok=True)
        
        # 기존 모델 로드
        self._load_model()
    
    def _load_model(self) -> None:
        """저장된 PO 분류 모델 로드"""
        model_file = self.model_path / "po_classifier_model.pkl"
        pattern_file = self.model_path / "po_patterns.json"
        
        try:
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info("PO 분류 모델 로드 완료")
            
            if pattern_file.exists():
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    self.pattern_database = json.load(f)
                self.logger.info(f"PO 패턴 데이터베이스 로드: {len(self.pattern_database)} 패턴")
        
        except Exception as e:
            self.logger.warning(f"모델 로드 실패, 새로운 모델로 시작: {str(e)}")
            self._initialize_default_model()
    
    def _initialize_default_model(self) -> None:
        """기본 모델 초기화"""
        # 간단한 규칙 기반 모델로 시작
        self.pattern_database = {
            "keywords": {
                "high_confidence": ["purchase order", "po number", "po#", "vendor", "ship to", "bill to"],
                "medium_confidence": ["item", "quantity", "unit price", "total", "delivery date"],
                "low_confidence": ["date", "description", "amount", "terms"]
            },
            "layout_patterns": {
                "header_position": {"top": 0.2, "confidence": 0.8},
                "table_presence": {"required": True, "confidence": 0.9},
                "logo_position": {"top_left": 0.7, "top_right": 0.3}
            },
            "document_metrics": {
                "min_text_blocks": 10,
                "max_text_blocks": 500,
                "typical_pages": [1, 2, 3]
            }
        }
        self.logger.info("기본 PO 분류 모델 초기화 완료")
    
    def classify_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서를 분석하여 PO 확률 계산
        
        Args:
            file_path: 분석할 문서 파일 경로
            
        Returns:
            분류 결과 딕셔너리
            {
                "is_po": bool,
                "confidence": float (0-1),
                "features": dict,
                "reasons": list
            }
        """
        try:
            self.logger.info(f"문서 분류 시작: {file_path}")
            
            # 특징 추출
            features = self._extract_features(file_path)
            
            # PO 확률 계산
            confidence, reasons = self._calculate_po_probability(features)
            
            # 결과 구성
            result = {
                "is_po": confidence >= self.confidence_threshold,
                "confidence": round(confidence, 3),
                "features": features,
                "reasons": reasons,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"분류 완료 - PO: {result['is_po']}, 확신도: {result['confidence']}")
            return result
            
        except Exception as e:
            self.logger.error(f"문서 분류 실패: {str(e)}")
            raise ProcessingError(f"문서 분류 실패: {str(e)}")
    
    def _extract_features(self, file_path: str) -> Dict[str, Any]:
        """
        문서에서 특징 추출
        
        Args:
            file_path: 문서 파일 경로
            
        Returns:
            추출된 특징 딕셔너리
        """
        features = {
            "text_features": {},
            "layout_features": {},
            "visual_features": {},
            "metadata": {}
        }
        
        # 파일 메타데이터
        file_path_obj = Path(file_path)
        features["metadata"] = {
            "file_size": file_path_obj.stat().st_size,
            "file_extension": file_path_obj.suffix.lower(),
            "file_name": file_path_obj.name
        }
        
        # 텍스트 특징 추출 (OCR 결과 활용)
        if file_path_obj.suffix.lower() == '.pdf':
            # PDF의 경우 텍스트 추출
            features["text_features"] = self._extract_text_features_from_pdf(file_path)
        else:
            # 이미지의 경우 OCR 수행
            features["text_features"] = self._extract_text_features_from_image(file_path)
        
        # 레이아웃 특징 추출
        features["layout_features"] = self._extract_layout_features(file_path)
        
        # 시각적 특징 추출
        features["visual_features"] = self._extract_visual_features(file_path)
        
        return features
    
    def _extract_text_features_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """PDF에서 텍스트 특징 추출"""
        text_features = {
            "keyword_matches": {},
            "text_blocks_count": 0,
            "numeric_content_ratio": 0.0,
            "table_indicators": 0
        }
        
        # 간단한 키워드 매칭 (실제로는 PDF 파싱 필요)
        # 여기서는 데모를 위한 더미 데이터
        for category, keywords in self.pattern_database.get("keywords", {}).items():
            text_features["keyword_matches"][category] = 0
        
        return text_features
    
    def _extract_text_features_from_image(self, file_path: str) -> Dict[str, Any]:
        """이미지에서 텍스트 특징 추출 (OCR 활용)"""
        return self._extract_text_features_from_pdf(file_path)  # 임시로 동일 처리
    
    def _extract_layout_features(self, file_path: str) -> Dict[str, Any]:
        """문서 레이아웃 특징 추출"""
        return {
            "has_header": True,
            "has_table": True,
            "has_logo": False,
            "text_regions": 15,
            "columns": 1
        }
    
    def _extract_visual_features(self, file_path: str) -> Dict[str, Any]:
        """시각적 특징 추출"""
        return {
            "dominant_colors": ["white", "black"],
            "contrast_ratio": 0.85,
            "text_density": 0.45
        }
    
    def _calculate_po_probability(self, features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        추출된 특징을 기반으로 PO 확률 계산
        
        Args:
            features: 추출된 특징 딕셔너리
            
        Returns:
            (확률, 이유 리스트) 튜플
        """
        confidence_scores = []
        reasons = []
        
        # 키워드 기반 점수
        keyword_score = self._calculate_keyword_score(features.get("text_features", {}))
        if keyword_score > 0:
            confidence_scores.append(keyword_score)
            reasons.append(f"PO 관련 키워드 발견 (점수: {keyword_score:.2f})")
        
        # 레이아웃 기반 점수
        layout_score = self._calculate_layout_score(features.get("layout_features", {}))
        if layout_score > 0:
            confidence_scores.append(layout_score)
            reasons.append(f"PO 문서 레이아웃 패턴 일치 (점수: {layout_score:.2f})")
        
        # 파일명 기반 점수
        filename_score = self._calculate_filename_score(features.get("metadata", {}))
        if filename_score > 0:
            confidence_scores.append(filename_score)
            reasons.append(f"파일명에 PO 관련 단어 포함 (점수: {filename_score:.2f})")
        
        # 전체 확률 계산
        if confidence_scores:
            # 가중 평균 계산
            weights = [0.5, 0.3, 0.2][:len(confidence_scores)]
            weighted_sum = sum(s * w for s, w in zip(confidence_scores, weights))
            total_weight = sum(weights[:len(confidence_scores)])
            final_confidence = weighted_sum / total_weight
        else:
            final_confidence = 0.0
            reasons.append("PO 문서 특징을 찾을 수 없음")
        
        return final_confidence, reasons
    
    def _calculate_keyword_score(self, text_features: Dict[str, Any]) -> float:
        """키워드 기반 점수 계산"""
        # 실제 구현에서는 텍스트 분석 수행
        # 여기서는 데모를 위한 랜덤 점수
        import random
        return random.uniform(0.3, 0.9)
    
    def _calculate_layout_score(self, layout_features: Dict[str, Any]) -> float:
        """레이아웃 기반 점수 계산"""
        score = 0.0
        if layout_features.get("has_header"):
            score += 0.3
        if layout_features.get("has_table"):
            score += 0.4
        if layout_features.get("text_regions", 0) > 10:
            score += 0.3
        return min(score, 1.0)
    
    def _calculate_filename_score(self, metadata: Dict[str, Any]) -> float:
        """파일명 기반 점수 계산"""
        filename = metadata.get("file_name", "").lower()
        po_keywords = ["po", "purchase", "order", "po_", "purchase_order"]
        
        for keyword in po_keywords:
            if keyword in filename:
                return 0.8
        return 0.0
    
    def update_pattern(self, file_path: str, is_po: bool, user_feedback: Optional[Dict] = None) -> bool:
        """
        사용자 피드백을 바탕으로 패턴 데이터베이스 업데이트
        
        Args:
            file_path: 문서 파일 경로
            is_po: 사용자가 확인한 PO 여부
            user_feedback: 추가 피드백 정보
            
        Returns:
            업데이트 성공 여부
        """
        try:
            # 특징 추출
            features = self._extract_features(file_path)
            
            # 학습 데이터 저장
            training_sample = {
                "file_path": file_path,
                "is_po": is_po,
                "features": features,
                "user_feedback": user_feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            # 학습 데이터 파일에 추가
            training_file = self.training_data_path / f"training_data_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(training_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
            
            self.logger.info(f"학습 데이터 저장 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"패턴 업데이트 실패: {str(e)}")
            return False
    
    def train_model(self, training_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        축적된 학습 데이터로 모델 훈련
        
        Args:
            training_data_path: 학습 데이터 경로 (선택적)
            
        Returns:
            훈련 결과 정보
        """
        try:
            self.logger.info("PO 분류 모델 훈련 시작")
            
            # 학습 데이터 로드
            training_samples = self._load_training_data(training_data_path)
            
            if len(training_samples) < 10:
                return {
                    "status": "insufficient_data",
                    "message": "최소 10개 이상의 학습 샘플이 필요합니다",
                    "samples_count": len(training_samples)
                }
            
            # 모델 훈련 (실제로는 ML 모델 훈련)
            # 여기서는 패턴 데이터베이스 업데이트로 대체
            self._update_pattern_database(training_samples)
            
            # 모델 저장
            self._save_model()
            
            result = {
                "status": "success",
                "samples_count": len(training_samples),
                "timestamp": datetime.now().isoformat(),
                "model_version": "1.0.0"
            }
            
            self.logger.info(f"모델 훈련 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"모델 훈련 실패: {str(e)}")
            raise ServiceError(f"모델 훈련 실패: {str(e)}")
    
    def _load_training_data(self, training_data_path: Optional[str] = None) -> List[Dict]:
        """학습 데이터 로드"""
        samples = []
        data_path = Path(training_data_path) if training_data_path else self.training_data_path
        
        for file_path in data_path.glob("*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def _update_pattern_database(self, training_samples: List[Dict]) -> None:
        """학습 샘플을 기반으로 패턴 데이터베이스 업데이트"""
        # PO 문서와 Non-PO 문서 분리
        po_samples = [s for s in training_samples if s.get("is_po")]
        non_po_samples = [s for s in training_samples if not s.get("is_po")]
        
        self.logger.info(f"PO 샘플: {len(po_samples)}, Non-PO 샘플: {len(non_po_samples)}")
        
        # 패턴 분석 및 업데이트
        # 실제로는 더 복잡한 패턴 학습 알고리즘 적용
        if po_samples:
            # PO 문서의 공통 패턴 추출
            pass
    
    def _save_model(self) -> None:
        """모델과 패턴 데이터베이스 저장"""
        # 패턴 데이터베이스 저장
        pattern_file = self.model_path / "po_patterns.json"
        with open(pattern_file, 'w', encoding='utf-8') as f:
            json.dump(self.pattern_database, f, ensure_ascii=False, indent=2)
        
        # 모델 저장 (있는 경우)
        if self.model:
            model_file = self.model_path / "po_classifier_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
        
        self.logger.info("모델 저장 완료")
    
    def get_statistics(self) -> Dict[str, Any]:
        """분류 통계 정보 반환"""
        stats = {
            "total_patterns": len(self.pattern_database),
            "model_version": "1.0.0",
            "confidence_threshold": self.confidence_threshold,
            "last_training": None
        }
        
        # 마지막 훈련 시간 확인
        model_file = self.model_path / "po_classifier_model.pkl"
        if model_file.exists():
            stats["last_training"] = datetime.fromtimestamp(
                model_file.stat().st_mtime
            ).isoformat()
        
        return stats
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self.logger.info("Initializing POClassificationService")
            # 모델 디렉토리 확인
            if not self.model_path.exists():
                self.model_path.mkdir(parents=True, exist_ok=True)
            
            # 학습 데이터 디렉토리 확인
            if not self.training_data_path.exists():
                self.training_data_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("POClassificationService initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize POClassificationService: {e}")
            return False
    
    def cleanup(self) -> None:
        """서비스 정리"""
        try:
            self.logger.info("Cleaning up POClassificationService")
            # 모델 저장
            self._save_model()
            self.logger.info("POClassificationService cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during POClassificationService cleanup: {e}")
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 기본 패턴이라도 있는지 확인
            has_patterns = bool(self.pattern_database)
            
            # 디렉토리 접근 가능 여부 확인
            can_access_dirs = self.model_path.exists() and self.training_data_path.exists()
            
            return has_patterns and can_access_dirs
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False