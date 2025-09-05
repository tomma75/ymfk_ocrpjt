import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pytesseract
from PIL import Image
from datetime import datetime

from utils.logger_util import get_logger
from services.ocr_learning_service import OCRLearningService
from config import settings

class OCRIntegrationService:
    """OCR 통합 서비스 - 라벨 생성 시 실제 OCR 수행 및 학습 연동"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.ocr_learning_service = OCRLearningService()
        
        # Tesseract 경로 설정
        self._setup_tesseract()
        
    def _setup_tesseract(self):
        """Tesseract 경로 설정"""
        # 플랫폼에 따른 경로 설정
        import platform
        system = platform.system()
        
        if system == "Linux" or system == "Darwin":  # Linux or MacOS
            tesseract_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "tesseract"  # PATH에 있는 경우
            ]
        else:  # Windows
            tesseract_paths = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "tesseract", "tesseract.exe"),
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
        
        tesseract_found = False
        for path in tesseract_paths:
            try:
                if path == "tesseract":  # PATH에서 찾기
                    import subprocess
                    result = subprocess.run(["which", "tesseract"], capture_output=True, text=True)
                    if result.returncode == 0:
                        pytesseract.pytesseract.tesseract_cmd = "tesseract"
                        self.logger.info(f"Tesseract found in PATH")
                        tesseract_found = True
                        break
                elif os.path.exists(path) and os.access(path, os.X_OK):
                    pytesseract.pytesseract.tesseract_cmd = path
                    self.logger.info(f"Tesseract found at: {path}")
                    tesseract_found = True
                    break
            except Exception as e:
                self.logger.debug(f"Error checking tesseract at {path}: {e}")
                
        if not tesseract_found:
            self.logger.warning("Tesseract not found. OCR functionality may be limited.")
            # 기본값으로 설정 (시스템 PATH에 있을 수 있음)
            pytesseract.pytesseract.tesseract_cmd = "tesseract"
    
    def perform_ocr_on_bbox(self, image_path: str, bbox: Dict) -> Dict[str, Any]:
        """특정 bbox 영역에 대해 OCR 수행"""
        try:
            # 이미지 읽기
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return {
                    'ocr_text': '',
                    'ocr_confidence': 0.0,
                    'error': 'Failed to load image'
                }
            
            # BGR to RGB 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # bbox 영역 추출
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', bbox.get('w', 0))
            height = bbox.get('height', bbox.get('h', 0))
            
            # 영역 추출
            cropped = image[y:y+height, x:x+width]
            
            # 전처리
            cropped = self._preprocess_image(cropped)
            
            # OCR 수행
            try:
                ocr_text = pytesseract.image_to_string(cropped, lang='eng', config='--psm 7')
                ocr_data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 7')
                
                # 신뢰도 계산
                confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                return {
                    'ocr_text': ocr_text.strip(),
                    'ocr_confidence': avg_confidence / 100.0,
                    'success': True
                }
                
            except Exception as ocr_error:
                self.logger.error(f"OCR failed: {str(ocr_error)}")
                return {
                    'ocr_text': '',
                    'ocr_confidence': 0.0,
                    'error': str(ocr_error)
                }
                
        except Exception as e:
            self.logger.error(f"Error in perform_ocr_on_bbox: {str(e)}")
            return {
                'ocr_text': '',
                'ocr_confidence': 0.0,
                'error': str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """OCR을 위한 이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def process_label_with_ocr(self, image_path: str, label_data: Dict) -> Dict:
        """라벨 데이터에 OCR 수행하여 ocr_original 필드 추가"""
        
        # 라벨 데이터가 v1 형식인지 v2 형식인지 확인
        if 'bboxes' in label_data:
            # v1 형식
            return self._process_v1_label_with_ocr(image_path, label_data)
        elif 'entities' in label_data:
            # v2 형식
            return self._process_v2_label_with_ocr(image_path, label_data)
        else:
            self.logger.error("Unknown label format")
            return label_data
    
    def _process_v1_label_with_ocr(self, image_path: str, label_data: Dict) -> Dict:
        """v1 형식 라벨에 OCR 추가"""
        document_type = label_data.get('class', '')
        
        # bboxes 처리
        for bbox in label_data.get('bboxes', []):
            # OCR 수행 여부 결정
            should_ocr, predicted_label = self.ocr_learning_service.should_ocr_bbox(
                bbox, document_type, 'unknown'
            )
            
            if should_ocr:
                # OCR 수행
                ocr_result = self.perform_ocr_on_bbox(image_path, bbox)
                
                # OCR 결과를 학습 서비스로 처리
                if ocr_result.get('success'):
                    learning_result = self.ocr_learning_service.process_with_learning(
                        {
                            'text': ocr_result['ocr_text'],
                            'confidence': ocr_result['ocr_confidence']
                        },
                        bbox,
                        document_type,
                        'unknown'
                    )
                    
                    # 결과 저장
                    bbox['ocr_original'] = ocr_result['ocr_text']
                    bbox['ocr_confidence'] = ocr_result['ocr_confidence']
                    
                    # 이미 수정된 값이 있으면 그대로 유지
                    if bbox.get('was_corrected', False) and 'text' in bbox:
                        # 사용자가 이미 수정한 값이 있으면 그대로 유지
                        pass
                    elif learning_result.get('confidence', 0) > 0.8:
                        # 자동 보정된 텍스트가 있으면 적용
                        bbox['text'] = learning_result['corrected_text']
                        bbox['was_corrected'] = learning_result['corrected_text'] != ocr_result['ocr_text']
                    else:
                        # 신뢰도가 낮으면 OCR 원본 사용
                        bbox['text'] = ocr_result['ocr_text']
                        bbox['was_corrected'] = False
            else:
                # OCR 수행하지 않음
                bbox['ocr_original'] = ''
                bbox['ocr_confidence'] = 0.0
                bbox['was_corrected'] = False
        
        # items 배열도 동기화
        if 'items' in label_data:
            for item in label_data['items']:
                for label in item.get('labels', []):
                    # 해당하는 bbox 찾기
                    for bbox in label_data['bboxes']:
                        if (bbox['x'] == label['bbox'][0] and 
                            bbox['y'] == label['bbox'][1] and
                            bbox['label'] == label['label']):
                            label['ocr_original'] = bbox.get('ocr_original', '')
                            label['ocr_confidence'] = bbox.get('ocr_confidence', 0)
                            label['was_corrected'] = bbox.get('was_corrected', False)
                            label['text'] = bbox.get('text', '')
                            break
        
        return label_data
    
    def _process_v2_label_with_ocr(self, image_path: str, label_data: Dict) -> Dict:
        """v2 형식 라벨에 OCR 추가"""
        document_type = label_data.get('document_metadata', {}).get('document_type', '')
        layout_pattern = label_data.get('layout_analysis', {}).get('layout_pattern', 'unknown')
        
        # entities 처리
        for entity in label_data.get('entities', []):
            bbox = entity.get('bbox', {})
            
            # OCR 수행 여부 결정
            should_ocr, predicted_label = self.ocr_learning_service.should_ocr_bbox(
                bbox, document_type, layout_pattern
            )
            
            if should_ocr:
                # OCR 수행
                ocr_result = self.perform_ocr_on_bbox(image_path, bbox)
                
                if ocr_result.get('success'):
                    # OCR 결과 저장
                    if 'ocr_results' not in entity:
                        entity['ocr_results'] = {}
                    
                    entity['ocr_results']['tesseract_raw'] = ocr_result['ocr_text']
                    entity['ocr_results']['tesseract_confidence'] = ocr_result['ocr_confidence']
                    entity['ocr_results']['corrected_value'] = ocr_result['ocr_text']  # 초기값은 원본과 동일
                    
                    # 학습 서비스로 처리
                    learning_result = self.ocr_learning_service.process_with_learning(
                        {
                            'text': ocr_result['ocr_text'],
                            'confidence': ocr_result['ocr_confidence']
                        },
                        bbox,
                        document_type,
                        layout_pattern
                    )
                    
                    # 이미 수정된 값이 있는지 확인
                    was_corrected = entity.get('ocr_results', {}).get('was_corrected', False)
                    existing_corrected = entity.get('ocr_results', {}).get('corrected_value', '')
                    
                    # 이미 수정된 값이 있으면 그대로 유지
                    if was_corrected and existing_corrected and existing_corrected != ocr_result['ocr_text']:
                        entity['text']['value'] = existing_corrected
                        entity['ocr_results']['corrected_value'] = existing_corrected
                        entity['ocr_results']['was_corrected'] = True
                    elif learning_result.get('confidence', 0) > 0.8:
                        # 자동 보정 적용
                        entity['text']['value'] = learning_result['corrected_text']
                        entity['ocr_results']['corrected_value'] = learning_result['corrected_text']
                        entity['ocr_results']['was_corrected'] = learning_result['corrected_text'] != ocr_result['ocr_text']
                    else:
                        entity['text']['value'] = ocr_result['ocr_text']
                        entity['ocr_results']['was_corrected'] = False
                    
                    entity['text']['confidence'] = ocr_result['ocr_confidence']
        
        # OCR 엔진 정보 업데이트
        if 'document_metadata' in label_data:
            label_data['document_metadata']['ocr_engine'] = 'tesseract_with_learning'
            label_data['document_metadata']['ocr_confidence'] = self._calculate_avg_confidence(label_data)
        
        return label_data
    
    def _calculate_avg_confidence(self, label_data: Dict) -> float:
        """전체 OCR 신뢰도 계산"""
        confidences = []
        
        if 'entities' in label_data:
            for entity in label_data['entities']:
                ocr_conf = entity.get('ocr_results', {}).get('tesseract_confidence', 0)
                if ocr_conf > 0:
                    confidences.append(ocr_conf)
        elif 'bboxes' in label_data:
            for bbox in label_data['bboxes']:
                ocr_conf = bbox.get('ocr_confidence', 0)
                if ocr_conf > 0:
                    confidences.append(ocr_conf)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def update_labels_with_ocr(self, pdf_filename: str, page_number: int) -> bool:
        """기존 라벨 파일에 OCR 데이터 추가"""
        try:
            # 라벨 파일 경로
            label_filename = f"{Path(pdf_filename).stem}_page{page_number:03d}_label.json"
            label_path = Path(settings.PROCESSED_DATA_DIR) / "labels" / label_filename
            
            if not label_path.exists():
                self.logger.error(f"Label file not found: {label_path}")
                return False
            
            # 이미지 파일 경로
            image_filename = f"{Path(pdf_filename).stem}_page_{page_number:03d}.png"
            image_path = Path(settings.PROCESSED_DATA_DIR) / "images" / image_filename
            
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return False
            
            # 라벨 데이터 로드
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # OCR 수행
            updated_label_data = self.process_label_with_ocr(str(image_path), label_data)
            
            # 저장
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(updated_label_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Updated label with OCR: {label_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating labels with OCR: {str(e)}")
            return False