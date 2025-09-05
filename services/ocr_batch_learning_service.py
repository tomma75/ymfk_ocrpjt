import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict
import joblib

from utils.logger_util import get_logger
from services.ocr_learning_service import OCRLearningService
from config import get_application_config

class OCRBatchLearningService:
    """OCR 일괄 학습 서비스 - 누적된 라벨 데이터로부터 일괄 학습"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.ocr_learning_service = OCRLearningService()
        
        # config에서 경로 가져오기
        config = get_application_config()
        self.label_dir = Path(config.processed_data_directory) / "labels"
        self.label_v2_dir = Path(config.processed_data_directory) / "labels_v2"
        
    def collect_training_data(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """모든 라벨 파일에서 학습 데이터 수집"""
        training_data = []
        statistics = {
            'total_files': 0,
            'total_corrections': 0,
            'corrections_by_label': defaultdict(int),
            'ocr_errors': defaultdict(int)
        }
        
        # 일반 라벨 파일 처리
        for label_file in self.label_dir.glob("*_label.json"):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                statistics['total_files'] += 1
                
                # bboxes에서 OCR 수정 데이터 수집
                for bbox in data.get('bboxes', []):
                    ocr_original = bbox.get('ocr_original', '')
                    text = bbox.get('text', '')
                    
                    # OCR 원본과 수정된 텍스트가 다른 경우
                    if ocr_original and text and ocr_original != text:
                        training_item = {
                            'file': label_file.name,
                            'label': bbox.get('label', ''),
                            'ocr_original': ocr_original,
                            'corrected_text': text,
                            'bbox': {
                                'x': bbox.get('x', 0),
                                'y': bbox.get('y', 0),
                                'width': bbox.get('width', 0),
                                'height': bbox.get('height', 0)
                            },
                            'confidence': bbox.get('ocr_confidence', 0),
                            'document_class': data.get('class', '')
                        }
                        training_data.append(training_item)
                        
                        # 통계 업데이트
                        statistics['total_corrections'] += 1
                        statistics['corrections_by_label'][bbox.get('label', 'unknown')] += 1
                        
                        # OCR 오류 패턴 분석
                        error_type = self._analyze_ocr_error(ocr_original, text)
                        if error_type:
                            statistics['ocr_errors'][error_type] += 1
                            
            except Exception as e:
                self.logger.error(f"Error processing {label_file}: {str(e)}")
                continue
        
        # v2 라벨 파일도 처리 (있는 경우)
        if self.label_v2_dir.exists():
            for label_file in self.label_v2_dir.glob("*_label_v2.json"):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # entities에서 OCR 수정 데이터 수집
                    for entity in data.get('entities', []):
                        ocr_results = entity.get('ocr_results', {})
                        ocr_original = ocr_results.get('tesseract_raw', '')
                        corrected = ocr_results.get('corrected_value', '')
                        current_text = entity.get('text', {}).get('value', '')
                        
                        # 수정된 경우
                        if ocr_original and (corrected != ocr_original or current_text != ocr_original):
                            training_item = {
                                'file': label_file.name,
                                'label': entity.get('label', {}).get('primary', ''),
                                'ocr_original': ocr_original,
                                'corrected_text': current_text or corrected,
                                'bbox': entity.get('bbox', {}),
                                'confidence': ocr_results.get('tesseract_confidence', 0),
                                'document_class': data.get('document_metadata', {}).get('document_type', '')
                            }
                            training_data.append(training_item)
                            statistics['total_corrections'] += 1
                            
                except Exception as e:
                    self.logger.error(f"Error processing v2 file {label_file}: {str(e)}")
                    continue
        
        return training_data, statistics
    
    def _analyze_ocr_error(self, original: str, corrected: str) -> str:
        """OCR 오류 유형 분석"""
        if not original or not corrected:
            return None
            
        # 문자 치환 오류
        if len(original) == len(corrected):
            for i, (o, c) in enumerate(zip(original, corrected)):
                if o != c:
                    # 일반적인 OCR 오류 패턴
                    if (o == 'I' and c == '1') or (o == 'l' and c == '1'):
                        return "I_to_1"
                    elif (o == 'O' and c == '0') or (o == 'o' and c == '0'):
                        return "O_to_0"
                    elif (o == 'S' and c == '5') or (o == 's' and c == '5'):
                        return "S_to_5"
                    elif (o == 'B' and c == '8'):
                        return "B_to_8"
                    else:
                        return f"char_substitution_{o}_to_{c}"
        
        # 길이가 다른 경우
        elif len(original) != len(corrected):
            if len(original) > len(corrected):
                return "extra_chars"
            else:
                return "missing_chars"
        
        return "other"
    
    def train_batch(self, training_data: List[Dict]) -> Dict[str, Any]:
        """일괄 학습 수행"""
        self.logger.info(f"Starting batch training with {len(training_data)} samples")
        
        # defaultdict 사용하여 자동으로 빈 리스트 생성
        results = {
            'total_samples': len(training_data),
            'processed': 0,
            'failed': 0,
            'improvements': defaultdict(list),  # defaultdict로 자동 빈 리스트 생성
            'start_time': datetime.now().isoformat()
        }
        
        # 라벨별로 그룹화
        grouped_data = defaultdict(list)
        for item in training_data:
            grouped_data[item['label']].append(item)
        
        # 각 라벨별로 학습
        for label, items in grouped_data.items():
            self.logger.info(f"Training label '{label}' with {len(items)} samples")
            
            for item in items:
                try:
                    # OCR 학습 서비스에 전달할 형식으로 변환
                    entity = {
                        'entity_id': f"batch_{results['processed']}",
                        'bbox': item['bbox'],
                        'ocr_results': {
                            'tesseract_raw': item['ocr_original'],
                            'tesseract_confidence': item['confidence']
                        }
                    }
                    
                    user_correction = {
                        'label': label,
                        'text': item['corrected_text']
                    }
                    
                    # 학습 수행
                    self.ocr_learning_service.learn_from_correction(entity, user_correction)
                    
                    results['processed'] += 1
                    
                    # 개선 사항 기록
                    if item['ocr_original'] != item['corrected_text']:
                        # defaultdict이므로 자동으로 빈 리스트가 생성됨
                        results['improvements'][label].append({
                            'from': item['ocr_original'],
                            'to': item['corrected_text'],
                            'confidence': item['confidence']
                        })
                        
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error processing item: {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    self.logger.error(f"Item data: {item}")
                    self.logger.error(f"Label: {label}")
                    self.logger.error(f"Current improvements: {results.get('improvements', {})}")
                    results['failed'] += 1
        
        # 학습 완료 후 모델 저장
        self._save_learned_models()
        
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'
        
        return results
    
    def _save_learned_models(self):
        """학습된 모델 저장"""
        try:
            # OCR 학습 서비스의 데이터 저장
            save_path = Path("data/models/ocr_corrections")
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 위치 패턴 저장
            with open(save_path / "position_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(self.ocr_learning_service.position_patterns, f, ensure_ascii=False, indent=2)
            
            # 텍스트 패턴 저장
            with open(save_path / "text_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(self.ocr_learning_service.text_patterns, f, ensure_ascii=False, indent=2)
            
            # 정확도 메트릭 저장
            with open(save_path / "accuracy_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(self.ocr_learning_service.accuracy_metrics, f, ensure_ascii=False, indent=2)
            
            # 보정 히스토리 저장
            with open(save_path / "correction_history.json", 'w', encoding='utf-8') as f:
                json.dump(self.ocr_learning_service.correction_history, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Learned models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def get_learning_report(self, statistics: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """학습 결과 리포트 생성"""
        # defaultdict를 일반 dict로 변환
        improvements_dict = {}
        if 'improvements' in results:
            # defaultdict인 경우 일반 dict로 변환
            if isinstance(results['improvements'], defaultdict):
                improvements_dict = dict(results['improvements'])
            elif isinstance(results['improvements'], dict):
                improvements_dict = results['improvements']
        
        report = {
            'summary': {
                'total_files_processed': statistics['total_files'],
                'total_corrections_found': statistics['total_corrections'],
                'total_samples_trained': results['total_samples'],
                'successfully_processed': results['processed'],
                'failed': results['failed']
            },
            'corrections_by_label': dict(statistics['corrections_by_label']),
            'common_ocr_errors': dict(statistics['ocr_errors']),
            'improvements': improvements_dict,
            'training_time': {
                'start': results.get('start_time'),
                'end': results.get('end_time')
            },
            'current_accuracy': self._calculate_current_accuracy()
        }
        
        return report
    
    def _calculate_current_accuracy(self) -> Dict[str, float]:
        """현재 학습된 모델의 정확도 계산"""
        accuracy = {}
        
        # accuracy_metrics가 딕셔너리가 아닌 경우 처리
        if not hasattr(self.ocr_learning_service, 'accuracy_metrics'):
            return accuracy
            
        metrics = self.ocr_learning_service.accuracy_metrics
        
        # label_accuracy가 있는 새로운 형식 처리
        if isinstance(metrics, dict) and 'label_accuracy' in metrics:
            return metrics.get('label_accuracy', {})
        
        # 레거시 형식 처리 (라벨별 메트릭이 직접 저장된 경우)
        elif isinstance(metrics, dict):
            for label, label_metrics in metrics.items():
                if isinstance(label_metrics, dict) and label_metrics.get('total_corrections', 0) > 0:
                    accuracy[label] = label_metrics.get('current_accuracy', 0.0)
        
        return accuracy
    
    def run_batch_learning(self) -> Dict[str, Any]:
        """전체 일괄 학습 프로세스 실행"""
        self.logger.info("Starting OCR batch learning process")
        
        # 1. 학습 데이터 수집
        training_data, statistics = self.collect_training_data()
        
        if not training_data:
            return {
                'status': 'no_data',
                'message': 'No training data found. Make sure OCR has been performed and corrections have been made.'
            }
        
        # 2. 일괄 학습 수행
        results = self.train_batch(training_data)
        
        # 3. 학습 리포트 생성
        report = self.get_learning_report(statistics, results)
        
        # 4. 리포트 저장
        report_path = Path("data/models/ocr_corrections") / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Batch learning completed. Report saved to {report_path}")
        
        return report