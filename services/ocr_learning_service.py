import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import re
from collections import defaultdict
from pathlib import Path

class OCRLearningService:
    """Tesseract OCR 결과를 사용자 보정으로 학습하는 서비스"""
    
    def __init__(self):
        self.learning_data_path = Path("data/models/ocr_corrections")
        self.learning_data_path.mkdir(parents=True, exist_ok=True)
        
        # 학습 데이터 로드
        self.correction_history = self._load_correction_history()
        self.position_patterns = self._load_position_patterns()
        self.text_patterns = self._load_text_patterns()
        self.accuracy_metrics = self._load_accuracy_metrics()
        
        # OCR 오류 매핑
        self.common_ocr_errors = {
            '1': ['I', 'l', '|', 'i'],
            '0': ['O', 'o', 'Q'],
            '5': ['S', 's'],
            '8': ['B', '3'],
            '6': ['b'],
            '9': ['g'],
            'rn': ['m'],
            'cl': ['d']
        }
        
    def should_ocr_bbox(self, bbox: Dict, document_type: str, layout_pattern: str) -> Tuple[bool, str]:
        """bbox에 OCR이 필요한지 판단"""
        
        # 1. 위치 기반 라벨 예측
        predicted_label = self._predict_label_by_position(
            bbox, document_type, layout_pattern
        )
        
        # 중요한 라벨들만 OCR 수행
        important_labels = [
            'Order number', 'Shipping line', 'Case mark',
            'Item number', 'Part number', 'Delivery date',
            'Quantity', 'Unit price', 'Line total',
            'Net amount (total)', 'Part description'
        ]
        
        should_ocr = predicted_label in important_labels or predicted_label == "unknown"
        
        return should_ocr, predicted_label
    
    def process_with_learning(self, ocr_result: Dict, bbox: Dict, 
                            document_type: str, layout_pattern: str) -> Dict:
        """OCR 결과를 학습된 패턴으로 보정"""
        
        # 1. 위치 기반 라벨 예측
        predicted_label = self._predict_label_by_position(
            bbox, document_type, layout_pattern
        )
        
        # 2. 텍스트 패턴 기반 보정
        corrected_text = self._correct_ocr_text(
            ocr_result['text'], 
            predicted_label,
            ocr_result.get('confidence', 0)
        )
        
        # 3. 신뢰도 계산
        confidence = self._calculate_confidence(
            ocr_result, bbox, predicted_label, corrected_text
        )
        
        return {
            'original_ocr': ocr_result['text'],
            'corrected_text': corrected_text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'needs_review': confidence < 0.8
        }
    
    def learn_from_correction(self, entity: Dict, user_correction: Dict) -> None:
        """사용자 보정으로부터 학습"""
        
        entity_id = entity['entity_id']
        label = user_correction['label']
        
        # 1. 보정 이력 저장
        self._save_correction_history(entity, user_correction)
        
        # 2. 위치 패턴 업데이트
        self._update_position_pattern(label, entity['bbox'])
        
        # 3. 텍스트 패턴 업데이트
        self._update_text_pattern(
            label, 
            entity.get('ocr_results', {}).get('tesseract_raw', ''),
            user_correction['text']
        )
        
        # 4. 정확도 메트릭 업데이트
        self._update_accuracy_metrics(label)
        
        # 5. 학습 중단 체크
        if self._should_stop_learning(label):
            self._mark_learning_complete(label)
    
    def _predict_label_by_position(self, bbox: Dict, doc_type: str, 
                                  layout: str) -> str:
        """위치 기반 라벨 예측"""
        
        key = f"{doc_type}_{layout}"
        if key not in self.position_patterns:
            return "unknown"
        
        best_match = None
        best_score = 0
        
        for label, pattern in self.position_patterns[key].items():
            score = self._calculate_position_score(bbox, pattern)
            if score > best_score:
                best_score = score
                best_match = label
        
        return best_match if best_score > 0.7 else "unknown"
    
    def _correct_ocr_text(self, ocr_text: str, label: str, 
                         ocr_confidence: float) -> str:
        """OCR 텍스트 보정"""
        
        if label not in self.text_patterns:
            return ocr_text
        
        pattern_data = self.text_patterns[label]
        
        # 1. 일반적인 OCR 오류 수정
        corrected = self._fix_common_ocr_errors(ocr_text)
        
        # 2. 라벨별 특정 패턴 적용
        if 'regex' in pattern_data:
            corrected = self._apply_regex_correction(
                corrected, pattern_data['regex']
            )
        
        # 3. 학습된 보정 패턴 적용
        if 'learned_corrections' in pattern_data:
            for error, correction in pattern_data['learned_corrections'].items():
                if error in corrected:
                    corrected = corrected.replace(error, correction)
        
        return corrected
    
    def _calculate_confidence(self, ocr_result: Dict, bbox: Dict, 
                            label: str, corrected_text: str) -> float:
        """보정 결과의 신뢰도 계산"""
        
        # 가중치
        weights = {
            'ocr_confidence': 0.3,
            'position_match': 0.3,
            'pattern_match': 0.2,
            'correction_history': 0.2
        }
        
        scores = {
            'ocr_confidence': ocr_result.get('confidence', 0),
            'position_match': self._get_position_confidence(bbox, label),
            'pattern_match': self._get_pattern_confidence(corrected_text, label),
            'correction_history': self._get_history_confidence(
                ocr_result['text'], corrected_text, label
            )
        }
        
        # 학습 진행도에 따라 가중치 조정
        if label in self.accuracy_metrics:
            accuracy = self.accuracy_metrics[label]['current_accuracy']
            if accuracy > 0.9:
                weights['correction_history'] = 0.4
                weights['ocr_confidence'] = 0.2
        
        # 최종 신뢰도 계산
        confidence = sum(
            scores[key] * weights[key] for key in weights
        )
        
        return min(confidence, 1.0)
    
    def _should_stop_learning(self, label: str) -> bool:
        """학습 중단 조건 확인"""
        
        if label not in self.accuracy_metrics:
            return False
        
        metrics = self.accuracy_metrics[label]
        
        # 정확도 95% 이상
        if metrics['current_accuracy'] >= 0.95:
            return True
        
        # 최근 50개 샘플에서 개선 없음
        if len(metrics['accuracy_history']) >= 50:
            recent = metrics['accuracy_history'][-50:]
            if max(recent) - min(recent) < 0.02:
                return True
        
        return False
    
    def _save_correction_history(self, entity: Dict, correction: Dict) -> None:
        """보정 이력 저장"""
        
        history_file = self.learning_data_path / "correction_history.json"
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'entity_id': entity['entity_id'],
            'label': correction['label'],
            'original': entity.get('ocr_results', {}).get('tesseract_raw', ''),
            'corrected': correction['text'],
            'bbox': entity['bbox'],
            'document_info': {
                'filename': correction.get('filename'),
                'page': correction.get('page'),
                'layout_pattern': correction.get('layout_pattern')
            }
        }
        
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 이전 형식 호환성 처리
                if isinstance(data, dict):
                    history = data.get('corrections', [])
                elif isinstance(data, list):
                    history = data
                else:
                    history = []
        else:
            history = []
        
        history.append(entry)
        
        # 최근 10000개만 유지
        if len(history) > 10000:
            history = history[-10000:]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_learning_status(self) -> Dict:
        """전체 학습 상태 반환"""
        
        status = {
            'overall_accuracy': 0,
            'labels': {},
            'total_corrections': 0,
            'learning_complete': False
        }
        
        accuracies = []
        
        for label, metrics in self.accuracy_metrics.items():
            status['labels'][label] = {
                'accuracy': metrics['current_accuracy'],
                'corrections': metrics['correction_count'],
                'learning_active': metrics['learning_active']
            }
            
            accuracies.append(metrics['current_accuracy'])
            status['total_corrections'] += metrics['correction_count']
        
        if accuracies:
            status['overall_accuracy'] = np.mean(accuracies)
            status['learning_complete'] = all(
                acc >= 0.95 for acc in accuracies
            )
        
        return status
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """일반적인 OCR 오류 수정"""
        corrected = text
        
        # 숫자 관련 오류 수정
        if any(char.isdigit() for char in text):
            for correct, errors in self.common_ocr_errors.items():
                for error in errors:
                    # 주변 문자가 숫자인 경우에만 교체
                    pattern = f"(?<=\\d){re.escape(error)}|{re.escape(error)}(?=\\d)"
                    corrected = re.sub(pattern, correct, corrected)
        
        return corrected
    
    def _load_correction_history(self) -> List[Dict]:
        """보정 이력 로드"""
        history_file = self.learning_data_path / "correction_history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _load_position_patterns(self) -> Dict:
        """위치 패턴 로드"""
        pattern_file = self.learning_data_path / "position_patterns.json"
        if pattern_file.exists():
            with open(pattern_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_text_patterns(self) -> Dict:
        """텍스트 패턴 로드"""
        pattern_file = self.learning_data_path / "text_patterns.json"
        if pattern_file.exists():
            with open(pattern_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_accuracy_metrics(self) -> Dict:
        """정확도 메트릭 로드"""
        metrics_file = self.learning_data_path / "accuracy_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _update_position_pattern(self, label: str, bbox: Dict) -> None:
        """위치 패턴 업데이트"""
        pattern_file = self.learning_data_path / "position_patterns.json"
        
        if pattern_file.exists():
            with open(pattern_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
        else:
            patterns = {}
        
        # 문서 타입별로 패턴 저장
        doc_key = "purchase_order_pattern_B"  # 기본값
        
        if doc_key not in patterns:
            patterns[doc_key] = {}
            
        if label not in patterns[doc_key]:
            patterns[doc_key][label] = {
                'x_range': [bbox['x'], bbox['x']],
                'y_range': [bbox['y'], bbox['y']],
                'width_range': [bbox.get('width', 100), bbox.get('width', 100)],
                'height_range': [bbox.get('height', 50), bbox.get('height', 50)],
                'count': 0
            }
        
        # 범위 업데이트
        pattern = patterns[doc_key][label]
        pattern['x_range'][0] = min(pattern['x_range'][0], bbox['x'])
        pattern['x_range'][1] = max(pattern['x_range'][1], bbox['x'])
        pattern['y_range'][0] = min(pattern['y_range'][0], bbox['y'])
        pattern['y_range'][1] = max(pattern['y_range'][1], bbox['y'])
        if 'width' in bbox:
            pattern['width_range'][0] = min(pattern['width_range'][0], bbox['width'])
            pattern['width_range'][1] = max(pattern['width_range'][1], bbox['width'])
        if 'height' in bbox:
            pattern['height_range'][0] = min(pattern['height_range'][0], bbox['height'])
            pattern['height_range'][1] = max(pattern['height_range'][1], bbox['height'])
        pattern['count'] += 1
        
        with open(pattern_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
    
    def _update_text_pattern(self, label: str, ocr_text: str, corrected_text: str) -> None:
        """텍스트 패턴 업데이트"""
        pattern_file = self.learning_data_path / "text_patterns.json"
        
        if pattern_file.exists():
            with open(pattern_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
        else:
            patterns = {}
        
        if label not in patterns:
            patterns[label] = {
                'learned_corrections': {},
                'common_patterns': []
            }
        
        # OCR 오류 패턴 학습
        if ocr_text != corrected_text:
            patterns[label]['learned_corrections'][ocr_text] = corrected_text
        
        # 패턴 추출
        if corrected_text:
            if corrected_text.isdigit():
                patterns[label]['common_patterns'].append('numeric')
            elif re.match(r'^[A-Z0-9]+$', corrected_text):
                patterns[label]['common_patterns'].append('alphanumeric_upper')
            elif re.match(r'^\d{2}-\d{2}-\d{4}$', corrected_text):
                patterns[label]['common_patterns'].append('date_mm-dd-yyyy')
        
        with open(pattern_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
    
    def _update_accuracy_metrics(self, label: str) -> None:
        """정확도 메트릭 업데이트"""
        metrics_file = self.learning_data_path / "accuracy_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        if label not in metrics:
            metrics[label] = {
                'correction_count': 0,
                'current_accuracy': 0.0,
                'accuracy_history': [],
                'learning_active': True
            }
        
        metrics[label]['correction_count'] += 1
        
        # 간단한 정확도 계산 (실제로는 더 복잡한 로직 필요)
        # 보정 횟수가 증가할수록 정확도가 향상된다고 가정
        correction_count = metrics[label]['correction_count']
        new_accuracy = min(0.5 + (correction_count * 0.01), 0.99)
        
        metrics[label]['current_accuracy'] = new_accuracy
        metrics[label]['accuracy_history'].append(new_accuracy)
        
        # 최근 100개만 유지
        if len(metrics[label]['accuracy_history']) > 100:
            metrics[label]['accuracy_history'] = metrics[label]['accuracy_history'][-100:]
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    def _calculate_position_score(self, bbox: Dict, pattern: Dict) -> float:
        """위치 기반 점수 계산"""
        score = 0.0
        
        # x 좌표 범위 확인
        if pattern['x_range'][0] <= bbox['x'] <= pattern['x_range'][1]:
            score += 0.25
        
        # y 좌표 범위 확인
        if pattern['y_range'][0] <= bbox['y'] <= pattern['y_range'][1]:
            score += 0.25
        
        # 너비 범위 확인
        if 'width' in bbox and pattern['width_range'][0] <= bbox['width'] <= pattern['width_range'][1]:
            score += 0.25
        
        # 높이 범위 확인
        if 'height' in bbox and pattern['height_range'][0] <= bbox['height'] <= pattern['height_range'][1]:
            score += 0.25
        
        return score
    
    def _get_position_confidence(self, bbox: Dict, label: str) -> float:
        """위치 신뢰도 계산"""
        pattern_file = self.learning_data_path / "position_patterns.json"
        if not pattern_file.exists():
            return 0.5
        
        with open(pattern_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        
        doc_key = "purchase_order_pattern_B"
        if doc_key not in patterns or label not in patterns[doc_key]:
            return 0.5
        
        return self._calculate_position_score(bbox, patterns[doc_key][label])
    
    def _get_pattern_confidence(self, text: str, label: str) -> float:
        """패턴 신뢰도 계산"""
        pattern_file = self.learning_data_path / "text_patterns.json"
        if not pattern_file.exists():
            return 0.5
        
        with open(pattern_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        
        if label not in patterns:
            return 0.5
        
        # 학습된 보정 패턴과 일치하는지 확인
        if text in patterns[label]['learned_corrections'].values():
            return 0.9
        
        # 일반 패턴과 일치하는지 확인
        if 'numeric' in patterns[label]['common_patterns'] and text.isdigit():
            return 0.8
        
        return 0.5
    
    def _get_history_confidence(self, ocr_text: str, corrected_text: str, label: str) -> float:
        """보정 이력 기반 신뢰도"""
        # 보정 이력에서 동일한 패턴이 있는지 확인
        for entry in self.correction_history:
            if (entry.get('label') == label and 
                entry.get('original') == ocr_text and 
                entry.get('corrected') == corrected_text):
                return 0.9
        
        return 0.5
    
    def _mark_learning_complete(self, label: str) -> None:
        """학습 완료 표시"""
        metrics_file = self.learning_data_path / "accuracy_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            if label in metrics:
                metrics[label]['learning_active'] = False
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    def _apply_regex_correction(self, text: str, regex_pattern: str) -> str:
        """정규식 기반 보정"""
        # 간단한 예시 구현
        if regex_pattern == '^451\\d{7}$' and len(text) == 10:
            # Order number 패턴
            corrected = re.sub(r'[OoIl]', '0', text)
            corrected = re.sub(r'[Ss]', '5', text)
            return corrected
        
        return text