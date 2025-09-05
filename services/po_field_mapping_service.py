"""
PO Field Auto-Mapping Engine Service

이 서비스는 PO 문서의 필드를 자동으로 매핑하고 템플릿 학습을 수행합니다.
다양한 PO 형식을 학습하여 새로운 문서에서 필드를 자동으로 찾아 매핑합니다.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import pickle

class POFieldMappingService:
    """PO 필드 자동 매핑 및 템플릿 학습 서비스"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # 데이터 디렉토리 설정
        self.data_dir = Path(config.processed_data_directory) / 'po_field_mapping'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 템플릿 저장 경로
        self.templates_file = self.data_dir / 'templates.json'
        self.mapping_rules_file = self.data_dir / 'mapping_rules.json'
        self.learning_data_file = self.data_dir / 'learning_data.json'
        
        # 필드 템플릿 및 규칙
        self.templates = self._load_templates()
        self.mapping_rules = self._load_mapping_rules()
        self.learning_data = self._load_learning_data()
        
        # PO 필드 정의
        self.po_fields = {
            'po_number': {
                'patterns': [r'PO\s*#?\s*:?\s*(\S+)', r'Purchase Order\s*:?\s*(\S+)', r'Order\s*No\.?\s*:?\s*(\S+)'],
                'keywords': ['PO', 'Purchase Order', 'Order No', 'Order Number'],
                'data_type': 'alphanumeric',
                'required': True
            },
            'vendor': {
                'patterns': [r'Vendor\s*:?\s*(.+)', r'Supplier\s*:?\s*(.+)', r'From\s*:?\s*(.+)'],
                'keywords': ['Vendor', 'Supplier', 'From', 'Seller'],
                'data_type': 'text',
                'required': True
            },
            'ship_to': {
                'patterns': [r'Ship\s*To\s*:?\s*(.+)', r'Deliver\s*To\s*:?\s*(.+)', r'Destination\s*:?\s*(.+)'],
                'keywords': ['Ship To', 'Deliver To', 'Destination', 'Delivery Address'],
                'data_type': 'text',
                'required': True
            },
            'order_date': {
                'patterns': [r'Order\s*Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'],
                'keywords': ['Order Date', 'Date', 'Purchase Date'],
                'data_type': 'date',
                'required': True
            },
            'delivery_date': {
                'patterns': [r'Delivery\s*Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'Expected\s*Delivery\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'],
                'keywords': ['Delivery Date', 'Expected Delivery', 'Ship Date'],
                'data_type': 'date',
                'required': False
            },
            'item_number': {
                'patterns': [r'Item\s*#?\s*:?\s*(\S+)', r'Part\s*#?\s*:?\s*(\S+)', r'Product\s*Code\s*:?\s*(\S+)'],
                'keywords': ['Item', 'Part', 'Product Code', 'SKU'],
                'data_type': 'alphanumeric',
                'required': False,
                'is_line_item': True
            },
            'description': {
                'patterns': [r'Description\s*:?\s*(.+)', r'Item\s*Description\s*:?\s*(.+)'],
                'keywords': ['Description', 'Item Description', 'Product'],
                'data_type': 'text',
                'required': False,
                'is_line_item': True
            },
            'quantity': {
                'patterns': [r'Qty\s*:?\s*(\d+)', r'Quantity\s*:?\s*(\d+)', r'QTY\s*:?\s*(\d+)'],
                'keywords': ['Qty', 'Quantity', 'QTY'],
                'data_type': 'number',
                'required': True,
                'is_line_item': True
            },
            'unit_price': {
                'patterns': [r'Unit\s*Price\s*:?\s*\$?(\d+\.?\d*)', r'Price\s*:?\s*\$?(\d+\.?\d*)'],
                'keywords': ['Unit Price', 'Price', 'Rate'],
                'data_type': 'currency',
                'required': False,
                'is_line_item': True
            },
            'total_amount': {
                'patterns': [r'Total\s*:?\s*\$?(\d+\.?\d*)', r'Amount\s*:?\s*\$?(\d+\.?\d*)', r'Net\s*Amount\s*:?\s*\$?(\d+\.?\d*)'],
                'keywords': ['Total', 'Amount', 'Net Amount', 'Grand Total'],
                'data_type': 'currency',
                'required': True
            }
        }
        
        self.logger.info("PO Field Mapping Service initialized")
    
    def _load_templates(self) -> Dict[str, Any]:
        """템플릿 로드"""
        if self.templates_file.exists():
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_templates(self):
        """템플릿 저장"""
        with open(self.templates_file, 'w', encoding='utf-8') as f:
            json.dump(self.templates, f, indent=2, ensure_ascii=False)
    
    def _load_mapping_rules(self) -> Dict[str, Any]:
        """매핑 규칙 로드"""
        if self.mapping_rules_file.exists():
            with open(self.mapping_rules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._initialize_default_rules()
    
    def _save_mapping_rules(self):
        """매핑 규칙 저장"""
        with open(self.mapping_rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping_rules, f, indent=2, ensure_ascii=False)
    
    def _load_learning_data(self) -> List[Dict[str, Any]]:
        """학습 데이터 로드"""
        if self.learning_data_file.exists():
            with open(self.learning_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_learning_data(self):
        """학습 데이터 저장"""
        with open(self.learning_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_default_rules(self) -> Dict[str, Any]:
        """기본 매핑 규칙 초기화"""
        return {
            'field_positions': {},  # 필드별 일반적인 위치
            'field_relationships': {},  # 필드 간 관계
            'vendor_patterns': {},  # 벤더별 패턴
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }
        }
    
    def map_fields(self, ocr_data: List[Dict[str, Any]], template_id: Optional[str] = None) -> Dict[str, Any]:
        """
        OCR 데이터에서 PO 필드 자동 매핑
        
        Args:
            ocr_data: OCR 결과 데이터 (bbox, text 포함)
            template_id: 사용할 템플릿 ID (선택적)
        
        Returns:
            매핑된 필드 정보
        """
        try:
            # 템플릿이 지정된 경우 템플릿 기반 매핑
            if template_id and template_id in self.templates:
                mapped_fields = self._map_with_template(ocr_data, template_id)
            else:
                # 패턴 기반 자동 매핑
                mapped_fields = self._auto_map_fields(ocr_data)
            
            # 유효성 검증
            validation_result = self._validate_mapped_fields(mapped_fields)
            
            return {
                'mapped_fields': mapped_fields,
                'validation': validation_result,
                'confidence': self._calculate_mapping_confidence(mapped_fields),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to map fields: {e}")
            return {
                'error': str(e),
                'mapped_fields': {},
                'validation': {'valid': False, 'errors': [str(e)]}
            }
    
    def _auto_map_fields(self, ocr_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """패턴 기반 자동 필드 매핑"""
        mapped_fields = {}
        
        # 모든 텍스트를 하나의 문자열로 결합 (위치 정보 포함)
        text_blocks = []
        for item in ocr_data:
            if 'text' in item and item['text']:
                text_blocks.append({
                    'text': item['text'],
                    'bbox': item.get('bbox', {}),
                    'confidence': item.get('confidence', 0)
                })
        
        # 각 필드에 대해 매핑 시도
        for field_name, field_config in self.po_fields.items():
            field_value = self._find_field_value(text_blocks, field_config)
            if field_value:
                mapped_fields[field_name] = field_value
        
        # 라인 아이템 그룹화
        line_items = self._group_line_items(text_blocks, mapped_fields)
        if line_items:
            mapped_fields['line_items'] = line_items
        
        return mapped_fields
    
    def _find_field_value(self, text_blocks: List[Dict], field_config: Dict) -> Optional[Dict[str, Any]]:
        """특정 필드 값 찾기"""
        best_match = None
        best_confidence = 0
        
        for block in text_blocks:
            text = block['text']
            
            # 패턴 매칭
            for pattern in field_config.get('patterns', []):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    confidence = block.get('confidence', 0.5)
                    
                    if confidence > best_confidence:
                        best_match = {
                            'value': value,
                            'bbox': block['bbox'],
                            'confidence': confidence,
                            'pattern_matched': pattern
                        }
                        best_confidence = confidence
            
            # 키워드 매칭
            if not best_match:
                for keyword in field_config.get('keywords', []):
                    if keyword.lower() in text.lower():
                        # 키워드 다음의 값 추출
                        value = self._extract_value_after_keyword(text, keyword, field_config['data_type'])
                        if value:
                            confidence = block.get('confidence', 0.5) * 0.8  # 키워드 매칭은 신뢰도 낮춤
                            if confidence > best_confidence:
                                best_match = {
                                    'value': value,
                                    'bbox': block['bbox'],
                                    'confidence': confidence,
                                    'keyword_matched': keyword
                                }
                                best_confidence = confidence
        
        return best_match
    
    def _extract_value_after_keyword(self, text: str, keyword: str, data_type: str) -> Optional[str]:
        """키워드 다음의 값 추출"""
        # 키워드 위치 찾기
        keyword_pos = text.lower().find(keyword.lower())
        if keyword_pos == -1:
            return None
        
        # 키워드 다음 텍스트
        after_text = text[keyword_pos + len(keyword):].strip()
        
        # 콜론이나 탭 제거
        after_text = re.sub(r'^[:|\t]+', '', after_text).strip()
        
        # 데이터 타입에 따른 추출
        if data_type == 'alphanumeric':
            match = re.match(r'(\S+)', after_text)
            return match.group(1) if match else None
        elif data_type == 'number':
            match = re.match(r'(\d+)', after_text)
            return match.group(1) if match else None
        elif data_type == 'currency':
            match = re.match(r'\$?(\d+\.?\d*)', after_text)
            return match.group(1) if match else None
        elif data_type == 'date':
            match = re.match(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', after_text)
            return match.group(1) if match else None
        else:  # text
            # 다음 줄이나 필드까지
            match = re.match(r'(.+?)(?:\n|$)', after_text)
            return match.group(1).strip() if match else after_text.strip()
    
    def _group_line_items(self, text_blocks: List[Dict], mapped_fields: Dict) -> List[Dict[str, Any]]:
        """라인 아이템 그룹화"""
        line_items = []
        
        # 라인 아이템 필드 식별
        line_item_fields = {k: v for k, v in self.po_fields.items() if v.get('is_line_item')}
        
        # Y 좌표로 정렬하여 행 단위로 그룹화
        sorted_blocks = sorted(text_blocks, key=lambda x: (x['bbox'].get('y', 0), x['bbox'].get('x', 0)))
        
        # 행별로 그룹화 (Y 좌표가 비슷한 블록들)
        rows = []
        current_row = []
        last_y = None
        y_threshold = 10  # Y 좌표 차이 임계값
        
        for block in sorted_blocks:
            y = block['bbox'].get('y', 0)
            if last_y is None or abs(y - last_y) <= y_threshold:
                current_row.append(block)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [block]
            last_y = y
        
        if current_row:
            rows.append(current_row)
        
        # 각 행에서 라인 아이템 추출
        for row in rows:
            item = {}
            for field_name, field_config in line_item_fields.items():
                for block in row:
                    # 간단한 패턴 매칭으로 라인 아이템 값 추출
                    if field_config['data_type'] == 'number' and re.match(r'^\d+$', block['text']):
                        if 'quantity' not in item:
                            item['quantity'] = block['text']
                    elif field_config['data_type'] == 'currency' and re.match(r'^\$?\d+\.?\d*$', block['text']):
                        if 'unit_price' not in item:
                            item['unit_price'] = block['text']
                    elif field_config['data_type'] == 'alphanumeric' and re.match(r'^[A-Z0-9-]+$', block['text']):
                        if 'item_number' not in item:
                            item['item_number'] = block['text']
            
            if item and ('quantity' in item or 'item_number' in item):
                line_items.append(item)
        
        return line_items
    
    def _map_with_template(self, ocr_data: List[Dict[str, Any]], template_id: str) -> Dict[str, Any]:
        """템플릿 기반 필드 매핑"""
        template = self.templates[template_id]
        mapped_fields = {}
        
        for field_name, field_template in template.get('fields', {}).items():
            # 템플릿의 위치 정보를 사용하여 필드 찾기
            field_value = self._find_field_by_position(
                ocr_data,
                field_template.get('position'),
                field_template.get('patterns', [])
            )
            if field_value:
                mapped_fields[field_name] = field_value
        
        return mapped_fields
    
    def _find_field_by_position(self, ocr_data: List[Dict], position: Dict, patterns: List[str]) -> Optional[Dict]:
        """위치 정보를 기반으로 필드 찾기"""
        if not position:
            return None
        
        target_x = position.get('x', 0)
        target_y = position.get('y', 0)
        tolerance = position.get('tolerance', 50)
        
        best_match = None
        best_distance = float('inf')
        
        for item in ocr_data:
            bbox = item.get('bbox', {})
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            
            # 유클리드 거리 계산
            distance = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5
            
            if distance <= tolerance and distance < best_distance:
                # 패턴 확인
                text = item.get('text', '')
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        best_match = {
                            'value': text,
                            'bbox': bbox,
                            'confidence': item.get('confidence', 0),
                            'distance': distance
                        }
                        best_distance = distance
                        break
        
        return best_match
    
    def learn_template(self, ocr_data: List[Dict[str, Any]], mapped_fields: Dict[str, Any], 
                       vendor_name: Optional[str] = None) -> str:
        """
        새로운 템플릿 학습
        
        Args:
            ocr_data: OCR 결과 데이터
            mapped_fields: 수동으로 매핑된 필드
            vendor_name: 벤더 이름
        
        Returns:
            생성된 템플릿 ID
        """
        try:
            # 템플릿 ID 생성
            template_id = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if vendor_name:
                template_id = f"{vendor_name}_{template_id}"
            
            # 템플릿 구조 생성
            template = {
                'id': template_id,
                'vendor': vendor_name,
                'created_at': datetime.now().isoformat(),
                'fields': {},
                'layout_signature': self._generate_layout_signature(ocr_data)
            }
            
            # 각 필드의 위치 및 패턴 학습
            for field_name, field_value in mapped_fields.items():
                if isinstance(field_value, dict) and 'bbox' in field_value:
                    template['fields'][field_name] = {
                        'position': field_value['bbox'],
                        'patterns': [field_value.get('value', '')],
                        'confidence': field_value.get('confidence', 1.0)
                    }
            
            # 템플릿 저장
            self.templates[template_id] = template
            self._save_templates()
            
            # 학습 데이터에 추가
            self.learning_data.append({
                'template_id': template_id,
                'timestamp': datetime.now().isoformat(),
                'ocr_data': ocr_data,
                'mapped_fields': mapped_fields
            })
            self._save_learning_data()
            
            # 매핑 규칙 업데이트
            self._update_mapping_rules(mapped_fields, vendor_name)
            
            self.logger.info(f"Learned new template: {template_id}")
            return template_id
            
        except Exception as e:
            self.logger.error(f"Failed to learn template: {e}")
            raise
    
    def _generate_layout_signature(self, ocr_data: List[Dict]) -> str:
        """레이아웃 시그니처 생성 (템플릿 매칭용)"""
        # 주요 텍스트 블록의 상대 위치를 기반으로 시그니처 생성
        signature_parts = []
        
        # 정렬된 블록
        sorted_blocks = sorted(ocr_data, key=lambda x: (x.get('bbox', {}).get('y', 0), 
                                                        x.get('bbox', {}).get('x', 0)))
        
        for i, block in enumerate(sorted_blocks[:10]):  # 상위 10개 블록만 사용
            bbox = block.get('bbox', {})
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            text_len = len(block.get('text', ''))
            
            # 상대 위치와 텍스트 길이로 시그니처 생성
            signature_parts.append(f"{i}:{x//10}:{y//10}:{text_len//5}")
        
        return '|'.join(signature_parts)
    
    def _update_mapping_rules(self, mapped_fields: Dict[str, Any], vendor_name: Optional[str] = None):
        """매핑 규칙 업데이트"""
        # 필드 위치 정보 업데이트
        for field_name, field_value in mapped_fields.items():
            if isinstance(field_value, dict) and 'bbox' in field_value:
                if field_name not in self.mapping_rules['field_positions']:
                    self.mapping_rules['field_positions'][field_name] = []
                
                self.mapping_rules['field_positions'][field_name].append({
                    'bbox': field_value['bbox'],
                    'vendor': vendor_name,
                    'timestamp': datetime.now().isoformat()
                })
        
        # 벤더별 패턴 업데이트
        if vendor_name:
            if vendor_name not in self.mapping_rules['vendor_patterns']:
                self.mapping_rules['vendor_patterns'][vendor_name] = {}
            
            for field_name, field_value in mapped_fields.items():
                if isinstance(field_value, dict):
                    self.mapping_rules['vendor_patterns'][vendor_name][field_name] = {
                        'value': field_value.get('value'),
                        'pattern': field_value.get('pattern_matched') or field_value.get('keyword_matched')
                    }
        
        self._save_mapping_rules()
    
    def _validate_mapped_fields(self, mapped_fields: Dict[str, Any]) -> Dict[str, Any]:
        """매핑된 필드 유효성 검증"""
        errors = []
        warnings = []
        
        # 필수 필드 확인
        for field_name, field_config in self.po_fields.items():
            if field_config.get('required') and field_name not in mapped_fields:
                errors.append(f"Required field '{field_name}' is missing")
        
        # 데이터 타입 검증
        for field_name, field_value in mapped_fields.items():
            if field_name in self.po_fields:
                expected_type = self.po_fields[field_name]['data_type']
                if not self._validate_data_type(field_value, expected_type):
                    warnings.append(f"Field '{field_name}' has invalid data type")
        
        # 비즈니스 규칙 검증
        if 'line_items' in mapped_fields and mapped_fields['line_items']:
            for i, item in enumerate(mapped_fields['line_items']):
                if 'quantity' in item and 'unit_price' in item:
                    try:
                        qty = float(item['quantity'])
                        price = float(item['unit_price'].replace('$', ''))
                        if qty <= 0 or price < 0:
                            warnings.append(f"Line item {i+1} has invalid quantity or price")
                    except ValueError:
                        warnings.append(f"Line item {i+1} has non-numeric quantity or price")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _validate_data_type(self, value: Any, expected_type: str) -> bool:
        """데이터 타입 검증"""
        if isinstance(value, dict):
            value = value.get('value', '')
        
        if expected_type == 'alphanumeric':
            return bool(re.match(r'^[A-Za-z0-9-_]+$', str(value)))
        elif expected_type == 'number':
            try:
                float(value)
                return True
            except:
                return False
        elif expected_type == 'currency':
            return bool(re.match(r'^\$?\d+\.?\d*$', str(value)))
        elif expected_type == 'date':
            return bool(re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', str(value)))
        else:  # text
            return isinstance(value, str) and len(value) > 0
    
    def _calculate_mapping_confidence(self, mapped_fields: Dict[str, Any]) -> float:
        """매핑 신뢰도 계산"""
        if not mapped_fields:
            return 0.0
        
        total_confidence = 0
        field_count = 0
        
        for field_name, field_value in mapped_fields.items():
            if isinstance(field_value, dict) and 'confidence' in field_value:
                total_confidence += field_value['confidence']
                field_count += 1
            elif field_name != 'line_items':  # 라인 아이템 제외
                total_confidence += 0.5  # 기본 신뢰도
                field_count += 1
        
        # 필수 필드 가중치
        required_fields_mapped = sum(1 for f in self.po_fields 
                                    if self.po_fields[f].get('required') and f in mapped_fields)
        required_fields_total = sum(1 for f in self.po_fields if self.po_fields[f].get('required'))
        
        if required_fields_total > 0:
            required_ratio = required_fields_mapped / required_fields_total
        else:
            required_ratio = 1.0
        
        # 최종 신뢰도 계산
        if field_count > 0:
            avg_confidence = total_confidence / field_count
            final_confidence = avg_confidence * 0.7 + required_ratio * 0.3
        else:
            final_confidence = 0.0
        
        return min(1.0, final_confidence)
    
    def get_template_suggestions(self, ocr_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """적합한 템플릿 제안"""
        suggestions = []
        
        # 레이아웃 시그니처 생성
        current_signature = self._generate_layout_signature(ocr_data)
        
        # 모든 템플릿과 비교
        for template_id, template in self.templates.items():
            similarity = self._calculate_signature_similarity(
                current_signature,
                template.get('layout_signature', '')
            )
            
            if similarity > 0.6:  # 60% 이상 유사도
                suggestions.append({
                    'template_id': template_id,
                    'vendor': template.get('vendor'),
                    'similarity': similarity,
                    'created_at': template.get('created_at')
                })
        
        # 유사도 순으로 정렬
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return suggestions[:5]  # 상위 5개만 반환
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """레이아웃 시그니처 유사도 계산"""
        if not sig1 or not sig2:
            return 0.0
        
        parts1 = sig1.split('|')
        parts2 = sig2.split('|')
        
        if len(parts1) != len(parts2):
            return 0.0
        
        matches = 0
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                matches += 1
        
        return matches / len(parts1)
    
    def export_templates(self) -> Dict[str, Any]:
        """템플릿 내보내기"""
        return {
            'templates': self.templates,
            'mapping_rules': self.mapping_rules,
            'export_date': datetime.now().isoformat(),
            'template_count': len(self.templates)
        }
    
    def import_templates(self, template_data: Dict[str, Any]) -> bool:
        """템플릿 가져오기"""
        try:
            if 'templates' in template_data:
                self.templates.update(template_data['templates'])
                self._save_templates()
            
            if 'mapping_rules' in template_data:
                for key, value in template_data['mapping_rules'].items():
                    if key in self.mapping_rules:
                        if isinstance(self.mapping_rules[key], dict):
                            self.mapping_rules[key].update(value)
                        else:
                            self.mapping_rules[key] = value
                self._save_mapping_rules()
            
            self.logger.info(f"Imported {len(template_data.get('templates', {}))} templates")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import templates: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """서비스 통계 조회"""
        return {
            'total_templates': len(self.templates),
            'vendors': list(set(t.get('vendor') for t in self.templates.values() if t.get('vendor'))),
            'learning_samples': len(self.learning_data),
            'field_coverage': {
                field: len([t for t in self.templates.values() 
                          if field in t.get('fields', {})])
                for field in self.po_fields.keys()
            },
            'last_updated': max([t.get('created_at', '') for t in self.templates.values()], default='N/A')
        }
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            self.logger.info("Initializing POFieldMappingService")
            
            # 데이터 디렉토리 생성
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # 초기 데이터 로드
            self.templates = self._load_templates()
            self.mapping_rules = self._load_mapping_rules()
            self.learning_data = self._load_learning_data()
            
            self.logger.info("POFieldMappingService initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize POFieldMappingService: {e}")
            return False
    
    def cleanup(self) -> None:
        """서비스 정리"""
        try:
            self.logger.info("Cleaning up POFieldMappingService")
            
            # 현재 데이터 저장
            self._save_templates()
            self._save_mapping_rules()
            self._save_learning_data()
            
            self.logger.info("POFieldMappingService cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during POFieldMappingService cleanup: {e}")
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 데이터 디렉토리 확인
            if not self.data_dir.exists():
                return False
            
            # 템플릿 파일 접근 가능 여부 확인
            if self.templates_file.exists() and not self.templates_file.is_file():
                return False
            
            # 매핑 규칙 파일 접근 가능 여부 확인
            if self.mapping_rules_file.exists() and not self.mapping_rules_file.is_file():
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Health check failed for POFieldMappingService: {e}")
            return False