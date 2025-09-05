# 고급 어노테이션 구조 가이드

## 개요
이 문서는 OCR 수준의 자동 라벨링을 위한 상세한 어노테이션 구조를 설명합니다.

## 1. 핵심 개선사항

### 1.1 관계 정보 (Relationships)
```json
"relationships": [
  {
    "target_id": "ent_011",
    "relation_type": "same_row",
    "spatial_relation": "right_of",
    "distance": 147,
    "confidence": 0.98
  }
]
```
- **spatial_relation**: left_of, right_of, above, below, same_row, same_column
- **distance**: 픽셀 단위 거리
- **relation_type**: 의미적 관계 (same_row, item_group, header_group 등)

### 1.2 컨텍스트 정보 (Context)
```json
"context": {
  "nearby_entities": {
    "left": {"entity_id": "ent_x", "text": "Ship to", "distance": 100},
    "right": {"entity_id": "ent_y", "text": "FOB", "distance": 200}
  },
  "table_context": {
    "row_index": 0,
    "column_index": 0,
    "column_header": "Item"
  }
}
```

### 1.3 레이아웃 분석 (Layout Analysis)
```json
"layout_analysis": {
  "regions": [
    {
      "region_id": "header",
      "bbox": [0, 0, 2480, 400],
      "type": "header",
      "contains": ["company_logo", "document_title"]
    }
  ],
  "columns": [
    {"start_x": 122, "end_x": 232, "column_type": "item_number"}
  ]
}
```

### 1.4 특징 벡터 확장 (Extended Features)
```json
"features": {
  "position": {
    "x_normalized": 0.876,
    "region": "header",
    "quadrant": "top_right"
  },
  "text_properties": {
    "pattern": "\\d{10}",
    "matches_regex": ["order_number_pattern"]
  },
  "visual_properties": {
    "font_size_estimated": 14,
    "is_bold": true
  },
  "semantic_properties": {
    "business_meaning": "purchase_order_identifier"
  }
}
```

## 2. 모델 학습 개선 방안

### 2.1 다층 특징 추출
1. **위치 특징**: 절대/상대 좌표, 영역, 사분면
2. **텍스트 특징**: 패턴, 길이, 문자 유형 분포
3. **시각적 특징**: 폰트 크기, 굵기, 정렬
4. **의미적 특징**: 비즈니스 의미, 검증 규칙

### 2.2 관계 기반 학습
```python
def extract_relational_features(entity, all_entities):
    features = {}
    # 같은 행의 다른 엔티티들
    same_row_entities = find_same_row_entities(entity, all_entities)
    features['has_item_number_left'] = any(e.label == 'Item number' for e in same_row_entities)
    
    # 주변 텍스트 패턴
    features['left_text_pattern'] = get_text_pattern(entity.context.nearby_entities.left)
    
    # 구조적 위치
    features['in_table'] = entity.context.structural_context.is_in_table
    return features
```

### 2.3 템플릿 매칭
```python
def match_template(document):
    templates = load_templates()
    best_match = None
    best_score = 0
    
    for template in templates:
        score = calculate_template_similarity(document, template)
        if score > best_score:
            best_match = template
            best_score = score
    
    return best_match, best_score
```

## 3. 구현 전략

### 3.1 단계적 접근
1. **1단계**: 기본 관계 정보 추가 (same_row, 거리)
2. **2단계**: 레이아웃 영역 분석
3. **3단계**: 템플릿 학습 및 매칭
4. **4단계**: 시각적 특징 추출

### 3.2 하이브리드 모델
```python
class HybridOCRLabeler:
    def __init__(self):
        self.position_model = PositionBasedClassifier()
        self.text_model = TextPatternClassifier()
        self.relation_model = RelationalClassifier()
        self.template_matcher = TemplateMatcher()
        self.rule_engine = RuleBasedValidator()
    
    def predict(self, entity, document):
        # 각 모델의 예측
        predictions = {
            'position': self.position_model.predict(entity),
            'text': self.text_model.predict(entity),
            'relation': self.relation_model.predict(entity, document),
            'template': self.template_matcher.predict(entity, document)
        }
        
        # 앙상블
        final_prediction = self.ensemble(predictions)
        
        # 규칙 기반 검증
        return self.rule_engine.validate(final_prediction, entity)
```

## 4. 데이터 수집 전략

### 4.1 점진적 라벨링
1. 핵심 필드만 먼저 라벨링
2. 관계 정보 추가
3. 상세 특징 추가
4. 템플릿 정보 완성

### 4.2 반자동화
```python
def semi_automatic_labeling(document, existing_labels):
    # 기존 라벨 기반 예측
    predictions = model.predict(document)
    
    # 신뢰도 높은 예측만 자동 적용
    auto_labels = [p for p in predictions if p.confidence > 0.9]
    
    # 나머지는 수동 검토
    manual_review = [p for p in predictions if p.confidence <= 0.9]
    
    return auto_labels, manual_review
```

## 5. 검증 및 품질 관리

### 5.1 일관성 검사
- 같은 행의 y 좌표 차이 검증
- 필수 필드 존재 여부 확인
- 비즈니스 규칙 검증 (가격 * 수량 = 합계)

### 5.2 품질 지표
```json
"quality_metrics": {
  "completeness": 0.95,  // 필수 필드 완성도
  "consistency": 0.89,   // 관계 일관성
  "confidence": 0.92     // 전체 신뢰도
}
```

## 6. 마이그레이션 계획

### 6.1 기존 데이터 변환
```python
def migrate_to_advanced_format(old_annotation):
    new_annotation = {
        "annotation_version": "2.0",
        "entities": []
    }
    
    # 기본 정보 변환
    for bbox in old_annotation['bboxes']:
        entity = create_entity_from_bbox(bbox)
        # 관계 정보 추론
        entity['relationships'] = infer_relationships(entity, old_annotation['bboxes'])
        new_annotation['entities'].append(entity)
    
    # 레이아웃 분석 추가
    new_annotation['layout_analysis'] = analyze_layout(new_annotation['entities'])
    
    return new_annotation
```

이 구조를 사용하면 실제 OCR 엔진처럼 문서를 이해하고 라벨을 예측할 수 있습니다.