#!/usr/bin/env python3
"""
Auto OCR extraction and label suggestion enabler
바운딩 박스 그리기 시 자동 OCR 추출 및 라벨 추천 기능 활성화
"""

import json
from pathlib import Path

def enable_auto_features():
    """자동 OCR 및 라벨 추천 기능 활성화"""
    
    # 1. 설정 파일 업데이트
    config_file = Path("config/settings.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 자동 기능 활성화 설정
    config.update({
        "auto_ocr": {
            "enabled": True,
            "trigger_on_bbox_draw": True,  # 박스 그리기 시 자동 OCR
            "confidence_threshold": 0.8,
            "languages": ["eng", "kor"]
        },
        "auto_labeling": {
            "enabled": True,
            "suggest_labels": True,  # 자동 라벨 추천
            "top_k_suggestions": 3,  # 상위 3개 라벨 추천
            "use_hybrid_model": True,  # 하이브리드 모델 사용
            "confidence_threshold": 0.7
        },
        "label_types": [
            "Order number",
            "Part number", 
            "Delivery date",
            "Quantity",
            "Unit price",
            "Net amount",
            "Net amount (total)",
            "Shipping line",
            "Case mark",
            "Item number"
        ]
    })
    
    # 설정 저장
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ 자동 OCR 및 라벨 추천 기능이 활성화되었습니다.")
    
    # 2. 웹 인터페이스 업데이트 스크립트 생성
    js_update = """
// labeling.js에 추가할 코드
// 바운딩 박스 그리기 완료 시 자동 OCR 및 라벨 추천

function onBboxDrawComplete(bbox) {
    // 자동 OCR 실행
    executeOCR(bbox).then(ocrResult => {
        // OCR 결과를 텍스트 필드에 자동 입력
        updateTextFromOCR(bbox, ocrResult);
        
        // 라벨 추천 받기
        suggestLabels(bbox, ocrResult).then(suggestions => {
            // 추천 라벨 표시
            displayLabelSuggestions(suggestions);
        });
    });
}

async function executeOCR(bbox) {
    const response = await fetch('/api/execute-ocr', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image_path: currentImagePath,
            bbox: bbox
        })
    });
    return response.json();
}

async function suggestLabels(bbox, ocrText) {
    const response = await fetch('/api/suggest-labels', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            bbox: bbox,
            text: ocrText,
            page_number: currentPageNumber
        })
    });
    return response.json();
}
"""
    
    with open("static/js/auto_features.js", 'w', encoding='utf-8') as f:
        f.write(js_update)
    
    # 3. 실행 명령어 안내
    print("\n📝 사용 방법:")
    print("1. 웹 인터페이스 실행: python web_interface.py")
    print("2. 이미지에서 바운딩 박스를 그리면:")
    print("   - 자동으로 OCR이 실행됩니다")
    print("   - 10개 라벨 중 가장 적합한 라벨이 추천됩니다")
    print("   - 추천된 라벨을 클릭하여 적용할 수 있습니다")
    
    return True

if __name__ == "__main__":
    enable_auto_features()
    
    # 학습된 모델 확인
    model_path = Path("data/models/hybrid_model.pkl")
    if model_path.exists():
        print(f"\n✅ 학습된 모델 발견: {model_path}")
    else:
        print("\n⚠️  학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요:")
        print("   python train_hybrid_model.py")