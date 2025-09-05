
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
