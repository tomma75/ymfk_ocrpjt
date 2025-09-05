/**
 * Labeling System JavaScript
 * Comprehensive labeling functionality extracted from index.html
 * Includes bounding box drawing, group management, PDF navigation, auto-labeling, and model training
 */

// ===========================
// GLOBAL VARIABLES
// ===========================

// Drawing and selection state
let isDrawing = false;
let startX, startY;
let currentBox = null;
let selectedBoxId = null;
let bboxData = [];

// Label data and file management
let currentLabelData = null;
let currentFullFilename = null;
let labelingData = [];

// OCR tracking for learning
let ocrOriginalValues = {};  // Store original OCR values by bbox id
let ocrResults = [];  // Store OCR results separately from bbox data

// PDF navigation
let currentPdfPage = 1;
let totalPdfPages = 1;
let currentPdfFile = null;
let currentPngPrefix = null;  // PNG file prefix (with timestamp)

// Group management
let currentGroupId = null;
let groupMode = false;

// Task management
let currentTaskId = null;
let progressInterval = null;

// File management
let uploadedFilesList = [];
let allFiles = [];
let filteredFiles = [];

// 페이지별 수정된 데이터 임시 저장
let modifiedPageData = {};  // {pageNum: {bboxData: [...], hasUnsavedChanges: true}}

// ===========================
// UTILITY FUNCTIONS (from base.html)
// ===========================

/**
 * API request wrapper with error handling
 */
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        showAlert('요청 처리 중 오류가 발생했습니다.', 'error');
        throw error;
    }
}

/**
 * Show loading spinner
 */
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="spinner"></div>';
    }
}

/**
 * Hide loading spinner
 */
function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
    }
}

/**
 * Show alert notification - Use parent showAlert if available
 */
if (typeof window.showAlert === 'undefined') {
    window.showAlert = function(message, type = 'info') {
        // Fallback if base.html's showAlert is not available
        console.log(`[${type.toUpperCase()}] ${message}`);
    };
}

/**
 * Format file size in human readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ===========================
// DATA LOADING AND SAVING
// ===========================

/**
 * Load labeling data from server
 */
async function loadLabelingData() {
    try {
        showLoading('labelingContent');
        const response = await fetch('/api/labels');
        const data = await response.json();
        
        labelingData = data;
        displayLabelingData();
        showAlert('라벨링 데이터를 불러왔습니다.', 'success');
        
    } catch (error) {
        showAlert(`라벨링 데이터 로드 실패: ${error}`, 'error');
    } finally {
        hideLoading('labelingContent');
    }
}

/**
 * Display general labeling data (legacy format)
 */
function displayLabelingData() {
    const content = document.getElementById('labelingContent');
    content.innerHTML = '';
    
    if (labelingData.length === 0) {
        content.innerHTML = '<p>라벨링 데이터가 없습니다.</p>';
        return;
    }
    
    labelingData.forEach((item, index) => {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'label-form';
        labelDiv.innerHTML = `
            <h3>라벨 ${index + 1}</h3>
            <div class="form-group">
                <label>파일명</label>
                <input type="text" value="${item.filename}" readonly>
            </div>
            <div class="form-group">
                <label>클래스</label>
                <select id="class_${index}">
                    <option value="purchase_order" ${item.class === 'purchase_order' ? 'selected' : ''}>Purchase Order</option>
                    <option value="invoice" ${item.class === 'invoice' ? 'selected' : ''}>Invoice</option>
                    <option value="receipt" ${item.class === 'receipt' ? 'selected' : ''}>Receipt</option>
                    <option value="other" ${item.class === 'other' ? 'selected' : ''}>Other</option>
                </select>
            </div>
            <div class="form-group">
                <label>텍스트</label>
                <textarea id="text_${index}">${item.text || ''}</textarea>
            </div>
            <div class="form-group">
                <label>Bounding Box (x, y, width, height)</label>
                <input type="text" id="bbox_${index}" value="${item.bbox ? item.bbox.join(', ') : ''}">
            </div>
        `;
        content.appendChild(labelDiv);
    });
}

/**
 * Save labeling data (both legacy and current formats)
 */
async function saveLabelingData() {
    // Current editing data has priority
    if (currentLabelData && bboxData.length > 0) {
        await saveCurrentLabelData();
        return;
    }
    
    // Legacy format handling
    if (labelingData && labelingData.length > 0) {
        try {
            const updatedData = labelingData.map((item, index) => {
                const classElem = document.getElementById(`class_${index}`);
                const textElem = document.getElementById(`text_${index}`);
                const bboxElem = document.getElementById(`bbox_${index}`);
                
                if (classElem && textElem && bboxElem) {
                    return {
                        ...item,
                        class: classElem.value,
                        text: textElem.value,
                        bbox: bboxElem.value.split(',').map(v => parseInt(v.trim()))
                    };
                }
                return item;
            });
            
            const response = await fetch('/api/labels', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(updatedData)
            });
            
            if (response.ok) {
                showAlert('라벨링 데이터가 저장되었습니다.', 'success');
            } else {
                showAlert('저장 실패', 'error');
            }
            
        } catch (error) {
            showAlert(`저장 중 오류 발생: ${error}`, 'error');
        }
    } else {
        showAlert('저장할 데이터가 없습니다.', 'warning');
    }
}

/**
 * Save current label data (with bounding boxes and groups)
 */
async function saveCurrentLabelData() {
    if (!currentLabelData && !currentFullFilename) {
        showAlert('저장할 라벨 데이터가 없습니다.', 'error');
        return;
    }
    
    try {
        // 현재 화면에 입력된 모든 데이터를 먼저 동기화
        syncAllBboxData();
        
        const labelData = {
            filename: currentLabelData?.filename || currentFullFilename,
            filepath: currentLabelData?.filepath || currentFullFilename,
            pageNumber: currentPdfPage || 1,
            class: document.getElementById('documentClass').value,
            bboxData: bboxData.map(bbox => ({
                ...bbox,
                ocr_original: bbox.ocr_original || '',
                ocr_confidence: bbox.ocr_confidence || 0,
                was_corrected: bbox.was_corrected || false
            }))
        };
        
        console.log('Saving label data:', labelData);
        console.log('Total bboxData count:', labelData.bboxData.length);
        
        // 디버깅: Shipping line 데이터 확인
        const shippingLineData = labelData.bboxData.find(b => b.label === 'Shipping line');
        if (shippingLineData) {
            console.log('=== SAVING SHIPPING LINE ===');
            console.log('Full bbox data:', shippingLineData);
            console.log('text:', shippingLineData.text);
            console.log('ocr_original:', shippingLineData.ocr_original);
            console.log('was_corrected:', shippingLineData.was_corrected);
        }
        
        // 전송할 JSON 데이터 출력
        console.log('JSON being sent:', JSON.stringify(labelData, null, 2));
        
        const response = await fetch('/api/labels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(labelData)
        });
        
        if (response.ok) {
            showAlert('라벨 데이터가 저장되었습니다.', 'success');
            
            // 자동 모델 업데이트 비활성화 - 수동으로만 학습 실행
            // updateModelIncrementally(labelData);
            
            // 메모리에서 저장 플래그 초기화
            if (modifiedPageData[currentPdfPage]) {
                modifiedPageData[currentPdfPage].hasUnsavedChanges = false;
                console.log(`Page ${currentPdfPage} marked as saved in memory`);
            }
            
            // 저장 후에는 화면 갱신하지 않고 현재 상태 유지
            // 사용자가 입력한 값이 그대로 유지되도록 함
        } else {
            const error = await response.json();
            showAlert(`저장 실패: ${error.error}`, 'error');
        }
    } catch (error) {
        showAlert(`저장 중 오류 발생: ${error}`, 'error');
    }
}

/**
 * Load labeling data for specific file
 */
async function loadLabelingDataForFile(filename) {
    try {
        currentPdfPage = 1;
        // totalPdfPages는 나중에 loadPdfInfo에서 설정됨
        
        console.log(`=== loadLabelingDataForFile START ===`);
        console.log(`Loading labeling data for file: ${filename}`);
        currentFullFilename = filename;
        
        // Extract original PDF name from timestamped filename
        let originalPdfName = filename;
        const timestampMatch = filename.match(/^\d{8}_\d{6}_(.+?)(?:\.pdf)?$/i);
        if (timestampMatch) {
            originalPdfName = timestampMatch[1];
            if (!originalPdfName.toLowerCase().endsWith('.pdf')) {
                originalPdfName += '.pdf';
            }
            console.log(`Extracted original PDF name: ${originalPdfName}`);
        }
        
        // 캐시 무효화를 위해 타임스탬프 추가
        const response = await fetch(`/api/labels/${encodeURIComponent(filename)}?page=${currentPdfPage}&t=${Date.now()}`, {
            cache: 'no-cache',
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        });
        const data = await response.json();
        
        console.log('Label data received:', data);
        console.log('Bboxes:', data.bboxes);
        
        // Show labeling interface
        document.getElementById('labelingContent').style.display = 'none';
        document.getElementById('labelingInterface').style.display = 'block';
        document.getElementById('saveCurrentBtn').style.display = 'inline-flex';
        
        // Set file info
        document.getElementById('labelingFilename').value = data.filename || filename;
        document.getElementById('documentClass').value = data.class || 'purchase_order';
        
        // Handle PDF files
        if (originalPdfName && originalPdfName.toLowerCase().endsWith('.pdf')) {
            currentPdfFile = originalPdfName;
            console.log(`This is a PDF file: ${originalPdfName}`);
            
            // Set PNG prefix (파일명에서 .pdf 제거)
            if (timestampMatch) {
                currentPngPrefix = filename.replace(/\.pdf$/i, '');
                console.log(`PNG prefix set from full filename: ${currentPngPrefix}`);
            } else if (data.image_path) {
                const match = data.image_path.match(/\/(\d{8}_\d{6}_[^/]+)_page_/);
                if (match) {
                    currentPngPrefix = match[1];
                    console.log(`PNG prefix extracted from image_path: ${currentPngPrefix}`);
                }
            }
            
            // PNG prefix가 여전히 .pdf를 포함하고 있다면 제거
            if (currentPngPrefix && currentPngPrefix.toLowerCase().endsWith('.pdf')) {
                currentPngPrefix = currentPngPrefix.replace(/\.pdf$/i, '');
                console.log(`Removed .pdf from PNG prefix: ${currentPngPrefix}`);
            }
            
            // 전체 파일명으로 PDF 정보 로드 시도
            await loadPdfInfo(filename);
            
            // 만약 totalPdfPages가 여전히 1이고 currentPngPrefix가 있다면, 직접 체크
            if (totalPdfPages === 1 && currentPngPrefix) {
                console.log('PDF info failed to get page count, checking PNG files directly...');
                // 간단히 10페이지까지만 체크
                for (let i = 2; i <= 10; i++) {
                    const pageStr = String(i).padStart(3, '0');
                    const pngFilename = `${currentPngPrefix}_page_${pageStr}.png`;
                    const testPath = `/api/view/${encodeURIComponent(pngFilename)}`;
                    try {
                        const testResponse = await fetch(testPath, { method: 'HEAD' });
                        if (testResponse.ok) {
                            totalPdfPages = i;
                            console.log(`Found at least ${i} pages`);
                        } else {
                            break;
                        }
                    } catch (e) {
                        break;
                    }
                }
                console.log(`Direct PNG check found ${totalPdfPages} pages`);
                updatePageInfo();
            }
            
            const pdfNav = document.getElementById('pdfNavigation');
            if (pdfNav) {
                pdfNav.style.display = 'block';
                pdfNav.classList.add('visible');
            }
            
            if (data.current_page) {
                currentPdfPage = data.current_page;
            }
            updatePageInfo();
        } else {
            console.log(`Not a PDF file: ${filename}`);
            const pdfNav = document.getElementById('pdfNavigation');
            if (pdfNav) {
                pdfNav.style.display = 'none';
                pdfNav.classList.remove('visible');
            }
            currentPdfFile = null;
            currentPngPrefix = null;
        }
        
        // Load image
        if (data.image_path) {
            await loadImage(data.image_path);
        }
        
        // Display existing bounding boxes
        if (data.ocr_results && Array.isArray(data.ocr_results)) {
            displayOCRResults(data.ocr_results);
        } else if (data.bboxes && Array.isArray(data.bboxes) && data.bboxes.length > 0) {
            displayExistingBboxes(data.bboxes);
            
            // 대시보드에서 편집 버튼으로 진입한 경우 OCR 실행
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('file')) {
                console.log('Entered from dashboard, executing OCR on all empty bboxes');
                
                // OCR 결과가 없는 bbox들에 대해서만 OCR 실행
                setTimeout(() => {
                    executeOCROnAllEmpty();
                }, 500);
            }
        } else if (data.bbox && Array.isArray(data.bbox) && data.bbox.length > 0) {
            displayExistingBbox(data.bbox);
        }
        
        currentLabelData = data;
        
        console.log(`=== loadLabelingDataForFile END ===`);
        console.log(`Final state:`);
        console.log(`- currentPdfFile: ${currentPdfFile}`);
        console.log(`- currentFullFilename: ${currentFullFilename}`);
        console.log(`- currentPngPrefix: ${currentPngPrefix}`);
        console.log(`- currentPdfPage: ${currentPdfPage}`);
        console.log(`- totalPdfPages: ${totalPdfPages}`);
        
        // window 객체에도 설정 (템플릿과의 호환성을 위해)
        window.currentPdfFile = currentPdfFile;
        window.currentFullFilename = currentFullFilename;
        window.currentPngPrefix = currentPngPrefix;
        window.currentPdfPage = currentPdfPage;
        window.totalPdfPages = totalPdfPages;
        
    } catch (error) {
        showAlert(`파일 라벨 데이터 로드 실패: ${error}`, 'error');
        console.error('loadLabelingDataForFile error:', error);
    }
}

/**
 * Load image and set up container
 */
async function loadImage(imagePath) {
    return new Promise((resolve, reject) => {
        const img = document.getElementById('labelingImage');
        img.style.display = 'block';
        
        // Store the image path for OCR
        const canvas = document.getElementById('imageCanvas');
        if (canvas) {
            canvas.setAttribute('data-image-path', imagePath);
        }
        // Also store in image element as data attribute
        img.setAttribute('data-image-path', imagePath);
        
        img.onload = function() {
            console.log('Image loaded successfully');
            console.log(`Image dimensions: ${img.naturalWidth}x${img.naturalHeight}`);
            
            const container = document.getElementById('imageContainer');
            container.style.width = img.naturalWidth + 'px';
            container.style.height = img.naturalHeight + 'px';
            
            // Also set the bboxOverlays dimensions to match the image
            const overlaysContainer = document.getElementById('bboxOverlays');
            if (overlaysContainer) {
                overlaysContainer.style.width = img.naturalWidth + 'px';
                overlaysContainer.style.height = img.naturalHeight + 'px';
                console.log(`Set bboxOverlays dimensions to: ${img.naturalWidth}x${img.naturalHeight}`);
            }
            
            resolve();
        };
        
        img.onerror = function() {
            console.error('Image load error:', imagePath);
            
            if (currentLabelData && currentLabelData.filename && currentLabelData.filename.toLowerCase().endsWith('.pdf')) {
                console.log('Trying alternative PDF path...');
                const processedImagePath = `/api/view/${encodeURIComponent(currentLabelData.filename.replace('.pdf', `_page_${String(currentPdfPage).padStart(3, '0')}.png`))}`;
                console.log(`Alternative path: ${processedImagePath}`);
                img.src = processedImagePath;
            } else {
                reject(new Error(`이미지를 불러올 수 없습니다: ${imagePath}`));
            }
        };
        
        img.src = imagePath;
        console.log('Setting image source:', imagePath);
    });
}

// ===========================
// BOUNDING BOX DRAWING
// ===========================

/**
 * Start drawing mode
 */
function startDrawing() {
    isDrawing = true;
    document.getElementById('drawBtn').textContent = '🔴 그리기 중...';
    document.getElementById('imageContainer').style.cursor = 'crosshair';
    
    const container = document.getElementById('imageContainer');
    container.onmousedown = handleMouseDown;
    container.onmousemove = handleMouseMove;
    container.onmouseup = handleMouseUp;
}

/**
 * Stop drawing mode
 */
function stopDrawing() {
    isDrawing = false;
    document.getElementById('drawBtn').innerHTML = '<span>✏️</span> Bounding Box 그리기';
    document.getElementById('imageContainer').style.cursor = 'default';
    
    const container = document.getElementById('imageContainer');
    container.onmousedown = null;
    container.onmousemove = null;
    container.onmouseup = null;
}

/**
 * Handle mouse down for drawing
 */
function handleMouseDown(e) {
    if (!isDrawing) return;
    
    const rect = e.target.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    
    currentBox = document.createElement('div');
    currentBox.className = 'bbox-overlay';
    currentBox.style.left = startX + 'px';
    currentBox.style.top = startY + 'px';
    currentBox.style.width = '0px';
    currentBox.style.height = '0px';
    
    document.getElementById('bboxOverlays').appendChild(currentBox);
}

/**
 * Handle mouse move for drawing
 */
function handleMouseMove(e) {
    if (!isDrawing || !currentBox) return;
    
    const rect = e.target.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    const width = Math.abs(currentX - startX);
    const height = Math.abs(currentY - startY);
    
    currentBox.style.width = width + 'px';
    currentBox.style.height = height + 'px';
    currentBox.style.left = Math.min(currentX, startX) + 'px';
    currentBox.style.top = Math.min(currentY, startY) + 'px';
}

/**
 * Handle mouse up for drawing
 */
function handleMouseUp(e) {
    if (!isDrawing || !currentBox) return;
    
    const rect = e.target.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    const bbox = {
        id: Date.now(),
        x: Math.min(startX, endX),
        y: Math.min(startY, endY),
        width: Math.abs(endX - startX),
        height: Math.abs(endY - startY),
        text: '',
        label: '',
        ocr_original: '',
        ocr_confidence: 0,
        was_corrected: false
    };
    
    if (bbox.width > 10 && bbox.height > 10) {
        bboxData.push(bbox);
        
        // OCR 데이터가 없는 경우에만 실행
        if (!bbox.ocr_original) {
            // Execute OCR on the drawn bbox
            executeOCROnBbox(bbox).then(ocrResult => {
                if (ocrResult && ocrResult.ocr_text) {
                    bbox.text = ocrResult.ocr_text;
                    bbox.ocr_original = ocrResult.ocr_text;
                    bbox.ocr_confidence = ocrResult.ocr_confidence;
                    
                    // 자동 라벨 추천 요청
                    suggestLabelsForBbox(bbox);
                    
                    // Update the bbox list to show OCR result
                    updateBoundingBoxList();
                }
            });
        }
        currentBox.id = 'bbox-' + bbox.id;
        
        addResizeHandles(currentBox);
        makeDraggable(currentBox, bbox);
        
        currentBox.onclick = (e) => {
            if (!e.target.classList.contains('resize-handle')) {
                selectBoundingBox(bbox.id);
            }
        };
        
        const label = document.createElement('div');
        label.className = 'bbox-label';
        label.textContent = `Box ${bboxData.length}`;
        currentBox.appendChild(label);
        
        updateBoundingBoxList();
    } else {
        currentBox.remove();
    }
    
    currentBox = null;
    stopDrawing();
}

/**
 * Add a bounding box manually
 */
async function addBoundingBox() {
    const bbox = {
        id: Date.now(),
        x: 50,
        y: 50,
        width: 200,
        height: 50,
        text: '',
        label: '',
        ocr_original: '',
        ocr_confidence: 0,
        was_corrected: false
    };
    
    // OCR 데이터가 없는 경우에만 실행
    if (!bbox.ocr_original) {
        // Execute OCR on the new bbox
        const ocrResult = await executeOCROnBbox(bbox);
        if (ocrResult && ocrResult.ocr_text) {
            bbox.text = ocrResult.ocr_text;
            bbox.ocr_original = ocrResult.ocr_text;
            bbox.ocr_confidence = ocrResult.ocr_confidence;
        }
    }
    
    bboxData.push(bbox);
    createBboxElement(bbox);
    updateBoundingBoxList();
    selectBoundingBox(bbox.id);
}

// ===========================
// BOUNDING BOX MANAGEMENT
// ===========================

/**
 * Select a bounding box
 */
function selectBoundingBox(id) {
    // Clear previous selection
    document.querySelectorAll('.bbox-overlay').forEach(box => {
        box.classList.remove('selected');
    });
    document.querySelectorAll('.bbox-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Set new selection
    selectedBoxId = id;
    document.getElementById('bbox-' + id).classList.add('selected');
    const itemElement = document.getElementById('bbox-item-' + id);
    if (itemElement) {
        itemElement.classList.add('selected');
        
        // 자동 스크롤 - 우측 패널에서 해당 항목으로 스크롤
        itemElement.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
    
    // Auto-add to group in group mode
    if (groupMode && currentGroupId) {
        const bbox = bboxData.find(b => b.id === id);
        if (bbox && !bbox.group_id) {
            bbox.group_id = currentGroupId;
            
            const element = document.getElementById(`bbox-${bbox.id}`);
            if (element) {
                element.style.border = '3px solid #ff9800';
                element.style.backgroundColor = 'rgba(255, 152, 0, 0.2)';
            }
            
            updateBboxList();
            showAlert(`항목이 그룹 '${currentGroupId}'에 자동으로 추가되었습니다.`, 'info');
        }
    }
}

/**
 * Clear bounding box selection
 */
function clearSelection() {
    selectedBoxId = null;
    document.querySelectorAll('.bbox-overlay').forEach(box => {
        box.classList.remove('selected');
    });
    document.querySelectorAll('.bbox-item').forEach(item => {
        item.classList.remove('selected');
    });
}

/**
 * Clear all labels from current page
 */
function clearAllLabels() {
    if (!currentLabelData || !currentLabelData.items) {
        showAlert('현재 페이지에 라벨이 없습니다.', 'info');
        return;
    }
    
    if (!confirm('정말로 현재 페이지의 모든 라벨을 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다.')) {
        return;
    }
    
    // 모든 라벨 데이터 초기화
    currentLabelData.items = [];
    
    // UI에서 모든 bbox 제거
    const overlaysContainer = document.getElementById('bboxOverlays');
    if (overlaysContainer) {
        overlaysContainer.innerHTML = '';
    }
    
    // 리스트 업데이트
    updateBoundingBoxList();
    updateBboxList();
    
    // 선택 해제
    selectedBoxId = null;
    
    // 그룹 정보 초기화
    if (groupMode) {
        document.getElementById('currentGroupInfo').style.display = 'none';
        currentGroupId = null;
        updateGroupSelect();
    }
    
    showAlert('모든 라벨이 초기화되었습니다. 저장하려면 "라벨 저장" 버튼을 클릭하세요.', 'success');
}

/**
 * Delete selected bounding box
 */
function deleteSelectedBox() {
    if (!selectedBoxId) {
        showAlert('삭제할 Bounding Box를 선택해주세요.', 'error');
        return;
    }
    
    document.getElementById('bbox-' + selectedBoxId).remove();
    bboxData = bboxData.filter(b => b.id !== selectedBoxId);
    
    updateBoundingBoxList();
    selectedBoxId = null;
}

/**
 * Execute OCR on a bounding box
 */
/**
 * 바운딩 박스에 대한 라벨 추천 받기
 */
async function suggestLabelsForBbox(bbox) {
    try {
        const response = await fetch('/api/suggest_labels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                bbox: bbox,
                text: bbox.text || bbox.ocr_original || '',
                page_number: currentPdfPage || 1
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.suggestions && result.suggestions.length > 0) {
                // 가장 높은 신뢰도의 라벨을 자동 선택
                const topSuggestion = result.suggestions[0];
                bbox.label = topSuggestion.label;
                
                // UI 업데이트
                updateBoundingBoxList();
                selectBoundingBox(bbox.id);
                
                // 라벨 추천 알림 표시
                showLabelSuggestions(result.suggestions, bbox.id);
            }
        }
    } catch (error) {
        console.error('Label suggestion failed:', error);
    }
}

/**
 * 라벨 추천 결과를 UI에 표시
 */
function showLabelSuggestions(suggestions, bboxId) {
    // 기존 추천 팝업 제거
    const existingPopup = document.querySelector('.label-suggestions-popup');
    if (existingPopup) {
        existingPopup.remove();
    }
    
    // 새 추천 팝업 생성
    const popup = document.createElement('div');
    popup.className = 'label-suggestions-popup';
    popup.innerHTML = `
        <div class="suggestions-header">추천 라벨:</div>
        ${suggestions.map(s => `
            <div class="suggestion-item" onclick="applyLabelSuggestion('${bboxId}', '${s.label}')">
                <span class="label-name">${s.label}</span>
                <span class="confidence">${Math.round(s.confidence * 100)}%</span>
            </div>
        `).join('')}
    `;
    
    // bbox 요소 근처에 팝업 배치
    const bboxElement = document.getElementById(`bbox-${bboxId}`);
    if (bboxElement) {
        const rect = bboxElement.getBoundingClientRect();
        popup.style.position = 'absolute';
        popup.style.left = (rect.right + 10) + 'px';
        popup.style.top = rect.top + 'px';
        popup.style.zIndex = '10000';
        document.body.appendChild(popup);
        
        // 3초 후 자동 제거
        setTimeout(() => {
            if (popup.parentNode) {
                popup.remove();
            }
        }, 3000);
    }
}

/**
 * 추천된 라벨 적용
 */
function applyLabelSuggestion(bboxId, label) {
    const bbox = bboxData.find(b => b.id === parseInt(bboxId));
    if (bbox) {
        bbox.label = label;
        updateBoundingBoxList();
        selectBoundingBox(parseInt(bboxId));
    }
    
    // 추천 팝업 제거
    const popup = document.querySelector('.label-suggestions-popup');
    if (popup) {
        popup.remove();
    }
}

async function executeOCROnBbox(bbox) {
    try {
        // Try multiple ways to get image path
        let imagePath = null;
        
        // Method 1: From image element src (primary method)
        const img = document.getElementById('labelingImage');
        if (img && img.src) {
            // Extract path from src URL
            const url = new URL(img.src);
            const pathname = url.pathname;
            
            // Handle different API endpoints
            if (pathname.startsWith('/api/pdf_to_image/')) {
                // Extract filename from pdf_to_image endpoint
                const filename = pathname.replace('/api/pdf_to_image/', '');
                // Convert PDF filename to PNG with correct page number
                const pageNum = (currentPdfPage || 1).toString().padStart(3, '0');
                const pngFilename = filename.replace('.pdf', `_page_${pageNum}.png`);
                imagePath = 'data/processed/images/' + pngFilename;
            } else if (pathname.startsWith('/api/view/')) {
                // View endpoint - already has PNG filename
                const filename = decodeURIComponent(pathname.replace('/api/view/', ''));
                imagePath = 'data/processed/images/' + filename;
            } else if (pathname.startsWith('/api/image/')) {
                // Regular image endpoint
                imagePath = pathname.replace('/api/image/', '');
                if (!imagePath.includes('data/processed/images/')) {
                    imagePath = 'data/processed/images/' + imagePath;
                }
            } else {
                // Fallback: assume it's a direct path
                imagePath = pathname;
            }
        }
        
        // Method 2: From current label data
        if (!imagePath && currentLabelData && currentLabelData.image_path) {
            imagePath = currentLabelData.image_path;
        }
        
        // Method 3: From image data attribute if available
        if (!imagePath && img) {
            imagePath = img.getAttribute('data-image-path');
        }
        
        if (!imagePath) {
            console.error('No image path available from any source');
            return null;
        }
        
        console.log('Executing OCR with image path:', imagePath);
        console.log('Bbox:', bbox);
        
        const response = await fetch('/api/execute_ocr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: imagePath,
                bbox: {
                    x: bbox.x,
                    y: bbox.y,
                    width: bbox.width,
                    height: bbox.height
                }
            })
        });
        
        const result = await response.json();
        console.log('OCR result:', result);
        
        if (result.error) {
            console.error('OCR error:', result.error);
            return null;
        }
        
        return result;
    } catch (error) {
        console.error('OCR execution failed:', error);
        return null;
    }
}

/**
 * Update bounding box data
 */
function updateBboxData(id, field, value) {
    const bbox = bboxData.find(b => b.id === id);
    if (bbox) {
        // Track original OCR value when text is first set
        if (field === 'text' && !ocrOriginalValues[id] && bbox.ocr_original) {
            ocrOriginalValues[id] = {
                original: bbox.ocr_original,
                confidence: bbox.ocr_confidence,
                modified: false
            };
        }
        
        // Check if user modified the text
        if (field === 'text') {
            // OCR 원본값이 있고 값이 다르면 수정됨으로 표시
            if (bbox.ocr_original && value !== bbox.ocr_original) {
                bbox.was_corrected = true;
            } else if (bbox.ocr_original && value === bbox.ocr_original) {
                bbox.was_corrected = false;
            }
            
            if (ocrOriginalValues[id] && value !== ocrOriginalValues[id].original) {
                ocrOriginalValues[id].modified = true;
            }
        }
        
        bbox[field] = value;
        console.log(`Updated bbox ${id} field ${field} to:`, value);
    }
}

/**
 * Sync all bbox data from input fields
 */
function syncAllBboxData() {
    console.log('Syncing all bbox data from input fields...');
    bboxData.forEach(bbox => {
        // 라벨 필드 동기화
        const labelInput = document.getElementById(`bbox-label-${bbox.id}`);
        if (labelInput && labelInput.value !== bbox.label) {
            bbox.label = labelInput.value;
            console.log(`Synced label for bbox ${bbox.id}:`, bbox.label);
        }
        
        // 텍스트 필드 동기화
        const textInput = document.getElementById(`bbox-text-${bbox.id}`);
        if (textInput && textInput.value !== bbox.text) {
            // OCR 원본값과 다르면 was_corrected 플래그 설정
            if (bbox.ocr_original && textInput.value !== bbox.ocr_original) {
                bbox.was_corrected = true;
            } else if (bbox.ocr_original && textInput.value === bbox.ocr_original) {
                bbox.was_corrected = false;
            }
            
            bbox.text = textInput.value;
            console.log(`Synced text for bbox ${bbox.id}:`, bbox.text);
            console.log(`was_corrected: ${bbox.was_corrected}`);
        }
        
        // 그룹 필드 동기화
        const groupInput = document.getElementById(`bbox-group-${bbox.id}`);
        if (groupInput && groupInput.value !== bbox.group_id) {
            bbox.group_id = groupInput.value;
            console.log(`Synced group for bbox ${bbox.id}:`, bbox.group_id);
        }
    });
}

/**
 * Delete bounding box
 */
function deleteBbox(id) {
    if (confirm('이 Bounding Box를 삭제하시겠습니까?')) {
        // Remove from data array
        bboxData = bboxData.filter(b => b.id !== id);
        
        // Remove from DOM
        const element = document.getElementById('bbox-' + id);
        if (element) {
            element.remove();
        }
        
        // Update list
        updateBoundingBoxList();
        showAlert('Bounding Box가 삭제되었습니다.', 'success');
    }
}

/**
 * Show group modal for setting group ID
 */
function showGroupModal(id) {
    const bbox = bboxData.find(b => b.id === id);
    if (!bbox) return;
    
    const currentGroup = bbox.group_id || '';
    const newGroup = prompt(`그룹 ID를 입력하세요 (예: ITEM_00001)\n현재 그룹: ${currentGroup}`, currentGroup);
    
    if (newGroup !== null) {
        bbox.group_id = newGroup.trim();
        
        // Update visual appearance
        const element = document.getElementById('bbox-' + id);
        if (element) {
            if (newGroup.trim()) {
                element.style.borderColor = '#ff9800';
                element.style.backgroundColor = 'rgba(255, 152, 0, 0.1)';
            } else {
                element.style.borderColor = '#ff0000';
                element.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
            }
        }
        
        // Update list
        updateBoundingBoxList();
        showAlert('그룹이 설정되었습니다.', 'success');
    }
}

/**
 * Add new group
 */
function addNewGroup() {
    const groupName = prompt('새 그룹 이름을 입력하세요 (예: ITEM_00001)');
    if (!groupName || !groupName.trim()) return;
    
    // Get selected bounding boxes
    const selectedBoxes = bboxData.filter(bbox => {
        const element = document.getElementById('bbox-item-' + bbox.id);
        return element && element.classList.contains('selected');
    });
    
    if (selectedBoxes.length === 0) {
        showAlert('그룹에 추가할 Bounding Box를 먼저 선택해주세요.', 'warning');
        return;
    }
    
    // Apply group to selected boxes
    selectedBoxes.forEach(bbox => {
        bbox.group_id = groupName.trim();
        
        // Update visual appearance
        const element = document.getElementById('bbox-' + bbox.id);
        if (element) {
            element.style.borderColor = '#ff9800';
            element.style.backgroundColor = 'rgba(255, 152, 0, 0.1)';
        }
    });
    
    updateBoundingBoxList();
    showAlert(`${selectedBoxes.length}개의 Bounding Box가 그룹 "${groupName}"에 추가되었습니다.`, 'success');
}

/**
 * Auto group by row position
 */
function autoGroupByRow() {
    if (!bboxData.length) {
        showAlert('그룹화할 Bounding Box가 없습니다.', 'warning');
        return;
    }
    
    // Sort by Y position
    const sortedBoxes = [...bboxData].sort((a, b) => a.y - b.y);
    
    // Group by similar Y position (within 20px threshold)
    const groups = [];
    let currentGroup = [sortedBoxes[0]];
    
    for (let i = 1; i < sortedBoxes.length; i++) {
        const prevBox = sortedBoxes[i - 1];
        const currBox = sortedBoxes[i];
        
        if (Math.abs(currBox.y - prevBox.y) <= 20) {
            currentGroup.push(currBox);
        } else {
            groups.push(currentGroup);
            currentGroup = [currBox];
        }
    }
    groups.push(currentGroup);
    
    // Assign group IDs
    groups.forEach((group, groupIndex) => {
        const groupId = `AUTO_GROUP_${groupIndex + 1}`;
        group.forEach(bbox => {
            bbox.group_id = groupId;
            
            // Update visual appearance
            const element = document.getElementById('bbox-' + bbox.id);
            if (element) {
                element.style.borderColor = '#ff9800';
                element.style.backgroundColor = 'rgba(255, 152, 0, 0.1)';
            }
        });
    });
    
    updateBoundingBoxList();
    showAlert(`${groups.length}개의 그룹이 자동으로 생성되었습니다.`, 'success');
}

/**
 * Clear all groups
 */
function clearAllGroups() {
    if (!confirm('모든 그룹을 초기화하시겠습니까?')) return;
    
    bboxData.forEach(bbox => {
        bbox.group_id = '';
        
        // Reset visual appearance
        const element = document.getElementById('bbox-' + bbox.id);
        if (element) {
            element.style.borderColor = '#ff0000';
            element.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
        }
    });
    
    updateBoundingBoxList();
    showAlert('모든 그룹이 초기화되었습니다.', 'success');
}

/**
 * Save and refresh bbox list
 */
async function saveAndRefreshBboxList() {
    try {
        // 현재 화면에 입력된 모든 데이터를 먼저 동기화
        syncAllBboxData();
        
        console.log('=== BEFORE SAVE ===');
        const shippingLineBefore = bboxData.find(b => b.label === 'Shipping line');
        if (shippingLineBefore) {
            console.log('Shipping line before save:', shippingLineBefore.text);
        }
        
        // Save current label data
        await saveCurrentLabelData();
        
        // 저장 성공 메시지 표시
        showAlert('저장되었습니다.', 'success');
        
        // 목록만 다시 그리기 (데이터 리로드 없이 현재 bboxData 사용)
        updateBoundingBoxList();
        updateBboxList();
        
    } catch (error) {
        showAlert('저장 중 오류가 발생했습니다.', 'error');
        console.error('Save and refresh error:', error);
    }
}

/**
 * Update bounding box list display
 */
function updateBoundingBoxList() {
    const listContainer = document.getElementById('bboxList');
    listContainer.innerHTML = '';
    
    console.log('updateBoundingBoxList - bboxData:', bboxData);
    
    bboxData.forEach((bbox, index) => {
        console.log(`Rendering bbox ${index}:`, bbox);
        const item = document.createElement('div');
        item.className = 'bbox-item';
        item.id = 'bbox-item-' + bbox.id;
        item.onclick = (e) => {
            // Prevent selection when clicking on inputs or buttons
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'BUTTON') {
                return;
            }
            
            // Toggle selection with Ctrl/Cmd key
            if (e.ctrlKey || e.metaKey) {
                item.classList.toggle('selected');
            } else {
                // Clear other selections
                document.querySelectorAll('.bbox-item').forEach(el => el.classList.remove('selected'));
                item.classList.add('selected');
                selectBoundingBox(bbox.id);
            }
        };
        
        const groupBadge = bbox.group_id ? 
            `<span class="badge badge-info" style="margin-left: 10px;">${bbox.group_id}</span>` : '';
        
        item.innerHTML = `
            <div class="bbox-item-header">
                <div class="bbox-item-title">
                    <span>Bounding Box ${index + 1}</span>
                    ${groupBadge}
                </div>
                <div class="bbox-item-actions">
                    <button class="group-btn" onclick="showGroupModal(${bbox.id})" title="그룹 설정">
                        <i class="fas fa-layer-group">🔗</i>
                    </button>
                    <button class="delete-btn" onclick="deleteBbox(${bbox.id})" title="삭제">
                        <i class="fas fa-trash">🗑️</i>
                    </button>
                </div>
            </div>
            <div class="form-group">
                <label>위치 (x, y, width, height)</label>
                <input type="text" class="form-control" value="${bbox.x}, ${bbox.y}, ${bbox.width}, ${bbox.height}" readonly>
            </div>
            <div class="form-group">
                <label>라벨</label>
                <input type="text" class="form-control" id="bbox-label-${bbox.id}" value="${bbox.label}" 
                       oninput="updateBboxData(${bbox.id}, 'label', this.value)"
                       onchange="updateBboxData(${bbox.id}, 'label', this.value)"
                       placeholder="예: 구매 번호, 공급업체명 등">
            </div>
            <div class="form-group">
                <label>추출된 텍스트</label>
                <textarea class="form-control" id="bbox-text-${bbox.id}" 
                          oninput="updateBboxData(${bbox.id}, 'text', this.value)"
                          onchange="updateBboxData(${bbox.id}, 'text', this.value)"
                          placeholder="OCR로 추출된 텍스트">${bbox.text}</textarea>
            </div>
            <div class="form-group">
                <label>그룹</label>
                <input type="text" id="bbox-group-${bbox.id}" value="${bbox.group_id || ''}" 
                       oninput="updateBboxData(${bbox.id}, 'group_id', this.value)"
                       onchange="updateBboxData(${bbox.id}, 'group_id', this.value)"
                       placeholder="그룹 ID (예: item_00001)">
            </div>
        `;
        
        listContainer.appendChild(item);
    });
}

/**
 * Create bounding box element on image
 */
function createBboxElement(bbox) {
    console.log(`createBboxElement called for bbox:`, bbox);
    
    const overlaysContainer = document.getElementById('bboxOverlays');
    if (!overlaysContainer) {
        console.error('ERROR: bboxOverlays container not found in createBboxElement!');
        return;
    }
    
    const boxDiv = document.createElement('div');
    boxDiv.className = 'bbox-overlay';
    boxDiv.id = 'bbox-' + bbox.id;
    boxDiv.style.left = bbox.x + 'px';
    boxDiv.style.top = bbox.y + 'px';
    boxDiv.style.width = bbox.width + 'px';
    boxDiv.style.height = bbox.height + 'px';
    
    // 클릭 이벤트 추가 - 클릭 시 선택 및 자동 스크롤
    boxDiv.onclick = function(e) {
        e.stopPropagation();
        selectBoundingBox(bbox.id);
    };
    
    console.log(`Created bbox element with styles:`, {
        id: boxDiv.id,
        left: boxDiv.style.left,
        top: boxDiv.style.top,
        width: boxDiv.style.width,
        height: boxDiv.style.height
    });
    
    // Apply group styling if group_id exists
    if (bbox.group_id && bbox.group_id !== '-') {
        boxDiv.style.borderColor = '#ff9800';
        boxDiv.style.backgroundColor = 'rgba(255, 152, 0, 0.1)';
    }
    
    addResizeHandles(boxDiv);
    makeDraggable(boxDiv, bbox);
    
    boxDiv.onclick = (e) => {
        if (!e.target.classList.contains('resize-handle')) {
            selectBoundingBox(bbox.id);
        }
    };
    
    const label = document.createElement('div');
    label.className = 'bbox-label';
    label.textContent = bbox.confidence ? `${Math.round(bbox.confidence)}%` : `Box ${bboxData.indexOf(bbox) + 1}`;
    boxDiv.appendChild(label);
    
    overlaysContainer.appendChild(boxDiv);
    console.log(`Appended bbox element to container. Total elements now: ${overlaysContainer.children.length}`);
}

/**
 * Add resize handles to bounding box
 */
function addResizeHandles(element) {
    const handles = ['nw', 'ne', 'sw', 'se'];
    handles.forEach(pos => {
        const handle = document.createElement('div');
        handle.className = `resize-handle ${pos}`;
        handle.onmousedown = (e) => startResize(e, pos, element);
        element.appendChild(handle);
    });
}

/**
 * Make bounding box draggable
 */
function makeDraggable(element, bbox) {
    let isDragging = false;
    let currentX;
    let currentY;
    let initialX;
    let initialY;
    
    element.onmousedown = dragStart;
    
    function dragStart(e) {
        if (e.target.classList.contains('resize-handle')) return;
        
        initialX = e.clientX - element.offsetLeft;
        initialY = e.clientY - element.offsetTop;
        
        isDragging = true;
        
        document.onmousemove = drag;
        document.onmouseup = dragEnd;
    }
    
    function drag(e) {
        if (!isDragging) return;
        
        e.preventDefault();
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
        
        element.style.left = currentX + 'px';
        element.style.top = currentY + 'px';
        
        bbox.x = currentX;
        bbox.y = currentY;
    }
    
    function dragEnd() {
        isDragging = false;
        document.onmousemove = null;
        document.onmouseup = null;
        updateBoundingBoxList();
    }
}

/**
 * Start resizing bounding box
 */
function startResize(e, handle, element) {
    e.stopPropagation();
    const bbox = bboxData.find(b => b.id === parseInt(element.id.replace('bbox-', '')));
    if (!bbox) return;
    
    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = parseInt(element.style.width);
    const startHeight = parseInt(element.style.height);
    const startLeft = parseInt(element.style.left);
    const startTop = parseInt(element.style.top);
    
    document.onmousemove = (e) => {
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        
        switch(handle) {
            case 'se':
                element.style.width = Math.max(20, startWidth + dx) + 'px';
                element.style.height = Math.max(20, startHeight + dy) + 'px';
                bbox.width = Math.max(20, startWidth + dx);
                bbox.height = Math.max(20, startHeight + dy);
                break;
            case 'sw':
                element.style.width = Math.max(20, startWidth - dx) + 'px';
                element.style.height = Math.max(20, startHeight + dy) + 'px';
                element.style.left = Math.min(startLeft + startWidth - 20, startLeft + dx) + 'px';
                bbox.width = Math.max(20, startWidth - dx);
                bbox.height = Math.max(20, startHeight + dy);
                bbox.x = parseInt(element.style.left);
                break;
            case 'ne':
                element.style.width = Math.max(20, startWidth + dx) + 'px';
                element.style.height = Math.max(20, startHeight - dy) + 'px';
                element.style.top = Math.min(startTop + startHeight - 20, startTop + dy) + 'px';
                bbox.width = Math.max(20, startWidth + dx);
                bbox.height = Math.max(20, startHeight - dy);
                bbox.y = parseInt(element.style.top);
                break;
            case 'nw':
                element.style.width = Math.max(20, startWidth - dx) + 'px';
                element.style.height = Math.max(20, startHeight - dy) + 'px';
                element.style.left = Math.min(startLeft + startWidth - 20, startLeft + dx) + 'px';
                element.style.top = Math.min(startTop + startHeight - 20, startTop + dy) + 'px';
                bbox.width = Math.max(20, startWidth - dx);
                bbox.height = Math.max(20, startHeight - dy);
                bbox.x = parseInt(element.style.left);
                bbox.y = parseInt(element.style.top);
                break;
        }
    };
    
    document.onmouseup = () => {
        document.onmousemove = null;
        document.onmouseup = null;
        updateBoundingBoxList();
    };
}

// ===========================
// BBOX DATA DISPLAY
// ===========================

/**
 * Display OCR results as bounding boxes
 */
function displayOCRResults(ocrResults) {
    if (!Array.isArray(ocrResults)) return;
    
    bboxData = [];
    document.getElementById('bboxOverlays').innerHTML = '';
    
    ocrResults.forEach((result, index) => {
        if (result.words && Array.isArray(result.words)) {
            result.words.forEach((word, wordIndex) => {
                if (word.bbox) {
                    const bbox = {
                        id: Date.now() + wordIndex,
                        x: word.bbox.x || 0,
                        y: word.bbox.y || 0,
                        width: word.bbox.width || 100,
                        height: word.bbox.height || 30,
                        text: word.text || '',
                        label: '',
                        confidence: word.confidence || 0
                    };
                    
                    bboxData.push(bbox);
                    createBboxElement(bbox);
                }
            });
        }
    });
    
    updateBoundingBoxList();
}

/**
 * Display existing bbox data (legacy format)
 */
function displayExistingBbox(bboxArray) {
    bboxData = [];
    document.getElementById('bboxOverlays').innerHTML = '';
    
    bboxArray.forEach((coords, index) => {
        if (coords && coords.length >= 4) {
            const bbox = {
                id: Date.now() + index,
                x: coords[0],
                y: coords[1],
                width: coords[2],
                height: coords[3],
                text: '',
                label: ''
            };
            
            bboxData.push(bbox);
            createBboxElement(bbox);
        }
    });
    
    updateBoundingBoxList();
}

/**
 * Display existing bboxes (new format with groups)
 */
function displayExistingBboxes(bboxesArray) {
    bboxData = [];
    const overlaysContainer = document.getElementById('bboxOverlays');
    
    console.log('=== displayExistingBboxes START ===');
    console.log('Received bboxesArray:', JSON.stringify(bboxesArray, null, 2));
    console.log('overlaysContainer element:', overlaysContainer);
    
    if (!overlaysContainer) {
        console.error('ERROR: bboxOverlays element not found!');
        return;
    }
    
    overlaysContainer.innerHTML = '';
    
    bboxesArray.forEach((bboxInfo, index) => {
        console.log(`Processing bbox ${index}:`, bboxInfo);
        
        // Shipping line 디버깅
        if (bboxInfo.label === 'Shipping line') {
            console.log('=== SHIPPING LINE BBOX ===');
            console.log('Raw bbox data:', bboxInfo);
            console.log('text:', bboxInfo.text);
            console.log('ocr_original:', bboxInfo.ocr_original);
            console.log('was_corrected:', bboxInfo.was_corrected);
            console.log('=== END SHIPPING LINE ===');
        }
        
        const bbox = {
            id: Date.now() + index,
            x: bboxInfo.x || 0,
            y: bboxInfo.y || 0,
            width: bboxInfo.width || 100,
            height: bboxInfo.height || 30,
            text: bboxInfo.text || '',
            label: bboxInfo.label || '',
            group_id: bboxInfo.group_id || '',
            ocr_original: bboxInfo.ocr_original || '',
            ocr_confidence: bboxInfo.ocr_confidence || 0,
            was_corrected: bboxInfo.was_corrected || false
        };
        console.log(`Created bbox object:`, bbox);
        
        bboxData.push(bbox);
        createBboxElement(bbox);
    });
    
    console.log('Final bboxData array:', bboxData);
    console.log('Number of bbox elements created:', overlaysContainer.children.length);
    
    updateBoundingBoxList();
    updateBboxList();
    
    if (groupMode) {
        updateGroupSelect();
    }
    
    console.log('=== displayExistingBboxes END ===');
    
    // OCR 데이터가 없는 bbox만 확인
    const needsOCR = bboxData.some(bbox => !bbox.ocr_original || bbox.ocr_original === '');
    
    if (needsOCR) {
        // OCR 데이터가 없는 경우에만 자동 실행
        setTimeout(() => {
            executeOCROnAllEmpty();
        }, 1000); // 500ms에서 1000ms로 증가
    }
    
    // Debug: Check if elements are actually visible
    setTimeout(() => {
        const container = document.getElementById('bboxOverlays');
        const imageContainer = document.getElementById('imageContainer');
        const image = document.getElementById('labelingImage');
        
        console.log('=== DEBUG: Visibility Check ===');
        console.log('bboxOverlays:', container);
        console.log('- offsetWidth:', container.offsetWidth);
        console.log('- offsetHeight:', container.offsetHeight);
        console.log('- children count:', container.children.length);
        console.log('- computed style display:', window.getComputedStyle(container).display);
        console.log('- computed style position:', window.getComputedStyle(container).position);
        
        console.log('imageContainer:', imageContainer);
        console.log('- offsetWidth:', imageContainer.offsetWidth);
        console.log('- offsetHeight:', imageContainer.offsetHeight);
        
        console.log('image:', image);
        console.log('- naturalWidth:', image.naturalWidth);
        console.log('- naturalHeight:', image.naturalHeight);
        console.log('- displayed:', image.style.display);
        
        // Check each bbox element
        Array.from(container.children).forEach((child, index) => {
            console.log(`Bbox element ${index}:`, child.id);
            console.log('- offsetWidth:', child.offsetWidth);
            console.log('- offsetHeight:', child.offsetHeight);
            console.log('- style.left:', child.style.left);
            console.log('- style.top:', child.style.top);
            console.log('- computed visibility:', window.getComputedStyle(child).visibility);
        });
        console.log('=== END DEBUG ===');
    }, 1000);
}

// ===========================
// QUICK LABELING
// ===========================

/**
 * Set quick label for selected bounding box
 */
function setQuickLabel(labelText) {
    if (!selectedBoxId) {
        showAlert('먼저 Bounding Box를 선택해주세요.', 'error');
        return;
    }
    
    const bbox = bboxData.find(b => b.id === selectedBoxId);
    if (bbox) {
        bbox.label = labelText;
        const labelInput = document.getElementById(`bbox-label-${selectedBoxId}`);
        if (labelInput) {
            labelInput.value = labelText;
        }
        showAlert(`라벨이 '${labelText}'로 설정되었습니다.`, 'success');
    }
}

// ===========================
// GROUP MANAGEMENT
// ===========================

/**
 * Toggle group mode
 */
function toggleGroupMode() {
    groupMode = !groupMode;
    const btn = document.getElementById('groupModeBtn');
    const panel = document.getElementById('groupManagementPanel');
    
    if (groupMode) {
        btn.innerHTML = '<span>🔗</span> 그룹 모드 ON';
        btn.classList.remove('btn-info');
        btn.classList.add('btn-success');
        panel.style.display = 'block';
        updateGroupSelect();
    } else {
        btn.innerHTML = '<span>🔗</span> 그룹 모드 OFF';
        btn.classList.remove('btn-success');
        btn.classList.add('btn-info');
        panel.style.display = 'none';
    }
}

/**
 * Auto-group bounding boxes by rows (Y position)
 */
function autoGroupByRows() {
    if (bboxData.length === 0) {
        showAlert('그룹화할 라벨이 없습니다.', 'warning');
        return;
    }
    
    const sortedBboxes = [...bboxData].sort((a, b) => a.y - b.y);
    
    let groups = [];
    let currentGroup = [];
    let currentGroupY = null;
    let groupId = 1;
    const yThreshold = 15;
    
    sortedBboxes.forEach(bbox => {
        if (currentGroupY === null || Math.abs(bbox.y - currentGroupY) <= yThreshold) {
            if (currentGroupY === null) currentGroupY = bbox.y;
            currentGroup.push(bbox);
        } else {
            if (currentGroup.length > 0) {
                const gId = `item_${String(groupId).padStart(5, '0')}`;
                currentGroup.forEach(b => b.group_id = gId);
                groups.push({id: gId, items: currentGroup});
                groupId++;
            }
            currentGroup = [bbox];
            currentGroupY = bbox.y;
        }
    });
    
    if (currentGroup.length > 0) {
        const gId = `item_${String(groupId).padStart(5, '0')}`;
        currentGroup.forEach(b => b.group_id = gId);
        groups.push({id: gId, items: currentGroup});
    }
    
    showAlert(`${groups.length}개의 그룹으로 자동 분류되었습니다.`, 'success');
    updateBboxList();
    highlightGroups();
}

/**
 * Update group selection dropdown
 */
function updateGroupSelect() {
    const select = document.getElementById('currentGroupSelect');
    select.innerHTML = '<option value="">그룹 선택...</option>';
    
    const groups = new Set();
    bboxData.forEach(bbox => {
        if (bbox.group_id) {
            groups.add(bbox.group_id);
        }
    });
    
    Array.from(groups).sort().forEach(groupId => {
        const option = document.createElement('option');
        option.value = groupId;
        option.textContent = groupId;
        select.appendChild(option);
    });
}

/**
 * Select a group
 */
function selectGroup(groupId) {
    currentGroupId = groupId;
    const btn = document.getElementById('addToGroupBtn');
    btn.disabled = !groupId;
    
    if (groupId) {
        bboxData.forEach(bbox => {
            const element = document.getElementById(`bbox-${bbox.id}`);
            if (element) {
                if (bbox.group_id === groupId) {
                    element.style.border = '3px solid #ff9800';
                    element.style.backgroundColor = 'rgba(255, 152, 0, 0.2)';
                } else {
                    element.style.border = '2px solid #4caf50';
                    element.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
                }
            }
        });
    }
}

/**
 * Create new group
 */
function createNewGroup() {
    const timestamp = Date.now();
    const groupId = `item_${String(timestamp).slice(-5).padStart(5, '0')}`;
    currentGroupId = groupId;
    
    const select = document.getElementById('currentGroupSelect');
    const option = document.createElement('option');
    option.value = groupId;
    option.textContent = groupId;
    select.appendChild(option);
    select.value = groupId;
    
    document.getElementById('addToGroupBtn').disabled = false;
    showAlert(`새 그룹 '${groupId}'가 생성되었습니다.`, 'success');
}

/**
 * Add selected bbox to current group
 */
function addToCurrentGroup() {
    if (!currentGroupId) {
        showAlert('먼저 그룹을 선택하세요.', 'error');
        return;
    }
    
    if (!selectedBoxId) {
        showAlert('그룹에 추가할 bbox를 선택하세요.', 'error');
        return;
    }
    
    const bbox = bboxData.find(b => b.id === selectedBoxId);
    if (bbox) {
        bbox.group_id = currentGroupId;
        
        const element = document.getElementById(`bbox-${bbox.id}`);
        if (element) {
            element.style.border = '3px solid #ff9800';
            element.style.backgroundColor = 'rgba(255, 152, 0, 0.2)';
        }
        
        updateBboxList();
        showAlert(`항목이 그룹 '${currentGroupId}'에 추가되었습니다.`, 'success');
    }
}

/**
 * Remove selected bbox from group
 */
function removeFromGroup() {
    if (!selectedBoxId) {
        showAlert('그룹에서 제거할 bbox를 선택하세요.', 'error');
        return;
    }
    
    const bbox = bboxData.find(b => b.id === selectedBoxId);
    if (bbox && bbox.group_id) {
        const groupId = bbox.group_id;
        delete bbox.group_id;
        
        const element = document.getElementById(`bbox-${bbox.id}`);
        if (element) {
            element.style.border = '2px solid #4caf50';
            element.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
        }
        
        updateBboxList();
        showAlert(`항목이 그룹 '${groupId}'에서 제거되었습니다.`, 'success');
    }
}

/**
 * Highlight groups with different colors
 */
function highlightGroups() {
    const colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#f44336'];
    const groups = {};
    
    bboxData.forEach(bbox => {
        if (bbox.group_id && bbox.group_id !== '-') {
            if (!groups[bbox.group_id]) {
                groups[bbox.group_id] = [];
            }
            groups[bbox.group_id].push(bbox);
        }
    });
    
    // Reset all bboxes to default style
    bboxData.forEach(bbox => {
        const element = document.getElementById(`bbox-${bbox.id}`);
        if (element) {
            if (!bbox.group_id || bbox.group_id === '-') {
                element.style.border = '2px solid #4caf50';
                element.style.backgroundColor = 'rgba(76, 175, 80, 0.1)';
            }
        }
    });
    
    // Apply group colors
    let colorIndex = 0;
    Object.keys(groups).forEach(groupId => {
        const color = colors[colorIndex % colors.length];
        groups[groupId].forEach(bbox => {
            const element = document.getElementById(`bbox-${bbox.id}`);
            if (element) {
                element.style.border = `2px solid ${color}`;
                element.style.backgroundColor = `${color}20`;
            }
        });
        colorIndex++;
    });
}

/**
 * Update bbox list with grouping
 */
function updateBboxList() {
    const listDiv = document.getElementById('bboxList');
    if (!listDiv) return;
    
    console.log('=== updateBboxList called ===');
    console.log('Current bboxData:', bboxData);
    
    // Shipping line 데이터 디버깅
    const shippingLineData = bboxData.find(b => b.label === 'Shipping line');
    if (shippingLineData) {
        console.log('Shipping line in updateBboxList:', {
            text: shippingLineData.text,
            ocr_original: shippingLineData.ocr_original,
            was_corrected: shippingLineData.was_corrected
        });
        
        // Shipping line 텍스트가 C로 시작하지 않으면 경고
        if (shippingLineData.text && !shippingLineData.text.startsWith('C')) {
            console.warn('WARNING: Shipping line text lost "C" prefix:', shippingLineData.text);
        }
    }
    
    if (bboxData.length === 0) {
        listDiv.innerHTML = '<p>그려진 Bounding Box가 없습니다.</p>';
        return;
    }
    
    const groups = {};
    const ungrouped = [];
    
    bboxData.forEach(bbox => {
        if (bbox.group_id && bbox.group_id !== '-') {
            if (!groups[bbox.group_id]) {
                groups[bbox.group_id] = [];
            }
            groups[bbox.group_id].push(bbox);
        } else {
            ungrouped.push(bbox);
        }
    });
    
    let html = '<div style="max-height: 400px; overflow-y: auto;">';
    
    // Grouped items
    Object.keys(groups).forEach(groupId => {
        html += `<div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; cursor: pointer;" onclick="selectGroup('${groupId}')">`;
        html += `<h5 style="margin: 0 0 10px 0;">📦 ${groupId}</h5>`;
        
        groups[groupId].forEach((bbox, index) => {
            html += `
                <div id="bbox-item-${bbox.id}" class="bbox-item" style="margin-left: 20px; margin-bottom: 5px; padding: 5px; border-radius: 3px; cursor: pointer;" onclick="selectBoundingBox(${bbox.id}); event.stopPropagation();">
                    <span>${index + 1}. ${bbox.label || '(라벨 없음)'}</span>
                    <span style="color: #666;"> - ${bbox.text || ''}</span>
                </div>
            `;
        });
        html += '</div>';
    });
    
    // Ungrouped items
    if (ungrouped.length > 0) {
        html += '<div style="margin-bottom: 15px; padding: 10px; border: 1px solid #ffc107; border-radius: 5px;">';
        html += '<h5 style="margin: 0 0 10px 0;">📝 그룹 미지정</h5>';
        
        ungrouped.forEach((bbox, index) => {
            html += `
                <div id="bbox-item-${bbox.id}" class="bbox-item" style="margin-left: 20px; margin-bottom: 5px; padding: 5px; border-radius: 3px; cursor: pointer;" onclick="selectBoundingBox(${bbox.id})">
                    <span>${index + 1}. ${bbox.label || '(라벨 없음)'}</span>
                    <span style="color: #666;"> - ${bbox.text || ''}</span>
                </div>
            `;
        });
        html += '</div>';
    }
    
    html += '</div>';
    listDiv.innerHTML = html;
    
    if (groupMode) {
        updateGroupSelect();
    }
}

// ===========================
// PDF NAVIGATION
// ===========================

/**
 * Load PDF information
 */
async function loadPdfInfo(filename) {
    try {
        console.log(`=== loadPdfInfo START ===`);
        console.log(`Loading PDF info for: ${filename}`);
        
        const pdfStem = filename.replace('.pdf', '').replace('.PDF', '');
        
        const apiUrl = `/api/pdf_info/${encodeURIComponent(filename)}`;
        console.log(`Calling PDF info API: ${apiUrl}`);
        
        const response = await fetch(apiUrl);
        console.log(`PDF info API response status: ${response.status}`);
        
        if (response.ok) {
            const data = await response.json();
            console.log(`PDF info API response data:`, data);
            totalPdfPages = data.page_count || 1;
            console.log(`PDF API returned ${totalPdfPages} pages for ${filename}`);
        } else {
            console.error(`PDF info API failed with status ${response.status}`);
            const errorText = await response.text();
            console.error(`Error response:`, errorText);
        }
        
        // Check actual pages using PNG prefix
        if (currentPngPrefix) {
            console.log(`Checking actual pages using PNG prefix: ${currentPngPrefix}`);
            let maxPage = 1;
            
            for (let i = 1; i <= 20; i++) {
                const pageStr = String(i).padStart(3, '0');
                const pngFilename = `${currentPngPrefix}_page_${pageStr}.png`;
                const pagePath = `/api/view/${encodeURIComponent(pngFilename)}`;
                try {
                    const imgResponse = await fetch(pagePath, { method: 'HEAD' });
                    if (imgResponse.ok) {
                        maxPage = i;
                        console.log(`Found page ${i} at ${pagePath}`);
                    } else {
                        console.log(`Page ${i} not found at ${pagePath}`);
                        break;
                    }
                } catch (e) {
                    console.error(`Error checking page ${i}:`, e);
                    break;
                }
            }
            
            if (maxPage > totalPdfPages) {
                totalPdfPages = maxPage;
                console.log(`Updated total pages to ${totalPdfPages} based on PNG files`);
            }
        }
        
        updatePageInfo();
        console.log(`=== loadPdfInfo END ===`);
        console.log(`Final totalPdfPages: ${totalPdfPages}`);
    } catch (error) {
        console.error('PDF 정보 로드 중 오류:', error);
        totalPdfPages = 1;
        updatePageInfo();
    }
}

/**
 * Update page info display
 */
function updatePageInfo() {
    document.getElementById('pageInfo').textContent = `${currentPdfPage} / ${totalPdfPages}`;
    
    document.getElementById('prevPageBtn').disabled = currentPdfPage <= 1;
    document.getElementById('nextPageBtn').disabled = currentPdfPage >= totalPdfPages;
}

/**
 * Save current page data to memory before page navigation
 */
function saveCurrentPageToMemory() {
    if (bboxData.length > 0) {
        syncAllBboxData(); // 화면의 모든 입력값 동기화
        
        modifiedPageData[currentPdfPage] = {
            bboxData: JSON.parse(JSON.stringify(bboxData)), // Deep copy
            hasUnsavedChanges: true,
            documentClass: document.getElementById('documentClass')?.value || 'purchase_order'
        };
        
        console.log(`Saved page ${currentPdfPage} to memory:`, modifiedPageData[currentPdfPage]);
    }
}

/**
 * Go to previous page
 */
function previousPage() {
    if (currentPdfPage > 1) {
        // 자동 저장 (수정사항이 있을 때만)
        syncAllBboxData(); // 화면의 모든 입력값 동기화
        if (bboxData.length > 0) {
            console.log('Auto-saving before page navigation...');
            saveCurrentLabelData();
            
            // 저장 완료까지 잠시 대기
            setTimeout(() => {
                currentPdfPage--;
                loadPdfPage(currentPdfPage);
            }, 500);
        } else {
            currentPdfPage--;
            loadPdfPage(currentPdfPage);
        }
    }
}

/**
 * Go to next page
 */
function nextPage() {
    console.log(`nextPage called. Current: ${currentPdfPage}, Total: ${totalPdfPages}`);
    if (currentPdfPage < totalPdfPages) {
        // 자동 저장 (수정사항이 있을 때만)
        syncAllBboxData(); // 화면의 모든 입력값 동기화
        if (bboxData.length > 0) {
            console.log('Auto-saving before page navigation...');
            saveCurrentLabelData();
            
            // 저장 완료까지 잠시 대기
            setTimeout(() => {
                currentPdfPage++;
                console.log(`Moving to page ${currentPdfPage}`);
                loadPdfPage(currentPdfPage);
            }, 500);
        } else {
            currentPdfPage++;
            console.log(`Moving to page ${currentPdfPage}`);
            loadPdfPage(currentPdfPage);
        }
    } else {
        console.log('Already at last page');
    }
}

/**
 * Load specific PDF page
 */
async function loadPdfPage(pageNum) {
    if (!currentPdfFile && !currentFullFilename) {
        console.error('No PDF file to load');
        return;
    }
    
    console.log(`=== loadPdfPage START ===`);
    console.log(`Loading page ${pageNum}...`);
    console.log(`currentPdfFile: ${currentPdfFile}`);
    console.log(`currentFullFilename: ${currentFullFilename}`);
    console.log(`currentPngPrefix: ${currentPngPrefix}`);
    
    // Clear bboxes
    bboxData = [];
    document.getElementById('bboxOverlays').innerHTML = '';
    updateBoundingBoxList();
    
    // Update page info
    currentPdfPage = pageNum;
    updatePageInfo();
    
    // Load new page image and labels
    const img = document.getElementById('labelingImage');
    let imagePath;
    
    if (currentPngPrefix) {
        const pageStr = String(pageNum).padStart(3, '0');
        const pngFilename = `${currentPngPrefix}_page_${pageStr}.png`;
        imagePath = `/api/view/${encodeURIComponent(pngFilename)}`;
        console.log(`Loading PNG directly: ${imagePath}`);
        console.log(`PNG filename: ${pngFilename}`);
    } else if (currentFullFilename) {
        // 전체 파일명이 있는 경우 PDF 변환 사용
        imagePath = `/api/pdf_to_image/${encodeURIComponent(currentFullFilename)}?page=${pageNum}`;
        console.log(`Loading via PDF conversion with full filename: ${imagePath}`);
    } else {
        imagePath = `/api/pdf_to_image/${encodeURIComponent(currentPdfFile)}?page=${pageNum}`;
        console.log(`Loading via PDF conversion: ${imagePath}`);
    }
    
    console.log(`Loading PDF page ${pageNum}: ${imagePath}`);
    
    img.onload = function() {
        console.log(`Page ${pageNum} loaded successfully`);
        const container = document.getElementById('imageContainer');
        container.style.width = img.naturalWidth + 'px';
        container.style.height = img.naturalHeight + 'px';
        
        // Also set the bboxOverlays dimensions to match the image
        const overlaysContainer = document.getElementById('bboxOverlays');
        if (overlaysContainer) {
            overlaysContainer.style.width = img.naturalWidth + 'px';
            overlaysContainer.style.height = img.naturalHeight + 'px';
        }
        
        // Make sure image is visible
        img.style.display = 'block';
    };
    
    img.onerror = function() {
        console.error(`Failed to load page ${pageNum}, trying alternative path`);
        if (!imagePath.includes('pdf_to_image')) {
            // PNG 직접 로드 실패 시 PDF 변환 시도
            const altPath = currentFullFilename ? 
                `/api/pdf_to_image/${encodeURIComponent(currentFullFilename)}?page=${pageNum}` :
                `/api/pdf_to_image/${encodeURIComponent(currentPdfFile)}?page=${pageNum}`;
            console.log(`Retrying with pdf_to_image endpoint: ${altPath}`);
            img.src = altPath;
        } else if (currentPngPrefix && !imagePath.includes('view')) {
            // PDF 변환 실패 시 PNG 직접 로드 시도
            const pageStr = String(pageNum).padStart(3, '0');
            const pngFilename = `${currentPngPrefix}_page_${pageStr}.png`;
            const altPath = `/api/view/${encodeURIComponent(pngFilename)}`;
            console.log(`Retrying with PNG direct load: ${altPath}`);
            console.log(`PNG filename for retry: ${pngFilename}`);
            img.src = altPath;
        } else {
            showAlert('이미지를 불러올 수 없습니다.', 'error');
        }
    };
    
    img.src = imagePath;
    console.log(`Image src set to: ${imagePath}`);
    
    // Load label data for current page
    try {
        // 항상 서버에서 최신 데이터를 로드 (메모리 캐시 사용하지 않음)
        const filenameForLabels = currentFullFilename || currentPdfFile;
        const response = await fetch(`/api/labels/${encodeURIComponent(filenameForLabels)}?page=${pageNum}&t=${Date.now()}`, {
            cache: 'no-cache',
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        });
        console.log('Label API URL:', `/api/labels/${encodeURIComponent(filenameForLabels)}?page=${pageNum}`);
        console.log('Label response status:', response.status);
        if (response.ok) {
            const data = await response.json();
            console.log('Label data received:', data);
            
            if (data.class) {
                document.getElementById('documentClass').value = data.class;
            }
            
            if (data.image_path) {
                const match = data.image_path.match(/\/(\d{8}_\d{6}_[^/]+)_page_/);
                if (match) {
                    currentPngPrefix = match[1];
                    console.log(`PNG prefix updated for page ${pageNum}: ${currentPngPrefix}`);
                }
            }
            
            console.log('Label data for page:', data);
            
            // 디버깅을 위한 상세 로그
            if (data.bboxes) {
                console.log('=== BBOX DATA DEBUG ===');
                data.bboxes.forEach((bbox, idx) => {
                    if (bbox.label === 'Shipping line') {
                        console.log(`Bbox ${idx} - Shipping line:`, {
                            text: bbox.text,
                            ocr_original: bbox.ocr_original,
                            was_corrected: bbox.was_corrected
                        });
                    }
                });
                console.log('=== END BBOX DEBUG ===');
            }
            
            if (data.bboxes && Array.isArray(data.bboxes) && data.bboxes.length > 0) {
                console.log(`Found ${data.bboxes.length} bboxes for page ${pageNum}`);
                displayExistingBboxes(data.bboxes);
                
                // 페이지 이동 후 OCR 자동 실행
                setTimeout(() => {
                    console.log('Auto-executing OCR after page navigation');
                    executeOCROnAllEmpty();
                }, 1000);
            } else if (data.items && Array.isArray(data.items)) {
                // items 형식의 데이터를 bboxes 형식으로 변환
                console.log(`Found ${data.items.length} items for page ${pageNum}`);
                const bboxes = [];
                data.items.forEach(item => {
                    if (item.labels && Array.isArray(item.labels)) {
                        item.labels.forEach(label => {
                            if (label.bbox && Array.isArray(label.bbox) && label.bbox.length >= 4) {
                                bboxes.push({
                                    x: label.bbox[0],
                                    y: label.bbox[1],
                                    width: label.bbox[2],
                                    height: label.bbox[3],
                                    text: label.text || '',
                                    label: label.label || '',
                                    group_id: item.group_id || '',
                                    ocr_original: label.ocr_original || '',
                                    ocr_confidence: label.ocr_confidence || 0,
                                    was_corrected: label.was_corrected || false
                                });
                            }
                        });
                    }
                });
                if (bboxes.length > 0) {
                    displayExistingBboxes(bboxes);
                    
                    // 페이지 이동 후 OCR 자동 실행
                    setTimeout(() => {
                        console.log('Auto-executing OCR after page navigation');
                        executeOCROnAllEmpty();
                    }, 1000);
                }
            }
        }
    } catch (error) {
        console.error('Failed to load label data for page', error);
    }
    
    console.log(`=== loadPdfPage END (page ${pageNum}) ===`);
}

// ===========================
// AUTO-LABELING
// ===========================

/**
 * Get auto-labeling suggestions
 */
async function getAutoLabels() {
    if (!currentLabelData || !currentLabelData.filename) {
        showAlert('먼저 파일을 선택해주세요.', 'error');
        return;
    }
    
    try {
        showAlert('자동 라벨 제안을 가져오는 중...', 'info');
        
        // 자동 라벨 제안 API 호출 (모델 기반)
        const url = `/api/auto_label/${encodeURIComponent(currentLabelData.filename)}?page=${currentPdfPage}`;
        const response = await fetch(url);
        const data = await response.json();
        
        if (!response.ok) {
            showAlert(data.error || '자동 라벨 제안을 가져올 수 없습니다.', 'error');
            return;
        }
        
        if (data.suggestions && data.suggestions.length > 0) {
            displaySuggestedBboxes(data.suggestions);
            showAlert(`${data.suggestions.length}개의 라벨이 자동으로 제안되었습니다.`, 'success');
        } else {
            showAlert('제안할 라벨이 없습니다. 모델 학습이 필요할 수 있습니다.', 'info');
        }
        
    } catch (error) {
        showAlert(`자동 라벨 제안 중 오류: ${error}`, 'error');
    }
}


/**
 * Determine label type based on text content
 */
function determineLabelType(text) {
    if (!text) return 'text';
    
    const textLower = text.toLowerCase();
    
    // 키워드 기반 라벨 타입 결정
    if (textLower.includes('item') || textLower.includes('품목')) {
        return 'item';
    } else if (textLower.includes('quantity') || textLower.includes('qty') || textLower.includes('수량')) {
        return 'quantity';
    } else if (textLower.includes('price') || textLower.includes('단가') || textLower.includes('금액')) {
        return 'price';
    } else if (textLower.includes('total') || textLower.includes('합계')) {
        return 'total';
    } else if (textLower.includes('date') || /\d{4}[\-\/]\d{2}[\-\/]\d{2}/.test(text)) {
        return 'date';
    } else if (/^\d+$/.test(text.trim())) {
        return 'number';
    } else if (textLower.includes('description') || textLower.includes('설명')) {
        return 'description';
    } else if (textLower.includes('unit') || textLower.includes('단위')) {
        return 'unit';
    } else {
        return 'text';
    }
}

/**
 * Get group number from group ID
 */
function getGroupNumber(groupId) {
    if (!groupId || groupId === '-') return 0;
    
    // ITEM_00010, ITEM_00020 등에서 아이템 번호 추출
    if (groupId.startsWith('ITEM_')) {
        const itemNum = parseInt(groupId.split('_')[1]);
        if (!isNaN(itemNum)) {
            // 아이템 번호를 10으로 나눈 몫을 기준으로 그룹 색상 할당
            // 00010 -> 1, 00020 -> 2, ... 00050 -> 5, 00060 -> 1 (순환)
            return ((Math.floor(itemNum / 10) - 1) % 5) + 1;
        }
    }
    
    // 기타 특수 그룹들
    if (groupId === 'ORDER_INFO') return 1;
    if (groupId === 'SHIPPING_INFO') return 2;
    if (groupId.startsWith('AMOUNT_')) return 3;
    if (groupId.startsWith('GROUP_')) {
        const num = parseInt(groupId.split('_')[1]);
        return (num % 5) + 1;
    }
    
    return 0;
}

/**
 * Display suggested bounding boxes
 */
function displaySuggestedBboxes(suggestions) {
    // "text" 라벨 제외
    const filteredSuggestions = suggestions.filter(s => s.label && s.label.toLowerCase() !== 'text');
    
    if (filteredSuggestions.length === 0) {
        showAlert('유효한 라벨 제안이 없습니다.', 'info');
        return;
    }
    
    // 기존 bbox 데이터 초기화 (옵션)
    if (confirm(`${filteredSuggestions.length}개의 라벨을 추가하시겠습니까? 기존 라벨은 유지됩니다.`)) {
        // 기존 라벨은 유지하고 새 제안만 추가
    } else {
        return;
    }
    
    filteredSuggestions.forEach((suggestion, index) => {
        // 페이지 번호 확인 - 페이지가 지정되지 않았거나 현재 페이지와 일치하는 경우만 처리
        if (suggestion.page && suggestion.page !== currentPdfPage) {
            console.log(`[DEBUG] Skipping suggestion ${index}: page=${suggestion.page}, currentPdfPage=${currentPdfPage}`);
            return;
        }
        console.log(`[DEBUG] Processing suggestion ${index}: label=${suggestion.label}, page=${suggestion.page}, currentPdfPage=${currentPdfPage}`);
        
        const bbox = {
            id: Date.now() + index,
            x: suggestion.x,
            y: suggestion.y,
            width: suggestion.width,
            height: suggestion.height,
            text: suggestion.text || '',
            label: suggestion.label || 'Unknown',
            confidence: suggestion.confidence || 0,
            group_id: suggestion.group || suggestion.group_id || '-',
            is_suggestion: true,
            is_group: suggestion.is_group || false
        };
        
        bboxData.push(bbox);
        createSuggestedBboxElement(bbox);
    });
    
    console.log(`[DEBUG] Total bboxData after suggestions: ${bboxData.length}`);
    console.log('[DEBUG] bboxData:', bboxData);
    
    // 자동 그룹핑 적용
    if (suggestions.some(s => s.group_id && s.group_id !== '-')) {
        showAlert('자동 그룹핑이 적용되었습니다.', 'info');
    }
    
    updateBoundingBoxList();
    updateBboxList();
    
    // 자동 저장 옵션 추가
    if (suggestions.length > 0 && confirm('제안된 라벨을 자동으로 저장하시겠습니까?')) {
        saveBoundingBoxData();
    }
}

/**
 * Create suggested bbox element
 */
function createSuggestedBboxElement(bbox) {
    const boxDiv = document.createElement('div');
    boxDiv.className = 'bbox-overlay suggested';
    boxDiv.id = 'bbox-' + bbox.id;
    boxDiv.style.left = bbox.x + 'px';
    boxDiv.style.top = bbox.y + 'px';
    boxDiv.style.width = bbox.width + 'px';
    boxDiv.style.height = bbox.height + 'px';
    
    // 그룹에 따른 색상 적용
    if (bbox.group_id && bbox.group_id !== '-') {
        const groupNum = getGroupNumber(bbox.group_id);
        if (groupNum > 0 && groupNum <= 5) {
            boxDiv.classList.add(`group-${groupNum}`);
        }
    }
    
    addResizeHandles(boxDiv);
    makeDraggable(boxDiv, bbox);
    
    boxDiv.onclick = (e) => {
        if (!e.target.classList.contains('resize-handle')) {
            selectBoundingBox(bbox.id);
        }
    };
    
    const label = document.createElement('div');
    label.className = 'bbox-label';
    // 그룹 정보가 있으면 함께 표시
    if (bbox.group_id && bbox.group_id !== '-') {
        label.textContent = `[${bbox.group_id}] ${bbox.label || 'Unknown'}`;
    } else {
        label.textContent = bbox.label || 'Unknown';
    }
    boxDiv.appendChild(label);
    
    if (bbox.confidence) {
        const confidence = document.createElement('div');
        confidence.className = 'confidence-badge';
        confidence.textContent = `${Math.round(bbox.confidence * 100)}%`;
        boxDiv.appendChild(confidence);
    }
    
    document.getElementById('bboxOverlays').appendChild(boxDiv);
}

// ===========================
// MODEL TRAINING
// ===========================

/**
 * 학습 진행 상황 모달 생성
 */
function createTrainingProgressModal() {
    const modal = document.createElement('div');
    modal.className = 'training-progress-modal';
    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-content">
            <h3>모델 학습 진행 중</h3>
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="trainingProgressFill"></div>
                </div>
                <div class="progress-text">
                    <span id="trainingProgressPercent">0%</span>
                </div>
            </div>
            <div class="progress-message" id="trainingProgressMessage">
                초기화 중...
            </div>
            <div class="progress-details" id="trainingProgressDetails"></div>
        </div>
    `;
    return modal;
}

/**
 * 학습 진행 상황 업데이트
 */
function updateTrainingProgress(modal, message, progress, details = null) {
    const fillElement = modal.querySelector('#trainingProgressFill');
    const percentElement = modal.querySelector('#trainingProgressPercent');
    const messageElement = modal.querySelector('#trainingProgressMessage');
    const detailsElement = modal.querySelector('#trainingProgressDetails');
    
    if (fillElement) {
        fillElement.style.width = `${progress}%`;
    }
    if (percentElement) {
        percentElement.textContent = `${Math.round(progress)}%`;
    }
    if (messageElement) {
        messageElement.textContent = message;
    }
    if (detailsElement && details) {
        detailsElement.innerHTML = details;
    }
}

/**
 * 학습 진행 상황 모달 제거
 */
function removeTrainingProgressModal(modal) {
    if (modal && modal.parentNode) {
        modal.remove();
    }
}

/**
 * Train model with current data
 */
async function trainModel() {
    if (!confirm('현재까지 라벨링된 모든 데이터로 모델을 학습하시겠습니까?\n\nOCR 학습도 함께 진행됩니다.')) {
        return;
    }
    
    // 진행 상황 표시 모달 생성
    const progressModal = createTrainingProgressModal();
    document.body.appendChild(progressModal);
    
    try {
        // 학습 시작
        updateTrainingProgress(progressModal, '모델 학습을 시작합니다...', 0);
        
        const response = await fetch('/api/train_model', {
            method: 'POST'
        });
        
        // 스트리밍 응답 처리
        let result = null;
        if (response.headers.get('content-type')?.includes('text/event-stream')) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // 마지막 불완전한 라인은 버퍼에 유지
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.progress !== undefined) {
                                updateTrainingProgress(progressModal, data.message, data.progress);
                            } else {
                                // progress가 없는 데이터는 최종 결과
                                result = data;
                            }
                        } catch (e) {
                            console.error('Progress parsing error:', e, line);
                        }
                    }
                }
            }
        } else {
            result = await response.json();
        }
        
        if (response.ok && result && result.status === 'success') {
            let message = `모델 학습 완료! 총 ${result.total_samples}개 샘플, ${result.unique_labels}개 라벨 클래스`;
            
            // OCR 학습 결과 추가
            if (result.ocr_learning) {
                if (result.ocr_learning.ocr_training === 'completed') {
                    const summary = result.ocr_learning.summary;
                    message += `\n\nOCR 학습 완료:`;
                    message += `\n- 처리된 파일: ${summary.total_files_processed}개`;
                    message += `\n- 발견된 보정: ${summary.total_corrections_found}개`;
                    message += `\n- 학습된 샘플: ${summary.successfully_processed}개`;
                    
                    // 라벨별 보정 현황
                    if (result.ocr_learning.corrections_by_label) {
                        message += `\n\n라벨별 보정 현황:`;
                        for (const [label, count] of Object.entries(result.ocr_learning.corrections_by_label)) {
                            message += `\n- ${label}: ${count}개`;
                        }
                    }
                    
                    // 현재 정확도
                    if (result.ocr_learning.current_accuracy) {
                        message += `\n\n현재 정확도:`;
                        for (const [label, accuracy] of Object.entries(result.ocr_learning.current_accuracy)) {
                            message += `\n- ${label}: ${(accuracy * 100).toFixed(1)}%`;
                        }
                    }
                } else if (result.ocr_learning.ocr_training === 'no_data') {
                    message += `\n\nOCR 학습: 학습할 데이터가 없습니다.`;
                    message += `\nOCR을 수행하고 보정한 후 다시 시도해주세요.`;
                }
            }
            
            // 진행 상황 100% 표시
            updateTrainingProgress(progressModal, '학습 완료!', 100);
            
            // 모달 제거
            setTimeout(() => {
                removeTrainingProgressModal(progressModal);
            }, 1500);
            
            showAlert(message, 'success', 10000); // 10초간 표시
        } else {
            removeTrainingProgressModal(progressModal);
            showAlert(result.message || '모델 학습 실패', 'error');
        }
        
    } catch (error) {
        removeTrainingProgressModal(progressModal);
        showAlert(`모델 학습 중 오류: ${error}`, 'error');
    }
}

/**
 * Reset model
 */
async function resetModel() {
    if (!confirm('모델을 초기화하시겠습니까?\n\n⚠️ 주의:\n- 모든 학습된 모델이 삭제됩니다\n- OCR 학습 데이터가 초기화됩니다\n- 기존 데이터는 백업 폴더에 저장됩니다\n\n계속하시겠습니까?')) {
        return;
    }
    
    // 추가 확인
    if (!confirm('정말로 모델을 초기화하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
        return;
    }
    
    try {
        showAlert('모델 초기화를 시작합니다...', 'info');
        
        const response = await fetch('/api/reset_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ confirm: true })
        });
        
        const result = await response.json();
        
        if (response.ok && result.status === 'success') {
            let message = '✅ 모델 초기화 완료!\n\n';
            message += `백업 위치: ${result.backup_location}\n`;
            message += `백업된 파일: ${result.backed_up_files.length}개\n`;
            
            if (result.removed_reports > 0) {
                message += `정리된 리포트: ${result.removed_reports}개\n`;
            }
            
            message += '\n이제 새로운 데이터로 모델을 학습할 수 있습니다.';
            
            showAlert(message, 'success', 8000);
        } else {
            showAlert(result.error || '모델 초기화 실패', 'error');
        }
        
    } catch (error) {
        showAlert(`모델 초기화 중 오류: ${error}`, 'error');
    }
}

/**
 * Show model statistics
 */
async function showModelStats() {
    try {
        const response = await fetch('/api/model_stats');
        const stats = await response.json();
        
        if (response.ok) {
            let message = '모델 통계:\n';
            message += `- 학습 상태: ${stats.is_trained ? '학습됨' : '미학습'}\n`;
            
            if (stats.training_stats) {
                message += `- 총 샘플 수: ${stats.training_stats.total_samples}\n`;
                message += `- 마지막 학습: ${stats.training_stats.last_training_time || 'N/A'}\n`;
                message += `- 라벨 분포:\n`;
                
                for (const [label, count] of Object.entries(stats.training_stats.label_distribution || {})) {
                    message += `  • ${label}: ${count}개\n`;
                }
            }
            
            showAlert(message, 'info');
        } else {
            showAlert('모델 통계를 가져올 수 없습니다.', 'error');
        }
        
    } catch (error) {
        showAlert(`통계 조회 중 오류: ${error}`, 'error');
    }
}

/**
 * Update model incrementally (background)
 */
async function updateModelIncrementally(newAnnotation) {
    try {
        const response = await fetch('/api/train_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                incremental: true,
                annotation: newAnnotation
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.status === 'success') {
                console.log('모델이 자동으로 업데이트되었습니다.');
            }
        }
    } catch (error) {
        console.error('모델 업데이트 중 오류:', error);
    }
}

// ===========================
// FILE MANAGEMENT
// ===========================

/**
 * Delete collected file
 */
async function deleteCollectedFile(filePath, fileName) {
    if (!confirm(`정말로 '${fileName}' 파일을 삭제하시겠습니까?\n관련된 모든 파일(이미지, 라벨 등)도 함께 삭제됩니다.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/delete/collected/${encodeURIComponent(filePath)}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const result = await response.json();
            showAlert(`${fileName} 파일이 삭제되었습니다.`, 'success');
            
            // Refresh file list if function exists
            if (typeof refreshFiles === 'function') {
                refreshFiles();
            }
        } else {
            const error = await response.json();
            showAlert(`파일 삭제 실패: ${error.error}`, 'error');
        }
    } catch (error) {
        showAlert(`파일 삭제 중 오류: ${error}`, 'error');
    }
}

// ===========================
// INITIALIZATION
// ===========================

/**
 * Initialize labeling system
 */
function initializeLabelingSystem() {
    console.log('Labeling system initialized');
    
    // Set up event listeners for file input if it exists
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const files = Array.from(e.target.files);
            uploadFiles(files);
        });
    }
    
    // Set up drag and drop if containers exist
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('drop', handleDrop);
    }
}

/**
 * Handle drag over
 */
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

/**
 * Handle drag leave
 */
function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

/**
 * Handle drop
 */
function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    uploadFiles(files);
}

/**
 * Upload files
 */
async function uploadFiles(files) {
    const validFiles = files.filter(file => {
        const ext = file.name.toLowerCase().split('.').pop();
        return ['pdf', 'png', 'jpg', 'jpeg'].includes(ext);
    });
    
    if (validFiles.length === 0) {
        showAlert('유효한 파일이 없습니다. PDF, PNG, JPG 파일만 업로드 가능합니다.', 'error');
        return;
    }
    
    for (const file of validFiles) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                showAlert(`${file.name} 업로드 성공`, 'success');
                uploadedFilesList.push(result);
                updateUploadedFilesList();
            } else {
                showAlert(`${file.name} 업로드 실패`, 'error');
            }
        } catch (error) {
            showAlert(`${file.name} 업로드 중 오류 발생`, 'error');
        }
    }
}

/**
 * Update uploaded files list
 */
function updateUploadedFilesList() {
    const container = document.getElementById('uploadedFiles');
    if (!container) return;
    
    container.innerHTML = '<h3>업로드된 파일</h3>';
    
    uploadedFilesList.forEach(file => {
        const fileDiv = document.createElement('div');
        fileDiv.className = 'file-item';
        fileDiv.innerHTML = `
            <div class="file-info">
                <strong>${file.name}</strong><br>
                <small>크기: ${formatFileSize(file.size)}</small>
            </div>
            <div class="file-actions">
                <button class="btn btn-small btn-primary" onclick="processFile('${file.id}')">처리</button>
                <button class="btn btn-small btn-warning" onclick="removeFile('${file.id}')">삭제</button>
            </div>
        `;
        container.appendChild(fileDiv);
    });
}

// Test function for verifying save functionality
async function testSaveFunction() {
    console.log('=== Testing Save Function ===');
    
    // Create test bbox data
    const testBbox = {
        id: Date.now(),
        x: 100,
        y: 100,
        width: 200,
        height: 50,
        text: 'Test Text',
        label: 'Test Label',
        group_id: 'TEST_GROUP'
    };
    
    // Add to bboxData if empty
    if (bboxData.length === 0) {
        bboxData.push(testBbox);
        console.log('Added test bbox to bboxData:', testBbox);
    }
    
    // Log current state
    console.log('Current bboxData:', bboxData);
    console.log('Current labelData:', currentLabelData);
    console.log('Current PDF page:', currentPdfPage);
    console.log('Document class:', document.getElementById('documentClass')?.value);
    
    // Try to save
    try {
        await saveCurrentLabelData();
        console.log('Save function completed');
    } catch (error) {
        console.error('Save function error:', error);
    }
    
    console.log('=== Test Complete ===');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeLabelingSystem();
    
    // Add test button for debugging
    setTimeout(() => {
        const testBtn = document.createElement('button');
        testBtn.textContent = 'Test Save';
        testBtn.style.position = 'fixed';
        testBtn.style.bottom = '10px';
        testBtn.style.right = '10px';
        testBtn.style.zIndex = '9999';
        testBtn.style.backgroundColor = '#4CAF50';
        testBtn.style.color = 'white';
        testBtn.style.padding = '10px 20px';
        testBtn.style.border = 'none';
        testBtn.style.borderRadius = '5px';
        testBtn.style.cursor = 'pointer';
        testBtn.onclick = testSaveFunction;
        document.body.appendChild(testBtn);
    }, 1000);
});

/**
 * Execute OCR only on empty bounding boxes (without OCR data)
 */
async function executeOCROnAllEmpty() {
    const emptyBboxes = bboxData.filter(bbox => !bbox.ocr_original || bbox.ocr_original === '');
    
    if (emptyBboxes.length === 0) {
        console.log('All bboxes already have OCR data');
        // OCR 데이터가 이미 있는 경우에도 OCR 결과 목록을 업데이트
        
        // 기존 OCR 데이터를 OCR 결과 목록에 추가
        ocrResults = [];
        bboxData.forEach(bbox => {
            if (bbox.ocr_original) {
                ocrResults.push({
                    bbox_id: bbox.id,
                    ocr_text: bbox.ocr_original,
                    ocr_confidence: bbox.ocr_confidence || 0,
                    corrected_text: bbox.text,
                    predicted_label: bbox.label,
                    learning_confidence: bbox.was_corrected ? 0.9 : 0.5
                });
            }
        });
        
        updateOCRResultList();
        return;
    }
    
    console.log(`Executing OCR on ${emptyBboxes.length} empty bboxes`);
    
    // Show spinner
    const spinner = document.getElementById('ocrProcessingSpinner');
    const content = document.getElementById('ocrResultContent');
    if (spinner) {
        spinner.style.display = 'flex';
    }
    if (content) {
        content.style.display = 'none';
    }
    
    // OCR 결과 초기화
    ocrResults = [];
    
    for (const bbox of emptyBboxes) {
        const ocrResult = await executeOCROnBbox(bbox);
        if (ocrResult && ocrResult.ocr_text) {
            bbox.text = ocrResult.ocr_text;
            bbox.ocr_original = ocrResult.ocr_text;
            bbox.ocr_confidence = ocrResult.ocr_confidence;
            
            // OCR 결과 추가
            ocrResults.push({
                bbox_id: bbox.id,
                ocr_text: ocrResult.ocr_text,
                ocr_confidence: ocrResult.ocr_confidence,
                corrected_text: bbox.text,
                predicted_label: bbox.label,
                learning_confidence: 0.5
            });
        }
    }
    
    // Hide spinner
    if (spinner) {
        spinner.style.display = 'none';
    }
    if (content) {
        content.style.display = 'block';
    }
    
    updateBoundingBoxList();
    updateOCRResultList();
    
    // 스피너가 확실히 숨겨지도록 추가 확인
    setTimeout(() => {
        const spinnerCheck = document.getElementById('ocrProcessingSpinner');
        if (spinnerCheck && spinnerCheck.style.display !== 'none') {
            console.log('Force hiding spinner');
            spinnerCheck.style.display = 'none';
        }
        const contentCheck = document.getElementById('ocrResultContent');
        if (contentCheck && contentCheck.style.display === 'none') {
            console.log('Force showing content');
            contentCheck.style.display = 'block';
        }
    }, 500);
}

/**
 * Execute OCR on all bounding boxes
 */
async function executeOCROnAll() {
    if (bboxData.length === 0) {
        showAlert('Bounding Box가 없습니다.', 'warning');
        return;
    }
    
    // Show spinner
    const spinner = document.getElementById('ocrProcessingSpinner');
    const content = document.getElementById('ocrResultContent');
    if (spinner) {
        spinner.style.display = 'flex';
        console.log('Spinner shown');
    }
    if (content) {
        content.style.display = 'none';
        console.log('Content hidden');
    }
    
    showAlert('OCR 실행 중...', 'info');
    ocrResults = [];  // Clear previous results
    
    // Execute OCR only for necessary bboxes
    for (const bbox of bboxData) {
        // Check if OCR is needed for this bbox
        const shouldOcrResponse = await fetch('/api/should_ocr_bbox', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: {
                    x: bbox.x,
                    y: bbox.y,
                    width: bbox.width,
                    height: bbox.height
                },
                document_type: 'purchase_order',
                layout_pattern: currentLabelData?.layout_pattern || 'pattern_B'
            })
        });
        
        const shouldOcrData = await shouldOcrResponse.json();
        
        if (shouldOcrData.should_ocr) {
            const ocrResult = await executeOCROnBbox(bbox);
            if (ocrResult) {
                // Apply learning-based corrections
                const correctedResult = await applyOCRLearning(bbox, ocrResult);
                
                // bbox에 OCR 데이터 저장
                bbox.ocr_original = ocrResult.ocr_text || '';
                bbox.ocr_confidence = ocrResult.ocr_confidence || 0;
                
                // OCR 결과와 현재 텍스트가 다르면 was_corrected 표시
                if (bbox.text && bbox.text !== bbox.ocr_original) {
                    bbox.was_corrected = true;
                } else if (!bbox.text) {
                    // 텍스트가 없으면 OCR 결과 사용
                    bbox.text = bbox.ocr_original;
                    bbox.was_corrected = false;
                }
                
                ocrResults.push({
                    bbox_id: bbox.id,
                    x: bbox.x,
                    y: bbox.y,
                    width: bbox.width,
                    height: bbox.height,
                    original_text: bbox.text || '',
                    original_label: bbox.label || '',
                    ocr_text: ocrResult.ocr_text || '',
                    ocr_confidence: ocrResult.ocr_confidence || 0,
                    corrected_text: correctedResult.corrected_text || ocrResult.ocr_text || '',
                    predicted_label: correctedResult.predicted_label || shouldOcrData.predicted_label || '',
                    learning_confidence: correctedResult.confidence || 0
                });
            }
        } else {
            // Skip OCR but still add predicted label
            ocrResults.push({
                bbox_id: bbox.id,
                x: bbox.x,
                y: bbox.y,
                width: bbox.width,
                height: bbox.height,
                original_text: bbox.text || '',
                original_label: bbox.label || '',
                ocr_text: '(OCR 생략)',
                ocr_confidence: 0,
                corrected_text: bbox.text || '',
                predicted_label: shouldOcrData.predicted_label || '',
                learning_confidence: 0.5
            });
        }
    }
    
    updateOCRResultList();
    
    // Hide spinner (reuse variables from above)
    if (spinner) spinner.style.display = 'none';
    if (content) content.style.display = 'block';
    
    showAlert('OCR 실행 완료!', 'success');
}

/**
 * Apply OCR learning corrections
 */
async function applyOCRLearning(bbox, ocrResult) {
    try {
        const response = await fetch('/api/apply_ocr_learning', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bbox: {
                    x: bbox.x,
                    y: bbox.y,
                    width: bbox.width,
                    height: bbox.height
                },
                ocr_result: ocrResult,
                document_type: 'purchase_order',
                layout_pattern: currentLabelData?.layout_pattern || 'pattern_B'
            })
        });
        
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Failed to apply OCR learning:', error);
    }
    
    return {
        corrected_text: ocrResult.ocr_text,
        predicted_label: '',
        confidence: 0
    };
}

/**
 * Update OCR result list display
 */
function updateOCRResultList() {
    console.log('updateOCRResultList called with', ocrResults.length, 'results');
    
    const listContainer = document.getElementById('ocrResultContent');
    if (!listContainer) {
        console.error('ocrResultContent container not found');
        // Fallback to old container
        const oldContainer = document.getElementById('ocrResultList');
        if (oldContainer) {
            console.log('Using fallback ocrResultList container');
            if (ocrResults.length === 0) {
                oldContainer.innerHTML = '<p>OCR 결과가 없습니다.</p>';
            }
        }
        return;
    }
    
    if (ocrResults.length === 0) {
        listContainer.innerHTML = '<p>OCR 결과가 없습니다.</p>';
        return;
    }
    
    let html = '';
    ocrResults.forEach((result, index) => {
        const confidenceClass = result.learning_confidence > 0.8 ? 'high' : 
                              result.learning_confidence > 0.5 ? 'medium' : 'low';
        
        html += `
            <div class="ocr-result-item" style="margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>Box ${index + 1}</strong>
                    <span class="confidence-${confidenceClass}" style="font-size: 12px;">
                        신뢰도: ${(result.learning_confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div style="margin-top: 5px;">
                    <label>OCR 원본:</label> ${result.ocr_text || '(없음)'}<br>
                    <label>보정된 텍스트:</label> <strong>${result.corrected_text || '(없음)'}</strong><br>
                    <label>예측 라벨:</label> <strong>${result.predicted_label || '(없음)'}</strong>
                </div>
                <div style="margin-top: 5px; font-size: 11px; color: #666;">
                    위치: (${result.x}, ${result.y}) | 크기: ${result.width}x${result.height}
                </div>
            </div>
        `;
    });
    
    listContainer.innerHTML = html + `
        <style>
            .confidence-high { color: #28a745; font-weight: bold; }
            .confidence-medium { color: #ffc107; }
            .confidence-low { color: #dc3545; }
        </style>
    `;
}

/**
 * Apply OCR results to bounding box list
 */
function applyOCRToBbox() {
    if (ocrResults.length === 0) {
        showAlert('적용할 OCR 결과가 없습니다.', 'warning');
        return;
    }
    
    let appliedCount = 0;
    
    // Apply OCR results to matching bboxes
    ocrResults.forEach(result => {
        const bbox = bboxData.find(b => b.id === result.bbox_id);
        if (bbox) {
            // Store original values for learning
            if (!bbox.ocr_original) {
                bbox.ocr_original = result.ocr_text;
                bbox.ocr_confidence = result.ocr_confidence;
            }
            
            // Apply corrected text and label
            if (result.corrected_text) {
                bbox.text = result.corrected_text;
            }
            if (result.predicted_label) {
                bbox.label = result.predicted_label;
            }
            
            appliedCount++;
        }
    });
    
    // Update the display
    updateBoundingBoxList();
    
    showAlert(`${appliedCount}개의 OCR 결과가 적용되었습니다.`, 'success');
}

/**
 * Show OCR Learning Status
 */
async function showOCRLearningStatus() {
    try {
        const response = await fetch('/api/ocr_learning_status');
        const status = await response.json();
        
        let statusHTML = `
            <h2>OCR 학습 상태</h2>
            <div class="status-overview">
                <p><strong>전체 정확도:</strong> ${(status.overall_accuracy * 100).toFixed(1)}%</p>
                <p><strong>총 보정 횟수:</strong> ${status.total_corrections}</p>
                <p><strong>학습 완료 여부:</strong> ${status.learning_complete ? '✅ 완료' : '🔄 진행중'}</p>
            </div>
            <h3>라벨별 상태</h3>
            <table class="status-table">
                <thead>
                    <tr>
                        <th>라벨</th>
                        <th>정확도</th>
                        <th>보정 횟수</th>
                        <th>학습 상태</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        for (const [label, data] of Object.entries(status.labels || {})) {
            const accuracy = (data.accuracy * 100).toFixed(1);
            const isActive = data.learning_active;
            const statusIcon = accuracy >= 95 ? '✅' : '🔄';
            
            statusHTML += `
                <tr>
                    <td>${label}</td>
                    <td>${accuracy}%</td>
                    <td>${data.corrections}</td>
                    <td>${statusIcon} ${isActive ? '학습중' : '완료'}</td>
                </tr>
            `;
        }
        
        statusHTML += `
                </tbody>
            </table>
            <style>
                .status-overview {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }
                .status-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .status-table th, .status-table td {
                    padding: 8px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                .status-table th {
                    background: #e9ecef;
                }
            </style>
        `;
        
        // Show in modal or alert
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 10000; max-width: 600px; max-height: 80vh; overflow-y: auto;';
        
        modal.innerHTML = statusHTML + '<button onclick="this.parentElement.remove()" style="margin-top: 20px;">닫기</button>';
        
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('Failed to get OCR learning status:', error);
        showAlert('OCR 학습 상태를 불러올 수 없습니다.', 'error');
    }
}

// Export functions for global access
window.loadLabelingDataForFile = loadLabelingDataForFile;

// Export all functions through labelingSystem
window.labelingSystem = {
    // Data functions
    loadLabelingData,
    saveLabelingData,
    saveCurrentLabelData,
    loadLabelingDataForFile,
    
    // Drawing functions
    startDrawing,
    stopDrawing,
    addBoundingBox,
    deleteSelectedBox,
    clearSelection,
    clearAllLabels,
    setQuickLabel,
    
    // Group functions
    toggleGroupMode,
    autoGroupByRows,
    createNewGroup,
    addToCurrentGroup,
    removeFromGroup,
    highlightGroups,
    
    // PDF functions
    previousPage,
    nextPage,
    loadPdfPage,
    
    // Auto-labeling functions
    getAutoLabels,
    
    // OCR Learning functions
    showOCRLearningStatus,
    executeOCROnAll,
    applyOCRToBbox,
    updateOCRResultList,
    trainModel,
    resetModel,
    showModelStats,
    
    // File functions
    deleteCollectedFile,
    uploadFiles
};