# YOKOGAWA OCR 데이터 준비 시스템

YOKOGAWA 문서의 OCR 학습을 위한 데이터 준비 시스템입니다.

## 주요 기능

- PDF/이미지 파일 업로드 및 관리
- 웹 기반 라벨링 인터페이스
- OCR 수행 및 보정
- OCR 학습 시스템 (일괄 학습)
- 모델 학습 및 자동 라벨 제안
- 데이터 증강 및 검증

## 빠른 시작

### 1. 환경 설정

```bash
# Python 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 실행

```bash
# 웹 인터페이스 실행 (기본)
python main.py

# 또는 직접 웹 서버 실행
python web_interface.py
```

웹 브라우저에서 http://localhost:5000 접속

### 3. 사용 방법

1. **파일 업로드**: Upload 메뉴에서 PDF 파일 업로드
2. **라벨링**: Labeling 메뉴에서 문서 선택 후 라벨링 작업
3. **OCR 실행**: "OCR 실행" 버튼으로 텍스트 추출
4. **라벨 저장**: 작업 완료 후 "저장" 버튼
5. **모델 학습**: "모델 학습" 버튼으로 OCR 및 라벨 모델 학습

## OCR 학습 시스템

### 작동 방식

1. OCR 실행 시 원본 텍스트가 `ocr_original` 필드에 저장
2. 사용자가 텍스트 수정 시 `was_corrected` 플래그 자동 설정
3. "모델 학습" 버튼 클릭 시 모든 보정 데이터로 일괄 학습
4. 학습된 패턴은 향후 OCR 수행 시 자동 적용

### 학습 데이터 위치

- 라벨 데이터: `data/processed/labels/`
- v2 형식 라벨: `data/processed/labels_v2/`
- 학습 모델: `data/models/`

## 고급 사용법

### 다른 실행 모드

```bash
# 전체 파이프라인 실행
python main.py --mode full

# 개별 단계 실행
python main.py --mode collection    # 데이터 수집
python main.py --mode labeling      # 라벨링
python main.py --mode augmentation  # 데이터 증강
python main.py --mode validation    # 검증
```

### 기존 라벨에 OCR 추가

```bash
# 특정 파일에 OCR 추가
python update_existing_labels_with_ocr.py --file 파일명.json

# 모든 라벨 파일에 OCR 추가
python update_existing_labels_with_ocr.py --all
```

## 프로젝트 구조

```
ymfk_ocrpjt/
├── config/           # 설정 파일
├── services/         # 핵심 서비스
│   ├── ocr_integration_service.py    # OCR 통합
│   ├── ocr_learning_service.py       # OCR 학습
│   └── ocr_batch_learning_service.py # 일괄 학습
├── static/          # 웹 정적 파일
├── templates/       # HTML 템플릿
├── data/           # 데이터 디렉토리
│   ├── raw/        # 원본 파일
│   ├── processed/  # 처리된 파일
│   └── models/     # 학습된 모델
└── main.py         # 메인 실행 파일
```

## 문제 해결

### Tesseract OCR 설치

- Windows: `tesseract/` 폴더에 포함됨
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

### 필요한 Python 패키지

```
flask
opencv-python
pytesseract
pillow
numpy
scikit-learn
PyPDF2
```

## 라이선스

YOKOGAWA 내부 사용 전용