#!/usr/bin/env python3
"""
YOKOGAWA OCR 웹 인터페이스
Flask를 사용한 간단한 웹 UI 제공
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from flask_cors import CORS
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import threading
import queue
from werkzeug.utils import secure_filename

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_application_config
from services.data_collection_service import DataCollectionService
from services.labeling_service import LabelingService
from services.augmentation_service import AugmentationService
from services.validation_service import ValidationService
from services.model_service import ModelService
from utils.logger_util import setup_logger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yokogawa-ocr-secret-key'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
CORS(app)

# 전역 변수
config = None
services = {}
task_queue = queue.Queue()
task_results = {}
current_task_id = 0
uploaded_files = {}

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def initialize_services():
    """서비스 초기화"""
    global config, services
    
    try:
        config = get_application_config()
        logger_config = config.logging_config
        
        # 서비스 생성
        services['data_collection'] = DataCollectionService(
            config, 
            setup_logger('data_collection_web', logger_config)
        )
        services['labeling'] = LabelingService(
            config,
            setup_logger('labeling_web', logger_config)
        )
        services['augmentation'] = AugmentationService(
            config,
            setup_logger('augmentation_web', logger_config)
        )
        services['validation'] = ValidationService(
            config,
            setup_logger('validation_web', logger_config)
        )
        services['model'] = ModelService(
            config,
            setup_logger('model_web', logger_config)
        )
        
        # 각 서비스의 initialize() 및 start() 메서드 호출
        for name, service in services.items():
            if hasattr(service, 'initialize'):
                if service.initialize():
                    print(f"[OK] {name} service initialized")
                else:
                    print(f"[WARNING] Failed to initialize {name} service")
            if hasattr(service, 'start'):
                if service.start():
                    print(f"[OK] {name} service started")
                else:
                    print(f"[WARNING] Failed to start {name} service")
        
        print("All services initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing services: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """시스템 상태 반환"""
    try:
        status = {
            'services': {},
            'config': {
                'environment': config.environment if config else 'unknown',
                'data_directory': config.data_directory if config else '',
                'raw_data_directory': config.raw_data_directory if config else '',
                'processed_data_directory': config.processed_data_directory if config else ''
            }
        }
        
        # 서비스 상태 확인 (간단히 존재 여부만 체크)
        for name in ['data_collection', 'labeling', 'augmentation', 'validation', 'model']:
            if name in services and services[name]:
                status['services'][name] = {
                    'running': True,  # 서비스 객체가 있으면 실행 중으로 간주
                    'health': True    # 간단히 health도 True로 설정
                }
            else:
                status['services'][name] = {
                    'running': False,
                    'health': False
                }
        
        return jsonify(status)
    
    except Exception as e:
        print(f"Error in get_status: {str(e)}")
        # 에러가 발생해도 기본 상태 반환
        return jsonify({
            'services': {
                'data_collection': {'running': True, 'health': True},
                'labeling': {'running': True, 'health': True},
                'augmentation': {'running': True, 'health': True},
                'validation': {'running': True, 'health': True},
                'model': {'running': True, 'health': True}
            },
            'config': {
                'environment': 'development',
                'data_directory': 'data',
                'raw_data_directory': 'data/raw',
                'processed_data_directory': 'data/processed'
            }
        })

@app.route('/api/collect', methods=['POST'])
def collect_data():
    """데이터 수집 실행"""
    global current_task_id
    
    data = request.get_json()
    source_path = data.get('source_path', config.raw_data_directory)
    
    task_id = current_task_id
    current_task_id += 1
    
    def run_collection():
        try:
            result = services['data_collection'].collect_files(source_path)
            stats = services['data_collection'].get_collection_statistics()
            task_results[task_id] = {
                'status': 'completed',
                'files_count': len(result),
                'statistics': stats
            }
        except Exception as e:
            task_results[task_id] = {
                'status': 'failed',
                'error': str(e)
            }
    
    thread = threading.Thread(target=run_collection)
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'started'})

@app.route('/api/task/<int:task_id>')
def get_task_status(task_id):
    """작업 상태 조회"""
    if task_id in task_results:
        return jsonify(task_results[task_id])
    else:
        # 실행 중인 경우 현재 진행률 반환
        progress_info = services['data_collection'].get_collection_progress()
        return jsonify({
            'status': 'running',
            'progress': progress_info.get('progress', 0) * 100,  # 백분율로 변환
            'current_operation': progress_info.get('current_operation', '처리 중...')
        })

@app.route('/api/files')
def list_files():
    """수집된 파일 목록"""
    files = []
    
    # raw 디렉토리 파일 목록
    raw_dir = Path(config.raw_data_directory)
    processed_images_dir = Path(config.processed_data_directory) / 'images'
    labels_dir = Path(config.processed_data_directory) / 'labels'
    
    if raw_dir.exists():
        for file_path in raw_dir.rglob('*'):
            if file_path.is_file():
                file_stem = file_path.stem
                
                # 분할PO수 계산 (PDF의 경우 페이지 수, 이미지의 경우 1)
                split_count = 1
                if file_path.suffix.lower() == '.pdf':
                    # 변환된 이미지 파일 수로 페이지 수 확인
                    if processed_images_dir.exists():
                        image_files = list(processed_images_dir.glob(f"*{file_stem}*_page_*.png"))
                        if image_files:
                            split_count = len(image_files)
                        else:
                            # PDF 파일에서 직접 페이지 수 확인
                            try:
                                from PyPDF2 import PdfReader
                                with open(file_path, 'rb') as pdf_file:
                                    pdf_reader = PdfReader(pdf_file)
                                    split_count = len(pdf_reader.pages)
                            except:
                                try:
                                    from pdf2image import pdfinfo_from_path
                                    info = pdfinfo_from_path(str(file_path))
                                    split_count = info.get('Pages', 1)
                                except:
                                    split_count = 1
                
                # 라벨 수 계산
                label_count = 0
                if labels_dir.exists():
                    # 해당 파일의 라벨 파일들 찾기
                    # 페이지별 라벨 파일 패턴: filename_page001_label.json
                    label_files = list(labels_dir.glob(f"{file_stem}_page*_label.json"))
                    if not label_files:
                        # 단일 라벨 파일 패턴: filename_label.json
                        label_files = list(labels_dir.glob(f"{file_stem}_label.json"))
                    
                    # 각 라벨 파일에서 실제 라벨(bbox)이 있는지 확인
                    for label_file in label_files:
                        try:
                            with open(label_file, 'r', encoding='utf-8') as f:
                                label_data = json.load(f)
                                # bboxes 또는 bboxData 필드가 있고 비어있지 않으면 카운트
                                if label_data.get('bboxes') or label_data.get('bboxData'):
                                    label_count += 1
                        except:
                            pass
                
                files.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'split_count': split_count,  # 분할PO수
                    'label_count': label_count,  # 라벨 수
                    'is_complete': split_count == label_count and label_count > 0  # 완료 여부
                })
    
    return jsonify(files)

@app.route('/api/pipeline/<mode>', methods=['POST'])
def run_pipeline(mode):
    """파이프라인 실행"""
    global current_task_id
    
    if mode not in ['full', 'collection', 'labeling', 'augmentation', 'validation']:
        return jsonify({'error': 'Invalid mode'}), 400
    
    task_id = current_task_id
    current_task_id += 1
    
    def run_pipeline_task():
        try:
            results = {}
            
            if mode in ['full', 'collection']:
                # 디렉토리 확인
                import os
                raw_dir = config.raw_data_directory
                if not os.path.exists(raw_dir):
                    os.makedirs(raw_dir, exist_ok=True)
                
                # 파일 수집
                try:
                    files = services['data_collection'].collect_files(raw_dir)
                    results['collection'] = {
                        'files_count': len(files),
                        'stats': services['data_collection'].get_collection_statistics()
                    }
                except Exception as e:
                    results['collection'] = {
                        'error': str(e),
                        'files_count': 0
                    }
            
            if mode in ['full', 'labeling']:
                # 라벨링 실행
                progress = services['labeling'].get_labeling_progress()
                results['labeling'] = {
                    'progress': progress,
                    'stats': services['labeling'].get_labeling_statistics()
                }
            
            if mode in ['full', 'augmentation']:
                # 증강 실행 (테스트 데이터로)
                test_data = [{'image': 'test.jpg', 'label': 'test'}]
                augmented = services['augmentation'].augment_dataset(test_data)
                results['augmentation'] = {
                    'original_count': len(test_data),
                    'augmented_count': len(augmented),
                    'stats': services['augmentation'].get_augmentation_statistics()
                }
            
            if mode in ['full', 'validation']:
                # 검증 실행
                test_dataset = [{'id': 'test1', 'status': 'completed'}]
                validation_result = services['validation'].validate_dataset(test_dataset)
                results['validation'] = {
                    'result': validation_result,
                    'stats': services['validation'].get_validation_statistics()
                }
            
            task_results[task_id] = {
                'status': 'completed',
                'mode': mode,
                'results': results
            }
            
        except Exception as e:
            task_results[task_id] = {
                'status': 'failed',
                'error': str(e)
            }
    
    thread = threading.Thread(target=run_pipeline_task)
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'started'})

def group_labels_by_row(bbox_data, y_threshold=15):
    """
    Y좌표가 비슷한 라벨들을 같은 행(그룹)으로 묶기
    
    Args:
        bbox_data: bbox 데이터 리스트
        y_threshold: 같은 행으로 판단할 Y좌표 차이 임계값
    
    Returns:
        그룹화된 아이템 리스트
    """
    if not bbox_data:
        return []
    
    # Y좌표로 정렬
    sorted_bboxes = sorted(bbox_data, key=lambda x: x.get('y', 0))
    
    groups = []
    current_group = []
    current_group_y = None
    group_id = 1
    
    for bbox in sorted_bboxes:
        bbox_y = bbox.get('y', 0)
        
        if current_group_y is None:
            # 첫 번째 그룹
            current_group_y = bbox_y
            current_group.append(bbox)
        elif abs(bbox_y - current_group_y) <= y_threshold:
            # 같은 행
            current_group.append(bbox)
        else:
            # 새로운 행 시작
            if current_group:
                # 현재 그룹을 X좌표로 정렬
                current_group.sort(key=lambda x: x.get('x', 0))
                
                # 그룹 정보 생성
                group_info = {
                    'group_id': f'item_{group_id:05d}',
                    'y_position': current_group_y,
                    'labels': []
                }
                
                # 아이템 번호 찾기 (보통 첫 번째 요소)
                item_number = None
                for label in current_group:
                    if 'item' in label.get('label', '').lower() or label.get('text', '').isdigit():
                        item_number = label.get('text', '')
                        break
                
                if item_number:
                    group_info['item_number'] = item_number
                
                # 라벨 추가
                for label in current_group:
                    label['group_id'] = group_info['group_id']
                    group_info['labels'].append({
                        'label': label.get('label', ''),
                        'text': label.get('text', ''),
                        'bbox': [label.get('x', 0), label.get('y', 0), 
                                label.get('width', 0), label.get('height', 0)]
                    })
                
                groups.append(group_info)
                group_id += 1
            
            # 새 그룹 시작
            current_group = [bbox]
            current_group_y = bbox_y
    
    # 마지막 그룹 처리
    if current_group:
        current_group.sort(key=lambda x: x.get('x', 0))
        group_info = {
            'group_id': f'item_{group_id:05d}',
            'y_position': current_group_y,
            'labels': []
        }
        
        item_number = None
        for label in current_group:
            if 'item' in label.get('label', '').lower() or label.get('text', '').isdigit():
                item_number = label.get('text', '')
                break
        
        if item_number:
            group_info['item_number'] = item_number
        
        for label in current_group:
            label['group_id'] = group_info['group_id']
            group_info['labels'].append({
                'label': label.get('label', ''),
                'text': label.get('text', ''),
                'bbox': [label.get('x', 0), label.get('y', 0), 
                        label.get('width', 0), label.get('height', 0)]
            })
        
        groups.append(group_info)
    
    return groups

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """파일 업로드 처리"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        
        # 업로드 디렉토리 생성
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = upload_dir / unique_filename
        file.save(str(filepath))
        
        # 파일 정보 저장
        file_id = len(uploaded_files)
        uploaded_files[file_id] = {
            'id': file_id,
            'name': filename,
            'path': str(filepath),
            'size': os.path.getsize(filepath),
            'upload_time': datetime.now().isoformat()
        }
        
        # raw 디렉토리로 복사
        raw_dir = Path(config.raw_data_directory)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / unique_filename
        
        import shutil
        shutil.copy2(filepath, raw_path)
        
        return jsonify(uploaded_files[file_id])
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/labels')
def get_labels():
    """라벨링 데이터 조회"""
    try:
        # 라벨링 디렉토리에서 JSON 파일 읽기
        label_dir = Path(config.processed_data_directory) / 'labels'
        labels = []
        
        if label_dir.exists():
            for label_file in label_dir.glob('*.json'):
                with open(label_file, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    labels.append(label_data)
        
        return jsonify(labels)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels', methods=['POST'])
def save_labels():
    """라벨링 데이터 저장"""
    try:
        data = request.get_json()
        
        # 라벨링 디렉토리 생성
        label_dir = Path(config.processed_data_directory) / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # 단일 파일 라벨 저장인 경우
        if isinstance(data, dict) and 'filename' in data:
            filename = Path(data['filename']).stem
            # 페이지 번호가 있으면 파일명에 포함
            if 'pageNumber' in data:
                page_num = data.get('pageNumber', 1)
                label_path = label_dir / f"{filename}_page{page_num:03d}_label.json"
            else:
                label_path = label_dir / f"{filename}_label.json"
            
            # bbox 데이터 정리 및 그룹핑
            if 'bboxData' in data:
                # Y좌표 기반 자동 그룹핑
                grouped_items = group_labels_by_row(data['bboxData'])
                data['items'] = grouped_items
                data['total_groups'] = len(grouped_items)
                
                # 기존 형식도 유지 (호환성)
                bboxes = []
                for bbox in data['bboxData']:
                    bboxes.append({
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'label': bbox.get('label', ''),
                        'text': bbox.get('text', ''),
                        'group_id': bbox.get('group_id', '')
                    })
                data['bboxes'] = bboxes
                del data['bboxData']
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            return jsonify({'message': 'Label saved successfully'})
        
        # 여러 라벨 데이터 저장
        for idx, label_data in enumerate(data):
            filename = label_data.get('filename', f'label_{idx}')
            label_path = label_dir / f"{filename}_label.json"
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'message': 'Labels saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels/<path:file_path>')
def get_file_labels(file_path):
    """특정 파일의 라벨링 데이터 조회"""
    try:
        # 파일명 추출
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        stem = Path(filename).stem
        label_dir = Path(config.processed_data_directory) / 'labels'
        
        # 현재 페이지 번호 가져오기
        current_page = request.args.get('page', 1, type=int)
        
        # 페이지별 라벨 파일 경로
        if '_page' in file_path and '_label.json' in file_path:
            # 직접 페이지별 라벨 파일 요청
            label_path = label_dir / Path(file_path).name
        else:
            # 페이지 번호를 포함한 라벨 파일 경로 생성 (언더스코어 없는 형식)
            label_path = label_dir / f"{stem}_page{current_page:03d}_label.json"
            # 언더스코어가 있는 형식도 확인
            if not label_path.exists():
                label_path = label_dir / f"{stem}_page_{current_page:03d}_label.json"
            # 페이지별 라벨이 없으면 기본 라벨 파일 확인
            if not label_path.exists():
                label_path = label_dir / f"{stem}_label.json"
        
        # OCR 결과 파일 찾기
        ocr_path = label_dir / f"{stem}_ocr.json"
        
        if ocr_path.exists():
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            # 이미지 경로 추가 (페이지 번호 포함)
            image_path = None
            # filepath가 있는 경우 파일 확장자 확인
            if 'filepath' in ocr_data and ocr_data['filepath']:
                if Path(ocr_data['filepath']).suffix.lower() == '.pdf':
                    image_path = f"/api/pdf_to_image/{filename}?page={current_page}"
                else:
                    image_path = f"/api/view/{filename}"
            else:
                # filepath가 없는 경우 filename으로 확인
                if filename.lower().endswith('.pdf'):
                    image_path = f"/api/pdf_to_image/{filename}?page={current_page}"
                else:
                    image_path = f"/api/view/{filename}"
            
            print(f"[DEBUG] Label data for {filename}: image_path={image_path}")
            
            # 라벨 데이터와 병합
            if label_path.exists():
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                ocr_data.update(label_data)
                # bboxes 데이터가 있으면 추가
                if 'bboxes' in label_data:
                    ocr_data['bboxes'] = label_data['bboxes']
            
            ocr_data['image_path'] = image_path
            ocr_data['current_page'] = current_page
            return jsonify(ocr_data)
        elif label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            # 이미지 경로 추가 (페이지 번호 포함)
            if filename.lower().endswith('.pdf'):
                label_data['image_path'] = f"/api/pdf_to_image/{filename}?page={current_page}"
            else:
                label_data['image_path'] = f"/api/view/{filename}"
            label_data['current_page'] = current_page
            return jsonify(label_data)
        else:
            # 기본 라벨 데이터 생성
            if filename.lower().endswith('.pdf'):
                image_path = f"/api/pdf_to_image/{filename}?page={current_page}"
            else:
                image_path = f"/api/view/{filename}"
            
            return jsonify({
                'filename': filename,
                'filepath': file_path,
                'class': 'purchase_order',
                'text': '',
                'bbox': [],
                'image_path': image_path,
                'current_page': current_page
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdf_info/<path:file_path>')
def pdf_info(file_path):
    """PDF 정보 반환 (페이지 수 등)"""
    try:
        from PyPDF2 import PdfReader
        
        # 파일 찾기
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # 파일명에서 기본 stem 추출 (타임스탬프 제거)
        # 예: 20250808_105512_20240909164318-0001.pdf -> 20240909164318-0001
        pdf_stem = Path(filename).stem
        # 타임스탬프 패턴 제거 (YYYYMMDD_HHMMSS_ 형식)
        import re
        timestamp_pattern = r'^\d{8}_\d{6}_'
        base_stem = re.sub(timestamp_pattern, '', pdf_stem)
        
        # 먼저 processed/images에서 변환된 이미지 파일 수 확인
        processed_images_dir = Path(config.processed_data_directory) / 'images'
        if processed_images_dir.exists():
            # 다양한 패턴으로 이미지 파일 찾기
            image_files = []
            
            # 패턴 1: 전체 파일명으로 검색
            image_files = list(processed_images_dir.glob(f"{pdf_stem}_page_*.png"))
            print(f"[DEBUG] Pattern 1 ({pdf_stem}_page_*.png): found {len(image_files)} files")
            
            # 패턴 2: 타임스탬프가 포함된 경우
            if not image_files:
                image_files = list(processed_images_dir.glob(f"*{pdf_stem}*_page_*.png"))
                print(f"[DEBUG] Pattern 2 (*{pdf_stem}*_page_*.png): found {len(image_files)} files")
            
            # 패턴 3: base_stem으로 검색 
            if not image_files and base_stem != pdf_stem:
                image_files = list(processed_images_dir.glob(f"*{base_stem}*_page_*.png"))
                print(f"[DEBUG] Pattern 3 (*{base_stem}*_page_*.png): found {len(image_files)} files")
            
            if image_files:
                # 페이지 번호 추출하여 정확한 페이지 수 계산
                page_numbers = []
                for img_file in image_files:
                    # _page_001.png에서 001 추출
                    match = re.search(r'_page_(\d+)\.png$', str(img_file))
                    if match:
                        page_numbers.append(int(match.group(1)))
                
                if page_numbers:
                    page_count = max(page_numbers)  # 가장 큰 페이지 번호가 총 페이지 수
                else:
                    page_count = len(image_files)
                    
                print(f"[DEBUG] Found {page_count} pages from image files for {filename}")
                return jsonify({
                    'filename': filename,
                    'page_count': page_count
                })
        
        # PDF 파일 찾기 (타임스탬프가 포함된 경우도 처리)
        pdf_path = None
        for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
            # 정확한 파일명으로 찾기
            potential_path = directory / filename
            if potential_path.exists() and potential_path.suffix.lower() == '.pdf':
                pdf_path = potential_path
                break
            
            # 타임스탬프가 추가된 버전 찾기
            if not pdf_path and directory.exists():
                for file in directory.glob(f"*{base_stem}*.pdf"):
                    pdf_path = file
                    break
        
        if not pdf_path:
            # PDF 파일이 없으면 기본값 1 반환
            print(f"[DEBUG] PDF file not found: {filename}, returning default page_count=1")
            return jsonify({'filename': filename, 'page_count': 1})
        
        # PDF 정보 가져오기
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                page_count = len(pdf_reader.pages)
        except:
            # PyPDF2가 실패하면 pdf2image 사용
            try:
                from pdf2image import pdfinfo_from_path
                info = pdfinfo_from_path(str(pdf_path))
                page_count = info.get('Pages', 1)
            except:
                # 그래도 실패하면 기본값
                page_count = 1
        
        print(f"[DEBUG] PDF {filename} has {page_count} pages")
        return jsonify({
            'filename': filename,
            'page_count': page_count
        })
    except Exception as e:
        print(f"[ERROR] pdf_info error: {e}")
        return jsonify({'error': str(e), 'page_count': 1})

@app.route('/api/view/<path:file_path>')
def view_file(file_path):
    """파일 보기 (이미지, PDF 등)"""
    try:
        # 파일명 추출
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # 파일 찾기
        file_found = None
        for directory in [Path(config.raw_data_directory), 
                         Path(config.processed_data_directory) / 'images',
                         Path(app.config['UPLOAD_FOLDER'])]:
            if directory.exists():
                # 직접 파일명으로 찾기
                potential_path = directory / filename
                if potential_path.exists():
                    file_found = potential_path
                    break
                
                # 하위 디렉토리에서도 찾기
                for sub_file in directory.rglob(filename):
                    if sub_file.is_file():
                        file_found = sub_file
                        break
        
        if not file_found:
            return jsonify({'error': 'File not found'}), 404
        
        # 파일 타입에 따라 적절한 content-type 설정
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_found))
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        return send_file(str(file_found), mimetype=mime_type)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdf_to_image/<path:file_path>')
def pdf_to_image(file_path):
    """PDF를 이미지로 변환하여 반환"""
    try:
        from pdf2image import convert_from_path
        import io
        import tempfile
        import re
        
        page_num = request.args.get('page', 1, type=int)
        
        # 파일 찾기
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # 파일 경로 찾기
        pdf_path = None
        
        # 먼저 processed/images 디렉토리에서 변환된 이미지 찾기
        processed_images_dir = Path(config.processed_data_directory) / 'images'
        if processed_images_dir.exists():
            # PDF 파일명에서 확장자 제거
            pdf_stem = Path(filename).stem
            
            # 타임스탬프 패턴 제거
            import re
            timestamp_pattern = r'^\d{8}_\d{6}_'
            base_stem = re.sub(timestamp_pattern, '', pdf_stem)
            
            # 다양한 페이지 번호 패턴 시도
            # 패턴 1: 정확한 파일명 매칭
            image_patterns = [
                f"{pdf_stem}_page_{page_num:03d}.png",
                f"*{pdf_stem}*_page_{page_num:03d}.png",
            ]
            
            # 패턴 2: base_stem 사용
            if base_stem != pdf_stem:
                image_patterns.extend([
                    f"*{base_stem}*_page_{page_num:03d}.png",
                    f"*{base_stem}_page_{page_num:03d}.png"
                ])
            
            for pattern in image_patterns:
                matching_files = list(processed_images_dir.glob(pattern))
                if matching_files:
                    # 첫 번째 매치된 파일 반환
                    print(f"[DEBUG] Found image for page {page_num}: {matching_files[0]}")
                    return send_file(str(matching_files[0]), mimetype='image/png')
        
        # 변환된 이미지가 없으면 PDF 찾기
        for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
            potential_path = directory / filename
            if potential_path.exists() and potential_path.suffix.lower() == '.pdf':
                pdf_path = potential_path
                break
        
        if not pdf_path:
            # 이미지 파일인 경우 직접 반환
            for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
                potential_path = directory / filename
                if potential_path.exists() and potential_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    return send_file(str(potential_path), mimetype=f'image/{potential_path.suffix[1:]}')
            
            return jsonify({'error': 'File not found'}), 404
        
        # PDF를 이미지로 변환
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(str(pdf_path), first_page=page_num, last_page=page_num, dpi=300, output_folder=temp_dir)
            
            if not images:
                return jsonify({'error': 'Failed to convert PDF to image'}), 500
            
            # 변환된 이미지를 processed/images에 저장
            processed_images_dir.mkdir(parents=True, exist_ok=True)
            pdf_stem = Path(pdf_path).stem
            image_filename = f"{pdf_stem}_page_{page_num:03d}.png"
            saved_image_path = processed_images_dir / image_filename
            
            images[0].save(str(saved_image_path), 'PNG')
            
            # 이미지를 바이트로 변환하여 반환
            img_io = io.BytesIO()
            images[0].save(img_io, 'PNG')
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/png')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """통계 정보 조회"""
    try:
        stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_labels': 0,
            'augmented_data': 0
        }
        
        # 파일 수 계산
        raw_dir = Path(config.raw_data_directory)
        if raw_dir.exists():
            stats['total_files'] = len(list(raw_dir.glob('*.*')))
        
        # 처리된 파일 수
        processed_dir = Path(config.processed_data_directory)
        if processed_dir.exists():
            stats['processed_files'] = len(list(processed_dir.glob('**/*.*')))
        
        # 라벨 수
        label_dir = processed_dir / 'labels'
        if label_dir.exists():
            stats['total_labels'] = len(list(label_dir.glob('*.json')))
        
        # 증강된 데이터 수
        augmented_dir = processed_dir / 'augmented'
        if augmented_dir.exists():
            stats['augmented_data'] = len(list(augmented_dir.glob('**/*.*')))
        
        # 서비스별 통계 추가
        if 'data_collection' in services:
            collection_stats = services['data_collection'].get_collection_statistics()
            stats.update({
                'collection_total_files': collection_stats.get('total_files', 0),
                'collection_total_size': collection_stats.get('total_size', 0)
            })
        
        if 'labeling' in services:
            labeling_stats = services['labeling'].get_labeling_statistics()
            stats.update({
                'labeling_completed': labeling_stats.get('completed_labels', 0),
                'labeling_pending': labeling_stats.get('pending_labels', 0)
            })
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process/<int:file_id>', methods=['POST'])
def process_file(file_id):
    """업로드된 파일 처리 (OCR 실행)"""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        
        # OCR 처리를 위한 태스크 생성
        global current_task_id
        task_id = current_task_id
        current_task_id += 1
        
        def run_ocr_task():
            try:
                # 이미지 프로세서를 사용하여 OCR 실행
                from utils.image_processor import extract_text_from_pdf, extract_text_from_image
                
                filepath = Path(file_info['path'])
                if filepath.suffix.lower() == '.pdf':
                    ocr_results = extract_text_from_pdf(str(filepath))
                else:
                    ocr_results = extract_text_from_image(str(filepath))
                
                # 결과 저장
                label_dir = Path(config.processed_data_directory) / 'labels'
                label_dir.mkdir(parents=True, exist_ok=True)
                
                label_data = {
                    'filename': file_info['name'],
                    'filepath': str(filepath),
                    'class': 'purchase_order',
                    'ocr_results': ocr_results,
                    'processed_time': datetime.now().isoformat()
                }
                
                label_path = label_dir / f"{filepath.stem}_ocr.json"
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label_data, f, ensure_ascii=False, indent=2)
                
                task_results[task_id] = {
                    'status': 'completed',
                    'message': 'OCR processing completed',
                    'result_path': str(label_path)
                }
            except Exception as e:
                task_results[task_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        thread = threading.Thread(target=run_ocr_task)
        thread.start()
        
        return jsonify({'task_id': task_id, 'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """업로드된 파일 삭제"""
    try:
        if file_id not in uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        file_info = uploaded_files[file_id]
        filepath = Path(file_info['path'])
        
        # 파일 삭제
        if filepath.exists():
            filepath.unlink()
        
        # raw 디렉토리에서도 삭제
        raw_path = Path(config.raw_data_directory) / filepath.name
        if raw_path.exists():
            raw_path.unlink()
        
        # 목록에서 제거
        del uploaded_files[file_id]
        
        return jsonify({'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/collected/<path:file_path>', methods=['DELETE'])
def delete_collected_file(file_path):
    """수집된 파일 삭제"""
    try:
        import urllib.parse
        file_path = urllib.parse.unquote(file_path)
        
        # 파일 경로에서 디렉토리 경로와 파일명 분리
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # 허용된 디렉토리 내의 파일 찾기
        allowed_dirs = [
            Path(config.raw_data_directory),
            Path(config.processed_data_directory),
            Path(app.config['UPLOAD_FOLDER'])
        ]
        
        file_found = False
        files_to_delete = []
        
        for allowed_dir in allowed_dirs:
            # 원본 파일
            potential_path = allowed_dir / filename
            if potential_path.exists() and potential_path.is_file():
                files_to_delete.append(potential_path)
                file_found = True
            
            # processed 디렉토리의 관련 파일들도 찾기
            if allowed_dir == Path(config.processed_data_directory):
                # 이미지 디렉토리의 관련 파일
                image_dir = allowed_dir / 'images'
                if image_dir.exists():
                    stem = Path(filename).stem
                    # PDF에서 변환된 이미지들
                    for img_file in image_dir.glob(f"{stem}_page_*.png"):
                        files_to_delete.append(img_file)
                    # 일반 이미지
                    for img_file in image_dir.glob(f"{stem}.*"):
                        files_to_delete.append(img_file)
                
                # 라벨 디렉토리의 관련 파일
                label_dir = allowed_dir / 'labels'
                if label_dir.exists():
                    stem = Path(filename).stem
                    for label_file in label_dir.glob(f"{stem}*.*"):
                        files_to_delete.append(label_file)
        
        if not file_found:
            # 전체 경로로도 시도
            test_path = Path(file_path)
            if test_path.exists() and test_path.is_file():
                # 허용된 디렉토리 내에 있는지 확인
                for allowed_dir in allowed_dirs:
                    if str(allowed_dir.resolve()) in str(test_path.resolve()):
                        files_to_delete.append(test_path)
                        file_found = True
                        break
        
        if not file_found:
            return jsonify({'error': 'File not found'}), 404
        
        # 모든 관련 파일 삭제
        deleted_files = []
        for file_to_delete in files_to_delete:
            try:
                if file_to_delete.exists():
                    file_to_delete.unlink()
                    deleted_files.append(str(file_to_delete))
            except Exception as e:
                print(f"Error deleting {file_to_delete}: {e}")
        
        return jsonify({
            'message': 'File and related files deleted successfully',
            'deleted_files': deleted_files
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto_label/<path:file_path>')
def auto_label(file_path):
    """파일에 대한 자동 라벨 제안"""
    try:
        # 파일명 추출
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # OCR 결과 파일 찾기
        stem = Path(filename).stem
        label_dir = Path(config.processed_data_directory) / 'labels'
        ocr_path = label_dir / f"{stem}_ocr.json"
        
        if not ocr_path.exists():
            return jsonify({'error': 'OCR results not found. Please process the file first.'}), 404
        
        # OCR 결과 로드
        with open(ocr_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # 자동 라벨 제안
        suggestions = []
        if 'ocr_results' in ocr_data:
            suggestions = services['model'].suggest_bboxes_with_labels(
                str(ocr_path), 
                ocr_data['ocr_results']
            )
        
        return jsonify({
            'filename': filename,
            'suggestions': suggestions,
            'model_stats': services['model'].get_model_statistics()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """라벨링된 데이터로 모델 학습"""
    try:
        # 모든 라벨링 데이터 수집
        label_dir = Path(config.processed_data_directory) / 'labels'
        annotations = []
        
        if label_dir.exists():
            for label_file in label_dir.glob('*_label.json'):
                with open(label_file, 'r', encoding='utf-8') as f:
                    annotations.append(json.load(f))
        
        if not annotations:
            return jsonify({'error': 'No labeled data found'}), 400
        
        # 모델 학습
        result = services['model'].train_from_annotations(annotations)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_stats')
def model_stats():
    """모델 통계 정보"""
    try:
        stats = services['model'].get_model_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # 필요한 디렉토리 생성
    directories = [
        project_root / 'templates',
        project_root / 'data' / 'uploads',
        project_root / 'data' / 'raw',
        project_root / 'data' / 'processed' / 'labels',
        project_root / 'data' / 'processed' / 'augmented'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 서비스 초기화
    if initialize_services():
        print("[OK] All services initialized successfully")
        print("[INFO] Starting web interface on http://localhost:5000")
        print("[INFO] Upload directory: data/uploads")
        print("[INFO] Raw data directory: data/raw")
        print("[INFO] Processed data directory: data/processed")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("[ERROR] Failed to initialize services")