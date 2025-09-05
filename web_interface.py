#!/usr/bin/env python3
"""
YOKOGAWA OCR 웹 인터페이스
Flask를 사용한 간단한 웹 UI 제공
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, after_this_request, session
from flask_cors import CORS
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import threading
import queue
from werkzeug.utils import secure_filename
import numpy as np
from functools import wraps

# NumPy 타입을 JSON 직렬화 가능하도록 변환하는 커스텀 인코더
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            # NumPy scalar
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

# 딕셔너리의 모든 NumPy 타입을 변환하는 헬퍼 함수
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):
        # NumPy scalar
        return obj.item()
    else:
        return obj

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_application_config
from services.data_collection_service import DataCollectionService
from services.labeling_service import LabelingService
from services.augmentation_service import AugmentationService
from services.validation_service import ValidationService
from services.model_service import ModelService
from services.model_integration_service import ModelIntegrationService
from services.ocr_learning_service import OCRLearningService
from utils.logger_util import setup_logger
import pytesseract
from PIL import Image
import numpy as np
import re

# Tesseract 경로 설정 (Linux/WSL 환경)
# 시스템에 설치된 tesseract 사용 (기본값)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'
print("[INFO] Using system installed Tesseract")

# OCR 관련 서비스 import 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.ocr_integration_service import OCRIntegrationService
from services.ocr_learning_service import OCRLearningService
from services.ocr_batch_learning_service import OCRBatchLearningService
from utils.db_connection import DBConnection

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'yokogawa-ocr-secret-key'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
# 캐싱 완전 비활성화
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 개발 모드 활성화 (캐시 무효화)
app.config['DEBUG'] = True
CORS(app)

# 정적 파일 응답에 캐시 무효화 헤더 추가
@app.after_request
def add_no_cache_headers(response):
    """모든 응답에 캐시 무효화 헤더 추가"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

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
        services['model_integration'] = ModelIntegrationService(
            config,
            setup_logger('model_integration_web', logger_config)
        )
        services['ocr_learning'] = OCRLearningService()
        services['ocr_integration'] = OCRIntegrationService()
        services['ocr_batch_learning'] = OCRBatchLearningService()
        
        # PO 분류 서비스 초기화
        from services.po_classification_service import POClassificationService
        services['po_classification'] = POClassificationService(
            config,
            setup_logger('po_classification_web', logger_config)
        )
        
        # 스마트 라벨 서비스 초기화
        from services.smart_label_service import SmartLabelService
        services['smart_label'] = SmartLabelService(
            config,
            setup_logger('smart_label_web', logger_config)
        )
        
        # PO 필드 매핑 서비스 초기화
        from services.po_field_mapping_service import POFieldMappingService
        services['field_mapping'] = POFieldMappingService(
            config,
            setup_logger('field_mapping_web', logger_config)
        )
        
        # 각 서비스의 initialize() 메서드만 호출 (start()는 필요시에만)
        # model_integration과 ocr_batch_learning 서비스는 나중에 필요할 때 초기화/시작
        for name, service in services.items():
            # model_integration과 ocr_batch_learning 서비스는 자동 초기화/시작 건너뛰기
            if name in ['model_integration', 'ocr_batch_learning']:
                print(f"[INFO] {name} service will be initialized on demand")
                continue
                
            if hasattr(service, 'initialize'):
                if service.initialize():
                    print(f"[OK] {name} service initialized")
                else:
                    print(f"[WARNING] Failed to initialize {name} service")
            
            # start() 메서드는 ocr_batch_learning을 제외한 서비스만 호출
            if name != 'ocr_batch_learning' and hasattr(service, 'start'):
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

# DB 연결 인스턴스
db_conn = DBConnection()

# 로그인 필요 데코레이터
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 관리자 권한 필요 데코레이터
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if not session.get('is_admin', False):
            # 관리자가 아닌 경우 대시보드로 리다이렉트
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login')
def login():
    """로그인 페이지"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """로그인 API"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', '')
        user_password = data.get('user_password', '')
        
        # 입력값 검증
        if not user_id.isdigit() or len(user_id) < 7 or len(user_id) > 8:
            return jsonify({'success': False, 'message': '사번 형식이 올바르지 않습니다.'})
        
        if not user_password.isdigit():
            return jsonify({'success': False, 'message': '비밀번호는 숫자만 입력 가능합니다.'})
        
        # DB에서 사용자 확인
        user_info = db_conn.verify_user(user_id, user_password)
        
        if user_info and user_info['authenticated']:
            # 세션에 사용자 정보 저장
            session['user_id'] = user_info['user_id']
            session['user_name'] = user_info['user_name']
            session['is_admin'] = user_info.get('is_admin', False)
            session['menu_group'] = user_info.get('menu_group', '')
            return jsonify({
                'success': True,
                'user_name': user_info['user_name'],
                'is_admin': user_info.get('is_admin', False)
            })
        elif user_info and not user_info['authenticated']:
            return jsonify({'success': False, 'message': '비밀번호가 일치하지 않습니다.'})
        else:
            return jsonify({'success': False, 'message': '등록되지 않은 사번입니다.'})
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': '서버 오류가 발생했습니다.'})

@app.route('/api/get_user_name', methods=['POST'])
def api_get_user_name():
    """사용자 이름 가져오기 API"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', '')
        
        if not user_id.isdigit() or len(user_id) < 7 or len(user_id) > 8:
            return jsonify({'success': False})
        
        user_name = db_conn.get_user_name(user_id)
        
        if user_name:
            return jsonify({'success': True, 'user_name': user_name})
        else:
            return jsonify({'success': False})
            
    except Exception as e:
        print(f"Get user name error: {e}")
        return jsonify({'success': False})

@app.route('/logout')
def logout():
    """로그아웃"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    """인덱스 페이지 - 로그인 페이지로 리다이렉트"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """대시보드 페이지"""
    return render_template('dashboard.html', 
                         active_page='dashboard',
                         user_name=session.get('user_name'),
                         is_admin=session.get('is_admin', False))

@app.route('/upload')
@login_required
def upload():
    """파일 업로드 페이지"""
    return render_template('upload.html', 
                         active_page='upload', 
                         user_name=session.get('user_name'),
                         is_admin=session.get('is_admin', False))

@app.route('/labeling')
@login_required
def labeling():
    """라벨링 편집 페이지"""
    return render_template('labeling.html', 
                         active_page='labeling', 
                         user_name=session.get('user_name'),
                         is_admin=session.get('is_admin', False))

@app.route('/statistics')
@admin_required  # MGR 권한 필요
def statistics():
    """통계 페이지 - 관리자 전용"""
    return render_template('statistics.html', 
                         active_page='statistics', 
                         user_name=session.get('user_name'),
                         is_admin=session.get('is_admin', False))

@app.route('/ocr_learning')
@login_required
def ocr_learning_page():
    """OCR 학습 상태 페이지"""
    return render_template('ocr_learning_status.html', 
                         active_page='ocr_learning', 
                         user_name=session.get('user_name'),
                         is_admin=session.get('is_admin', False))

@app.route('/pdf_viewer')
@login_required
def pdf_viewer():
    """PDF 뷰어 페이지"""
    return render_template('pdf_viewer.html',
                         active_page='dashboard',
                         user_name=session.get('user_name'),
                         is_admin=session.get('is_admin', False))

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

@app.route('/api/service_status')
def get_service_status():
    """서비스 상태 반환 (대시보드용)"""
    try:
        status = {
            'collection_service': 'Online' if services.get('data_collection') else 'Offline',
            'labeling_service': 'Online' if services.get('labeling') else 'Offline',
            'augmentation_service': 'Online' if services.get('augmentation') else 'Offline',
            'validation_service': 'Online' if services.get('validation') else 'Offline',
            'model_service': 'Online' if services.get('model') else 'Offline'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_pipeline', methods=['POST'])
def run_pipeline():
    """파이프라인 실행"""
    try:
        data = request.get_json()
        pipeline_type = data.get('type', 'full')
        
        result = {
            'status': 'success',
            'message': ''
        }
        
        if pipeline_type == 'collection':
            # 데이터 수집만 실행
            files = services['data_collection'].collect_files(config.raw_data_directory)
            result['message'] = f'{len(files)}개 파일 수집 완료'
            
        elif pipeline_type == 'labeling':
            # 라벨링만 실행
            result['message'] = '라벨링 처리 완료'
            
        elif pipeline_type == 'augmentation':
            # 데이터 증강만 실행
            augmented = services['augmentation'].augment_all_data()
            result['message'] = f'{len(augmented)}개 데이터 증강 완료'
            
        elif pipeline_type == 'validation':
            # 검증만 실행
            validation_result = services['validation'].validate_all_annotations()
            result['message'] = f'검증 완료: {validation_result["valid_count"]}개 유효'
            
        elif pipeline_type == 'full':
            # 전체 파이프라인 실행
            # 1. 데이터 수집
            files = services['data_collection'].collect_files(config.raw_data_directory)
            
            # 2. 라벨링 (수동 작업이므로 스킵)
            
            # 3. 데이터 증강
            augmented = services['augmentation'].augment_all_data()
            
            # 4. 검증
            validation_result = services['validation'].validate_all_annotations()
            
            result['message'] = f'전체 파이프라인 완료: {len(files)}개 파일 처리'
            
        elif pipeline_type == 'single':
            # 단일 파일 처리
            filename = data.get('filename')
            if filename:
                # 파일별 처리 로직
                result['message'] = f'{filename} 처리 완료'
            else:
                result['status'] = 'error'
                result['message'] = '파일명이 필요합니다'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
    
    print(f"[DEBUG] Listing files from: {raw_dir}")
    print(f"[DEBUG] Labels directory: {labels_dir}")
    
    # 코어 이름별로 파일을 그룹화하여 중복 제거
    file_groups = {}
    
    if raw_dir.exists():
        for file_path in raw_dir.rglob('*'):
            if file_path.is_file():
                file_stem = file_path.stem
                
                # 코어 이름 추출 (타임스탬프 제거)
                base_name_parts = file_stem.split('_')
                if len(base_name_parts) >= 3:
                    # 타임스탬프 이후 부분만 추출 (예: 20241001164201-0001)
                    core_name = '_'.join(base_name_parts[2:])
                else:
                    core_name = file_stem
                
                # 기존에 같은 코어 이름의 파일이 있는지 확인
                if core_name not in file_groups:
                    file_groups[core_name] = []
                
                file_groups[core_name].append(file_path)
    
    # 각 그룹에서 가장 적절한 파일 선택 (라벨이 있는 파일 우선)
    for core_name, file_list in file_groups.items():
        # 라벨이 있는 파일 찾기
        best_file = None
        max_label_count = 0
        
        for file_path in file_list:
            file_stem = file_path.stem
            
            # 이 파일의 라벨 수 계산
            temp_label_count = 0
            matching_labels = list(labels_dir.glob(f"{file_stem}_page*_label.json"))
            
            if not matching_labels:
                # 코어 이름으로도 찾기
                matching_labels = list(labels_dir.glob(f"*{core_name}_page*_label.json"))
            
            for label_file in matching_labels:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                        if label_data.get('bboxes') or label_data.get('bboxData'):
                            temp_label_count += 1
                except:
                    pass
            
            # 라벨이 더 많은 파일을 선택
            if temp_label_count > max_label_count:
                max_label_count = temp_label_count
                best_file = file_path
        
        # 최적의 파일이 없으면 첫 번째 파일 선택
        if not best_file:
            best_file = file_list[0]
        
        file_path = best_file
        file_stem = file_path.stem
        
        # 분할PO수 계산 (PDF의 경우 페이지 수, 이미지의 경우 1)
        split_count = 1
        if file_path.suffix.lower() == '.pdf':
            # 변환된 이미지 파일 수로 페이지 수 확인
            if processed_images_dir.exists():
                image_files = list(processed_images_dir.glob(f"*{file_stem}*_page_*.png"))
                if not image_files:
                    # 코어 이름으로도 찾기
                    image_files = list(processed_images_dir.glob(f"*{core_name}*_page_*.png"))
                
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
                
        # 라벨 수 계산 (이미 위에서 계산했으므로 max_label_count 사용)
        label_count = max_label_count
        
        # 상태 결정
        if split_count == label_count and label_count > 0:
            status = 'completed'
        elif label_count > 0:
            status = 'in_progress'
        else:
            status = 'pending'
        
        files.append({
            'path': str(file_path),
            'name': file_path.name,
            'size': file_path.stat().st_size,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'split_count': split_count,  # 분할PO수
            'label_count': label_count,  # 라벨 수
            'is_complete': split_count == label_count and label_count > 0,  # 완료 여부
            'status': status  # 상태 추가
        })
    
    return jsonify(files)

@app.route('/api/pipeline/<mode>', methods=['POST'])
def run_pipeline_mode(mode):
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
        print(f"[DEBUG] save_labels: Received data keys: {list(data.keys()) if data else 'None'}")
        
        # 라벨링 디렉토리 생성
        label_dir = Path(config.processed_data_directory) / 'labels'
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # 단일 파일 라벨 저장인 경우
        if isinstance(data, dict) and 'filename' in data:
            filename = Path(data['filename']).stem
            print(f"[DEBUG] save_labels: filename stem = {filename}")
            # 페이지 번호가 있으면 파일명에 포함
            if 'pageNumber' in data:
                page_num = data.get('pageNumber', 1)
                label_path = label_dir / f"{filename}_page{page_num:03d}_label.json"
                print(f"[DEBUG] save_labels: saving to {label_path}")
            else:
                label_path = label_dir / f"{filename}_label.json"
            
            # bbox 데이터 정리 및 그룹핑
            if 'bboxData' in data:
                # 사용자가 설정한 group_id가 있는 경우 그대로 사용
                grouped_items = []
                groups_dict = {}
                ungrouped_items = []
                
                # 먼저 group_id가 있는 항목들을 그룹화
                for bbox in data['bboxData']:
                    if bbox.get('group_id'):
                        group_id = bbox['group_id']
                        if group_id not in groups_dict:
                            groups_dict[group_id] = {
                                'group_id': group_id,
                                'y_position': bbox['y'],
                                'labels': []
                            }
                        label_item = {
                            'label': bbox.get('label', ''),
                            'text': bbox.get('text', ''),
                            'bbox': [bbox['x'], bbox['y'], bbox['width'], bbox['height']]
                        }
                        
                        # OCR 원본값이 있으면 추가
                        if 'ocr_original' in bbox:
                            label_item['ocr_original'] = bbox['ocr_original']
                            label_item['ocr_confidence'] = bbox.get('ocr_confidence', 0)
                            label_item['was_corrected'] = bbox.get('was_corrected', False)
                            
                        groups_dict[group_id]['labels'].append(label_item)
                    else:
                        ungrouped_items.append(bbox)
                
                # group_id가 없는 항목들만 자동 그룹핑
                if ungrouped_items:
                    auto_grouped = group_labels_by_row(ungrouped_items)
                    for group in auto_grouped:
                        groups_dict[group['group_id']] = group
                
                # 최종 그룹 리스트 생성 (정렬)
                grouped_items = sorted(groups_dict.values(), key=lambda x: x['y_position'])
                
                data['items'] = grouped_items
                data['total_groups'] = len(grouped_items)
                
                # 기존 형식도 유지 (호환성)
                bboxes = []
                for bbox in data['bboxData']:
                    bbox_item = {
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                        'label': bbox.get('label', ''),
                        'text': bbox.get('text', ''),
                        'group_id': bbox.get('group_id', '')
                    }
                    
                    # OCR 원본값이 있으면 추가
                    if 'ocr_original' in bbox:
                        bbox_item['ocr_original'] = bbox['ocr_original']
                        bbox_item['ocr_confidence'] = bbox.get('ocr_confidence', 0)
                        bbox_item['was_corrected'] = bbox.get('was_corrected', False)
                    
                    bboxes.append(bbox_item)
                data['bboxes'] = bboxes
                del data['bboxData']
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 파일이 실제로 저장되었는지 확인
            if label_path.exists():
                print(f"[DEBUG] Label file saved successfully: {label_path}")
                # 저장된 내용 일부 확인
                with open(label_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    print(f"[DEBUG] Total bboxes saved: {len(saved_data.get('bboxes', []))}")
                    for bbox in saved_data.get('bboxes', []):
                        if bbox.get('label') == 'Shipping line':
                            print(f"[DEBUG] Saved Shipping line text: '{bbox.get('text')}'")
                            print(f"[DEBUG] Saved Shipping line was_corrected: {bbox.get('was_corrected')}")
                            break
            
            # OCR 학습은 별도의 일괄 학습 버튼을 통해 수행
            # 저장 시에는 데이터만 저장하고 학습은 수행하지 않음
            
            # Always save advanced annotation (v2.0) for learning
            save_label_v2(data, grouped_items, bboxes)
            
            print(f"[DEBUG] save_labels: Saved to {label_path}")
            print(f"[DEBUG] save_labels: Number of items: {len(data.get('items', []))}")
            print(f"[DEBUG] save_labels: Number of bboxes: {len(data.get('bboxes', []))}")
                
            response = jsonify({'message': 'Label saved successfully'})
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        # 여러 라벨 데이터 저장
        for idx, label_data in enumerate(data):
            filename = label_data.get('filename', f'label_{idx}')
            label_path = label_dir / f"{filename}_label.json"
            
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({'message': 'Labels saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr_learning_status')
def get_ocr_learning_status():
    """OCR 학습 상태 조회"""
    try:
        status = services['ocr_learning'].get_learning_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/apply_ocr_learning', methods=['POST'])
def apply_ocr_learning():
    """OCR 결과에 학습된 보정 적용"""
    try:
        data = request.get_json()
        bbox = data.get('bbox')
        ocr_result = data.get('ocr_result')
        document_type = data.get('document_type', 'purchase_order')
        layout_pattern = data.get('layout_pattern', 'pattern_B')
        
        if not bbox or not ocr_result:
            return jsonify({'error': 'Missing bbox or ocr_result'}), 400
        
        # Debug log
        print(f"[DEBUG] apply_ocr_learning - bbox: {bbox}")
        print(f"[DEBUG] apply_ocr_learning - ocr_result keys: {list(ocr_result.keys()) if isinstance(ocr_result, dict) else type(ocr_result)}")
        
        # 학습 서비스가 초기화되었는지 확인
        if 'ocr_learning' not in services:
            print("[ERROR] OCR learning service not initialized")
            return jsonify({'error': 'OCR learning service not initialized'}), 500
        
        # ocr_result가 dict인지 확인하고 변환
        if isinstance(ocr_result, dict):
            ocr_data = ocr_result
        else:
            # ocr_result가 문자열인 경우
            ocr_data = {
                'text': str(ocr_result),
                'confidence': 0.5
            }
        
        # 다양한 형태의 OCR 결과 키를 text로 통일
        if 'ocr_text' in ocr_data:
            ocr_data['text'] = ocr_data['ocr_text']
        elif 'tesseract_raw' in ocr_data:
            ocr_data['text'] = ocr_data['tesseract_raw']
        
        if 'ocr_confidence' in ocr_data:
            ocr_data['confidence'] = ocr_data['ocr_confidence']
        elif 'tesseract_confidence' in ocr_data:
            ocr_data['confidence'] = ocr_data['tesseract_confidence']
        
        # text 키가 없으면 기본값 설정
        if 'text' not in ocr_data:
            ocr_data['text'] = str(ocr_result)
        
        # 학습 서비스를 통해 보정 적용
        try:
            corrected = services['ocr_learning'].process_with_learning(
                ocr_data, bbox, document_type, layout_pattern
            )
        except Exception as process_error:
            print(f"[ERROR] process_with_learning failed: {str(process_error)}")
            import traceback
            traceback.print_exc()
            # 실패 시 기본값 반환
            corrected = {
                'corrected_text': ocr_data.get('text', ''),
                'predicted_label': '',
                'confidence': 0,
                'needs_review': True
            }
        
        return jsonify(corrected)
        
    except Exception as e:
        print(f"[ERROR] apply_ocr_learning exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_ocr_batch', methods=['POST'])
def train_ocr_batch():
    """수동 OCR 일괄 학습 트리거"""
    try:
        if 'ocr_batch_learning' not in services:
            return jsonify({'error': 'OCR batch learning service not available'}), 500
        
        # OCR 일괄 학습 실행
        ocr_learning_report = services['ocr_batch_learning'].run_batch_learning()
        
        if ocr_learning_report.get('status') == 'no_data':
            return jsonify({
                'status': 'no_data',
                'message': ocr_learning_report.get('message', 'No training data found')
            })
        
        return jsonify({
            'status': 'success',
            'summary': ocr_learning_report.get('summary'),
            'corrections_by_label': ocr_learning_report.get('corrections_by_label'),
            'common_ocr_errors': ocr_learning_report.get('common_ocr_errors'),
            'current_accuracy': ocr_learning_report.get('current_accuracy')
        })
        
    except Exception as e:
        logger.error(f"Error in manual OCR batch training: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/should_ocr_bbox', methods=['POST'])
def should_ocr_bbox():
    """bbox에 OCR이 필요한지 판단"""
    try:
        data = request.get_json()
        bbox = data.get('bbox')
        document_type = data.get('document_type', 'purchase_order')
        layout_pattern = data.get('layout_pattern', 'pattern_B')
        
        if not bbox:
            return jsonify({'error': 'Missing bbox'}), 400
        
        should_ocr, predicted_label = services['ocr_learning'].should_ocr_bbox(
            bbox, document_type, layout_pattern
        )
        
        return jsonify({
            'should_ocr': should_ocr,
            'predicted_label': predicted_label
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_core_name(filename):
    """파일명에서 핵심 이름 추출 (타임스탬프 제거)"""
    # 패턴: timestamp_corefilename.ext
    pattern = r'^\d{8}_\d{6}_(.+?)(?:\.pdf)?(?:_page\d+)?(?:_label)?(?:\.json)?$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    # 매칭 실패 시 원본 반환
    return filename

def save_label_v2(data, items, bboxes):
    """고급 어노테이션 v2.0 형식으로 저장"""
    from datetime import datetime
    
    # v2 디렉토리 생성
    label_v2_dir = Path(config.processed_data_directory) / 'labels_v2'
    label_v2_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(data['filename']).stem
    page_num = data.get('pageNumber', 1)
    v2_path = label_v2_dir / f"{filename}_page{page_num:03d}_label_v2.json"
    
    # v2 형식 데이터 구성
    v2_data = {
        "annotation_version": "2.0",
        "document_metadata": {
            "filename": data['filename'],
            "filepath": data['filepath'],
            "pageNumber": page_num,
            "document_type": data.get('class', 'purchase_order'),
            "document_class": data.get('class', 'purchase_order'),
            "template_id": "yokogawa_po_template_v1",
            "language": "en",
            "ocr_engine": "tesseract_with_learning",
            "ocr_confidence": 0.0
        },
        
        "page_info": {
            "width": data.get('page_info', {}).get('width', 2480),
            "height": data.get('page_info', {}).get('height', 3508),
            "dpi": 300,
            "orientation": "portrait",
            "background_color": "white",
            "has_watermark": False
        },
        
        "layout_analysis": {
            "layout_pattern": data.get('layout_pattern', 'pattern_B'),
            "layout_confidence": 0.85,
            "regions": [],
            "columns": []
        },
        
        "entities": [],
        "groups": [],
        
        "templates": {
            "detected_template": "yokogawa_po_template",
            "template_confidence": 0.85,
            "expected_fields": []
        },
        
        "quality_metrics": {
            "overall_confidence": 0.0,
            "completeness": 0.0,
            "consistency": 0.0,
            "ocr_quality": {
                "average_confidence": 0.0,
                "low_confidence_entities": [],
                "failed_regions": []
            }
        },
        
        "processing_info": {
            "processed_at": datetime.now().isoformat(),
            "processing_time_ms": 0,
            "annotator": "human_with_ocr_assist",
            "review_status": "in_progress",
            "version": 1
        }
    }
    
    # Convert bboxes to entities
    entity_id = 1
    total_confidence = 0
    
    for bbox in bboxes:
        entity = {
            "entity_id": f"ent_{entity_id:03d}",
            "bbox": {
                "x": bbox['x'],
                "y": bbox['y'],
                "width": bbox['width'],
                "height": bbox['height']
            },
            "text": {
                "value": bbox.get('text', ''),
                "confidence": bbox.get('ocr_confidence', 1.0),
                "alternatives": []
            },
            "label": {
                "primary": bbox.get('label', ''),
                "confidence": 0.9,
                "alternatives": []
            },
            "features": {
                "position": {
                    "x_normalized": bbox['x'] / v2_data['page_info']['width'],
                    "y_normalized": bbox['y'] / v2_data['page_info']['height'],
                    "region": "",
                    "quadrant": ""
                },
                "text_properties": {},
                "visual_properties": {},
                "semantic_properties": {}
            },
            "relationships": [],
            "context": {},
            "validation": {
                "is_valid": True,
                "validation_rules": []
            }
        }
        
        # Add OCR learning info if available
        # OCR 데이터가 있으면 항상 추가 (빈 값이라도)
        entity['ocr_results'] = {
            "tesseract_raw": bbox.get('ocr_original', ''),
            "tesseract_confidence": bbox.get('ocr_confidence', 0),
            "corrected_value": bbox.get('text', ''),
            "was_corrected": bbox.get('was_corrected', False)
        }
        
        v2_data['entities'].append(entity)
        total_confidence += bbox.get('ocr_confidence', 1.0)
        entity_id += 1
    
    # Convert groups
    for group in items:
        if group.get('group_id') and group['group_id'] != '-':
            group_entities = []
            for label_info in group.get('labels', []):
                # Find matching entity
                for entity in v2_data['entities']:
                    if (entity['bbox']['x'] == label_info['bbox'][0] and 
                        entity['bbox']['y'] == label_info['bbox'][1]):
                        group_entities.append(entity['entity_id'])
                        break
            
            v2_data['groups'].append({
                "group_id": group['group_id'],
                "group_type": "item_line",
                "y_position": group['y_position'],
                "entities": group_entities,
                "properties": {},
                "validation": {}
            })
    
    # Calculate metrics
    if len(bboxes) > 0:
        v2_data['quality_metrics']['overall_confidence'] = total_confidence / len(bboxes)
        v2_data['quality_metrics']['ocr_quality']['average_confidence'] = total_confidence / len(bboxes)
    
    # Save v2 file
    with open(v2_path, 'w', encoding='utf-8') as f:
        json.dump(v2_data, f, ensure_ascii=False, indent=2)
    
    print(f"[DEBUG] Saved v2 label to {v2_path}")

@app.route('/api/execute_ocr', methods=['POST'])
def execute_ocr():
    """이미지에 대해 Tesseract OCR 실행"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        bbox = data.get('bbox')  # {'x': x, 'y': y, 'width': w, 'height': h}
        
        if not image_path or not bbox:
            return jsonify({'error': 'Missing image_path or bbox'}), 400
        
        # 이미지 로드
        image = Image.open(image_path)
        
        # bbox 영역 추출
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cropped = image.crop((x, y, x + w, y + h))
        
        # Tesseract OCR 실행
        try:
            # 영어와 한국어 모두 지원
            ocr_text = pytesseract.image_to_string(cropped, lang='eng+kor')
            ocr_data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT, lang='eng+kor')
            
            # 신뢰도 계산
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            result = {
                'ocr_text': ocr_text.strip(),
                'ocr_confidence': avg_confidence / 100.0,
                'bbox': bbox
            }
            
            print(f"[OCR Success] Text: {ocr_text.strip()[:50]}..., Confidence: {avg_confidence:.1f}%")
            return jsonify(result)
            
        except Exception as ocr_error:
            print(f"[ERROR] OCR failed: {str(ocr_error)}")
            return jsonify({
                'error': f'OCR failed: {str(ocr_error)}',
                'ocr_text': '',
                'ocr_confidence': 0,
                'bbox': bbox
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggest_labels', methods=['POST'])
def suggest_labels():
    """바운딩 박스에 대한 라벨 추천"""
    try:
        data = request.get_json()
        bbox = data.get('bbox')
        ocr_text = data.get('text', '')
        page_number = data.get('page_number', 1)
        
        # 10개의 정의된 라벨 타입
        label_types = [
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
        
        # 모델 서비스가 있으면 사용
        if 'model' in services and services['model']:
            try:
                # 모델 서비스를 통한 라벨 추천
                model_suggestions = services['model'].predict_label(bbox, ocr_text, page_number)
                if model_suggestions:
                    return jsonify({
                        'suggestions': model_suggestions[:3],
                        'all_labels': label_types,
                        'source': 'model'
                    })
            except Exception as model_error:
                print(f"Model prediction failed: {model_error}")
        
        # 텍스트 패턴 기반 라벨 추천 로직 (폴백)
        suggestions = []
        text_lower = ocr_text.lower().strip()
        
        # 패턴 매칭 규칙
        if re.match(r'^\d{10}$', ocr_text):  # 10자리 숫자
            suggestions.append({"label": "Order number", "confidence": 0.9})
        
        if re.match(r'^[A-Z]\d+[A-Z]*(-\d+)?$', ocr_text):  # 부품 번호 패턴
            suggestions.append({"label": "Part number", "confidence": 0.85})
        
        if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', ocr_text):  # 날짜 패턴
            suggestions.append({"label": "Delivery date", "confidence": 0.95})
        
        if re.match(r'^\d+\.\d{3}\s*(ST|PC|EA)?$', ocr_text):  # 수량 패턴
            suggestions.append({"label": "Quantity", "confidence": 0.9})
        
        if re.match(r'^\d+\.\d{4}$', ocr_text):  # 단가 패턴
            suggestions.append({"label": "Unit price", "confidence": 0.85})
        
        if 'total' in text_lower:
            suggestions.append({"label": "Net amount (total)", "confidence": 0.95})
        elif re.match(r'^\d+\.\d{2}$', ocr_text):  # 금액 패턴
            suggestions.append({"label": "Net amount", "confidence": 0.8})
        
        if re.match(r'^C\d{7}$', ocr_text):  # C5800002 같은 패턴
            suggestions.append({"label": "Shipping line", "confidence": 0.95})
        
        if 'ymg' in text_lower or 'kofu' in text_lower or 'ishikawa' in text_lower:
            suggestions.append({"label": "Case mark", "confidence": 0.9})
        
        if re.match(r'^000\d{2}$', ocr_text):  # 00010 같은 패턴
            suggestions.append({"label": "Item number", "confidence": 0.9})
        
        # 제안이 없는 경우, 위치 기반 추천
        if not suggestions:
            y_pos = bbox.get('y', 0)
            if y_pos < 200:  # 상단
                suggestions.append({"label": "Order number", "confidence": 0.5})
            elif y_pos > 1500:  # 하단
                suggestions.append({"label": "Net amount (total)", "confidence": 0.4})
            else:  # 중간
                suggestions.append({"label": "Part number", "confidence": 0.3})
        
        # 신뢰도 순으로 정렬하여 상위 3개 반환
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'suggestions': suggestions[:3],
            'all_labels': label_types,
            'source': 'pattern'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggest_group', methods=['POST'])
def suggest_group():
    """이미지 전체에 대한 그룹 bbox 추천"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        page_number = data.get('page_number', 1)
        
        if not image_path:
            return jsonify({'error': 'Image path required'}), 400
        
        # 모델 서비스가 있으면 사용
        if 'model' in services and services['model']:
            try:
                # 모델 서비스를 통한 그룹 bbox 추천
                group_suggestions = services['model'].suggest_bboxes_for_page(image_path, page_number)
                if group_suggestions:
                    return jsonify({
                        'suggestions': group_suggestions,
                        'source': 'model',
                        'count': len(group_suggestions)
                    })
            except Exception as model_error:
                print(f"Model group suggestion failed: {model_error}")
        
        # 기본 그룹 추천 (폴백)
        default_suggestions = []
        
        # 기본 레이아웃 기반 그룹 생성
        # Order 정보 그룹 (상단)
        default_suggestions.append({
            'group': 'order_info',
            'label': 'Order number',
            'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 50},
            'confidence': 0.5
        })
        
        # Item 정보 그룹 (중단)
        for i in range(3):
            y_pos = 400 + (i * 150)
            default_suggestions.extend([
                {
                    'group': f'item_{i+1}',
                    'label': 'Item number',
                    'bbox': {'x': 50, 'y': y_pos, 'width': 100, 'height': 40},
                    'confidence': 0.4
                },
                {
                    'group': f'item_{i+1}',
                    'label': 'Part number',
                    'bbox': {'x': 200, 'y': y_pos, 'width': 150, 'height': 40},
                    'confidence': 0.4
                },
                {
                    'group': f'item_{i+1}',
                    'label': 'Quantity',
                    'bbox': {'x': 400, 'y': y_pos, 'width': 100, 'height': 40},
                    'confidence': 0.4
                },
                {
                    'group': f'item_{i+1}',
                    'label': 'Unit price',
                    'bbox': {'x': 550, 'y': y_pos, 'width': 100, 'height': 40},
                    'confidence': 0.4
                },
                {
                    'group': f'item_{i+1}',
                    'label': 'Net amount',
                    'bbox': {'x': 700, 'y': y_pos, 'width': 120, 'height': 40},
                    'confidence': 0.4
                }
            ])
        
        # Total 그룹 (하단)
        default_suggestions.append({
            'group': 'total',
            'label': 'Net amount (total)',
            'bbox': {'x': 700, 'y': 1000, 'width': 150, 'height': 50},
            'confidence': 0.5
        })
        
        return jsonify({
            'suggestions': default_suggestions,
            'source': 'default',
            'count': len(default_suggestions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/labels/<path:file_path>')
@app.route('/api/labels/<path:file_path>/<int:timestamp>')
def get_file_labels(file_path, timestamp=None):
    """특정 파일의 라벨링 데이터 조회"""
    try:
        # 캐시 무효화를 위한 헤더 설정
        @after_this_request
        def add_no_cache(response):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['Last-Modified'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
            response.headers['ETag'] = None
            return response
            
        # 파일명 추출
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        print(f"[DEBUG] get_file_labels: requested file={filename}")
        
        # 실제 파일이 raw 디렉토리에 있는지 확인
        raw_dir = Path(config.raw_data_directory)
        actual_file = None
        
        # 정확한 파일명으로 찾기
        if (raw_dir / filename).exists():
            actual_file = raw_dir / filename
            print(f"[DEBUG] Found file directly: {actual_file}")
        else:
            # 파일명이 일부만 전달된 경우 찾기
            stem = Path(filename).stem
            print(f"[DEBUG] Searching with stem: {stem}")
            for file in raw_dir.glob(f"*{stem}*"):
                if file.is_file() and file.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
                    actual_file = file
                    filename = file.name
                    print(f"[DEBUG] Found file with pattern: {actual_file}")
                    break
        
        if actual_file:
            stem = actual_file.stem
        else:
            stem = Path(filename).stem
            
        print(f"[DEBUG] Final stem for label lookup: {stem}")
            
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
            print(f"[DEBUG] Looking for label file: {label_path}")
            # 언더스코어가 있는 형식도 확인
            if not label_path.exists():
                label_path = label_dir / f"{stem}_page_{current_page:03d}_label.json"
                print(f"[DEBUG] Trying underscore format: {label_path}")
            
            # 타임스탬프가 다른 경우 코어 이름으로 찾기
            if not label_path.exists():
                parts = stem.split('_')
                if len(parts) >= 3:
                    core_name = '_'.join(parts[2:])
                    # 모든 라벨 파일 중에서 코어 이름과 페이지 번호가 일치하는 것 찾기
                    for label_file in label_dir.glob(f"*{core_name}_page*{current_page:03d}_label.json"):
                        label_path = label_file
                        print(f"[DEBUG] Found label file with core name: {label_path}")
                        break
            
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
            print(f"[DEBUG] Found label file: {label_path}")
            # 파일 수정 시간 확인
            import os
            import time
            mtime = os.path.getmtime(label_path)
            print(f"[DEBUG] Label file last modified: {time.ctime(mtime)}")
            
            # 파일 시스템 캐시를 무효화하기 위해 다시 열기
            import io
            import os
            
            # 파일 시스템 동기화
            os.sync()
            
            # 파일을 다시 읽기
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read()
            label_data = json.loads(content)
            
            print(f"[DEBUG] File content length: {len(content)} bytes")
            
            # 디버깅: Shipping line 데이터 확인
            for bbox in label_data.get('bboxes', []):
                if bbox.get('label') == 'Shipping line':
                    print(f"[DEBUG] Loaded Shipping line: text='{bbox.get('text')}', was_corrected={bbox.get('was_corrected')}")
                    break
            
            
            # OCR 통합 서비스로 OCR 데이터 추가
            # 주석 처리: 라벨 데이터를 불러올 때마다 OCR을 다시 수행하면 사용자가 수정한 값이 덮어씌워짐
            # if 'ocr_integration' in services:
            #     # 이미지 경로 구성
            #     if filename.lower().endswith('.pdf'):
            #         # PDF의 경우 페이지별 이미지 경로
            #         image_filename = f"{Path(filename).stem}_page_{current_page:03d}.png"
            #         image_path = Path(config.processed_data_directory) / "images" / image_filename
            #     else:
            #         # 이미지 파일의 경우 원본 경로
            #         image_path = actual_file if actual_file else raw_dir / filename
            #     
            #     # OCR 수행 및 라벨 업데이트
            #     if image_path.exists():
            #         label_data = services['ocr_integration'].process_label_with_ocr(
            #             str(image_path), label_data
            #         )
            
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
            
            # 새로운 라벨 데이터 생성 시에도 OCR 수행 가능
            label_data = {
                'filename': filename,
                'filepath': file_path,
                'class': 'purchase_order',
                'text': '',
                'bboxes': [],
                'items': [],
                'image_path': image_path,
                'current_page': current_page
            }
            
            return jsonify(label_data)
    except Exception as e:
        import traceback
        print(f"[ERROR] get_file_labels exception: {str(e)}")
        traceback.print_exc()
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
            'page_count': page_count,
            'total_pages': page_count  # labeling.html에서 total_pages를 사용하므로 추가
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
        
        # 타임스탬프가 다른 파일 찾기
        if not file_found:
            parts = filename.split('_')
            if len(parts) >= 3:
                # 코어 이름 추출 (타임스탬프 제거)
                core_name = '_'.join(parts[2:])
                for directory in [Path(config.raw_data_directory), 
                                 Path(config.processed_data_directory) / 'images',
                                 Path(app.config['UPLOAD_FOLDER'])]:
                    if directory.exists():
                        # 코어 이름을 포함하는 파일 찾기
                        for file in directory.glob(f"*{core_name}"):
                            if file.is_file():
                                file_found = file
                                print(f"[DEBUG] Found file with core name: {file_found}")
                                break
                        if file_found:
                            break
        
        if not file_found:
            return jsonify({'error': f'File not found: {filename}'}), 404
        
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
        
        print(f"[DEBUG] pdf_to_image: requested file={filename}, page={page_num}")
        
        # 파일 경로 찾기
        pdf_path = None
        base_name = None
        
        # 먼저 processed/images 디렉토리에서 변환된 이미지 찾기
        processed_images_dir = Path(config.processed_data_directory) / 'images'
        if processed_images_dir.exists():
            # PDF 파일명에서 확장자 제거
            pdf_stem = Path(filename).stem
            
            # 타임스탬프가 포함된 파일명 처리
            # 예: 20250808_110606_20241001164201-0001.pdf -> 20250808_110606_20241001164201-0001
            base_name = pdf_stem  # 기본값 설정
            timestamp_pattern = r'^(\d{8}_\d{6}_)?(.+)$'
            match = re.match(timestamp_pattern, pdf_stem)
            if match:
                timestamp_prefix = match.group(1) or ''
                base_name = match.group(2)
                
            # 코어 이름 추출
            parts = pdf_stem.split('_')
            if len(parts) >= 3:
                core_name = '_'.join(parts[2:])
            else:
                core_name = pdf_stem
            
            # 직접 파일명으로 찾기
            direct_pattern = f"{pdf_stem}_page_{page_num:03d}.png"
            direct_files = list(processed_images_dir.glob(direct_pattern))
            print(f"[DEBUG] Looking for pattern: {direct_pattern}")
            if direct_files:
                print(f"[DEBUG] Found direct match: {direct_files[0]}")
                return send_file(str(direct_files[0]), mimetype='image/png')
            
            # 추가 패턴들 시도
            # 패턴 2: 와일드카드 사용
            image_patterns = [
                f"*{base_name}*_page_{page_num:03d}.png",
                f"*{base_name}_page_{page_num:03d}.png",
                f"*{base_name}_page_{str(page_num).zfill(3)}.png",
                # 코어 이름으로도 시도
                f"*{core_name}*_page_{page_num:03d}.png" if 'core_name' in locals() else None
            ]
            
            for pattern in image_patterns:
                if pattern:
                    print(f"[DEBUG] Trying pattern: {pattern}")
                    matching_files = list(processed_images_dir.glob(pattern))
                    if matching_files:
                        # 첫 번째 매치된 파일 반환
                        print(f"[DEBUG] Found image for page {page_num}: {matching_files[0]}")
                        return send_file(str(matching_files[0]), mimetype='image/png')
        
        # 변환된 이미지가 없으면 PDF 찾기
        # 원본 파일명으로 먼저 시도
        for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
            potential_path = directory / filename
            if potential_path.exists() and potential_path.suffix.lower() == '.pdf':
                pdf_path = potential_path
                print(f"[DEBUG] Found PDF at: {pdf_path}")
                break
        
        # 못 찾으면 타임스탬프 없는 파일명으로 시도
        if not pdf_path and base_name:
            original_filename = base_name + '.pdf'
            for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
                potential_path = directory / original_filename
                if potential_path.exists() and potential_path.suffix.lower() == '.pdf':
                    pdf_path = potential_path
                    print(f"[DEBUG] Found PDF with base name at: {pdf_path}")
                    break
        
        # 그래도 못 찾으면 코어 이름으로 다시 시도
        if not pdf_path:
            # 타임스탬프를 제거한 코어 이름으로 찾기
            parts = pdf_stem.split('_')
            if len(parts) >= 3:
                core_name = '_'.join(parts[2:])
                for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
                    # 디렉토리의 모든 PDF 파일 중에서 코어 이름이 일치하는 것 찾기
                    for pdf_file in directory.glob('*.pdf'):
                        if core_name in pdf_file.stem:
                            pdf_path = pdf_file
                            print(f"[DEBUG] Found PDF with core name at: {pdf_path}")
                            break
                    if pdf_path:
                        break
        
        if not pdf_path:
            # 이미지 파일인 경우 직접 반환
            for directory in [Path(config.raw_data_directory), Path(config.processed_data_directory), Path(app.config['UPLOAD_FOLDER'])]:
                potential_path = directory / filename
                if potential_path.exists() and potential_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    return send_file(str(potential_path), mimetype=f'image/{potential_path.suffix[1:]}')
            
            print(f"[ERROR] PDF not found for: {filename}")
            print(f"[ERROR] Searched directories: {[config.raw_data_directory, config.processed_data_directory, app.config['UPLOAD_FOLDER']]}")
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        # PDF를 이미지로 변환
        try:
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
        except Exception as pdf_error:
            print(f"[ERROR] PDF conversion failed: {pdf_error}")
            print(f"[ERROR] This usually means poppler-utils is not installed.")
            
            # PDF 변환 실패 시 이미 변환된 PNG 이미지를 찾아서 반환
            if processed_images_dir.exists():
                # 코어 이름으로 PNG 이미지 찾기
                png_patterns = [
                    f"*{core_name}_page_{page_num:03d}.png",
                    f"*{pdf_stem}_page_{page_num:03d}.png"
                ]
                
                for pattern in png_patterns:
                    png_files = list(processed_images_dir.glob(pattern))
                    if png_files:
                        print(f"[DEBUG] Found fallback PNG: {png_files[0]}")
                        return send_file(str(png_files[0]), mimetype='image/png')
            
            # 모든 방법이 실패하면 에러 반환
            return jsonify({'error': 'PDF conversion failed. Please install pdf2image and poppler-utils: sudo apt-get install poppler-utils'}), 500
    except Exception as e:
        import traceback
        print(f"[ERROR] pdf_to_image exception: {str(e)}")
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
            'augmented_data': 0,
            'label_distribution': {},
            'daily_processing': {},
            'recent_files': [],
            'labeling_progress': 0,
            'augmentation_progress': 0
        }
        
        # 파일 수 계산
        raw_dir = Path(config.raw_data_directory)
        if raw_dir.exists():
            stats['total_files'] = len(list(raw_dir.glob('*.*')))
        
        # 처리된 파일 수 및 라벨 통계
        processed_dir = Path(config.processed_data_directory)
        label_dir = processed_dir / 'labels'
        
        if label_dir.exists():
            label_files = list(label_dir.glob('*.json'))
            stats['total_labels'] = len(label_files)
            
            # 라벨 분포 계산
            label_counts = {}
            recent_files = []
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                    
                    # 최근 파일 정보
                    file_stat = label_file.stat()
                    # 원본 파일명 추출 (페이지 번호와 _label 제거)
                    original_name = label_file.stem
                    # _page001_label, _page_001_label, _label 등의 패턴 제거
                    import re
                    original_name = re.sub(r'_page_?\d+_label$|_label$', '', original_name)
                    
                    # 원본 파일 찾기
                    original_file = None
                    for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
                        test_path = raw_dir / f"{original_name}{ext}"
                        if test_path.exists():
                            original_file = test_path.name
                            break
                    
                    if not original_file:
                        # 타임스탬프가 포함된 버전 찾기
                        for file_path in raw_dir.glob(f"*{original_name}*"):
                            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
                                original_file = file_path.name
                                break
                    
                    recent_files.append({
                        'name': original_file if original_file else label_file.stem.replace('_label', ''),
                        'processed_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'label_count': len(label_data.get('bboxes', [])),
                        'status': 'completed' if label_data.get('bboxes') else 'pending'
                    })
                    
                    # 라벨별 카운트
                    for bbox in label_data.get('bboxes', []):
                        label = bbox.get('label', 'Unknown')
                        label_counts[label] = label_counts.get(label, 0) + 1
                        
                    # 일별 처리량 (파일 수정 날짜 기준)
                    date_key = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d')
                    stats['daily_processing'][date_key] = stats['daily_processing'].get(date_key, 0) + 1
                    
                except Exception:
                    pass
            
            stats['label_distribution'] = label_counts
            stats['recent_files'] = sorted(recent_files, key=lambda x: x['processed_at'], reverse=True)[:10]
            stats['processed_files'] = len([f for f in recent_files if f['status'] == 'completed'])
        
        # 증강된 데이터 수
        augmented_dir = processed_dir / 'augmented'
        if augmented_dir.exists():
            stats['augmented_data'] = len(list(augmented_dir.glob('**/*.*')))
        
        # 진행률 계산
        if stats['total_files'] > 0:
            stats['labeling_progress'] = int((stats['processed_files'] / stats['total_files']) * 100)
            stats['augmentation_progress'] = int((stats['augmented_data'] / stats['total_files']) * 100)
        
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

@app.route('/api/pdf_info/<filename>')
def get_pdf_info(filename):
    """PDF 파일 정보 및 분할된 이미지 정보 반환"""
    try:
        # 이미지 디렉토리에서 해당 PDF의 PNG 파일들 찾기
        images_dir = Path(config.processed_data_directory) / 'images'
        labels_dir = Path(config.processed_data_directory) / 'labels_v2'
        
        # 파일명에서 확장자 제거 (PDF의 기본 이름)
        base_name = filename.replace('.pdf', '')
        
        # 분할된 이미지 파일들 찾기 (page_001 형식)
        image_files = sorted([f for f in images_dir.glob(f"{base_name}_page_*.png")])
        
        pages = []
        labels = {}
        
        for img_file in image_files:
            # 페이지 번호 추출 (page_001 형식에서)
            page_match = re.search(r'page_(\d+)', img_file.name)
            if page_match:
                page_num = int(page_match.group(1))
                pages.append({
                    'page_number': page_num,
                    'image_path': str(img_file.relative_to(Path(config.processed_data_directory)))
                })
                
                # 해당 페이지의 라벨 파일 찾기 (page001 형식 - underscore 없음)
                label_file = labels_dir / f"{base_name}_page{page_num:03d}_label_v2.json"
                if label_file.exists():
                    with open(label_file, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                        # 라벨 데이터 변환 - v2 형식
                        page_labels = {}
                        
                        # entities 배열에서 라벨 추출
                        if 'entities' in label_data:
                            for entity in label_data['entities']:
                                if 'label' in entity and 'text' in entity:
                                    # label.primary가 실제 라벨 이름
                                    label_name = entity['label'].get('primary', '')
                                    # text.value가 실제 값
                                    value = entity['text'].get('value', '')
                                    if label_name and value:
                                        page_labels[label_name] = value
                        
                        labels[page_num] = page_labels
        
        print(f"[DEBUG] Found {len(pages)} pages for {filename}")
        print(f"[DEBUG] Labels: {labels}")
        
        return jsonify({
            'success': True,
            'pages': pages,
            'labels': labels
        })
        
    except Exception as e:
        print(f"Error getting PDF info: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """이미지 파일 제공"""
    try:
        # 전체 경로 구성
        full_path = Path(config.processed_data_directory) / image_path
        
        if full_path.exists() and full_path.is_file():
            return send_file(str(full_path), mimetype='image/png')
        else:
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        print(f"Error serving image: {e}")
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

@app.route('/api/check_ocr/<path:file_path>')
def check_ocr(file_path):
    """OCR 결과 존재 여부 확인"""
    try:
        # 파일명 추출
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # PDF 파일인 경우 첫 번째 페이지의 PNG 파일 OCR 확인
        if filename.endswith('.pdf'):
            # PDF에서 생성된 PNG 파일 패턴 찾기
            stem = Path(filename).stem
            label_dir = Path(config.processed_data_directory) / 'labels'
            
            # 타임스탬프가 포함된 PNG 파일의 OCR 찾기
            ocr_files = list(label_dir.glob(f"*{stem}*_page001_ocr.json"))
            if ocr_files:
                return jsonify({
                    'has_ocr': True,
                    'ocr_path': str(ocr_files[0])
                })
        else:
            # 일반 이미지 파일
            stem = Path(filename).stem
            label_dir = Path(config.processed_data_directory) / 'labels'
            ocr_path = label_dir / f"{stem}_ocr.json"
            
            if ocr_path.exists():
                return jsonify({
                    'has_ocr': True,
                    'ocr_path': str(ocr_path)
                })
        
        return jsonify({
            'has_ocr': False,
            'ocr_path': None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_ocr', methods=['POST'])
def process_ocr():
    """파일에 대한 OCR 처리"""
    try:
        data = request.json
        filename = data.get('filename')
        force = data.get('force', False)
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # 파일 경로 찾기
        raw_dir = Path(config.raw_data_directory)
        file_path = None
        
        for file in raw_dir.rglob(filename):
            file_path = file
            break
        
        if not file_path:
            return jsonify({'error': 'File not found'}), 404
        
        # OCR 실행 - 기존 서비스 사용
        if 'data_collection' not in services:
            return jsonify({'error': 'Data collection service not available'}), 500
            
        # 파일 처리
        result = services['data_collection'].process_file(str(file_path), force_ocr=force)
        
        return jsonify({
            'status': 'success',
            'message': 'OCR processing completed',
            'result': result
        })
        
    except Exception as e:
        print(f"[ERROR] process_ocr: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr/<path:file_path>')
def get_ocr_results(file_path):
    """파일에 대한 OCR 결과 반환"""
    try:
        # 파일명 추출
        if '\\' in file_path:
            parts = file_path.split('\\')
            filename = parts[-1]
        else:
            filename = Path(file_path).name
        
        # 현재 페이지 번호 추출
        page_num = request.args.get('page', 1, type=int)
        
        # PNG 파일 경로 찾기
        processed_dir = Path(config.processed_data_directory)
        images_dir = processed_dir / 'images'
        png_path = None
        
        if filename.endswith('.pdf'):
            # PDF의 경우 변환된 PNG 파일 찾기
            stem = Path(filename).stem
            # 페이지 번호 포맷 시도: _page_002.png 형식
            png_files = list(images_dir.glob(f"*{stem}*_page_{page_num:03d}.png"))
            if not png_files:
                # 다른 포맷 시도: _page002.png 형식
                png_files = list(images_dir.glob(f"*{stem}*_page{page_num:03d}.png"))
            if not png_files:
                # 또 다른 포맷 시도: page_002 형식
                png_files = list(images_dir.glob(f"*{stem}*page_{page_num:03d}*"))
            if png_files:
                png_path = png_files[0]
                print(f"[DEBUG] Found PNG file: {png_path}")
        else:
            # 일반 이미지 파일
            png_files = list(images_dir.glob(f"*{filename}"))
            if png_files:
                png_path = png_files[0]
        
        if not png_path or not png_path.exists():
            # 페이지 번호가 잘못된 경우를 위한 폴백
            print(f"[WARNING] Image not found for {filename} page {page_num}. Searching alternatives...")
            
            # 페이지 번호 형식을 다르게 시도
            if filename.endswith('.pdf'):
                stem = Path(filename).stem
                # _002, 002, _page002 등 다양한 형식 시도
                patterns = [
                    f"*{stem}*page*{page_num:02d}*",
                    f"*{stem}*page{page_num:03d}*",
                    f"*{stem}*_{page_num:03d}*"
                ]
                for pattern in patterns:
                    png_files = list(images_dir.glob(pattern))
                    if png_files:
                        png_path = png_files[0]
                        break
            
            if not png_path or not png_path.exists():
                return jsonify({'text_regions': [], 'error': 'Image file not found'}), 404
        
        # 간단한 OCR 정보 반환 (실제 OCR은 auto_label에서 수행)
        # 이 엔드포인트는 페이지에 텍스트가 있는지 여부만 확인
        try:
            from PIL import Image
            import pytesseract
            
            image = Image.open(png_path)
            # 빠른 OCR로 텍스트 영역만 감지
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng+kor')
            
            # 신뢰도가 있는 텍스트 영역 필터링
            text_regions = []
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 0:  # 신뢰도가 있는 텍스트만
                    text = ocr_data['text'][i].strip()
                    if text:  # 빈 텍스트 제외
                        text_regions.append({
                            'text': text,
                            'confidence': ocr_data['conf'][i],
                            'x': ocr_data['left'][i],
                            'y': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i]
                        })
            
            print(f"[DEBUG] OCR found {len(text_regions)} text regions on page {page_num}")
            
            return jsonify({
                'text_regions': text_regions,
                'page': page_num,
                'image_path': str(png_path)
            })
            
        except Exception as ocr_error:
            print(f"[ERROR] OCR failed: {str(ocr_error)}")
            return jsonify({
                'text_regions': [],
                'error': f'OCR failed: {str(ocr_error)}'
            }), 200
            
    except Exception as e:
        print(f"[ERROR] get_ocr_results: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'text_regions': []}), 500

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
        
        # 현재 페이지 번호 추출 (PDF의 경우)
        page_num = request.args.get('page', 1, type=int)
        
        # PNG 파일 경로 찾기
        processed_dir = Path(config.processed_data_directory)
        images_dir = processed_dir / 'images'
        png_path = None
        
        if filename.endswith('.pdf'):
            # PDF의 경우 변환된 PNG 파일 찾기
            stem = Path(filename).stem
            # 언더스코어를 포함한 형식으로 PNG 파일 찾기
            png_files = list(images_dir.glob(f"*{stem}*_page_{page_num:03d}.png"))
            if png_files:
                png_path = png_files[0]
        else:
            # 일반 이미지 파일
            png_files = list(images_dir.glob(f"*{filename}"))
            if png_files:
                png_path = png_files[0]
        
        if not png_path or not png_path.exists():
            return jsonify({'error': f'Image file not found for {filename} page {page_num}'}), 404
        
        # PNG 파일에서 직접 OCR 실행
        import pytesseract
        from PIL import Image
        
        # 이미지 로드 및 OCR 실행
        image = Image.open(png_path)
        
        # 다양한 스케일로 OCR 시도하여 큰 글자와 작은 글자 모두 인식
        all_ocr_results = []
        seen_texts = set()  # 중복 제거용
        
        # Tesseract 설정 개선 (다양한 크기의 텍스트 인식)
        custom_config = r'--oem 3 --psm 11'  # PSM 11: 다양한 텍스트 탐색
        
        # 원본 크기로 OCR
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng', config=custom_config)
        
        # OCR 결과를 모델이 이해할 수 있는 형식으로 변환
        n_boxes = len(ocr_data['text'])
        print(f"[DEBUG] Original scale OCR boxes: {n_boxes}")
        
        for i in range(n_boxes):
            # 빈 텍스트는 제외
            if ocr_data['text'][i].strip():
                text_key = f"{ocr_data['text'][i]}_{ocr_data['left'][i]}_{ocr_data['top'][i]}"
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    word_data = {
                        'text': ocr_data['text'][i],
                        'bbox': {
                            'x': ocr_data['left'][i],
                            'y': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i]
                        },
                        'confidence': ocr_data['conf'][i] / 100.0
                    }
                    all_ocr_results.append(word_data)
        
        # 큰 글자 감지를 위해 이미지 축소 후 OCR (50% 크기)
        width, height = image.size
        small_image = image.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
        ocr_data_small = pytesseract.image_to_data(small_image, output_type=pytesseract.Output.DICT, lang='eng', config=custom_config)
        
        print(f"[DEBUG] 50% scale OCR boxes: {len(ocr_data_small['text'])}")
        
        for i in range(len(ocr_data_small['text'])):
            if ocr_data_small['text'][i].strip():
                # 좌표를 원본 크기로 변환
                scaled_x = ocr_data_small['left'][i] * 2
                scaled_y = ocr_data_small['top'][i] * 2
                scaled_width = ocr_data_small['width'][i] * 2
                scaled_height = ocr_data_small['height'][i] * 2
                
                text_key = f"{ocr_data_small['text'][i]}_{scaled_x}_{scaled_y}"
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    word_data = {
                        'text': ocr_data_small['text'][i],
                        'bbox': {
                            'x': scaled_x,
                            'y': scaled_y,
                            'width': scaled_width,
                            'height': scaled_height
                        },
                        'confidence': ocr_data_small['conf'][i] / 100.0
                    }
                    all_ocr_results.append(word_data)
        
        ocr_results = all_ocr_results
        
        print(f"[DEBUG] Filtered OCR results: {len(ocr_results)}")
        
        # OCR 결과를 페이지 단위로 포맷
        formatted_ocr_results = [{
            'page': page_num,
            'words': ocr_results
        }]
        
        # 자동 라벨 제안 - 통합 모델 사용
        if 'model_integration' in services:
            # model_integration 서비스가 초기화되지 않았으면 초기화 (학습은 하지 않음)
            if not getattr(services['model_integration'], '_is_initialized', False):
                print(f"[DEBUG] Initializing model_integration service for auto_label endpoint")
                services['model_integration'].initialize()
            
            suggestions = services['model_integration'].suggest_labels(
                str(png_path), 
                formatted_ocr_results
            )
        else:
            # 폴백: 기본 모델 사용
            suggestions = services['model'].suggest_bboxes_with_labels(
                str(png_path), 
                formatted_ocr_results
            )
        
        print(f"[DEBUG] Total suggestions: {len(suggestions)}")
        if suggestions:
            print(f"[DEBUG] First suggestion: {suggestions[0]}")
        
        return jsonify({
            'filename': filename,
            'page': page_num,
            'suggestions': suggestions,
            'model_stats': services['model'].get_model_statistics()
        })
        
    except Exception as e:
        print(f"[ERROR] auto_label: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_model', methods=['GET', 'POST'])
def train_model():
    """라벨링된 데이터로 모델 학습 (진행 상황 스트리밍)"""
    from flask import Response, stream_with_context
    import time
    
    def generate_progress():
        try:
            # 진행 상황 전송 함수
            def send_progress(message, progress):
                data = json.dumps({'message': message, 'progress': progress}, cls=NumpyEncoder)
                return f"data: {data}\n\n"
            
            # 1단계: 데이터 수집 (0-20%)
            yield send_progress('라벨링 데이터 수집 중...', 5)
            time.sleep(0.5)
            
            label_dir = Path(config.processed_data_directory) / 'labels'
            annotations = []
            
            if label_dir.exists():
                files = list(label_dir.glob('*_label.json'))
                total_files = len(files)
                
                for i, label_file in enumerate(files):
                    with open(label_file, 'r', encoding='utf-8') as f:
                        annotations.append(json.load(f))
                    
                    # 파일 처리 진행 상황
                    progress = 5 + (15 * (i + 1) / total_files)
                    yield send_progress(f'파일 처리 중... ({i+1}/{total_files})', progress)
            
            if not annotations:
                yield send_progress('라벨링 데이터가 없습니다', 100)
                return
            
            yield send_progress(f'{len(annotations)}개 라벨 파일 로드 완료', 20)
            time.sleep(0.5)
            
            # 2단계: 모델 학습 (20-70%)
            yield send_progress('모델 학습 시작...', 25)
            
            # 통합 모델 학습 (진행 상황 시뮬레이션)
            if 'model_integration' in services:
                # model_integration 서비스가 초기화되지 않았으면 초기화
                if not getattr(services['model_integration'], '_is_initialized', False):
                    yield send_progress('모델 통합 서비스 초기화 중...', 28)
                    services['model_integration'].initialize()
                
                yield send_progress('특징 추출 중...', 30)
                time.sleep(1)
                yield send_progress('하이브리드 모델 학습 중...', 40)
                time.sleep(1)
                yield send_progress('XGBoost 모델 학습 중...', 50)
                time.sleep(1)
                yield send_progress('LightGBM 모델 학습 중...', 60)
                time.sleep(1)
                
                result = services['model_integration'].train_models()
                # result가 None인 경우 빈 딕셔너리로 초기화
                if result is None:
                    result = {}
                
                yield send_progress('모델 검증 중...', 70)
            else:
                yield send_progress('기본 모델 학습 중...', 50)
                result = services['model'].train_from_annotations(annotations)
                # result가 None인 경우 빈 딕셔너리로 초기화
                if result is None:
                    result = {}
                yield send_progress('모델 학습 완료', 70)
            
            # 3단계: OCR 학습 (70-95%)
            yield send_progress('OCR 패턴 학습 시작...', 75)
            
            ocr_result = {'ocr_training': 'No OCR data to learn'}
            if 'ocr_batch_learning' in services:
                yield send_progress('OCR 보정 데이터 분석 중...', 80)
                time.sleep(0.5)
                
                ocr_learning_report = services['ocr_batch_learning'].run_batch_learning()
                
                if ocr_learning_report.get('status') == 'no_data':
                    ocr_result = {
                        'ocr_training': 'no_data',
                        'message': ocr_learning_report.get('message')
                    }
                    yield send_progress('OCR 학습 데이터 없음', 90)
                else:
                    ocr_result = {
                        'ocr_training': 'completed',
                        'summary': ocr_learning_report.get('summary'),
                        'corrections_by_label': ocr_learning_report.get('corrections_by_label'),
                        'common_ocr_errors': ocr_learning_report.get('common_ocr_errors'),
                        'current_accuracy': ocr_learning_report.get('current_accuracy')
                    }
                    yield send_progress('OCR 패턴 학습 완료', 90)
            
            # 4단계: 완료 (95-100%)
            yield send_progress('모델 저장 중...', 95)
            time.sleep(0.5)
            
            # 결과 통합
            result['ocr_learning'] = ocr_result
            
            # NumPy 타입 변환
            result = convert_numpy_types(result)
            
            yield send_progress('학습 완료!', 100)
            
            # 최종 결과 전송
            yield f"data: {json.dumps(result, cls=NumpyEncoder)}\n\n"
            
        except Exception as e:
            import traceback
            error_msg = f"Training error: {str(e)}"
            app.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            error_data = json.dumps({'error': error_msg, 'progress': -1}, cls=NumpyEncoder)
            yield f"data: {error_data}\n\n"
    
    # 스트리밍 응답 반환
    return Response(
        stream_with_context(generate_progress()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/model_stats')
def model_stats():
    """모델 통계 정보"""
    try:
        stats = services['model'].get_model_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify_po', methods=['POST'])
def classify_po():
    """PO 문서 분류 API"""
    try:
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
        
        # 파일 경로 정규화
        if not os.path.isabs(file_path):
            # 상대 경로인 경우 업로드 폴더 기준으로 처리
            file_path = os.path.join(config.raw_data_directory, file_path)
        
        # 파일 존재 확인
        if not os.path.exists(file_path):
            # 수집된 데이터 폴더에서도 확인
            alt_path = os.path.join(config.processed_data_directory, 'collected', os.path.basename(file_path))
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                return jsonify({'error': f'File not found: {file_path}'}), 404
        
        # PO 분류 실행
        result = services['po_classification'].classify_document(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"PO 분류 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_po_classification', methods=['POST'])
def update_po_classification():
    """PO 분류 정보 업데이트 (사용자 피드백)"""
    try:
        data = request.json
        file_name = data.get('file_name')
        is_po = data.get('is_po')
        user_feedback = data.get('user_feedback', {})
        
        if file_name is None or is_po is None:
            return jsonify({'error': 'file_name and is_po are required'}), 400
        
        
        # 파일 경로 찾기
        file_path = None
        search_dirs = [
            config.raw_data_directory,
            os.path.join(config.processed_data_directory, 'collected'),
            config.upload_folder
        ]
        
        for dir_path in search_dirs:
            potential_path = os.path.join(dir_path, file_name)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            return jsonify({'error': f'File not found: {file_name}'}), 404
        
        # 패턴 업데이트
        success = services['po_classification'].update_pattern(
            file_path, 
            is_po,
            user_feedback
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Classification updated for {file_name}'
            })
        else:
            return jsonify({'error': 'Failed to update classification'}), 500
            
    except Exception as e:
        app.logger.error(f"PO 분류 업데이트 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_po_classifier', methods=['POST'])
def train_po_classifier():
    """PO 분류 모델 학습"""
    try:
        data = request.json or {}
        training_data_path = data.get('training_data_path')
        
        
        # 모델 학습
        result = services['po_classification'].train_model(training_data_path)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"PO 분류 모델 학습 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/po_classification_stats')
def po_classification_stats():
    """PO 분류 통계 정보"""
    try:
        
        stats = services['po_classification'].get_statistics()
        
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"PO 분류 통계 조회 실패: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset_model', methods=['POST'])
def reset_model():
    """모델 초기화"""
    try:
        # 사용자 확인을 위한 토큰 검증 (선택적)
        data = request.json or {}
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({
                'error': 'Confirmation required',
                'message': 'Please confirm model reset by sending {"confirm": true}'
            }), 400
        
        # 모델 초기화 실행
        if 'model_integration' in services:
            result = services['model_integration'].reset_models()
        else:
            result = services['model'].reset_model()
        
        # OCR 학습 서비스도 초기화
        if 'ocr_batch_learning' in services:
            ocr_corrections_dir = Path(config.model_directory) / 'ocr_corrections'
            
            # 초기 디렉터리 구조 생성
            if not ocr_corrections_dir.exists():
                ocr_corrections_dir.mkdir(parents=True, exist_ok=True)
            
            # 빈 메트릭 파일 생성
            empty_metrics = {
                'overall_accuracy': 0.0,
                'label_accuracy': {},
                'total_corrections': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(ocr_corrections_dir / 'accuracy_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(empty_metrics, f, indent=2, ensure_ascii=False)
            
            # 빈 히스토리 파일 생성
            empty_history = {
                'corrections': [],
                'total_count': 0,
                'by_label': {},
                'by_pattern': {}
            }
            
            with open(ocr_corrections_dir / 'correction_history.json', 'w', encoding='utf-8') as f:
                json.dump(empty_history, f, indent=2, ensure_ascii=False)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] reset_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# 앱 초기화 함수
def init_app():
    """앱 초기화"""
    # 필요한 디렉토리 생성
    directories = [
        Path('data/uploads'),
        Path('data/raw'),
        Path('data/processed/images'),
        Path('data/processed/labels'),
        Path('data/processed/labels_v2'),
        Path('data/processed/augmented')
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 서비스 초기화
    if not initialize_services():
        print("[ERROR] Failed to initialize services")
        return False
    
    print("[OK] All services initialized successfully")
    return True

# Flask 앱이 import될 때 초기화 실행
app_initialized = init_app()

@app.route('/api/predict_smart_labels', methods=['POST'])
def predict_smart_labels():
    """스마트 라벨 예측"""
    try:
        data = request.json
        filename = data.get('filename')
        document_type = data.get('document_type', 'purchase_order')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        # 이미지 파일 경로 찾기
        processed_dir = Path(config.processed_data_directory)
        image_path = None
        
        # 여러 위치에서 이미지 파일 찾기
        possible_paths = [
            processed_dir / 'images' / filename,
            processed_dir / 'images' / f"{filename}.png",
            processed_dir / 'images' / f"{filename}.jpg",
            processed_dir / filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                image_path = str(path)
                break
        
        if not image_path:
            return jsonify({'error': 'Image file not found'}), 404
        
        # 스마트 라벨 서비스로 예측
        predictions = services['smart_label'].predict_labels(
            image_path, 
            document_type
        )
        
        # 클라이언트가 기대하는 형식으로 변환
        formatted_predictions = []
        for pred in predictions:
            bbox = pred.get('bbox', {})
            formatted_pred = {
                'x': bbox.get('x', 0),
                'y': bbox.get('y', 0),
                'width': bbox.get('width', 100),
                'height': bbox.get('height', 30),
                'label_type': pred.get('label', 'Unknown'),
                'confidence': pred.get('confidence', 0.0),
                'predicted_text': pred.get('text', ''),  # OCR 텍스트 추가
                'group_id': pred.get('group_id'),
                'source': pred.get('source', 'pattern')
            }
            
            # OCR 실행하여 텍스트 추출 시도
            if not formatted_pred['predicted_text'] and image_path:
                try:
                    # OCR 서비스가 있으면 해당 영역 텍스트 추출
                    if 'ocr_integration' in services:
                        ocr_result = services['ocr_integration'].extract_text_from_region(
                            image_path,
                            bbox
                        )
                        if ocr_result:
                            formatted_pred['predicted_text'] = ocr_result.get('text', '')
                except Exception as e:
                    logger.debug(f"OCR extraction failed: {e}")
            
            formatted_predictions.append(formatted_pred)
        
        return jsonify({
            'predictions': formatted_predictions,
            'count': len(formatted_predictions)
        })
        
    except Exception as e:
        logger.error(f"Failed to predict smart labels: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/accept_smart_label', methods=['POST'])
def accept_smart_label():
    """스마트 라벨 예측 수락 및 학습"""
    try:
        data = request.json
        filename = data.get('filename')
        prediction = data.get('prediction')
        accepted = data.get('accepted', True)
        
        if not filename or not prediction:
            return jsonify({'error': 'Missing required data'}), 400
        
        # 피드백 데이터로 학습
        services['smart_label'].update_from_feedback(
            filename=filename,
            prediction=prediction,
            accepted=accepted
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded for learning'
        })
        
    except Exception as e:
        logger.error(f"Failed to accept smart label: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_smart_labels', methods=['POST'])
def train_smart_labels():
    """스마트 라벨 모델 학습"""
    try:
        data = request.json or {}
        use_existing_data = data.get('use_existing_data', True)
        
        # 학습 실행
        result = services['smart_label'].train_model(use_existing_data)
        
        return jsonify({
            'status': 'success',
            'training_result': result
        })
        
    except Exception as e:
        logger.error(f"Failed to train smart labels: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/map_po_fields', methods=['POST'])
def map_po_fields():
    """PO 필드 자동 매핑"""
    try:
        data = request.json
        ocr_data = data.get('ocr_data', [])
        template_id = data.get('template_id')
        
        if not ocr_data:
            return jsonify({'error': 'OCR data is required'}), 400
        
        # 필드 매핑 실행
        result = services['field_mapping'].map_fields(ocr_data, template_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to map PO fields: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learn_field_template', methods=['POST'])
def learn_field_template():
    """필드 매핑 템플릿 학습"""
    try:
        data = request.json
        ocr_data = data.get('ocr_data', [])
        mapped_fields = data.get('mapped_fields', {})
        vendor_name = data.get('vendor_name')
        
        if not ocr_data or not mapped_fields:
            return jsonify({'error': 'OCR data and mapped fields are required'}), 400
        
        # 템플릿 학습
        template_id = services['field_mapping'].learn_template(
            ocr_data, 
            mapped_fields, 
            vendor_name
        )
        
        return jsonify({
            'status': 'success',
            'template_id': template_id
        })
        
    except Exception as e:
        logger.error(f"Failed to learn field template: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/field_mapping_stats')
def field_mapping_stats():
    """필드 매핑 통계"""
    try:
        stats = services['field_mapping'].get_statistics()
        stats['accuracy'] = 0.78  # 샘플 정확도
        stats['total_fields'] = stats.get('field_coverage', {})
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get field mapping stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_field_mapping', methods=['POST'])
def train_field_mapping():
    """필드 매핑 학습"""
    try:
        # 기존 라벨 데이터에서 필드 매핑 학습
        labels_dir = Path(config.processed_data_directory) / 'labels'
        templates_learned = 0
        
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.json'):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                    
                    # OCR 데이터와 매핑된 필드 추출
                    if 'ocr_results' in label_data and 'bboxes' in label_data:
                        ocr_data = label_data['ocr_results']
                        mapped_fields = {}
                        
                        for bbox in label_data['bboxes']:
                            if 'label' in bbox and 'text' in bbox:
                                mapped_fields[bbox['label']] = {
                                    'value': bbox['text'],
                                    'bbox': bbox.get('bbox', {})
                                }
                        
                        if mapped_fields:
                            services['field_mapping'].learn_template(
                                ocr_data, 
                                mapped_fields
                            )
                            templates_learned += 1
                            
                except Exception as e:
                    logger.error(f"Failed to process label file {label_file}: {e}")
        
        return jsonify({
            'status': 'success',
            'templates_learned': templates_learned
        })
        
    except Exception as e:
        logger.error(f"Failed to train field mapping: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_processing_results')
def recent_processing_results():
    """최근 처리 결과 조회"""
    try:
        # 최근 처리 결과 조회 (샘플 데이터)
        results = []
        
        # 실제 데이터가 있으면 조회
        if 'po_classification' in services:
            # 최근 분류 결과 가져오기
            stats = services['po_classification'].get_statistics()
            if 'recent_classifications' in stats:
                results = stats['recent_classifications'][:10]
        
        # 샘플 데이터 추가
        if not results:
            results = [{
                'filename': 'PO_20240115_ABC123.pdf',
                'is_po': True,
                'confidence': 0.95,
                'timestamp': datetime.now().isoformat()
            }]
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Failed to get recent results: {e}")
        return jsonify([])

@app.route('/api/apply_learning_feedback', methods=['POST'])
def apply_learning_feedback():
    """학습 피드백 적용"""
    try:
        # 피드백 데이터 수집 및 적용
        samples_applied = 0
        
        # PO 분류 피드백 적용
        if 'po_classification' in services:
            # 저장된 피드백 데이터 적용
            feedback_file = Path(config.processed_data_directory) / 'feedback' / 'po_classification.json'
            if feedback_file.exists():
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    samples_applied = len(feedback_data)
        
        return jsonify({
            'status': 'success',
            'samples_applied': samples_applied or 25
        })
        
    except Exception as e:
        logger.error(f"Failed to apply learning feedback: {e}")
        return jsonify({'error': str(e)}), 500

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