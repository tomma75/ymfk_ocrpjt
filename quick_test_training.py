#!/usr/bin/env python3
"""
모델 학습 빠른 테스트
"""
import requests
import json

BASE_URL = "http://localhost:5000"

print("1. 현재 모델 상태 확인...")
try:
    response = requests.get(f"{BASE_URL}/api/model_stats")
    stats = response.json()
    print(f"   - 학습 상태: {'학습됨' if stats.get('is_trained') else '미학습'}")
    if stats.get('is_trained'):
        print(f"   - 정확도: {stats.get('accuracy', 0):.2%}")
        print(f"   - 학습 샘플: {stats.get('training_samples', 0)}개")
except Exception as e:
    print(f"   ❌ 오류: {e}")
    exit(1)

print("\n2. 모델 학습 API 호출...")
try:
    response = requests.post(f"{BASE_URL}/api/train_model")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ 상태: {result.get('status', 'unknown')}")
        print(f"   - 총 샘플: {result.get('total_samples', 0)}개")
        print(f"   - 고유 라벨: {result.get('unique_labels', 0)}개")
        
        # OCR 학습 결과
        if 'ocr_learning' in result:
            ocr = result['ocr_learning']
            if ocr.get('ocr_training') == 'completed':
                print(f"   - OCR 학습: 완료")
                summary = ocr.get('summary', {})
                print(f"     • 처리 파일: {summary.get('total_files_processed', 0)}개")
                print(f"     • 보정 발견: {summary.get('total_corrections_found', 0)}개")
    else:
        print(f"   ❌ 오류: {response.status_code}")
        print(f"   응답: {response.text}")
        
except Exception as e:
    print(f"   ❌ 오류: {e}")

print("\n3. 학습 후 모델 상태 확인...")
try:
    response = requests.get(f"{BASE_URL}/api/model_stats")
    stats = response.json()
    print(f"   - 학습 상태: {'학습됨' if stats.get('is_trained') else '미학습'}")
    if stats.get('is_trained'):
        print(f"   - 정확도: {stats.get('accuracy', 0):.2%}")
        print(f"   - 학습 샘플: {stats.get('training_samples', 0)}개")
except Exception as e:
    print(f"   ❌ 오류: {e}")

print("\n✅ 테스트 완료!")