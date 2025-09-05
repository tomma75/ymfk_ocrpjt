#!/bin/bash

echo "캐시 및 임시 파일 정리 중..."

# Python 캐시 제거
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# Flask 캐시 제거 (있는 경우)
rm -rf .webassets-cache 2>/dev/null
rm -rf static/.webassets-cache 2>/dev/null

echo "캐시 정리 완료!"
echo "서버를 재시작하세요: python web_interface.py"