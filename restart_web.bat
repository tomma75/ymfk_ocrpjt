@echo off
echo Stopping existing Python processes...
taskkill /F /IM python.exe 2>nul || echo No Python process to kill

echo Starting YOKOGAWA OCR Web Interface...
cd /d "D:\8.간접업무자동화\1. PO시트입력자동화\YMFK_OCR"
python main.py --mode web

pause