@echo off
echo Starting SOLOMOND AI with GPU acceleration...

REM GPU 활성화 설정
set CUDA_VISIBLE_DEVICES=0

REM UTF-8 인코딩 설정
chcp 65001 > nul

REM Streamlit 실행
streamlit run conference_analysis_UNIFIED_COMPLETE.py --server.port 8610

pause