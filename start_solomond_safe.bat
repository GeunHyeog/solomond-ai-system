@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1
python -m streamlit run jewelry_stt_ui_v23_real_fixed.py --server.port 8503
pause
