#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 설정 파일
자동 생성됨 - 2025-07-23 18:36:26
"""

import os
from pathlib import Path

# 기본 설정
CONFIG = {
    # 서버 설정
    "BASE_URL": "https://www.jckonline.com",
    "PORT": 8765,
    
    # 경로 설정  
    "DATA_PATH": "C:\\').free / (1024**3)",
    "UPLOAD_PATH": "./uploads",
    "RESULTS_PATH": "./results",
    
    # API 설정
    "API_TIMEOUT": 30,
    "MAX_FILE_SIZE": 100 * 1024 * 1024,  # 100MB
    
    # 연락처
    "CONTACT_EMAIL": "SOLOMONDd.jgh@gmail.com",
    
    # 환경별 설정
    "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
    "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
}

# 환경변수 오버라이드
def load_config():
    """환경변수로 설정 오버라이드"""
    config = CONFIG.copy()
    
    # 환경변수가 있으면 사용
    if os.getenv("SOLOMOND_BASE_URL"):
        config["BASE_URL"] = os.getenv("SOLOMOND_BASE_URL")
    
    if os.getenv("SOLOMOND_PORT"):
        config["PORT"] = int(os.getenv("SOLOMOND_PORT"))
    
    return config

# 전역 설정 인스턴스
SETTINGS = load_config()
