# config/settings.py
"""
솔로몬드 AI 시스템 v3.0 - 시스템 설정 모듈
모든 설정과 환경 변수를 중앙 관리
"""

import os
from pathlib import Path
from typing import Dict, List

# 시스템 기본 정보
SYSTEM_INFO = {
    "name": "솔로몬드 AI 시스템",
    "version": "v3.0 (MCP 통합)",
    "developer": "전근혁 (솔로몬드 대표, 한국보석협회 사무국장)",
    "description": "실제 내용을 읽고 분석하는 차세대 AI 플랫폼",
    "license": "MIT License"
}

# 서버 설정
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": False,
    "reload": False,
    "log_level": "info"
}

# 파일 처리 설정
FILE_CONFIG = {
    "max_file_size": 200 * 1024 * 1024,  # 200MB
    "max_files": 10,
    "chunk_size": 8 * 1024 * 1024,  # 8MB 청크
    "temp_dir": "./temp",
    "upload_dir": "./uploads",
    "allowed_extensions": {
        "audio": [".mp3", ".wav", ".m4a"],
        "document": [".pdf", ".docx", ".txt"],
        "image": [".jpg", ".jpeg", ".png", ".gif"],
        "video": [".mp4", ".mov", ".avi", ".mkv"]
    }
}

# AI 분석 설정
AI_CONFIG = {
    "whisper_model": "base",  # tiny, base, small, medium, large
    "language": "ko",  # 기본 언어: 한국어
    "translation_enabled": True,
    "keyword_extraction": True,
    "summary_enabled": True,
    "max_keywords": 20
}

# 메모리 관리 설정
MEMORY_CONFIG = {
    "max_memory_usage": 500 * 1024 * 1024,  # 500MB
    "cleanup_interval": 300,  # 5분마다 정리
    "cache_enabled": True,
    "cache_ttl": 3600  # 1시간 캐시
}

# 4단계 워크플로우 설정
WORKFLOW_CONFIG = {
    "enable_step1": True,   # 정보입력
    "enable_step2": True,   # 업로드
    "enable_step3": True,   # 검토
    "enable_step4": True,   # 보고서
    "auto_progress": True,  # 자동 진행
    "save_sessions": True,  # 세션 저장
    "session_timeout": 3600  # 1시간
}

# UI/UX 설정
UI_CONFIG = {
    "theme": "modern",
    "language": "ko",
    "mobile_friendly": True,
    "progress_bar": True,
    "animations": True,
    "dark_mode": False
}

# 주얼리 업계 특화 설정
JEWELRY_CONFIG = {
    "enable_jewelry_terms": True,
    "gemstone_recognition": True,
    "metal_analysis": True,
    "price_analysis": False,  # 추후 구현
    "custom_keywords": [
        "다이아몬드", "루비", "사파이어", "에메랄드",
        "금", "은", "플래티넘", "팔라듐",
        "반지", "목걸이", "귀걸이", "팔찌",
        "캐럿", "등급", "컷", "투명도", "색상"
    ]
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_enabled": True,
    "file_path": "./logs/system.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# 보안 설정
SECURITY_CONFIG = {
    "cors_enabled": True,
    "cors_origins": ["*"],
    "rate_limiting": False,  # 추후 구현
    "api_key_required": False,  # 추후 구현
    "file_validation": True
}

# 개발 환경 설정
def get_env_config() -> Dict:
    """환경에 따른 설정 반환"""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return {
            **SERVER_CONFIG,
            "debug": False,
            "log_level": "warning"
        }
    elif env == "testing":
        return {
            **SERVER_CONFIG,
            "port": 8081,
            "debug": True
        }
    else:  # development
        return {
            **SERVER_CONFIG,
            "debug": True,
            "reload": True
        }

# 디렉토리 생성 함수
def ensure_directories():
    """필요한 디렉토리들을 생성"""
    dirs = [
        FILE_CONFIG["temp_dir"],
        FILE_CONFIG["upload_dir"],
        Path(LOGGING_CONFIG["file_path"]).parent
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# 설정 검증 함수
def validate_config() -> bool:
    """설정 값들이 올바른지 검증"""
    try:
        # 메모리 설정 검증
        if MEMORY_CONFIG["max_memory_usage"] < 100 * 1024 * 1024:  # 최소 100MB
            raise ValueError("메모리 설정이 너무 낮습니다")
        
        # 파일 크기 검증
        if FILE_CONFIG["max_file_size"] > 1024 * 1024 * 1024:  # 최대 1GB
            raise ValueError("파일 크기 제한이 너무 큽니다")
        
        # Whisper 모델 검증
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if AI_CONFIG["whisper_model"] not in valid_models:
            raise ValueError(f"지원하지 않는 Whisper 모델: {AI_CONFIG['whisper_model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 검증 실패: {e}")
        return False

# 설정 출력 함수
def print_config_summary():
    """현재 설정 요약 출력"""
    print(f"""
🚀 {SYSTEM_INFO['name']} {SYSTEM_INFO['version']}
========================================
👤 개발자: {SYSTEM_INFO['developer']}
🌐 서버: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}
💾 최대 파일: {FILE_CONFIG['max_file_size'] // 1024 // 1024}MB
🤖 Whisper 모델: {AI_CONFIG['whisper_model']}
🔄 4단계 워크플로우: {'✅ 활성화' if WORKFLOW_CONFIG['enable_step1'] else '❌ 비활성화'}
💎 주얼리 특화: {'✅ 활성화' if JEWELRY_CONFIG['enable_jewelry_terms'] else '❌ 비활성화'}
========================================
""")

if __name__ == "__main__":
    # 설정 검증 및 출력
    if validate_config():
        ensure_directories()
        print_config_summary()
    else:
        print("❌ 설정에 문제가 있습니다. 시스템을 시작할 수 없습니다.")