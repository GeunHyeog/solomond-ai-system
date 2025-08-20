#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Unicode 문제 해결 시스템
Windows CP949 인코딩 문제 완전 해결

해결 방법:
1. 로그에서 이모지 제거
2. UTF-8 강제 설정
3. Windows 콘솔 호환성
"""

import os
import sys
import re
import logging
from pathlib import Path

def fix_unicode_issues():
    """유니코드 문제 해결"""
    print("=== Unicode 문제 해결 시작 ===")
    
    # 1. 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'ko_KR.UTF-8'
    
    # 2. 윈도우 콘솔 코드페이지 설정
    try:
        os.system('chcp 65001 > nul')
        print("[SUCCESS] 윈도우 콘솔 UTF-8 설정 완료")
    except:
        print("! 콘솔 설정 실패 (관리자 권한 필요할 수 있음)")
    
    # 3. 로깅 설정 수정
    setup_safe_logging()
    
    print("=== Unicode 문제 해결 완료 ===")

def setup_safe_logging():
    """안전한 로깅 설정 (이모지 없음)"""
    # 기존 핸들러 제거
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 새로운 핸들러 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                'system_logs/unicode_safe.log', 
                encoding='utf-8',
                mode='a'
            )
        ],
        force=True
    )
    
    print("[SUCCESS] 안전한 로깅 시스템 설정 완료")

def remove_emojis_from_text(text: str) -> str:
    """텍스트에서 이모지 제거"""
    # 이모지 패턴 제거
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def safe_print(text: str):
    """안전한 출력 함수"""
    try:
        # 이모지 제거 후 출력
        safe_text = remove_emojis_from_text(text)
        print(safe_text)
    except UnicodeEncodeError:
        # ASCII만 출력
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        print(ascii_text)

def create_safe_logger(name: str):
    """안전한 로거 생성"""
    logger = logging.getLogger(name)
    
    # 커스텀 핸들러 (이모지 제거)
    class SafeHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                safe_msg = remove_emojis_from_text(msg)
                self.stream.write(safe_msg + self.terminator)
                self.flush()
            except:
                self.handleError(record)
    
    if not logger.handlers:
        handler = SafeHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# 안전한 출력 함수들
def info(message: str):
    """정보 출력 (안전)"""
    safe_print(f"[INFO] {message}")

def success(message: str):
    """성공 메시지 (안전)"""
    safe_print(f"[SUCCESS] {message}")

def warning(message: str):
    """경고 메시지 (안전)"""
    safe_print(f"[WARNING] {message}")

def error(message: str):
    """오류 메시지 (안전)"""
    safe_print(f"[ERROR] {message}")

if __name__ == "__main__":
    # Unicode 문제 해결 실행
    fix_unicode_issues()
    
    # 테스트
    logger = create_safe_logger(__name__)
    
    info("시스템 보호 체계 초기화 완료")
    success("시스템 백업 생성 완료")
    warning("Tesseract 초기화 실패")
    error("Enhanced OCR 테스트 실패")
    
    # 일반적인 메시지들
    safe_print("SOLOMOND AI v7.1 Enhanced System 검증")
    safe_print("시스템 보호 모듈: 정상")
    safe_print("통합 제어기: 정상")
    safe_print("Enhanced OCR 엔진: 정상")
    
    print("Unicode 안전 시스템 테스트 완료!")