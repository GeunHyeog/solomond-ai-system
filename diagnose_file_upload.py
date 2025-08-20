#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파일 업로드 문제 진단 및 수정 도구
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
import os

def diagnose_upload_system():
    """파일 업로드 시스템 진단"""
    
    print("=== 파일 업로드 시스템 진단 ===")
    
    # 1. 경로 확인
    user_files_dir = Path("user_files")
    print(f"user_files 디렉토리 존재: {user_files_dir.exists()}")
    
    if user_files_dir.exists():
        jga_dir = user_files_dir / "JGA2025_D1"
        print(f"JGA2025_D1 폴더 존재: {jga_dir.exists()}")
        
        if jga_dir.exists():
            files = list(jga_dir.glob("*"))
            print(f"JGA2025_D1 폴더 내 파일 수: {len(files)}")
            
            # 파일 타입별 분류
            images = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            videos = [f for f in files if f.suffix.lower() in ['.mov', '.mp4', '.avi']]
            audios = [f for f in files if f.suffix.lower() in ['.wav', '.mp3', '.m4a']]
            
            print(f"이미지 파일: {len(images)}개")
            print(f"비디오 파일: {len(videos)}개")
            print(f"오디오 파일: {len(audios)}개")
            
            # 대용량 파일 확인
            large_files = [f for f in files if f.stat().st_size > 1024*1024*100]  # 100MB+
            print(f"대용량 파일 (100MB+): {len(large_files)}개")
            
            for lf in large_files:
                size_mb = lf.stat().st_size / (1024*1024)
                print(f"  - {lf.name}: {size_mb:.1f}MB")
    
    # 2. 임시 디렉토리 확인
    temp_dir = Path(tempfile.gettempdir())
    print(f"임시 디렉토리: {temp_dir}")
    print(f"임시 디렉토리 쓰기 가능: {os.access(temp_dir, os.W_OK)}")
    
    # 3. Streamlit 설정 확인
    print("\n=== Streamlit 설정 확인 ===")
    try:
        print(f"Streamlit 버전: {st.__version__}")
        print("Streamlit 설정 권장사항:")
        print("  maxUploadSize = 5000  # 5GB")
        print("  maxMessageSize = 1000  # 1GB")
    except:
        print("Streamlit 정보 확인 불가")
    
    return True

def create_test_files():
    """테스트용 작은 파일들 생성"""
    
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # 작은 텍스트 파일 생성
    (test_dir / "test.txt").write_text("테스트 파일입니다.", encoding='utf-8')
    
    # 가짜 이미지 파일 (작은 크기)
    fake_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00\x007n\xf9$\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01r\xdd\xe4\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    (test_dir / "test.png").write_bytes(fake_image_content)
    
    print(f"테스트 파일 생성: {test_dir}")
    return test_dir

if __name__ == "__main__":
    diagnose_upload_system()
    create_test_files()