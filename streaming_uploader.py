#!/usr/bin/env python3
"""
🚀 스트리밍 업로더 - 대용량 파일 메모리 안전 처리
3GB+ 파일도 메모리 초과 없이 안전하게 업로드
"""

import os
import tempfile
import shutil
from pathlib import Path
import streamlit as st
from typing import Optional, Generator

class StreamingUploader:
    """메모리 안전 스트리밍 업로더"""
    
    def __init__(self, chunk_size: int = 8192):  # 8KB 청크
        self.chunk_size = chunk_size
        self.temp_dir = Path(tempfile.gettempdir()) / "solomond_streaming"
        self.temp_dir.mkdir(exist_ok=True)
    
    def stream_to_temp(self, uploaded_file) -> Optional[Path]:
        """
        업로드된 파일을 스트리밍 방식으로 임시 파일에 저장
        메모리 사용량을 청크 크기로 제한
        """
        try:
            # 임시 파일 경로 생성
            temp_path = self.temp_dir / f"temp_{uploaded_file.name}"
            
            # 파일 크기 확인
            file_size = uploaded_file.size
            st.info(f"📁 스트리밍 업로드: {uploaded_file.name} ({file_size / (1024*1024*1024):.2f}GB)")
            
            # 진행률 바 생성
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 스트리밍 방식으로 파일 저장
            with open(temp_path, 'wb') as f:
                bytes_written = 0
                
                # 청크 단위로 읽기/쓰기
                while True:
                    chunk = uploaded_file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    bytes_written += len(chunk)
                    
                    # 진행률 업데이트
                    progress = min(bytes_written / file_size, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"업로드 중... {bytes_written / (1024*1024):.1f}MB / {file_size / (1024*1024):.1f}MB")
            
            # 완료
            progress_bar.progress(1.0)
            status_text.text("✅ 스트리밍 업로드 완료!")
            
            return temp_path
            
        except Exception as e:
            st.error(f"❌ 스트리밍 업로드 실패: {e}")
            return None
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            st.warning(f"임시 파일 정리 중 오류: {e}")
    
    def get_file_info(self, file_path: Path) -> dict:
        """파일 정보 반환"""
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "size_mb": stat.st_size / (1024*1024),
            "size_gb": stat.st_size / (1024*1024*1024),
            "path": str(file_path)
        }

def handle_large_file_upload():
    """대용량 파일 업로드 핸들러 (Streamlit 위젯 대체)"""
    
    st.markdown("### 🚀 대용량 파일 스트리밍 업로드")
    st.markdown("**5GB까지 지원 - 메모리 안전 보장**")
    
    # 스트리밍 업로더 초기화
    uploader = StreamingUploader()
    
    # 파일 업로드 위젯 (작은 파일용)
    uploaded_files = st.file_uploader(
        "파일 선택 (대용량 파일은 자동으로 스트리밍 처리됩니다)",
        accept_multiple_files=True,
        type=['jpg', 'jpeg', 'png', 'bmp', 'wav', 'mp3', 'm4a', 'flac', 'mp4', 'avi', 'mov', 'txt']
    )
    
    processed_files = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_size_gb = uploaded_file.size / (1024*1024*1024)
            
            if file_size_gb > 1.0:  # 1GB 이상은 스트리밍 처리
                st.warning(f"🔄 대용량 파일 감지: {uploaded_file.name} ({file_size_gb:.2f}GB)")
                st.info("스트리밍 모드로 안전하게 처리합니다...")
                
                # 스트리밍 처리
                temp_path = uploader.stream_to_temp(uploaded_file)
                if temp_path:
                    file_info = uploader.get_file_info(temp_path)
                    processed_files.append({
                        "name": uploaded_file.name,
                        "temp_path": temp_path,
                        "size": uploaded_file.size,
                        "streaming": True
                    })
            else:
                # 일반 처리 (1GB 미만)
                processed_files.append({
                    "name": uploaded_file.name,
                    "file_obj": uploaded_file,
                    "size": uploaded_file.size,
                    "streaming": False
                })
    
    return processed_files, uploader

if __name__ == "__main__":
    st.title("🚀 스트리밍 업로더 테스트")
    files, uploader = handle_large_file_upload()
    
    if files:
        st.write("처리된 파일:")
        for file_info in files:
            streaming_text = "스트리밍" if file_info["streaming"] else "일반"
            st.write(f"- {file_info['name']} ({streaming_text})")