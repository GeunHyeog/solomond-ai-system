#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 파일 핸들러 - 업로드 문제 완전 해결
Enhanced File Handler for Upload Issues Resolution
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import mimetypes
import hashlib
import time
import os

class EnhancedFileHandler:
    """향상된 파일 처리기"""
    
    def __init__(self, max_size_mb: int = 5000):
        self.max_size_mb = max_size_mb
        self.temp_dir = Path(tempfile.gettempdir()) / "solomond_uploads"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 지원 파일 형식
        self.supported_types = {
            'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            'audio': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'],
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv'],
            'text': ['.txt', '.md', '.csv', '.json', '.xml'],
            'document': ['.pdf', '.docx', '.doc', '.pptx', '.ppt']
        }
    
    def safe_upload_interface(self) -> List[Any]:
        """안전한 파일 업로드 인터페이스"""
        
        st.markdown("### 📂 안전한 파일 업로드")
        
        # 업로드 타입 선택
        upload_mode = st.radio(
            "업로드 방식 선택:",
            ["개별 파일 업로드", "로컬 폴더 선택", "테스트 파일 사용"],
            horizontal=True
        )
        
        uploaded_files = []
        
        if upload_mode == "개별 파일 업로드":
            uploaded_files = self._handle_direct_upload()
        elif upload_mode == "로컬 폴더 선택":
            uploaded_files = self._handle_local_folder()
        elif upload_mode == "테스트 파일 사용":
            uploaded_files = self._handle_test_files()
        
        return uploaded_files
    
    def _handle_direct_upload(self) -> List[Any]:
        """직접 업로드 처리"""
        
        st.markdown("**지원 형식**: 이미지, 음성, 비디오, 텍스트 파일")
        
        # 파일 업로더
        uploaded_files = st.file_uploader(
            "파일을 선택하세요 (최대 5GB)",
            accept_multiple_files=True,
            type=self._get_all_extensions(),
            help="대용량 파일도 안전하게 처리됩니다",
            key="enhanced_uploader"
        )
        
        if uploaded_files:
            # 파일 정보 표시
            total_size = sum(f.size for f in uploaded_files)
            size_mb = total_size / (1024 * 1024)
            
            st.success(f"✅ {len(uploaded_files)}개 파일 업로드됨 ({size_mb:.1f}MB)")
            
            # 파일별 상세 정보
            with st.expander("📋 업로드된 파일 정보"):
                for i, file in enumerate(uploaded_files):
                    file_size_mb = file.size / (1024 * 1024)
                    file_type = self._get_file_type(file.name)
                    st.write(f"{i+1}. **{file.name}** ({file_size_mb:.1f}MB) - {file_type}")
        
        return uploaded_files or []
    
    def _handle_local_folder(self) -> List[Any]:
        """로컬 폴더 처리"""
        
        user_files_dir = Path("user_files")
        
        if not user_files_dir.exists():
            st.warning("user_files 폴더가 없습니다.")
            return []
        
        # 폴더 목록 가져오기
        folders = [f for f in user_files_dir.iterdir() if f.is_dir()]
        
        if not folders:
            st.warning("user_files에 폴더가 없습니다.")
            return []
        
        # 폴더 선택
        folder_names = [f.name for f in folders]
        selected_folder_name = st.selectbox(
            "📁 분석할 폴더 선택:",
            folder_names,
            key="folder_selector"
        )
        
        if selected_folder_name:
            selected_folder = user_files_dir / selected_folder_name
            
            # 폴더 내 파일 스캔
            files = self._scan_folder_files(selected_folder)
            
            if files:
                st.success(f"✅ {len(files)}개 파일 발견")
                
                # 파일 정보 표시
                total_size_mb = sum(f['size_mb'] for f in files)
                st.info(f"📊 총 크기: {total_size_mb:.1f}MB ({total_size_mb/1024:.2f}GB)")
                
                # 파일 목록
                with st.expander("📋 발견된 파일 목록"):
                    for file_info in files:
                        icon = self._get_file_icon(file_info['type'])
                        st.write(f"{icon} **{file_info['name']}** ({file_info['size_mb']:.1f}MB)")
                
                # 가상 업로드 객체 생성
                return self._create_virtual_upload_objects(files)
            else:
                st.warning("지원되는 파일이 없습니다.")
        
        return []
    
    def _handle_test_files(self) -> List[Any]:
        """테스트 파일 처리"""
        
        test_files_dir = Path("test_files")
        
        if not test_files_dir.exists():
            # 테스트 파일 생성
            st.info("테스트 파일을 생성합니다...")
            self._create_test_files()
        
        if test_files_dir.exists():
            files = self._scan_folder_files(test_files_dir)
            
            if files:
                st.success(f"✅ {len(files)}개 테스트 파일 준비됨")
                return self._create_virtual_upload_objects(files)
        
        st.error("테스트 파일 생성 실패")
        return []
    
    def _scan_folder_files(self, folder_path: Path) -> List[Dict]:
        """폴더 내 파일 스캔"""
        
        files = []
        all_extensions = self._get_all_extensions()
        
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in all_extensions or ext[1:] in all_extensions:  # .제거 버전도 체크
                    try:
                        size = file_path.stat().st_size
                        files.append({
                            'path': file_path,
                            'name': file_path.name,
                            'size_mb': size / (1024 * 1024),
                            'type': ext,
                            'full_path': str(file_path)
                        })
                    except (OSError, PermissionError):
                        continue
        
        # 크기 순 정렬 (큰 파일 먼저)
        files.sort(key=lambda x: x['size_mb'], reverse=True)
        return files
    
    def _create_virtual_upload_objects(self, files: List[Dict]) -> List[Any]:
        """가상 업로드 객체 생성"""
        
        virtual_objects = []
        
        for file_info in files:
            # Streamlit 업로드 파일과 호환되는 객체 생성
            class VirtualUploadedFile:
                def __init__(self, file_info):
                    self.name = file_info['name']
                    self.size = int(file_info['size_mb'] * 1024 * 1024)
                    self.type = mimetypes.guess_type(file_info['name'])[0] or 'application/octet-stream'
                    self._file_path = file_info['path']
                
                def read(self):
                    with open(self._file_path, 'rb') as f:
                        return f.read()
                
                def getvalue(self):
                    return self.read()
                
                def seek(self, pos):
                    pass  # 가상 구현
                
                def tell(self):
                    return 0  # 가상 구현
            
            virtual_objects.append(VirtualUploadedFile(file_info))
        
        return virtual_objects
    
    def _create_test_files(self):
        """테스트 파일 생성"""
        
        test_dir = Path("test_files")
        test_dir.mkdir(exist_ok=True)
        
        # 텍스트 파일
        (test_dir / "sample_text.txt").write_text(
            "이것은 테스트용 텍스트 파일입니다.\n컨퍼런스 분석 테스트를 위한 샘플 내용입니다.",
            encoding='utf-8'
        )
        
        # 가짜 이미지 (1x1 PNG)
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00\x007n\xf9$\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01r\xdd\xe4\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
        (test_dir / "test_image.png").write_bytes(png_data)
        
        # JSON 설정 파일
        import json
        config = {
            "conference_name": "테스트 컨퍼런스",
            "date": "2025-08-20",
            "participants": ["발표자1", "발표자2"],
            "topics": ["AI", "분석", "테스트"]
        }
        (test_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    
    def _get_all_extensions(self) -> List[str]:
        """지원하는 모든 확장자 리스트"""
        
        all_exts = []
        for type_exts in self.supported_types.values():
            all_exts.extend(type_exts)
        
        # .없는 버전도 추가 (Streamlit 호환)
        return all_exts + [ext[1:] for ext in all_exts]
    
    def _get_file_type(self, filename: str) -> str:
        """파일 타입 확인"""
        
        ext = Path(filename).suffix.lower()
        
        for type_name, extensions in self.supported_types.items():
            if ext in extensions:
                return type_name
        
        return "unknown"
    
    def _get_file_icon(self, file_type: str) -> str:
        """파일 타입별 아이콘"""
        
        icons = {
            '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️',
            '.wav': '🎵', '.mp3': '🎵', '.m4a': '🎵',
            '.mp4': '🎬', '.mov': '🎬', '.avi': '🎬',
            '.txt': '📄', '.md': '📄', '.json': '📄'
        }
        
        return icons.get(file_type, '📎')
    
    def save_uploaded_files(self, uploaded_files: List[Any]) -> List[str]:
        """업로드된 파일들을 안전하게 저장"""
        
        saved_paths = []
        
        for uploaded_file in uploaded_files:
            try:
                # 안전한 파일명 생성
                safe_name = self._make_safe_filename(uploaded_file.name)
                temp_path = self.temp_dir / safe_name
                
                # 파일 저장
                with open(temp_path, 'wb') as f:
                    if hasattr(uploaded_file, '_file_path'):
                        # 로컬 파일인 경우
                        with open(uploaded_file._file_path, 'rb') as src:
                            shutil.copyfileobj(src, f)
                    else:
                        # 업로드된 파일인 경우
                        f.write(uploaded_file.getvalue())
                
                saved_paths.append(str(temp_path))
                
            except Exception as e:
                st.error(f"파일 저장 실패: {uploaded_file.name} - {e}")
        
        return saved_paths
    
    def _make_safe_filename(self, filename: str) -> str:
        """안전한 파일명 생성"""
        
        # 타임스탬프 추가로 중복 방지
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        
        # 안전한 문자만 유지
        safe_name = "".join(c for c in name if c.isalnum() or c in "._-")
        
        return f"{safe_name}_{timestamp}{ext}"

# 전역 인스턴스
enhanced_handler = EnhancedFileHandler()

def get_enhanced_file_upload() -> List[Any]:
    """향상된 파일 업로드 인터페이스 (전역 함수)"""
    return enhanced_handler.safe_upload_interface()

if __name__ == "__main__":
    # 테스트 코드
    print("Enhanced File Handler 테스트")
    handler = EnhancedFileHandler()
    
    # 테스트 파일 생성
    handler._create_test_files()
    print("테스트 파일 생성 완료")
    
    # 로컬 파일 스캔 테스트
    user_files = Path("user_files")
    if user_files.exists():
        for folder in user_files.iterdir():
            if folder.is_dir():
                files = handler._scan_folder_files(folder)
                print(f"{folder.name}: {len(files)}개 파일")