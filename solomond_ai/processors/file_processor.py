"""
파일 처리기
업로드된 파일들의 형식 검증, 변환, 정리를 담당
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import shutil
import tempfile

class FileProcessor:
    """파일 처리 및 관리 클래스"""
    
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(exist_ok=True)
        
        self.supported_formats = {
            'audio': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'], 
            'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'],
            'text': ['.txt', '.md', '.json', '.csv', '.log', '.py', '.js', '.html', '.xml', '.pdf', '.docx']
        }
        
    def organize_files_by_type(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """파일들을 타입별로 분류"""
        organized = {
            'audio': [],
            'image': [], 
            'video': [],
            'text': [],
            'unknown': []
        }
        
        for file_path in file_paths:
            file_type = self.detect_file_type(file_path)
            organized[file_type].append(file_path)
        
        return organized
    
    def detect_file_type(self, file_path: str) -> str:
        """파일 타입 감지"""
        file_ext = Path(file_path).suffix.lower()
        
        for file_type, extensions in self.supported_formats.items():
            if file_ext in extensions:
                return file_type
        
        return 'unknown'
    
    def validate_files(self, file_paths: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
        """파일 유효성 검증"""
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            try:
                path_obj = Path(file_path)
                
                # 파일 존재 확인
                if not path_obj.exists():
                    invalid_files.append({
                        'file_path': file_path,
                        'error': 'File does not exist'
                    })
                    continue
                
                # 파일 크기 확인 (10GB 제한)
                file_size = path_obj.stat().st_size
                if file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                    invalid_files.append({
                        'file_path': file_path,
                        'error': 'File too large (>10GB)'
                    })
                    continue
                
                # 파일 형식 확인
                if self.detect_file_type(file_path) == 'unknown':
                    invalid_files.append({
                        'file_path': file_path,
                        'error': 'Unsupported file format'
                    })
                    continue
                
                # 파일 읽기 권한 확인
                if not os.access(file_path, os.R_OK):
                    invalid_files.append({
                        'file_path': file_path,
                        'error': 'No read permission'
                    })
                    continue
                
                valid_files.append(file_path)
                
            except Exception as e:
                invalid_files.append({
                    'file_path': file_path,
                    'error': f'Validation error: {str(e)}'
                })
        
        return valid_files, invalid_files
    
    def copy_files_to_workspace(self, file_paths: List[str]) -> Dict[str, str]:
        """파일들을 작업 공간으로 복사"""
        copied_files = {}
        
        for file_path in file_paths:
            try:
                source_path = Path(file_path)
                # 작업 공간에 고유한 파일명 생성
                timestamp = int(time.time() * 1000)
                dest_filename = f"{timestamp}_{source_path.name}"
                dest_path = self.work_dir / dest_filename
                
                shutil.copy2(source_path, dest_path)
                copied_files[file_path] = str(dest_path)
                
                logging.info(f"Copied {file_path} to {dest_path}")
                
            except Exception as e:
                logging.error(f"Failed to copy {file_path}: {e}")
                copied_files[file_path] = None
        
        return copied_files
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일 상세 정보 추출"""
        try:
            path_obj = Path(file_path)
            stat_info = path_obj.stat()
            
            return {
                'file_name': path_obj.name,
                'file_path': str(path_obj.absolute()),
                'file_size': stat_info.st_size,
                'file_size_mb': round(stat_info.st_size / (1024 * 1024), 2),
                'file_type': self.detect_file_type(file_path),
                'file_extension': path_obj.suffix.lower(),
                'created_time': stat_info.st_ctime,
                'modified_time': stat_info.st_mtime,
                'is_readable': os.access(file_path, os.R_OK)
            }
            
        except Exception as e:
            return {
                'file_name': Path(file_path).name,
                'file_path': file_path,
                'error': str(e)
            }
    
    def convert_audio_format(self, input_path: str, output_format: str = 'wav') -> Optional[str]:
        """오디오 형식 변환 (FFmpeg 필요)"""
        try:
            import subprocess
            
            input_path_obj = Path(input_path)
            output_path = self.work_dir / f"{input_path_obj.stem}.{output_format}"
            
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz 샘플링 (Whisper 권장)
                '-ac', '1',      # 모노
                str(output_path),
                '-y'  # 덮어쓰기
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Audio converted: {input_path} -> {output_path}")
                return str(output_path)
            else:
                logging.error(f"Audio conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Audio conversion error: {e}")
            return None
    
    def cleanup_workspace(self):
        """작업 공간 정리"""
        try:
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
                logging.info(f"Cleaned up workspace: {self.work_dir}")
        except Exception as e:
            logging.error(f"Failed to cleanup workspace: {e}")
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """작업 공간 정보"""
        try:
            files = list(self.work_dir.glob('*'))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                'workspace_path': str(self.work_dir),
                'file_count': len(files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'files': [f.name for f in files]
            }
        except Exception as e:
            return {
                'workspace_path': str(self.work_dir),
                'error': str(e)
            }