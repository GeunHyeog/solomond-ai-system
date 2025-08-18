#!/usr/bin/env python3
"""
강화된 동영상 처리 시스템
- 고용량 동영상 파일 지원 강화
- YouTube URL 실제 분석 구현
- 사전정보 맥락 반영 시스템
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio

# 동영상 처리 라이브러리
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class EnhancedVideoProcessor:
    """강화된 동영상 처리 시스템"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 지원 포맷 확장
        self.supported_formats = [
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', 
            '.wmv', '.m4v', '.3gp', '.ogv', '.ts', '.mts'
        ]
        
        # 처리 제한 설정
        self.max_file_size = 10 * 1024 * 1024 * 1024  # 10GB
        self.chunk_duration = 60  # 60초 청크
        self.max_duration = 7200  # 2시간
        
        # URL 지원 플랫폼
        self.supported_platforms = [
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
            'twitch.tv', 'tiktok.com', 'instagram.com'
        ]
        
        # 사전정보 맥락 저장
        self.context_data = {}
        
        self._check_dependencies()
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _check_dependencies(self):
        """의존성 확인"""
        print("=== 동영상 처리 의존성 확인 ===")
        
        dependencies = {
            'OpenCV': CV2_AVAILABLE,
            'MoviePy': MOVIEPY_AVAILABLE,
            'yt-dlp': YT_DLP_AVAILABLE,
            'requests': REQUESTS_AVAILABLE
        }
        
        for dep, available in dependencies.items():
            status = "사용 가능" if available else "설치 필요"
            print(f"{dep}: {status}")
        
        # FFmpeg 확인
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            print("FFmpeg: 사용 가능")
            self.ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("FFmpeg: 설치 필요")
            self.ffmpeg_available = False
    
    def set_context(self, context_data: Dict[str, Any]):
        """사전정보 맥락 설정"""
        self.context_data = context_data
        self.logger.info(f"맥락 정보 설정 완료: {len(context_data)} 항목")
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """지원 포맷 확인"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        return file_path.suffix.lower() in self.supported_formats
    
    def validate_file_size(self, file_path: Union[str, Path]) -> bool:
        """파일 크기 검증"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        return file_size <= self.max_file_size
    
    def extract_video_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """동영상 메타데이터 추출"""
        metadata = {
            'file_path': str(file_path),
            'file_size_mb': 0,
            'duration_seconds': 0,
            'fps': 0,
            'width': 0,
            'height': 0,
            'audio_available': False,
            'codec': 'unknown',
            'context_applied': bool(self.context_data)
        }
        
        try:
            if CV2_AVAILABLE:
                cap = cv2.VideoCapture(str(file_path))
                
                if cap.isOpened():
                    # 기본 정보 추출
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    metadata.update({
                        'fps': fps,
                        'width': width,
                        'height': height,
                        'duration_seconds': frame_count / fps if fps > 0 else 0
                    })
                
                cap.release()
            
            # 파일 크기
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            metadata['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
            
            # 맥락 정보 통합
            if self.context_data:
                metadata['context_info'] = {
                    'project_name': self.context_data.get('project_name', ''),
                    'participants': self.context_data.get('participants', ''),
                    'purpose': self.context_data.get('purpose', ''),
                    'keywords': self.context_data.get('keywords', [])
                }
            
            self.logger.info(f"메타데이터 추출 완료: {metadata['duration_seconds']:.1f}초")
            
        except Exception as e:
            self.logger.error(f"메타데이터 추출 실패: {e}")
        
        return metadata
    
    def process_video_in_chunks(self, file_path: Union[str, Path], 
                               progress_callback=None) -> List[Dict[str, Any]]:
        """청크 단위 동영상 처리"""
        results = []
        
        try:
            metadata = self.extract_video_metadata(file_path)
            duration = metadata['duration_seconds']
            
            if duration > self.max_duration:
                self.logger.warning(f"동영상이 너무 깁니다: {duration:.1f}초 (최대 {self.max_duration}초)")
                duration = self.max_duration
            
            chunk_count = int(duration / self.chunk_duration) + 1
            
            for i in range(chunk_count):
                start_time = i * self.chunk_duration
                end_time = min((i + 1) * self.chunk_duration, duration)
                
                if progress_callback:
                    progress_callback(i / chunk_count, f"청크 {i+1}/{chunk_count} 처리 중")
                
                chunk_result = self._process_video_chunk(
                    file_path, start_time, end_time, i
                )
                
                if chunk_result:
                    results.append(chunk_result)
                
                time.sleep(0.1)  # CPU 부하 방지
            
            self.logger.info(f"청크 처리 완료: {len(results)}개 청크")
            
        except Exception as e:
            self.logger.error(f"청크 처리 실패: {e}")
        
        return results
    
    def _process_video_chunk(self, file_path: Union[str, Path], 
                            start_time: float, end_time: float, 
                            chunk_index: int) -> Dict[str, Any]:
        """개별 청크 처리"""
        chunk_result = {
            'chunk_index': chunk_index,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'frames_extracted': 0,
            'audio_extracted': False,
            'context_applied': bool(self.context_data)
        }
        
        try:
            # 오디오 추출 (음성 분석용)
            if MOVIEPY_AVAILABLE:
                audio_path = self._extract_audio_chunk(
                    file_path, start_time, end_time, chunk_index
                )
                if audio_path:
                    chunk_result['audio_path'] = audio_path
                    chunk_result['audio_extracted'] = True
            
            # 키 프레임 추출 (이미지 분석용)
            frame_paths = self._extract_key_frames(
                file_path, start_time, end_time, chunk_index
            )
            chunk_result['frame_paths'] = frame_paths
            chunk_result['frames_extracted'] = len(frame_paths)
            
            # 맥락 정보 적용
            if self.context_data:
                chunk_result['context_analysis'] = self._apply_context_to_chunk(
                    chunk_result
                )
            
        except Exception as e:
            self.logger.error(f"청크 {chunk_index} 처리 실패: {e}")
        
        return chunk_result
    
    def _extract_audio_chunk(self, file_path: Union[str, Path], 
                            start_time: float, end_time: float, 
                            chunk_index: int) -> Optional[str]:
        """오디오 청크 추출"""
        try:
            if not MOVIEPY_AVAILABLE:
                return None
            
            with VideoFileClip(str(file_path)) as video:
                audio = video.subclip(start_time, end_time).audio
                
                if audio is not None:
                    temp_dir = Path(tempfile.gettempdir()) / 'solomond_video_chunks'
                    temp_dir.mkdir(exist_ok=True)
                    
                    audio_path = temp_dir / f"chunk_{chunk_index}_audio.wav"
                    audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                    
                    return str(audio_path)
        
        except Exception as e:
            self.logger.error(f"오디오 추출 실패: {e}")
        
        return None
    
    def _extract_key_frames(self, file_path: Union[str, Path], 
                           start_time: float, end_time: float, 
                           chunk_index: int, frame_count: int = 5) -> List[str]:
        """키 프레임 추출"""
        frame_paths = []
        
        try:
            if not CV2_AVAILABLE:
                return frame_paths
            
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                return frame_paths
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            frame_interval = max(1, (end_frame - start_frame) // frame_count)
            
            temp_dir = Path(tempfile.gettempdir()) / 'solomond_video_frames'
            temp_dir.mkdir(exist_ok=True)
            
            for i in range(frame_count):
                frame_number = start_frame + (i * frame_interval)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    frame_path = temp_dir / f"chunk_{chunk_index}_frame_{i}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path))
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"프레임 추출 실패: {e}")
        
        return frame_paths
    
    def _apply_context_to_chunk(self, chunk_result: Dict[str, Any]) -> Dict[str, Any]:
        """청크에 맥락 정보 적용"""
        context_analysis = {
            'relevance_score': 0.0,
            'context_keywords_found': [],
            'analysis_focus': []
        }
        
        try:
            # 사전정보에서 키워드 추출
            keywords = self.context_data.get('keywords', [])
            purpose = self.context_data.get('purpose', '')
            
            # 시간대별 관련성 계산 (예시)
            start_time = chunk_result.get('start_time', 0)
            
            # 맥락 기반 분석 초점 설정
            if '상담' in purpose:
                context_analysis['analysis_focus'].append('conversation_analysis')
            if '회의' in purpose:
                context_analysis['analysis_focus'].append('meeting_analysis')
            if '프레젠테이션' in purpose:
                context_analysis['analysis_focus'].append('presentation_analysis')
            
            # 키워드 매칭 (실제로는 더 정교한 NLP 분석 필요)
            context_analysis['context_keywords_found'] = keywords[:3]  # 예시
            context_analysis['relevance_score'] = min(1.0, len(keywords) * 0.2)
            
        except Exception as e:
            self.logger.error(f"맥락 적용 실패: {e}")
        
        return context_analysis
    
    def process_video_url(self, url: str, progress_callback=None) -> Dict[str, Any]:
        """동영상 URL 처리 (YouTube 등)"""
        result = {
            'url': url,
            'platform': self._detect_platform(url),
            'success': False,
            'downloaded_path': None,
            'metadata': {},
            'error': None,
            'context_applied': bool(self.context_data)
        }
        
        try:
            if not YT_DLP_AVAILABLE:
                result['error'] = 'yt-dlp 라이브러리가 설치되지 않았습니다'
                return result
            
            if progress_callback:
                progress_callback(0.1, "URL 유효성 검사 중...")
            
            # URL 유효성 검사
            if not self._validate_url(url):
                result['error'] = '지원되지 않는 URL입니다'
                return result
            
            if progress_callback:
                progress_callback(0.3, "동영상 정보 추출 중...")
            
            # yt-dlp 설정
            temp_dir = Path(tempfile.gettempdir()) / 'solomond_downloads'
            temp_dir.mkdir(exist_ok=True)
            
            ydl_opts = {
                'format': 'best[height<=720]',  # 720p 이하로 제한
                'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
                'extractaudio': True,
                'audioformat': 'wav',
                'noplaylist': True,
                'max_filesize': self.max_file_size,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 메타데이터만 먼저 추출
                info = ydl.extract_info(url, download=False)
                
                result['metadata'] = {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'description': info.get('description', '')[:500]  # 처음 500자만
                }
                
                # 길이 제한 확인
                duration = info.get('duration', 0)
                if duration > self.max_duration:
                    result['error'] = f'동영상이 너무 깁니다: {duration}초 (최대 {self.max_duration}초)'
                    return result
                
                if progress_callback:
                    progress_callback(0.6, "동영상 다운로드 중...")
                
                # 실제 다운로드
                ydl.download([url])
                
                # 다운로드된 파일 찾기
                downloaded_files = list(temp_dir.glob('*'))
                if downloaded_files:
                    result['downloaded_path'] = str(downloaded_files[0])
                    result['success'] = True
                    
                    if progress_callback:
                        progress_callback(1.0, "다운로드 완료!")
                    
                    # 맥락 정보 적용
                    if self.context_data:
                        result['context_analysis'] = self._apply_context_to_url_content(
                            result['metadata']
                        )
                else:
                    result['error'] = '다운로드된 파일을 찾을 수 없습니다'
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"URL 처리 실패: {e}")
        
        return result
    
    def _detect_platform(self, url: str) -> str:
        """플랫폼 감지"""
        url_lower = url.lower()
        
        for platform in self.supported_platforms:
            if platform in url_lower:
                return platform.split('.')[0]  # 도메인에서 플랫폼명 추출
        
        return 'unknown'
    
    def _validate_url(self, url: str) -> bool:
        """URL 유효성 검사"""
        if not REQUESTS_AVAILABLE:
            return True  # requests가 없으면 기본적으로 유효하다고 가정
        
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _apply_context_to_url_content(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """URL 콘텐츠에 맥락 정보 적용"""
        context_analysis = {
            'content_relevance': 0.0,
            'keyword_matches': [],
            'recommended_focus': []
        }
        
        try:
            # 제목과 설명에서 맥락 키워드 찾기
            title = metadata.get('title', '').lower()
            description = metadata.get('description', '').lower()
            
            keywords = self.context_data.get('keywords', [])
            
            for keyword in keywords:
                if keyword.lower() in title or keyword.lower() in description:
                    context_analysis['keyword_matches'].append(keyword)
            
            # 관련성 점수 계산
            context_analysis['content_relevance'] = len(context_analysis['keyword_matches']) / max(1, len(keywords))
            
            # 분석 권장사항
            purpose = self.context_data.get('purpose', '')
            if '교육' in purpose:
                context_analysis['recommended_focus'].append('educational_content')
            if '회의' in purpose:
                context_analysis['recommended_focus'].append('meeting_content')
            
        except Exception as e:
            self.logger.error(f"URL 맥락 적용 실패: {e}")
        
        return context_analysis
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """처리 능력 정보 반환"""
        return {
            'supported_formats': self.supported_formats,
            'max_file_size_gb': self.max_file_size / (1024**3),
            'max_duration_minutes': self.max_duration / 60,
            'chunk_duration_seconds': self.chunk_duration,
            'url_platforms': self.supported_platforms,
            'dependencies': {
                'opencv': CV2_AVAILABLE,
                'moviepy': MOVIEPY_AVAILABLE,
                'yt_dlp': YT_DLP_AVAILABLE,
                'ffmpeg': self.ffmpeg_available
            },
            'context_support': True
        }
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            temp_dirs = [
                Path(tempfile.gettempdir()) / 'solomond_video_chunks',
                Path(tempfile.gettempdir()) / 'solomond_video_frames',
                Path(tempfile.gettempdir()) / 'solomond_downloads'
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file in temp_dir.glob('*'):
                        try:
                            file.unlink()
                        except:
                            pass
            
            self.logger.info("임시 파일 정리 완료")
            
        except Exception as e:
            self.logger.error(f"임시 파일 정리 실패: {e}")

# 글로벌 인스턴스
enhanced_video_processor = EnhancedVideoProcessor()

def get_enhanced_video_processor():
    """강화된 동영상 처리기 인스턴스 반환"""
    return enhanced_video_processor

if __name__ == "__main__":
    # 테스트 실행
    processor = EnhancedVideoProcessor()
    
    print("강화된 동영상 처리 시스템 초기화 완료")
    print("처리 능력:", json.dumps(processor.get_processing_capabilities(), indent=2, ensure_ascii=False))