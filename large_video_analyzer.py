#!/usr/bin/env python3
"""
대용량 비디오 파일 분석기
- 3.24GB MOV 파일 최적화 처리
- 메타데이터 추출 및 샘플 분석
- 컨퍼런스 관련성 평가
"""

import os
import sys
import time
import json
import cv2
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class LargeVideoAnalyzer:
    """대용량 비디오 분석기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"large_video_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'analysis_approach': 'metadata_and_sampling',
            'target_files': []
        }
        
        print("대용량 비디오 분석기 초기화")
        self._check_dependencies()
    
    def _check_dependencies(self):
        """의존성 확인"""
        print("\n--- 의존성 확인 ---")
        
        # OpenCV 확인
        try:
            print(f"OpenCV 버전: {cv2.__version__}")
            self.opencv_available = True
        except Exception as e:
            print(f"[WARNING] OpenCV 문제: {e}")
            self.opencv_available = False
        
        # FFmpeg 확인
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[OK] FFmpeg 사용 가능")
                self.ffmpeg_available = True
            else:
                print("[WARNING] FFmpeg 실행 문제")
                self.ffmpeg_available = False
        except Exception as e:
            print(f"[WARNING] FFmpeg 확인 실패: {e}")
            self.ffmpeg_available = False
    
    def find_video_files(self) -> List[Dict[str, Any]]:
        """비디오 파일 탐색"""
        print("\n--- 비디오 파일 탐색 ---")
        
        video_path = project_root / 'user_files' / 'videos'
        
        if not video_path.exists():
            print(f"[ERROR] 비디오 폴더 없음: {video_path}")
            return []
        
        video_files = []
        for video_file in video_path.glob('*.MOV'):
            file_size_gb = video_file.stat().st_size / (1024**3)
            file_info = {
                'file_path': str(video_file),
                'file_name': video_file.name,
                'file_size_gb': file_size_gb,
                'file_size_bytes': video_file.stat().st_size,
                'is_large_file': file_size_gb > 1.0,  # 1GB 이상
                'priority': 'high' if file_size_gb > 2.0 else 'medium'
            }
            video_files.append(file_info)
        
        # 크기순 정렬
        video_files.sort(key=lambda x: x['file_size_gb'], reverse=True)
        
        self.analysis_session['target_files'] = video_files
        
        print(f"[OK] 비디오 파일 발견: {len(video_files)}개")
        for i, file_info in enumerate(video_files, 1):
            print(f"  {i}. {file_info['file_name']} ({file_info['file_size_gb']:.2f}GB)")
        
        return video_files
    
    def analyze_video_metadata(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """비디오 메타데이터 분석 (빠른 처리)"""
        print(f"\n--- 비디오 메타데이터 분석 ---")
        print(f"파일: {file_info['file_name']} ({file_info['file_size_gb']:.2f}GB)")
        
        analysis_start = time.time()
        
        try:
            # 1. OpenCV로 기본 정보 추출
            metadata = self._extract_basic_metadata(file_info['file_path'])
            
            # 2. FFmpeg로 상세 정보 추출 (가능한 경우)
            if self.ffmpeg_available:
                ffmpeg_info = self._extract_ffmpeg_metadata(file_info['file_path'])
                metadata.update(ffmpeg_info)
            
            # 3. 컨퍼런스 관련성 평가
            conference_assessment = self._assess_video_conference_relevance(metadata, file_info)
            
            processing_time = time.time() - analysis_start
            
            result = {
                'file_info': file_info,
                'video_metadata': metadata,
                'conference_assessment': conference_assessment,
                'processing_info': {
                    'analysis_method': 'metadata_extraction',
                    'processing_time': processing_time,
                    'tools_used': ['opencv'] + (['ffmpeg'] if self.ffmpeg_available else [])
                },
                'status': 'success'
            }
            
            print(f"[OK] 메타데이터 분석 완료 ({processing_time:.1f}초)")
            print(f"비디오 길이: {metadata.get('duration_minutes', 0):.1f}분")
            print(f"해상도: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
            
            return result
            
        except Exception as e:
            error_result = {
                'file_info': file_info,
                'error': str(e),
                'processing_time': time.time() - analysis_start,
                'status': 'error'
            }
            
            print(f"[ERROR] 메타데이터 분석 실패: {e}")
            return error_result
    
    def _extract_basic_metadata(self, video_path: str) -> Dict[str, Any]:
        """OpenCV로 기본 메타데이터 추출"""
        metadata = {}
        
        if not self.opencv_available:
            return {'error': 'OpenCV 사용 불가'}
        
        cap = cv2.VideoCapture(video_path)
        
        try:
            # 기본 속성
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 길이 계산
            if metadata['fps'] > 0:
                metadata['duration_seconds'] = metadata['total_frames'] / metadata['fps']
                metadata['duration_minutes'] = metadata['duration_seconds'] / 60
            else:
                metadata['duration_seconds'] = 0
                metadata['duration_minutes'] = 0
            
            # 해상도 품질 평가
            total_pixels = metadata['width'] * metadata['height']
            if total_pixels >= 1920 * 1080:
                metadata['quality'] = 'HD'
            elif total_pixels >= 1280 * 720:
                metadata['quality'] = 'SD'
            else:
                metadata['quality'] = 'Low'
            
            print(f"  OpenCV 정보: {metadata['width']}x{metadata['height']}, {metadata['fps']:.1f}fps, {metadata['duration_minutes']:.1f}분")
            
        except Exception as e:
            metadata['opencv_error'] = str(e)
        
        finally:
            cap.release()
        
        return metadata
    
    def _extract_ffmpeg_metadata(self, video_path: str) -> Dict[str, Any]:
        """FFmpeg로 상세 메타데이터 추출"""
        metadata = {}
        
        try:
            # FFprobe 명령어로 메타데이터 추출
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                ffmpeg_data = json.loads(result.stdout)
                
                # 포맷 정보
                if 'format' in ffmpeg_data:
                    format_info = ffmpeg_data['format']
                    metadata['format_name'] = format_info.get('format_name', '')
                    metadata['duration_ffmpeg'] = float(format_info.get('duration', 0))
                    metadata['bit_rate'] = int(format_info.get('bit_rate', 0))
                    metadata['size'] = int(format_info.get('size', 0))
                
                # 스트림 정보
                if 'streams' in ffmpeg_data:
                    for stream in ffmpeg_data['streams']:
                        if stream.get('codec_type') == 'video':
                            metadata['codec_name'] = stream.get('codec_name', '')
                            metadata['pixel_format'] = stream.get('pix_fmt', '')
                            metadata['color_space'] = stream.get('color_space', '')
                        elif stream.get('codec_type') == 'audio':
                            metadata['audio_codec'] = stream.get('codec_name', '')
                            metadata['audio_channels'] = stream.get('channels', 0)
                            metadata['audio_sample_rate'] = stream.get('sample_rate', 0)
                
                print(f"  FFmpeg 정보: 코덱={metadata.get('codec_name', 'Unknown')}, 비트레이트={metadata.get('bit_rate', 0)}")
                
            else:
                metadata['ffmpeg_error'] = result.stderr
        
        except Exception as e:
            metadata['ffmpeg_error'] = str(e)
        
        return metadata
    
    def _assess_video_conference_relevance(self, metadata: Dict[str, Any], file_info: Dict[str, Any]) -> Dict[str, Any]:
        """비디오 컨퍼런스 관련성 평가"""
        relevance_score = 0
        assessment_factors = []
        
        # 1. 파일 크기 평가 (대용량은 중요한 내용일 가능성 높음)
        if file_info['file_size_gb'] > 2.0:
            relevance_score += 30
            assessment_factors.append('대용량 파일 (중요 콘텐츠 가능성)')
        
        # 2. 길이 평가 (긴 영상은 컨퍼런스 가능성 높음)
        duration_minutes = metadata.get('duration_minutes', 0)
        if duration_minutes > 30:
            relevance_score += 25
            assessment_factors.append('장시간 영상 (컨퍼런스 형태)')
        elif duration_minutes > 10:
            relevance_score += 15
            assessment_factors.append('중간 길이 영상')
        
        # 3. 해상도 평가 (HD 영상은 전문적 촬영)
        quality = metadata.get('quality', '')
        if quality == 'HD':
            relevance_score += 20
            assessment_factors.append('HD 화질 (전문적 촬영)')
        elif quality == 'SD':
            relevance_score += 10
            assessment_factors.append('적정 화질')
        
        # 4. 파일명 패턴 평가
        file_name = file_info['file_name'].upper()
        if 'IMG_' in file_name:
            relevance_score += 15
            assessment_factors.append('표준 카메라 파일명 패턴')
        
        # 5. 오디오 정보 평가
        if metadata.get('audio_channels', 0) >= 2:
            relevance_score += 10
            assessment_factors.append('스테레오 오디오 (고품질 녹음)')
        
        # 관련성 레벨 결정
        if relevance_score >= 70:
            relevance_level = 'high'
        elif relevance_score >= 50:
            relevance_level = 'medium'
        else:
            relevance_level = 'low'
        
        return {
            'relevance_score': int(relevance_score),
            'relevance_level': relevance_level,
            'assessment_factors': assessment_factors,
            'is_likely_conference_video': bool(relevance_score >= 50),
            'recommended_analysis': 'sample_extraction' if relevance_score >= 60 else 'basic_info_only'
        }
    
    def extract_video_samples(self, file_info: Dict[str, Any], sample_count: int = 5) -> Dict[str, Any]:
        """비디오 샘플 프레임 추출"""
        print(f"\n--- 비디오 샘플 추출 ---")
        print(f"파일: {file_info['file_name']}")
        print(f"샘플 수: {sample_count}개")
        
        if not self.opencv_available:
            return {'error': 'OpenCV 사용 불가'}
        
        extraction_start = time.time()
        samples = []
        
        cap = cv2.VideoCapture(file_info['file_path'])
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                return {'error': '비디오 정보 읽기 실패'}
            
            # 샘플 추출 간격 계산
            sample_interval = max(1, total_frames // (sample_count + 1))
            
            for i in range(1, sample_count + 1):
                frame_number = i * sample_interval
                
                # 프레임 위치로 이동
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # 타임스탬프 계산
                    timestamp_seconds = frame_number / fps
                    timestamp_str = str(timedelta(seconds=int(timestamp_seconds)))
                    
                    # 프레임 분석
                    frame_analysis = self._analyze_frame(frame, i, timestamp_str)
                    
                    # 샘플 저장 경로
                    sample_filename = f"video_sample_{self.analysis_session['session_id']}_{i:02d}.jpg"
                    sample_path = project_root / sample_filename
                    
                    # 프레임 저장
                    cv2.imwrite(str(sample_path), frame)
                    
                    sample_info = {
                        'sample_id': i,
                        'frame_number': frame_number,
                        'timestamp': timestamp_str,
                        'sample_path': str(sample_path),
                        'frame_analysis': frame_analysis
                    }
                    samples.append(sample_info)
                    
                    print(f"  샘플 {i}: {timestamp_str} → {sample_filename}")
                
                else:
                    print(f"  [WARNING] 프레임 {frame_number} 읽기 실패")
        
        except Exception as e:
            return {'error': f'샘플 추출 실패: {str(e)}'}
        
        finally:
            cap.release()
        
        processing_time = time.time() - extraction_start
        
        result = {
            'total_samples_extracted': len(samples),
            'samples': samples,
            'extraction_info': {
                'total_frames': total_frames,
                'sample_interval': sample_interval,
                'processing_time': processing_time
            },
            'status': 'success'
        }
        
        print(f"[OK] 샘플 추출 완료: {len(samples)}개 ({processing_time:.1f}초)")
        
        return result
    
    def _analyze_frame(self, frame: np.ndarray, sample_id: int, timestamp: str) -> Dict[str, Any]:
        """개별 프레임 분석"""
        try:
            # 기본 정보
            height, width, channels = frame.shape
            
            # 색상 분석
            mean_color = np.mean(frame, axis=(0, 1))
            brightness = np.mean(mean_color)
            
            # 텍스트 영역 감지 (간단한 방법)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 가장자리 검출로 텍스트 영역 추정
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # 색상 분포 분석
            hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
            
            color_variance = np.var([np.var(hist_b), np.var(hist_g), np.var(hist_r)])
            
            # 프레젠테이션 슬라이드 가능성 평가
            is_likely_presentation = self._assess_presentation_likelihood(
                brightness, edge_density, color_variance, width, height
            )
            
            return {
                'dimensions': {'width': width, 'height': height},
                'brightness': float(brightness),
                'edge_density': float(edge_density),
                'color_variance': float(color_variance),
                'mean_color': [float(c) for c in mean_color],
                'presentation_likelihood': is_likely_presentation
            }
        
        except Exception as e:
            return {'error': f'프레임 분석 실패: {str(e)}'}
    
    def _assess_presentation_likelihood(self, brightness: float, edge_density: float, 
                                      color_variance: float, width: int, height: int) -> Dict[str, Any]:
        """프레젠테이션 슬라이드 가능성 평가"""
        likelihood_score = 0
        factors = []
        
        # 1. 밝기 평가 (프레젠테이션은 보통 밝음)
        if 100 < brightness < 200:
            likelihood_score += 25
            factors.append('적정 밝기 (프레젠테이션 적합)')
        
        # 2. 가장자리 밀도 (텍스트가 많으면 높음)
        if edge_density > 0.1:
            likelihood_score += 30
            factors.append('높은 텍스트 밀도')
        elif edge_density > 0.05:
            likelihood_score += 15
            factors.append('중간 텍스트 밀도')
        
        # 3. 화면 비율 (16:9 또는 4:3)
        aspect_ratio = width / height if height > 0 else 1
        if 1.7 < aspect_ratio < 1.8:  # 16:9
            likelihood_score += 20
            factors.append('16:9 프레젠테이션 비율')
        elif 1.3 < aspect_ratio < 1.4:  # 4:3
            likelihood_score += 15
            factors.append('4:3 프레젠테이션 비율')
        
        # 4. 색상 다양성 (프레젠테이션은 보통 단순함)
        if color_variance < 1000:
            likelihood_score += 15
            factors.append('단순한 색상 구성')
        
        # 가능성 레벨 결정
        if likelihood_score >= 70:
            likelihood_level = 'high'
        elif likelihood_score >= 40:
            likelihood_level = 'medium'
        else:
            likelihood_level = 'low'
        
        return {
            'likelihood_score': likelihood_score,
            'likelihood_level': likelihood_level,
            'assessment_factors': factors,
            'is_likely_presentation': likelihood_score >= 40
        }
    
    def generate_video_analysis_summary(self, metadata_result: Dict[str, Any], 
                                       samples_result: Dict[str, Any]) -> Dict[str, Any]:
        """비디오 분석 종합 요약"""
        print("\n--- 비디오 분석 종합 요약 생성 ---")
        
        if 'error' in metadata_result or 'error' in samples_result:
            return {'error': '분석 결과에 오류가 있습니다.'}
        
        file_info = metadata_result['file_info']
        metadata = metadata_result['video_metadata']
        conference_assessment = metadata_result['conference_assessment']
        samples = samples_result['samples']
        
        # 샘플 프레임 분석 통계
        presentation_frames = sum(1 for sample in samples 
                                if sample['frame_analysis'].get('presentation_likelihood', {}).get('is_likely_presentation', False))
        
        avg_brightness = np.mean([sample['frame_analysis'].get('brightness', 0) for sample in samples])
        avg_edge_density = np.mean([sample['frame_analysis'].get('edge_density', 0) for sample in samples])
        
        # 종합 평가
        overall_assessment = self._generate_overall_assessment(
            conference_assessment, presentation_frames, len(samples), metadata
        )
        
        summary = {
            'file_summary': {
                'name': file_info['file_name'],
                'size_gb': file_info['file_size_gb'],
                'duration_minutes': metadata.get('duration_minutes', 0),
                'resolution': f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                'quality': metadata.get('quality', 'Unknown'),
                'format': metadata.get('format_name', 'MOV')
            },
            'content_analysis': {
                'conference_relevance': conference_assessment['relevance_level'],
                'presentation_frames_ratio': f"{presentation_frames}/{len(samples)}",
                'average_brightness': round(avg_brightness, 1),
                'average_text_density': round(avg_edge_density, 3),
                'video_type_assessment': overall_assessment['video_type']
            },
            'technical_assessment': {
                'codec': metadata.get('codec_name', 'Unknown'),
                'bitrate_mbps': round(metadata.get('bit_rate', 0) / 1000000, 1) if metadata.get('bit_rate') else 0,
                'audio_quality': 'Stereo' if metadata.get('audio_channels', 0) >= 2 else 'Mono',
                'overall_quality': metadata.get('quality', 'Unknown')
            },
            'analysis_insights': overall_assessment['insights'],
            'recommended_actions': overall_assessment['recommendations'],
            'sample_extraction_summary': {
                'total_samples': len(samples),
                'presentation_samples': presentation_frames,
                'sample_files': [sample['sample_path'] for sample in samples]
            }
        }
        
        print("[OK] 종합 요약 생성 완료")
        return summary
    
    def _generate_overall_assessment(self, conference_assessment: Dict, presentation_frames: int, 
                                   total_samples: int, metadata: Dict) -> Dict[str, Any]:
        """전체 평가 생성"""
        insights = []
        recommendations = []
        
        # 비디오 유형 판단
        duration_minutes = metadata.get('duration_minutes', 0)
        presentation_ratio = presentation_frames / total_samples if total_samples > 0 else 0
        
        if duration_minutes > 30 and presentation_ratio > 0.6:
            video_type = '컨퍼런스 발표 영상 (높은 확률)'
            insights.append(f"{duration_minutes:.0f}분 장시간 영상에서 {presentation_ratio*100:.0f}% 프레젠테이션 화면 감지")
            recommendations.append("전체 영상에 대한 상세 STT 분석 권장")
        elif duration_minutes > 15 and presentation_ratio > 0.4:
            video_type = '세미나/워크샵 영상 (가능성 높음)'
            insights.append(f"중간 길이 영상에서 프레젠테이션 요소 확인")
            recommendations.append("핵심 구간 선별하여 텍스트 추출 분석")
        elif presentation_ratio > 0.7:
            video_type = '프레젠테이션 화면 녹화'
            insights.append("화면의 대부분이 프레젠테이션 슬라이드로 구성")
            recommendations.append("OCR 기반 슬라이드 텍스트 추출 우선")
        else:
            video_type = '일반 영상 (컨퍼런스 관련성 낮음)'
            insights.append("컨퍼런스 특성보다는 일반적인 영상 콘텐츠")
            recommendations.append("기본적인 메타데이터 분석으로 충분")
        
        # 파일 크기 기반 인사이트
        file_size_gb = metadata.get('size', 0) / (1024**3) if metadata.get('size') else 0
        if file_size_gb > 2:
            insights.append(f"대용량 파일 ({file_size_gb:.1f}GB) - 고품질 또는 장시간 콘텐츠")
        
        # 추가 권장사항
        if conference_assessment.get('is_likely_conference_video', False):
            recommendations.append("오디오 분석과 연계하여 종합적 컨퍼런스 인사이트 도출")
            recommendations.append("이미지 OCR 결과와 교차 검증")
        
        return {
            'video_type': video_type,
            'insights': insights,
            'recommendations': recommendations
        }
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """완전한 대용량 비디오 분석 실행"""
        print("\n=== 대용량 비디오 분석 실행 ===")
        
        # 1. 비디오 파일 탐색
        video_files = self.find_video_files()
        
        if not video_files:
            return {'error': '분석할 비디오 파일이 없습니다.'}
        
        # 2. 가장 큰 파일 선택 (대용량 파일)
        target_file = video_files[0]
        
        print(f"\n대상 파일: {target_file['file_name']} ({target_file['file_size_gb']:.2f}GB)")
        
        # 3. 메타데이터 분석
        metadata_result = self.analyze_video_metadata(target_file)
        
        if metadata_result.get('status') != 'success':
            return metadata_result
        
        # 4. 샘플 프레임 추출 (컨퍼런스 관련성이 높은 경우만)
        samples_result = {'samples': [], 'status': 'skipped'}
        
        if metadata_result['conference_assessment']['is_likely_conference_video']:
            print("\n컨퍼런스 관련성 높음 - 샘플 추출 진행")
            samples_result = self.extract_video_samples(target_file, sample_count=5)
        else:
            print("\n컨퍼런스 관련성 낮음 - 샘플 추출 생략")
            samples_result = {
                'total_samples_extracted': 0,
                'samples': [],
                'status': 'skipped_low_relevance'
            }
        
        # 5. 종합 분석 요약
        if samples_result.get('status') == 'success' or samples_result.get('status') == 'skipped_low_relevance':
            summary = self.generate_video_analysis_summary(metadata_result, samples_result)
        else:
            summary = {'error': '샘플 추출 실패로 요약 생성 불가'}
        
        # 6. 최종 결과 구성
        final_result = {
            'analysis_info': {
                'session_id': self.analysis_session['session_id'],
                'analysis_timestamp': datetime.now().isoformat(),
                'target_file': target_file['file_name']
            },
            'metadata_analysis': metadata_result,
            'sample_extraction': samples_result,
            'comprehensive_summary': summary,
            'processing_performance': {
                'total_analysis_time': time.time() - time.mktime(datetime.fromisoformat(self.analysis_session['start_time']).timetuple()),
                'analysis_efficiency': f"{target_file['file_size_gb']:.2f}GB 파일 분석 완료"
            }
        }
        
        return final_result
    
    def save_analysis_results(self, final_result: Dict[str, Any]) -> str:
        """분석 결과 저장"""
        report_path = project_root / f"large_video_analysis_{self.analysis_session['session_id']}.json"
        
        # 세션 정보 업데이트
        self.analysis_session.update({
            'final_results': final_result,
            'completion_time': datetime.now().isoformat()
        })
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_session, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 대용량 비디오 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("대용량 비디오 파일 분석")
    print("=" * 50)
    
    # 비디오 분석기 초기화
    analyzer = LargeVideoAnalyzer()
    
    # 완전한 분석 실행
    final_result = analyzer.run_complete_analysis()
    
    if 'error' in final_result:
        print(f"[ERROR] 분석 실패: {final_result['error']}")
        return final_result
    
    # 결과 저장
    report_path = analyzer.save_analysis_results(final_result)
    
    # 요약 출력
    print(f"\n{'='*50}")
    print("대용량 비디오 분석 완료")
    print(f"{'='*50}")
    
    # 파일 정보
    summary = final_result.get('comprehensive_summary', {})
    file_summary = summary.get('file_summary', {})
    print(f"\n[FILE] 파일 정보:")
    print(f"  파일명: {file_summary.get('name', 'Unknown')}")
    print(f"  크기: {file_summary.get('size_gb', 0):.2f}GB")
    print(f"  길이: {file_summary.get('duration_minutes', 0):.1f}분")
    print(f"  해상도: {file_summary.get('resolution', 'Unknown')}")
    print(f"  품질: {file_summary.get('quality', 'Unknown')}")
    
    # 콘텐츠 분석
    content_analysis = summary.get('content_analysis', {})
    print(f"\n[CONTENT] 콘텐츠 분석:")
    print(f"  컨퍼런스 관련성: {content_analysis.get('conference_relevance', 'Unknown')}")
    print(f"  프레젠테이션 비율: {content_analysis.get('presentation_frames_ratio', '0/0')}")
    print(f"  영상 유형: {content_analysis.get('video_type_assessment', 'Unknown')}")
    
    # 기술적 평가
    tech_assessment = summary.get('technical_assessment', {})
    print(f"\n[TECHNICAL] 기술적 평가:")
    print(f"  코덱: {tech_assessment.get('codec', 'Unknown')}")
    print(f"  비트레이트: {tech_assessment.get('bitrate_mbps', 0)}Mbps")
    print(f"  오디오: {tech_assessment.get('audio_quality', 'Unknown')}")
    
    # 인사이트
    insights = summary.get('analysis_insights', [])
    print(f"\n[INSIGHTS] 분석 인사이트:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    
    # 권장사항
    recommendations = summary.get('recommended_actions', [])
    print(f"\n[RECOMMENDATIONS] 권장사항:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 성능 정보
    performance = final_result.get('processing_performance', {})
    print(f"\n[PERFORMANCE] 처리 성능:")
    print(f"  총 분석 시간: {performance.get('total_analysis_time', 0):.1f}초")
    print(f"  분석 효율성: {performance.get('analysis_efficiency', 'Unknown')}")
    
    print(f"\n[FILE] 상세 보고서: {Path(report_path).name}")
    
    return final_result

if __name__ == "__main__":
    main()