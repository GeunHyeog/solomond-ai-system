#!/usr/bin/env python3
"""
오류 복구 분석기
성공률 96.4% → 99%+ 향상을 위한 대체 분석 방법
"""

import os
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorRecoveryAnalyzer:
    """실패한 파일에 대한 대체 분석 방법 제공"""
    
    def __init__(self):
        self.fallback_methods = {
            'audio': ['basic_info', 'metadata_only', 'partial_stt'],
            'image': ['basic_ocr', 'text_detection', 'metadata_only'],
            'video': ['metadata_only', 'basic_info', 'frame_extraction'],
            'document': ['text_extraction', 'basic_parsing', 'metadata_only']
        }
        
    def recover_failed_analysis(self, file_path: str, file_type: str, original_error: str) -> Dict[str, Any]:
        """실패한 분석에 대한 복구 시도"""
        logger.info(f"[RECOVERY] 실패 파일 복구 시도: {os.path.basename(file_path)}")
        
        recovery_result = {
            "status": "partial_success",
            "recovery_method": None,
            "partial_data": {},
            "original_error": original_error,
            "recovery_time": time.time()
        }
        
        try:
            if file_type == 'audio':
                recovery_result.update(self._recover_audio_analysis(file_path))
            elif file_type == 'image':
                recovery_result.update(self._recover_image_analysis(file_path))
            elif file_type == 'video':
                recovery_result.update(self._recover_video_analysis(file_path))
            elif file_type == 'document':
                recovery_result.update(self._recover_document_analysis(file_path))
            else:
                recovery_result.update(self._recover_generic_analysis(file_path))
                
            recovery_result["recovery_time"] = time.time() - recovery_result["recovery_time"]
            logger.info(f"[RECOVERY] 복구 완료: {recovery_result.get('recovery_method', 'generic')}")
            
        except Exception as e:
            logger.error(f"[RECOVERY] 복구 실패: {str(e)}")
            recovery_result["status"] = "recovery_failed"
            recovery_result["recovery_error"] = str(e)
            
        return recovery_result
    
    def _recover_audio_analysis(self, file_path: str) -> Dict[str, Any]:
        """음성 파일 복구 분석"""
        try:
            # 기본 파일 정보
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            partial_data = {
                "file_info": {
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "format": file_ext,
                    "estimated_duration": self._estimate_audio_duration(file_size, file_ext)
                },
                "fallback_transcription": f"[음성 파일 감지됨: {file_ext} 형식, {partial_data.get('file_info', {}).get('size_mb', 0)}MB]",
                "confidence": 0.3,  # 낮은 신뢰도
                "keywords": self._extract_filename_keywords(file_path)
            }
            
            return {
                "recovery_method": "audio_metadata",
                "partial_data": partial_data
            }
            
        except Exception as e:
            return {"recovery_method": "audio_failed", "recovery_error": str(e)}
    
    def _recover_image_analysis(self, file_path: str) -> Dict[str, Any]:
        """이미지 파일 복구 분석"""
        try:
            # PIL로 기본 이미지 정보 시도
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    
                partial_data = {
                    "image_info": {
                        "dimensions": f"{width}x{height}",
                        "mode": mode,
                        "estimated_text": "[이미지에서 텍스트 추출 실패 - 기본 정보만 제공]"
                    },
                    "detected_text": f"[이미지 파일: {width}x{height} {mode}]",
                    "confidence": 0.2,
                    "keywords": self._extract_filename_keywords(file_path)
                }
                
                return {
                    "recovery_method": "image_metadata",
                    "partial_data": partial_data
                }
                
            except ImportError:
                # PIL 없으면 기본 정보만
                file_size = os.path.getsize(file_path)
                partial_data = {
                    "basic_info": {
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "detected_text": f"[이미지 파일 감지됨: {os.path.splitext(file_path)[1]}]"
                    },
                    "confidence": 0.1,
                    "keywords": self._extract_filename_keywords(file_path)
                }
                
                return {
                    "recovery_method": "image_basic",
                    "partial_data": partial_data
                }
                
        except Exception as e:
            return {"recovery_method": "image_failed", "recovery_error": str(e)}
    
    def _recover_video_analysis(self, file_path: str) -> Dict[str, Any]:
        """비디오 파일 복구 분석"""
        try:
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            partial_data = {
                "video_info": {
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "format": file_ext,
                    "estimated_duration": self._estimate_video_duration(file_size, file_ext)
                },
                "summary": f"[비디오 파일 감지됨: {file_ext} 형식, {round(file_size / (1024 * 1024), 2)}MB]",
                "confidence": 0.2,
                "keywords": self._extract_filename_keywords(file_path)
            }
            
            return {
                "recovery_method": "video_metadata",
                "partial_data": partial_data
            }
            
        except Exception as e:
            return {"recovery_method": "video_failed", "recovery_error": str(e)}
    
    def _recover_document_analysis(self, file_path: str) -> Dict[str, Any]:
        """문서 파일 복구 분석"""
        try:
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 텍스트 파일이면 직접 읽기 시도
            if file_ext in ['.txt', '.md', '.csv']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:1000]  # 처음 1000자만
                    
                    partial_data = {
                        "extracted_text": content,
                        "confidence": 0.8,
                        "keywords": self._extract_text_keywords(content)
                    }
                    
                    return {
                        "recovery_method": "text_direct_read",
                        "partial_data": partial_data
                    }
                except:
                    pass
            
            # 기본 메타데이터
            partial_data = {
                "document_info": {
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "format": file_ext,
                    "estimated_content": f"[문서 파일: {file_ext} 형식]"
                },
                "confidence": 0.1,
                "keywords": self._extract_filename_keywords(file_path)
            }
            
            return {
                "recovery_method": "document_metadata",
                "partial_data": partial_data
            }
            
        except Exception as e:
            return {"recovery_method": "document_failed", "recovery_error": str(e)}
    
    def _recover_generic_analysis(self, file_path: str) -> Dict[str, Any]:
        """일반 파일 복구 분석"""
        try:
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            
            partial_data = {
                "file_info": {
                    "name": file_name,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "format": file_ext,
                    "analysis_note": "원본 분석 실패 - 기본 정보만 제공"
                },
                "confidence": 0.1,
                "keywords": self._extract_filename_keywords(file_path)
            }
            
            return {
                "recovery_method": "generic_metadata",
                "partial_data": partial_data
            }
            
        except Exception as e:
            return {"recovery_method": "generic_failed", "recovery_error": str(e)}
    
    def _estimate_audio_duration(self, file_size_bytes: int, file_ext: str) -> str:
        """파일 크기로 오디오 길이 추정"""
        # 대략적인 추정 (비트레이트 기반)
        if file_ext in ['.mp3']:
            # MP3: 평균 128kbps
            duration_seconds = file_size_bytes / (128 * 1000 / 8)
        elif file_ext in ['.wav']:
            # WAV: 평균 1411kbps (CD 품질)
            duration_seconds = file_size_bytes / (1411 * 1000 / 8)
        elif file_ext in ['.m4a']:
            # M4A: 평균 256kbps
            duration_seconds = file_size_bytes / (256 * 1000 / 8)
        else:
            # 기본값
            duration_seconds = file_size_bytes / (192 * 1000 / 8)
        
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        return f"약 {minutes}분 {seconds}초"
    
    def _estimate_video_duration(self, file_size_bytes: int, file_ext: str) -> str:
        """파일 크기로 비디오 길이 추정"""
        # 대략적인 추정 (해상도별 평균 비트레이트)
        if file_size_bytes > 500 * 1024 * 1024:  # 500MB 이상
            # HD 비디오 추정: 5Mbps
            duration_seconds = file_size_bytes / (5 * 1000 * 1000 / 8)
        else:
            # SD 비디오 추정: 2Mbps
            duration_seconds = file_size_bytes / (2 * 1000 * 1000 / 8)
        
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        return f"약 {minutes}분 {seconds}초"
    
    def _extract_filename_keywords(self, file_path: str) -> list:
        """파일명에서 키워드 추출"""
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # 주요 키워드 패턴
        jewelry_terms = ['diamond', 'gold', 'silver', 'jewelry', 'ring', 'necklace', 
                        '다이아몬드', '금', '은', '주얼리', '반지', '목걸이']
        
        business_terms = ['meeting', 'conference', 'presentation', 'report',
                         '회의', '발표', '보고서', '컨퍼런스']
        
        found_keywords = []
        filename_lower = filename.lower()
        
        for term in jewelry_terms + business_terms:
            if term.lower() in filename_lower:
                found_keywords.append(term)
        
        # 숫자나 날짜도 키워드로 추가
        import re
        dates = re.findall(r'\d{4}[-_]\d{2}[-_]\d{2}', filename)
        numbers = re.findall(r'\d+', filename)
        
        found_keywords.extend(dates[:2])  # 최대 2개 날짜
        found_keywords.extend(numbers[:3])  # 최대 3개 숫자
        
        return found_keywords[:10]  # 최대 10개
    
    def _extract_text_keywords(self, text: str) -> list:
        """텍스트에서 키워드 추출"""
        if not text:
            return []
        
        # 간단한 키워드 추출
        words = text.split()
        keywords = []
        
        for word in words:
            if len(word) > 3:  # 3글자 이상만
                keywords.append(word.strip('.,!?;:'))
        
        return list(set(keywords))[:15]  # 중복 제거 후 최대 15개

# 전역 인스턴스
error_recovery_analyzer = ErrorRecoveryAnalyzer()

def recover_failed_analysis(file_path: str, file_type: str, original_error: str) -> Dict[str, Any]:
    """실패한 분석 복구 (전역 접근용)"""
    return error_recovery_analyzer.recover_failed_analysis(file_path, file_type, original_error)