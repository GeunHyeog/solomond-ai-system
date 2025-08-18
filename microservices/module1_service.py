#!/usr/bin/env python3
"""
🎯 Module 1: 컨퍼런스 분석 마이크로서비스
- Streamlit UI의 핵심 기능을 FastAPI로 변환
- 스마트 메모리 매니저와 완전 통합
- 포트 충돌 해결 (8501 → 8001)
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 최적화된 컴포넌트들 import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.optimized_ai_loader import optimized_loader
from core.smart_memory_manager import get_memory_stats
from core.robust_file_processor import robust_processor
from config.security_config import get_system_config

logger = logging.getLogger(__name__)

# Pydantic 모델들
class AnalysisRequest(BaseModel):
    """분석 요청 모델"""
    session_id: Optional[str] = None
    analysis_type: str = "comprehensive"  # 'quick', 'comprehensive', 'detailed'
    context_info: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    """분석 결과 모델"""
    session_id: str
    status: str
    results: Dict[str, Any]
    processing_time: float
    timestamp: datetime

class ConferenceAnalysisService:
    """컨퍼런스 분석 서비스"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
    async def analyze_files(self, files: List[UploadFile], 
                          analysis_type: str = "comprehensive",
                          context_info: Optional[Dict] = None) -> Dict[str, Any]:
        """파일 분석 실행"""
        session_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 세션 정보 저장
            self.active_sessions[session_id] = {
                'start_time': start_time,
                'status': 'processing',
                'files_count': len(files),
                'analysis_type': analysis_type
            }
            
            results = {
                'session_id': session_id,
                'files_processed': [],
                'audio_analysis': {},
                'image_analysis': {},
                'video_analysis': {},
                'summary': {}
            }
            
            # 파일별 처리
            for file in files:
                file_result = await self._process_single_file(file, analysis_type)
                results['files_processed'].append(file_result)
                
                # 파일 타입별 분류
                if file_result['type'] == 'audio':
                    results['audio_analysis'][file.filename] = file_result
                elif file_result['type'] == 'image':
                    results['image_analysis'][file.filename] = file_result
                elif file_result['type'] == 'video':
                    results['video_analysis'][file.filename] = file_result
            
            # 통합 분석 (컨텍스트 정보 활용)
            if context_info:
                results['context'] = context_info
                results['summary'] = await self._generate_contextual_summary(
                    results, context_info
                )
            else:
                results['summary'] = await self._generate_summary(results)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 세션 완료
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['processing_time'] = processing_time
            
            return {
                'session_id': session_id,
                'status': 'success',
                'results': results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"분석 실패 {session_id}: {e}")
            
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'failed'
                self.active_sessions[session_id]['error'] = str(e)
            
            raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")
    
    async def _process_single_file(self, file: UploadFile, 
                                 analysis_type: str) -> Dict[str, Any]:
        """단일 파일 처리"""
        file_content = await file.read()
        file_size = len(file_content)
        file_extension = Path(file.filename).suffix.lower()
        
        result = {
            'filename': file.filename,
            'size_bytes': file_size,
            'extension': file_extension,
            'type': self._detect_file_type(file_extension),
            'analysis': {}
        }
        
        try:
            # 파일 타입별 분석
            if result['type'] == 'audio':
                result['analysis'] = await self._analyze_audio_file(
                    file_content, analysis_type
                )
            elif result['type'] == 'image':
                result['analysis'] = await self._analyze_image_file(
                    file_content, analysis_type
                )
            elif result['type'] == 'video':
                result['analysis'] = await self._analyze_video_file(
                    file_content, analysis_type
                )
            else:
                result['analysis'] = {'error': '지원되지 않는 파일 형식'}
            
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"파일 처리 실패 {file.filename}: {e}")
            result['status'] = 'failed'
            result['analysis'] = {'error': str(e)}
        
        return result
    
    def _detect_file_type(self, extension: str) -> str:
        """파일 확장자로 타입 감지"""
        audio_exts = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        if extension in audio_exts:
            return 'audio'
        elif extension in image_exts:
            return 'image'
        elif extension in video_exts:
            return 'video'
        else:
            return 'unknown'
    
    async def _analyze_audio_file(self, content: bytes, 
                                analysis_type: str) -> Dict[str, Any]:
        """오디오 파일 분석 (강화된 파일 처리 포함)"""
        try:
            # robust_file_processor로 파일 전처리 (m4a → wav 변환 포함)
            file_info = await robust_processor.process_file(
                content, "audio_file.tmp", target_format='.wav'
            )
            
            if file_info.conversion_path is None:
                return {'error': '파일 전처리 실패'}
            
            # Whisper STT 모델 사용
            model_size = "small" if analysis_type == "detailed" else "base"
            
            with optimized_loader.get_whisper_model(model_size) as whisper_model:
                result = whisper_model.transcribe(file_info.conversion_path)
                
                analysis_result = {
                    'transcript': result['text'],
                    'language': result.get('language', 'unknown'),
                    'confidence': getattr(result, 'avg_logprob', 0.0),
                    'segments': result.get('segments', []),
                    'model_used': f"whisper_{model_size}",
                    'file_info': {
                        'original_size': file_info.original_size,
                        'processed_size': file_info.processed_size,
                        'format_converted': file_info.is_converted,
                        'duration': file_info.duration
                    }
                }
                
                # 추가 분석 (컨텍스트별)
                if analysis_type in ['comprehensive', 'detailed']:
                    analysis_result['speaker_info'] = self._analyze_speakers(result)
                    analysis_result['key_topics'] = self._extract_topics(result['text'])
            
            # 임시 파일 정리
            robust_processor.cleanup_temp_files(file_info)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"오디오 분석 실패: {e}")
            return {'error': str(e)}
    
    async def _analyze_image_file(self, content: bytes, 
                                analysis_type: str) -> Dict[str, Any]:
        """이미지 파일 분석 (강화된 파일 처리 포함)"""
        try:
            # robust_file_processor로 파일 전처리
            file_info = await robust_processor.process_file(
                content, "image_file.tmp"
            )
            
            if file_info.conversion_path is None:
                return {'error': '이미지 파일 전처리 실패'}
            
            # EasyOCR 사용
            languages = ['en', 'ko'] if analysis_type == 'detailed' else ['en']
            
            with optimized_loader.get_easyocr_reader(languages) as reader:
                ocr_results = reader.readtext(file_info.conversion_path)
                
                analysis_result = {
                    'text_blocks': [],
                    'total_text': '',
                    'languages_detected': languages,
                    'confidence_avg': 0.0,
                    'file_info': {
                        'original_size': file_info.original_size,
                        'processed_size': file_info.processed_size,
                        'format': file_info.format
                    }
                }
                
                total_confidence = 0
                for (bbox, text, confidence) in ocr_results:
                    analysis_result['text_blocks'].append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    analysis_result['total_text'] += text + ' '
                    total_confidence += confidence
                
                if ocr_results:
                    analysis_result['confidence_avg'] = total_confidence / len(ocr_results)
                
                # 추가 분석
                if analysis_type in ['comprehensive', 'detailed']:
                    analysis_result['key_info'] = self._extract_key_info(
                        analysis_result['total_text']
                    )
            
            # 임시 파일 정리
            robust_processor.cleanup_temp_files(file_info)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"이미지 분석 실패: {e}")
            return {'error': str(e)}
    
    async def _analyze_video_file(self, content: bytes, 
                                analysis_type: str) -> Dict[str, Any]:
        """비디오 파일 분석 (기본 구현)"""
        return {
            'message': '비디오 분석 기능 개발 예정',
            'size_bytes': len(content),
            'analysis_type': analysis_type
        }
    
    def _analyze_speakers(self, whisper_result: Dict) -> Dict[str, Any]:
        """화자 분석 (기본 구현)"""
        segments = whisper_result.get('segments', [])
        
        return {
            'total_segments': len(segments),
            'estimated_speakers': min(len(segments) // 10 + 1, 5),  # 추정
            'speaking_time': sum(seg.get('end', 0) - seg.get('start', 0) 
                               for seg in segments)
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """주요 토픽 추출 (기본 구현)"""
        # 실제로는 더 정교한 NLP 분석 필요
        keywords = ['컨퍼런스', '발표', '토론', '질문', '답변', '주얼리', '보석', '다이아몬드']
        found_topics = [kw for kw in keywords if kw in text]
        
        return found_topics or ['일반 대화']
    
    def _extract_key_info(self, text: str) -> Dict[str, Any]:
        """텍스트에서 핵심 정보 추출"""
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'has_numbers': any(c.isdigit() for c in text),
            'has_korean': any('\uac00' <= c <= '\ud7a3' for c in text)
        }
    
    async def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """결과 요약 생성"""
        return {
            'total_files': len(results['files_processed']),
            'audio_files': len(results['audio_analysis']),
            'image_files': len(results['image_analysis']),
            'video_files': len(results['video_analysis']),
            'processing_summary': '분석 완료',
            'recommendations': ['추가 분석 가능', '결과 다운로드 가능']
        }
    
    async def _generate_contextual_summary(self, results: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 기반 요약 생성"""
        base_summary = await self._generate_summary(results)
        
        # 컨텍스트 정보 활용
        if context.get('event_type') == 'conference':
            base_summary['event_analysis'] = {
                'type': 'conference',
                'relevance': 'high',
                'insights': '컨퍼런스 내용 분석 완료'
            }
        
        return base_summary

# FastAPI 앱 생성
app = FastAPI(
    title="Module 1: 컨퍼런스 분석 서비스",
    description="음성, 이미지, 비디오 통합 분석",
    version="4.0.0"
)

# 서비스 인스턴스
service = ConferenceAnalysisService()

@app.get("/health")
async def health_check():
    """헬스체크"""
    memory_stats = get_memory_stats()
    
    return {
        "status": "healthy",
        "service": "module1_conference",
        "version": "4.0.0",
        "memory": {
            "loaded_models": memory_stats.get('loaded_models', 0),
            "memory_percent": memory_stats.get('memory_info', {}).get('percent', 0)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    analysis_type: str = "comprehensive",
    context_info: Optional[str] = None
):
    """파일 분석 API"""
    if not files:
        raise HTTPException(status_code=400, detail="분석할 파일이 없습니다")
    
    # 컨텍스트 정보 파싱
    context = {}
    if context_info:
        try:
            import json
            context = json.loads(context_info)
        except:
            context = {"raw_context": context_info}
    
    # 분석 실행
    result = await service.analyze_files(files, analysis_type, context)
    
    return result

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    if session_id not in service.active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return service.active_sessions[session_id]

@app.get("/sessions")
async def list_sessions():
    """전체 세션 목록"""
    return {
        "total_sessions": len(service.active_sessions),
        "sessions": service.active_sessions
    }

@app.get("/stats")
async def get_service_stats():
    """서비스 통계"""
    memory_stats = get_memory_stats()
    
    return {
        "service_info": {
            "name": "module1_conference",
            "version": "4.0.0",
            "uptime": "실행 중"
        },
        "memory": memory_stats,
        "sessions": {
            "total": len(service.active_sessions),
            "active": len([s for s in service.active_sessions.values() 
                          if s['status'] == 'processing']),
            "completed": len([s for s in service.active_sessions.values() 
                             if s['status'] == 'completed'])
        },
        "file_processor": {
            "ffmpeg_available": robust_processor.ffmpeg_available,
            "max_file_size": robust_processor.max_file_size,
            "supported_formats": robust_processor.get_supported_formats()
        }
    }

@app.post("/analyze/large-file")
async def analyze_large_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_type: str = "comprehensive"
):
    """대용량 파일 청킹 분석"""
    session_id = str(uuid.uuid4())
    
    try:
        # 대용량 파일 청킹 처리
        async def chunk_callback(chunk_num: int, chunk_size: int, total_size: int):
            logger.info(f"청크 {chunk_num} 처리 중: {chunk_size} bytes, 총 {total_size:,} bytes")
        
        # 스트림으로 처리
        file_stream = file.file
        file_info = await robust_processor.process_large_file_chunked(
            file_stream, file.filename, chunk_callback
        )
        
        # 분석 실행
        result = await service.analyze_files([file], analysis_type)
        
        return {
            "session_id": session_id,
            "status": "success",
            "file_info": {
                "filename": file.filename,
                "original_size": file_info.original_size,
                "processed_size": file_info.processed_size,
                "chunks_processed": "대용량 파일 청킹 완료"
            },
            "analysis_result": result
        }
        
    except ValueError as e:
        if "파일 크기 초과" in str(e):
            raise HTTPException(
                status_code=413, 
                detail=f"파일이 너무 큽니다 (최대 {robust_processor.max_file_size:,} bytes)"
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"대용량 파일 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """지원되는 파일 형식 조회"""
    formats = robust_processor.get_supported_formats()
    
    return {
        "file_processor": {
            "ffmpeg_available": robust_processor.ffmpeg_available,
            "max_file_size_bytes": robust_processor.max_file_size,
            "max_file_size_gb": round(robust_processor.max_file_size / (1024**3), 1),
            "chunk_size_mb": round(robust_processor.chunk_size / (1024**2), 1)
        },
        "supported_formats": formats,
        "format_details": {
            "audio": {
                "description": "음성 파일 (Whisper STT로 분석)",
                "conversion": "m4a, mp3, aac → wav (16kHz, 모노)"
            },
            "image": {
                "description": "이미지 파일 (EasyOCR로 텍스트 추출)",
                "languages": ["en", "ko"]
            },
            "video": {
                "description": "비디오 파일 (기본 정보만 추출, 향후 확장 예정)",
                "status": "개발 중"
            }
        }
    }

if __name__ == "__main__":
    logger.info("🎯 Module 1 컨퍼런스 분석 서비스 시작: http://localhost:8001")
    
    uvicorn.run(
        "module1_service:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )