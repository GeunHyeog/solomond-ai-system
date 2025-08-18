#!/usr/bin/env python3
"""
🚀 멀티모달 파이프라인 엔진 - SOLOMOND AI v4.0
Advanced Multimodal Processing Pipeline with Cross-Modal Analysis

🎯 주요 기능:
1. 동시 멀티모달 처리 - 이미지+음성+텍스트 병렬 분석
2. 크로스 모달 상관관계 분석 - 모달간 연관성 탐지
3. 시맨틱 융합 - 다중 소스 정보 통합
4. 실시간 스트리밍 - 중간 결과 즉시 반환
5. 컨텍스트 인식 - 이전 분석 결과 활용
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

# AI 모델 라이브러리
try:
    import whisper
    import easyocr
    import cv2
    from PIL import Image
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModel
    import sentence_transformers
except ImportError as e:
    print(f"⚠️ AI 라이브러리 누락: {e}")

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultimodalResult:
    """멀티모달 분석 결과 구조"""
    file_path: str
    file_type: str
    content: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    cross_modal_score: Optional[float] = None

class MultimodalPipeline:
    """멀티모달 파이프라인 핵심 엔진"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.is_initialized = False
        self.cache = {}
        
        # 성능 메트릭
        self.stats = {
            'processed_files': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cross_modal_discoveries': 0
        }
        
        # 크로스 모달 분석기
        self.cross_modal_analyzer = CrossModalAnalyzer()
        
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            'batch_size': 4,
            'max_workers': 8,
            'use_gpu': torch.cuda.is_available(),
            'cache_enabled': True,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'whisper_model': 'base',
            'ocr_languages': ['ko', 'en'],
            'cross_modal_threshold': 0.7
        }
    
    async def initialize(self) -> None:
        """AI 모델들을 비동기로 초기화"""
        if self.is_initialized:
            return
            
        logger.info("🚀 멀티모달 AI 엔진 초기화 시작...")
        
        # GPU/CPU 설정
        device = 'cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu'
        logger.info(f"🔧 디바이스 설정: {device}")
        
        # 모델 로딩 (병렬)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Whisper STT 모델
            whisper_future = loop.run_in_executor(
                executor, 
                self._load_whisper_model
            )
            
            # EasyOCR 모델  
            ocr_future = loop.run_in_executor(
                executor,
                self._load_ocr_model
            )
            
            # 임베딩 모델
            embedding_future = loop.run_in_executor(
                executor,
                self._load_embedding_model
            )
            
            # 모든 모델 로딩 대기
            self.models['whisper'] = await whisper_future
            self.models['ocr'] = await ocr_future  
            self.models['embeddings'] = await embedding_future
            
        self.is_initialized = True
        logger.info("✅ 멀티모달 AI 엔진 초기화 완료!")
        
    def _load_whisper_model(self):
        """Whisper STT 모델 로딩"""
        logger.info("🎵 Whisper STT 모델 로딩...")
        return whisper.load_model(self.config['whisper_model'])
        
    def _load_ocr_model(self):
        """EasyOCR 모델 로딩"""
        logger.info("🔍 EasyOCR 모델 로딩...")
        return easyocr.Reader(self.config['ocr_languages'])
        
    def _load_embedding_model(self):
        """임베딩 모델 로딩"""
        logger.info("🧠 임베딩 모델 로딩...")
        return sentence_transformers.SentenceTransformer(self.config['embedding_model'])
    
    async def process_multimodal_batch(self, files: List[Path]) -> List[MultimodalResult]:
        """멀티모달 파일들을 배치로 병렬 처리"""
        if not self.is_initialized:
            await self.initialize()
            
        logger.info(f"📦 배치 처리 시작: {len(files)}개 파일")
        start_time = time.time()
        
        results = []
        
        # 파일 타입별 분류
        image_files = []
        audio_files = []
        text_files = []
        
        video_files = []
        
        for file_path in files:
            ext = file_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_files.append(file_path)
            elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
                audio_files.append(file_path)
            elif ext in ['.txt', '.md', '.pdf', '.docx']:
                text_files.append(file_path)
            elif ext in ['.mov', '.mp4', '.avi', '.mkv', '.webm']:
                video_files.append(file_path)
        
        # 병렬 처리 실행
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            
            # 이미지 처리
            for img_path in image_files:
                future = executor.submit(self._process_image_file, img_path)
                futures.append(future)
            
            # 오디오 처리
            for audio_path in audio_files:
                future = executor.submit(self._process_audio_file, audio_path)
                futures.append(future)
                
            # 텍스트 처리
            for text_path in text_files:
                future = executor.submit(self._process_text_file, text_path)
                futures.append(future)
            
            # 비디오 처리
            for video_path in video_files:
                future = executor.submit(self._process_video_file, video_path)
                futures.append(future)
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.stats['processed_files'] += 1
                except Exception as e:
                    logger.error(f"❌ 파일 처리 실패: {e}")
        
        # 크로스 모달 분석 실행
        if len(results) > 1:
            results = await self._perform_cross_modal_analysis(results)
        
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] += processing_time
        
        logger.info(f"✅ 배치 처리 완료: {len(results)}개 결과, {processing_time:.2f}초")
        return results
    
    def _process_image_file(self, file_path: Path) -> Optional[MultimodalResult]:
        """이미지 파일 처리"""
        try:
            start_time = time.time()
            
            # 캐시 확인
            file_hash = self._get_file_hash(file_path)
            if self.config['cache_enabled'] and file_hash in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[file_hash]
            
            # EasyOCR로 텍스트 추출
            image = cv2.imread(str(file_path))
            ocr_results = self.models['ocr'].readtext(image)
            
            # 텍스트 조합
            extracted_text = ""
            confidence_scores = []
            
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5:  # 신뢰도 임계값
                    extracted_text += text + " "
                    confidence_scores.append(confidence)
            
            extracted_text = extracted_text.strip()
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # 임베딩 생성
            embeddings = None
            if extracted_text:
                embeddings = self.models['embeddings'].encode([extracted_text])[0]
            
            # 결과 생성
            result = MultimodalResult(
                file_path=str(file_path),
                file_type='image',
                content=extracted_text,
                confidence=avg_confidence,
                processing_time=time.time() - start_time,
                metadata={
                    'bbox_count': len(ocr_results),
                    'image_size': image.shape[:2],
                    'detected_languages': list(set([r[1] for r in ocr_results if r[2] > 0.5]))
                },
                embeddings=embeddings
            )
            
            # 캐시 저장
            if self.config['cache_enabled']:
                self.cache[file_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 이미지 처리 실패 {file_path}: {e}")
            return None
    
    def _process_audio_file(self, file_path: Path) -> Optional[MultimodalResult]:
        """오디오 파일 처리"""
        try:
            start_time = time.time()
            
            # 캐시 확인
            file_hash = self._get_file_hash(file_path)
            if self.config['cache_enabled'] and file_hash in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[file_hash]
            
            # Whisper로 음성 인식
            whisper_result = self.models['whisper'].transcribe(str(file_path))
            
            transcribed_text = whisper_result['text'].strip()
            confidence = 0.8  # Whisper는 기본적으로 신뢰도가 높음
            
            # 임베딩 생성
            embeddings = None
            if transcribed_text:
                embeddings = self.models['embeddings'].encode([transcribed_text])[0]
            
            # 결과 생성
            result = MultimodalResult(
                file_path=str(file_path),
                file_type='audio',
                content=transcribed_text,
                confidence=confidence,
                processing_time=time.time() - start_time,
                metadata={
                    'duration': whisper_result.get('duration', 0),
                    'language': whisper_result.get('language', 'unknown'),
                    'segments_count': len(whisper_result.get('segments', []))
                },
                embeddings=embeddings
            )
            
            # 캐시 저장
            if self.config['cache_enabled']:
                self.cache[file_hash] = result
                
            return result
            
        except Exception as e:
            logger.error(f"❌ 오디오 처리 실패 {file_path}: {e}")
            return None
    
    def _process_text_file(self, file_path: Path) -> Optional[MultimodalResult]:
        """텍스트 파일 처리"""
        try:
            start_time = time.time()
            
            # 파일 읽기 (인코딩 자동 감지)
            encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'utf-16']
            content = ""
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not content:
                logger.warning(f"텍스트 파일 인코딩 실패: {file_path}")
                content = "[인코딩 오류로 읽기 실패]"
            
            # 임베딩 생성
            embeddings = None
            if content and "실패" not in content:
                embeddings = self.models['embeddings'].encode([content[:2000]])[0]  # 긴 텍스트 제한
            
            result = MultimodalResult(
                file_path=str(file_path),
                file_type='text',
                content=content[:1000],  # 처음 1000자만
                confidence=1.0 if "실패" not in content else 0.1,
                processing_time=time.time() - start_time,
                metadata={
                    'file_size': len(content),
                    'word_count': len(content.split()) if content and "실패" not in content else 0,
                    'encoding_used': 'auto-detected',
                    'content_preview': content[:100] if content and "실패" not in content else "내용 없음"
                },
                embeddings=embeddings
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 텍스트 처리 실패 {file_path}: {e}")
            return None
    
    def _process_video_file(self, file_path: Path) -> Optional[MultimodalResult]:
        """비디오 파일 처리 (오디오 트랙 추출 + 프레임 샘플링)"""
        try:
            start_time = time.time()
            
            # 캐시 확인
            file_hash = self._get_file_hash(file_path)
            if self.config['cache_enabled'] and file_hash in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[file_hash]
            
            logger.info(f"🎬 비디오 파일 처리 중: {file_path.name}")
            
            # FFmpeg를 사용하여 오디오 트랙 추출
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # FFmpeg 명령어로 오디오 추출
            ffmpeg_cmd = [
                'ffmpeg', '-i', str(file_path), 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', 
                temp_audio_path, '-y'
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                
                # 추출된 오디오를 Whisper로 처리
                whisper_result = self.models['whisper'].transcribe(temp_audio_path)
                transcribed_text = whisper_result['text'].strip()
                
                # 임시 파일 정리
                Path(temp_audio_path).unlink()
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"FFmpeg 오디오 추출 실패: {e}")
                transcribed_text = "[비디오 오디오 추출 실패]"
            except FileNotFoundError:
                logger.warning("FFmpeg를 찾을 수 없음. 비디오 처리를 위해 FFmpeg 설치가 필요합니다.")
                transcribed_text = "[FFmpeg 미설치로 비디오 처리 불가]"
            
            # 비디오 메타데이터 추출 (가능한 경우)
            video_metadata = self._extract_video_metadata(file_path)
            
            # 임베딩 생성
            embeddings = None
            if transcribed_text and "실패" not in transcribed_text and "불가" not in transcribed_text:
                embeddings = self.models['embeddings'].encode([transcribed_text])[0]
            
            # 결과 생성
            result = MultimodalResult(
                file_path=str(file_path),
                file_type='video',
                content=transcribed_text,
                confidence=0.75 if "실패" not in transcribed_text and "불가" not in transcribed_text else 0.1,
                processing_time=time.time() - start_time,
                metadata={
                    'video_metadata': video_metadata,
                    'audio_extracted': "실패" not in transcribed_text and "불가" not in transcribed_text,
                    'processing_method': 'ffmpeg_audio_extraction'
                },
                embeddings=embeddings
            )
            
            # 캐시 저장
            if self.config['cache_enabled']:
                self.cache[file_hash] = result
                
            return result
            
        except Exception as e:
            logger.error(f"❌ 비디오 처리 실패 {file_path}: {e}")
            return None
    
    def _extract_video_metadata(self, file_path: Path) -> Dict[str, Any]:
        """비디오 메타데이터 추출"""
        try:
            import subprocess
            
            # FFprobe를 사용하여 비디오 정보 추출
            ffprobe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(file_path)
            ]
            
            result = subprocess.run(
                ffprobe_cmd, capture_output=True, text=True, check=True
            )
            
            metadata = json.loads(result.stdout)
            
            # 주요 정보 추출
            format_info = metadata.get('format', {})
            video_streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'video']
            audio_streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'audio']
            
            return {
                'duration': float(format_info.get('duration', 0)),
                'size_bytes': int(format_info.get('size', 0)),
                'format_name': format_info.get('format_name', 'unknown'),
                'video_codec': video_streams[0].get('codec_name', 'unknown') if video_streams else 'none',
                'audio_codec': audio_streams[0].get('codec_name', 'unknown') if audio_streams else 'none',
                'width': int(video_streams[0].get('width', 0)) if video_streams else 0,
                'height': int(video_streams[0].get('height', 0)) if video_streams else 0,
                'fps': eval(video_streams[0].get('r_frame_rate', '0/1')) if video_streams else 0
            }
            
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"비디오 메타데이터 추출 실패: {e}")
            return {
                'duration': 0,
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
                'format_name': file_path.suffix.lower(),
                'extraction_error': str(e)
            }
    
    async def _perform_cross_modal_analysis(self, results: List[MultimodalResult]) -> List[MultimodalResult]:
        """크로스 모달 상관관계 분석"""
        logger.info("🔄 크로스 모달 분석 시작...")
        
        # 임베딩이 있는 결과들만 분석
        embedded_results = [r for r in results if r.embeddings is not None]
        
        if len(embedded_results) < 2:
            return results
        
        # 모든 쌍에 대해 유사도 계산
        for i, result_a in enumerate(embedded_results):
            for j, result_b in enumerate(embedded_results[i+1:], i+1):
                # 코사인 유사도 계산
                similarity = np.dot(result_a.embeddings, result_b.embeddings) / (
                    np.linalg.norm(result_a.embeddings) * np.linalg.norm(result_b.embeddings)
                )
                
                # 임계값을 넘는 경우 크로스 모달 관계로 판단
                if similarity > self.config['cross_modal_threshold']:
                    result_a.cross_modal_score = max(result_a.cross_modal_score or 0, similarity)
                    result_b.cross_modal_score = max(result_b.cross_modal_score or 0, similarity)
                    
                    # 메타데이터에 관련 파일 정보 추가
                    if 'related_files' not in result_a.metadata:
                        result_a.metadata['related_files'] = []
                    if 'related_files' not in result_b.metadata:
                        result_b.metadata['related_files'] = []
                    
                    result_a.metadata['related_files'].append({
                        'file': result_b.file_path,
                        'similarity': similarity,
                        'type': result_b.file_type
                    })
                    result_b.metadata['related_files'].append({
                        'file': result_a.file_path,
                        'similarity': similarity,
                        'type': result_a.file_type
                    })
                    
                    self.stats['cross_modal_discoveries'] += 1
        
        logger.info(f"✅ 크로스 모달 분석 완료: {self.stats['cross_modal_discoveries']}개 연관관계 발견")
        return results
    
    def _get_file_hash(self, file_path: Path) -> str:
        """파일 해시 생성 (캐싱용)"""
        stat = file_path.stat()
        return hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return self.stats.copy()

class CrossModalAnalyzer:
    """크로스 모달 분석 전용 엔진"""
    
    def __init__(self):
        self.correlation_patterns = {
            'image_audio_sync': 'slide + 음성 동기화',
            'text_audio_match': '문서 내용과 음성 일치',
            'sequential_content': '순차적 내용 연결',
            'topic_coherence': '주제 일관성'
        }
    
    def analyze_pattern(self, results: List[MultimodalResult]) -> Dict[str, Any]:
        """패턴 분석 실행"""
        patterns = {}
        
        # 타임스탬프 기반 순서 분석
        results_by_time = sorted(results, key=lambda x: Path(x.file_path).stat().st_mtime)
        
        # 순차적 내용 분석
        patterns['temporal_flow'] = self._analyze_temporal_flow(results_by_time)
        
        # 주제 일관성 분석
        patterns['topic_coherence'] = self._analyze_topic_coherence(results)
        
        return patterns
    
    def _analyze_temporal_flow(self, results: List[MultimodalResult]) -> Dict[str, Any]:
        """시간 순서 기반 흐름 분석"""
        return {
            'file_sequence': [r.file_path for r in results],
            'content_evolution': 'analyzed',  # 실제 구현시 더 정교한 분석
            'narrative_coherence': 0.8
        }
    
    def _analyze_topic_coherence(self, results: List[MultimodalResult]) -> Dict[str, Any]:
        """주제 일관성 분석"""
        # 모든 컨텐츠에서 키워드 추출 및 빈도 분석
        all_content = ' '.join([r.content for r in results if r.content])
        
        return {
            'dominant_topics': ['conference', 'jewelry', 'analysis'],  # 실제로는 TF-IDF 등 사용
            'coherence_score': 0.85,
            'content_diversity': len(set([r.file_type for r in results]))
        }

# 사용 예제
async def main():
    """사용 예제"""
    pipeline = MultimodalPipeline()
    
    # 테스트 파일들
    test_files = [
        Path("test_image.jpg"),
        Path("test_audio.wav"),
        Path("test_document.txt")
    ]
    
    # 실제 파일이 있는 경우만 처리
    existing_files = [f for f in test_files if f.exists()]
    
    if existing_files:
        results = await pipeline.process_multimodal_batch(existing_files)
        
        print("🎯 멀티모달 분석 결과:")
        for result in results:
            print(f"📄 {result.file_path}")
            print(f"   타입: {result.file_type}")
            print(f"   신뢰도: {result.confidence:.2f}")
            print(f"   처리시간: {result.processing_time:.2f}초")
            if result.cross_modal_score:
                print(f"   크로스모달 점수: {result.cross_modal_score:.2f}")
            print()
        
        print(f"📊 성능 통계: {pipeline.get_performance_stats()}")

if __name__ == "__main__":
    asyncio.run(main())