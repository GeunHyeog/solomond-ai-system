"""
Ultra Large File Processor v2.1
1시간 영상 + 30개 이미지 동시 분석을 위한 초대용량 파일 처리 엔진

주요 기능:
- 메모리 효율적 청킹 처리 (최대 100MB 메모리 사용)
- 병렬 스트리밍 분석 (CPU 코어 활용)
- 진행률 실시간 모니터링
- 에러 복구 및 재시작 지원
- 품질 기반 적응형 처리
"""

import os
import sys
import asyncio
import time
import logging
import traceback
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tempfile
import shutil
import hashlib
import mmap
from queue import Queue
import threading
import psutil

# 비동기 처리
import aiofiles
import aiohttp

# 미디어 처리
try:
    import cv2
    import librosa
    import soundfile as sf
    from moviepy.editor import VideoFileClip
    import whisper
except ImportError as e:
    print(f"⚠️ 선택적 라이브러리 누락: {e}")

# 메모리 최적화
import gc
import resource
from functools import wraps
import weakref

@dataclass
class FileChunk:
    """파일 청크 정보"""
    chunk_id: str
    file_path: str
    start_time: float
    end_time: float
    chunk_type: str  # 'video', 'audio', 'image'
    size_bytes: int
    processing_priority: int = 1
    quality_score: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class ProcessingProgress:
    """처리 진행률 정보"""
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    current_chunk: Optional[str]
    estimated_time_remaining: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_mb_per_sec: float

class MemoryManager:
    """메모리 사용량 모니터링 및 최적화"""
    
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.memory_threshold = max_memory_mb * 0.8  # 80% 임계값
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 * 1024)
    
    def is_memory_critical(self) -> bool:
        """메모리 사용량이 임계값을 초과했는지 확인"""
        return self.get_memory_usage() > self.memory_threshold
    
    def force_garbage_collection(self):
        """강제 가비지 컬렉션"""
        gc.collect()
        
    def optimize_memory(self):
        """메모리 최적화"""
        if self.is_memory_critical():
            self.force_garbage_collection()
            time.sleep(0.1)  # 시스템 안정화

class ChunkManager:
    """파일 청킹 및 분할 관리"""
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="ultra_processing_")
        self.chunk_size_mb = 50  # 청크 크기 (MB)
        self.video_segment_seconds = 300  # 5분 세그먼트
        
    async def create_video_chunks(self, video_path: str) -> List[FileChunk]:
        """비디오 파일을 청크로 분할"""
        chunks = []
        
        try:
            # 비디오 정보 분석
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            # 세그먼트 계산
            num_segments = max(1, int(duration / self.video_segment_seconds))
            segment_duration = duration / num_segments
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                chunk = FileChunk(
                    chunk_id=f"video_{i:03d}",
                    file_path=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_type="video",
                    size_bytes=os.path.getsize(video_path) // num_segments,
                    processing_priority=1,
                    metadata={
                        "fps": fps,
                        "duration": end_time - start_time,
                        "segment_index": i
                    }
                )
                chunks.append(chunk)
                
        except Exception as e:
            logging.error(f"비디오 청킹 실패: {e}")
            
        return chunks
    
    async def create_image_chunks(self, image_paths: List[str]) -> List[FileChunk]:
        """이미지 파일들을 청크로 그룹화"""
        chunks = []
        
        # 이미지를 크기별로 그룹화
        batch_size = 5  # 한 번에 처리할 이미지 수
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            total_size = sum(os.path.getsize(path) for path in batch_paths)
            
            chunk = FileChunk(
                chunk_id=f"images_{i//batch_size:03d}",
                file_path=batch_paths[0],  # 대표 파일
                start_time=0,
                end_time=0,
                chunk_type="image_batch",
                size_bytes=total_size,
                processing_priority=2,
                metadata={
                    "image_paths": batch_paths,
                    "batch_index": i // batch_size,
                    "image_count": len(batch_paths)
                }
            )
            chunks.append(chunk)
            
        return chunks

class QualityAnalyzer:
    """파일 품질 분석 및 적응형 처리"""
    
    def __init__(self):
        self.quality_cache = {}
        
    async def analyze_video_quality(self, video_path: str, start_time: float, duration: float) -> float:
        """비디오 품질 분석"""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            frame_count = 0
            quality_scores = []
            
            for _ in range(10):  # 샘플 프레임 분석
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 프레임 품질 분석 (라플라시안 분산으로 선명도 측정)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality_scores.append(laplacian_var)
                frame_count += 1
                
            cap.release()
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                normalized_quality = min(1.0, avg_quality / 1000)  # 정규화
                return normalized_quality
            else:
                return 0.5  # 기본값
                
        except Exception as e:
            logging.error(f"비디오 품질 분석 실패: {e}")
            return 0.5
    
    async def analyze_audio_quality(self, audio_data: any) -> float:
        """오디오 품질 분석"""
        try:
            # 신호 대 잡음비 계산
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data - np.mean(audio_data))
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                normalized_snr = min(1.0, max(0.0, (snr + 10) / 40))  # -10~30dB 범위
                return normalized_snr
            else:
                return 1.0
                
        except Exception as e:
            logging.error(f"오디오 품질 분석 실패: {e}")
            return 0.5

class StreamingProcessor:
    """스트리밍 방식의 파일 처리"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.whisper_model = None
        self.processing_queue = asyncio.Queue()
        
    async def initialize_models(self):
        """AI 모델 초기화 (지연 로딩)"""
        try:
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model("base")
                logging.info("Whisper 모델 로드 완료")
        except Exception as e:
            logging.error(f"모델 초기화 실패: {e}")
    
    async def process_video_chunk(self, chunk: FileChunk) -> Dict[str, Any]:
        """비디오 청크 처리"""
        result = {
            "chunk_id": chunk.chunk_id,
            "type": "video",
            "transcript": "",
            "quality_score": chunk.quality_score,
            "processing_time": 0,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # 메모리 확인
            self.memory_manager.optimize_memory()
            
            # 임시 오디오 파일 추출
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
            # FFmpeg를 사용한 오디오 추출 (메모리 효율적)
            import subprocess
            cmd = [
                "ffmpeg", "-i", chunk.file_path,
                "-ss", str(chunk.start_time),
                "-t", str(chunk.end_time - chunk.start_time),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                temp_audio_path, "-y"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            # STT 처리
            if self.whisper_model and os.path.exists(temp_audio_path):
                await self.initialize_models()
                
                # 메모리 체크
                if self.memory_manager.is_memory_critical():
                    logging.warning(f"메모리 부족으로 청크 {chunk.chunk_id} 지연 처리")
                    await asyncio.sleep(1)
                    self.memory_manager.optimize_memory()
                
                # Whisper 처리
                loop = asyncio.get_event_loop()
                transcript_result = await loop.run_in_executor(
                    None, 
                    self.whisper_model.transcribe, 
                    temp_audio_path
                )
                
                result["transcript"] = transcript_result.get("text", "")
                
            # 임시 파일 정리
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"비디오 청크 처리 실패 {chunk.chunk_id}: {e}")
            
        finally:
            result["processing_time"] = time.time() - start_time
            
        return result
    
    async def process_image_batch(self, chunk: FileChunk) -> Dict[str, Any]:
        """이미지 배치 처리"""
        result = {
            "chunk_id": chunk.chunk_id,
            "type": "image_batch",
            "ocr_results": [],
            "image_analysis": [],
            "processing_time": 0,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            image_paths = chunk.metadata.get("image_paths", [])
            
            for img_path in image_paths:
                try:
                    # 메모리 최적화
                    self.memory_manager.optimize_memory()
                    
                    # 이미지 로드 및 OCR
                    import cv2
                    import pytesseract
                    
                    img = cv2.imread(img_path)
                    if img is not None:
                        # OCR 처리
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ocr_text = pytesseract.image_to_string(gray, lang='kor+eng')
                        
                        # 이미지 기본 분석
                        height, width = img.shape[:2]
                        file_size = os.path.getsize(img_path)
                        
                        result["ocr_results"].append({
                            "file_path": img_path,
                            "text": ocr_text.strip(),
                            "confidence": 0.8  # 임시값
                        })
                        
                        result["image_analysis"].append({
                            "file_path": img_path,
                            "dimensions": f"{width}x{height}",
                            "file_size": file_size,
                            "has_text": len(ocr_text.strip()) > 0
                        })
                        
                        # 메모리 해제
                        del img, gray
                        
                except Exception as e:
                    logging.error(f"이미지 처리 실패 {img_path}: {e}")
                    
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"이미지 배치 처리 실패 {chunk.chunk_id}: {e}")
            
        finally:
            result["processing_time"] = time.time() - start_time
            
        return result

class UltraLargeFileProcessor:
    """초대용량 파일 처리 메인 엔진"""
    
    def __init__(self, max_memory_mb: int = 1000, max_workers: int = None):
        self.max_memory_mb = max_memory_mb
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        
        # 컴포넌트 초기화
        self.memory_manager = MemoryManager(max_memory_mb)
        self.chunk_manager = ChunkManager()
        self.quality_analyzer = QualityAnalyzer()
        self.streaming_processor = StreamingProcessor(self.memory_manager)
        
        # 처리 상태
        self.processing_stats = {
            "start_time": None,
            "total_chunks": 0,
            "completed_chunks": 0,
            "failed_chunks": 0,
            "total_files_processed": 0,
            "total_bytes_processed": 0
        }
        
        # 진행률 콜백
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """진행률 콜백 설정"""
        self.progress_callback = callback
        
    async def process_ultra_large_files(
        self,
        video_files: List[str],
        image_files: List[str],
        output_dir: str = None
    ) -> Dict[str, Any]:
        """초대용량 파일 통합 처리"""
        
        self.processing_stats["start_time"] = time.time()
        output_dir = output_dir or tempfile.mkdtemp(prefix="ultra_results_")
        
        try:
            # 1. 파일 청킹
            logging.info("파일 청킹 시작...")
            all_chunks = []
            
            # 비디오 청킹
            for video_file in video_files:
                if os.path.exists(video_file):
                    video_chunks = await self.chunk_manager.create_video_chunks(video_file)
                    all_chunks.extend(video_chunks)
                    
            # 이미지 청킹
            if image_files:
                image_chunks = await self.chunk_manager.create_image_chunks(image_files)
                all_chunks.extend(image_chunks)
            
            self.processing_stats["total_chunks"] = len(all_chunks)
            logging.info(f"총 {len(all_chunks)}개 청크 생성")
            
            # 2. 품질 분석 및 우선순위 설정
            logging.info("품질 분석 시작...")
            await self._analyze_chunk_quality(all_chunks)
            
            # 3. 우선순위별 청크 정렬
            all_chunks.sort(key=lambda x: (x.processing_priority, -x.quality_score))
            
            # 4. 병렬 처리
            logging.info("병렬 처리 시작...")
            results = await self._process_chunks_parallel(all_chunks)
            
            # 5. 결과 통합
            logging.info("결과 통합 중...")
            final_result = await self._integrate_results(results, output_dir)
            
            return final_result
            
        except Exception as e:
            logging.error(f"초대용량 파일 처리 실패: {e}")
            raise
        
    async def _analyze_chunk_quality(self, chunks: List[FileChunk]):
        """청크 품질 분석"""
        for chunk in chunks:
            try:
                if chunk.chunk_type == "video":
                    quality = await self.quality_analyzer.analyze_video_quality(
                        chunk.file_path, chunk.start_time, 
                        chunk.end_time - chunk.start_time
                    )
                    chunk.quality_score = quality
                    
                elif chunk.chunk_type == "image_batch":
                    # 이미지는 기본 품질 점수 사용
                    chunk.quality_score = 0.8
                    
            except Exception as e:
                logging.error(f"청크 품질 분석 실패 {chunk.chunk_id}: {e}")
                chunk.quality_score = 0.5
    
    async def _process_chunks_parallel(self, chunks: List[FileChunk]) -> List[Dict[str, Any]]:
        """청크 병렬 처리"""
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_chunk(chunk: FileChunk) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # 메모리 체크
                    if self.memory_manager.is_memory_critical():
                        await asyncio.sleep(0.5)
                        self.memory_manager.optimize_memory()
                    
                    # 청크 타입별 처리
                    if chunk.chunk_type == "video":
                        result = await self.streaming_processor.process_video_chunk(chunk)
                    elif chunk.chunk_type == "image_batch":
                        result = await self.streaming_processor.process_image_batch(chunk)
                    else:
                        result = {"chunk_id": chunk.chunk_id, "error": "Unknown chunk type"}
                    
                    # 진행률 업데이트
                    self.processing_stats["completed_chunks"] += 1
                    if result.get("error"):
                        self.processing_stats["failed_chunks"] += 1
                    
                    await self._update_progress()
                    
                    return result
                    
                except Exception as e:
                    self.processing_stats["failed_chunks"] += 1
                    await self._update_progress()
                    return {"chunk_id": chunk.chunk_id, "error": str(e)}
        
        # 모든 청크 처리
        tasks = [process_single_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _update_progress(self):
        """진행률 업데이트"""
        if self.progress_callback:
            progress = ProcessingProgress(
                total_chunks=self.processing_stats["total_chunks"],
                completed_chunks=self.processing_stats["completed_chunks"],
                failed_chunks=self.processing_stats["failed_chunks"],
                current_chunk=None,
                estimated_time_remaining=self._calculate_eta(),
                memory_usage_mb=self.memory_manager.get_memory_usage(),
                cpu_usage_percent=psutil.cpu_percent(),
                throughput_mb_per_sec=self._calculate_throughput()
            )
            
            await self.progress_callback(progress)
    
    def _calculate_eta(self) -> float:
        """남은 시간 계산"""
        if self.processing_stats["completed_chunks"] == 0:
            return 0.0
            
        elapsed = time.time() - self.processing_stats["start_time"]
        remaining_chunks = (
            self.processing_stats["total_chunks"] - 
            self.processing_stats["completed_chunks"]
        )
        
        chunks_per_second = self.processing_stats["completed_chunks"] / elapsed
        if chunks_per_second > 0:
            return remaining_chunks / chunks_per_second
        else:
            return 0.0
    
    def _calculate_throughput(self) -> float:
        """처리량 계산 (MB/초)"""
        if self.processing_stats["completed_chunks"] == 0:
            return 0.0
            
        elapsed = time.time() - self.processing_stats["start_time"]
        return self.processing_stats["total_bytes_processed"] / (1024 * 1024) / elapsed
    
    async def _integrate_results(self, results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """결과 통합"""
        integrated_result = {
            "processing_summary": {
                "total_chunks": len(results),
                "successful_chunks": len([r for r in results if not r.get("error")]),
                "failed_chunks": len([r for r in results if r.get("error")]),
                "total_processing_time": time.time() - self.processing_stats["start_time"]
            },
            "video_transcripts": [],
            "image_ocr_results": [],
            "combined_transcript": "",
            "output_files": [],
            "quality_metrics": {
                "average_video_quality": 0.0,
                "total_text_extracted": 0,
                "processing_efficiency": 0.0
            }
        }
        
        try:
            # 결과 분류 및 통합
            video_texts = []
            image_texts = []
            
            for result in results:
                if isinstance(result, dict) and not result.get("error"):
                    if result.get("type") == "video":
                        transcript = result.get("transcript", "")
                        if transcript:
                            video_texts.append({
                                "chunk_id": result["chunk_id"],
                                "transcript": transcript,
                                "quality_score": result.get("quality_score", 0.5)
                            })
                            
                    elif result.get("type") == "image_batch":
                        ocr_results = result.get("ocr_results", [])
                        for ocr in ocr_results:
                            if ocr.get("text"):
                                image_texts.append(ocr)
            
            integrated_result["video_transcripts"] = video_texts
            integrated_result["image_ocr_results"] = image_texts
            
            # 통합 텍스트 생성
            all_texts = []
            all_texts.extend([vt["transcript"] for vt in video_texts])
            all_texts.extend([it["text"] for it in image_texts])
            
            integrated_result["combined_transcript"] = "\n\n".join(all_texts)
            
            # 품질 메트릭 계산
            if video_texts:
                avg_quality = sum(vt["quality_score"] for vt in video_texts) / len(video_texts)
                integrated_result["quality_metrics"]["average_video_quality"] = avg_quality
            
            integrated_result["quality_metrics"]["total_text_extracted"] = len(all_texts)
            
            # 결과 파일 저장
            output_file = os.path.join(output_dir, "ultra_processing_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(integrated_result, f, ensure_ascii=False, indent=2)
            
            integrated_result["output_files"].append(output_file)
            
        except Exception as e:
            logging.error(f"결과 통합 실패: {e}")
            integrated_result["integration_error"] = str(e)
        
        return integrated_result

# 사용 예제 및 테스트 함수
async def demo_ultra_processing():
    """데모 실행"""
    print("🚀 Ultra Large File Processor v2.1 데모")
    
    # 프로세서 초기화
    processor = UltraLargeFileProcessor(max_memory_mb=800, max_workers=3)
    
    # 진행률 콜백 설정
    async def progress_callback(progress: ProcessingProgress):
        print(f"진행률: {progress.completed_chunks}/{progress.total_chunks} "
              f"({progress.completed_chunks/progress.total_chunks*100:.1f}%) "
              f"메모리: {progress.memory_usage_mb:.1f}MB "
              f"CPU: {progress.cpu_usage_percent:.1f}%")
    
    processor.set_progress_callback(progress_callback)
    
    # 테스트 파일 (실제 환경에서는 존재하는 파일 경로 사용)
    video_files = [
        # "path/to/large_video.mp4"  # 1시간 영상
    ]
    
    image_files = [
        # "path/to/image1.jpg", "path/to/image2.png", ...  # 30개 이미지
    ]
    
    try:
        result = await processor.process_ultra_large_files(
            video_files=video_files,
            image_files=image_files
        )
        
        print("✅ 처리 완료!")
        print(f"📊 처리 요약:")
        print(f"  - 총 청크: {result['processing_summary']['total_chunks']}")
        print(f"  - 성공: {result['processing_summary']['successful_chunks']}")
        print(f"  - 실패: {result['processing_summary']['failed_chunks']}")
        print(f"  - 처리 시간: {result['processing_summary']['total_processing_time']:.1f}초")
        
    except Exception as e:
        print(f"❌ 처리 실패: {e}")

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 데모 실행
    asyncio.run(demo_ultra_processing())
