#!/usr/bin/env python3
"""
최적화된 실제 분석 엔진
메모리 누수 방지, 모델 캐싱, 청크 처리 통합
"""

import os
import time
import logging
import gc
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

# 최적화 시스템들 import
from .memory_cleanup_manager import get_global_memory_manager, register_temp_file, emergency_cleanup
from .optimized_model_loader import get_optimized_model_loader, load_whisper, load_easyocr, load_transformers
from .chunk_processor_optimized import get_global_chunk_processor, ChunkType, process_audio_chunked, process_image_chunked

# 기존 import들
from utils.logger import get_logger

class OptimizedRealAnalysisEngine:
    """최적화된 실제 분석 엔진"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # 최적화 시스템들 초기화
        self.memory_manager = get_global_memory_manager()
        self.model_loader = get_optimized_model_loader()
        self.chunk_processor = get_global_chunk_processor()
        
        # 진행 상황 콜백 등록
        self.chunk_processor.add_progress_callback(self._chunk_progress_callback)
        
        # 분석 통계
        self.analysis_stats = {
            "total_files": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_processing_time": 0,
            "memory_peaks": [],
            "chunk_processing_times": []
        }
        
        self.logger.info("✅ 최적화된 분석 엔진 초기화 완료")
    
    def _chunk_progress_callback(self, current: int, total: int, message: str = "") -> None:
        """청크 처리 진행 상황 콜백"""
        progress = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"청크 진행: {current}/{total} ({progress:.1f}%) - {message}")
    
    def analyze_audio_file_optimized(self, file_path: str, language: str = "korean") -> Dict[str, Any]:
        """최적화된 오디오 파일 분석"""
        start_time = time.time()
        self.logger.info(f"🎵 최적화된 오디오 분석 시작: {file_path}")
        
        try:
            # 메모리 상태 확인
            before_memory = self.memory_manager.get_memory_usage()
            
            # 청크 단위 처리를 위한 프로세서 함수
            def process_audio_chunk(chunk_file: str, chunk_info) -> Dict[str, Any]:
                """오디오 청크 처리 함수"""
                try:
                    # Whisper 모델 로딩 (캐싱됨)
                    model = load_whisper(model_name="base", device="cpu")
                    
                    # 음성 텍스트 변환
                    result = model.transcribe(chunk_file, language=language)
                    
                    return {
                        "text": result.get("text", ""),
                        "segments": result.get("segments", []),
                        "chunk_index": chunk_info.index,
                        "start_time": chunk_info.start_time,
                        "end_time": chunk_info.end_time
                    }
                except Exception as e:
                    self.logger.error(f"오디오 청크 처리 실패: {e}")
                    return {"text": "", "error": str(e)}
            
            # 결과 병합 함수
            def merge_audio_results(chunk_results: List[Dict]) -> Dict[str, Any]:
                """오디오 분석 결과 병합"""
                full_text = ""
                all_segments = []
                
                # 청크 순서대로 정렬
                sorted_results = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
                
                for result in sorted_results:
                    if result.get("text"):
                        full_text += " " + result["text"]
                    if result.get("segments"):
                        all_segments.extend(result["segments"])
                
                return {
                    "text": full_text.strip(),
                    "segments": all_segments,
                    "total_chunks": len(sorted_results)
                }
            
            # 청크 단위 처리 실행
            result = process_audio_chunked(
                file_path=file_path,
                processor_func=process_audio_chunk,
                merger_func=merge_audio_results
            )
            
            # 후처리 및 통계
            processing_time = time.time() - start_time
            after_memory = self.memory_manager.get_memory_usage()
            
            # 통계 업데이트
            self.analysis_stats["total_files"] += 1
            self.analysis_stats["total_processing_time"] += processing_time
            self.analysis_stats["memory_peaks"].append(after_memory["rss_mb"])
            
            if result["success"]:
                self.analysis_stats["successful_analyses"] += 1
                self.logger.info(
                    f"✅ 오디오 분석 완료: {file_path} "
                    f"({processing_time:.2f}s, "
                    f"{result['successful_chunks']}/{result['total_chunks']} 청크, "
                    f"메모리: {before_memory['rss_mb']:.1f}→{after_memory['rss_mb']:.1f}MB)"
                )
            else:
                self.analysis_stats["failed_analyses"] += 1
                self.logger.error(f"❌ 오디오 분석 실패: {file_path}")
            
            return {
                "success": result["success"],
                "file_path": file_path,
                "file_type": "audio",
                "text": result["result"]["text"] if result["result"] else "",
                "segments": result["result"]["segments"] if result["result"] else [],
                "processing_time": processing_time,
                "chunk_stats": {
                    "total_chunks": result["total_chunks"],
                    "successful_chunks": result["successful_chunks"],
                    "success_rate": result["success_rate"]
                },
                "memory_usage": {
                    "before_mb": before_memory["rss_mb"],
                    "after_mb": after_memory["rss_mb"],
                    "peak_mb": max(before_memory["rss_mb"], after_memory["rss_mb"])
                },
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.analysis_stats["failed_analyses"] += 1
            
            self.logger.error(f"❌ 오디오 분석 전체 실패: {file_path} - {e}")
            
            return {
                "success": False,
                "file_path": file_path,
                "file_type": "audio",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def analyze_image_file_optimized(self, file_path: str, languages: List[str] = ['ko', 'en']) -> Dict[str, Any]:
        """최적화된 이미지 파일 분석"""
        start_time = time.time()
        self.logger.info(f"🖼️ 최적화된 이미지 분석 시작: {file_path}")
        
        try:
            # 메모리 상태 확인
            before_memory = self.memory_manager.get_memory_usage()
            
            # 이미지는 보통 단일 파일로 처리하지만, 대용량인 경우 청크 처리
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb > 50:  # 50MB 이상인 경우 청크 처리
                return self._analyze_large_image_chunked(file_path, languages)
            else:
                return self._analyze_single_image(file_path, languages)
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.analysis_stats["failed_analyses"] += 1
            
            self.logger.error(f"❌ 이미지 분석 전체 실패: {file_path} - {e}")
            
            return {
                "success": False,
                "file_path": file_path,
                "file_type": "image",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _analyze_single_image(self, file_path: str, languages: List[str]) -> Dict[str, Any]:
        """단일 이미지 분석"""
        start_time = time.time()
        
        try:
            # EasyOCR 모델 로딩 (캐싱됨)
            reader = load_easyocr(lang_list=languages, gpu=False)
            
            # OCR 실행
            results = reader.readtext(file_path)
            
            # 결과 정리
            extracted_texts = []
            confidence_scores = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # 30% 이상 신뢰도만 포함
                    extracted_texts.append(text)
                    confidence_scores.append(confidence)
            
            full_text = " ".join(extracted_texts)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self.analysis_stats["total_files"] += 1
            self.analysis_stats["successful_analyses"] += 1
            self.analysis_stats["total_processing_time"] += processing_time
            
            self.logger.info(f"✅ 이미지 분석 완료: {file_path} ({processing_time:.2f}s)")
            
            return {
                "success": True,
                "file_path": file_path,
                "file_type": "image",
                "text": full_text,
                "ocr_results": results,
                "text_blocks": extracted_texts,
                "confidence_scores": confidence_scores,
                "average_confidence": avg_confidence,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ 단일 이미지 분석 실패: {file_path} - {e}")
            
            return {
                "success": False,
                "file_path": file_path,
                "file_type": "image",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _analyze_large_image_chunked(self, file_path: str, languages: List[str]) -> Dict[str, Any]:
        """대용량 이미지 청크 분석"""
        start_time = time.time()
        
        try:
            # 청크 단위 처리를 위한 프로세서 함수
            def process_image_chunk(chunk_file: str, chunk_info) -> Dict[str, Any]:
                """이미지 청크 처리 함수"""
                try:
                    # EasyOCR 모델 로딩 (캐싱됨)
                    reader = load_easyocr(lang_list=languages, gpu=False)
                    
                    # OCR 실행
                    results = reader.readtext(chunk_file)
                    
                    # 결과 정리
                    extracted_texts = []
                    confidence_scores = []
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3:
                            extracted_texts.append(text)
                            confidence_scores.append(confidence)
                    
                    return {
                        "texts": extracted_texts,
                        "confidences": confidence_scores,
                        "raw_results": results,
                        "chunk_index": chunk_info.index
                    }
                except Exception as e:
                    self.logger.error(f"이미지 청크 처리 실패: {e}")
                    return {"texts": [], "error": str(e)}
            
            # 결과 병합 함수
            def merge_image_results(chunk_results: List[Dict]) -> Dict[str, Any]:
                """이미지 분석 결과 병합"""
                all_texts = []
                all_confidences = []
                all_raw_results = []
                
                # 청크 순서대로 정렬
                sorted_results = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
                
                for result in sorted_results:
                    if result.get("texts"):
                        all_texts.extend(result["texts"])
                    if result.get("confidences"):
                        all_confidences.extend(result["confidences"])
                    if result.get("raw_results"):
                        all_raw_results.extend(result["raw_results"])
                
                full_text = " ".join(all_texts)
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                
                return {
                    "text": full_text,
                    "text_blocks": all_texts,
                    "confidence_scores": all_confidences,
                    "average_confidence": avg_confidence,
                    "ocr_results": all_raw_results,
                    "total_chunks": len(sorted_results)
                }
            
            # 청크 단위 처리 실행
            result = process_image_chunked(
                file_path=file_path,
                processor_func=process_image_chunk,
                merger_func=merge_image_results
            )
            
            processing_time = time.time() - start_time
            
            # 통계 업데이트
            self.analysis_stats["total_files"] += 1
            self.analysis_stats["total_processing_time"] += processing_time
            
            if result["success"]:
                self.analysis_stats["successful_analyses"] += 1
                self.logger.info(f"✅ 대용량 이미지 분석 완료: {file_path} ({processing_time:.2f}s)")
            else:
                self.analysis_stats["failed_analyses"] += 1
                
            return {
                "success": result["success"],
                "file_path": file_path,
                "file_type": "image",
                "text": result["result"]["text"] if result["result"] else "",
                "text_blocks": result["result"]["text_blocks"] if result["result"] else [],
                "confidence_scores": result["result"]["confidence_scores"] if result["result"] else [],
                "average_confidence": result["result"]["average_confidence"] if result["result"] else 0,
                "ocr_results": result["result"]["ocr_results"] if result["result"] else [],
                "processing_time": processing_time,
                "chunk_stats": {
                    "total_chunks": result["total_chunks"],
                    "successful_chunks": result["successful_chunks"],
                    "success_rate": result["success_rate"]
                },
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.analysis_stats["failed_analyses"] += 1
            
            self.logger.error(f"❌ 대용량 이미지 분석 실패: {file_path} - {e}")
            
            return {
                "success": False,
                "file_path": file_path,
                "file_type": "image",
                "error": str(e),
                "processing_time": processing_time
            }
    
    def create_comprehensive_summary(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """종합 요약 생성"""
        start_time = time.time()
        
        try:
            # 모든 텍스트 결합
            all_texts = []
            successful_files = []
            failed_files = []
            
            for result in analysis_results:
                if result.get("success", False):
                    successful_files.append(result["file_path"])
                    if result.get("text"):
                        all_texts.append(result["text"])
                else:
                    failed_files.append(result["file_path"])
            
            combined_text = " ".join(all_texts)
            
            if not combined_text.strip():
                return {
                    "success": False,
                    "error": "분석된 텍스트가 없습니다.",
                    "summary": "",
                    "key_points": [],
                    "file_stats": {
                        "total_files": len(analysis_results),
                        "successful_files": len(successful_files),
                        "failed_files": len(failed_files)
                    }
                }
            
            # Transformers 모델로 요약 생성
            try:
                summarizer = load_transformers("facebook/bart-large-cnn", "summarization")
                
                # 텍스트가 너무 긴 경우 청크 단위로 처리
                max_length = 1024
                if len(combined_text) > max_length:
                    chunks = [combined_text[i:i+max_length] for i in range(0, len(combined_text), max_length)]
                    chunk_summaries = []
                    
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:  # 최소 길이 체크
                            try:
                                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                                chunk_summaries.append(summary[0]['summary_text'])
                            except:
                                continue
                    
                    # 청크 요약들을 다시 요약
                    if chunk_summaries:
                        final_text = " ".join(chunk_summaries)
                        if len(final_text) > max_length:
                            final_text = final_text[:max_length]
                        
                        final_summary = summarizer(final_text, max_length=200, min_length=50, do_sample=False)
                        summary_text = final_summary[0]['summary_text']
                    else:
                        summary_text = "요약 생성에 실패했습니다."
                else:
                    summary = summarizer(combined_text, max_length=200, min_length=50, do_sample=False)
                    summary_text = summary[0]['summary_text']
                
            except Exception as e:
                self.logger.warning(f"Transformers 요약 실패, 기본 요약 사용: {e}")
                # 간단한 텍스트 추출 요약
                sentences = combined_text.split('.')
                summary_text = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else combined_text
            
            # 키 포인트 추출 (간단한 방식)
            sentences = [s.strip() for s in combined_text.split('.') if len(s.strip()) > 10]
            key_points = sentences[:5]  # 상위 5개 문장
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"✅ 종합 요약 생성 완료 ({processing_time:.2f}s)")
            
            return {
                "success": True,
                "summary": summary_text,
                "key_points": key_points,
                "full_text": combined_text,
                "processing_time": processing_time,
                "file_stats": {
                    "total_files": len(analysis_results),
                    "successful_files": len(successful_files),
                    "failed_files": len(failed_files),
                    "success_rate": len(successful_files) / len(analysis_results) if analysis_results else 0
                },
                "successful_files": successful_files,
                "failed_files": failed_files
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ 종합 요약 생성 실패: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "summary": "",
                "key_points": []
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        memory_status = self.memory_manager.get_status()
        model_cache_info = self.model_loader.get_cache_info()
        
        return {
            "analysis_stats": self.analysis_stats,
            "memory_status": memory_status,
            "model_cache_info": model_cache_info,
            "optimization_features": {
                "memory_management": True,
                "model_caching": True,
                "chunk_processing": True,
                "auto_cleanup": memory_status["cleanup_running"]
            }
        }
    
    def cleanup_system(self) -> Dict[str, Any]:
        """시스템 정리"""
        self.logger.info("🧹 시스템 정리 시작")
        
        # 메모리 응급 정리
        memory_result = emergency_cleanup()
        
        # 모델 캐시 정리
        model_cleared = self.model_loader.clear_cache()
        
        # 청크 프로세서는 자동으로 정리됨 (임시 파일들)
        
        result = {
            "memory_cleanup": memory_result,
            "model_cache_cleared": model_cleared,
            "cleanup_time": datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ 시스템 정리 완료: {result}")
        return result

# 전역 인스턴스
global_optimized_analysis_engine = OptimizedRealAnalysisEngine()

def get_optimized_analysis_engine() -> OptimizedRealAnalysisEngine:
    """최적화된 분석 엔진 반환"""
    return global_optimized_analysis_engine

# 편의 함수들
def analyze_file_optimized(file_path: str, file_type: str = "auto") -> Dict[str, Any]:
    """파일 분석 (최적화) - 편의 함수"""
    engine = get_optimized_analysis_engine()
    
    if file_type == "auto":
        # 파일 확장자로 타입 결정
        ext = Path(file_path).suffix.lower()
        if ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            file_type = "audio"
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            file_type = "image"
        else:
            return {"success": False, "error": f"지원하지 않는 파일 형식: {ext}"}
    
    if file_type == "audio":
        return engine.analyze_audio_file_optimized(file_path)
    elif file_type == "image":
        return engine.analyze_image_file_optimized(file_path)
    else:
        return {"success": False, "error": f"지원하지 않는 파일 타입: {file_type}"}