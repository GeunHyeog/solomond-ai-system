"""
솔로몬드 AI 시스템 - 고용량 다중분석 전용 LLM 요약 엔진
GEMMA 기반으로 5GB 파일 50개 동시 처리 최적화

특징:
- 청크 단위 스트리밍 처리 (메모리 효율성)
- 다중 소스 통합 요약 (크로스 검증)
- 주얼리 도메인 특화 분석
- 대용량 배치 처리 최적화
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import gc
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import re

# 외부 라이브러리 (실제 구현 시 필요)
try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    logging.warning("GEMMA 라이브러리 없음. 모의 모드로 실행")

class ProcessingMode(Enum):
    STREAMING = "streaming"  # 스트리밍 처리 (대용량)
    BATCH = "batch"         # 배치 처리 (중간 용량)
    MEMORY = "memory"       # 메모리 처리 (소용량)

class SummaryType(Enum):
    EXECUTIVE = "executive"     # 경영진 요약
    TECHNICAL = "technical"     # 기술적 요약
    BUSINESS = "business"       # 비즈니스 요약
    COMPREHENSIVE = "comprehensive"  # 종합 요약

@dataclass
class ChunkInfo:
    """텍스트 청크 정보"""
    chunk_id: str
    source_file: str
    source_type: str  # audio, video, document, image
    text: str
    token_count: int
    jewelry_terms: List[str] = field(default_factory=list)
    confidence: float = 0.0
    processing_time: float = 0.0

@dataclass
class SummaryRequest:
    """요약 요청 정보"""
    session_id: str
    title: str
    chunks: List[ChunkInfo]
    summary_type: SummaryType
    max_length: int = 2000
    focus_keywords: List[str] = field(default_factory=list)
    language: str = "ko"
    priority_sources: List[str] = field(default_factory=list)

class AdvancedLLMSummarizer:
    """고용량 다중분석 전용 LLM 요약 엔진"""
    
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_context_length = 4096
        self.chunk_size = 1024
        self.overlap_size = 128
        
        # 주얼리 특화 프롬프트 템플릿
        self.jewelry_prompts = {
            "executive": """다음 주얼리 업계 정보를 경영진 관점에서 요약해주세요:
- 핵심 비즈니스 인사이트
- 시장 기회 및 위험
- 실행 가능한 전략적 권장사항
- 재무적 영향 분석

내용: {content}

요약 (한국어, 500자 이내):""",
            
            "technical": """다음 주얼리 기술 정보를 전문가 관점에서 요약해주세요:
- 보석학적 특징 및 품질 분석
- 가공 기술 및 처리 방법
- 감정 및 인증 관련 사항
- 기술적 트렌드 및 혁신

내용: {content}

기술 요약 (한국어, 800자 이내):""",
            
            "business": """다음 주얼리 비즈니스 정보를 실무진 관점에서 요약해주세요:
- 가격 동향 및 시장 분석
- 거래 조건 및 상거래 정보
- 고객 동향 및 선호도 변화
- 영업 전략 및 마케팅 인사이트

내용: {content}

비즈니스 요약 (한국어, 600자 이내):""",
            
            "comprehensive": """다음 주얼리 업계 종합 정보를 전체적으로 요약해주세요:
- 시장 현황 및 전망
- 기술적 특징 및 품질 분석
- 비즈니스 기회 및 전략
- 업계 트렌드 및 미래 방향

내용: {content}

종합 요약 (한국어, 1200자 이내):"""
        }
        
        # 메모리 사용량 추적
        self.memory_usage = {
            "peak_memory": 0,
            "current_memory": 0,
            "chunks_processed": 0,
            "gc_collections": 0
        }
        
        logging.info(f"고급 LLM 요약 엔진 초기화 (모델: {model_name})")
    
    async def initialize_model(self):
        """모델 초기화 (지연 로딩)"""
        if self.model is not None:
            return
            
        try:
            if GEMMA_AVAILABLE:
                print("🤖 GEMMA 모델 로딩 중...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.device == "cuda" else None
                )
                print("✅ GEMMA 모델 로딩 완료")
            else:
                print("⚠️ GEMMA 모의 모드로 실행")
                
        except Exception as e:
            logging.error(f"모델 초기화 실패: {e}")
            print("❌ 모델 로딩 실패, 모의 모드로 전환")
    
    async def process_large_batch(self, 
                                session_data: Dict,
                                processing_mode: ProcessingMode = ProcessingMode.STREAMING) -> Dict:
        """대용량 배치 처리 메인 함수"""
        print(f"🚀 대용량 배치 처리 시작 (모드: {processing_mode.value})")
        
        session_id = session_data.get("session_id", f"batch_{int(time.time())}")
        start_time = time.time()
        
        try:
            # 1. 모델 초기화
            await self.initialize_model()
            
            # 2. 입력 데이터 검증 및 분류
            validated_data = await self._validate_and_classify_inputs(session_data)
            
            # 3. 청크 단위 분할 (메모리 효율성)
            chunks = await self._create_optimized_chunks(validated_data, processing_mode)
            
            # 4. 병렬 처리 최적화
            processed_chunks = await self._process_chunks_parallel(chunks, processing_mode)
            
            # 5. 계층적 요약 생성
            hierarchical_summary = await self._generate_hierarchical_summary(processed_chunks)
            
            # 6. 크로스 검증 및 품질 평가
            quality_assessment = await self._assess_summary_quality(hierarchical_summary, processed_chunks)
            
            # 7. 최종 통합 보고서 생성
            final_report = await self._generate_final_report(
                session_id, 
                hierarchical_summary, 
                quality_assessment,
                processing_mode,
                start_time
            )
            
            # 8. 메모리 정리
            await self._cleanup_memory()
            
            print(f"✅ 대용량 배치 처리 완료 ({time.time() - start_time:.1f}초)")
            return final_report
            
        except Exception as e:
            logging.error(f"대용량 배치 처리 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "processing_mode": processing_mode.value
            }
    
    async def _validate_and_classify_inputs(self, session_data: Dict) -> Dict:
        """입력 데이터 검증 및 분류"""
        print("📋 입력 데이터 검증 중...")
        
        files = session_data.get("files", [])
        total_size = sum(f.get("size_mb", 0) for f in files)
        
        # 용량별 처리 모드 자동 결정
        if total_size > 2000:  # 2GB 이상
            recommended_mode = ProcessingMode.STREAMING
        elif total_size > 500:  # 500MB 이상
            recommended_mode = ProcessingMode.BATCH
        else:
            recommended_mode = ProcessingMode.MEMORY
        
        # 파일 타입별 분류
        classified_files = {
            "audio": [],
            "video": [],
            "documents": [],
            "images": []
        }
        
        for file_info in files:
            file_type = self._detect_file_type(file_info.get("filename", ""))
            if file_type in classified_files:
                classified_files[file_type].append(file_info)
        
        return {
            "total_files": len(files),
            "total_size_mb": total_size,
            "recommended_mode": recommended_mode,
            "classified_files": classified_files,
            "processing_complexity": self._calculate_complexity(files)
        }
    
    async def _create_optimized_chunks(self, validated_data: Dict, mode: ProcessingMode) -> List[ChunkInfo]:
        """최적화된 청크 생성"""
        print(f"🔄 청크 생성 중... (모드: {mode.value})")
        
        chunks = []
        chunk_id_counter = 0
        
        for file_type, files in validated_data["classified_files"].items():
            for file_info in files:
                # 각 파일의 텍스트 추출 (이미 처리된 상태라고 가정)
                text_content = file_info.get("processed_text", "")
                
                if not text_content:
                    continue
                
                # 청크 크기 결정 (모드에 따라 다름)
                chunk_size = self._get_optimal_chunk_size(mode, len(text_content))
                
                # 텍스트를 청크로 분할
                text_chunks = self._split_text_into_chunks(text_content, chunk_size)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk_info = ChunkInfo(
                        chunk_id=f"chunk_{chunk_id_counter:04d}",
                        source_file=file_info.get("filename", "unknown"),
                        source_type=file_type,
                        text=chunk_text,
                        token_count=len(chunk_text.split()),
                        jewelry_terms=self._extract_jewelry_terms(chunk_text),
                        confidence=file_info.get("confidence", 0.8)
                    )
                    chunks.append(chunk_info)
                    chunk_id_counter += 1
        
        print(f"✅ 청크 생성 완료: {len(chunks)}개")
        return chunks
    
    async def _process_chunks_parallel(self, chunks: List[ChunkInfo], mode: ProcessingMode) -> List[ChunkInfo]:
        """병렬 청크 처리"""
        print(f"⚡ 병렬 처리 시작: {len(chunks)}개 청크")
        
        # 동시 처리 수 결정 (모드와 시스템 리소스에 따라)
        max_concurrent = self._get_max_concurrent_tasks(mode)
        
        processed_chunks = []
        
        # 청크를 배치로 나누어 처리
        for i in range(0, len(chunks), max_concurrent):
            batch = chunks[i:i + max_concurrent]
            
            # 배치 내 병렬 처리
            tasks = [self._process_single_chunk(chunk) for chunk in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 수집
            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"청크 처리 오류: {result}")
                    continue
                processed_chunks.append(result)
            
            # 메모리 정리 (중간 단계)
            if i % (max_concurrent * 3) == 0:
                await self._intermediate_memory_cleanup()
        
        print(f"✅ 병렬 처리 완료: {len(processed_chunks)}개")
        return processed_chunks
    
    async def _process_single_chunk(self, chunk: ChunkInfo) -> ChunkInfo:
        """개별 청크 처리"""
        start_time = time.time()
        
        try:
            # 주얼리 용어 강화
            enhanced_text = await self._enhance_jewelry_content(chunk.text)
            
            # 핵심 정보 추출
            key_info = await self._extract_key_information(enhanced_text)
            
            # 요약 생성 (청크 수준)
            chunk_summary = await self._generate_chunk_summary(enhanced_text, chunk.source_type)
            
            # 결과 업데이트
            chunk.text = chunk_summary
            chunk.jewelry_terms = self._extract_jewelry_terms(chunk_summary)
            chunk.processing_time = time.time() - start_time
            
            return chunk
            
        except Exception as e:
            logging.error(f"청크 처리 오류 ({chunk.chunk_id}): {e}")
            chunk.confidence = 0.0
            chunk.processing_time = time.time() - start_time
            return chunk
    
    async def _generate_chunk_summary(self, text: str, source_type: str) -> str:
        """청크 수준 요약 생성"""
        if GEMMA_AVAILABLE and self.model is not None:
            return await self._generate_with_gemma(text, "business")
        else:
            return await self._generate_mock_summary(text, source_type)
    
    async def _generate_with_gemma(self, text: str, summary_type: str) -> str:
        """GEMMA 모델로 요약 생성"""
        try:
            prompt = self.jewelry_prompts[summary_type].format(content=text)
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=self.max_context_length, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = generated_text[len(prompt):].strip()
            
            return summary
            
        