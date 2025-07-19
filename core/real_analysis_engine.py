#!/usr/bin/env python3
"""
실제 분석 엔진 - 가짜 분석을 실제 분석으로 교체
Whisper STT + EasyOCR + 무료 AI 모델 통합
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

# 실제 분석 라이브러리들
import whisper
import easyocr
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False

class RealAnalysisEngine:
    """실제 파일 분석 엔진"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 분석 모델들 초기화
        self.whisper_model = None
        self.ocr_reader = None
        self.nlp_pipeline = None
        
        # 성능 추적
        self.analysis_stats = {
            "total_files": 0,
            "successful_analyses": 0,
            "total_processing_time": 0,
            "last_analysis_time": None
        }
        
        self.logger.info("🚀 실제 분석 엔진 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _lazy_load_whisper(self, model_size: str = "base") -> whisper.Whisper:
        """Whisper 모델 지연 로딩"""
        if self.whisper_model is None:
            self.logger.info(f"🎤 Whisper {model_size} 모델 로딩...")
            start_time = time.time()
            self.whisper_model = whisper.load_model(model_size)
            load_time = time.time() - start_time
            self.logger.info(f"✅ Whisper 로드 완료 ({load_time:.1f}초)")
        return self.whisper_model
    
    def _lazy_load_ocr(self) -> easyocr.Reader:
        """EasyOCR 모델 지연 로딩"""
        if self.ocr_reader is None:
            self.logger.info("🖼️ EasyOCR 한/영 모델 로딩...")
            start_time = time.time()
            self.ocr_reader = easyocr.Reader(['ko', 'en'])
            load_time = time.time() - start_time
            self.logger.info(f"✅ EasyOCR 로드 완료 ({load_time:.1f}초)")
        return self.ocr_reader
    
    def _lazy_load_nlp(self) -> Optional[any]:
        """NLP 파이프라인 지연 로딩"""
        if not transformers_available:
            return None
            
        if self.nlp_pipeline is None:
            try:
                self.logger.info("🧠 NLP 모델 로딩...")
                start_time = time.time()
                self.nlp_pipeline = pipeline("summarization", 
                                           model="facebook/bart-large-cnn")
                load_time = time.time() - start_time
                self.logger.info(f"✅ NLP 로드 완료 ({load_time:.1f}초)")
            except Exception as e:
                self.logger.warning(f"NLP 모델 로드 실패: {e}")
                return None
        return self.nlp_pipeline
    
    def analyze_audio_file(self, file_path: str, language: str = "ko") -> Dict[str, Any]:
        """실제 음성 파일 분석"""
        self.logger.info(f"🎤 실제 음성 분석 시작: {os.path.basename(file_path)}")
        
        start_time = time.time()
        
        try:
            # Whisper 모델 로드
            model = self._lazy_load_whisper()
            
            # 음성-텍스트 변환
            self.logger.info("🔄 음성-텍스트 변환 중...")
            result = model.transcribe(file_path, language=language)
            
            processing_time = time.time() - start_time
            
            # 결과 분석
            text = result["text"]
            segments = result["segments"]
            detected_language = result["language"]
            
            # 텍스트 요약 (NLP 모델 사용 가능시)
            summary = self._generate_summary(text)
            
            # 주얼리 키워드 분석
            jewelry_keywords = self._extract_jewelry_keywords(text)
            
            analysis_result = {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                "processing_time": round(processing_time, 1),
                "detected_language": detected_language,
                "segments_count": len(segments),
                "text_length": len(text),
                "full_text": text,
                "summary": summary,
                "jewelry_keywords": jewelry_keywords,
                "segments": segments,
                "analysis_type": "real_whisper_stt",
                "timestamp": datetime.now().isoformat()
            }
            
            self._update_stats(processing_time, True)
            self.logger.info(f"✅ 음성 분석 완료 ({processing_time:.1f}초)")
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"음성 분석 실패: {str(e)}"
            self.logger.error(error_msg)
            self._update_stats(time.time() - start_time, False)
            
            return {
                "status": "error",
                "error": error_msg,
                "file_name": os.path.basename(file_path),
                "analysis_type": "real_whisper_stt",
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_image_file(self, file_path: str) -> Dict[str, Any]:
        """실제 이미지 파일 OCR 분석"""
        self.logger.info(f"🖼️ 실제 이미지 분석 시작: {os.path.basename(file_path)}")
        
        start_time = time.time()
        
        try:
            # OCR 모델 로드
            reader = self._lazy_load_ocr()
            
            # OCR 텍스트 추출
            self.logger.info("🔄 이미지 텍스트 추출 중...")
            results = reader.readtext(file_path)
            
            processing_time = time.time() - start_time
            
            # 결과 처리
            detected_texts = []
            total_confidence = 0
            
            for bbox, text, confidence in results:
                detected_texts.append({
                    "text": text,
                    "confidence": round(confidence, 3),
                    "bbox": bbox
                })
                total_confidence += confidence
            
            avg_confidence = total_confidence / len(results) if results else 0
            full_text = ' '.join([item["text"] for item in detected_texts])
            
            # 텍스트 요약
            summary = self._generate_summary(full_text)
            
            # 주얼리 키워드 분석
            jewelry_keywords = self._extract_jewelry_keywords(full_text)
            
            analysis_result = {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                "processing_time": round(processing_time, 1),
                "blocks_detected": len(results),
                "average_confidence": round(avg_confidence, 3),
                "full_text": full_text,
                "summary": summary,
                "jewelry_keywords": jewelry_keywords,
                "detailed_results": detected_texts,
                "analysis_type": "real_easyocr",
                "timestamp": datetime.now().isoformat()
            }
            
            self._update_stats(processing_time, True)
            self.logger.info(f"✅ 이미지 분석 완료 ({processing_time:.1f}초)")
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"이미지 분석 실패: {str(e)}"
            self.logger.error(error_msg)
            self._update_stats(time.time() - start_time, False)
            
            return {
                "status": "error", 
                "error": error_msg,
                "file_name": os.path.basename(file_path),
                "analysis_type": "real_easyocr",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성"""
        if not text or len(text.strip()) < 50:
            return "텍스트가 너무 짧아 요약을 생성할 수 없습니다."
        
        # NLP 모델 사용 가능시
        nlp = self._lazy_load_nlp()
        if nlp and len(text) > 100:
            try:
                # 긴 텍스트는 자르기
                if len(text) > 1024:
                    text = text[:1024]
                
                summary_result = nlp(text, max_length=100, min_length=30, do_sample=False)
                return summary_result[0]['summary_text']
            except Exception as e:
                self.logger.debug(f"NLP 요약 실패: {e}")
        
        # 기본 요약 (첫 100자)
        return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_jewelry_keywords(self, text: str) -> List[str]:
        """주얼리 관련 키워드 추출"""
        if not text:
            return []
        
        jewelry_terms = [
            # 영어 주얼리 용어
            "diamond", "gold", "silver", "platinum", "jewelry", "jewellery", 
            "ring", "necklace", "bracelet", "earring", "pendant", "gemstone",
            "ruby", "sapphire", "emerald", "pearl", "crystal", "luxury",
            "carat", "cut", "clarity", "color", "certificate", "GIA",
            
            # 한국어 주얼리 용어  
            "다이아몬드", "금", "은", "백금", "주얼리", "반지", "목걸이", 
            "팔찌", "귀걸이", "펜던트", "보석", "루비", "사파이어", 
            "에메랄드", "진주", "크리스탈", "럭셔리", "캐럿", "커팅",
            "투명도", "색상", "인증서", "지아"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in jewelry_terms:
            if term.lower() in text_lower:
                found_keywords.append(term)
        
        return list(set(found_keywords))  # 중복 제거
    
    def _update_stats(self, processing_time: float, success: bool):
        """통계 업데이트"""
        self.analysis_stats["total_files"] += 1
        self.analysis_stats["total_processing_time"] += processing_time
        if success:
            self.analysis_stats["successful_analyses"] += 1
        self.analysis_stats["last_analysis_time"] = datetime.now().isoformat()
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 통계 반환"""
        total_files = self.analysis_stats["total_files"]
        if total_files == 0:
            return self.analysis_stats
        
        stats = self.analysis_stats.copy()
        stats["success_rate"] = round(
            (stats["successful_analyses"] / total_files) * 100, 1
        )
        stats["average_processing_time"] = round(
            stats["total_processing_time"] / total_files, 1
        )
        
        return stats

# 전역 분석 엔진 인스턴스
global_analysis_engine = RealAnalysisEngine()

def analyze_file_real(file_path: str, file_type: str) -> Dict[str, Any]:
    """파일 실제 분석 (간편 사용)"""
    if file_type == "audio":
        return global_analysis_engine.analyze_audio_file(file_path)
    elif file_type == "image":
        return global_analysis_engine.analyze_image_file(file_path)
    else:
        return {
            "status": "error",
            "error": f"지원하지 않는 파일 타입: {file_type}",
            "file_name": os.path.basename(file_path),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # 테스트 실행
    print("🚀 실제 분석 엔진 테스트")
    print("=" * 50)
    
    engine = RealAnalysisEngine()
    
    # 테스트 파일들
    test_files = [
        ("/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1/새로운 녹음 2.m4a", "audio"),
        ("/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1/IMG_2160.JPG", "image")
    ]
    
    for file_path, file_type in test_files:
        if os.path.exists(file_path):
            print(f"\n🧪 테스트: {os.path.basename(file_path)}")
            result = analyze_file_real(file_path, file_type)
            print(f"결과: {result.get('status', 'unknown')}")
            if result.get('status') == 'success':
                print(f"처리시간: {result.get('processing_time', 0)}초")
                if 'full_text' in result:
                    text = result['full_text']
                    print(f"추출 텍스트: {text[:100]}{'...' if len(text) > 100 else ''}")
        else:
            print(f"⚠️ 파일 없음: {file_path}")
    
    # 통계 출력
    print(f"\n📊 분석 통계:")
    stats = engine.get_analysis_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ 실제 분석 엔진 테스트 완료!")