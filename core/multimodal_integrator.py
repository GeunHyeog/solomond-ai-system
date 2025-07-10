"""
솔로몬드 AI 시스템 - 멀티모달 통합 분석 엔진 v3.0
음성, 비디오, 이미지, 문서, 웹 데이터를 통합하여 종합 분석 및 결론 도출
품질 분석 + 한국어 최종 요약 통합
"""

import asyncio
import io
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import json
import logging
from collections import defaultdict, Counter
import re
from pathlib import Path

# 각 모듈 import
from .analyzer import STTAnalyzer, get_stt_analyzer
from .video_processor import get_video_processor, extract_audio_from_video
from .image_processor import get_image_processor, process_image_file, process_document_file
from .web_crawler import get_web_crawler, crawl_url, crawl_jewelry_news
from .jewelry_ai_engine import JewelryAIEngine
from .cross_validation_visualizer import CrossValidationVisualizer
from .speaker_analyzer import SpeakerAnalyzer

# 새로운 품질 분석 및 한국어 요약 모듈
from .quality_analyzer import get_quality_analyzer, analyze_audio_quality, analyze_image_quality
from .korean_summarizer import get_korean_summarizer, analyze_situation_in_korean

class MultimodalIntegrator:
    """멀티모달 데이터 통합 분석 클래스 v3.0"""
    
    def __init__(self):
        # 각 프로세서 인스턴스
        self.stt_analyzer = None
        self.video_processor = get_video_processor()
        self.image_processor = get_image_processor()
        self.web_crawler = get_web_crawler()
        self.ai_engine = JewelryAIEngine()
        self.cross_validator = CrossValidationVisualizer()
        self.speaker_analyzer = SpeakerAnalyzer()
        
        # 새로운 품질 분석 및 한국어 요약 모듈
        self.quality_analyzer = get_quality_analyzer()
        self.korean_summarizer = get_korean_summarizer()
        
        # 분석 가중치 (소스별 신뢰도)
        self.source_weights = {
            "audio": 1.0,      # 음성 (기본)
            "video": 0.9,      # 비디오 (음성에서 추출)
            "image": 0.8,      # 이미지 OCR
            "document": 0.9,   # 문서 (높은 신뢰도)
            "web": 0.6         # 웹 크롤링 (낮은 신뢰도)
        }
        
        # 주얼리 용어 카테고리 매핑
        self.jewelry_categories = {
            "gems": ["다이아몬드", "루비", "사파이어", "에메랄드", "진주"],
            "metals": ["금", "은", "백금", "플래티넘"],
            "grading": ["4C", "캐럿", "컷", "컬러", "클래리티"],
            "certification": ["GIA", "AGS", "감정서", "인증서"],
            "business": ["가격", "할인", "도매", "소매", "무역"],
            "techniques": ["세팅", "가공", "연마", "조각"]
        }
        
        logging.info("멀티모달 통합 분석 엔진 v3.0 초기화 완료 (품질분석+한국어요약 통합)")
    
    def _get_stt_analyzer(self):
        """STT 분석기 지연 초기화"""
        if self.stt_analyzer is None:
            self.stt_analyzer = get_stt_analyzer()
        return self.stt_analyzer
    
    async def process_multimodal_session(self, 
                                       session_data: Dict,
                                       analysis_depth: str = "comprehensive",
                                       situation_type: str = "auto") -> Dict:
        """
        멀티모달 세션 통합 처리 v3.0
        
        Args:
            session_data: 세션 데이터 (각 소스별 파일/URL 정보)
            analysis_depth: 분석 깊이 ("quick", "standard", "comprehensive")
            situation_type: 상황 타입 ("auto", "seminar", "meeting", "lecture", "conference")
            
        Returns:
            통합 분석 결과 (품질분석 + 한국어 요약 포함)
        """
        try:
            print("🔄 멀티모달 세션 분석 v3.0 시작")
            
            # 세션 정보 초기화
            session_id = session_data.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            session_title = session_data.get("title", "Untitled Session")
            
            # 각 소스별 처리 결과 저장
            source_results = {}
            
            # 1. 음성 파일 처리 (품질 분석 포함)
            if "audio_files" in session_data:
                source_results["audio"] = await self._process_audio_sources_with_quality(session_data["audio_files"])
            
            # 2. 비디오 파일 처리 (품질 분석 포함)
            if "video_files" in session_data:
                source_results["video"] = await self._process_video_sources_with_quality(session_data["video_files"])
            
            # 3. 이미지/문서 파일 처리 (품질 분석 포함)
            if "document_files" in session_data:
                source_results["documents"] = await self._process_document_sources_with_quality(session_data["document_files"])
            
            # 4. 웹 소스 처리
            if "web_urls" in session_data:
                source_results["web"] = await self._process_web_sources(session_data["web_urls"])
            
            # 5. 통합 분석 수행
            integrated_analysis = await self._perform_integrated_analysis(
                source_results, 
                analysis_depth
            )
            
            # 6. 크로스 검증 수행
            cross_validation = await self._perform_cross_validation(source_results)
            
            # 7. 한국어 종합 분석 (새로운 기능)
            korean_analysis = await self.korean_summarizer.analyze_situation_comprehensively(
                {"source_results": source_results},
                situation_type=situation_type,
                focus_areas=None
            )
            
            # 8. 최종 인사이트 생성 (품질 정보 포함)
            final_insights = await self._generate_final_insights_with_quality(
                integrated_analysis,
                cross_validation,
                korean_analysis,
                session_data
            )
            
            # 9. 종합 리포트 생성 v3.0
            comprehensive_report = self._generate_comprehensive_report_v3(
                session_id,
                session_title,
                source_results,
                integrated_analysis,
                cross_validation,
                korean_analysis,
                final_insights
            )
            
            print(f"✅ 멀티모달 세션 분석 v3.0 완료: {len(source_results)}개 소스 통합 + 품질분석 + 한국어요약")
            return comprehensive_report
            
        except Exception as e:
            logging.error(f"멀티모달 세션 분석 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_data.get("session_id", "unknown")
            }
    
    async def _process_audio_sources_with_quality(self, audio_files: List[Dict]) -> Dict:
        """음성 파일들 처리 (품질 분석 포함)"""
        print("🎤 음성 소스 처리 중... (품질 분석 포함)")
        
        results = []
        quality_results = []
        total_duration = 0
        combined_text = ""
        
        stt_analyzer = self._get_stt_analyzer()
        
        for audio_file in audio_files:
            try:
                # 1. 품질 분석 먼저 수행
                quality_result = await self.quality_analyzer.analyze_audio_quality(
                    audio_file["content"],
                    audio_file.get("filename", "unknown.wav")
                )
                quality_results.append(quality_result)
                
                # 2. STT 분석 수행
                if "content" in audio_file:
                    result = await stt_analyzer.analyze_audio(
                        audio_file["content"],
                        audio_file.get("filename", "unknown.wav"),
                        language=audio_file.get("language", "ko"),
                        enable_jewelry_enhancement=True
                    )
                    
                    if result.get("success"):
                        # 품질 정보 통합
                        result["quality_analysis"] = quality_result
                        results.append(result)
                        total_duration += result.get("duration", 0)
                        combined_text += f" {result.get('enhanced_text', '')}"
                        
                        # 화자 분석 추가
                        if hasattr(self.speaker_analyzer, 'analyze_speakers'):
                            speaker_result = await self.speaker_analyzer.analyze_speakers(
                                audio_file["content"],
                                audio_file.get("filename", "unknown.wav")
                            )
                            result["speaker_analysis"] = speaker_result
                
            except Exception as e:
                logging.error(f"음성 파일 처리 오류: {e}")
                continue
        
        # 전체 품질 평가
        overall_quality = self._calculate_overall_audio_quality(quality_results)
        
        return {
            "source_type": "audio",
            "files_processed": len(results),
            "files_successful": len([r for r in results if r.get("success")]),
            "total_duration": total_duration,
            "combined_text": combined_text.strip(),
            "individual_results": results,
            "quality_analysis": {
                "overall_quality": overall_quality,
                "individual_quality": quality_results,
                "quality_issues": [q.get("improvement_suggestions", []) for q in quality_results if not q.get("success", True) or q.get("quality_metrics", {}).get("overall_quality", 1.0) < 0.6]
            },
            "summary": {
                "total_words": len(combined_text.split()),
                "average_confidence": sum(r.get("confidence", 0) for r in results) / max(len(results), 1),
                "quality_score": overall_quality
            }
        }
    
    async def _process_video_sources_with_quality(self, video_files: List[Dict]) -> Dict:
        """비디오 파일들 처리 (품질 분석 포함)"""
        print("🎥 비디오 소스 처리 중... (품질 분석 포함)")
        
        results = []
        total_duration = 0
        combined_text = ""
        
        for video_file in video_files:
            try:
                # 비디오에서 음성 추출
                extraction_result = await extract_audio_from_video(
                    video_file["content"],
                    video_file.get("filename", "unknown.mp4")
                )
                
                if extraction_result.get("success"):
                    # 추출된 음성에 대한 품질 분석
                    audio_quality = await self.quality_analyzer.analyze_audio_quality(
                        extraction_result["audio_content"],
                        extraction_result["extracted_filename"]
                    )
                    
                    # 추출된 음성으로 STT 수행
                    stt_analyzer = self._get_stt_analyzer()
                    stt_result = await stt_analyzer.analyze_audio(
                        extraction_result["audio_content"],
                        extraction_result["extracted_filename"],
                        language=video_file.get("language", "ko"),
                        enable_jewelry_enhancement=True
                    )
                    
                    if stt_result.get("success"):
                        # 비디오 정보와 STT 결과 결합
                        combined_result = {
                            **stt_result,
                            "video_info": extraction_result,
                            "source_file": video_file.get("filename"),
                            "audio_quality_analysis": audio_quality
                        }
                        
                        results.append(combined_result)
                        total_duration += stt_result.get("duration", 0)
                        combined_text += f" {stt_result.get('enhanced_text', '')}"
                
            except Exception as e:
                logging.error(f"비디오 파일 처리 오류: {e}")
                continue
        
        return {
            "source_type": "video",
            "files_processed": len(results),
            "files_successful": len([r for r in results if r.get("success")]),
            "total_duration": total_duration,
            "combined_text": combined_text.strip(),
            "individual_results": results,
            "summary": {
                "total_words": len(combined_text.split()),
                "extraction_method": "ffmpeg",
                "quality_info": "음성 추출 후 품질 분석 완료"
            }
        }
    
    async def _process_document_sources_with_quality(self, document_files: List[Dict]) -> Dict:
        """문서/이미지 파일들 처리 (품질 분석 포함)"""
        print("📄 문서 소스 처리 중... (품질 분석 포함)")
        
        results = []
        quality_results = []
        combined_text = ""
        total_pages = 0
        
        for doc_file in document_files:
            try:
                # 파일 타입에 따라 처리
                file_type = self.image_processor.get_file_type(doc_file.get("filename", ""))
                
                # 1. 품질 분석 먼저 수행 (이미지인 경우)
                if file_type == "image":
                    # PPT 화면 여부 자동 감지
                    filename = doc_file.get("filename", "")
                    is_ppt_hint = any(keyword in filename.lower() for keyword in ["ppt", "slide", "presentation", "screen"])
                    
                    image_quality = await self.quality_analyzer.analyze_image_quality(
                        doc_file["content"],
                        filename,
                        is_ppt_screen=is_ppt_hint
                    )
                    quality_results.append(image_quality)
                    
                    # 2. OCR 처리
                    result = await process_image_file(
                        doc_file["content"],
                        filename,
                        enhance_quality=True,
                        ocr_method="auto"
                    )
                    
                    if result.get("success"):
                        result["quality_analysis"] = image_quality
                        
                elif file_type == "document":
                    result = await process_document_file(
                        doc_file["content"],
                        doc_file.get("filename", "unknown.pdf")
                    )
                    
                    # 문서의 경우 간단한 품질 평가
                    if result.get("success"):
                        doc_quality = {
                            "success": True,
                            "filename": doc_file.get("filename", "unknown.pdf"),
                            "quality_assessment": "문서 처리 성공",
                            "text_length": len(result.get("text", "")),
                            "overall_quality": 0.9 if len(result.get("text", "")) > 100 else 0.6
                        }
                        result["quality_analysis"] = doc_quality
                        quality_results.append(doc_quality)
                else:
                    continue
                
                if result.get("success"):
                    results.append(result)
                    
                    # 텍스트 추출
                    if "text" in result:
                        combined_text += f" {result['text']}"
                    elif "ocr_results" in result and "text" in result["ocr_results"]:
                        combined_text += f" {result['ocr_results']['text']}"
                    
                    # 페이지 수 계산
                    if "page_count" in result:
                        total_pages += result["page_count"]
                    else:
                        total_pages += 1
                
            except Exception as e:
                logging.error(f"문서 파일 처리 오류: {e}")
                continue
        
        # 전체 품질 평가
        overall_quality = self._calculate_overall_document_quality(quality_results)
        
        return {
            "source_type": "documents",
            "files_processed": len(results),
            "files_successful": len([r for r in results if r.get("success")]),
            "total_pages": total_pages,
            "combined_text": combined_text.strip(),
            "individual_results": results,
            "quality_analysis": {
                "overall_quality": overall_quality,
                "individual_quality": quality_results,
                "ppt_screens_detected": len([q for q in quality_results if q.get("ppt_analysis", {}).get("is_ppt_screen", False)]),
                "ocr_quality_summary": self._summarize_ocr_quality(quality_results)
            },
            "summary": {
                "total_words": len(combined_text.split()),
                "document_types": list(set(r.get("file_type", "unknown") for r in results)),
                "quality_score": overall_quality
            }
        }
    
    async def _process_web_sources(self, web_urls: List[str]) -> Dict:
        """웹 소스들 처리"""
        print("🌐 웹 소스 처리 중...")
        
        results = []
        combined_text = ""
        
        async with self.web_crawler:
            for url in web_urls:
                try:
                    result = await self.web_crawler.process_url(
                        url,
                        content_type="auto",
                        extract_video=False
                    )
                    
                    if result.get("success"):
                        results.append(result)
                        
                        # 텍스트 추출
                        if "content" in result:
                            combined_text += f" {result['content']}"
                        elif "text" in result:
                            combined_text += f" {result['text']}"
                
                except Exception as e:
                    logging.error(f"웹 URL 처리 오류 ({url}): {e}")
                    continue
        
        return {
            "source_type": "web",
            "urls_processed": len(results),
            "urls_successful": len([r for r in results if r.get("success")]),
            "combined_text": combined_text.strip(),
            "individual_results": results,
            "summary": {
                "total_words": len(combined_text.split()),
                "content_types": list(set(r.get("content_type", "unknown") for r in results))
            }
        }
    
    def _calculate_overall_audio_quality(self, quality_results: List[Dict]) -> float:
        """전체 음성 품질 계산"""
        if not quality_results:
            return 0.5
        
        successful_results = [q for q in quality_results if q.get("success")]
        if not successful_results:
            return 0.3
        
        scores = []
        for result in successful_results:
            quality_metrics = result.get("quality_metrics", {})
            overall_quality = quality_metrics.get("overall_quality", 0.5)
            scores.append(overall_quality)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_overall_document_quality(self, quality_results: List[Dict]) -> float:
        """전체 문서 품질 계산"""
        if not quality_results:
            return 0.5
        
        successful_results = [q for q in quality_results if q.get("success")]
        if not successful_results:
            return 0.3
        
        scores = []
        for result in successful_results:
            overall_quality = result.get("overall_quality", 0.5)
            scores.append(overall_quality)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _summarize_ocr_quality(self, quality_results: List[Dict]) -> Dict:
        """OCR 품질 요약"""
        ocr_summary = {
            "total_files": len(quality_results),
            "successful_ocr": 0,
            "average_confidence": 0.0,
            "ppt_optimized": 0,
            "issues_found": []
        }
        
        confidences = []
        
        for result in quality_results:
            if result.get("success"):
                ocr_quality = result.get("ocr_quality", {})
                if "average_confidence" in ocr_quality:
                    confidences.append(ocr_quality["average_confidence"])
                    ocr_summary["successful_ocr"] += 1
                
                # PPT 최적화 확인
                ppt_analysis = result.get("ppt_analysis", {})
                if ppt_analysis.get("is_ppt_screen"):
                    ocr_summary["ppt_optimized"] += 1
                
                # 품질 이슈 수집
                improvement_suggestions = result.get("improvement_suggestions", [])
                if improvement_suggestions:
                    ocr_summary["issues_found"].extend(improvement_suggestions[:2])
        
        if confidences:
            ocr_summary["average_confidence"] = sum(confidences) / len(confidences)
        
        return ocr_summary
    
    async def _perform_integrated_analysis(self, 
                                         source_results: Dict,
                                         analysis_depth: str) -> Dict:
        """통합 분석 수행"""
        print("🧠 통합 AI 분석 수행 중...")
        
        # 모든 텍스트 결합
        all_texts = []
        source_weights_applied = {}
        
        for source_type, result in source_results.items():
            if result and "combined_text" in result:
                text = result["combined_text"]
                weight = self.source_weights.get(source_type, 0.5)
                
                all_texts.append({
                    "text": text,
                    "source": source_type,
                    "weight": weight,
                    "word_count": len(text.split())
                })
                source_weights_applied[source_type] = weight
        
        # 가중 평균으로 통합 텍스트 생성
        combined_text = " ".join([item["text"] for item in all_texts])
        
        # 주얼리 AI 엔진으로 고급 분석
        ai_analysis = await self.ai_engine.analyze_jewelry_content(
            combined_text,
            analysis_type="comprehensive" if analysis_depth == "comprehensive" else "standard"
        )
        
        # 소스별 주얼리 용어 분석
        source_term_analysis = {}
        for source_type, result in source_results.items():
            if result and "combined_text" in result:
                terms = self._extract_jewelry_terms(result["combined_text"])
                source_term_analysis[source_type] = terms
        
        # 주제 일관성 분석
        topic_consistency = self._analyze_topic_consistency(source_results)
        
        # 신뢰도 점수 계산 (품질 정보 반영)
        confidence_score = self._calculate_confidence_score_with_quality(source_results, source_weights_applied)
        
        return {
            "ai_analysis": ai_analysis,
            "source_term_analysis": source_term_analysis,
            "topic_consistency": topic_consistency,
            "confidence_score": confidence_score,
            "total_word_count": len(combined_text.split()),
            "source_distribution": {
                source: len(result.get("combined_text", "").split())
                for source, result in source_results.items()
                if result
            },
            "analysis_depth": analysis_depth
        }
    
    def _calculate_confidence_score_with_quality(self, source_results: Dict, weights: Dict) -> float:
        """품질 정보를 반영한 신뢰도 점수 계산"""
        total_weight = 0
        weighted_confidence = 0
        
        for source_type, result in source_results.items():
            if result:
                weight = weights.get(source_type, 0.5)
                
                # 기본 신뢰도
                source_confidence = result.get("summary", {}).get("average_confidence", 0.5)
                
                # 품질 점수 반영
                quality_score = result.get("summary", {}).get("quality_score", 0.7)
                
                # 품질과 신뢰도 조합 (7:3 비율)
                combined_confidence = source_confidence * 0.7 + quality_score * 0.3
                
                weighted_confidence += combined_confidence * weight
                total_weight += weight
        
        return round(weighted_confidence / max(total_weight, 1), 3)
    
    async def _perform_cross_validation(self, source_results: Dict) -> Dict:
        """크로스 검증 수행"""
        print("🔍 크로스 검증 수행 중...")
        
        # 소스 간 일치도 매트릭스 생성
        if hasattr(self.cross_validator, 'calculate_consensus_matrix'):
            consensus_matrix = await self.cross_validator.calculate_consensus_matrix(source_results)
        else:
            consensus_matrix = self._simple_consensus_matrix(source_results)
        
        # 주요 키워드 일치도 분석
        keyword_consistency = self._analyze_keyword_consistency(source_results)
        
        # 타임라인 일치성 (시간 정보가 있는 경우)
        timeline_consistency = self._analyze_timeline_consistency(source_results)
        
        return {
            "consensus_matrix": consensus_matrix,
            "keyword_consistency": keyword_consistency,
            "timeline_consistency": timeline_consistency,
            "overall_consistency": self._calculate_overall_consistency(consensus_matrix)
        }
    
    def _simple_consensus_matrix(self, source_results: Dict) -> Dict:
        """간단한 합의 매트릭스 계산"""
        sources = list(source_results.keys())
        matrix = {}
        
        for i, source1 in enumerate(sources):
            matrix[source1] = {}
            for j, source2 in enumerate(sources):
                if i == j:
                    similarity = 1.0
                else:
                    # 텍스트 유사도 간단 계산
                    text1 = source_results[source1].get("combined_text", "")
                    text2 = source_results[source2].get("combined_text", "")
                    similarity = self._calculate_text_similarity(text1, text2)
                
                matrix[source1][source2] = round(similarity, 3)
        
        return matrix
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 구현)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_jewelry_terms(self, text: str) -> Dict:
        """텍스트에서 주얼리 용어 추출"""
        terms_by_category = defaultdict(list)
        text_lower = text.lower()
        
        for category, terms in self.jewelry_categories.items():
            for term in terms:
                if term.lower() in text_lower:
                    terms_by_category[category].append(term)
        
        return dict(terms_by_category)
    
    def _analyze_topic_consistency(self, source_results: Dict) -> Dict:
        """주제 일관성 분석"""
        all_terms = defaultdict(int)
        source_terms = {}
        
        # 각 소스별 용어 빈도 계산
        for source_type, result in source_results.items():
            if result and "combined_text" in result:
                terms = self._extract_jewelry_terms(result["combined_text"])
                source_terms[source_type] = terms
                
                for category, category_terms in terms.items():
                    for term in category_terms:
                        all_terms[term] += 1
        
        # 공통 주제 식별
        common_terms = {term: count for term, count in all_terms.items() if count >= 2}
        
        return {
            "common_terms": common_terms,
            "source_terms": source_terms,
            "consistency_score": len(common_terms) / max(len(all_terms), 1)
        }
    
    def _analyze_keyword_consistency(self, source_results: Dict) -> Dict:
        """키워드 일치도 분석"""
        keyword_counts = defaultdict(lambda: defaultdict(int))
        
        for source_type, result in source_results.items():
            if result and "combined_text" in result:
                text = result["combined_text"].lower()
                words = text.split()
                
                # 주얼리 키워드 빈도 계산
                for category, terms in self.jewelry_categories.items():
                    for term in terms:
                        if term.lower() in text:
                            keyword_counts[category][source_type] += text.count(term.lower())
        
        return dict(keyword_counts)
    
    def _analyze_timeline_consistency(self, source_results: Dict) -> Dict:
        """타임라인 일치성 분석"""
        # 간단한 구현 - 실제로는 더 정교한 시간 정보 추출 필요
        timeline_info = {}
        
        for source_type, result in source_results.items():
            if result:
                timeline_info[source_type] = {
                    "processing_time": result.get("processing_time", ""),
                    "duration": result.get("total_duration", 0),
                    "timestamp": datetime.now().isoformat()
                }
        
        return timeline_info
    
    def _calculate_overall_consistency(self, consensus_matrix: Dict) -> float:
        """전체 일관성 점수 계산"""
        if not consensus_matrix:
            return 0.0
        
        total_similarity = 0
        count = 0
        
        for source1, similarities in consensus_matrix.items():
            for source2, similarity in similarities.items():
                if source1 != source2:  # 자기 자신과의 비교 제외
                    total_similarity += similarity
                    count += 1
        
        return round(total_similarity / max(count, 1), 3)
    
    async def _generate_final_insights_with_quality(self, 
                                                  integrated_analysis: Dict,
                                                  cross_validation: Dict,
                                                  korean_analysis: Dict,
                                                  session_data: Dict) -> Dict:
        """최종 인사이트 생성 (품질 정보 포함)"""
        print("💡 최종 인사이트 생성 중... (품질 정보 포함)")
        
        # 기존 인사이트
        basic_insights = await self._generate_basic_insights(integrated_analysis, cross_validation, session_data)
        
        # 품질 관련 인사이트
        quality_insights = self._generate_quality_insights(korean_analysis)
        
        # 한국어 분석 인사이트
        korean_insights = self._extract_korean_insights(korean_analysis)
        
        return {
            **basic_insights,
            "quality_insights": quality_insights,
            "korean_analysis_insights": korean_insights,
            "generated_at": datetime.now().isoformat()
        }
    
    async def _generate_basic_insights(self, 
                                     integrated_analysis: Dict,
                                     cross_validation: Dict,
                                     session_data: Dict) -> Dict:
        """기본 인사이트 생성"""
        # 핵심 발견사항
        key_findings = []
        
        # 1. 주요 주제
        ai_analysis = integrated_analysis.get("ai_analysis", {})
        if "main_topics" in ai_analysis:
            key_findings.append(f"주요 주제: {', '.join(ai_analysis['main_topics'])}")
        
        # 2. 신뢰도 평가
        confidence = integrated_analysis.get("confidence_score", 0)
        if confidence >= 0.8:
            key_findings.append("높은 신뢰도의 분석 결과")
        elif confidence >= 0.6:
            key_findings.append("보통 수준의 신뢰도")
        else:
            key_findings.append("추가 검증이 필요한 분석 결과")
        
        # 3. 일관성 평가
        consistency = cross_validation.get("overall_consistency", 0)
        if consistency >= 0.7:
            key_findings.append("소스 간 높은 일관성 확인")
        elif consistency >= 0.5:
            key_findings.append("소스 간 적당한 일관성")
        else:
            key_findings.append("소스 간 일관성 부족 - 추가 검토 필요")
        
        # 4. 데이터 품질 평가
        total_words = integrated_analysis.get("total_word_count", 0)
        if total_words >= 1000:
            key_findings.append("충분한 양의 데이터로 신뢰할 만한 분석")
        elif total_words >= 500:
            key_findings.append("적당한 양의 데이터")
        else:
            key_findings.append("제한적인 데이터 양 - 추가 자료 수집 권장")
        
        # 실행 가능한 권장사항
        recommendations = []
        
        # 주얼리 비즈니스 관점 권장사항
        jewelry_analysis = ai_analysis.get("jewelry_insights", {})
        if jewelry_analysis.get("business_opportunities"):
            recommendations.extend(jewelry_analysis["business_opportunities"][:3])
        
        # 기술적 권장사항
        if confidence < 0.7:
            recommendations.append("더 많은 데이터 소스를 추가하여 분석 정확도 향상")
        
        if consistency < 0.6:
            recommendations.append("소스 간 불일치 부분에 대한 세부 검증 수행")
        
        # 다음 단계 제안
        next_steps = [
            "핵심 발견사항에 대한 상세 분석 수행",
            "이해관계자들과 결과 공유 및 피드백 수집",
            "실행 계획 수립 및 우선순위 설정"
        ]
        
        return {
            "key_findings": key_findings,
            "recommendations": recommendations,
            "next_steps": next_steps,
            "quality_assessment": {
                "confidence_level": "높음" if confidence >= 0.8 else "보통" if confidence >= 0.6 else "낮음",
                "consistency_level": "높음" if consistency >= 0.7 else "보통" if consistency >= 0.5 else "낮음",
                "data_sufficiency": "충분" if total_words >= 1000 else "보통" if total_words >= 500 else "부족"
            }
        }
    
    def _generate_quality_insights(self, korean_analysis: Dict) -> Dict:
        """품질 관련 인사이트 생성"""
        quality_assessment = korean_analysis.get("quality_assessment", {})
        
        insights = {
            "overall_quality_score": quality_assessment.get("overall_score", 0.5),
            "quality_issues": quality_assessment.get("issues_found", []),
            "improvement_recommendations": quality_assessment.get("recommendations", []),
            "source_quality_breakdown": quality_assessment.get("source_qualities", {}),
            "recording_quality_tips": []
        }
        
        # 품질 개선 팁 생성
        overall_score = quality_assessment.get("overall_score", 0.5)
        if overall_score < 0.7:
            insights["recording_quality_tips"] = [
                "📱 현장 녹화 시 마이크를 화자에게 더 가까이 배치",
                "💡 PPT 화면 촬영 시 조명과 각도 최적화",
                "🔇 주변 소음이 적은 환경에서 녹화",
                "📷 흔들림 방지를 위한 삼각대 사용 권장"
            ]
        else:
            insights["recording_quality_tips"] = [
                "✅ 현재 녹화 품질이 우수합니다!",
                "📊 지속적인 품질 유지를 위해 동일한 환경 설정 권장"
            ]
        
        return insights
    
    def _extract_korean_insights(self, korean_analysis: Dict) -> Dict:
        """한국어 분석 인사이트 추출"""
        if not korean_analysis.get("success"):
            return {"error": "한국어 분석 실패"}
        
        return {
            "situation_type": korean_analysis.get("analysis_info", {}).get("situation_name", "일반 상황"),
            "executive_summary": korean_analysis.get("final_summary", {}).get("executive_summary", ""),
            "key_business_findings": korean_analysis.get("business_analysis", {}).get("market_insights", [])[:3],
            "actionable_items": korean_analysis.get("actionable_insights", {}).get("immediate_actions", [])[:3],
            "strategic_recommendations": korean_analysis.get("business_analysis", {}).get("strategic_recommendations", [])[:3]
        }
    
    def _generate_comprehensive_report_v3(self, 
                                        session_id: str,
                                        session_title: str,
                                        source_results: Dict,
                                        integrated_analysis: Dict,
                                        cross_validation: Dict,
                                        korean_analysis: Dict,
                                        final_insights: Dict) -> Dict:
        """종합 리포트 생성 v3.0"""
        print("📊 종합 리포트 v3.0 생성 중...")
        
        # 요약 통계
        summary_stats = {
            "session_id": session_id,
            "session_title": session_title,
            "sources_processed": len([s for s in source_results.values() if s]),
            "total_files": sum(r.get("files_processed", r.get("urls_processed", 0)) for r in source_results.values() if r),
            "successful_files": sum(r.get("files_successful", r.get("urls_successful", 0)) for r in source_results.values() if r),
            "total_words": integrated_analysis.get("total_word_count", 0),
            "confidence_score": integrated_analysis.get("confidence_score", 0),
            "consistency_score": cross_validation.get("overall_consistency", 0),
            "korean_analysis_success": korean_analysis.get("success", False)
        }
        
        # 성능 메트릭 v3.0
        performance_metrics = {
            "processing_time": "실시간 계산됨",
            "source_distribution": integrated_analysis.get("source_distribution", {}),
            "analysis_depth": integrated_analysis.get("analysis_depth", "standard"),
            "quality_indicators": {
                "data_quality": "높음" if summary_stats["total_words"] >= 1000 else "보통",
                "source_reliability": "높음" if summary_stats["confidence_score"] >= 0.8 else "보통",
                "cross_validation": "통과" if summary_stats["consistency_score"] >= 0.6 else "검토 필요",
                "korean_analysis": "성공" if korean_analysis.get("success") else "실패"
            },
            "new_features": {
                "quality_analysis": "활성화됨",
                "korean_summarization": "활성화됨",
                "ppt_optimization": "활성화됨",
                "noise_analysis": "활성화됨"
            }
        }
        
        return {
            "success": True,
            "session_info": {
                "session_id": session_id,
                "title": session_title,
                "generated_at": datetime.now().isoformat(),
                "report_version": "3.0",
                "features": "품질분석 + 한국어요약 통합"
            },
            "summary_statistics": summary_stats,
            "source_results": source_results,
            "integrated_analysis": integrated_analysis,
            "cross_validation": cross_validation,
            "korean_analysis": korean_analysis,  # 새로운 섹션
            "final_insights": final_insights,
            "performance_metrics": performance_metrics,
            "report_type": "multimodal_comprehensive_v3"
        }


# 전역 인스턴스
_multimodal_integrator_instance = None

def get_multimodal_integrator() -> MultimodalIntegrator:
    """전역 멀티모달 통합기 인스턴스 반환"""
    global _multimodal_integrator_instance
    if _multimodal_integrator_instance is None:
        _multimodal_integrator_instance = MultimodalIntegrator()
    return _multimodal_integrator_instance

# 편의 함수들
async def process_multimodal_session(session_data: Dict, **kwargs) -> Dict:
    """멀티모달 세션 처리 편의 함수 v3.0"""
    integrator = get_multimodal_integrator()
    return await integrator.process_multimodal_session(session_data, **kwargs)

async def analyze_mixed_content_with_quality(files_data: List[Dict], 
                                           urls: List[str] = None,
                                           situation_type: str = "auto") -> Dict:
    """혼합 콘텐츠 품질 분석 편의 함수"""
    session_data = {
        "session_id": f"mixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "title": "Mixed Content Analysis with Quality Check"
    }
    
    # 파일들을 타입별로 분류
    audio_files = []
    video_files = []
    document_files = []
    
    for file_data in files_data:
        filename = file_data.get("filename", "")
        file_ext = Path(filename).suffix.lower()
        
        if file_ext in ['.mp3', '.wav', '.m4a', '.aac']:
            audio_files.append(file_data)
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.append(file_data)
        else:
            document_files.append(file_data)
    
    if audio_files:
        session_data["audio_files"] = audio_files
    if video_files:
        session_data["video_files"] = video_files
    if document_files:
        session_data["document_files"] = document_files
    if urls:
        session_data["web_urls"] = urls
    
    return await process_multimodal_session(
        session_data, 
        analysis_depth="comprehensive",
        situation_type=situation_type
    )

def get_integration_capabilities_v3() -> Dict:
    """멀티모달 통합 기능 정보 v3.0 반환"""
    integrator = get_multimodal_integrator()
    return {
        "supported_sources": ["audio", "video", "images", "documents", "web"],
        "source_weights": integrator.source_weights,
        "jewelry_categories": list(integrator.jewelry_categories.keys()),
        "analysis_depths": ["quick", "standard", "comprehensive"],
        "situation_types": ["auto", "seminar", "meeting", "lecture", "conference"],
        "output_formats": ["comprehensive_report_v3", "korean_summary", "quality_analysis"],
        "new_features_v3": [
            "실시간 품질 분석 (음성 노이즈, 이미지 OCR)",
            "PPT 화면 특화 분석 및 최적화",
            "다국어 → 한국어 통합 요약",
            "상황별 맞춤 분석 (세미나/회의/강의/컨퍼런스)",
            "비즈니스 관점 인사이트 자동 생성",
            "현장 촬영 품질 개선 제안"
        ]
    }

if __name__ == "__main__":
    # 테스트 코드
    async def test_integrator_v3():
        print("멀티모달 통합 분석 엔진 v3.0 테스트")
        capabilities = get_integration_capabilities_v3()
        print(f"통합 기능 v3.0: {capabilities}")
    
    asyncio.run(test_integrator_v3())
