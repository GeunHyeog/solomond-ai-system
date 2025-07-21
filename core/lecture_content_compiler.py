#!/usr/bin/env python3
"""
강의 내용 종합 컴파일러
여러 파일의 분석 결과를 하나의 완성된 강의 내용으로 정리
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

class LectureContentCompiler:
    """분석 결과를 종합하여 강의 내용을 생성하는 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 강의 주제별 분류 키워드
        self.topic_keywords = {
            "jewelry_business": [
                "다이아몬드", "금", "은", "백금", "주얼리", "보석", "jewelry", "diamond", "gold", 
                "silver", "platinum", "gemstone", "ring", "necklace", "bracelet", "earring"
            ],
            "technical_process": [
                "제작", "가공", "세팅", "커팅", "연마", "manufacturing", "cutting", "polishing", 
                "setting", "crafting", "technique", "process"
            ],
            "market_trend": [
                "시장", "트렌드", "가격", "판매", "고객", "브랜드", "market", "trend", "price", 
                "sales", "customer", "brand", "luxury", "premium"
            ],
            "quality_certification": [
                "품질", "인증", "등급", "검사", "GIA", "감정", "quality", "certification", 
                "grade", "evaluation", "assessment", "standard"
            ],
            "design_innovation": [
                "디자인", "혁신", "스타일", "패션", "트렌드", "창의", "design", "innovation", 
                "style", "fashion", "creative", "artistic"
            ]
        }
        
        # 강의 구조 템플릿
        self.lecture_structure = {
            "title": "",
            "overview": "",
            "main_topics": [],
            "detailed_content": {},
            "key_insights": [],
            "practical_applications": [],
            "conclusion": "",
            "source_files": [],
            "metadata": {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def compile_lecture_content(self, analysis_results: List[Dict[str, Any]], 
                              custom_title: str = None) -> Dict[str, Any]:
        """
        여러 분석 결과를 종합하여 강의 내용 생성
        
        Args:
            analysis_results: 각 파일의 분석 결과 리스트
            custom_title: 사용자 지정 강의 제목
        
        Returns:
            종합된 강의 내용 딕셔너리
        """
        start_time = time.time()
        self.logger.info(f"[LECTURE] 강의 내용 컴파일 시작: {len(analysis_results)}개 파일")
        
        try:
            # 1. 기본 구조 초기화
            lecture = self.lecture_structure.copy()
            lecture["metadata"] = {
                "compilation_date": datetime.now().isoformat(),
                "total_files": len(analysis_results),
                "compilation_time": 0
            }
            
            # 2. 파일 데이터 분석 및 분류
            categorized_content = self._categorize_content(analysis_results)
            
            # 3. 강의 제목 생성
            lecture["title"] = custom_title or self._generate_title(categorized_content)
            
            # 4. 개요 생성
            lecture["overview"] = self._generate_overview(categorized_content)
            
            # 5. 주요 주제 식별
            lecture["main_topics"] = self._identify_main_topics(categorized_content)
            
            # 6. 세부 내용 구성
            lecture["detailed_content"] = self._organize_detailed_content(categorized_content)
            
            # 7. 핵심 인사이트 추출
            lecture["key_insights"] = self._extract_key_insights(categorized_content)
            
            # 8. 실용적 응용 방안
            lecture["practical_applications"] = self._generate_practical_applications(categorized_content)
            
            # 9. 결론 작성
            lecture["conclusion"] = self._generate_conclusion(categorized_content)
            
            # 10. 소스 파일 정보
            lecture["source_files"] = self._compile_source_info(analysis_results)
            
            # 11. 메타데이터 완성
            processing_time = time.time() - start_time
            lecture["metadata"]["compilation_time"] = round(processing_time, 2)
            lecture["metadata"]["quality_score"] = self._calculate_quality_score(lecture)
            
            self.logger.info(f"[LECTURE] 강의 내용 컴파일 완료 ({processing_time:.1f}초)")
            
            return {
                "status": "success",
                "lecture_content": lecture,
                "compilation_stats": {
                    "files_processed": len(analysis_results),
                    "topics_identified": len(lecture["main_topics"]),
                    "insights_generated": len(lecture["key_insights"]),
                    "processing_time": round(processing_time, 2)
                }
            }
            
        except Exception as e:
            error_msg = f"강의 내용 컴파일 실패: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "status": "error",
                "error": error_msg,
                "compilation_time": round(time.time() - start_time, 2)
            }
    
    def _categorize_content(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """분석 결과를 주제별로 분류"""
        categorized = {
            "jewelry_business": [],
            "technical_process": [],
            "market_trend": [],
            "quality_certification": [],
            "design_innovation": [],
            "general": []
        }
        
        for result in analysis_results:
            if result.get("status") != "success" and result.get("status") != "partial_success":
                continue
            
            # 텍스트 내용 추출
            content_text = self._extract_text_content(result)
            if not content_text:
                continue
            
            # 키워드 기반 분류
            category = self._classify_by_keywords(content_text)
            categorized[category].append({
                "result": result,
                "content": content_text,
                "file_name": result.get("file_name", "unknown"),
                "file_type": self._determine_file_type(result)
            })
        
        return categorized
    
    def _extract_text_content(self, result: Dict[str, Any]) -> str:
        """분석 결과에서 텍스트 내용 추출"""
        text_sources = [
            "full_text",  # OCR 결과
            "transcribed_text",  # STT 결과
            "extracted_text",  # 문서 추출
            "summary",  # 요약
            "enhanced_text"  # 향상된 텍스트
        ]
        
        for source in text_sources:
            if source in result and result[source]:
                return str(result[source])
        
        # 부분 성공의 경우 partial_data에서 추출
        if result.get("status") == "partial_success" and "partial_data" in result:
            partial_data = result["partial_data"]
            if "detected_text" in partial_data:
                return str(partial_data["detected_text"])
            if "fallback_transcription" in partial_data:
                return str(partial_data["fallback_transcription"])
        
        return ""
    
    def _classify_by_keywords(self, text: str) -> str:
        """키워드 기반 내용 분류"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            scores[category] = score
        
        # 최고 점수 카테고리 반환, 점수가 0이면 'general'
        max_category = max(scores.items(), key=lambda x: x[1])
        return max_category[0] if max_category[1] > 0 else "general"
    
    def _determine_file_type(self, result: Dict[str, Any]) -> str:
        """파일 타입 결정"""
        if "analysis_type" in result:
            analysis_type = result["analysis_type"]
            if "whisper" in analysis_type or "stt" in analysis_type:
                return "audio"
            elif "ocr" in analysis_type:
                return "image"
            elif "document" in analysis_type:
                return "document"
            elif "video" in analysis_type:
                return "video"
        
        # 파일 확장자로 추정
        file_name = result.get("file_name", "")
        if any(ext in file_name.lower() for ext in ['.mp3', '.wav', '.m4a']):
            return "audio"
        elif any(ext in file_name.lower() for ext in ['.jpg', '.png', '.jpeg']):
            return "image"
        elif any(ext in file_name.lower() for ext in ['.pdf', '.docx', '.doc']):
            return "document"
        elif any(ext in file_name.lower() for ext in ['.mp4', '.mov', '.avi']):
            return "video"
        
        return "unknown"
    
    def _generate_title(self, categorized_content: Dict[str, List[Dict]]) -> str:
        """강의 제목 생성"""
        # 가장 많은 내용이 있는 카테고리 찾기
        main_category = max(categorized_content.items(), 
                           key=lambda x: len(x[1]) if x[0] != 'general' else 0)
        
        category_titles = {
            "jewelry_business": "주얼리 비즈니스 및 업계 동향",
            "technical_process": "주얼리 제작 기술 및 공정",
            "market_trend": "주얼리 시장 분석 및 트렌드",
            "quality_certification": "주얼리 품질 관리 및 인증",
            "design_innovation": "주얼리 디자인 혁신",
            "general": "주얼리 업계 종합 분석"
        }
        
        main_title = category_titles.get(main_category[0], "주얼리 업계 분석")
        
        # 날짜 추가
        date_str = datetime.now().strftime("%Y.%m.%d")
        
        return f"{main_title} - {date_str} 종합 강의"
    
    def _generate_overview(self, categorized_content: Dict[str, List[Dict]]) -> str:
        """강의 개요 생성"""
        total_files = sum(len(items) for items in categorized_content.values())
        
        overview = f"본 강의는 {total_files}개의 분석 자료를 바탕으로 구성된 주얼리 업계 종합 분석 내용입니다.\n\n"
        
        # 카테고리별 내용 요약
        for category, items in categorized_content.items():
            if not items:
                continue
            
            category_names = {
                "jewelry_business": "주얼리 비즈니스",
                "technical_process": "기술 및 제작 공정",
                "market_trend": "시장 동향",
                "quality_certification": "품질 및 인증",
                "design_innovation": "디자인 혁신",
                "general": "기타 관련 내용"
            }
            
            category_name = category_names.get(category, category)
            file_types = list(set(item["file_type"] for item in items))
            
            overview += f"• {category_name}: {len(items)}개 자료 ({', '.join(file_types)})\n"
        
        overview += "\n이러한 다양한 자료를 통해 주얼리 업계의 현황과 미래 전망을 종합적으로 살펴보겠습니다."
        
        return overview
    
    def _identify_main_topics(self, categorized_content: Dict[str, List[Dict]]) -> List[str]:
        """주요 주제 식별"""
        topics = []
        
        for category, items in categorized_content.items():
            if not items:
                continue
            
            topic_names = {
                "jewelry_business": "주얼리 비즈니스 전략",
                "technical_process": "제작 기술 및 공정 혁신",
                "market_trend": "시장 분석 및 소비자 트렌드",
                "quality_certification": "품질 관리 및 인증 시스템",
                "design_innovation": "디자인 트렌드 및 창작 기법",
                "general": "업계 일반 동향"
            }
            
            if category in topic_names:
                topics.append(topic_names[category])
        
        return topics
    
    def _organize_detailed_content(self, categorized_content: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """세부 내용 구성"""
        detailed = {}
        
        for category, items in categorized_content.items():
            if not items:
                continue
            
            category_content = {
                "summary": self._generate_category_summary(category, items),
                "key_points": self._extract_key_points(items),
                "supporting_data": self._compile_supporting_data(items),
                "file_sources": [item["file_name"] for item in items]
            }
            
            detailed[category] = category_content
        
        return detailed
    
    def _generate_category_summary(self, category: str, items: List[Dict]) -> str:
        """카테고리별 요약 생성"""
        all_text = " ".join(item["content"] for item in items)
        
        # 간단한 요약 로직 (실제로는 더 정교한 NLP 기법 사용 가능)
        sentences = all_text.split('.')[:5]  # 처음 5문장
        summary = '. '.join(sentences)
        
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        return summary
    
    def _extract_key_points(self, items: List[Dict]) -> List[str]:
        """핵심 포인트 추출"""
        key_points = []
        
        for item in items:
            content = item["content"]
            
            # 주얼리 관련 키워드가 포함된 문장들 찾기
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(keyword in sentence.lower() 
                       for keywords in self.topic_keywords.values() 
                       for keyword in keywords):
                    key_points.append(sentence)
        
        return list(set(key_points))[:10]  # 중복 제거 후 최대 10개
    
    def _compile_supporting_data(self, items: List[Dict]) -> List[Dict]:
        """지원 데이터 컴파일"""
        supporting_data = []
        
        for item in items:
            result = item["result"]
            
            data_entry = {
                "file_name": item["file_name"],
                "file_type": item["file_type"],
                "confidence": result.get("average_confidence", result.get("confidence", 0)),
                "processing_time": result.get("processing_time", 0),
                "content_length": len(item["content"])
            }
            
            # 추가 메타데이터
            if "jewelry_keywords" in result:
                data_entry["jewelry_keywords"] = result["jewelry_keywords"]
            
            supporting_data.append(data_entry)
        
        return supporting_data
    
    def _extract_key_insights(self, categorized_content: Dict[str, List[Dict]]) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        
        total_files = sum(len(items) for items in categorized_content.values())
        
        # 전체적인 인사이트
        insights.append(f"총 {total_files}개의 다양한 자료를 통해 주얼리 업계의 다면적 분석이 가능했습니다.")
        
        # 카테고리별 인사이트
        for category, items in categorized_content.items():
            if not items:
                continue
            
            category_insights = {
                "jewelry_business": f"주얼리 비즈니스 관련 {len(items)}개 자료에서 시장 경쟁력과 브랜드 전략의 중요성이 강조되었습니다.",
                "technical_process": f"기술 공정 관련 {len(items)}개 자료를 통해 제작 기술의 혁신과 품질 향상 방안을 확인했습니다.",
                "market_trend": f"시장 동향 {len(items)}개 자료에서 소비자 선호도 변화와 새로운 시장 기회를 발견했습니다.",
                "quality_certification": f"품질 인증 관련 {len(items)}개 자료에서 국제 표준의 중요성과 신뢰성 확보 방안을 확인했습니다.",
                "design_innovation": f"디자인 혁신 {len(items)}개 자료를 통해 창의적 접근법과 트렌드 반영의 필요성을 파악했습니다."
            }
            
            if category in category_insights:
                insights.append(category_insights[category])
        
        return insights
    
    def _generate_practical_applications(self, categorized_content: Dict[str, List[Dict]]) -> List[str]:
        """실용적 응용 방안 생성"""
        applications = []
        
        # 카테고리별 실용적 응용 방안
        category_applications = {
            "jewelry_business": [
                "시장 분석 결과를 활용한 마케팅 전략 수립",
                "경쟁사 분석을 통한 차별화 전략 개발",
                "고객 세그먼트별 맞춤형 제품 기획"
            ],
            "technical_process": [
                "제작 공정 최적화를 통한 효율성 향상",
                "품질 관리 시스템 구축 및 운영",
                "신기술 도입을 통한 경쟁력 강화"
            ],
            "market_trend": [
                "트렌드 예측을 통한 선제적 상품 개발",
                "소비자 니즈 분석 기반 서비스 개선",
                "새로운 시장 진출 전략 수립"
            ],
            "quality_certification": [
                "국제 인증 획득을 통한 신뢰성 확보",
                "품질 기준 정립 및 관리 체계 구축",
                "소비자 신뢰도 향상 방안 수립"
            ],
            "design_innovation": [
                "창의적 디자인 프로세스 도입",
                "고객 참여형 디자인 개발",
                "지속가능한 디자인 철학 구축"
            ]
        }
        
        for category, items in categorized_content.items():
            if items and category in category_applications:
                applications.extend(category_applications[category])
        
        return applications
    
    def _generate_conclusion(self, categorized_content: Dict[str, List[Dict]]) -> str:
        """결론 생성"""
        total_files = sum(len(items) for items in categorized_content.values())
        active_categories = [cat for cat, items in categorized_content.items() if items]
        
        conclusion = f"본 강의를 통해 {total_files}개의 다양한 자료를 분석하여 주얼리 업계의 "
        conclusion += f"{len(active_categories)}개 주요 분야에 대한 종합적인 이해를 얻을 수 있었습니다.\n\n"
        
        conclusion += "주요 성과:\n"
        conclusion += "• 다각도의 업계 분석을 통한 통찰력 획득\n"
        conclusion += "• 실무 적용 가능한 전략적 방향성 도출\n"
        conclusion += "• 미래 지향적 발전 방안 제시\n\n"
        
        conclusion += "앞으로도 지속적인 시장 모니터링과 혁신적 접근을 통해 "
        conclusion += "주얼리 업계의 성장과 발전에 기여할 수 있는 방안을 모색해야 할 것입니다."
        
        return conclusion
    
    def _compile_source_info(self, analysis_results: List[Dict[str, Any]]) -> List[Dict]:
        """소스 파일 정보 컴파일"""
        source_info = []
        
        for result in analysis_results:
            info = {
                "file_name": result.get("file_name", "unknown"),
                "status": result.get("status", "unknown"),
                "file_type": self._determine_file_type(result),
                "processing_time": result.get("processing_time", 0),
                "content_length": len(self._extract_text_content(result))
            }
            
            if "analysis_type" in result:
                info["analysis_method"] = result["analysis_type"]
            
            source_info.append(info)
        
        return source_info
    
    def _calculate_quality_score(self, lecture: Dict[str, Any]) -> float:
        """강의 품질 점수 계산"""
        score = 0
        
        # 기본 구조 완성도 (40%)
        if lecture["title"]: score += 10
        if lecture["overview"]: score += 10
        if lecture["main_topics"]: score += 10
        if lecture["conclusion"]: score += 10
        
        # 내용 풍부도 (40%)
        if len(lecture["detailed_content"]) > 0: score += 15
        if len(lecture["key_insights"]) > 0: score += 15
        if len(lecture["practical_applications"]) > 0: score += 10
        
        # 소스 다양성 (20%)
        file_types = set(src["file_type"] for src in lecture["source_files"])
        score += min(20, len(file_types) * 5)
        
        return min(100, score)

# 전역 컴파일러 인스턴스
lecture_compiler = LectureContentCompiler()

def compile_comprehensive_lecture(analysis_results: List[Dict[str, Any]], 
                                custom_title: str = None) -> Dict[str, Any]:
    """종합 강의 내용 컴파일 (편의 함수)"""
    return lecture_compiler.compile_lecture_content(analysis_results, custom_title)

if __name__ == "__main__":
    # 테스트 코드
    print("🎓 강의 내용 컴파일러 테스트")
    
    # 샘플 분석 결과
    sample_results = [
        {
            "status": "success",
            "file_name": "sample_audio.m4a",
            "full_text": "주얼리 디자인의 새로운 트렌드에 대해 논의하겠습니다. 다이아몬드 세팅 기술이 발전하고 있습니다.",
            "analysis_type": "real_whisper_stt",
            "processing_time": 15.2
        },
        {
            "status": "success", 
            "file_name": "jewelry_image.jpg",
            "full_text": "PREMIUM DIAMOND RING COLLECTION 2024",
            "analysis_type": "real_easyocr",
            "processing_time": 3.5
        }
    ]
    
    result = compile_comprehensive_lecture(sample_results, "테스트 강의")
    print(f"컴파일 결과: {result['status']}")
    
    if result["status"] == "success":
        lecture = result["lecture_content"]
        print(f"제목: {lecture['title']}")
        print(f"주요 주제 수: {len(lecture['main_topics'])}")
        print(f"품질 점수: {lecture['metadata']['quality_score']}")