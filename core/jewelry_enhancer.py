"""
솔로몬드 AI 시스템 - 주얼리 특화 STT 후처리 모듈
Jewelry Industry Specialized Speech-to-Text Enhancement Engine
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

class JewelrySTTEnhancer:
    """주얼리 업계 특화 STT 후처리 엔진"""
    
    def __init__(self, terms_db_path: Optional[str] = None):
        """
        초기화
        
        Args:
            terms_db_path: 주얼리 용어 데이터베이스 파일 경로
        """
        self.terms_db = None
        self.correction_cache = {}
        
        # 데이터베이스 파일 경로 설정
        if terms_db_path is None:
            current_dir = Path(__file__).parent.parent
            terms_db_path = current_dir / "data" / "jewelry_terms.json"
        
        self.load_terms_database(terms_db_path)
        
        # 성능 최적화를 위한 빠른 검색 인덱스 구축
        self._build_search_indices()
        
    def load_terms_database(self, file_path: str) -> bool:
        """주얼리 용어 데이터베이스 로드"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.terms_db = json.load(f)
                print(f"✅ 주얼리 용어 DB 로드 성공: {file_path}")
                return True
            else:
                print(f"⚠️ 용어 DB 파일 없음: {file_path}")
                # 기본 용어 사전 생성
                self.terms_db = self._create_minimal_terms_db()
                return False
        except Exception as e:
            print(f"❌ 용어 DB 로드 실패: {e}")
            self.terms_db = self._create_minimal_terms_db()
            return False
    
    def _create_minimal_terms_db(self) -> Dict:
        """최소한의 용어 데이터베이스 생성"""
        return {
            "jewelry_terms_db": {
                "version": "1.0-minimal",
                "common_corrections": {
                    "pronunciation_fixes": {
                        "다이몬드": "다이아몬드",
                        "디아몬드": "다이아몬드",
                        "새파이어": "사파이어",
                        "에머랄드": "에메랄드",
                        "캐럿": "캐럿",
                        "지아이에이": "GIA",
                        "포씨": "4C"
                    }
                }
            }
        }
    
    def _build_search_indices(self):
        """빠른 검색을 위한 인덱스 구축"""
        self.all_terms = []
        self.correction_map = {}
        
        if not self.terms_db or "jewelry_terms_db" not in self.terms_db:
            return
        
        db = self.terms_db["jewelry_terms_db"]
        
        # 모든 카테고리에서 용어 수집
        categories = ["precious_stones", "grading_4c", "grading_institutes", 
                     "business_terms", "technical_terms", "market_analysis", "education_terms"]
        
        for category in categories:
            if category in db:
                self._extract_terms_from_category(db[category])
        
        # 일반적인 수정사항 추가
        if "common_corrections" in db:
            corrections = db["common_corrections"]
            if "pronunciation_fixes" in corrections:
                self.correction_map.update(corrections["pronunciation_fixes"])
            if "common_mistakes" in corrections:
                self.correction_map.update(corrections["common_mistakes"])
        
        print(f"📚 인덱스 구축 완료: {len(self.all_terms)}개 용어, {len(self.correction_map)}개 수정사항")
    
    def _extract_terms_from_category(self, category_data: Dict):
        """카테고리에서 모든 용어 추출"""
        if isinstance(category_data, dict):
            for key, value in category_data.items():
                if isinstance(value, dict):
                    # 하위 카테고리 처리
                    for lang in ["korean", "english", "chinese"]:
                        if lang in value and isinstance(value[lang], list):
                            self.all_terms.extend(value[lang])
                    
                    # 잘못된 발음 수정
                    if "common_mistakes" in value and isinstance(value["common_mistakes"], list):
                        correct_term = value.get("korean", [])
                        if correct_term and isinstance(correct_term, list):
                            for mistake in value["common_mistakes"]:
                                self.correction_map[mistake] = correct_term[0]
                    
                    # 재귀적으로 하위 데이터 처리
                    self._extract_terms_from_category(value)
                elif isinstance(value, list):
                    self.all_terms.extend(value)
    
    def enhance_transcription(self, transcribed_text: str, 
                            detected_language: str = "ko",
                            confidence_threshold: float = 0.7) -> Dict:
        """
        STT 결과를 주얼리 업계 특화로 개선
        
        Args:
            transcribed_text: 원본 STT 결과
            detected_language: 감지된 언어
            confidence_threshold: 수정 신뢰도 임계값
            
        Returns:
            개선된 결과 딕셔너리
        """
        if not transcribed_text.strip():
            return {
                "original_text": transcribed_text,
                "enhanced_text": transcribed_text,
                "corrections": [],
                "detected_terms": [],
                "confidence": 1.0
            }
        
        enhanced_text = transcribed_text
        corrections = []
        detected_terms = []
        
        # 1. 직접적인 용어 수정
        enhanced_text, direct_corrections = self._apply_direct_corrections(enhanced_text)
        corrections.extend(direct_corrections)
        
        # 2. 퍼지 매칭을 통한 유사 용어 수정
        enhanced_text, fuzzy_corrections = self._apply_fuzzy_corrections(
            enhanced_text, confidence_threshold
        )
        corrections.extend(fuzzy_corrections)
        
        # 3. 주얼리 용어 식별
        detected_terms = self._detect_jewelry_terms(enhanced_text)
        
        # 4. 문맥 기반 개선
        enhanced_text, context_corrections = self._apply_context_corrections(enhanced_text)
        corrections.extend(context_corrections)
        
        # 5. 전체 신뢰도 계산
        confidence = self._calculate_enhancement_confidence(
            transcribed_text, enhanced_text, corrections
        )
        
        return {
            "original_text": transcribed_text,
            "enhanced_text": enhanced_text,
            "corrections": corrections,
            "detected_terms": detected_terms,
            "confidence": confidence,
            "language": detected_language,
            "terms_count": len(detected_terms),
            "corrections_count": len(corrections)
        }
    
    def _apply_direct_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """직접적인 용어 수정 적용"""
        corrections = []
        enhanced_text = text
        
        for wrong_term, correct_term in self.correction_map.items():
            if wrong_term in enhanced_text:
                enhanced_text = enhanced_text.replace(wrong_term, correct_term)
                corrections.append({
                    "type": "direct_correction",
                    "original": wrong_term,
                    "corrected": correct_term,
                    "confidence": 0.95
                })
        
        return enhanced_text, corrections
    
    def _apply_fuzzy_corrections(self, text: str, threshold: float) -> Tuple[str, List[Dict]]:
        """퍼지 매칭을 통한 유사 용어 수정"""
        corrections = []
        words = text.split()
        enhanced_words = []
        
        for word in words:
            # 정확한 매치 확인
            if word in self.all_terms:
                enhanced_words.append(word)
                continue
            
            # 퍼지 매칭으로 유사한 용어 찾기
            best_match, similarity = self._find_best_fuzzy_match(word, threshold)
            
            if best_match and similarity >= threshold:
                enhanced_words.append(best_match)
                corrections.append({
                    "type": "fuzzy_correction", 
                    "original": word,
                    "corrected": best_match,
                    "confidence": similarity
                })
            else:
                enhanced_words.append(word)
        
        return " ".join(enhanced_words), corrections
    
    def _find_best_fuzzy_match(self, word: str, threshold: float) -> Tuple[Optional[str], float]:
        """가장 유사한 용어 찾기"""
        best_match = None
        best_similarity = 0.0
        
        for term in self.all_terms:
            if isinstance(term, str) and len(term) > 1:
                similarity = SequenceMatcher(None, word.lower(), term.lower()).ratio()
                
                # 길이 차이가 너무 크면 패스
                if abs(len(word) - len(term)) > max(len(word), len(term)) * 0.5:
                    continue
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = term
        
        return best_match, best_similarity
    
    def _detect_jewelry_terms(self, text: str) -> List[Dict]:
        """텍스트에서 주얼리 용어 식별"""
        detected_terms = []
        text_lower = text.lower()
        
        for term in self.all_terms:
            if isinstance(term, str) and term.lower() in text_lower:
                # 용어 카테고리 식별
                category = self._identify_term_category(term)
                detected_terms.append({
                    "term": term,
                    "category": category,
                    "position": text_lower.find(term.lower())
                })
        
        # 위치순으로 정렬
        detected_terms.sort(key=lambda x: x["position"])
        return detected_terms
    
    def _identify_term_category(self, term: str) -> str:
        """용어의 카테고리 식별"""
        if not self.terms_db or "jewelry_terms_db" not in self.terms_db:
            return "unknown"
        
        db = self.terms_db["jewelry_terms_db"]
        
        # 각 카테고리에서 용어 검색
        category_map = {
            "precious_stones": "보석",
            "grading_4c": "등급",
            "grading_institutes": "감정기관",
            "business_terms": "비즈니스",
            "technical_terms": "기술",
            "market_analysis": "시장분석",
            "education_terms": "교육"
        }
        
        for category_key, category_name in category_map.items():
            if self._term_in_category(term, db.get(category_key, {})):
                return category_name
        
        return "기타"
    
    def _term_in_category(self, term: str, category_data: Dict) -> bool:
        """특정 카테고리에 용어가 있는지 확인"""
        term_lower = term.lower()
        
        def search_recursive(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in ["korean", "english", "chinese"] and isinstance(value, list):
                        if any(term_lower == t.lower() for t in value if isinstance(t, str)):
                            return True
                    elif isinstance(value, (dict, list)):
                        if search_recursive(value):
                            return True
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and term_lower == item.lower():
                        return True
                    elif isinstance(item, dict) and search_recursive(item):
                        return True
            return False
        
        return search_recursive(category_data)
    
    def _apply_context_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """문맥 기반 용어 수정"""
        corrections = []
        enhanced_text = text
        
        # 주얼리 업계 특화 문맥 패턴
        context_patterns = [
            (r'(\d+)\s*c', r'\1캐럿', "무게 단위 정규화"),
            (r'(\d+)\s*k(?!\w)', r'\1K', "금 순도 정규화"),
            (r'pt\s*(\d+)', r'PT\1', "플래티넘 순도 정규화"),
            (r'vs\s*(\d+)', r'VS\1', "다이아몬드 등급 정규화"),
            (r'vvs\s*(\d+)', r'VVS\1', "다이아몬드 등급 정규화"),
            (r'지아이에이', r'GIA', "감정기관명 정규화"),
            (r'에이지에스', r'AGS', "감정기관명 정규화")
        ]
        
        for pattern, replacement, description in context_patterns:
            original_text = enhanced_text
            enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
            
            if original_text != enhanced_text:
                corrections.append({
                    "type": "context_correction",
                    "description": description,
                    "pattern": pattern,
                    "confidence": 0.9
                })
        
        return enhanced_text, corrections
    
    def _calculate_enhancement_confidence(self, original: str, enhanced: str, corrections: List[Dict]) -> float:
        """개선 신뢰도 계산"""
        if original == enhanced:
            return 1.0
        
        # 수정사항의 평균 신뢰도 계산
        if corrections:
            avg_confidence = sum(c.get("confidence", 0.7) for c in corrections) / len(corrections)
            return avg_confidence
        
        return 0.8  # 기본 신뢰도
    
    def analyze_jewelry_content(self, enhanced_result: Dict) -> Dict:
        """주얼리 콘텐츠 심층 분석"""
        text = enhanced_result.get("enhanced_text", "")
        detected_terms = enhanced_result.get("detected_terms", [])
        
        # 카테고리별 용어 분석
        category_analysis = {}
        for term_info in detected_terms:
            category = term_info["category"]
            if category not in category_analysis:
                category_analysis[category] = []
            category_analysis[category].append(term_info["term"])
        
        # 주제 식별
        topics = self._identify_topics(text, detected_terms)
        
        # 비즈니스 인사이트 추출
        insights = self._extract_business_insights(text, detected_terms)
        
        return {
            "category_analysis": category_analysis,
            "identified_topics": topics,
            "business_insights": insights,
            "technical_level": self._assess_technical_level(detected_terms),
            "language_complexity": self._assess_language_complexity(text)
        }
    
    def _identify_topics(self, text: str, detected_terms: List[Dict]) -> List[str]:
        """주요 주제 식별"""
        topics = []
        
        # 주제별 키워드 매핑
        topic_keywords = {
            "다이아몬드 등급평가": ["4C", "캐럿", "컷", "컬러", "클래리티", "GIA"],
            "보석 거래": ["도매가", "소매가", "할인", "재고", "주문"],
            "제품 기술": ["세팅", "가공", "연마", "표면처리"],
            "시장 분석": ["트렌드", "유행", "인기", "시장"],
            "고객 상담": ["추천", "상담", "선택", "구매"],
            "국제 무역": ["FOB", "수출", "수입", "통관", "관세"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if keyword_count >= 2:  # 2개 이상의 관련 키워드가 있으면 주제로 인식
                topics.append(topic)
        
        return topics
    
    def _extract_business_insights(self, text: str, detected_terms: List[Dict]) -> List[str]:
        """비즈니스 인사이트 추출"""
        insights = []
        
        # 가격 관련 언급
        if any(term["category"] == "비즈니스" for term in detected_terms):
            if "할인" in text or "세일" in text:
                insights.append("가격 할인 이벤트 관련 논의")
            if "재고" in text:
                insights.append("재고 관리 관련 논의")
        
        # 품질 관련 언급
        if any(term["category"] == "등급" for term in detected_terms):
            insights.append("품질 등급 및 평가 기준 논의")
        
        # 기술 관련 언급
        if any(term["category"] == "기술" for term in detected_terms):
            insights.append("기술적 세부사항 및 제조 과정 논의")
        
        return insights
    
    def _assess_technical_level(self, detected_terms: List[Dict]) -> str:
        """기술적 복잡도 평가"""
        technical_terms = [t for t in detected_terms if t["category"] in ["등급", "기술", "감정기관"]]
        
        if len(technical_terms) >= 5:
            return "고급"
        elif len(technical_terms) >= 2:
            return "중급"
        else:
            return "초급"
    
    def _assess_language_complexity(self, text: str) -> str:
        """언어 복잡도 평가"""
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length > 15:
            return "복잡"
        elif avg_sentence_length > 8:
            return "보통"
        else:
            return "단순"
    
    def generate_jewelry_summary(self, enhanced_result: Dict, analysis: Dict) -> str:
        """주얼리 업계 특화 요약 생성"""
        text = enhanced_result.get("enhanced_text", "")
        topics = analysis.get("identified_topics", [])
        category_analysis = analysis.get("category_analysis", {})
        
        summary_parts = []
        
        # 주요 주제
        if topics:
            summary_parts.append(f"🎯 주요 주제: {', '.join(topics)}")
        
        # 카테고리별 용어
        for category, terms in category_analysis.items():
            if terms:
                unique_terms = list(set(terms))
                summary_parts.append(f"📚 {category}: {', '.join(unique_terms[:3])}")
        
        # 비즈니스 인사이트
        insights = analysis.get("business_insights", [])
        if insights:
            summary_parts.append(f"💡 인사이트: {insights[0]}")
        
        # 요약문 생성
        if summary_parts:
            return "\\n".join(summary_parts)
        else:
            return "주얼리 관련 일반적인 논의가 진행되었습니다."
    
    def get_enhancement_stats(self) -> Dict:
        """개선 엔진 통계 정보"""
        return {
            "terms_database_version": self.terms_db.get("jewelry_terms_db", {}).get("version", "unknown"),
            "total_terms": len(self.all_terms),
            "correction_rules": len(self.correction_map),
            "categories": ["보석", "등급", "감정기관", "비즈니스", "기술", "시장분석", "교육"],
            "supported_languages": ["한국어", "영어", "중국어"],
            "features": [
                "직접 용어 수정",
                "퍼지 매칭 수정", 
                "문맥 기반 정규화",
                "주얼리 용어 식별",
                "주제 분석",
                "비즈니스 인사이트 추출",
                "업계 특화 요약"
            ]
        }

# 전역 인스턴스 (싱글톤 패턴)
_jewelry_enhancer_instance = None

def get_jewelry_enhancer() -> JewelrySTTEnhancer:
    """전역 주얼리 STT 개선 엔진 인스턴스 반환"""
    global _jewelry_enhancer_instance
    if _jewelry_enhancer_instance is None:
        _jewelry_enhancer_instance = JewelrySTTEnhancer()
    return _jewelry_enhancer_instance

def enhance_jewelry_transcription(transcribed_text: str, 
                                detected_language: str = "ko",
                                include_analysis: bool = True) -> Dict:
    """주얼리 특화 STT 결과 개선 (편의 함수)"""
    enhancer = get_jewelry_enhancer()
    
    # 기본 개선
    enhanced_result = enhancer.enhance_transcription(transcribed_text, detected_language)
    
    # 심층 분석 추가
    if include_analysis:
        analysis = enhancer.analyze_jewelry_content(enhanced_result)
        enhanced_result["analysis"] = analysis
        enhanced_result["summary"] = enhancer.generate_jewelry_summary(enhanced_result, analysis)
    
    return enhanced_result
