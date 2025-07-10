"""
솔로몬드 AI 시스템 - 한국어 최종 통합 요약 엔진
다국어 콘텐츠를 한국어로 통합 번역 및 주얼리 특화 요약 생성 모듈
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import re
from datetime import datetime
from collections import defaultdict, Counter
import openai
from pathlib import Path

# 번역 및 언어 감지 라이브러리
try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

# 주얼리 AI 엔진 import
from .jewelry_ai_engine import JewelryAIEngine

class KoreanFinalSummarizer:
    """다국어 → 한국어 통합 요약 생성 클래스"""
    
    def __init__(self):
        # 언어 코드 매핑
        self.language_mapping = {
            "ko": "한국어",
            "en": "영어", 
            "zh": "중국어",
            "zh-cn": "중국어(간체)",
            "zh-tw": "중국어(번체)",
            "ja": "일본어",
            "es": "스페인어",
            "fr": "프랑스어",
            "de": "독일어",
            "ru": "러시아어"
        }
        
        # 주얼리 전문 용어 한국어 매핑
        self.jewelry_term_mapping = {
            # 영어 → 한국어
            "diamond": "다이아몬드",
            "ruby": "루비",
            "sapphire": "사파이어", 
            "emerald": "에메랄드",
            "pearl": "진주",
            "gold": "금",
            "silver": "은",
            "platinum": "백금",
            "carat": "캐럿",
            "clarity": "투명도",
            "color": "컬러",
            "cut": "컷",
            "certification": "감정서",
            "GIA": "GIA",
            "AGS": "AGS",
            "setting": "세팅",
            "mounting": "마운팅",
            "prong": "프롱",
            "bezel": "베젤",
            "wholesale": "도매",
            "retail": "소매",
            "appraisal": "감정",
            "gemstone": "보석",
            "jewelry": "주얼리",
            "ring": "반지",
            "necklace": "목걸이",
            "earring": "귀걸이",
            "bracelet": "팔찌",
            "pendant": "펜던트",
            
            # 중국어 → 한국어 (간체)
            "钻石": "다이아몬드",
            "红宝石": "루비",
            "蓝宝石": "사파이어",
            "祖母绿": "에메랄드",
            "珍珠": "진주",
            "黄金": "금",
            "白银": "은",
            "铂金": "백금",
            "克拉": "캐럿",
            "净度": "투명도",
            "颜色": "컬러",
            "切工": "컷",
            "证书": "감정서",
            "珠宝": "주얼리",
            "戒指": "반지",
            "项链": "목걸이",
            "耳环": "귀걸이",
            
            # 일본어 → 한국어
            "ダイヤモンド": "다이아몬드",
            "ルビー": "루비",
            "サファイア": "사파이어",
            "エメラルド": "에메랄드",
            "真珠": "진주",
            "金": "금",
            "銀": "은",
            "プラチナ": "백금",
            "カラット": "캐럿",
            "透明度": "투명도",
            "カラー": "컬러",
            "カット": "컷",
            "鑑定書": "감정서",
            "ジュエリー": "주얼리",
            "指輪": "반지",
            "ネックレス": "목걸이",
            "イヤリング": "귀걸이"
        }
        
        # 번역기 초기화
        self.translator = None
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translator = Translator()
            except Exception as e:
                logging.warning(f"Google Translator 초기화 실패: {e}")
        
        # 주얼리 AI 엔진
        self.jewelry_ai = JewelryAIEngine()
        
        # 요약 스타일 템플릿
        self.summary_styles = {
            "comprehensive": {
                "title": "종합 분석 요약",
                "sections": ["핵심 내용", "주요 논점", "결론 및 시사점", "실행 방안"],
                "tone": "전문적이고 상세한"
            },
            "executive": {
                "title": "경영진 요약",
                "sections": ["핵심 메시지", "비즈니스 임팩트", "의사결정 포인트", "다음 단계"],
                "tone": "간결하고 핵심적인"
            },
            "technical": {
                "title": "기술적 요약", 
                "sections": ["기술적 내용", "전문 용어 해설", "기술적 시사점", "적용 방안"],
                "tone": "기술적이고 전문적인"
            },
            "business": {
                "title": "비즈니스 요약",
                "sections": ["시장 동향", "비즈니스 기회", "위험 요소", "전략적 제안"],
                "tone": "비즈니스 중심의 실용적인"
            }
        }
        
        logging.info("한국어 최종 통합 요약 엔진 초기화 완료")
    
    async def process_multilingual_session(self, 
                                         session_data: Dict,
                                         summary_style: str = "comprehensive",
                                         preserve_original: bool = True) -> Dict:
        """
        다국어 세션 데이터를 한국어로 통합 요약
        
        Args:
            session_data: 다국어 세션 데이터
            summary_style: 요약 스타일 ("comprehensive", "executive", "technical", "business")
            preserve_original: 원문 보존 여부
            
        Returns:
            한국어 통합 요약 결과
        """
        try:
            print(f"🇰🇷 다국어 → 한국어 통합 요약 시작")
            
            # 1. 다국어 콘텐츠 수집 및 분석
            multilingual_content = await self._collect_multilingual_content(session_data)
            
            # 2. 언어별 콘텐츠 번역
            translated_content = await self._translate_to_korean(multilingual_content)
            
            # 3. 주얼리 용어 정규화
            normalized_content = await self._normalize_jewelry_terms(translated_content)
            
            # 4. 내용 통합 및 중복 제거
            integrated_content = await self._integrate_content(normalized_content)
            
            # 5. 한국어 요약 생성
            korean_summary = await self._generate_korean_summary(
                integrated_content, 
                summary_style,
                session_data.get("session_type", "meeting")
            )
            
            # 6. 품질 평가 및 후처리
            quality_assessment = await self._assess_summary_quality(korean_summary, integrated_content)
            
            # 7. 최종 결과 구성
            result = {
                "success": True,
                "session_info": {
                    "session_id": session_data.get("session_id", "unknown"),
                    "session_type": session_data.get("session_type", "meeting"),
                    "processed_languages": list(multilingual_content.keys()),
                    "total_sources": sum(len(sources) for sources in multilingual_content.values())
                },
                "translation_analysis": {
                    "detected_languages": {lang: len(sources) for lang, sources in multilingual_content.items()},
                    "translation_quality": self._calculate_translation_quality(translated_content),
                    "jewelry_terms_mapped": len(self._get_mapped_terms(normalized_content))
                },
                "korean_summary": korean_summary,
                "quality_assessment": quality_assessment,
                "processing_metadata": {
                    "summary_style": summary_style,
                    "preserve_original": preserve_original,
                    "generated_at": datetime.now().isoformat(),
                    "processing_time": "실시간 계산됨"
                }
            }
            
            # 원문 보존 옵션
            if preserve_original:
                result["original_content"] = {
                    "multilingual_content": multilingual_content,
                    "translated_content": translated_content,
                    "normalized_content": normalized_content
                }
            
            print(f"✅ 한국어 통합 요약 완료: {len(korean_summary.get('content', ''))}자")
            return result
            
        except Exception as e:
            logging.error(f"다국어 → 한국어 통합 요약 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_data.get("session_id", "unknown")
            }
    
    async def _collect_multilingual_content(self, session_data: Dict) -> Dict[str, List[Dict]]:
        """다국어 콘텐츠 수집 및 언어별 분류"""
        try:
            multilingual_content = defaultdict(list)
            
            # 각 소스에서 텍스트 추출
            sources = [
                ("audio", session_data.get("audio_files", [])),
                ("video", session_data.get("video_files", [])), 
                ("documents", session_data.get("document_files", [])),
                ("web", session_data.get("web_urls", []))
            ]
            
            for source_type, source_list in sources:
                for item in source_list:
                    # 텍스트 추출
                    text_content = self._extract_text_from_source(item, source_type)
                    
                    if text_content and len(text_content.strip()) > 10:
                        # 언어 감지
                        detected_lang = await self._detect_language(text_content)
                        
                        multilingual_content[detected_lang].append({
                            "source_type": source_type,
                            "content": text_content,
                            "confidence": self._estimate_detection_confidence(text_content, detected_lang),
                            "word_count": len(text_content.split()),
                            "source_info": item.get("filename", item.get("url", "unknown"))
                        })
            
            return dict(multilingual_content)
            
        except Exception as e:
            logging.error(f"다국어 콘텐츠 수집 오류: {e}")
            return {}
    
    def _extract_text_from_source(self, item: Dict, source_type: str) -> str:
        """소스에서 텍스트 추출"""
        try:
            if source_type == "audio":
                return item.get("enhanced_text", item.get("text", ""))
            elif source_type == "video":
                return item.get("enhanced_text", item.get("text", ""))
            elif source_type == "documents":
                return item.get("text", item.get("content", ""))
            elif source_type == "web":
                return item.get("content", item.get("text", ""))
            else:
                return item.get("text", item.get("content", ""))
        except Exception as e:
            logging.error(f"텍스트 추출 오류 ({source_type}): {e}")
            return ""
    
    async def _detect_language(self, text: str) -> str:
        """텍스트 언어 감지"""
        try:
            if not LANGDETECT_AVAILABLE:
                # 간단한 휴리스틱 방법
                return self._simple_language_detection(text)
            
            # langdetect 사용
            detected = detect(text)
            
            # 신뢰도가 낮거나 감지 실패 시 휴리스틱 사용
            if detected == "ca" or len(text) < 50:  # 카탈로니아어로 잘못 감지되는 경우가 많음
                return self._simple_language_detection(text)
            
            return detected
            
        except (LangDetectError, Exception) as e:
            logging.warning(f"언어 감지 실패: {e}")
            return self._simple_language_detection(text)
    
    def _simple_language_detection(self, text: str) -> str:
        """간단한 휴리스틱 언어 감지"""
        try:
            # 한글 체크
            korean_chars = re.findall(r'[ㄱ-ㅎ가-힣]', text)
            if len(korean_chars) > len(text) * 0.3:
                return "ko"
            
            # 중국어 체크 (한자)
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            if len(chinese_chars) > len(text) * 0.3:
                return "zh"
            
            # 일본어 체크 (히라가나, 카타카나)
            japanese_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text)
            if len(japanese_chars) > len(text) * 0.2:
                return "ja"
            
            # 영어로 기본 설정
            return "en"
            
        except Exception as e:
            logging.error(f"간단한 언어 감지 오류: {e}")
            return "en"
    
    def _estimate_detection_confidence(self, text: str, detected_lang: str) -> float:
        """언어 감지 신뢰도 추정"""
        try:
            text_length = len(text)
            
            # 텍스트 길이 기반 신뢰도
            if text_length < 20:
                length_confidence = 0.3
            elif text_length < 100:
                length_confidence = 0.7
            else:
                length_confidence = 0.9
            
            # 언어별 특성 확인
            if detected_lang == "ko":
                korean_ratio = len(re.findall(r'[ㄱ-ㅎ가-힣]', text)) / max(len(text), 1)
                lang_confidence = korean_ratio
            elif detected_lang == "zh":
                chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', text)) / max(len(text), 1)
                lang_confidence = chinese_ratio
            elif detected_lang == "ja":
                japanese_ratio = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text)) / max(len(text), 1)
                lang_confidence = japanese_ratio
            else:  # 영어 등
                ascii_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
                lang_confidence = ascii_ratio
            
            return round((length_confidence + lang_confidence) / 2, 3)
            
        except Exception as e:
            logging.error(f"언어 감지 신뢰도 추정 오류: {e}")
            return 0.5
    
    async def _translate_to_korean(self, multilingual_content: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """다국어 콘텐츠를 한국어로 번역"""
        try:
            translated_content = {}
            
            for lang, sources in multilingual_content.items():
                if lang == "ko":
                    # 이미 한국어인 경우 그대로 유지
                    translated_content[lang] = sources
                else:
                    # 다른 언어는 한국어로 번역
                    translated_sources = []
                    
                    for source in sources:
                        original_text = source["content"]
                        
                        # 주얼리 용어 사전 기반 번역
                        translated_text = await self._translate_with_jewelry_context(
                            original_text, lang, "ko"
                        )
                        
                        translated_source = source.copy()
                        translated_source.update({
                            "original_content": original_text,
                            "translated_content": translated_text,
                            "original_language": lang,
                            "translation_method": "jewelry_enhanced"
                        })
                        
                        translated_sources.append(translated_source)
                    
                    translated_content[lang] = translated_sources
            
            return translated_content
            
        except Exception as e:
            logging.error(f"한국어 번역 오류: {e}")
            return multilingual_content
    
    async def _translate_with_jewelry_context(self, text: str, source_lang: str, target_lang: str) -> str:
        """주얼리 컨텍스트를 고려한 번역"""
        try:
            # 1. 주얼리 용어 사전 매핑 적용
            translated_text = self._apply_jewelry_term_mapping(text)
            
            # 2. Google Translate로 나머지 번역
            if self.translator and source_lang != target_lang:
                try:
                    # 긴 텍스트는 청크로 분할하여 번역
                    if len(text) > 5000:
                        chunks = self._split_text_into_chunks(translated_text, 4000)
                        translated_chunks = []
                        
                        for chunk in chunks:
                            translated_chunk = self.translator.translate(
                                chunk, src=source_lang, dest=target_lang
                            ).text
                            translated_chunks.append(translated_chunk)
                        
                        translated_text = " ".join(translated_chunks)
                    else:
                        translated_text = self.translator.translate(
                            translated_text, src=source_lang, dest=target_lang
                        ).text
                        
                except Exception as translate_error:
                    logging.warning(f"Google Translate 오류: {translate_error}")
                    # 번역 실패 시 용어 매핑만 적용된 텍스트 반환
            
            # 3. 번역 후 주얼리 용어 재정규화
            final_text = self._post_process_translation(translated_text)
            
            return final_text
            
        except Exception as e:
            logging.error(f"주얼리 컨텍스트 번역 오류: {e}")
            return text
    
    def _apply_jewelry_term_mapping(self, text: str) -> str:
        """주얼리 용어 사전 매핑 적용"""
        try:
            mapped_text = text
            
            # 용어 매핑 적용 (대소문자 구분 없이)
            for original_term, korean_term in self.jewelry_term_mapping.items():
                # 단어 경계를 고려한 치환
                pattern = r'\b' + re.escape(original_term) + r'\b'
                mapped_text = re.sub(pattern, korean_term, mapped_text, flags=re.IGNORECASE)
            
            return mapped_text
            
        except Exception as e:
            logging.error(f"주얼리 용어 매핑 오류: {e}")
            return text
    
    def _split_text_into_chunks(self, text: str, max_length: int = 4000) -> List[str]:
        """텍스트를 청크로 분할"""
        try:
            # 문장 단위로 분할
            sentences = re.split(r'[.!?。！？]\s+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < max_length:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logging.error(f"텍스트 청크 분할 오류: {e}")
            return [text]
    
    def _post_process_translation(self, translated_text: str) -> str:
        """번역 후처리"""
        try:
            # 일반적인 번역 오류 수정
            corrections = {
                "다이아 몬드": "다이아몬드",
                "루 비": "루비", 
                "사파 이어": "사파이어",
                "에메랄 드": "에메랄드",
                "캐 럿": "캐럿",
                "지 아": "GIA",
                "에이지에스": "AGS",
                "보석류": "주얼리",
                "귀중품": "주얼리"
            }
            
            processed_text = translated_text
            for wrong, correct in corrections.items():
                processed_text = processed_text.replace(wrong, correct)
            
            # 불필요한 공백 정리
            processed_text = re.sub(r'\s+', ' ', processed_text)
            processed_text = processed_text.strip()
            
            return processed_text
            
        except Exception as e:
            logging.error(f"번역 후처리 오류: {e}")
            return translated_text
    
    async def _normalize_jewelry_terms(self, translated_content: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """주얼리 용어 정규화"""
        try:
            normalized_content = {}
            
            for lang, sources in translated_content.items():
                normalized_sources = []
                
                for source in sources:
                    # 번역된 텍스트 또는 원본 텍스트 사용
                    content = source.get("translated_content", source.get("content", ""))
                    
                    # 주얼리 용어 정규화
                    normalized_text = self._normalize_jewelry_vocabulary(content)
                    
                    normalized_source = source.copy()
                    normalized_source["normalized_content"] = normalized_text
                    normalized_source["jewelry_terms"] = self._extract_jewelry_terms(normalized_text)
                    
                    normalized_sources.append(normalized_source)
                
                normalized_content[lang] = normalized_sources
            
            return normalized_content
            
        except Exception as e:
            logging.error(f"주얼리 용어 정규화 오류: {e}")
            return translated_content
    
    def _normalize_jewelry_vocabulary(self, text: str) -> str:
        """주얼리 어휘 정규화"""
        try:
            # 주얼리 업계 표준 용어로 통일
            vocabulary_mapping = {
                # 다이아몬드 등급 관련
                "투명도": "클래리티",
                "명도": "클래리티", 
                "투명성": "클래리티",
                "색상": "컬러",
                "색깔": "컬러",
                "절단": "컷",
                "커팅": "컷",
                "절삭": "컷",
                "무게": "캐럿",
                "중량": "캐럿",
                
                # 보석 종류
                "다이아": "다이아몬드",
                "다이야몬드": "다이아몬드",
                "홍옥": "루비",
                "청옥": "사파이어",
                "녹주석": "에메랄드",
                
                # 금속 종류
                "황금": "금",
                "순금": "금",
                "백은": "은",
                "순은": "은", 
                "백금": "플래티넘",
                
                # 주얼리 종류
                "목걸이": "네크리스",
                "목거리": "네크리스",
                "팔걸이": "브레이슬릿",
                "손목걸이": "브레이슬릿"
            }
            
            normalized_text = text
            for old_term, new_term in vocabulary_mapping.items():
                normalized_text = normalized_text.replace(old_term, new_term)
            
            return normalized_text
            
        except Exception as e:
            logging.error(f"주얼리 어휘 정규화 오류: {e}")
            return text
    
    def _extract_jewelry_terms(self, text: str) -> List[str]:
        """텍스트에서 주얼리 용어 추출"""
        try:
            jewelry_terms = []
            text_lower = text.lower()
            
            # 주얼리 용어 리스트
            all_terms = list(self.jewelry_term_mapping.values()) + list(self.jewelry_term_mapping.keys())
            
            for term in all_terms:
                if term.lower() in text_lower:
                    jewelry_terms.append(term)
            
            return list(set(jewelry_terms))
            
        except Exception as e:
            logging.error(f"주얼리 용어 추출 오류: {e}")
            return []
    
    async def _integrate_content(self, normalized_content: Dict[str, List[Dict]]) -> Dict:
        """콘텐츠 통합 및 중복 제거"""
        try:
            # 모든 소스의 정규화된 콘텐츠 수집
            all_content = []
            source_mapping = {}
            
            for lang, sources in normalized_content.items():
                for idx, source in enumerate(sources):
                    content = source.get("normalized_content", "")
                    if content and len(content.strip()) > 20:
                        source_key = f"{lang}_{idx}"
                        all_content.append({
                            "key": source_key,
                            "content": content,
                            "source_type": source.get("source_type", "unknown"),
                            "word_count": len(content.split()),
                            "jewelry_terms": source.get("jewelry_terms", []),
                            "original_language": lang
                        })
                        source_mapping[source_key] = source
            
            # 중복 콘텐츠 제거
            deduplicated_content = self._remove_duplicate_content(all_content)
            
            # 주제별 그룹화
            topic_groups = self._group_content_by_topics(deduplicated_content)
            
            # 시간순 정렬 (가능한 경우)
            chronological_content = self._arrange_chronologically(deduplicated_content)
            
            # 통합 텍스트 생성
            integrated_text = self._merge_content_intelligently(deduplicated_content, topic_groups)
            
            return {
                "integrated_text": integrated_text,
                "total_word_count": len(integrated_text.split()),
                "source_count": len(deduplicated_content),
                "topic_groups": topic_groups,
                "chronological_order": chronological_content,
                "language_distribution": self._calculate_language_distribution(deduplicated_content),
                "jewelry_terms_summary": self._summarize_jewelry_terms(deduplicated_content)
            }
            
        except Exception as e:
            logging.error(f"콘텐츠 통합 오류: {e}")
            return {"integrated_text": "", "error": str(e)}
    
    def _remove_duplicate_content(self, all_content: List[Dict]) -> List[Dict]:
        """중복 콘텐츠 제거"""
        try:
            unique_content = []
            seen_hashes = set()
            
            for item in all_content:
                content = item["content"]
                
                # 콘텐츠 해시 생성 (첫 100자 + 마지막 100자)
                content_hash = hash(content[:100] + content[-100:] if len(content) > 200 else content)
                
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_content.append(item)
                else:
                    # 중복 발견 시 더 긴 콘텐츠 선택
                    for i, existing in enumerate(unique_content):
                        if hash(existing["content"][:100] + existing["content"][-100:] if len(existing["content"]) > 200 else existing["content"]) == content_hash:
                            if len(content) > len(existing["content"]):
                                unique_content[i] = item
                            break
            
            return unique_content
            
        except Exception as e:
            logging.error(f"중복 콘텐츠 제거 오류: {e}")
            return all_content
    
    def _group_content_by_topics(self, content_list: List[Dict]) -> Dict[str, List[Dict]]:
        """주제별 콘텐츠 그룹화"""
        try:
            topic_groups = defaultdict(list)
            
            # 주제 키워드 정의
            topic_keywords = {
                "product_analysis": ["제품", "상품", "다이아몬드", "보석", "품질", "등급"],
                "market_trends": ["시장", "트렌드", "가격", "동향", "수요", "공급"],
                "business_strategy": ["전략", "계획", "목표", "방향", "정책", "사업"],
                "technology": ["기술", "기법", "방법", "공정", "제조", "가공"],
                "certification": ["감정", "인증", "GIA", "AGS", "감정서", "증명"],
                "customer_service": ["고객", "서비스", "상담", "응대", "만족", "관리"]
            }
            
            for item in content_list:
                content = item["content"].lower()
                max_matches = 0
                best_topic = "general"
                
                for topic, keywords in topic_keywords.items():
                    matches = sum(1 for keyword in keywords if keyword in content)
                    if matches > max_matches:
                        max_matches = matches
                        best_topic = topic
                
                topic_groups[best_topic].append(item)
            
            return dict(topic_groups)
            
        except Exception as e:
            logging.error(f"주제별 그룹화 오류: {e}")
            return {"general": content_list}
    
    def _arrange_chronologically(self, content_list: List[Dict]) -> List[Dict]:
        """시간순 정렬"""
        try:
            # 소스 타입별 우선순위 (일반적인 회의 진행 순서)
            type_priority = {
                "audio": 1,  # 주 발표/회의 내용
                "video": 2,  # 비디오 발표
                "documents": 3,  # 발표 자료
                "web": 4  # 참고 자료
            }
            
            # 우선순위와 단어 수 기준으로 정렬
            sorted_content = sorted(
                content_list,
                key=lambda x: (type_priority.get(x["source_type"], 5), -x["word_count"])
            )
            
            return sorted_content
            
        except Exception as e:
            logging.error(f"시간순 정렬 오류: {e}")
            return content_list
    
    def _merge_content_intelligently(self, content_list: List[Dict], topic_groups: Dict) -> str:
        """지능적 콘텐츠 병합"""
        try:
            merged_sections = []
            
            # 주제별로 콘텐츠 병합
            for topic, items in topic_groups.items():
                if not items:
                    continue
                
                topic_content = []
                for item in items:
                    content = item["content"].strip()
                    if content:
                        topic_content.append(content)
                
                if topic_content:
                    section_text = " ".join(topic_content)
                    merged_sections.append(section_text)
            
            # 전체 통합 텍스트
            integrated_text = "\n\n".join(merged_sections)
            
            # 중복 문장 제거
            integrated_text = self._remove_duplicate_sentences(integrated_text)
            
            return integrated_text
            
        except Exception as e:
            logging.error(f"지능적 콘텐츠 병합 오류: {e}")
            return " ".join([item["content"] for item in content_list])
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """중복 문장 제거"""
        try:
            sentences = re.split(r'[.!?。！？]\s+', text)
            unique_sentences = []
            seen_sentences = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    # 문장의 핵심 부분만 해시화
                    sentence_key = re.sub(r'\s+', ' ', sentence.lower())[:50]
                    if sentence_key not in seen_sentences:
                        seen_sentences.add(sentence_key)
                        unique_sentences.append(sentence)
            
            return ". ".join(unique_sentences) + "."
            
        except Exception as e:
            logging.error(f"중복 문장 제거 오류: {e}")
            return text
    
    def _calculate_language_distribution(self, content_list: List[Dict]) -> Dict:
        """언어 분포 계산"""
        try:
            language_stats = defaultdict(lambda: {"count": 0, "word_count": 0})
            
            for item in content_list:
                lang = item.get("original_language", "unknown")
                language_stats[lang]["count"] += 1
                language_stats[lang]["word_count"] += item.get("word_count", 0)
            
            total_sources = len(content_list)
            total_words = sum(item.get("word_count", 0) for item in content_list)
            
            distribution = {}
            for lang, stats in language_stats.items():
                distribution[lang] = {
                    "sources": stats["count"],
                    "source_percentage": round(stats["count"] / total_sources * 100, 1) if total_sources > 0 else 0,
                    "word_count": stats["word_count"],
                    "word_percentage": round(stats["word_count"] / total_words * 100, 1) if total_words > 0 else 0,
                    "language_name": self.language_mapping.get(lang, lang)
                }
            
            return distribution
            
        except Exception as e:
            logging.error(f"언어 분포 계산 오류: {e}")
            return {}
    
    def _summarize_jewelry_terms(self, content_list: List[Dict]) -> Dict:
        """주얼리 용어 요약"""
        try:
            all_terms = []
            for item in content_list:
                all_terms.extend(item.get("jewelry_terms", []))
            
            term_frequency = Counter(all_terms)
            
            return {
                "total_unique_terms": len(term_frequency),
                "most_frequent_terms": term_frequency.most_common(10),
                "all_terms": list(term_frequency.keys())
            }
            
        except Exception as e:
            logging.error(f"주얼리 용어 요약 오류: {e}")
            return {}
    
    async def _generate_korean_summary(self, 
                                     integrated_content: Dict,
                                     summary_style: str,
                                     session_type: str) -> Dict:
        """한국어 요약 생성"""
        try:
            print(f"📝 {summary_style} 스타일 한국어 요약 생성 중...")
            
            content_text = integrated_content.get("integrated_text", "")
            if not content_text:
                return {"error": "통합 콘텐츠가 비어있습니다"}
            
            # 스타일별 프롬프트 구성
            style_config = self.summary_styles.get(summary_style, self.summary_styles["comprehensive"])
            
            # 주얼리 AI 엔진으로 요약 생성
            summary_result = await self.jewelry_ai.analyze_jewelry_content(
                content_text,
                analysis_type="summary",
                language="korean",
                style=summary_style
            )
            
            # 구조화된 요약 생성
            structured_summary = self._create_structured_summary(
                content_text, 
                summary_result,
                style_config,
                session_type,
                integrated_content
            )
            
            return structured_summary
            
        except Exception as e:
            logging.error(f"한국어 요약 생성 오류: {e}")
            return {"error": str(e)}
    
    def _create_structured_summary(self, 
                                 content_text: str,
                                 ai_summary: Dict,
                                 style_config: Dict,
                                 session_type: str,
                                 integrated_content: Dict) -> Dict:
        """구조화된 요약 생성"""
        try:
            # 핵심 내용 추출
            key_points = self._extract_key_points(content_text)
            
            # 주얼리 관련 인사이트
            jewelry_insights = ai_summary.get("jewelry_insights", {})
            
            # 요약 섹션별 내용 생성
            sections = {}
            
            for section in style_config["sections"]:
                if section == "핵심 내용":
                    sections[section] = self._generate_key_content_section(key_points, jewelry_insights)
                elif section == "주요 논점":
                    sections[section] = self._generate_main_points_section(content_text, ai_summary)
                elif section == "결론 및 시사점":
                    sections[section] = self._generate_conclusions_section(ai_summary, jewelry_insights)
                elif section == "실행 방안":
                    sections[section] = self._generate_action_items_section(ai_summary, session_type)
                elif section == "핵심 메시지":
                    sections[section] = self._generate_key_message_section(key_points)
                elif section == "비즈니스 임팩트":
                    sections[section] = self._generate_business_impact_section(jewelry_insights)
                elif section == "의사결정 포인트":
                    sections[section] = self._generate_decision_points_section(ai_summary)
                elif section == "다음 단계":
                    sections[section] = self._generate_next_steps_section(ai_summary, session_type)
                elif section == "기술적 내용":
                    sections[section] = self._generate_technical_content_section(content_text)
                elif section == "전문 용어 해설":
                    sections[section] = self._generate_terminology_section(integrated_content)
                elif section == "시장 동향":
                    sections[section] = self._generate_market_trends_section(jewelry_insights)
                elif section == "비즈니스 기회":
                    sections[section] = self._generate_business_opportunities_section(jewelry_insights)
            
            # 전체 요약 텍스트 생성
            summary_text = f"# {style_config['title']}\n\n"
            for section_title, section_content in sections.items():
                if section_content:
                    summary_text += f"## {section_title}\n{section_content}\n\n"
            
            return {
                "title": style_config["title"],
                "style": summary_style,
                "tone": style_config["tone"],
                "sections": sections,
                "full_summary": summary_text.strip(),
                "word_count": len(summary_text.split()),
                "key_metrics": {
                    "original_word_count": len(content_text.split()),
                    "compression_ratio": round(len(summary_text.split()) / len(content_text.split()), 3) if content_text else 0,
                    "jewelry_terms_count": len(integrated_content.get("jewelry_terms_summary", {}).get("all_terms", [])),
                    "source_languages": len(integrated_content.get("language_distribution", {}))
                }
            }
            
        except Exception as e:
            logging.error(f"구조화된 요약 생성 오류: {e}")
            return {"error": str(e)}
    
    def _extract_key_points(self, content: str) -> List[str]:
        """핵심 포인트 추출"""
        try:
            # 문장을 분리하고 중요도 평가
            sentences = re.split(r'[.!?。！？]\s+', content)
            
            key_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    # 주얼리 용어나 중요 키워드가 포함된 문장 우선 선택
                    importance_score = 0
                    
                    # 주얼리 용어 가점
                    for term in self.jewelry_term_mapping.values():
                        if term in sentence:
                            importance_score += 2
                    
                    # 중요 키워드 가점
                    important_keywords = ["중요", "핵심", "주요", "결정", "전략", "계획", "목표", "결과", "결론"]
                    for keyword in important_keywords:
                        if keyword in sentence:
                            importance_score += 1
                    
                    if importance_score > 0:
                        key_sentences.append((sentence, importance_score))
            
            # 중요도 순으로 정렬하여 상위 5개 선택
            key_sentences.sort(key=lambda x: x[1], reverse=True)
            return [sentence for sentence, score in key_sentences[:5]]
            
        except Exception as e:
            logging.error(f"핵심 포인트 추출 오류: {e}")
            return []
    
    def _generate_key_content_section(self, key_points: List[str], jewelry_insights: Dict) -> str:
        """핵심 내용 섹션 생성"""
        try:
            content = []
            
            if key_points:
                content.append("**주요 내용:**")
                for i, point in enumerate(key_points, 1):
                    content.append(f"{i}. {point}")
            
            if jewelry_insights.get("main_topics"):
                content.append("\n**핵심 주제:**")
                for topic in jewelry_insights["main_topics"][:3]:
                    content.append(f"• {topic}")
            
            return "\n".join(content) if content else "핵심 내용을 추출할 수 없습니다."
            
        except Exception as e:
            logging.error(f"핵심 내용 섹션 생성 오류: {e}")
            return "섹션 생성 중 오류가 발생했습니다."
    
    def _generate_main_points_section(self, content: str, ai_summary: Dict) -> str:
        """주요 논점 섹션 생성"""
        try:
            points = []
            
            # AI 요약에서 주요 논점 추출
            if ai_summary.get("key_insights"):
                for insight in ai_summary["key_insights"][:4]:
                    points.append(f"• {insight}")
            
            if not points:
                # 직접 추출
                discussion_keywords = ["논의", "토론", "의견", "제안", "문제", "과제", "방안", "방법"]
                sentences = re.split(r'[.!?。！？]\s+', content)
                
                for sentence in sentences:
                    if any(keyword in sentence for keyword in discussion_keywords) and len(sentence) > 30:
                        points.append(f"• {sentence}")
                        if len(points) >= 3:
                            break
            
            return "\n".join(points) if points else "주요 논점을 식별할 수 없습니다."
            
        except Exception as e:
            logging.error(f"주요 논점 섹션 생성 오류: {e}")
            return "섹션 생성 중 오류가 발생했습니다."
    
    def _generate_conclusions_section(self, ai_summary: Dict, jewelry_insights: Dict) -> str:
        """결론 및 시사점 섹션 생성"""
        try:
            conclusions = []
            
            if jewelry_insights.get("business_implications"):
                conclusions.extend(jewelry_insights["business_implications"])
            
            if ai_summary.get("conclusions"):
                conclusions.extend(ai_summary["conclusions"])
            
            if not conclusions:
                conclusions = ["향후 주얼리 업계 동향에 대한 지속적인 모니터링이 필요합니다."]
            
            content = []
            for i, conclusion in enumerate(conclusions[:4], 1):
                content.append(f"{i}. {conclusion}")
            
            return "\n".join(content)
            
        except Exception as e:
            logging.error(f"결론 섹션 생성 오류: {e}")
            return "결론을 도출할 수 없습니다."
    
    def _generate_action_items_section(self, ai_summary: Dict, session_type: str) -> str:
        """실행 방안 섹션 생성"""
        try:
            actions = []
            
            if ai_summary.get("action_items"):
                actions.extend(ai_summary["action_items"])
            
            # 세션 타입별 기본 실행 방안
            if session_type == "seminar":
                actions.append("세미나 내용을 바탕으로 한 후속 교육 계획 수립")
            elif session_type == "meeting":
                actions.append("회의 결정사항에 대한 구체적 실행 계획 수립")
            elif session_type == "conference":
                actions.append("컨퍼런스 주요 내용을 조직 내 공유")
            
            if not actions:
                actions = ["주요 내용에 대한 세부 실행 계획 수립이 필요합니다."]
            
            content = []
            for i, action in enumerate(actions[:4], 1):
                content.append(f"{i}. {action}")
            
            return "\n".join(content)
            
        except Exception as e:
            logging.error(f"실행 방안 섹션 생성 오류: {e}")
            return "실행 방안을 제시할 수 없습니다."
    
    def _generate_business_impact_section(self, jewelry_insights: Dict) -> str:
        """비즈니스 임팩트 섹션 생성"""
        try:
            impacts = []
            
            if jewelry_insights.get("business_opportunities"):
                impacts.extend(jewelry_insights["business_opportunities"])
            
            if jewelry_insights.get("market_insights"):
                impacts.extend(jewelry_insights["market_insights"])
            
            if not impacts:
                impacts = ["주얼리 업계 비즈니스에 미치는 영향을 면밀히 분석할 필요가 있습니다."]
            
            content = []
            for impact in impacts[:3]:
                content.append(f"• {impact}")
            
            return "\n".join(content)
            
        except Exception as e:
            logging.error(f"비즈니스 임팩트 섹션 생성 오류: {e}")
            return "비즈니스 임팩트를 평가할 수 없습니다."
    
    def _calculate_translation_quality(self, translated_content: Dict) -> Dict:
        """번역 품질 계산"""
        try:
            total_sources = 0
            translation_scores = []
            
            for lang, sources in translated_content.items():
                for source in sources:
                    total_sources += 1
                    if "translated_content" in source:
                        # 간단한 번역 품질 추정
                        original_length = len(source.get("original_content", ""))
                        translated_length = len(source.get("translated_content", ""))
                        
                        # 길이 기반 품질 점수 (너무 짧거나 길면 품질 의심)
                        if original_length > 0:
                            length_ratio = translated_length / original_length
                            if 0.5 <= length_ratio <= 2.0:
                                quality_score = 0.8
                            else:
                                quality_score = 0.6
                        else:
                            quality_score = 0.5
                        
                        translation_scores.append(quality_score)
            
            avg_quality = sum(translation_scores) / len(translation_scores) if translation_scores else 0.5
            
            return {
                "average_quality": round(avg_quality, 3),
                "total_translations": len(translation_scores),
                "quality_level": "높음" if avg_quality >= 0.8 else "보통" if avg_quality >= 0.6 else "낮음"
            }
            
        except Exception as e:
            logging.error(f"번역 품질 계산 오류: {e}")
            return {"average_quality": 0.5, "quality_level": "알 수 없음"}
    
    def _get_mapped_terms(self, normalized_content: Dict) -> List[str]:
        """매핑된 주얼리 용어 수집"""
        try:
            all_terms = []
            for lang, sources in normalized_content.items():
                for source in sources:
                    all_terms.extend(source.get("jewelry_terms", []))
            
            return list(set(all_terms))
            
        except Exception as e:
            logging.error(f"매핑된 용어 수집 오류: {e}")
            return []
    
    async def _assess_summary_quality(self, korean_summary: Dict, integrated_content: Dict) -> Dict:
        """요약 품질 평가"""
        try:
            original_word_count = integrated_content.get("total_word_count", 0)
            summary_word_count = korean_summary.get("word_count", 0)
            
            # 압축률 평가
            compression_ratio = korean_summary.get("key_metrics", {}).get("compression_ratio", 0)
            
            # 내용 완성도 평가
            sections_completed = len([s for s in korean_summary.get("sections", {}).values() if s and "오류" not in s])
            total_sections = len(korean_summary.get("sections", {}))
            completeness = sections_completed / total_sections if total_sections > 0 else 0
            
            # 주얼리 용어 보존도
            original_terms = len(integrated_content.get("jewelry_terms_summary", {}).get("all_terms", []))
            summary_terms = korean_summary.get("key_metrics", {}).get("jewelry_terms_count", 0)
            term_preservation = min(1.0, summary_terms / original_terms) if original_terms > 0 else 1.0
            
            # 종합 품질 점수
            quality_score = (completeness * 0.4 + term_preservation * 0.3 + min(1.0, compression_ratio * 5) * 0.3) * 100
            
            return {
                "overall_score": round(quality_score, 1),
                "quality_level": "우수" if quality_score >= 80 else "양호" if quality_score >= 60 else "보통",
                "metrics": {
                    "completeness": round(completeness, 3),
                    "term_preservation": round(term_preservation, 3),
                    "compression_efficiency": round(compression_ratio, 3)
                },
                "recommendations": self._generate_quality_recommendations(quality_score, completeness, term_preservation)
            }
            
        except Exception as e:
            logging.error(f"요약 품질 평가 오류: {e}")
            return {"overall_score": 50, "quality_level": "평가 불가"}
    
    def _generate_quality_recommendations(self, quality_score: float, completeness: float, term_preservation: float) -> List[str]:
        """품질 개선 권장사항 생성"""
        try:
            recommendations = []
            
            if quality_score < 70:
                recommendations.append("전체적인 요약 품질 개선이 필요합니다.")
            
            if completeness < 0.8:
                recommendations.append("요약 섹션의 완성도를 높여주세요.")
            
            if term_preservation < 0.7:
                recommendations.append("주얼리 전문 용어의 보존을 개선해주세요.")
            
            if not recommendations:
                recommendations.append("우수한 품질의 요약이 생성되었습니다.")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"품질 권장사항 생성 오류: {e}")
            return ["품질 평가를 완료할 수 없습니다."]

# 전역 인스턴스
_korean_final_summarizer_instance = None

def get_korean_final_summarizer() -> KoreanFinalSummarizer:
    """전역 한국어 최종 요약기 인스턴스 반환"""
    global _korean_final_summarizer_instance
    if _korean_final_summarizer_instance is None:
        _korean_final_summarizer_instance = KoreanFinalSummarizer()
    return _korean_final_summarizer_instance

# 편의 함수들
async def generate_korean_summary(session_data: Dict, **kwargs) -> Dict:
    """한국어 요약 생성 편의 함수"""
    summarizer = get_korean_final_summarizer()
    return await summarizer.process_multilingual_session(session_data, **kwargs)

def check_korean_summarizer_support() -> Dict:
    """한국어 요약기 지원 상태 확인"""
    return {
        "libraries": {
            "langdetect": LANGDETECT_AVAILABLE,
            "googletrans": GOOGLETRANS_AVAILABLE,
            "jewelry_ai_engine": True
        },
        "features": {
            "language_detection": True,
            "multilingual_translation": GOOGLETRANS_AVAILABLE,
            "jewelry_term_mapping": True,
            "content_integration": True,
            "structured_summary": True
        },
        "supported_languages": list(KoreanFinalSummarizer().language_mapping.keys()),
        "summary_styles": list(KoreanFinalSummarizer().summary_styles.keys()),
        "jewelry_terms_count": len(KoreanFinalSummarizer().jewelry_term_mapping)
    }

if __name__ == "__main__":
    # 테스트 코드
    async def test_korean_summarizer():
        print("한국어 최종 통합 요약 엔진 테스트")
        support_info = check_korean_summarizer_support()
        print(f"지원 상태: {support_info}")
    
    import asyncio
    asyncio.run(test_korean_summarizer())
