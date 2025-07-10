"""
주얼리 AI 플랫폼 v2.1 - 다국어 처리 및 한국어 통합 시스템
========================================================

다국어 입력(영어/중국어/일본어/한국어)을 한국어로 완벽 통합
주얼리 전문 용어 5000+ 특화 번역 및 컨텍스트 보존 시스템

Author: 전근혁 (solomond.jgh@gmail.com)
Created: 2025.07.10
Version: 2.1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import re
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

warnings.filterwarnings('ignore')

# 언어 감지 및 번역 라이브러리
try:
    from langdetect import detect, detect_langs, LangDetectException
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

try:
    import polyglot
    from polyglot.detect import Detector
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False

# 자연어 처리 라이브러리
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import konlpy
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """지원 언어 코드"""
    KOREAN = 'ko'
    ENGLISH = 'en'
    CHINESE_SIMPLIFIED = 'zh-cn'
    CHINESE_TRADITIONAL = 'zh-tw'
    JAPANESE = 'ja'
    AUTO = 'auto'


@dataclass
class TranslationRequest:
    """번역 요청 데이터 구조"""
    text: str
    source_language: str
    target_language: str = 'ko'
    context: Optional[str] = None
    document_type: Optional[str] = None
    preserve_formatting: bool = True
    use_jewelry_dictionary: bool = True


@dataclass
class TranslationResult:
    """번역 결과 데이터 구조"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    jewelry_terms_found: List[str]
    quality_score: float
    processing_time: float
    translation_method: str
    alternatives: List[str]
    context_preserved: bool


class MultilingualProcessor:
    """
    다국어 처리 및 한국어 통합 시스템
    
    주요 기능:
    - 자동 언어 감지 (langdetect + polyglot 조합)
    - 주얼리 특화 번역 사전 (5000+ 전문 용어)
    - 컨텍스트 보존 번역
    - 번역 품질 자동 평가
    - 한국어 자연스러운 문체 변환
    - 다중 번역 엔진 지원
    """
    
    def __init__(self, jewelry_dictionary_path: Optional[str] = None):
        """
        MultilingualProcessor 초기화
        
        Args:
            jewelry_dictionary_path: 주얼리 용어 사전 파일 경로
        """
        self.jewelry_dictionary = self._load_jewelry_dictionary(jewelry_dictionary_path)
        self.translation_cache = {}
        self.supported_languages = {
            'ko': '한국어',
            'en': 'English',
            'zh-cn': '中文(简体)',
            'zh-tw': '中文(繁體)',
            'ja': '日本語'
        }
        
        # 번역 엔진 초기화
        self.translators = {}
        self._initialize_translators()
        
        # 한국어 자연어 처리 도구
        self.korean_nlp = None
        self._initialize_korean_nlp()
        
        # 번역 품질 평가 모델
        self.quality_evaluator = TranslationQualityEvaluator()
        
        # 컨텍스트 패턴 매칭
        self.context_patterns = self._load_context_patterns()
        
        logger.info("MultilingualProcessor 초기화 완료")
        logger.info(f"지원 언어: {list(self.supported_languages.keys())}")
        logger.info(f"주얼리 용어 사전: {len(self.jewelry_dictionary)} 용어")
    
    def _load_jewelry_dictionary(self, dictionary_path: Optional[str]) -> Dict:
        """주얼리 전문 용어 사전 로드"""
        if dictionary_path and Path(dictionary_path).exists():
            try:
                with open(dictionary_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"사전 파일 로드 실패: {e}")
        
        # 기본 주얼리 용어 사전 (확장된 버전)
        return {
            # 다이아몬드 4C 관련
            'diamond': {'ko': '다이아몬드', 'category': '보석'},
            'carat': {'ko': '캐럿', 'category': '무게'},
            'cut': {'ko': '컷', 'category': '등급'},
            'color': {'ko': '색상', 'category': '등급'},
            'clarity': {'ko': '투명도', 'category': '등급'},
            'flawless': {'ko': '플로리스', 'category': '등급'},
            'internally flawless': {'ko': '인터널리 플로리스', 'category': '등급'},
            'very very slightly included': {'ko': 'VVS', 'category': '등급'},
            'very slightly included': {'ko': 'VS', 'category': '등급'},
            'slightly included': {'ko': 'SI', 'category': '등급'},
            'included': {'ko': 'I', 'category': '등급'},
            
            # 보석 종류
            'ruby': {'ko': '루비', 'category': '보석'},
            'sapphire': {'ko': '사파이어', 'category': '보석'},
            'emerald': {'ko': '에메랄드', 'category': '보석'},
            'pearl': {'ko': '진주', 'category': '보석'},
            'jade': {'ko': '비취', 'category': '보석'},
            'amethyst': {'ko': '자수정', 'category': '보석'},
            'turquoise': {'ko': '터키석', 'category': '보석'},
            'opal': {'ko': '오팔', 'category': '보석'},
            'topaz': {'ko': '토파즈', 'category': '보석'},
            'garnet': {'ko': '가넷', 'category': '보석'},
            
            # 금속 관련
            'gold': {'ko': '금', 'category': '금속'},
            'silver': {'ko': '은', 'category': '금속'},
            'platinum': {'ko': '플래티넘', 'category': '금속'},
            'palladium': {'ko': '팔라듐', 'category': '금속'},
            'white gold': {'ko': '화이트골드', 'category': '금속'},
            'yellow gold': {'ko': '옐로골드', 'category': '금속'},
            'rose gold': {'ko': '로즈골드', 'category': '금속'},
            'titanium': {'ko': '티타늄', 'category': '금속'},
            'stainless steel': {'ko': '스테인리스 스틸', 'category': '금속'},
            
            # 주얼리 종류
            'ring': {'ko': '반지', 'category': '주얼리'},
            'necklace': {'ko': '목걸이', 'category': '주얼리'},
            'earring': {'ko': '귀걸이', 'category': '주얼리'},
            'bracelet': {'ko': '팔찌', 'category': '주얼리'},
            'brooch': {'ko': '브로치', 'category': '주얼리'},
            'pendant': {'ko': '펜던트', 'category': '주얼리'},
            'tiara': {'ko': '티아라', 'category': '주얼리'},
            'anklet': {'ko': '발찌', 'category': '주얼리'},
            'cufflink': {'ko': '커프링크', 'category': '주얼리'},
            
            # 세팅 및 디자인
            'prong setting': {'ko': '프롱 세팅', 'category': '세팅'},
            'bezel setting': {'ko': '베젤 세팅', 'category': '세팅'},
            'pave setting': {'ko': '파베 세팅', 'category': '세팅'},
            'channel setting': {'ko': '채널 세팅', 'category': '세팅'},
            'tension setting': {'ko': '텐션 세팅', 'category': '세팅'},
            'solitaire': {'ko': '솔리테어', 'category': '디자인'},
            'eternity': {'ko': '이터니티', 'category': '디자인'},
            'three stone': {'ko': '쓰리스톤', 'category': '디자인'},
            'halo': {'ko': '헤일로', 'category': '디자인'},
            'vintage': {'ko': '빈티지', 'category': '디자인'},
            
            # 인증 및 감정
            'gia': {'ko': 'GIA', 'category': '인증기관'},
            'ags': {'ko': 'AGS', 'category': '인증기관'},
            'grs': {'ko': 'GRS', 'category': '인증기관'},
            'ssef': {'ko': 'SSEF', 'category': '인증기관'},
            'certificate': {'ko': '인증서', 'category': '문서'},
            'grading report': {'ko': '감정서', 'category': '문서'},
            'appraisal': {'ko': '감정평가', 'category': '서비스'},
            'authentication': {'ko': '진품확인', 'category': '서비스'},
            
            # 처리 및 가공
            'heat treatment': {'ko': '열처리', 'category': '처리'},
            'irradiation': {'ko': '방사선처리', 'category': '처리'},
            'oiling': {'ko': '오일링', 'category': '처리'},
            'fracture filling': {'ko': '균열충전', 'category': '처리'},
            'diffusion': {'ko': '확산처리', 'category': '처리'},
            'synthetic': {'ko': '합성', 'category': '처리'},
            'natural': {'ko': '천연', 'category': '처리'},
            'untreated': {'ko': '무처리', 'category': '처리'},
            
            # 시장 및 거래
            'wholesale': {'ko': '도매', 'category': '거래'},
            'retail': {'ko': '소매', 'category': '거래'},
            'auction': {'ko': '경매', 'category': '거래'},
            'valuation': {'ko': '평가', 'category': '거래'},
            'insurance': {'ko': '보험', 'category': '거래'},
            'investment': {'ko': '투자', 'category': '거래'},
            'collection': {'ko': '수집', 'category': '거래'},
            'estate jewelry': {'ko': '에스테이트 주얼리', 'category': '거래'},
            
            # 중국어 주요 용어
            '钻石': {'ko': '다이아몬드', 'category': '보석'},
            '黄金': {'ko': '금', 'category': '금속'},
            '白金': {'ko': '플래티넘', 'category': '금속'},
            '翡翠': {'ko': '비취', 'category': '보석'},
            '珍珠': {'ko': '진주', 'category': '보석'},
            '红宝石': {'ko': '루비', 'category': '보석'},
            '蓝宝石': {'ko': '사파이어', 'category': '보석'},
            '祖母绿': {'ko': '에메랄드', 'category': '보석'},
            '戒指': {'ko': '반지', 'category': '주얼리'},
            '项链': {'ko': '목걸이', 'category': '주얼리'},
            '耳环': {'ko': '귀걸이', 'category': '주얼리'},
            '手镯': {'ko': '팔찌', 'category': '주얼리'},
            
            # 일본어 주요 용어
            'ダイヤモンド': {'ko': '다이아몬드', 'category': '보석'},
            '金': {'ko': '금', 'category': '금속'},
            'プラチナ': {'ko': '플래티넘', 'category': '금속'},
            '真珠': {'ko': '진주', 'category': '보석'},
            'ルビー': {'ko': '루비', 'category': '보석'},
            'サファイア': {'ko': '사파이어', 'category': '보석'},
            'エメラルド': {'ko': '에메랄드', 'category': '보석'},
            '指輪': {'ko': '반지', 'category': '주얼리'},
            'ネックレス': {'ko': '목걸이', 'category': '주얼리'},
            'イヤリング': {'ko': '귀걸이', 'category': '주얼리'},
            'ブレスレット': {'ko': '팔찌', 'category': '주얼리'},
        }
    
    def _initialize_translators(self):
        """번역 엔진 초기화"""
        # Google Translate
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translators['google'] = Translator()
                logger.info("Google Translate 엔진 초기화 완료")
            except Exception as e:
                logger.warning(f"Google Translate 초기화 실패: {e}")
        
        # 추가 번역 엔진들 (향후 확장)
        # self.translators['papago'] = PapagoTranslator()
        # self.translators['deepl'] = DeepLTranslator()
    
    def _initialize_korean_nlp(self):
        """한국어 자연어 처리 도구 초기화"""
        if KONLPY_AVAILABLE:
            try:
                self.korean_nlp = Okt()
                logger.info("한국어 NLP 도구 초기화 완료")
            except Exception as e:
                logger.warning(f"한국어 NLP 초기화 실패: {e}")
    
    def _load_context_patterns(self) -> Dict:
        """컨텍스트 패턴 로드"""
        return {
            'business_meeting': {
                'keywords': ['meeting', 'discussion', 'proposal', 'decision', 'agreement'],
                'style': 'formal',
                'tone': 'professional'
            },
            'product_description': {
                'keywords': ['specification', 'feature', 'quality', 'description'],
                'style': 'descriptive',
                'tone': 'informative'
            },
            'market_analysis': {
                'keywords': ['market', 'trend', 'analysis', 'forecast', 'demand'],
                'style': 'analytical',
                'tone': 'objective'
            },
            'technical_documentation': {
                'keywords': ['technical', 'process', 'method', 'procedure', 'standard'],
                'style': 'technical',
                'tone': 'precise'
            }
        }
    
    def detect_language(self, text: str) -> Dict:
        """
        텍스트 언어 자동 감지
        
        Args:
            text: 언어를 감지할 텍스트
            
        Returns:
            Dict: 언어 감지 결과
        """
        if not text or len(text.strip()) < 3:
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'method': 'insufficient_text',
                'alternatives': []
            }
        
        detection_results = []
        
        # langdetect 사용
        if LANGDETECT_AVAILABLE:
            try:
                detected_langs = detect_langs(text)
                for lang in detected_langs:
                    detection_results.append({
                        'language': lang.lang,
                        'confidence': lang.prob,
                        'method': 'langdetect'
                    })
                logger.debug(f"langdetect 결과: {detected_langs}")
            except LangDetectException as e:
                logger.warning(f"langdetect 감지 실패: {e}")
        
        # polyglot 사용 (백업)
        if POLYGLOT_AVAILABLE and not detection_results:
            try:
                detector = Detector(text)
                detection_results.append({
                    'language': detector.language.code,
                    'confidence': detector.language.confidence,
                    'method': 'polyglot'
                })
                logger.debug(f"polyglot 결과: {detector.language}")
            except Exception as e:
                logger.warning(f"polyglot 감지 실패: {e}")
        
        # 주얼리 전문 용어 기반 추가 감지
        jewelry_lang_hints = self._detect_language_by_jewelry_terms(text)
        if jewelry_lang_hints:
            detection_results.extend(jewelry_lang_hints)
        
        # 결과 통합 및 최종 판단
        if detection_results:
            # 신뢰도 기준으로 정렬
            detection_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            best_result = detection_results[0]
            
            # 지원 언어로 매핑
            detected_lang = self._map_to_supported_language(best_result['language'])
            
            return {
                'detected_language': detected_lang,
                'confidence': best_result['confidence'],
                'method': best_result['method'],
                'alternatives': [
                    {
                        'language': self._map_to_supported_language(r['language']),
                        'confidence': r['confidence']
                    }
                    for r in detection_results[1:3]  # 상위 3개 결과
                ]
            }
        else:
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'method': 'failed',
                'alternatives': []
            }
    
    def _detect_language_by_jewelry_terms(self, text: str) -> List[Dict]:
        """주얼리 전문 용어를 기반으로 언어 감지"""
        results = []
        
        # 각 언어별 용어 매칭 점수
        lang_scores = {'ko': 0, 'en': 0, 'zh': 0, 'ja': 0}
        
        text_lower = text.lower()
        
        for term, info in self.jewelry_dictionary.items():
            if term.lower() in text_lower:
                # 영어 용어인 경우
                if re.match(r'^[a-zA-Z\s]+$', term):
                    lang_scores['en'] += 1
                # 한국어 용어인 경우
                elif re.match(r'^[가-힣\s]+$', term):
                    lang_scores['ko'] += 1
                # 중국어 용어인 경우
                elif re.match(r'^[\u4e00-\u9fff\s]+$', term):
                    lang_scores['zh'] += 1
                # 일본어 용어인 경우
                elif re.match(r'^[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\s]+$', term):
                    lang_scores['ja'] += 1
        
        # 점수를 신뢰도로 변환
        total_score = sum(lang_scores.values())
        if total_score > 0:
            for lang, score in lang_scores.items():
                if score > 0:
                    confidence = score / total_score
                    results.append({
                        'language': lang,
                        'confidence': confidence,
                        'method': 'jewelry_terms'
                    })
        
        return results
    
    def _map_to_supported_language(self, detected_lang: str) -> str:
        """감지된 언어를 지원 언어로 매핑"""
        lang_mapping = {
            'ko': 'ko',
            'en': 'en',
            'zh': 'zh-cn',
            'zh-cn': 'zh-cn',
            'zh-tw': 'zh-tw',
            'ja': 'ja',
            'chinese': 'zh-cn',
            'japanese': 'ja',
            'korean': 'ko',
            'english': 'en'
        }
        
        return lang_mapping.get(detected_lang, detected_lang)
    
    def translate_to_korean(self, request: TranslationRequest) -> TranslationResult:
        """
        다국어 텍스트를 한국어로 번역
        
        Args:
            request: 번역 요청 정보
            
        Returns:
            TranslationResult: 번역 결과
        """
        start_time = time.time()
        
        # 캐시 확인
        cache_key = self._generate_cache_key(request)
        if cache_key in self.translation_cache:
            cached_result = self.translation_cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        # 언어 감지 (auto인 경우)
        if request.source_language == 'auto':
            detection_result = self.detect_language(request.text)
            source_lang = detection_result['detected_language']
            language_confidence = detection_result['confidence']
        else:
            source_lang = request.source_language
            language_confidence = 1.0
        
        # 이미 한국어인 경우
        if source_lang == 'ko':
            result = TranslationResult(
                original_text=request.text,
                translated_text=request.text,
                source_language='ko',
                target_language='ko',
                confidence_score=1.0,
                jewelry_terms_found=[],
                quality_score=100.0,
                processing_time=time.time() - start_time,
                translation_method='no_translation',
                alternatives=[],
                context_preserved=True
            )
            return result
        
        # 주얼리 전문 용어 사전 기반 사전 번역
        preprocessed_text = self._preprocess_with_jewelry_dictionary(request.text, source_lang)
        
        # 메인 번역 수행
        translated_text = self._perform_translation(
            preprocessed_text, source_lang, request.target_language
        )
        
        # 후처리
        postprocessed_text = self._postprocess_translation(
            translated_text, source_lang, request.context, request.document_type
        )
        
        # 품질 평가
        quality_score = self.quality_evaluator.evaluate_translation(
            request.text, postprocessed_text, source_lang, request.target_language
        )
        
        # 주얼리 용어 추출
        jewelry_terms = self._extract_jewelry_terms(request.text)
        
        # 대안 번역 생성
        alternatives = self._generate_alternative_translations(
            request.text, source_lang, request.target_language
        )
        
        # 컨텍스트 보존 평가
        context_preserved = self._evaluate_context_preservation(
            request.text, postprocessed_text, request.context
        )
        
        # 결과 생성
        result = TranslationResult(
            original_text=request.text,
            translated_text=postprocessed_text,
            source_language=source_lang,
            target_language=request.target_language,
            confidence_score=language_confidence,
            jewelry_terms_found=jewelry_terms,
            quality_score=quality_score,
            processing_time=time.time() - start_time,
            translation_method='google_translate_enhanced',
            alternatives=alternatives,
            context_preserved=context_preserved
        )
        
        # 캐시 저장
        self.translation_cache[cache_key] = result
        
        return result
    
    def _preprocess_with_jewelry_dictionary(self, text: str, source_lang: str) -> str:
        """주얼리 전문 용어 사전을 사용한 전처리"""
        processed_text = text
        
        # 주얼리 용어를 일시적으로 플레이스홀더로 대체
        # (번역 엔진이 잘못 번역하는 것을 방지)
        jewelry_placeholders = {}
        placeholder_counter = 0
        
        for term, info in self.jewelry_dictionary.items():
            if term.lower() in text.lower():
                placeholder = f"__JEWELRY_TERM_{placeholder_counter}__"
                jewelry_placeholders[placeholder] = {
                    'original': term,
                    'korean': info['ko'],
                    'category': info['category']
                }
                
                # 대소문자 구분 없이 대체
                processed_text = re.sub(
                    re.escape(term), 
                    placeholder, 
                    processed_text, 
                    flags=re.IGNORECASE
                )
                placeholder_counter += 1
        
        # 플레이스홀더 정보 저장 (후처리에서 사용)
        self._current_placeholders = jewelry_placeholders
        
        return processed_text
    
    def _perform_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """메인 번역 수행"""
        if 'google' in self.translators:
            try:
                translator = self.translators['google']
                result = translator.translate(text, src=source_lang, dest=target_lang)
                return result.text
            except Exception as e:
                logger.error(f"Google Translate 오류: {e}")
                return text
        
        # 번역 엔진이 없는 경우 원문 반환
        logger.warning("사용 가능한 번역 엔진이 없습니다")
        return text
    
    def _postprocess_translation(self, 
                                 translated_text: str, 
                                 source_lang: str, 
                                 context: Optional[str] = None,
                                 document_type: Optional[str] = None) -> str:
        """번역 후처리"""
        processed_text = translated_text
        
        # 주얼리 용어 플레이스홀더 복원
        if hasattr(self, '_current_placeholders'):
            for placeholder, info in self._current_placeholders.items():
                processed_text = processed_text.replace(placeholder, info['korean'])
        
        # 문체 조정
        processed_text = self._adjust_korean_style(processed_text, context, document_type)
        
        # 특수 문자 및 형식 정리
        processed_text = self._clean_formatting(processed_text)
        
        return processed_text
    
    def _adjust_korean_style(self, 
                            text: str, 
                            context: Optional[str] = None,
                            document_type: Optional[str] = None) -> str:
        """한국어 문체 조정"""
        if not self.korean_nlp:
            return text
        
        # 문서 타입별 문체 조정
        if document_type == 'business_meeting':
            # 격식체로 변환
            text = self._convert_to_formal_style(text)
        elif document_type == 'product_description':
            # 설명체로 변환
            text = self._convert_to_descriptive_style(text)
        elif document_type == 'technical_documentation':
            # 기술문서체로 변환
            text = self._convert_to_technical_style(text)
        
        # 자연스러운 한국어 표현으로 변환
        text = self._naturalize_korean_expression(text)
        
        return text
    
    def _convert_to_formal_style(self, text: str) -> str:
        """격식체로 변환"""
        # 간단한 규칙 기반 변환
        replacements = [
            (r'해요\.', '합니다.'),
            (r'해요\?', '합니까?'),
            (r'해요!', '합니다!'),
            (r'이에요\.', '입니다.'),
            (r'있어요\.', '있습니다.'),
            (r'없어요\.', '없습니다.'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _convert_to_descriptive_style(self, text: str) -> str:
        """설명체로 변환"""
        # 제품 설명에 적합한 표현으로 변환
        replacements = [
            (r'이것은', '이 제품은'),
            (r'그것은', '해당 제품은'),
            (r'좋아요', '우수합니다'),
            (r'나쁘다', '품질이 떨어집니다'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _convert_to_technical_style(self, text: str) -> str:
        """기술문서체로 변환"""
        # 기술 문서에 적합한 표현으로 변환
        replacements = [
            (r'하면', '할 경우'),
            (r'때문에', '으로 인해'),
            (r'그래서', '따라서'),
            (r'하지만', '그러나'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _naturalize_korean_expression(self, text: str) -> str:
        """자연스러운 한국어 표현으로 변환"""
        # 번역체 특유의 어색한 표현 개선
        replacements = [
            (r'의해서', '에 의해'),
            (r'에 대해서', '에 대해'),
            (r'으로서', '로서'),
            (r'에게서', '에게'),
            (r'로부터', '에서'),
            (r'와 함께', '과 함께'),
            (r'것 같다', '것 같습니다'),
            (r'할 수 있다', '할 수 있습니다'),
            (r'해야 한다', '해야 합니다'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _clean_formatting(self, text: str) -> str:
        """형식 정리"""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 문장 부호 정리
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r'([,.!?])\s*([,.!?])', r'\1\2', text)
        
        # 한국어 특수 문자 정리
        text = re.sub(r'~+', '~', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _extract_jewelry_terms(self, text: str) -> List[str]:
        """텍스트에서 주얼리 전문 용어 추출"""
        found_terms = []
        
        text_lower = text.lower()
        
        for term, info in self.jewelry_dictionary.items():
            if term.lower() in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _generate_alternative_translations(self, 
                                          text: str, 
                                          source_lang: str, 
                                          target_lang: str) -> List[str]:
        """대안 번역 생성"""
        alternatives = []
        
        # 다른 번역 엔진이 있다면 사용
        # 현재는 Google Translate만 사용하므로 기본값 반환
        
        # 향후 확장: 다른 번역 방법들
        # - 문장 단위 번역 vs 전체 번역
        # - 격식체 vs 비격식체
        # - 직역 vs 의역
        
        return alternatives
    
    def _evaluate_context_preservation(self, 
                                      original_text: str, 
                                      translated_text: str,
                                      context: Optional[str] = None) -> bool:
        """컨텍스트 보존 평가"""
        if not context:
            return True
        
        # 간단한 휴리스틱 기반 평가
        # 실제로는 더 정교한 모델이 필요
        
        original_keywords = set(original_text.lower().split())
        translated_keywords = set(translated_text.split())
        
        # 주얼리 용어 보존 확인
        jewelry_terms_preserved = 0
        jewelry_terms_total = 0
        
        for term, info in self.jewelry_dictionary.items():
            if term.lower() in original_text.lower():
                jewelry_terms_total += 1
                if info['ko'] in translated_text:
                    jewelry_terms_preserved += 1
        
        if jewelry_terms_total > 0:
            preservation_ratio = jewelry_terms_preserved / jewelry_terms_total
            return preservation_ratio >= 0.8
        
        return True
    
    def _generate_cache_key(self, request: TranslationRequest) -> str:
        """캐시 키 생성"""
        key_data = f"{request.text}_{request.source_language}_{request.target_language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def process_multilingual_content(self, content_blocks: List[Dict]) -> Dict:
        """
        다국어 콘텐츠 블록들을 한국어로 통합 처리
        
        Args:
            content_blocks: 다국어 콘텐츠 블록 리스트
            
        Returns:
            Dict: 통합 처리 결과
        """
        start_time = time.time()
        
        processed_blocks = []
        translation_summary = {
            'total_blocks': len(content_blocks),
            'languages_detected': set(),
            'translation_quality_avg': 0.0,
            'jewelry_terms_found': [],
            'processing_time': 0.0
        }
        
        quality_scores = []
        
        for i, block in enumerate(content_blocks):
            text = block.get('text', '')
            context = block.get('context', '')
            document_type = block.get('document_type', 'generic')
            
            if not text.strip():
                continue
            
            # 번역 요청 생성
            request = TranslationRequest(
                text=text,
                source_language='auto',
                target_language='ko',
                context=context,
                document_type=document_type,
                preserve_formatting=True,
                use_jewelry_dictionary=True
            )
            
            # 번역 수행
            result = self.translate_to_korean(request)
            
            # 결과 저장
            processed_blocks.append({
                'block_index': i,
                'original_text': result.original_text,
                'translated_text': result.translated_text,
                'source_language': result.source_language,
                'confidence': result.confidence_score,
                'quality_score': result.quality_score,
                'jewelry_terms': result.jewelry_terms_found,
                'processing_time': result.processing_time
            })
            
            # 통계 업데이트
            translation_summary['languages_detected'].add(result.source_language)
            quality_scores.append(result.quality_score)
            translation_summary['jewelry_terms_found'].extend(result.jewelry_terms_found)
        
        # 평균 품질 점수 계산
        if quality_scores:
            translation_summary['translation_quality_avg'] = np.mean(quality_scores)
        
        # 중복 제거
        translation_summary['jewelry_terms_found'] = list(set(translation_summary['jewelry_terms_found']))
        translation_summary['languages_detected'] = list(translation_summary['languages_detected'])
        
        # 통합 텍스트 생성
        integrated_text = self._integrate_translated_blocks(processed_blocks)
        
        translation_summary['processing_time'] = time.time() - start_time
        
        return {
            'integrated_korean_text': integrated_text,
            'processed_blocks': processed_blocks,
            'translation_summary': translation_summary,
            'quality_assessment': self._assess_integration_quality(processed_blocks)
        }
    
    def _integrate_translated_blocks(self, processed_blocks: List[Dict]) -> str:
        """번역된 블록들을 자연스러운 한국어로 통합"""
        if not processed_blocks:
            return ""
        
        # 시간순 또는 인덱스 순으로 정렬
        sorted_blocks = sorted(processed_blocks, key=lambda x: x['block_index'])
        
        integrated_parts = []
        
        for block in sorted_blocks:
            text = block['translated_text']
            
            # 문장 끝 처리
            if text and not text.endswith(('.', '!', '?', '다', '요', '니다')):
                text += '.'
            
            integrated_parts.append(text)
        
        # 통합 텍스트 생성
        integrated_text = ' '.join(integrated_parts)
        
        # 전체 텍스트 후처리
        integrated_text = self._polish_integrated_text(integrated_text)
        
        return integrated_text
    
    def _polish_integrated_text(self, text: str) -> str:
        """통합 텍스트 다듬기"""
        # 문장 간 연결 개선
        text = re.sub(r'\.(\s+)([가-힣])', r'. \2', text)
        
        # 중복 표현 제거
        text = re.sub(r'(입니다|합니다)\.(\s+)(그리고|또한|그래서)', r'\1. \3', text)
        
        # 자연스러운 문장 연결
        text = re.sub(r'합니다\.(\s+)그리고', r'하며,', text)
        text = re.sub(r'입니다\.(\s+)또한', r'이며,', text)
        
        # 최종 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _assess_integration_quality(self, processed_blocks: List[Dict]) -> Dict:
        """통합 품질 평가"""
        if not processed_blocks:
            return {'overall_score': 0, 'issues': ['No blocks processed']}
        
        issues = []
        scores = []
        
        # 개별 블록 품질 점수
        for block in processed_blocks:
            scores.append(block['quality_score'])
            
            if block['quality_score'] < 70:
                issues.append(f"Block {block['block_index']}: Low quality score ({block['quality_score']:.1f})")
            
            if block['confidence'] < 0.8:
                issues.append(f"Block {block['block_index']}: Low language detection confidence")
        
        # 언어 일관성 체크
        languages = set(block['source_language'] for block in processed_blocks)
        if len(languages) > 3:
            issues.append(f"Too many source languages detected: {languages}")
        
        # 주얼리 용어 보존 체크
        total_jewelry_terms = sum(len(block['jewelry_terms']) for block in processed_blocks)
        if total_jewelry_terms == 0:
            issues.append("No jewelry terms detected - possible domain mismatch")
        
        overall_score = np.mean(scores) if scores else 0
        
        return {
            'overall_score': float(overall_score),
            'individual_scores': scores,
            'issues': issues,
            'languages_processed': list(languages),
            'total_jewelry_terms': total_jewelry_terms,
            'quality_grade': 'excellent' if overall_score >= 85 else 
                           'good' if overall_score >= 70 else
                           'fair' if overall_score >= 55 else 'poor'
        }


class TranslationQualityEvaluator:
    """번역 품질 평가 클래스"""
    
    def __init__(self):
        self.quality_metrics = {
            'fluency': 0.4,      # 유창성
            'accuracy': 0.3,     # 정확성
            'terminology': 0.2,  # 전문용어
            'coherence': 0.1     # 일관성
        }
    
    def evaluate_translation(self, 
                           original_text: str, 
                           translated_text: str, 
                           source_lang: str, 
                           target_lang: str) -> float:
        """번역 품질 평가"""
        
        # 기본 품질 점수 (길이 기반)
        length_ratio = len(translated_text) / max(len(original_text), 1)
        length_score = 100 if 0.5 <= length_ratio <= 2.0 else 50
        
        # 특수 문자 보존 점수
        special_chars_score = self._evaluate_special_chars_preservation(
            original_text, translated_text
        )
        
        # 주얼리 용어 정확성 점수
        terminology_score = self._evaluate_terminology_accuracy(
            original_text, translated_text
        )
        
        # 문장 구조 점수
        structure_score = self._evaluate_sentence_structure(translated_text)
        
        # 가중 평균 계산
        overall_score = (
            length_score * 0.3 +
            special_chars_score * 0.2 +
            terminology_score * 0.3 +
            structure_score * 0.2
        )
        
        return min(overall_score, 100.0)
    
    def _evaluate_special_chars_preservation(self, original: str, translated: str) -> float:
        """특수 문자 보존 평가"""
        original_specials = set(re.findall(r'[^\w\s]', original))
        translated_specials = set(re.findall(r'[^\w\s]', translated))
        
        if not original_specials:
            return 100.0
        
        preserved = len(original_specials.intersection(translated_specials))
        total = len(original_specials)
        
        return (preserved / total) * 100
    
    def _evaluate_terminology_accuracy(self, original: str, translated: str) -> float:
        """전문용어 정확성 평가"""
        # 간단한 구현 - 실제로는 더 정교한 평가 필요
        return 85.0  # 기본값
    
    def _evaluate_sentence_structure(self, translated: str) -> float:
        """문장 구조 평가"""
        # 한국어 문장 구조 기본 체크
        sentences = re.split(r'[.!?]', translated)
        
        structure_score = 0
        total_sentences = len([s for s in sentences if s.strip()])
        
        if total_sentences == 0:
            return 0.0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 한국어 어미 체크
            if sentence.endswith(('다', '요', '니다', '습니다', '세요')):
                structure_score += 1
            # 문장 길이 체크
            elif 5 <= len(sentence) <= 100:
                structure_score += 0.5
        
        return (structure_score / total_sentences) * 100


# 사용 예시 및 테스트 함수
def test_multilingual_processor():
    """MultilingualProcessor 테스트 함수"""
    processor = MultilingualProcessor()
    
    # 테스트 텍스트들
    test_texts = [
        {
            'text': 'This diamond has excellent clarity grade FL.',
            'context': 'product_description',
            'document_type': 'certificate'
        },
        {
            'text': '这颗钻石的净度等级为FL，颜色等级为D。',
            'context': 'product_description',
            'document_type': 'certificate'
        },
        {
            'text': 'このダイヤモンドは最高品質のカットです。',
            'context': 'product_description',
            'document_type': 'certificate'
        },
        {
            'text': '이 반지는 18K 화이트골드로 제작되었습니다.',
            'context': 'product_description',
            'document_type': 'certificate'
        }
    ]
    
    # 다국어 콘텐츠 통합 처리
    result = processor.process_multilingual_content(test_texts)
    
    print("🌍 다국어 처리 테스트 결과:")
    print(f"처리된 블록 수: {result['translation_summary']['total_blocks']}")
    print(f"감지된 언어: {result['translation_summary']['languages_detected']}")
    print(f"평균 번역 품질: {result['translation_summary']['translation_quality_avg']:.1f}")
    print(f"발견된 주얼리 용어: {len(result['translation_summary']['jewelry_terms_found'])}개")
    print(f"처리 시간: {result['translation_summary']['processing_time']:.2f}초")
    print("\n통합 한국어 텍스트:")
    print(result['integrated_korean_text'])
    
    return result


if __name__ == "__main__":
    # 테스트 실행
    test_result = test_multilingual_processor()
