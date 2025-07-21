#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다국어 번역 모듈 - 솔로몬드 AI 시스템 확장

주얼리 업계 특화 다국어 번역 시스템
- 한국어, 영어, 중국어, 일본어, 태국어 지원
- 주얼리 전문 용어 정확 번역
- 실시간 번역 최적화

Author: 전근혁 (솔로몬드 대표)
Date: 2025.07.08
"""

import logging
from typing import Dict, Optional, List, Any
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    # 일관된 언어 감지를 위한 시드 설정
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

import re

logger = logging.getLogger(__name__)

class JewelryMultilingualTranslator:
    """솔로몬드 주얼리 업계 특화 다국어 번역기"""
    
    def __init__(self):
        """번역기 초기화"""
        if TRANSLATOR_AVAILABLE:
            self.translator = GoogleTranslator()
        else:
            self.translator = None
        self.jewelry_glossary = self._load_jewelry_glossary()
        self.supported_languages = ['ko', 'en', 'zh', 'ja', 'th']
        
        # 언어 코드 매핑
        self.language_names = {
            'ko': '한국어',
            'en': 'English', 
            'zh': '中文',
            'ja': '日本語',
            'th': 'ไทย'
        }
        
        logger.info("주얼리 다국어 번역기 초기화 완료")
        
    def _load_jewelry_glossary(self) -> Dict[str, Dict[str, str]]:
        """주얼리 전문 용어 다국어 사전 로딩"""
        glossary = {
            # 다이아몬드 관련
            "diamond": {
                "ko": "다이아몬드",
                "en": "diamond", 
                "zh": "钻石",
                "ja": "ダイヤモンド",
                "th": "เพชร"
            },
            "carat": {
                "ko": "캐럿",
                "en": "carat",
                "zh": "克拉", 
                "ja": "カラット",
                "th": "กะรัต"
            },
            "cut": {
                "ko": "컷",
                "en": "cut",
                "zh": "切工",
                "ja": "カット", 
                "th": "การตัด"
            },
            "color": {
                "ko": "컬러",
                "en": "color",
                "zh": "颜色",
                "ja": "カラー",
                "th": "สี"
            },
            "clarity": {
                "ko": "클래리티",
                "en": "clarity", 
                "zh": "净度",
                "ja": "クラリティ",
                "th": "ความใส"
            },
            "solitaire": {
                "ko": "솔리테어",
                "en": "solitaire",
                "zh": "单钻戒指",
                "ja": "ソリテール",
                "th": "โซลิแทร์"
            },
            
            # 금속 관련
            "gold": {
                "ko": "금",
                "en": "gold",
                "zh": "黄金",
                "ja": "ゴールド",
                "th": "ทอง"
            },
            "white_gold": {
                "ko": "화이트골드", 
                "en": "white gold",
                "zh": "白金",
                "ja": "ホワイトゴールド",
                "th": "ทองขาว"
            },
            "rose_gold": {
                "ko": "로즈골드",
                "en": "rose gold",
                "zh": "玫瑰金",
                "ja": "ローズゴールド", 
                "th": "ทองกุหลาบ"
            },
            "platinum": {
                "ko": "플래티넘",
                "en": "platinum",
                "zh": "铂金",
                "ja": "プラチナ",
                "th": "แพลทินัม"
            },
            "silver": {
                "ko": "은",
                "en": "silver", 
                "zh": "银",
                "ja": "シルバー",
                "th": "เงิน"
            },
            
            # 보석 관련
            "ruby": {
                "ko": "루비",
                "en": "ruby",
                "zh": "红宝石",
                "ja": "ルビー",
                "th": "ทับทิม"
            },
            "sapphire": {
                "ko": "사파이어", 
                "en": "sapphire",
                "zh": "蓝宝石",
                "ja": "サファイア",
                "th": "ไพลิน"
            },
            "emerald": {
                "ko": "에메랄드",
                "en": "emerald",
                "zh": "祖母绿",
                "ja": "エメラルド",
                "th": "มรกต"
            },
            "pearl": {
                "ko": "진주",
                "en": "pearl",
                "zh": "珍珠", 
                "ja": "パール",
                "th": "ไข่มุก"
            },
            
            # 주얼리 종류
            "ring": {
                "ko": "반지",
                "en": "ring",
                "zh": "戒指",
                "ja": "リング",
                "th": "แหวน"
            },
            "necklace": {
                "ko": "목걸이",
                "en": "necklace",
                "zh": "项链", 
                "ja": "ネックレス",
                "th": "สร้อยคอ"
            },
            "earring": {
                "ko": "귀걸이",
                "en": "earring", 
                "zh": "耳环",
                "ja": "イヤリング",
                "th": "ต่างหู"
            },
            "bracelet": {
                "ko": "팔찌",
                "en": "bracelet",
                "zh": "手镯",
                "ja": "ブレスレット",
                "th": "สร้อยข้อมือ"
            },
            
            # 세팅 방식
            "prong_setting": {
                "ko": "프롱 세팅",
                "en": "prong setting",
                "zh": "爪镶",
                "ja": "プロングセッティング",
                "th": "การติดตั้งแบบเล็บ"
            },
            "bezel_setting": {
                "ko": "베젤 세팅",
                "en": "bezel setting",
                "zh": "包镶",
                "ja": "ベゼルセッティング",
                "th": "การติดตั้งแบบล้อม"
            },
            "pave_setting": {
                "ko": "파베 세팅",
                "en": "pave setting",
                "zh": "密镶",
                "ja": "パヴェセッティング",
                "th": "การติดตั้งแบบปูพื้น"
            },
            
            # 비즈니스 용어
            "wholesale": {
                "ko": "도매",
                "en": "wholesale", 
                "zh": "批发",
                "ja": "卸売り",
                "th": "ขายส่ง"
            },
            "retail": {
                "ko": "소매",
                "en": "retail",
                "zh": "零售",
                "ja": "小売り", 
                "th": "ขายปลีก"
            },
            "certificate": {
                "ko": "감정서",
                "en": "certificate",
                "zh": "证书",
                "ja": "鑑定書",
                "th": "ใบรับรอง"
            },
            "appraisal": {
                "ko": "감정",
                "en": "appraisal",
                "zh": "评估",
                "ja": "鑑定",
                "th": "การประเมิน"
            },
            
            # 품질 등급
            "gia": {
                "ko": "GIA",
                "en": "GIA",
                "zh": "美国宝石学院",
                "ja": "GIA",
                "th": "GIA"
            },
            "four_c": {
                "ko": "4C",
                "en": "4C",
                "zh": "4C标准",
                "ja": "4C",
                "th": "4C"
            }
        }
        
        logger.info(f"주얼리 용어 사전 로딩: {len(glossary)}개 용어")
        return glossary
    
    def detect_language(self, text: str) -> str:
        """텍스트 언어 자동 감지"""
        try:
            if not text or not text.strip():
                return 'unknown'
            
            if not LANGDETECT_AVAILABLE:
                # 기본 언어 감지 (간단한 패턴 기반)
                if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text):
                    return 'ko'  # 한글
                elif any(ord(char) >= 0x4E00 and ord(char) <= 0x9FFF for char in text):
                    return 'zh'  # 중문
                elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in text):
                    return 'ja'  # 일문
                else:
                    return 'en'  # 기본값
            
            detected = detect(text)
            
            # 지원 언어로 매핑
            if detected in self.supported_languages:
                return detected
            elif detected == 'zh-cn' or detected == 'zh':
                return 'zh'
            else:
                return detected
                
        except Exception as e:
            logger.warning(f"언어 감지 실패: {str(e)}")
            return 'unknown'
    
    def enhance_with_glossary(self, text: str, source_lang: str, target_lang: str) -> str:
        """주얼리 용어 사전을 활용한 번역 개선"""
        enhanced_text = text
        
        for key, translations in self.jewelry_glossary.items():
            source_term = translations.get(source_lang, "").lower()
            target_term = translations.get(target_lang, "")
            
            if source_term and target_term and source_term in text.lower():
                pattern = re.compile(re.escape(source_term), re.IGNORECASE)
                enhanced_text = pattern.sub(target_term, enhanced_text)
        
        return enhanced_text
    
    def translate(self, 
                 text: str, 
                 target_lang: str, 
                 source_lang: Optional[str] = None) -> Dict[str, Any]:
        """
        텍스트 번역
        
        Args:
            text: 번역할 텍스트
            target_lang: 목표 언어 코드
            source_lang: 원본 언어 코드 (None이면 자동 감지)
            
        Returns:
            번역 결과 딕셔너리
        """
        try:
            if not text or not text.strip():
                return {
                    'translated_text': '',
                    'source_language': 'unknown',
                    'target_language': target_lang,
                    'confidence': 0.0,
                    'jewelry_enhanced': False
                }
            
            # 언어 감지
            if source_lang is None:
                source_lang = self.detect_language(text)
            
            # 같은 언어면 번역하지 않음
            if source_lang == target_lang:
                return {
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang, 
                    'confidence': 1.0,
                    'jewelry_enhanced': False
                }
            
            # Deep Translator API 사용
            if not self.translator:
                raise Exception("번역기를 사용할 수 없습니다")
            
            # deep-translator 사용법에 맞게 수정
            translated_text = self.translator.translate(
                text=text,
                target_language=target_lang
            )
            
            confidence = 0.8  # deep-translator는 confidence를 제공하지 않음
            
            # 주얼리 용어 사전으로 개선
            enhanced_text = self.enhance_with_glossary(
                translated_text, 
                source_lang, 
                target_lang
            )
            
            jewelry_enhanced = enhanced_text != translated_text
            
            return {
                'translated_text': enhanced_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': confidence if confidence else 0.8,
                'jewelry_enhanced': jewelry_enhanced,
                'original_translation': translated_text if jewelry_enhanced else None
            }
            
        except Exception as e:
            logger.error(f"번역 오류: {str(e)}")
            return {
                'translated_text': text,
                'source_language': source_lang or 'unknown', 
                'target_language': target_lang,
                'confidence': 0.0,
                'jewelry_enhanced': False,
                'error': str(e)
            }
    
    def translate_multiple(self, 
                          text: str, 
                          target_languages: List[str],
                          source_lang: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """여러 언어로 동시 번역"""
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        results = {}
        for target_lang in target_languages:
            if target_lang != source_lang:
                results[target_lang] = self.translate(text, target_lang, source_lang)
        
        return results
    
    def get_jewelry_terms(self, language: str) -> List[str]:
        """특정 언어의 주얼리 용어 목록 반환"""
        terms = []
        for term_data in self.jewelry_glossary.values():
            if language in term_data:
                terms.append(term_data[language])
        return terms
    
    def search_glossary(self, query: str, language: str = 'ko') -> List[Dict[str, Any]]:
        """주얼리 용어 사전 검색"""
        results = []
        query_lower = query.lower()
        
        for key, translations in self.jewelry_glossary.items():
            if language in translations:
                term = translations[language].lower()
                if query_lower in term:
                    results.append({
                        'key': key,
                        'translations': translations,
                        'matched_term': translations[language]
                    })
        
        return results
    
    def is_ready(self) -> bool:
        """번역 서비스 준비 상태 확인"""
        try:
            if not self.translator:
                return False
            # deep-translator로 간단한 테스트
            test_result = self.translator.translate(text="test", target_language='ko')
            return bool(test_result)
        except:
            return False
    
    def get_supported_languages(self) -> List[str]:
        """지원 언어 목록 반환"""
        return self.supported_languages
    
    def get_language_name(self, lang_code: str) -> str:
        """언어 코드로 언어명 반환"""
        return self.language_names.get(lang_code, lang_code)
