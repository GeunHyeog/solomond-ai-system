#!/usr/bin/env python3
"""
다국어 지원 확장 모듈 - 영어, 중국어, 일본어, 한국어 지원
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime

# 언어 감지를 위한 라이브러리들
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# 번역을 위한 라이브러리들
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

class MultilingualProcessor:
    """다국어 처리 클래스"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 지원 언어 설정
        self.supported_languages = {
            'ko': {
                'name': '한국어',
                'name_en': 'Korean',
                'whisper_code': 'ko',
                'jewelry_terms': [
                    '주얼리', '다이아몬드', '금', '은', '백금', '반지', '목걸이',
                    '팔찌', '귀걸이', '펜던트', '보석', '루비', '사파이어',
                    '에메랄드', '진주', '크리스탈', '럭셔리', '캐럿', '커팅',
                    '투명도', '색상', '인증서', '지아', '보석상', '쥬얼러',
                    '세팅', '프롱', '베젤', '채널', '파베', '마키즈', '오벌',
                    '라운드', '프린세스', '에메랄드컷', '아시아컷', '로즈골드'
                ]
            },
            'en': {
                'name': 'English',
                'name_en': 'English',
                'whisper_code': 'en',
                'jewelry_terms': [
                    'jewelry', 'jewellery', 'diamond', 'gold', 'silver', 'platinum',
                    'ring', 'necklace', 'bracelet', 'earring', 'pendant', 'gemstone',
                    'ruby', 'sapphire', 'emerald', 'pearl', 'crystal', 'luxury',
                    'carat', 'cut', 'clarity', 'color', 'certificate', 'GIA',
                    'setting', 'prong', 'bezel', 'channel', 'pave', 'marquise',
                    'oval', 'round', 'princess', 'emerald cut', 'cushion',
                    'rose gold', 'white gold', 'yellow gold', 'fine jewelry',
                    'fashion jewelry', 'engagement ring', 'wedding band',
                    'anniversary', 'birthstone', 'precious', 'semi-precious'
                ]
            },
            'zh': {
                'name': '中文',
                'name_en': 'Chinese',
                'whisper_code': 'zh',
                'jewelry_terms': [
                    '珠宝', '钻石', '黄金', '白银', '铂金', '戒指', '项链',
                    '手镯', '耳环', '吊坠', '宝石', '红宝石', '蓝宝石',
                    '祖母绿', '珍珠', '水晶', '奢侈品', '克拉', '切工',
                    '净度', '颜色', '证书', 'GIA', '镶嵌', '爪镶', '包镶',
                    '槽镶', '密镶', '马眼形', '椭圆形', '圆形', '公主方',
                    '祖母绿形', '垫形', '玫瑰金', '白金', '黄金', '高级珠宝',
                    '时尚珠宝', '订婚戒指', '结婚戒指', '周年纪念', '诞生石'
                ]
            },
            'ja': {
                'name': '日本語',
                'name_en': 'Japanese',
                'whisper_code': 'ja',
                'jewelry_terms': [
                    'ジュエリー', 'ダイヤモンド', '金', '銀', 'プラチナ', '指輪',
                    'ネックレス', 'ブレスレット', 'イヤリング', 'ペンダント',
                    '宝石', 'ルビー', 'サファイア', 'エメラルド', '真珠',
                    'クリスタル', 'ラグジュアリー', 'カラット', 'カット',
                    '透明度', '色', '証明書', 'GIA', 'セッティング', '爪留め',
                    'ベゼル', 'チャンネル', 'パヴェ', 'マーキス', 'オーバル',
                    'ラウンド', 'プリンセス', 'エメラルドカット', 'クッション',
                    'ローズゴールド', 'ホワイトゴールド', 'イエローゴールド',
                    'ファインジュエリー', 'ファッションジュエリー', '婚約指輪',
                    '結婚指輪', '記念日', '誕生石', '貴石', '半貴石'
                ]
            }
        }
        
        # 번역기 초기화
        self.translator = None
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translator = Translator()
                self.logger.info("[INFO] Google Translator 초기화 완료")
            except Exception as e:
                self.logger.warning(f"[WARNING] Google Translator 초기화 실패: {e}")
        
        self._check_dependencies()
    
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
    
    def _check_dependencies(self):
        """의존성 확인"""
        if LANGDETECT_AVAILABLE:
            self.logger.info("[INFO] langdetect 사용 가능 - 언어 자동 감지 지원")
        else:
            self.logger.warning("[WARNING] langdetect 미설치 - 언어 감지 제한됨")
        
        if GOOGLETRANS_AVAILABLE:
            self.logger.info("[INFO] googletrans 사용 가능 - 번역 지원")
        else:
            self.logger.warning("[WARNING] googletrans 미설치 - 번역 기능 제한됨")
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """텍스트 언어 감지"""
        if not text or len(text.strip()) < 3:
            return {
                "status": "error",
                "error": "텍스트가 너무 짧습니다 (최소 3자 필요)"
            }
        
        if not LANGDETECT_AVAILABLE:
            return {
                "status": "error",
                "error": "langdetect가 설치되지 않음",
                "install_command": "pip install langdetect"
            }
        
        try:
            self.logger.info(f"[INFO] 언어 감지 시작: {text[:50]}...")
            
            # 주 언어 감지
            detected_lang = detect(text)
            
            # 확률적 결과 (여러 가능성)
            lang_probs = detect_langs(text)
            
            # 지원 언어 매핑
            mapped_lang = self._map_language_code(detected_lang)
            
            # 결과 구성
            result = {
                "status": "success",
                "detected_language": detected_lang,
                "mapped_language": mapped_lang,
                "confidence": lang_probs[0].prob if lang_probs else 0.0,
                "all_probabilities": [
                    {
                        "language": str(lang.lang),
                        "probability": round(lang.prob, 3),
                        "mapped": self._map_language_code(str(lang.lang))
                    }
                    for lang in lang_probs[:5]  # 상위 5개만
                ],
                "is_supported": mapped_lang in self.supported_languages,
                "language_info": self.supported_languages.get(mapped_lang, {})
            }
            
            self.logger.info(f"[SUCCESS] 언어 감지 완료: {detected_lang} -> {mapped_lang}")
            return result
            
        except Exception as e:
            error_msg = f"언어 감지 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "text_sample": text[:100]
            }
    
    def _map_language_code(self, lang_code: str) -> str:
        """언어 코드 매핑"""
        # langdetect 결과를 우리 시스템 코드로 매핑
        mapping = {
            'ko': 'ko',
            'en': 'en', 
            'zh-cn': 'zh',
            'zh': 'zh',
            'ja': 'ja',
            'zh-tw': 'zh'  # 번체 중국어도 zh로 통합
        }
        
        return mapping.get(lang_code.lower(), lang_code)
    
    def translate_text(self, text: str, target_lang: str = 'ko', source_lang: str = 'auto') -> Dict[str, Any]:
        """텍스트 번역"""
        if not text or len(text.strip()) == 0:
            return {
                "status": "error",
                "error": "번역할 텍스트가 비어있습니다"
            }
        
        if not GOOGLETRANS_AVAILABLE or not self.translator:
            return {
                "status": "error",
                "error": "googletrans가 설치되지 않았거나 초기화되지 않음",
                "install_command": "pip install googletrans==4.0.0rc1"
            }
        
        try:
            self.logger.info(f"[INFO] 번역 시작: {source_lang} -> {target_lang}")
            
            # 번역 수행
            if source_lang == 'auto':
                translation = self.translator.translate(text, dest=target_lang)
                detected_source = translation.src
            else:
                translation = self.translator.translate(text, src=source_lang, dest=target_lang)
                detected_source = source_lang
            
            result = {
                "status": "success",
                "original_text": text,
                "translated_text": translation.text,
                "source_language": detected_source,
                "target_language": target_lang,
                "source_language_info": self.supported_languages.get(detected_source, {}),
                "target_language_info": self.supported_languages.get(target_lang, {}),
                "confidence": getattr(translation, 'confidence', None),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"[SUCCESS] 번역 완료: {detected_source} -> {target_lang}")
            return result
            
        except Exception as e:
            error_msg = f"번역 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "original_text": text[:100],
                "source_language": source_lang,
                "target_language": target_lang
            }
    
    def extract_multilingual_jewelry_keywords(self, text: str, language: str = 'auto') -> Dict[str, Any]:
        """다국어 주얼리 키워드 추출"""
        if not text:
            return {
                "status": "error",
                "error": "분석할 텍스트가 비어있습니다"
            }
        
        try:
            self.logger.info(f"[INFO] 다국어 키워드 추출 시작: {language}")
            
            # 언어 감지 (auto인 경우)
            if language == 'auto':
                lang_result = self.detect_language(text)
                if lang_result['status'] == 'success':
                    detected_lang = lang_result['mapped_language']
                else:
                    detected_lang = 'en'  # 기본값
            else:
                detected_lang = language
            
            # 해당 언어의 주얼리 용어 가져오기
            jewelry_terms = []
            matched_keywords = []
            
            # 감지된 언어의 키워드
            if detected_lang in self.supported_languages:
                jewelry_terms = self.supported_languages[detected_lang]['jewelry_terms']
                text_lower = text.lower()
                
                for term in jewelry_terms:
                    if term.lower() in text_lower:
                        matched_keywords.append({
                            "keyword": term,
                            "language": detected_lang,
                            "category": self._categorize_keyword(term, detected_lang)
                        })
            
            # 모든 언어에서 추가 검색 (다국어 혼재 텍스트 대응)
            cross_language_matches = []
            for lang_code, lang_info in self.supported_languages.items():
                if lang_code != detected_lang:
                    text_lower = text.lower()
                    for term in lang_info['jewelry_terms']:
                        if term.lower() in text_lower:
                            cross_language_matches.append({
                                "keyword": term,
                                "language": lang_code,
                                "category": self._categorize_keyword(term, lang_code)
                            })
            
            # 결과 구성
            result = {
                "status": "success",
                "detected_language": detected_lang,
                "primary_keywords": matched_keywords,
                "cross_language_keywords": cross_language_matches,
                "total_keywords": len(matched_keywords) + len(cross_language_matches),
                "keyword_density": round(
                    (len(matched_keywords) + len(cross_language_matches)) / max(len(text.split()), 1) * 100, 2
                ),
                "language_analysis": {
                    "primary_language": detected_lang,
                    "is_multilingual": len(cross_language_matches) > 0,
                    "languages_detected": list(set([detected_lang] + [kw['language'] for kw in cross_language_matches]))
                }
            }
            
            self.logger.info(f"[SUCCESS] 키워드 추출 완료: {len(matched_keywords) + len(cross_language_matches)}개")
            return result
            
        except Exception as e:
            error_msg = f"키워드 추출 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "text_sample": text[:100],
                "language": language
            }
    
    def _categorize_keyword(self, keyword: str, language: str) -> str:
        """키워드 카테고리 분류"""
        keyword_lower = keyword.lower()
        
        # 카테고리 정의 (다국어)
        categories = {
            'materials': {
                'ko': ['금', '은', '백금', '다이아몬드', '루비', '사파이어', '에메랄드', '진주'],
                'en': ['gold', 'silver', 'platinum', 'diamond', 'ruby', 'sapphire', 'emerald', 'pearl'],
                'zh': ['黄金', '白银', '铂金', '钻石', '红宝石', '蓝宝石', '祖母绿', '珍珠'],
                'ja': ['金', '銀', 'プラチナ', 'ダイヤモンド', 'ルビー', 'サファイア', 'エメラルド', '真珠']
            },
            'products': {
                'ko': ['반지', '목걸이', '팔찌', '귀걸이', '펜던트'],
                'en': ['ring', 'necklace', 'bracelet', 'earring', 'pendant'],
                'zh': ['戒指', '项链', '手镯', '耳环', '吊坠'],
                'ja': ['指輪', 'ネックレス', 'ブレスレット', 'イヤリング', 'ペンダント']
            },
            'technical': {
                'ko': ['캐럿', '커팅', '투명도', '색상', '인증서'],
                'en': ['carat', 'cut', 'clarity', 'color', 'certificate'],
                'zh': ['克拉', '切工', '净度', '颜色', '证书'],
                'ja': ['カラット', 'カット', '透明度', '色', '証明書']
            }
        }
        
        # 카테고리 매칭
        for category, lang_terms in categories.items():
            if language in lang_terms and keyword_lower in [term.lower() for term in lang_terms[language]]:
                return category
        
        return 'general'
    
    def create_multilingual_summary(self, analysis_results: Dict[str, Any], target_languages: List[str] = None) -> Dict[str, Any]:
        """다국어 요약 생성"""
        if target_languages is None:
            target_languages = ['ko', 'en']
        
        try:
            self.logger.info(f"[INFO] 다국어 요약 생성: {target_languages}")
            
            # 원본 요약 추출
            original_summary = analysis_results.get('summary', '')
            if not original_summary:
                return {
                    "status": "error",
                    "error": "요약할 내용이 없습니다"
                }
            
            summaries = {}
            
            # 각 언어로 번역
            for lang in target_languages:
                if lang == 'ko':  # 한국어는 원본으로 가정
                    summaries[lang] = {
                        "text": original_summary,
                        "language": lang,
                        "translation_status": "original"
                    }
                else:
                    # 번역 수행
                    translation_result = self.translate_text(original_summary, target_lang=lang, source_lang='ko')
                    
                    if translation_result['status'] == 'success':
                        summaries[lang] = {
                            "text": translation_result['translated_text'],
                            "language": lang,
                            "translation_status": "translated",
                            "confidence": translation_result.get('confidence')
                        }
                    else:
                        summaries[lang] = {
                            "text": f"Translation failed: {translation_result.get('error', 'Unknown error')}",
                            "language": lang,
                            "translation_status": "failed",
                            "error": translation_result.get('error')
                        }
            
            result = {
                "status": "success",
                "summaries": summaries,
                "target_languages": target_languages,
                "original_language": "ko",
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"[SUCCESS] 다국어 요약 생성 완료: {len(summaries)}개 언어")
            return result
            
        except Exception as e:
            error_msg = f"다국어 요약 생성 오류: {str(e)}"
            self.logger.error(f"[ERROR] {error_msg}")
            
            return {
                "status": "error",
                "error": error_msg,
                "target_languages": target_languages
            }
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """지원 언어 목록"""
        return {
            "supported_languages": self.supported_languages,
            "total_languages": len(self.supported_languages),
            "language_codes": list(self.supported_languages.keys()),
            "features": {
                "language_detection": LANGDETECT_AVAILABLE,
                "translation": GOOGLETRANS_AVAILABLE,
                "multilingual_keywords": True,
                "cross_language_analysis": True
            }
        }
    
    def get_installation_guide(self) -> Dict[str, Any]:
        """설치 가이드"""
        missing_packages = []
        
        if not LANGDETECT_AVAILABLE:
            missing_packages.append({
                "package": "langdetect",
                "command": "pip install langdetect",
                "purpose": "언어 자동 감지"
            })
        
        if not GOOGLETRANS_AVAILABLE:
            missing_packages.append({
                "package": "googletrans",
                "command": "pip install googletrans==4.0.0rc1",
                "purpose": "텍스트 번역"
            })
        
        return {
            "available_features": {
                "language_detection": LANGDETECT_AVAILABLE,
                "translation": GOOGLETRANS_AVAILABLE
            },
            "missing_packages": missing_packages,
            "install_all": "pip install langdetect googletrans==4.0.0rc1",
            "additional_notes": [
                "googletrans 버전 호환성에 주의하세요",
                "인터넷 연결이 필요한 기능들이 있습니다",
                "대용량 텍스트 번역 시 API 제한이 있을 수 있습니다"
            ]
        }

# 전역 인스턴스
multilingual_processor = MultilingualProcessor()

def detect_language(text: str) -> Dict[str, Any]:
    """언어 감지 (전역 접근용)"""
    return multilingual_processor.detect_language(text)

def translate_text(text: str, target_lang: str = 'ko', source_lang: str = 'auto') -> Dict[str, Any]:
    """텍스트 번역 (전역 접근용)"""
    return multilingual_processor.translate_text(text, target_lang, source_lang)