"""
솔로몬드 AI v2.3 - googletrans 없이 작동하는 수정된 다국어 번역기
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class BackupLanguageDetector:
    """백업 언어 감지기"""
    
    def __init__(self):
        # 언어별 특징 패턴
        self.patterns = {
            'ko': [
                r'[가-힣]',  # 한글
                r'(?:입니다|습니다|해요|에요|이다|다)$',  # 한국어 어미
                r'(?:그리고|그러나|하지만|따라서)'  # 한국어 접속사
            ],
            'en': [
                r'[a-zA-Z]',  # 영문자
                r'(?:the|and|or|but|in|on|at|to|for|of|with|by)\s',  # 영어 전치사/접속사
                r'(?:ing|ed|ly|tion|sion)$'  # 영어 접미사
            ],
            'zh': [
                r'[\u4e00-\u9fff]',  # 중국어 한자
                r'(?:的|是|在|有|和|或|但|因为|所以)',  # 중국어 조사/접속사
            ],
            'ja': [
                r'[\u3040-\u309f]',  # 히라가나
                r'[\u30a0-\u30ff]',  # 가타카나
                r'(?:です|ます|でした|ました|だ|である)',  # 일본어 어미
            ]
        }
    
    def detect_language(self, text: str) -> str:
        """언어 감지"""
        if not text or len(text.strip()) == 0:
            return 'en'
        
        text = text.lower().strip()
        scores = {}
        
        for lang, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text)
                score += len(matches)
            
            # 정규화
            scores[lang] = score / len(text) if len(text) > 0 else 0
        
        # 가장 높은 점수의 언어 반환
        detected_lang = max(scores, key=scores.get) if scores else 'en'
        
        # 임계값 확인 (너무 낮으면 영어로 기본 설정)
        if scores[detected_lang] < 0.1:
            detected_lang = 'en'
        
        logger.info(f"언어 감지 결과: {detected_lang} (점수: {scores})")
        return detected_lang

class BackupTranslator:
    """백업 번역기 - 간단한 규칙 기반 번역"""
    
    def __init__(self):
        self.detector = BackupLanguageDetector()
        
        # 주얼리 전문용어 번역 사전
        self.jewelry_dict = {
            'en_to_ko': {
                'diamond': '다이아몬드',
                'ruby': '루비', 
                'sapphire': '사파이어',
                'emerald': '에메랄드',
                'pearl': '진주',
                'gold': '금',
                'silver': '은',
                'platinum': '플래티넘',
                'carat': '캐럿',
                'cut': '컷',
                'color': '컬러',
                'clarity': '클래리티',
                'certificate': '감정서',
                'GIA': 'GIA',
                'quality': '품질',
                'grade': '등급',
                'price': '가격',
                'value': '가치',
                'investment': '투자',
                'jewelry': '주얼리',
                'ring': '반지',
                'necklace': '목걸이',
                'earring': '귀걸이',
                'bracelet': '팔찌',
                'watch': '시계'
            },
            'ko_to_en': {
                '다이아몬드': 'diamond',
                '루비': 'ruby',
                '사파이어': 'sapphire', 
                '에메랄드': 'emerald',
                '진주': 'pearl',
                '금': 'gold',
                '은': 'silver',
                '플래티넘': 'platinum',
                '캐럿': 'carat',
                '컷': 'cut',
                '컬러': 'color',
                '클래리티': 'clarity',
                '감정서': 'certificate',
                '품질': 'quality',
                '등급': 'grade',
                '가격': 'price',
                '가치': 'value',
                '투자': 'investment',
                '주얼리': 'jewelry',
                '반지': 'ring',
                '목걸이': 'necklace',
                '귀걸이': 'earring',
                '팔찌': 'bracelet',
                '시계': 'watch'
            }
        }
    
    def detect(self, text: str):
        """언어 감지 (googletrans 호환)"""
        lang = self.detector.detect_language(text)
        return type('DetectionResult', (), {'lang': lang})()
    
    def translate(self, text: str, dest: str = 'ko', src: str = 'auto'):
        """번역 실행 (googletrans 호환)"""
        
        if src == 'auto':
            detected = self.detect(text)
            src_lang = detected.lang
        else:
            src_lang = src
        
        # 같은 언어면 원문 반환
        if src_lang == dest:
            translated_text = text
        else:
            translated_text = self._translate_text(text, src_lang, dest)
        
        # googletrans 호환 결과 객체
        return type('TranslationResult', (), {
            'text': translated_text,
            'src': src_lang,
            'dest': dest
        })()
    
    def _translate_text(self, text: str, src_lang: str, dest_lang: str) -> str:
        """실제 번역 수행"""
        
        # 주얼리 전문용어 번역
        if src_lang == 'en' and dest_lang == 'ko':
            return self._apply_dictionary(text, self.jewelry_dict['en_to_ko'])
        elif src_lang == 'ko' and dest_lang == 'en':
            return self._apply_dictionary(text, self.jewelry_dict['ko_to_en'])
        else:
            # 기타 언어는 원문에 표시 추가
            return f"[{src_lang}→{dest_lang}] {text}"
    
    def _apply_dictionary(self, text: str, translation_dict: Dict[str, str]) -> str:
        """사전 기반 번역 적용"""
        
        result = text
        
        # 단어별로 번역 적용
        for source_word, target_word in translation_dict.items():
            # 대소문자 구분 없이 치환
            pattern = re.compile(re.escape(source_word), re.IGNORECASE)
            result = pattern.sub(target_word, result)
        
        return result

class JewelryMultilingualTranslator:
    """주얼리 전문 다국어 번역기 (백업 모드)"""
    
    def __init__(self):
        try:
            # googletrans 사용 시도
            from googletrans import Translator
            self.translator = Translator()
            self.backup_mode = False
            logger.info("✅ googletrans 모드 활성화")
        except (ImportError, AttributeError) as e:
            # 백업 번역기 사용
            self.translator = BackupTranslator()
            self.backup_mode = True
            logger.warning(f"⚠️ 백업 번역기 모드 활성화: {e}")
        
        self.supported_languages = {
            'ko': '한국어',
            'en': '영어', 
            'zh': '중국어',
            'ja': '일본어'
        }
    
    def detect_language(self, text: str) -> str:
        """언어 감지"""
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            logger.error(f"언어 감지 오류: {e}")
            return 'en'  # 기본값
    
    def translate_text(self, text: str, target_lang: str = 'ko') -> str:
        """텍스트 번역"""
        try:
            if not text or len(text.strip()) == 0:
                return text
            
            # 언어 감지
            detected_lang = self.detect_language(text)
            
            # 같은 언어면 원문 반환
            if detected_lang == target_lang:
                return text
            
            # 번역 실행
            result = self.translator.translate(text, dest=target_lang)
            translated_text = result.text
            
            # 백업 모드일 때 표시
            if self.backup_mode:
                translated_text = f"[백업번역] {translated_text}"
            
            logger.info(f"번역 완료: {detected_lang} → {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"번역 오류: {e}")
            return f"[번역실패] {text}"
    
    def translate_jewelry_terms(self, text: str) -> Dict[str, str]:
        """주얼리 전문용어 다국어 번역"""
        
        results = {}
        
        for lang_code, lang_name in self.supported_languages.items():
            try:
                translated = self.translate_text(text, lang_code)
                results[lang_code] = translated
            except Exception as e:
                logger.error(f"{lang_name} 번역 오류: {e}")
                results[lang_code] = text
        
        return results
    
    def extract_and_translate_keywords(self, text: str, target_lang: str = 'ko') -> List[Tuple[str, str]]:
        """키워드 추출 및 번역"""
        
        # 주얼리 관련 키워드 패턴
        jewelry_patterns = [
            r'\b(?:diamond|ruby|sapphire|emerald|pearl)\b',
            r'\b(?:carat|cut|color|clarity)\b',
            r'\b(?:GIA|AGS|SSEF|Gübelin)\b',
            r'\b(?:certification|certificate|grade|quality)\b',
            r'\b(?:다이아몬드|루비|사파이어|에메랄드|진주)\b',
            r'\b(?:캐럿|컷|컬러|클래리티)\b',
            r'\b(?:감정서|품질|등급)\b'
        ]
        
        extracted_keywords = []
        
        for pattern in jewelry_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    translated = self.translate_text(match, target_lang)
                    extracted_keywords.append((match, translated))
                except Exception as e:
                    logger.error(f"키워드 번역 오류 {match}: {e}")
                    extracted_keywords.append((match, match))
        
        # 중복 제거
        return list(set(extracted_keywords))
    
    def get_translation_summary(self, text: str) -> Dict[str, any]:
        """번역 요약 정보"""
        
        detected_lang = self.detect_language(text)
        
        summary = {
            'original_text': text,
            'detected_language': detected_lang,
            'detected_language_name': self.supported_languages.get(detected_lang, '알 수 없음'),
            'translations': self.translate_jewelry_terms(text),
            'keywords': self.extract_and_translate_keywords(text),
            'backup_mode': self.backup_mode,
            'translation_quality': 'high' if not self.backup_mode else 'basic'
        }
        
        return summary

# 전역 인스턴스
_global_translator = None

def get_jewelry_translator():
    """전역 번역기 인스턴스 반환"""
    global _global_translator
    if _global_translator is None:
        _global_translator = JewelryMultilingualTranslator()
    return _global_translator

# 호환성을 위한 별칭
Translator = BackupTranslator
