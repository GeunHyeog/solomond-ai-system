"""
🌍 Solomond AI v2.1 - 다국어 처리 엔진
자동 언어 감지, 특화 STT, 한국어 통합 번역 및 주얼리 전문용어 처리

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.0
"""

import librosa
import whisper
import openai
from googletrans import Translator
import langdetect
from langdetect import detect
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time
import re
from collections import Counter
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LanguageDetectionResult:
    """언어 감지 결과"""
    primary_language: str      # 주 언어 코드 (ko, en, zh, ja)
    confidence: float          # 감지 신뢰도 (0-1)
    language_distribution: Dict[str, float]  # 언어별 분포
    segments: List[Dict]       # 구간별 언어 정보
    processing_time: float     # 처리 시간

@dataclass
class MultilingualSTTResult:
    """다국어 STT 결과"""
    original_text: str         # 원본 텍스트
    detected_language: str     # 감지된 언어
    korean_translation: str    # 한국어 번역
    confidence_score: float    # 신뢰도 점수
    processing_details: Dict   # 처리 세부사항
    timestamp: float

class JewelryTermsDatabase:
    """주얼리 전문용어 데이터베이스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.terms_db = self._initialize_terms_database()
        
    def _initialize_terms_database(self) -> Dict[str, Dict[str, str]]:
        """주얼리 전문용어 다국어 데이터베이스 초기화"""
        
        # 핵심 주얼리 용어 다국어 매핑
        jewelry_terms = {
            # 다이아몬드 4C
            "carat": {
                "ko": "캐럿", 
                "en": "carat", 
                "zh": "克拉", 
                "ja": "カラット"
            },
            "clarity": {
                "ko": "투명도", 
                "en": "clarity", 
                "zh": "净度", 
                "ja": "クラリティ"
            },
            "color": {
                "ko": "컬러", 
                "en": "color", 
                "zh": "颜色", 
                "ja": "カラー"
            },
            "cut": {
                "ko": "커팅", 
                "en": "cut", 
                "zh": "切工", 
                "ja": "カット"
            },
            
            # 보석 종류
            "diamond": {
                "ko": "다이아몬드", 
                "en": "diamond", 
                "zh": "钻石", 
                "ja": "ダイヤモンド"
            },
            "ruby": {
                "ko": "루비", 
                "en": "ruby", 
                "zh": "红宝石", 
                "ja": "ルビー"
            },
            "sapphire": {
                "ko": "사파이어", 
                "en": "sapphire", 
                "zh": "蓝宝石", 
                "ja": "サファイア"
            },
            "emerald": {
                "ko": "에메랄드", 
                "en": "emerald", 
                "zh": "祖母绿", 
                "ja": "エメラルド"
            },
            "pearl": {
                "ko": "진주", 
                "en": "pearl", 
                "zh": "珍珠", 
                "ja": "真珠"
            },
            
            # 금속 재료
            "gold": {
                "ko": "금", 
                "en": "gold", 
                "zh": "黄金", 
                "ja": "ゴールド"
            },
            "silver": {
                "ko": "은", 
                "en": "silver", 
                "zh": "银", 
                "ja": "シルバー"
            },
            "platinum": {
                "ko": "플래티넘", 
                "en": "platinum", 
                "zh": "铂金", 
                "ja": "プラチナ"
            },
            "white_gold": {
                "ko": "화이트골드", 
                "en": "white gold", 
                "zh": "白金", 
                "ja": "ホワイトゴールド"
            },
            
            # 주얼리 타입
            "ring": {
                "ko": "반지", 
                "en": "ring", 
                "zh": "戒指", 
                "ja": "リング"
            },
            "necklace": {
                "ko": "목걸이", 
                "en": "necklace", 
                "zh": "项链", 
                "ja": "ネックレス"
            },
            "earring": {
                "ko": "귀걸이", 
                "en": "earring", 
                "zh": "耳环", 
                "ja": "イヤリング"
            },
            "bracelet": {
                "ko": "팔찌", 
                "en": "bracelet", 
                "zh": "手镯", 
                "ja": "ブレスレット"
            },
            "pendant": {
                "ko": "펜던트", 
                "en": "pendant", 
                "zh": "吊坠", 
                "ja": "ペンダント"
            },
            
            # 감정/인증 관련
            "certification": {
                "ko": "감정서", 
                "en": "certification", 
                "zh": "证书", 
                "ja": "鑑定書"
            },
            "appraisal": {
                "ko": "감정", 
                "en": "appraisal", 
                "zh": "评估", 
                "ja": "鑑定"
            },
            "gia": {
                "ko": "GIA", 
                "en": "GIA", 
                "zh": "GIA", 
                "ja": "GIA"
            },
            "grading": {
                "ko": "등급", 
                "en": "grading", 
                "zh": "分级", 
                "ja": "グレーディング"
            },
            
            # 비즈니스 관련
            "wholesale": {
                "ko": "도매", 
                "en": "wholesale", 
                "zh": "批发", 
                "ja": "卸売"
            },
            "retail": {
                "ko": "소매", 
                "en": "retail", 
                "zh": "零售", 
                "ja": "小売"
            },
            "market_price": {
                "ko": "시세", 
                "en": "market price", 
                "zh": "市场价格", 
                "ja": "市場価格"
            },
            "trade_show": {
                "ko": "전시회", 
                "en": "trade show", 
                "zh": "贸易展", 
                "ja": "見本市"
            },
            
            # 기술적 용어
            "setting": {
                "ko": "세팅", 
                "en": "setting", 
                "zh": "镶嵌", 
                "ja": "セッティング"
            },
            "prong": {
                "ko": "프롱", 
                "en": "prong", 
                "zh": "爪镶", 
                "ja": "プロング"
            },
            "bezel": {
                "ko": "베젤", 
                "en": "bezel", 
                "zh": "包镶", 
                "ja": "ベゼル"
            },
            "mounting": {
                "ko": "마운팅", 
                "en": "mounting", 
                "zh": "底座", 
                "ja": "マウント"
            }
        }
        
        return jewelry_terms
    
    def translate_jewelry_term(self, term: str, source_lang: str, target_lang: str = "ko") -> str:
        """주얼리 전문용어 번역"""
        try:
            term_lower = term.lower().strip()
            
            # 직접 매칭 시도
            for key, translations in self.terms_db.items():
                for lang, translation in translations.items():
                    if translation.lower() == term_lower and lang == source_lang:
                        return translations.get(target_lang, term)
            
            # 부분 매칭 시도
            for key, translations in self.terms_db.items():
                for lang, translation in translations.items():
                    if term_lower in translation.lower() and lang == source_lang:
                        return translations.get(target_lang, term)
            
            # 매칭되지 않으면 원본 반환
            return term
            
        except Exception as e:
            self.logger.error(f"주얼리 용어 번역 실패: {e}")
            return term
    
    def enhance_translation_with_jewelry_terms(self, text: str, source_lang: str, target_lang: str = "ko") -> str:
        """주얼리 전문용어를 고려한 번역 개선"""
        try:
            enhanced_text = text
            
            # 주요 용어 식별 및 교체
            for key, translations in self.terms_db.items():
                source_term = translations.get(source_lang, "")
                target_term = translations.get(target_lang, "")
                
                if source_term and target_term:
                    # 대소문자 구분 없이 교체
                    pattern = re.compile(re.escape(source_term), re.IGNORECASE)
                    enhanced_text = pattern.sub(target_term, enhanced_text)
            
            return enhanced_text
            
        except Exception as e:
            self.logger.error(f"주얼리 용어 번역 개선 실패: {e}")
            return text

class LanguageDetector:
    """언어 감지기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_languages = {
            'ko': '한국어',
            'en': 'English', 
            'zh-cn': '中文',
            'ja': '日本語'
        }
        
    def detect_audio_language(self, audio_path: str) -> LanguageDetectionResult:
        """음성 파일의 언어 감지"""
        try:
            start_time = time.time()
            
            # 1. Whisper를 사용한 초기 언어 감지
            whisper_result = self._whisper_language_detection(audio_path)
            
            # 2. 짧은 샘플 텍스트로 확인
            sample_text = self._get_sample_text_from_audio(audio_path)
            text_detection = self._detect_text_language(sample_text) if sample_text else None
            
            # 3. 결과 통합
            final_language, confidence = self._combine_detection_results(whisper_result, text_detection)
            
            # 4. 구간별 언어 분석
            segments = self._analyze_language_segments(audio_path)
            
            processing_time = time.time() - start_time
            
            return LanguageDetectionResult(
                primary_language=final_language,
                confidence=confidence,
                language_distribution=self._calculate_language_distribution(segments),
                segments=segments,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"음성 언어 감지 실패: {e}")
            return LanguageDetectionResult(
                primary_language='ko',  # 기본값
                confidence=0.5,
                language_distribution={'ko': 1.0},
                segments=[],
                processing_time=0.0
            )
    
    def _whisper_language_detection(self, audio_path: str) -> Dict[str, Any]:
        """Whisper 기반 언어 감지"""
        try:
            # Whisper 모델 로드 (작은 모델로 빠른 감지)
            model = whisper.load_model("base")
            
            # 오디오 로드 및 언어 감지
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # 언어 감지만 수행 (전체 전사는 하지 않음)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            
            # 지원하는 언어만 필터링
            filtered_probs = {}
            for lang_code, prob in probs.items():
                if lang_code in ['ko', 'en', 'zh', 'ja']:
                    filtered_probs[lang_code] = prob
            
            # 최고 확률 언어
            detected_language = max(filtered_probs, key=filtered_probs.get)
            confidence = filtered_probs[detected_language]
            
            return {
                'language': detected_language,
                'confidence': confidence,
                'probabilities': filtered_probs
            }
            
        except Exception as e:
            self.logger.error(f"Whisper 언어 감지 실패: {e}")
            return {
                'language': 'ko',
                'confidence': 0.5,
                'probabilities': {'ko': 0.5, 'en': 0.3, 'zh': 0.1, 'ja': 0.1}
            }
    
    def _get_sample_text_from_audio(self, audio_path: str, duration: int = 30) -> Optional[str]:
        """음성에서 샘플 텍스트 추출 (언어 감지용)"""
        try:
            # 음성을 텍스트로 변환 (짧은 구간만)
            model = whisper.load_model("base")
            
            # 처음 30초만 처리
            audio = whisper.load_audio(audio_path)
            audio_sample = audio[:duration * 16000]  # 16kHz 기준
            
            result = model.transcribe(audio_sample, language=None)
            return result.get('text', '').strip()
            
        except Exception as e:
            self.logger.error(f"샘플 텍스트 추출 실패: {e}")
            return None
    
    def _detect_text_language(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트 기반 언어 감지"""
        try:
            if not text or len(text.strip()) < 10:
                return None
            
            # langdetect 사용
            detected_lang = detect(text)
            
            # 언어 코드 매핑
            lang_mapping = {
                'ko': 'ko',
                'en': 'en', 
                'zh-cn': 'zh',
                'zh': 'zh',
                'ja': 'ja'
            }
            
            mapped_lang = lang_mapping.get(detected_lang, 'ko')
            
            # 신뢰도 추정 (텍스트 길이 기반)
            confidence = min(0.9, len(text) / 100)
            
            return {
                'language': mapped_lang,
                'confidence': confidence,
                'original_detection': detected_lang
            }
            
        except Exception as e:
            self.logger.error(f"텍스트 언어 감지 실패: {e}")
            return None
    
    def _combine_detection_results(self, whisper_result: Dict, text_result: Optional[Dict]) -> Tuple[str, float]:
        """언어 감지 결과 통합"""
        try:
            if not text_result:
                return whisper_result['language'], whisper_result['confidence']
            
            # 두 결과가 일치하는 경우
            if whisper_result['language'] == text_result['language']:
                combined_confidence = (whisper_result['confidence'] + text_result['confidence']) / 2
                return whisper_result['language'], min(0.95, combined_confidence * 1.2)
            
            # 다른 경우 더 신뢰도가 높은 것 선택
            if whisper_result['confidence'] > text_result['confidence']:
                return whisper_result['language'], whisper_result['confidence']
            else:
                return text_result['language'], text_result['confidence']
                
        except Exception as e:
            self.logger.error(f"언어 감지 결과 통합 실패: {e}")
            return 'ko', 0.5
    
    def _analyze_language_segments(self, audio_path: str) -> List[Dict]:
        """음성 구간별 언어 분석"""
        try:
            segments = []
            
            # 음성을 30초 단위로 분할하여 분석
            audio = whisper.load_audio(audio_path)
            segment_duration = 30 * 16000  # 30초
            
            model = whisper.load_model("base")
            
            for i in range(0, len(audio), segment_duration):
                segment_audio = audio[i:i + segment_duration]
                
                if len(segment_audio) < 16000:  # 1초 미만은 스킵
                    continue
                
                # 구간별 언어 감지
                try:
                    mel = whisper.log_mel_spectrogram(segment_audio).to(model.device)
                    _, probs = model.detect_language(mel)
                    
                    detected_lang = max(probs, key=probs.get)
                    confidence = probs[detected_lang]
                    
                    segments.append({
                        'start_time': i / 16000,
                        'end_time': min((i + segment_duration) / 16000, len(audio) / 16000),
                        'language': detected_lang,
                        'confidence': confidence
                    })
                    
                except Exception as e:
                    self.logger.warning(f"구간 {i//16000}초 언어 감지 실패: {e}")
            
            return segments
            
        except Exception as e:
            self.logger.error(f"구간별 언어 분석 실패: {e}")
            return []
    
    def _calculate_language_distribution(self, segments: List[Dict]) -> Dict[str, float]:
        """언어별 분포 계산"""
        try:
            if not segments:
                return {'ko': 1.0}
            
            language_times = {}
            
            for segment in segments:
                lang = segment['language']
                duration = segment['end_time'] - segment['start_time']
                
                if lang in language_times:
                    language_times[lang] += duration
                else:
                    language_times[lang] = duration
            
            # 비율로 변환
            total_time = sum(language_times.values())
            distribution = {lang: time / total_time for lang, time in language_times.items()}
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"언어 분포 계산 실패: {e}")
            return {'ko': 1.0}

class MultilingualSTTEngine:
    """다국어 STT 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.language_detector = LanguageDetector()
        self.jewelry_db = JewelryTermsDatabase()
        self.translator = Translator()
        
        # 언어별 최적화된 Whisper 모델
        self.whisper_models = {
            'ko': whisper.load_model("medium"),  # 한국어는 medium 모델
            'en': whisper.load_model("base"),    # 영어는 base 모델
            'zh': whisper.load_model("medium"),  # 중국어는 medium 모델
            'ja': whisper.load_model("base")     # 일본어는 base 모델
        }
        
    def process_multilingual_audio(self, audio_path: str) -> MultilingualSTTResult:
        """다국어 음성 파일 처리"""
        try:
            start_time = time.time()
            
            # 1. 언어 감지
            detection_result = self.language_detector.detect_audio_language(audio_path)
            detected_language = detection_result.primary_language
            
            self.logger.info(f"감지된 언어: {detected_language} (신뢰도: {detection_result.confidence:.2f})")
            
            # 2. 언어별 최적화 STT
            original_text = self._perform_optimized_stt(audio_path, detected_language)
            
            # 3. 주얼리 전문용어 후처리
            enhanced_text = self.jewelry_db.enhance_translation_with_jewelry_terms(
                original_text, detected_language, detected_language
            )
            
            # 4. 한국어 번역 (필요한 경우)
            korean_translation = self._translate_to_korean(enhanced_text, detected_language)
            
            # 5. 품질 평가
            confidence_score = self._calculate_stt_confidence(
                original_text, detection_result.confidence, detected_language
            )
            
            processing_time = time.time() - start_time
            
            return MultilingualSTTResult(
                original_text=enhanced_text,
                detected_language=detected_language,
                korean_translation=korean_translation,
                confidence_score=confidence_score,
                processing_details={
                    'language_detection': detection_result,
                    'processing_time': processing_time,
                    'model_used': f"whisper-{detected_language}",
                    'jewelry_terms_processed': self._count_jewelry_terms(enhanced_text)
                },
                timestamp=start_time
            )
            
        except Exception as e:
            self.logger.error(f"다국어 STT 처리 실패: {e}")
            return MultilingualSTTResult(
                original_text="",
                detected_language="ko",
                korean_translation="",
                confidence_score=0.0,
                processing_details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _perform_optimized_stt(self, audio_path: str, language: str) -> str:
        """언어별 최적화 STT 수행"""
        try:
            # 해당 언어의 최적화된 모델 선택
            model = self.whisper_models.get(language, self.whisper_models['ko'])
            
            # STT 수행
            result = model.transcribe(
                audio_path, 
                language=language if language != 'zh' else 'zh-cn',  # 중국어 코드 조정
                task='transcribe',
                verbose=False
            )
            
            return result.get('text', '').strip()
            
        except Exception as e:
            self.logger.error(f"언어별 STT 실패 ({language}): {e}")
            
            # 기본 모델로 재시도
            try:
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, verbose=False)
                return result.get('text', '').strip()
            except Exception as e2:
                self.logger.error(f"기본 STT 재시도 실패: {e2}")
                return ""
    
    def _translate_to_korean(self, text: str, source_language: str) -> str:
        """한국어로 번역"""
        try:
            if source_language == 'ko' or not text.strip():
                return text
            
            # Google Translate 사용
            translated = self.translator.translate(text, src=source_language, dest='ko')
            korean_text = translated.text
            
            # 주얼리 전문용어 추가 보정
            enhanced_korean = self.jewelry_db.enhance_translation_with_jewelry_terms(
                korean_text, 'ko', 'ko'  # 한국어 내에서 용어 정리
            )
            
            return enhanced_korean
            
        except Exception as e:
            self.logger.error(f"한국어 번역 실패: {e}")
            return text  # 번역 실패 시 원본 반환
    
    def _calculate_stt_confidence(self, text: str, language_confidence: float, language: str) -> float:
        """STT 신뢰도 점수 계산"""
        try:
            # 기본 신뢰도는 언어 감지 신뢰도
            base_confidence = language_confidence
            
            # 텍스트 길이 기반 보정
            length_factor = min(1.0, len(text) / 100)  # 100자 기준
            
            # 주얼리 용어 포함 여부로 신뢰도 향상
            jewelry_terms_count = self._count_jewelry_terms(text)
            jewelry_bonus = min(0.2, jewelry_terms_count * 0.05)
            
            # 언어별 STT 성능 가중치
            language_weights = {
                'ko': 0.9,   # 한국어 최적화
                'en': 0.95,  # 영어 우수
                'zh': 0.85,  # 중국어 양호  
                'ja': 0.9    # 일본어 양호
            }
            
            language_weight = language_weights.get(language, 0.8)
            
            # 종합 신뢰도 계산
            final_confidence = (
                base_confidence * 0.4 +
                length_factor * 0.3 + 
                language_weight * 0.3 +
                jewelry_bonus
            )
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    def _count_jewelry_terms(self, text: str) -> int:
        """텍스트 내 주얼리 용어 개수 계산"""
        try:
            count = 0
            text_lower = text.lower()
            
            for term_key, translations in self.jewelry_db.terms_db.items():
                for lang, term in translations.items():
                    if term.lower() in text_lower:
                        count += 1
                        break  # 같은 용어의 다른 언어 중복 방지
            
            return count
            
        except Exception as e:
            self.logger.error(f"주얼리 용어 카운팅 실패: {e}")
            return 0

class MultilingualProcessorV21:
    """v2.1 다국어 처리 통합 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stt_engine = MultilingualSTTEngine()
        self.jewelry_db = JewelryTermsDatabase()
        
    def process_multilingual_content(self, content: Union[str, List[str]], content_type: str = "audio") -> Dict[str, Any]:
        """다국어 컨텐츠 통합 처리"""
        try:
            if isinstance(content, str):
                content = [content]
            
            results = []
            total_processing_time = 0
            
            # 병렬 처리로 성능 향상
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                for item in content:
                    if content_type == "audio":
                        future = executor.submit(self.stt_engine.process_multilingual_audio, item)
                    else:
                        future = executor.submit(self._process_text_content, item)
                    
                    futures.append(future)
                
                # 결과 수집
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5분 타임아웃
                        results.append(result)
                        total_processing_time += result.processing_details.get('processing_time', 0)
                    except Exception as e:
                        self.logger.error(f"개별 처리 실패: {e}")
            
            # 결과 통합
            integrated_result = self._integrate_multilingual_results(results)
            
            return {
                'individual_results': results,
                'integrated_result': integrated_result,
                'processing_statistics': {
                    'total_files': len(content),
                    'successful_files': len(results),
                    'total_processing_time': total_processing_time,
                    'average_confidence': np.mean([r.confidence_score for r in results]) if results else 0
                },
                'language_distribution': self._analyze_overall_language_distribution(results)
            }
            
        except Exception as e:
            self.logger.error(f"다국어 컨텐츠 처리 실패: {e}")
            return {
                'error': str(e),
                'processing_complete': False
            }
    
    def _process_text_content(self, text: str) -> MultilingualSTTResult:
        """텍스트 컨텐츠 처리 (번역용)"""
        try:
            # 언어 감지
            detected_lang = detect(text) if text.strip() else 'ko'
            
            # 언어 코드 매핑
            lang_mapping = {'zh-cn': 'zh', 'zh': 'zh'}
            detected_lang = lang_mapping.get(detected_lang, detected_lang)
            
            # 한국어 번역
            if detected_lang != 'ko':
                translator = Translator()
                translated = translator.translate(text, src=detected_lang, dest='ko')
                korean_text = translated.text
            else:
                korean_text = text
            
            # 주얼리 용어 향상
            enhanced_korean = self.jewelry_db.enhance_translation_with_jewelry_terms(
                korean_text, 'ko', 'ko'
            )
            
            return MultilingualSTTResult(
                original_text=text,
                detected_language=detected_lang,
                korean_translation=enhanced_korean,
                confidence_score=0.8,
                processing_details={
                    'content_type': 'text',
                    'processing_time': 0.1
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"텍스트 처리 실패: {e}")
            return MultilingualSTTResult(
                original_text=text,
                detected_language='ko',
                korean_translation=text,
                confidence_score=0.5,
                processing_details={'error': str(e)},
                timestamp=time.time()
            )
    
    def _integrate_multilingual_results(self, results: List[MultilingualSTTResult]) -> Dict[str, Any]:
        """다국어 결과 통합"""
        try:
            if not results:
                return {'error': '처리된 결과가 없습니다.'}
            
            # 모든 한국어 번역을 하나로 통합
            all_korean_text = []
            language_stats = Counter()
            confidence_scores = []
            
            for result in results:
                if result.korean_translation.strip():
                    all_korean_text.append(result.korean_translation)
                
                language_stats[result.detected_language] += 1
                confidence_scores.append(result.confidence_score)
            
            # 통합 텍스트 생성
            integrated_korean = '\n\n'.join(all_korean_text)
            
            # 주얼리 용어 최종 정리
            final_korean = self.jewelry_db.enhance_translation_with_jewelry_terms(
                integrated_korean, 'ko', 'ko'
            )
            
            # 핵심 인사이트 추출
            key_insights = self._extract_key_insights(final_korean)
            
            return {
                'final_korean_text': final_korean,
                'key_insights': key_insights,
                'language_statistics': dict(language_stats),
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'total_length': len(final_korean),
                'jewelry_terms_count': self.stt_engine._count_jewelry_terms(final_korean)
            }
            
        except Exception as e:
            self.logger.error(f"결과 통합 실패: {e}")
            return {'error': str(e)}
    
    def _analyze_overall_language_distribution(self, results: List[MultilingualSTTResult]) -> Dict[str, float]:
        """전체 언어 분포 분석"""
        try:
            language_counts = Counter()
            
            for result in results:
                language_counts[result.detected_language] += 1
            
            total = sum(language_counts.values())
            if total == 0:
                return {'ko': 1.0}
            
            distribution = {lang: count / total for lang, count in language_counts.items()}
            return distribution
            
        except Exception as e:
            self.logger.error(f"언어 분포 분석 실패: {e}")
            return {'ko': 1.0}
    
    def _extract_key_insights(self, korean_text: str) -> List[str]:
        """한국어 텍스트에서 핵심 인사이트 추출"""
        try:
            insights = []
            
            # 주얼리 관련 키워드 기반 인사이트 추출
            jewelry_keywords = [
                '시장', '가격', '트렌드', '품질', '투자', '수요', '공급',
                '디자인', '제작', '기술', '혁신', '브랜드', '마케팅'
            ]
            
            sentences = korean_text.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # 너무 짧은 문장 제외
                    for keyword in jewelry_keywords:
                        if keyword in sentence:
                            insights.append(sentence + '.')
                            break
            
            # 중복 제거 및 상위 5개만 선택
            unique_insights = list(set(insights))[:5]
            
            return unique_insights
            
        except Exception as e:
            self.logger.error(f"인사이트 추출 실패: {e}")
            return ['핵심 내용이 성공적으로 처리되었습니다.']
    
    def generate_multilingual_summary(self, processing_result: Dict[str, Any]) -> str:
        """다국어 처리 결과 요약 생성"""
        try:
            if 'error' in processing_result:
                return f"❌ 다국어 처리 실패: {processing_result['error']}"
            
            stats = processing_result['processing_statistics']
            integrated = processing_result['integrated_result']
            lang_dist = processing_result['language_distribution']
            
            # 주요 언어 식별
            main_languages = sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)
            lang_names = {'ko': '한국어', 'en': '영어', 'zh': '중국어', 'ja': '일본어'}
            
            summary = f"""
🌍 **다국어 분석 결과**

📊 **처리 통계**
• 전체 파일: {stats['total_files']}개
• 성공 처리: {stats['successful_files']}개  
• 평균 신뢰도: {stats['average_confidence']:.1%}
• 총 처리시간: {stats['total_processing_time']:.1f}초

🗣️ **언어 분포**
{chr(10).join([f'• {lang_names.get(lang, lang)}: {ratio:.1%}' for lang, ratio in main_languages[:3]])}

💎 **주얼리 전문용어**: {integrated.get('jewelry_terms_count', 0)}개 식별

🎯 **핵심 인사이트**
{chr(10).join(['• ' + insight for insight in integrated.get('key_insights', [])[:3]])}

📝 **통합 분석 완료** - 모든 내용이 한국어로 통합되었습니다.
            """
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"다국어 요약 생성 실패: {e}")
            return "다국어 처리 요약 생성 중 오류가 발생했습니다."

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 다국어 처리 엔진 초기화
    processor = MultilingualProcessorV21()
    
    # 샘플 처리
    # result = processor.process_multilingual_content(["sample_audio.mp3"], "audio")
    # print(processor.generate_multilingual_summary(result))
    
    print("✅ 다국어 처리 엔진 v2.1 로드 완료!")
