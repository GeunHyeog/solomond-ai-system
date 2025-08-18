# -*- coding: utf-8 -*-
"""
향상된 다국어 스크립트 처리기 - 언어 감지 및 처리 개선
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
from utils.logger import get_logger

class EnhancedMultilingualProcessor:
    """향상된 다국어 스크립트 처리기"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # 향상된 언어별 정규식 패턴
        self.language_patterns = {
            'korean': {
                'hangul': re.compile(r'[가-힣]+'),
                'hangul_jamo': re.compile(r'[ㄱ-ㅎㅏ-ㅣ]+'),
                'korean_numbers': re.compile(r'[일이삼사오육칠팔구십백천만억조]+'),
                'korean_particles': re.compile(r'[은는이가을를의에서와과도만까지부터조차마저라도]+'),
                'korean_endings': re.compile(r'(습니다|ㅂ니다|어요|아요|지요|네요|군요|다고|라고)'),
                'korean_honorifics': re.compile(r'(선생님|씨|님|께서|드리|받으시|하시)')
            },
            'english': {
                'basic': re.compile(r'[a-zA-Z]+'),
                'words': re.compile(r'\b[a-zA-Z]{2,}\b'),
                'common_words': re.compile(r'\b(the|and|or|but|in|on|at|to|for|of|with|by|from|about|into|through|during|before|after|above|below|between|among|against)\b', re.IGNORECASE),
                'contractions': re.compile(r"\b\w+'\w+\b"),
                'english_patterns': re.compile(r'\b(is|are|was|were|have|has|had|do|does|did|will|would|can|could|should|must)\b', re.IGNORECASE),
                'articles': re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
            },
            'chinese': {
                'simplified': re.compile(r'[\u4e00-\u9fff]+'),
                'traditional': re.compile(r'[\u4e00-\u9fff\uf900-\ufaff]+'),
                'chinese_punctuation': re.compile(r'[，。、；：！？（）【】《》""''…—]')
            },
            'japanese': {
                'hiragana': re.compile(r'[ひらがな\u3040-\u309f]+'),
                'katakana': re.compile(r'[カタカナ\u30a0-\u30ff]+'),
                'kanji': re.compile(r'[漢字\u4e00-\u9faf]+'),
                'japanese_particles': re.compile(r'[はがをにへとでからまでもやより]')
            },
            'numbers': {
                'arabic': re.compile(r'\d+'),
                'roman': re.compile(r'\b[IVXLCDM]+\b'),
                'korean': re.compile(r'[일이삼사오육칠팔구십백천만억]+')
            },
            'punctuation': {
                'western': re.compile(r'[.,;:!?()[\]{}"\'-]'),
                'korean': re.compile(r'[。、，．；：！？（）［］｛｝"''""]'),
                'special': re.compile(r'[~@#$%^&*+=<>/\\|`_]'),
                'quotes': re.compile(r'["""''\'`]')
            }
        }
        
        # 언어별 고빈도 단어 및 패턴
        self.language_signatures = {
            'korean': {
                'high_frequency': ['그', '이', '저', '제', '내', '우리', '여러분', '사람', '시간', '일', '년', '월', '일', '때', '곳', '것', '수', '말', '생각', '안녕', '어서', '오세요', '감사', '합니다', '죄송', '미안', '괜찮', '좋아', '싫어'],
                'particles': ['은', '는', '이', '가', '을', '를', '의', '에', '서', '와', '과', '도', '만', '까지', '부터', '조차', '마저', '라도'],
                'endings': ['다', '요', '니다', '습니다', '어요', '아요', '지요', '네요', '군요', '죠', '거든요', '어서', '아서'],
                'functional': ['하다', '되다', '있다', '없다', '아니다', '같다', '다르다', '크다', '작다', '좋다', '나쁘다']
            },
            'english': {
                'high_frequency': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 'hi', 'hello', 'bye', 'goodbye', 'thanks', 'thank', 'please'],
                'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'where', 'when', 'how', 'why'],
                'verbs': ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'must'],
                'common_words': ['time', 'people', 'way', 'work', 'make', 'get', 'go', 'come', 'see', 'know', 'think', 'take', 'use', 'find', 'give']
            }
        }
        
        # 텍스트 품질 평가 기준
        self.quality_criteria = {
            'min_length': 3,
            'max_special_char_ratio': 0.3,
            'min_word_length': 1,
            'max_word_length': 50,
            'max_repeated_char_threshold': 4
        }
    
    def enhanced_language_detection(self, text: str) -> Dict[str, Any]:
        """향상된 다국어 언어 감지"""
        
        if not text or len(text.strip()) < self.quality_criteria['min_length']:
            return self._create_default_detection_result()
        
        # 텍스트 전처리
        cleaned_text = self._preprocess_text(text)
        
        # 문자체계 분석
        script_analysis = self._analyze_scripts(cleaned_text)
        
        # 언어별 시그니처 매칭
        signature_analysis = self._analyze_language_signatures(cleaned_text)
        
        # 구조적 패턴 분석
        structural_analysis = self._analyze_structural_patterns(cleaned_text)
        
        # 종합 점수 계산
        language_scores = self._calculate_language_scores(
            script_analysis, signature_analysis, structural_analysis, cleaned_text
        )
        
        # 최종 언어 결정
        detected_language, confidence = self._determine_final_language(language_scores)
        
        # 번역 필요성 판단
        needs_translation = self._determine_translation_need(detected_language, language_scores)
        
        # 추가 메타데이터
        text_statistics = self._calculate_advanced_statistics(text)
        quality_score = self._calculate_text_quality_score(text)
        
        return {
            'language': detected_language,
            'confidence': confidence,
            'needs_translation': needs_translation,
            'language_scores': language_scores,
            'script_analysis': script_analysis,
            'signature_analysis': signature_analysis,
            'structural_analysis': structural_analysis,
            'text_statistics': text_statistics,
            'quality_score': quality_score,
            'processing_metadata': {
                'original_length': len(text),
                'cleaned_length': len(cleaned_text),
                'processing_version': '2.0.0'
            }
        }
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 제어 문자 제거
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t ')
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _analyze_scripts(self, text: str) -> Dict[str, Any]:
        """문자체계 분석"""
        
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return {'ratios': {}, 'counts': {}, 'dominant_script': 'unknown'}
        
        # 각 문자체계별 매칭
        script_counts = {}
        
        # 한국어
        hangul_count = len(self.language_patterns['korean']['hangul'].findall(text))
        hangul_jamo_count = len(self.language_patterns['korean']['hangul_jamo'].findall(text))
        korean_total = hangul_count + hangul_jamo_count
        
        # 영어
        english_count = len(self.language_patterns['english']['basic'].findall(text))
        
        # 중국어
        chinese_count = len(self.language_patterns['chinese']['simplified'].findall(text))
        
        # 일본어
        hiragana_count = len(self.language_patterns['japanese']['hiragana'].findall(text))
        katakana_count = len(self.language_patterns['japanese']['katakana'].findall(text))
        kanji_count = len(self.language_patterns['japanese']['kanji'].findall(text))
        japanese_total = hiragana_count + katakana_count + kanji_count
        
        # 숫자
        number_count = len(self.language_patterns['numbers']['arabic'].findall(text))
        
        script_counts = {
            'korean': korean_total,
            'english': english_count,
            'chinese': chinese_count,
            'japanese': japanese_total,
            'numbers': number_count
        }
        
        # 비율 계산
        script_ratios = {k: v / total_chars for k, v in script_counts.items()}
        
        # 주요 문자체계 결정
        dominant_script = max(script_ratios.keys(), key=script_ratios.get) if script_ratios else 'unknown'
        
        return {
            'ratios': script_ratios,
            'counts': script_counts,
            'dominant_script': dominant_script,
            'total_characters': total_chars
        }
    
    def _analyze_language_signatures(self, text: str) -> Dict[str, Any]:
        """언어별 시그니처 분석"""
        
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        signature_scores = {}
        
        for language, signatures in self.language_signatures.items():
            language_score = 0
            matches_by_category = {}
            
            for category, word_list in signatures.items():
                matches = sum(1 for word in word_list if word in text_lower)
                matches_by_category[category] = matches
                
                # 카테고리별 가중치 적용
                if category == 'high_frequency':
                    language_score += matches * 2
                elif category in ['particles', 'endings', 'pronouns', 'verbs']:
                    language_score += matches * 3
                else:
                    language_score += matches
            
            # 정규화 (총 단어 수 대비)
            normalized_score = language_score / max(total_words, 1) if total_words > 0 else 0
            
            signature_scores[language] = {
                'raw_score': language_score,
                'normalized_score': normalized_score,
                'matches_by_category': matches_by_category
            }
        
        return {
            'signature_scores': signature_scores,
            'total_words_analyzed': total_words
        }
    
    def _analyze_structural_patterns(self, text: str) -> Dict[str, Any]:
        """구조적 패턴 분석"""
        
        patterns = {}
        
        # 한국어 구조적 특징
        korean_endings = len(self.language_patterns['korean']['korean_endings'].findall(text))
        korean_particles = len(self.language_patterns['korean']['korean_particles'].findall(text))
        korean_honorifics = len(self.language_patterns['korean']['korean_honorifics'].findall(text))
        
        patterns['korean_structural'] = {
            'endings': korean_endings,
            'particles': korean_particles,
            'honorifics': korean_honorifics,
            'total_score': korean_endings * 2 + korean_particles * 3 + korean_honorifics
        }
        
        # 영어 구조적 특징
        english_articles = len(self.language_patterns['english']['articles'].findall(text))
        english_contractions = len(self.language_patterns['english']['contractions'].findall(text))
        english_patterns = len(self.language_patterns['english']['english_patterns'].findall(text))
        
        patterns['english_structural'] = {
            'articles': english_articles,
            'contractions': english_contractions,
            'verb_patterns': english_patterns,
            'total_score': english_articles * 2 + english_contractions + english_patterns * 2
        }
        
        # 문장 구조 분석
        sentences = re.split(r'[.!?。！？]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
        
        patterns['sentence_structure'] = {
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_sentence_length': avg_sentence_length,
            'complexity_score': min(avg_sentence_length / 10, 1.0)  # 문장 복잡도
        }
        
        return patterns
    
    def _calculate_language_scores(self, script_analysis: Dict, signature_analysis: Dict, structural_analysis: Dict, cleaned_text: str) -> Dict[str, float]:
        """종합 언어 점수 계산"""
        
        language_scores = {'korean': 0.0, 'english': 0.0, 'mixed': 0.0, 'unknown': 0.0}
        
        # 1. 문자체계 점수 (40% 가중치)
        script_weight = 0.4
        for lang in ['korean', 'english']:
            if lang in script_analysis['ratios']:
                language_scores[lang] += script_analysis['ratios'][lang] * script_weight
        
        # 2. 시그니처 점수 (35% 가중치)
        signature_weight = 0.35
        for lang in ['korean', 'english']:
            if lang in signature_analysis['signature_scores']:
                normalized_score = signature_analysis['signature_scores'][lang]['normalized_score']
                language_scores[lang] += normalized_score * signature_weight
        
        # 3. 구조적 패턴 점수 (25% 가중치)
        structural_weight = 0.25
        
        # 한국어 구조적 점수
        korean_structural_score = structural_analysis['korean_structural']['total_score']
        max_korean_structural = max(korean_structural_score, 10)  # 정규화를 위한 최대값
        language_scores['korean'] += (korean_structural_score / max_korean_structural) * structural_weight
        
        # 영어 구조적 점수
        english_structural_score = structural_analysis['english_structural']['total_score']
        max_english_structural = max(english_structural_score, 10)  # 정규화를 위한 최대값
        language_scores['english'] += (english_structural_score / max_english_structural) * structural_weight
        
        # 혼합 언어 점수 계산 (개선된 로직)
        korean_score = language_scores['korean']
        english_score = language_scores['english']
        
        # 두 언어가 모두 일정 이상 나타나면 혼합 언어로 판정
        # 짧은 텍스트의 경우 더 낮은 임계값 사용
        korean_threshold = 0.05 if len(cleaned_text) < 20 else 0.1
        english_threshold = 0.05 if len(cleaned_text) < 20 else 0.1
        
        if korean_score > korean_threshold and english_score > english_threshold:
            # 혼합 정도에 따라 점수 계산
            balance_factor = 1 - abs(korean_score - english_score)  # 균형도
            mixed_base_score = (korean_score + english_score) / 2
            # 짧은 텍스트의 경우 혼합 점수에 더 큰 보너스 적용
            if len(cleaned_text) < 20:
                # 짧은 텍스트에서는 혼합 언어가 더 가능성이 높음
                length_bonus = 0.15
                balance_boost = 0.5 if balance_factor > 0.3 else 0.2
                language_scores['mixed'] = mixed_base_score * (1 + balance_factor * 0.4) + length_bonus + balance_boost
            else:
                language_scores['mixed'] = mixed_base_score * (1 + balance_factor * 0.3)
        
        # 알 수 없는 언어 점수
        if max(language_scores['korean'], language_scores['english'], language_scores['mixed']) < 0.2:
            language_scores['unknown'] = 0.3
        
        return language_scores
    
    def _determine_final_language(self, language_scores: Dict[str, float]) -> Tuple[str, float]:
        """최종 언어 및 신뢰도 결정"""
        
        # 최고 점수 언어 선택
        detected_language = max(language_scores.keys(), key=language_scores.get)
        confidence = language_scores[detected_language]
        
        # 신뢰도 조정
        if confidence < 0.3:
            detected_language = 'unknown'
            confidence = 0.5
        elif confidence > 1.0:
            confidence = 1.0
        
        return detected_language, round(confidence, 3)
    
    def _determine_translation_need(self, language: str, scores: Dict[str, float]) -> bool:
        """번역 필요성 판단"""
        
        if language in ['english', 'mixed']:
            return True
        elif language == 'korean':
            return False
        else:  # unknown or other languages
            # 영어 점수가 한국어 점수보다 높으면 번역 필요
            return scores.get('english', 0) > scores.get('korean', 0)
    
    def _calculate_advanced_statistics(self, text: str) -> Dict[str, Any]:
        """고급 텍스트 통계 계산"""
        
        words = text.split()
        sentences = re.split(r'[.!?。！？]+', text)
        
        # 기본 통계
        stats = {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len([s for s in sentences if s.strip()]),
            'average_word_length': round(sum(len(word) for word in words) / max(len(words), 1), 2),
            'average_sentence_length': round(sum(len(s.split()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1), 2)
        }
        
        # 문자 유형별 분석
        char_analysis = {
            'letters': len(re.findall(r'[a-zA-Z가-힣]', text)),
            'digits': len(re.findall(r'\d', text)),
            'spaces': len(re.findall(r'\s', text)),
            'punctuation': len(re.findall(r'[^\w\s]', text)),
            'special_chars': len(re.findall(r'[^\w\s가-힣]', text))
        }
        
        stats['character_analysis'] = char_analysis
        stats['text_density'] = round(char_analysis['letters'] / max(len(text), 1), 3)
        
        # 언어별 특수 통계
        stats['korean_syllable_count'] = len(re.findall(r'[가-힣]', text))
        stats['english_word_count'] = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        return stats
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """텍스트 품질 점수 계산"""
        
        if not text:
            return 0.0
        
        score = 1.0
        
        # 길이 검사
        length = len(text)
        if length < self.quality_criteria['min_length']:
            score *= 0.3
        elif length < 10:
            score *= 0.6
        elif length > 2000:
            score *= 0.9
        
        # 특수문자 비율 검사
        special_chars = len(re.findall(r'[^\w\s가-힣]', text))
        special_ratio = special_chars / length
        if special_ratio > self.quality_criteria['max_special_char_ratio']:
            score *= 0.7
        
        # 연속된 문자 검사
        repeated_pattern = re.compile(r'(.)\1{' + str(self.quality_criteria['max_repeated_char_threshold']) + ',}')
        if repeated_pattern.search(text):
            score *= 0.8
        
        # 단어 길이 분포
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < self.quality_criteria['min_word_length'] or avg_word_length > self.quality_criteria['max_word_length']:
                score *= 0.9
        
        # 공백 패턴
        if re.search(r'\s{3,}', text):  # 연속된 3개 이상의 공백
            score *= 0.95
        
        return round(min(score, 1.0), 3)
    
    def _create_default_detection_result(self) -> Dict[str, Any]:
        """기본 감지 결과 생성"""
        
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'needs_translation': False,
            'language_scores': {'korean': 0.0, 'english': 0.0, 'mixed': 0.0, 'unknown': 1.0},
            'script_analysis': {'ratios': {}, 'counts': {}, 'dominant_script': 'unknown'},
            'signature_analysis': {'signature_scores': {}, 'total_words_analyzed': 0},
            'structural_analysis': {},
            'text_statistics': {'total_characters': 0, 'total_words': 0, 'total_sentences': 0},
            'quality_score': 0.0,
            'processing_metadata': {
                'original_length': 0,
                'cleaned_length': 0,
                'processing_version': '2.0.0'
            }
        }
    
    def process_multilingual_segments(self, text: str) -> Dict[str, Any]:
        """다국어 텍스트 세그먼트 처리"""
        
        self.logger.info("다국어 세그먼트 처리 시작")
        
        # 전체 텍스트 언어 감지
        overall_detection = self.enhanced_language_detection(text)
        
        # 문장별 세분화
        sentences = re.split(r'[.!?。！？]+', text)
        segments = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 각 문장 개별 분석
            segment_detection = self.enhanced_language_detection(sentence)
            
            segment = {
                'segment_id': i + 1,
                'text': sentence,
                'language': segment_detection['language'],
                'confidence': segment_detection['confidence'],
                'needs_translation': segment_detection['needs_translation'],
                'quality_score': segment_detection['quality_score'],
                'word_count': len(sentence.split()),
                'char_count': len(sentence)
            }
            
            segments.append(segment)
        
        # 세그먼트 통계
        segment_stats = self._calculate_segment_statistics(segments)
        
        # 번역 권장사항
        translation_recommendations = self._generate_translation_recommendations(overall_detection, segments)
        
        result = {
            'overall_detection': overall_detection,
            'segments': segments,
            'segment_statistics': segment_stats,
            'translation_recommendations': translation_recommendations,
            'processing_metadata': {
                'total_segments': len(segments),
                'processing_timestamp': None,  # 실제 구현시 timestamp 추가
                'processor_version': '2.0.0'
            }
        }
        
        self.logger.info(f"세그먼트 처리 완료: {len(segments)}개 세그먼트")
        return result
    
    def _calculate_segment_statistics(self, segments: List[Dict]) -> Dict[str, Any]:
        """세그먼트 통계 계산"""
        
        if not segments:
            return {}
        
        # 언어별 세그먼트 분포
        language_distribution = {}
        for segment in segments:
            lang = segment['language']
            language_distribution[lang] = language_distribution.get(lang, 0) + 1
        
        # 번역 필요 세그먼트
        translation_needed = sum(1 for seg in segments if seg['needs_translation'])
        
        # 품질 점수 통계
        quality_scores = [seg['quality_score'] for seg in segments]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # 길이 통계
        word_counts = [seg['word_count'] for seg in segments]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        
        return {
            'language_distribution': language_distribution,
            'translation_needed_count': translation_needed,
            'translation_needed_ratio': translation_needed / len(segments),
            'average_quality_score': round(avg_quality, 3),
            'average_words_per_segment': round(avg_words, 2),
            'total_words': sum(word_counts),
            'quality_distribution': {
                'high': sum(1 for score in quality_scores if score >= 0.8),
                'medium': sum(1 for score in quality_scores if 0.5 <= score < 0.8),
                'low': sum(1 for score in quality_scores if score < 0.5)
            }
        }
    
    def _generate_translation_recommendations(self, overall_detection: Dict, segments: List[Dict]) -> Dict[str, Any]:
        """번역 권장사항 생성"""
        
        # 번역 필요 세그먼트 식별
        translation_segments = [seg for seg in segments if seg['needs_translation']]
        
        # 우선순위 계산
        if len(translation_segments) == 0:
            priority = 'none'
        elif len(translation_segments) / len(segments) > 0.7:
            priority = 'high'
        elif len(translation_segments) / len(segments) > 0.3:
            priority = 'medium'
        else:
            priority = 'low'
        
        # 권장 번역 엔진
        total_chars = sum(len(seg['text']) for seg in translation_segments)
        if total_chars > 1000:
            recommended_engine = 'professional'  # 전문 번역 서비스 권장
        elif total_chars > 300:
            recommended_engine = 'google'
        else:
            recommended_engine = 'dictionary'
        
        return {
            'translation_priority': priority,
            'segments_to_translate': len(translation_segments),
            'total_translation_chars': total_chars,
            'recommended_engine': recommended_engine,
            'batch_processing': len(translation_segments) > 5,
            'estimated_time_seconds': len(translation_segments) * 2,  # 세그먼트당 2초 추정
            'quality_threshold_met': all(seg['quality_score'] > 0.5 for seg in translation_segments)
        }

# 전역 인스턴스
_global_enhanced_multilingual_processor = None

def get_enhanced_multilingual_processor():
    """전역 향상된 다국어 처리기 인스턴스 반환"""
    global _global_enhanced_multilingual_processor
    if _global_enhanced_multilingual_processor is None:
        _global_enhanced_multilingual_processor = EnhancedMultilingualProcessor()
    return _global_enhanced_multilingual_processor

def enhanced_language_detection(text: str) -> Dict[str, Any]:
    """편의 함수: 향상된 언어 감지"""
    return get_enhanced_multilingual_processor().enhanced_language_detection(text)

def process_multilingual_segments(text: str) -> Dict[str, Any]:
    """편의 함수: 다국어 세그먼트 처리"""
    return get_enhanced_multilingual_processor().process_multilingual_segments(text)