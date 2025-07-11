#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1 - 다국어 처리 엔진
자동 언어 감지 + 언어별 STT 최적화 + 한국어 통합 분석

작성자: 전근혁 (솔로몬드 대표)
생성일: 2025.07.11
목적: 현장에서 다국어 혼용 상황 완벽 처리
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDetector:
    """언어 자동 감지기"""
    
    def __init__(self):
        # 주얼리 업계에서 자주 사용되는 언어별 키워드
        self.language_keywords = {
            'korean': {
                'words': ['다이아몬드', '반지', '목걸이', '귀걸이', '팔찌', '보석', '캐럿', '금', '은', '플래티넘',
                         '가격', '주문', '제작', '디자인', '감정서', '품질', '등급', '무게', '크기', '색상'],
                'patterns': [r'[가-힣]', r'원$', r'개$', r'번째']
            },
            'english': {
                'words': ['diamond', 'ring', 'necklace', 'earring', 'bracelet', 'jewelry', 'carat', 'gold', 
                         'silver', 'platinum', 'price', 'order', 'design', 'certificate', 'quality', 'grade'],
                'patterns': [r'\b[A-Za-z]+\b', r'\$\d+', r'\d+ct\b', r'\bVVS\d?\b', r'\bGIA\b']
            },
            'chinese': {
                'words': ['钻石', '戒指', '项链', '耳环', '手镯', '珠宝', '克拉', '黄金', '白银', '铂金',
                         '价格', '订单', '设计', '证书', '质量', '等级'],
                'patterns': [r'[\u4e00-\u9fff]', r'元$', r'克拉']
            },
            'japanese': {
                'words': ['ダイヤモンド', 'リング', 'ネックレス', 'ピアス', 'ブレスレット', 'ジュエリー', 
                         'カラット', 'ゴールド', 'シルバー', 'プラチナ', '価格', '注文', 'デザイン'],
                'patterns': [r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', r'円$', r'カラット']
            }
        }
        
        # 각 언어별 가중치
        self.language_weights = {
            'korean': 1.2,    # 한국어 우대 (최종 분석언어)
            'english': 1.0,   # 국제 표준
            'chinese': 0.9,   # 중국 시장
            'japanese': 0.8   # 일본 시장
        }
    
    def detect_language(self, text: str, confidence_threshold: float = 0.6) -> Dict:
        """텍스트에서 언어 감지"""
        if not text or not text.strip():
            return {"primary_language": "unknown", "confidence": 0.0, "language_distribution": {}}
        
        # 언어별 점수 계산
        language_scores = {}
        
        for lang, config in self.language_keywords.items():
            score = 0.0
            
            # 키워드 매칭
            for keyword in config['words']:
                if keyword.lower() in text.lower():
                    score += 2.0
            
            # 패턴 매칭
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 0.5
            
            # 가중치 적용
            score *= self.language_weights.get(lang, 1.0)
            
            # 텍스트 길이 대비 정규화
            language_scores[lang] = score / max(len(text.split()), 1)
        
        # 총 점수로 정규화하여 분포 계산
        total_score = sum(language_scores.values())
        if total_score == 0:
            return {"primary_language": "unknown", "confidence": 0.0, "language_distribution": {}}
        
        language_distribution = {lang: score/total_score for lang, score in language_scores.items()}
        
        # 주요 언어 결정
        primary_language = max(language_distribution, key=language_distribution.get)
        confidence = language_distribution[primary_language]
        
        result = {
            "primary_language": primary_language,
            "confidence": confidence,
            "language_distribution": language_distribution,
            "is_confident": confidence >= confidence_threshold,
            "is_multilingual": len([lang for lang, score in language_distribution.items() if score > 0.1]) > 1
        }
        
        logger.info(f"🌍 언어 감지 완료: {primary_language} ({confidence:.1%})")
        return result
    
    def detect_mixed_languages(self, segments: List[str]) -> List[Dict]:
        """여러 텍스트 세그먼트에서 언어 감지"""
        results = []
        
        for i, segment in enumerate(segments):
            detection = self.detect_language(segment)
            detection['segment_id'] = i
            detection['text'] = segment
            results.append(detection)
        
        return results


class JewelryTermTranslator:
    """주얼리 전문용어 번역기"""
    
    def __init__(self):
        # 주얼리 전문용어 다국어 사전
        self.jewelry_dictionary = {
            # 보석 종류
            'diamond': {'ko': '다이아몬드', 'en': 'diamond', 'zh': '钻石', 'ja': 'ダイヤモンド'},
            'ruby': {'ko': '루비', 'en': 'ruby', 'zh': '红宝石', 'ja': 'ルビー'},
            'sapphire': {'ko': '사파이어', 'en': 'sapphire', 'zh': '蓝宝石', 'ja': 'サファイア'},
            'emerald': {'ko': '에메랄드', 'en': 'emerald', 'zh': '祖母绿', 'ja': 'エメラルド'},
            'pearl': {'ko': '진주', 'en': 'pearl', 'zh': '珍珠', 'ja': '真珠'},
            
            # 품질 등급
            'carat': {'ko': '캐럿', 'en': 'carat', 'zh': '克拉', 'ja': 'カラット'},
            'clarity': {'ko': '투명도', 'en': 'clarity', 'zh': '净度', 'ja': 'クラリティ'},
            'color': {'ko': '색상', 'en': 'color', 'zh': '颜色', 'ja': 'カラー'},
            'cut': {'ko': '컷', 'en': 'cut', 'zh': '切工', 'ja': 'カット'},
            
            # 금속 종류
            'gold': {'ko': '금', 'en': 'gold', 'zh': '黄金', 'ja': 'ゴールド'},
            'silver': {'ko': '은', 'en': 'silver', 'zh': '银', 'ja': 'シルバー'},
            'platinum': {'ko': '플래티넘', 'en': 'platinum', 'zh': '铂金', 'ja': 'プラチナ'},
            
            # 제품 종류
            'ring': {'ko': '반지', 'en': 'ring', 'zh': '戒指', 'ja': 'リング'},
            'necklace': {'ko': '목걸이', 'en': 'necklace', 'zh': '项链', 'ja': 'ネックレス'},
            'earring': {'ko': '귀걸이', 'en': 'earring', 'zh': '耳环', 'ja': 'ピアス'},
            'bracelet': {'ko': '팔찌', 'en': 'bracelet', 'zh': '手镯', 'ja': 'ブレスレット'},
            
            # 비즈니스 용어
            'price': {'ko': '가격', 'en': 'price', 'zh': '价格', 'ja': '価格'},
            'quality': {'ko': '품질', 'en': 'quality', 'zh': '质量', 'ja': '品質'},
            'certificate': {'ko': '감정서', 'en': 'certificate', 'zh': '证书', 'ja': '鑑定書'},
            'wholesale': {'ko': '도매', 'en': 'wholesale', 'zh': '批发', 'ja': '卸売'},
            'retail': {'ko': '소매', 'en': 'retail', 'zh': '零售', 'ja': '小売'}
        }
        
        # 역방향 검색을 위한 인덱스
        self.reverse_index = {}
        for term, translations in self.jewelry_dictionary.items():
            for lang, translation in translations.items():
                if translation.lower() not in self.reverse_index:
                    self.reverse_index[translation.lower()] = []
                self.reverse_index[translation.lower()].append((term, lang))
    
    def translate_jewelry_terms(self, text: str, target_language: str = 'ko') -> str:
        """주얼리 전문용어를 한국어로 번역"""
        translated_text = text
        
        # 발견된 용어들 추적
        found_terms = []
        
        for term, translations in self.jewelry_dictionary.items():
            for source_lang, source_term in translations.items():
                if source_lang != target_language and source_term.lower() in text.lower():
                    target_term = translations.get(target_language, source_term)
                    
                    # 대소문자 구분 없이 교체
                    pattern = re.compile(re.escape(source_term), re.IGNORECASE)
                    translated_text = pattern.sub(target_term, translated_text)
                    
                    found_terms.append({
                        'original': source_term,
                        'translated': target_term,
                        'source_language': source_lang,
                        'target_language': target_language
                    })
        
        return translated_text, found_terms
    
    def extract_jewelry_terms(self, text: str) -> List[Dict]:
        """텍스트에서 주얼리 전문용어 추출"""
        found_terms = []
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            if word in self.reverse_index:
                for term, lang in self.reverse_index[word]:
                    found_terms.append({
                        'term': word,
                        'standard_term': term,
                        'language': lang,
                        'translations': self.jewelry_dictionary[term]
                    })
        
        return found_terms


class MultilingualProcessor:
    """통합 다국어 처리기"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.term_translator = JewelryTermTranslator()
        
        # STT 모델별 언어 지원 정보
        self.stt_models = {
            'whisper-korean': {'languages': ['korean'], 'accuracy': 0.95, 'specialty': 'korean_native'},
            'whisper-multilingual': {'languages': ['korean', 'english', 'chinese', 'japanese'], 'accuracy': 0.85, 'specialty': 'multilingual'},
            'whisper-english': {'languages': ['english'], 'accuracy': 0.92, 'specialty': 'english_native'}
        }
    
    def process_multilingual_content(self, content: str, content_type: str = 'transcript') -> Dict:
        """다국어 컨텐츠 통합 처리"""
        
        # 1. 언어 감지
        language_detection = self.language_detector.detect_language(content)
        
        # 2. 주얼리 전문용어 번역 (모든 언어를 한국어로)
        translated_content, translated_terms = self.term_translator.translate_jewelry_terms(
            content, target_language='ko'
        )
        
        # 3. 전문용어 추출
        extracted_terms = self.term_translator.extract_jewelry_terms(content)
        
        # 4. 최적 STT 모델 추천
        recommended_model = self._recommend_stt_model(language_detection)
        
        # 5. 번역 품질 평가
        translation_quality = self._evaluate_translation_quality(content, translated_content, translated_terms)
        
        result = {
            'original_content': content,
            'translated_content': translated_content,
            'language_detection': language_detection,
            'translated_terms': translated_terms,
            'extracted_terms': extracted_terms,
            'recommended_stt_model': recommended_model,
            'translation_quality': translation_quality,
            'processing_timestamp': datetime.now().isoformat(),
            'korean_summary': self._generate_korean_summary(translated_content, language_detection)
        }
        
        logger.info(f"🌍 다국어 처리 완료: {language_detection['primary_language']} → 한국어")
        return result
    
    def _recommend_stt_model(self, language_detection: Dict) -> Dict:
        """언어 감지 결과에 따른 최적 STT 모델 추천"""
        primary_lang = language_detection['primary_language']
        confidence = language_detection['confidence']
        is_multilingual = language_detection['is_multilingual']
        
        if is_multilingual:
            # 다국어 혼용 시 다국어 모델 사용
            return {
                'model': 'whisper-multilingual',
                'reason': '다국어 혼용 감지',
                'expected_accuracy': 0.85,
                'preprocessing_needed': True
            }
        elif primary_lang == 'korean' and confidence > 0.8:
            # 한국어 전용 모델
            return {
                'model': 'whisper-korean',
                'reason': '한국어 단일 언어 (고신뢰도)',
                'expected_accuracy': 0.95,
                'preprocessing_needed': False
            }
        elif primary_lang == 'english' and confidence > 0.8:
            # 영어 전용 모델
            return {
                'model': 'whisper-english',
                'reason': '영어 단일 언어 (고신뢰도)',
                'expected_accuracy': 0.92,
                'preprocessing_needed': False
            }
        else:
            # 기본적으로 다국어 모델 사용
            return {
                'model': 'whisper-multilingual',
                'reason': '언어 불확실 또는 저신뢰도',
                'expected_accuracy': 0.80,
                'preprocessing_needed': True
            }
    
    def _evaluate_translation_quality(self, original: str, translated: str, terms: List[Dict]) -> Dict:
        """번역 품질 평가"""
        
        # 번역된 용어 개수
        translated_count = len(terms)
        
        # 텍스트 길이 변화율
        length_ratio = len(translated) / max(len(original), 1)
        
        # 주얼리 용어 비율
        total_words = len(original.split())
        term_ratio = translated_count / max(total_words, 1)
        
        # 품질 점수 계산
        quality_score = min(1.0, (translated_count * 0.3 + term_ratio * 0.4 + min(length_ratio, 2.0) * 0.3))
        
        return {
            'quality_score': round(quality_score, 3),
            'translated_terms_count': translated_count,
            'term_coverage_ratio': round(term_ratio, 3),
            'length_change_ratio': round(length_ratio, 3),
            'quality_level': self._get_quality_level(quality_score)
        }
    
    def _get_quality_level(self, score: float) -> str:
        """품질 점수를 레벨로 변환"""
        if score >= 0.8:
            return "우수"
        elif score >= 0.6:
            return "양호"
        elif score >= 0.4:
            return "보통"
        else:
            return "개선필요"
    
    def _generate_korean_summary(self, translated_content: str, language_detection: Dict) -> str:
        """한국어 통합 요약 생성"""
        primary_lang = language_detection['primary_language']
        confidence = language_detection['confidence']
        
        summary_parts = []
        
        # 언어 감지 결과 요약
        if language_detection['is_multilingual']:
            lang_dist = language_detection['language_distribution']
            lang_percentages = [f"{lang}: {score:.1%}" for lang, score in lang_dist.items() if score > 0.1]
            summary_parts.append(f"다국어 환경 감지 ({', '.join(lang_percentages)})")
        else:
            summary_parts.append(f"주요 언어: {primary_lang} (신뢰도: {confidence:.1%})")
        
        # 번역된 내용 요약
        summary_parts.append(f"번역된 내용: {translated_content[:200]}...")
        
        return " | ".join(summary_parts)
    
    def batch_process_multilingual_files(self, files: List[Dict]) -> Dict:
        """여러 다국어 파일 일괄 처리"""
        results = {
            'files_processed': len(files),
            'processing_timestamp': datetime.now().isoformat(),
            'individual_results': [],
            'aggregated_analysis': {}
        }
        
        all_language_detections = []
        all_terms = []
        
        for file_info in files:
            file_result = self.process_multilingual_content(
                file_info['content'], 
                file_info.get('type', 'transcript')
            )
            file_result['file_info'] = file_info
            results['individual_results'].append(file_result)
            
            all_language_detections.append(file_result['language_detection'])
            all_terms.extend(file_result['extracted_terms'])
        
        # 전체 분석
        results['aggregated_analysis'] = self._generate_aggregated_analysis(
            all_language_detections, all_terms
        )
        
        return results
    
    def _generate_aggregated_analysis(self, language_detections: List[Dict], all_terms: List[Dict]) -> Dict:
        """전체 파일들의 통합 분석"""
        
        # 언어 분포 통계
        primary_languages = [ld['primary_language'] for ld in language_detections]
        language_counter = Counter(primary_languages)
        
        # 다국어 파일 비율
        multilingual_count = sum(1 for ld in language_detections if ld['is_multilingual'])
        multilingual_ratio = multilingual_count / max(len(language_detections), 1)
        
        # 전문용어 통계
        term_counter = Counter([term['standard_term'] for term in all_terms])
        
        return {
            'language_distribution': dict(language_counter),
            'multilingual_ratio': round(multilingual_ratio, 3),
            'most_common_language': language_counter.most_common(1)[0] if language_counter else None,
            'total_unique_terms': len(term_counter),
            'most_frequent_terms': term_counter.most_common(10),
            'recommended_strategy': self._recommend_processing_strategy(language_counter, multilingual_ratio)
        }
    
    def _recommend_processing_strategy(self, language_counter: Counter, multilingual_ratio: float) -> Dict:
        """처리 전략 추천"""
        most_common = language_counter.most_common(1)
        
        if multilingual_ratio > 0.5:
            return {
                'strategy': 'multilingual_focus',
                'description': '다국어 혼용이 빈번하므로 다국어 처리에 특화된 파이프라인 사용',
                'recommended_models': ['whisper-multilingual'],
                'preprocessing': ['language_segmentation', 'term_standardization']
            }
        elif most_common and most_common[0][0] == 'korean':
            return {
                'strategy': 'korean_optimized',
                'description': '한국어 중심 환경으로 한국어 특화 모델 사용',
                'recommended_models': ['whisper-korean'],
                'preprocessing': ['korean_text_normalization']
            }
        else:
            return {
                'strategy': 'balanced_approach',
                'description': '균형잡힌 다국어 접근법 사용',
                'recommended_models': ['whisper-multilingual', 'whisper-korean'],
                'preprocessing': ['language_detection', 'adaptive_routing']
            }


# 사용 예시 및 테스트
if __name__ == "__main__":
    processor = MultilingualProcessor()
    
    # 테스트 텍스트 (다국어 혼용)
    test_texts = [
        "안녕하세요, 다이아몬드 price를 문의드립니다. What's the carat?",
        "这个钻石戒指多少钱？ Quality는 어떤가요?",
        "18K gold ring with 1 carat diamond, 가격은 얼마인가요?",
        "주문하고 싶습니다. certificate는 GIA 감정서인가요?"
    ]
    
    print("🌍 다국어 처리 엔진 테스트")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 테스트 {i}: {text}")
        result = processor.process_multilingual_content(text)
        
        print(f"🌏 감지된 언어: {result['language_detection']['primary_language']} "
              f"({result['language_detection']['confidence']:.1%})")
        print(f"🔄 번역된 내용: {result['translated_content']}")
        print(f"💎 발견된 용어: {len(result['extracted_terms'])}개")
        print(f"🤖 추천 모델: {result['recommended_stt_model']['model']}")
        print(f"⭐ 번역 품질: {result['translation_quality']['quality_level']}")
    
    print("\n✅ 다국어 처리 엔진 테스트 완료")