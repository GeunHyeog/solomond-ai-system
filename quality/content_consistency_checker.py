"""
🔍 Content Consistency Checker v2.1
멀티모달 내용 일관성 검증 및 현장 최적화 모듈

주요 기능:
- 음성-이미지-문서 내용 매칭 검증
- 시간 동기화 품질 측정
- 다국어 번역 일관성 검증
- 주얼리 용어 통일성 분석
- 크로스 모달 신뢰도 계산
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

class ContentConsistencyChecker:
    """멀티모달 내용 일관성 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 일관성 검증 기준값
        self.consistency_thresholds = {
            'semantic_similarity_excellent': 0.8,    # 의미 유사도 80% 이상 = 우수
            'semantic_similarity_good': 0.6,         # 의미 유사도 60-80% = 양호
            'semantic_similarity_fair': 0.4,         # 의미 유사도 40-60% = 보통
            'semantic_similarity_poor': 0.2,         # 의미 유사도 20% 미만 = 불량
            
            'temporal_sync_excellent': 2.0,          # 시간 동기화 오차 2초 이내 = 우수
            'temporal_sync_good': 5.0,               # 시간 동기화 오차 5초 이내 = 양호
            'temporal_sync_fair': 10.0,              # 시간 동기화 오차 10초 이내 = 보통
            
            'terminology_consistency_excellent': 0.9, # 용어 일관성 90% 이상 = 우수
            'terminology_consistency_good': 0.8,      # 용어 일관성 80-90% = 양호
            'terminology_consistency_fair': 0.7,      # 용어 일관성 70-80% = 보통
        }
        
        # 주얼리 전문용어 매핑 (다국어)
        self.jewelry_term_mappings = {
            'diamond': ['다이아몬드', 'diamond', 'ダイヤモンド', '钻石'],
            'gold': ['금', 'gold', '金', '黄金'],
            'silver': ['은', 'silver', '銀', '银'],
            'platinum': ['백금', 'platinum', 'プラチナ', '铂金'],
            'carat': ['캐럿', '카라트', 'carat', 'カラット', '克拉'],
            'cut': ['컷', 'cut', 'カット', '切工'],
            'clarity': ['투명도', 'clarity', '透明度', '净度'],
            'color': ['색상', 'color', '色', '颜色'],
            'certificate': ['감정서', '인증서', 'certificate', '証明書', '证书'],
            'gemstone': ['보석', 'gemstone', '宝石', '宝石'],
            'jewelry': ['주얼리', '보석', 'jewelry', 'ジュエリー', '珠宝'],
            'setting': ['세팅', 'setting', 'セッティング', '镶嵌'],
            'appraisal': ['평가서', 'appraisal', '評価書', '评估'],
        }
        
        # 시간 표현 패턴
        self.time_patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',           # HH:MM:SS
            r'(\d{1,2}):(\d{2})',                   # HH:MM
            r'(\d+)\s*분\s*(\d+)\s*초',              # X분 Y초
            r'(\d+)\s*시간\s*(\d+)\s*분',            # X시간 Y분
            r'(\d+)\s*minutes?\s*(\d+)\s*seconds?',  # X minutes Y seconds
            r'(\d+)\s*hours?\s*(\d+)\s*minutes?',    # X hours Y minutes
        ]

    def check_content_consistency(self, 
                                audio_content: Dict = None,
                                image_content: Dict = None, 
                                document_content: Dict = None,
                                sync_data: Dict = None,
                                check_type: str = 'comprehensive') -> Dict:
        """
        멀티모달 내용 일관성 종합 검증
        
        Args:
            audio_content: 음성 인식 결과 (텍스트, 타임스탬프 등)
            image_content: 이미지 OCR 결과 (텍스트, 메타데이터 등)
            document_content: 문서 내용 (텍스트, 구조 등)
            sync_data: 시간 동기화 데이터
            check_type: 검증 유형 ('quick', 'comprehensive', 'jewelry_focused')
            
        Returns:
            Dict: 일관성 검증 결과
        """
        try:
            # 기본 정보
            results = {
                'timestamp': self._get_timestamp(),
                'check_type': check_type,
                'input_sources': self._identify_input_sources(
                    audio_content, image_content, document_content
                )
            }
            
            # 텍스트 내용 추출
            texts = self._extract_texts(audio_content, image_content, document_content)
            results['extracted_texts'] = texts
            
            # 의미적 일관성 검증
            if len(texts) >= 2:
                semantic_consistency = self.analyze_semantic_consistency(texts)
                results['semantic_consistency'] = semantic_consistency
            
            # 시간 동기화 검증
            if sync_data or self._has_temporal_data(audio_content, image_content):
                temporal_consistency = self.analyze_temporal_consistency(
                    audio_content, image_content, sync_data
                )
                results['temporal_consistency'] = temporal_consistency
            
            # 주얼리 용어 일관성 (주얼리 특화 모드)
            if check_type in ['comprehensive', 'jewelry_focused']:
                terminology_consistency = self.analyze_jewelry_terminology_consistency(texts)
                results['terminology_consistency'] = terminology_consistency
            
            # 구조적 일관성 검증
            structural_consistency = self.analyze_structural_consistency(
                audio_content, image_content, document_content
            )
            results['structural_consistency'] = structural_consistency
            
            # 번역 품질 검증 (다국어 내용이 있는 경우)
            translation_consistency = self.analyze_translation_consistency(texts)
            results['translation_consistency'] = translation_consistency
            
            # 전체 일관성 점수 계산
            overall_consistency = self.calculate_overall_consistency_score(results)
            results['overall_consistency'] = overall_consistency
            
            # 개선 권장사항 생성
            recommendations = self.generate_consistency_recommendations(results)
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            self.logger.error(f"내용 일관성 검증 오류: {str(e)}")
            return {
                'error': str(e),
                'overall_consistency': {'score': 0, 'level': 'error'}
            }

    def analyze_semantic_consistency(self, texts: Dict[str, str]) -> Dict:
        """의미적 일관성 분석"""
        try:
            text_sources = list(texts.keys())
            text_contents = list(texts.values())
            
            if len(text_contents) < 2:
                return {
                    'similarity_scores': {},
                    'average_similarity': 0.0,
                    'consistency_level': 'insufficient_data'
                }
            
            # 텍스트 간 유사도 계산
            similarity_scores = {}
            similarities = []
            
            for i in range(len(text_sources)):
                for j in range(i + 1, len(text_sources)):
                    source1, source2 = text_sources[i], text_sources[j]
                    text1, text2 = text_contents[i], text_contents[j]
                    
                    # 기본 문자열 유사도
                    basic_similarity = self._calculate_text_similarity(text1, text2)
                    
                    # 키워드 기반 유사도
                    keyword_similarity = self._calculate_keyword_similarity(text1, text2)
                    
                    # 주얼리 용어 기반 유사도
                    jewelry_similarity = self._calculate_jewelry_term_similarity(text1, text2)
                    
                    # 가중 평균
                    combined_similarity = (
                        basic_similarity * 0.4 +
                        keyword_similarity * 0.4 +
                        jewelry_similarity * 0.2
                    )
                    
                    pair_key = f"{source1}_vs_{source2}"
                    similarity_scores[pair_key] = {
                        'basic': round(basic_similarity, 3),
                        'keyword': round(keyword_similarity, 3),
                        'jewelry': round(jewelry_similarity, 3),
                        'combined': round(combined_similarity, 3)
                    }
                    similarities.append(combined_similarity)
            
            # 평균 유사도
            average_similarity = np.mean(similarities) if similarities else 0.0
            
            # 일관성 등급 분류
            consistency_level = self._classify_semantic_consistency(average_similarity)
            
            # 가장 유사한/다른 쌍 찾기
            if similarities:
                max_sim_idx = np.argmax(similarities)
                min_sim_idx = np.argmin(similarities)
                pair_keys = list(similarity_scores.keys())
                
                most_similar_pair = pair_keys[max_sim_idx]
                most_different_pair = pair_keys[min_sim_idx]
            else:
                most_similar_pair = None
                most_different_pair = None
            
            return {
                'similarity_scores': similarity_scores,
                'average_similarity': round(average_similarity, 3),
                'consistency_level': consistency_level,
                'most_similar_pair': most_similar_pair,
                'most_different_pair': most_different_pair,
                'similarity_distribution': {
                    'max': round(max(similarities), 3) if similarities else 0,
                    'min': round(min(similarities), 3) if similarities else 0,
                    'std': round(np.std(similarities), 3) if similarities else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"의미적 일관성 분석 오류: {str(e)}")
            return {
                'similarity_scores': {},
                'average_similarity': 0.0,
                'consistency_level': 'error'
            }

    def analyze_temporal_consistency(self, 
                                   audio_content: Dict = None,
                                   image_content: Dict = None,
                                   sync_data: Dict = None) -> Dict:
        """시간 동기화 일관성 분석"""
        try:
            # 타임스탬프 추출
            timestamps = self._extract_timestamps(audio_content, image_content, sync_data)
            
            if len(timestamps) < 2:
                return {
                    'sync_quality': 'insufficient_data',
                    'time_offsets': {},
                    'average_offset': 0.0
                }
            
            # 시간 오프셋 계산
            time_offsets = {}
            offsets = []
            
            timestamp_items = list(timestamps.items())
            for i in range(len(timestamp_items)):
                for j in range(i + 1, len(timestamp_items)):
                    source1, time1 = timestamp_items[i]
                    source2, time2 = timestamp_items[j]
                    
                    offset = abs(time1 - time2)
                    pair_key = f"{source1}_vs_{source2}"
                    time_offsets[pair_key] = round(offset, 2)
                    offsets.append(offset)
            
            # 평균 오프셋
            average_offset = np.mean(offsets) if offsets else 0.0
            
            # 동기화 품질 등급
            sync_quality = self._classify_temporal_sync(average_offset)
            
            # 시간 일관성 점수 (오프셋이 작을수록 높은 점수)
            max_acceptable_offset = 30.0  # 30초
            consistency_score = max(0.0, 1.0 - average_offset / max_acceptable_offset)
            
            return {
                'timestamps': timestamps,
                'time_offsets': time_offsets,
                'average_offset': round(average_offset, 2),
                'sync_quality': sync_quality,
                'consistency_score': round(consistency_score, 3),
                'temporal_analysis': {
                    'max_offset': round(max(offsets), 2) if offsets else 0,
                    'min_offset': round(min(offsets), 2) if offsets else 0,
                    'offset_std': round(np.std(offsets), 2) if offsets else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"시간 일관성 분석 오류: {str(e)}")
            return {
                'sync_quality': 'error',
                'time_offsets': {},
                'average_offset': 0.0
            }

    def analyze_jewelry_terminology_consistency(self, texts: Dict[str, str]) -> Dict:
        """주얼리 용어 일관성 분석"""
        try:
            # 각 텍스트에서 주얼리 용어 추출
            term_usage = {}
            
            for source, text in texts.items():
                extracted_terms = self._extract_jewelry_terms(text)
                term_usage[source] = extracted_terms
            
            # 용어 매핑 일관성 검증
            consistency_results = {}
            
            for standard_term, variants in self.jewelry_term_mappings.items():
                term_consistency = self._check_term_consistency_across_sources(
                    standard_term, variants, term_usage
                )
                if term_consistency['found_in_sources']:
                    consistency_results[standard_term] = term_consistency
            
            # 전체 용어 일관성 점수 계산
            if consistency_results:
                consistency_scores = [
                    result['consistency_score'] 
                    for result in consistency_results.values()
                ]
                overall_score = np.mean(consistency_scores)
            else:
                overall_score = 1.0  # 용어가 없으면 일관성 문제도 없음
            
            # 일관성 등급 분류
            consistency_level = self._classify_terminology_consistency(overall_score)
            
            # 불일치 용어 찾기
            inconsistent_terms = [
                term for term, result in consistency_results.items()
                if result['consistency_score'] < 0.7
            ]
            
            return {
                'term_usage_by_source': term_usage,
                'consistency_results': consistency_results,
                'overall_score': round(overall_score, 3),
                'consistency_level': consistency_level,
                'inconsistent_terms': inconsistent_terms,
                'term_statistics': {
                    'total_unique_terms': len(set().union(*[
                        terms.keys() for terms in term_usage.values()
                    ])),
                    'terms_checked': len(consistency_results),
                    'consistent_terms': len([
                        t for t in consistency_results.values() 
                        if t['consistency_score'] >= 0.7
                    ])
                }
            }
            
        except Exception as e:
            self.logger.error(f"주얼리 용어 일관성 분석 오류: {str(e)}")
            return {
                'term_usage_by_source': {},
                'overall_score': 0.0,
                'consistency_level': 'error'
            }

    def analyze_structural_consistency(self, 
                                     audio_content: Dict = None,
                                     image_content: Dict = None,
                                     document_content: Dict = None) -> Dict:
        """구조적 일관성 분석"""
        try:
            # 각 소스의 구조 정보 추출
            structure_info = {}
            
            if audio_content:
                structure_info['audio'] = self._analyze_audio_structure(audio_content)
            
            if image_content:
                structure_info['image'] = self._analyze_image_structure(image_content)
            
            if document_content:
                structure_info['document'] = self._analyze_document_structure(document_content)
            
            # 구조 일관성 검증
            consistency_checks = {}
            
            # 섹션/주제 일관성
            if len(structure_info) >= 2:
                section_consistency = self._check_section_consistency(structure_info)
                consistency_checks['sections'] = section_consistency
            
            # 정보 계층 일관성
            hierarchy_consistency = self._check_hierarchy_consistency(structure_info)
            consistency_checks['hierarchy'] = hierarchy_consistency
            
            # 전체 구조 일관성 점수
            if consistency_checks:
                structure_scores = [
                    check['score'] for check in consistency_checks.values()
                    if isinstance(check, dict) and 'score' in check
                ]
                overall_score = np.mean(structure_scores) if structure_scores else 0.5
            else:
                overall_score = 0.5
            
            return {
                'structure_info': structure_info,
                'consistency_checks': consistency_checks,
                'overall_score': round(overall_score, 3),
                'structure_level': self._classify_structural_consistency(overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"구조적 일관성 분석 오류: {str(e)}")
            return {
                'structure_info': {},
                'overall_score': 0.0,
                'structure_level': 'error'
            }

    def analyze_translation_consistency(self, texts: Dict[str, str]) -> Dict:
        """번역 품질/일관성 분석"""
        try:
            # 언어별 텍스트 분류
            language_texts = {}
            
            for source, text in texts.items():
                language = self._detect_primary_language(text)
                if language not in language_texts:
                    language_texts[language] = {}
                language_texts[language][source] = text
            
            if len(language_texts) < 2:
                return {
                    'languages_detected': list(language_texts.keys()),
                    'translation_quality': 'monolingual_content',
                    'consistency_score': 1.0
                }
            
            # 언어 간 내용 일관성 검증
            translation_checks = {}
            
            languages = list(language_texts.keys())
            for i in range(len(languages)):
                for j in range(i + 1, len(languages)):
                    lang1, lang2 = languages[i], languages[j]
                    
                    # 번역 품질 추정
                    quality_score = self._estimate_translation_quality(
                        language_texts[lang1], language_texts[lang2]
                    )
                    
                    translation_checks[f"{lang1}_to_{lang2}"] = quality_score
            
            # 전체 번역 일관성 점수
            if translation_checks:
                translation_scores = [
                    check['consistency_score'] 
                    for check in translation_checks.values()
                ]
                overall_score = np.mean(translation_scores)
            else:
                overall_score = 1.0
            
            return {
                'languages_detected': list(language_texts.keys()),
                'language_distribution': {
                    lang: len(texts) for lang, texts in language_texts.items()
                },
                'translation_checks': translation_checks,
                'consistency_score': round(overall_score, 3),
                'translation_quality': self._classify_translation_quality(overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"번역 일관성 분석 오류: {str(e)}")
            return {
                'languages_detected': [],
                'translation_quality': 'error',
                'consistency_score': 0.0
            }

    def calculate_overall_consistency_score(self, results: Dict) -> Dict:
        """전체 일관성 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                'semantic': 0.4,      # 의미적 일관성 40%
                'temporal': 0.2,      # 시간 일관성 20%
                'terminology': 0.2,   # 용어 일관성 20%
                'structural': 0.1,    # 구조적 일관성 10%
                'translation': 0.1    # 번역 일관성 10%
            }
            
            # 개별 점수 추출 및 정규화
            semantic_score = results.get('semantic_consistency', {}).get('average_similarity', 0)
            temporal_score = results.get('temporal_consistency', {}).get('consistency_score', 0)
            terminology_score = results.get('terminology_consistency', {}).get('overall_score', 0)
            structural_score = results.get('structural_consistency', {}).get('overall_score', 0)
            translation_score = results.get('translation_consistency', {}).get('consistency_score', 0)
            
            # 가중 평균 계산
            overall_score = (
                semantic_score * weights['semantic'] +
                temporal_score * weights['temporal'] +
                terminology_score * weights['terminology'] +
                structural_score * weights['structural'] +
                translation_score * weights['translation']
            )
            
            # 등급 분류
            if overall_score >= 0.8:
                level, status, color = 'excellent', '우수', '🟢'
            elif overall_score >= 0.6:
                level, status, color = 'good', '양호', '🟡'
            elif overall_score >= 0.4:
                level, status, color = 'fair', '보통', '🟠'
            else:
                level, status, color = 'poor', '불량', '🔴'
            
            return {
                'score': round(overall_score, 3),
                'percentage': round(overall_score * 100, 1),
                'level': level,
                'status': status,
                'color': color,
                'components': {
                    'semantic_score': round(semantic_score, 3),
                    'temporal_score': round(temporal_score, 3),
                    'terminology_score': round(terminology_score, 3),
                    'structural_score': round(structural_score, 3),
                    'translation_score': round(translation_score, 3)
                }
            }
            
        except Exception as e:
            self.logger.error(f"전체 일관성 점수 계산 오류: {str(e)}")
            return {
                'score': 0.0,
                'level': 'error',
                'status': '오류'
            }

    def generate_consistency_recommendations(self, results: Dict) -> List[Dict]:
        """일관성 개선 권장사항 생성"""
        recommendations = []
        
        try:
            # 의미적 일관성 권장사항
            semantic = results.get('semantic_consistency', {})
            if semantic.get('consistency_level') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '📝',
                    'title': '내용 일관성 부족',
                    'message': '음성, 이미지, 문서 간 내용이 일치하지 않습니다. 동일한 주제에 대해 일관된 설명을 하세요',
                    'action': 'improve_content_consistency'
                })
            
            # 시간 동기화 권장사항
            temporal = results.get('temporal_consistency', {})
            if temporal.get('sync_quality') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'warning',
                    'icon': '⏰',
                    'title': '시간 동기화 문제',
                    'message': '음성과 이미지의 시간이 일치하지 않습니다. 동시에 녹음/촬영하거나 시간을 맞춰주세요',
                    'action': 'improve_temporal_sync'
                })
            
            # 용어 일관성 권장사항
            terminology = results.get('terminology_consistency', {})
            if terminology.get('inconsistent_terms'):
                recommendations.append({
                    'type': 'info',
                    'icon': '💎',
                    'title': '주얼리 용어 불일치',
                    'message': f"용어 통일 필요: {', '.join(terminology.get('inconsistent_terms', [])[:3])}",
                    'action': 'standardize_jewelry_terminology'
                })
            
            # 구조적 일관성 권장사항
            structural = results.get('structural_consistency', {})
            if structural.get('structure_level') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'info',
                    'icon': '🏗️',
                    'title': '구조 개선 필요',
                    'message': '각 매체의 정보 구조를 일치시켜 주세요',
                    'action': 'improve_content_structure'
                })
            
            # 번역 품질 권장사항
            translation = results.get('translation_consistency', {})
            if translation.get('translation_quality') in ['poor', 'fair']:
                recommendations.append({
                    'type': 'info',
                    'icon': '🌐',
                    'title': '번역 품질 개선',
                    'message': '다국어 내용 간 번역 품질을 개선해주세요',
                    'action': 'improve_translation_quality'
                })
            
            # 전체 일관성 권장사항
            overall = results.get('overall_consistency', {})
            if overall.get('level') == 'poor':
                recommendations.append({
                    'type': 'critical',
                    'icon': '🔴',
                    'title': '전체 재검토 필요',
                    'message': '멀티모달 내용의 일관성이 매우 낮습니다. 전체적으로 재검토해주세요',
                    'action': 'comprehensive_review_needed'
                })
            elif overall.get('level') == 'excellent':
                recommendations.append({
                    'type': 'success',
                    'icon': '🟢',
                    'title': '일관성 우수',
                    'message': '모든 매체 간 내용이 잘 일치합니다',
                    'action': 'maintain_consistency'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"일관성 권장사항 생성 오류: {str(e)}")
            return [{
                'type': 'error',
                'icon': '❌',
                'title': '일관성 분석 오류',
                'message': '내용 일관성 분석 중 오류가 발생했습니다',
                'action': 'retry_consistency_check'
            }]

    # === 내부 유틸리티 함수들 ===
    
    def _identify_input_sources(self, audio_content, image_content, document_content) -> List[str]:
        """입력 소스 식별"""
        sources = []
        if audio_content:
            sources.append('audio')
        if image_content:
            sources.append('image') 
        if document_content:
            sources.append('document')
        return sources
    
    def _extract_texts(self, audio_content, image_content, document_content) -> Dict[str, str]:
        """각 소스에서 텍스트 추출"""
        texts = {}
        
        if audio_content and isinstance(audio_content, dict):
            if 'text' in audio_content:
                texts['audio'] = audio_content['text']
            elif 'transcription' in audio_content:
                texts['audio'] = audio_content['transcription']
        
        if image_content and isinstance(image_content, dict):
            if 'text' in image_content:
                texts['image'] = image_content['text']
            elif 'ocr_text' in image_content:
                texts['image'] = image_content['ocr_text']
        
        if document_content and isinstance(document_content, dict):
            if 'text' in document_content:
                texts['document'] = document_content['text']
            elif 'content' in document_content:
                texts['document'] = document_content['content']
        
        return texts
    
    def _has_temporal_data(self, audio_content, image_content) -> bool:
        """시간 데이터 존재 여부 확인"""
        temporal_keys = ['timestamp', 'time', 'created_at', 'recorded_at', 'captured_at']
        
        if audio_content and isinstance(audio_content, dict):
            if any(key in audio_content for key in temporal_keys):
                return True
        
        if image_content and isinstance(image_content, dict):
            if any(key in image_content for key in temporal_keys):
                return True
        
        return False
    
    def _extract_timestamps(self, audio_content, image_content, sync_data) -> Dict[str, float]:
        """타임스탬프 추출 및 통일"""
        timestamps = {}
        
        # 기준 시간 (Unix timestamp)
        try:
            if sync_data and 'reference_time' in sync_data:
                ref_time = sync_data['reference_time']
            else:
                ref_time = datetime.now().timestamp()
            
            # 음성 타임스탬프
            if audio_content and isinstance(audio_content, dict):
                audio_time = self._parse_timestamp(audio_content, ref_time)
                if audio_time is not None:
                    timestamps['audio'] = audio_time
            
            # 이미지 타임스탬프
            if image_content and isinstance(image_content, dict):
                image_time = self._parse_timestamp(image_content, ref_time)
                if image_time is not None:
                    timestamps['image'] = image_time
            
        except Exception as e:
            self.logger.error(f"타임스탬프 추출 오류: {str(e)}")
        
        return timestamps
    
    def _parse_timestamp(self, content: Dict, ref_time: float) -> Optional[float]:
        """개별 타임스탬프 파싱"""
        temporal_keys = ['timestamp', 'time', 'created_at', 'recorded_at', 'captured_at']
        
        for key in temporal_keys:
            if key in content:
                value = content[key]
                
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    try:
                        # ISO 형식 파싱 시도
                        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        return dt.timestamp()
                    except:
                        try:
                            # Unix timestamp 파싱 시도
                            return float(value)
                        except:
                            continue
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """기본 텍스트 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        
        # SequenceMatcher를 사용한 유사도
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity
    
    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """키워드 기반 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        
        # 단어 단위로 분할
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        # 자카드 유사도
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_jewelry_term_similarity(self, text1: str, text2: str) -> float:
        """주얼리 용어 기반 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        
        # 주얼리 용어 추출
        terms1 = self._extract_jewelry_terms(text1)
        terms2 = self._extract_jewelry_terms(text2)
        
        if not terms1 and not terms2:
            return 1.0  # 둘 다 주얼리 용어가 없으면 일관성 있음
        
        # 정규화된 용어로 변환
        normalized_terms1 = set()
        normalized_terms2 = set()
        
        for standard_term, variants in self.jewelry_term_mappings.items():
            for term in terms1:
                if term.lower() in [v.lower() for v in variants]:
                    normalized_terms1.add(standard_term)
            for term in terms2:
                if term.lower() in [v.lower() for v in variants]:
                    normalized_terms2.add(standard_term)
        
        # 자카드 유사도
        if not normalized_terms1 and not normalized_terms2:
            return 1.0
        
        intersection = len(normalized_terms1.intersection(normalized_terms2))
        union = len(normalized_terms1.union(normalized_terms2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_jewelry_terms(self, text: str) -> Dict[str, int]:
        """텍스트에서 주얼리 용어 추출"""
        terms = {}
        text_lower = text.lower()
        
        # 모든 주얼리 용어 변형 검색
        for standard_term, variants in self.jewelry_term_mappings.items():
            count = 0
            for variant in variants:
                count += len(re.findall(r'\b' + re.escape(variant.lower()) + r'\b', text_lower))
            
            if count > 0:
                terms[standard_term] = count
        
        # 추가 주얼리 관련 단어 검색
        additional_terms = [
            'ring', '반지', 'necklace', '목걸이', 'bracelet', '팔찌',
            'earring', '귀걸이', 'brooch', '브로치', 'pendant', '펜던트'
        ]
        
        for term in additional_terms:
            count = len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', text_lower))
            if count > 0:
                terms[term] = count
        
        return terms
    
    def _check_term_consistency_across_sources(self, standard_term: str, 
                                             variants: List[str], 
                                             term_usage: Dict) -> Dict:
        """소스 간 용어 일관성 검증"""
        found_variants = {}
        sources_with_term = []
        
        for source, terms in term_usage.items():
            if standard_term in terms:
                found_variants[source] = standard_term
                sources_with_term.append(source)
        
        # 용어가 발견된 소스가 없으면 검증 불가
        if not sources_with_term:
            return {
                'found_in_sources': [],
                'consistency_score': 1.0,  # 없으면 일관성 문제도 없음
                'variants_used': {}
            }
        
        # 모든 소스에서 같은 표준 용어를 사용하면 완전 일관성
        consistency_score = 1.0
        
        return {
            'found_in_sources': sources_with_term,
            'consistency_score': consistency_score,
            'variants_used': found_variants
        }
    
    def _detect_primary_language(self, text: str) -> str:
        """텍스트의 주요 언어 감지"""
        if not text:
            return 'unknown'
        
        # 한글 문자 비율
        korean_chars = len(re.findall(r'[가-힣]', text))
        # 영문 문자 비율
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        # 중문 문자 비율
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 일문 문자 비율
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        
        total_chars = korean_chars + english_chars + chinese_chars + japanese_chars
        
        if total_chars == 0:
            return 'unknown'
        
        # 가장 많은 문자의 언어 반환
        char_counts = {
            'korean': korean_chars,
            'english': english_chars,
            'chinese': chinese_chars,
            'japanese': japanese_chars
        }
        
        return max(char_counts, key=char_counts.get)
    
    def _estimate_translation_quality(self, texts1: Dict, texts2: Dict) -> Dict:
        """번역 품질 추정"""
        # 간단한 휴리스틱 기반 번역 품질 추정
        # 실제로는 더 정교한 번역 품질 측정 알고리즘 필요
        
        all_texts1 = ' '.join(texts1.values())
        all_texts2 = ' '.join(texts2.values())
        
        # 기본 유사도 (구조적 유사성)
        basic_similarity = self._calculate_text_similarity(all_texts1, all_texts2)
        
        # 주얼리 용어 매핑 일관성
        jewelry_consistency = self._calculate_jewelry_term_similarity(all_texts1, all_texts2)
        
        # 번역 품질 점수 (구조 + 용어 일관성)
        quality_score = (basic_similarity * 0.6 + jewelry_consistency * 0.4)
        
        return {
            'consistency_score': quality_score,
            'basic_similarity': basic_similarity,
            'jewelry_consistency': jewelry_consistency
        }
    
    def _analyze_audio_structure(self, audio_content: Dict) -> Dict:
        """음성 내용 구조 분석"""
        structure = {
            'type': 'audio',
            'has_timestamps': 'timestamp' in audio_content,
            'has_segments': 'segments' in audio_content,
            'estimated_duration': audio_content.get('duration', 0)
        }
        
        # 텍스트에서 구조 추정
        if 'text' in audio_content:
            text = audio_content['text']
            structure['word_count'] = len(text.split())
            structure['sentence_count'] = len(re.split(r'[.!?]', text))
            structure['paragraph_count'] = len(text.split('\n\n'))
        
        return structure
    
    def _analyze_image_structure(self, image_content: Dict) -> Dict:
        """이미지 내용 구조 분석"""
        structure = {
            'type': 'image',
            'has_ocr_text': 'text' in image_content or 'ocr_text' in image_content,
            'has_regions': 'regions' in image_content,
        }
        
        # OCR 텍스트에서 구조 추정
        text_key = 'text' if 'text' in image_content else 'ocr_text'
        if text_key in image_content:
            text = image_content[text_key]
            structure['text_length'] = len(text)
            structure['line_count'] = len(text.split('\n'))
        
        return structure
    
    def _analyze_document_structure(self, document_content: Dict) -> Dict:
        """문서 내용 구조 분석"""
        structure = {
            'type': 'document',
            'has_sections': 'sections' in document_content,
            'has_headers': 'headers' in document_content,
        }
        
        # 텍스트에서 구조 추정
        text_key = 'text' if 'text' in document_content else 'content'
        if text_key in document_content:
            text = document_content[text_key]
            structure['total_length'] = len(text)
            structure['paragraph_count'] = len(text.split('\n\n'))
            
            # 헤더 추정 (# 또는 대문자 라인)
            headers = re.findall(r'^[A-Z\s]{5,}$', text, re.MULTILINE)
            structure['estimated_headers'] = len(headers)
        
        return structure
    
    def _check_section_consistency(self, structure_info: Dict) -> Dict:
        """섹션 일관성 검증"""
        # 각 소스의 섹션/구조 정보 비교
        section_scores = []
        
        sources = list(structure_info.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                struct1, struct2 = structure_info[source1], structure_info[source2]
                
                # 구조 유사성 점수 계산 (간단한 휴리스틱)
                similarity = 0.5  # 기본값
                
                # 텍스트 길이 비교
                if 'word_count' in struct1 and 'text_length' in struct2:
                    ratio = min(struct1['word_count'], struct2['text_length']) / max(struct1['word_count'], struct2['text_length'])
                    similarity = ratio
                
                section_scores.append(similarity)
        
        avg_score = np.mean(section_scores) if section_scores else 0.5
        
        return {
            'score': round(avg_score, 3),
            'comparisons': len(section_scores)
        }
    
    def _check_hierarchy_consistency(self, structure_info: Dict) -> Dict:
        """정보 계층 일관성 검증"""
        # 간단한 계층 일관성 검증
        hierarchy_score = 0.5  # 기본값
        
        # 모든 소스가 비슷한 복잡도를 가지는지 확인
        complexities = []
        for source, struct in structure_info.items():
            complexity = 0
            
            if struct.get('paragraph_count', 0) > 3:
                complexity += 1
            if struct.get('estimated_headers', 0) > 1:
                complexity += 1
            if struct.get('sentence_count', 0) > 10:
                complexity += 1
            
            complexities.append(complexity)
        
        if complexities:
            # 복잡도 분산이 낮을수록 일관성 높음
            std_dev = np.std(complexities)
            hierarchy_score = max(0.0, 1.0 - std_dev / 3)
        
        return {
            'score': round(hierarchy_score, 3),
            'complexities': complexities
        }
    
    def _classify_semantic_consistency(self, similarity: float) -> str:
        """의미적 일관성 등급 분류"""
        if similarity >= self.consistency_thresholds['semantic_similarity_excellent']:
            return 'excellent'
        elif similarity >= self.consistency_thresholds['semantic_similarity_good']:
            return 'good'
        elif similarity >= self.consistency_thresholds['semantic_similarity_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_temporal_sync(self, offset: float) -> str:
        """시간 동기화 등급 분류"""
        if offset <= self.consistency_thresholds['temporal_sync_excellent']:
            return 'excellent'
        elif offset <= self.consistency_thresholds['temporal_sync_good']:
            return 'good'
        elif offset <= self.consistency_thresholds['temporal_sync_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_terminology_consistency(self, score: float) -> str:
        """용어 일관성 등급 분류"""
        if score >= self.consistency_thresholds['terminology_consistency_excellent']:
            return 'excellent'
        elif score >= self.consistency_thresholds['terminology_consistency_good']:
            return 'good'
        elif score >= self.consistency_thresholds['terminology_consistency_fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_structural_consistency(self, score: float) -> str:
        """구조적 일관성 등급 분류"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _classify_translation_quality(self, score: float) -> str:
        """번역 품질 등급 분류"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 사용 예제
if __name__ == "__main__":
    checker = ContentConsistencyChecker()
    
    print("🔍 Content Consistency Checker v2.1 - 테스트 시작")
    print("=" * 50)
    
    # 실제 사용 예제
    # audio_data = {'text': '다이아몬드 반지의 품질은 4C로 평가됩니다', 'timestamp': 1625097600}
    # image_data = {'text': 'Diamond ring quality: 4C evaluation', 'timestamp': 1625097605}
    # 
    # result = checker.check_content_consistency(
    #     audio_content=audio_data,
    #     image_content=image_data,
    #     check_type='jewelry_focused'
    # )
    # 
    # print(f"전체 일관성: {result['overall_consistency']['percentage']}%")
    # print(f"의미적 유사도: {result['semantic_consistency']['average_similarity']:.3f}")
    
    print("모듈 로드 완료 ✅")
