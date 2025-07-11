"""
🧠 Smart Content Merger v2.1
지능형 내용 병합 및 현장 최적화 모듈

주요 기능:
- 다중 파일 지능형 병합
- 중복 내용 자동 감지 및 제거
- 시간순 내용 정렬 및 연결
- 주얼리 전문용어 통합 관리
- 현장 실시간 내용 통합
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import re
import logging
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

class ContentItem:
    """내용 아이템 클래스"""
    def __init__(self, 
                 content: str, 
                 source_type: str, 
                 timestamp: float = None,
                 metadata: Dict = None):
        self.content = content
        self.source_type = source_type  # 'audio', 'image', 'document'
        self.timestamp = timestamp or datetime.now().timestamp()
        self.metadata = metadata or {}
        self.processed_content = None
        self.importance_score = 0.0
        self.similarity_groups = []
        self.merged_with = []

class SmartContentMerger:
    """지능형 내용 병합기"""
    
    def __init__(self, jewelry_mode: bool = True):
        self.logger = logging.getLogger(__name__)
        self.jewelry_mode = jewelry_mode
        
        # 병합 설정
        self.merge_config = {
            'similarity_threshold': 0.7,        # 유사도 임계값
            'time_window_seconds': 30.0,        # 시간 윈도우 (초)
            'min_content_length': 10,           # 최소 내용 길이
            'duplicate_threshold': 0.9,         # 중복 임계값
            'importance_weight_audio': 0.4,     # 음성 가중치
            'importance_weight_image': 0.4,     # 이미지 가중치
            'importance_weight_document': 0.2,  # 문서 가중치
        }
        
        # 주얼리 전문용어 매핑
        self.jewelry_terms = {
            'quality_terms': [
                'diamond', '다이아몬드', 'carat', '캐럿', 'cut', '컷',
                'clarity', '투명도', 'color', '색상', 'certificate', '감정서'
            ],
            'material_terms': [
                'gold', '금', 'silver', '은', 'platinum', '백금',
                'gemstone', '보석', 'ruby', '루비', 'emerald', '에메랄드'
            ],
            'product_terms': [
                'ring', '반지', 'necklace', '목걸이', 'earring', '귀걸이',
                'bracelet', '팔찌', 'pendant', '펜던트', 'brooch', '브로치'
            ],
            'business_terms': [
                'price', '가격', 'appraisal', '평가', 'collection', '컬렉션',
                'exhibition', '전시회', 'auction', '경매', 'investment', '투자'
            ]
        }
        
        # 중복 제거 패턴
        self.duplicate_patterns = [
            r'(\w+)\s+\1\s+\1+',               # 단어 반복
            r'(.{10,})\s*\1',                  # 긴 구문 반복
            r'(\S+\s+\S+)\s+\1',               # 단어 쌍 반복
        ]
        
        # 내용 분류 키워드
        self.content_categories = {
            'product_description': ['특징', '디자인', '소재', '크기', 'feature', 'design', 'material', 'size'],
            'quality_assessment': ['품질', '등급', '상태', 'quality', 'grade', 'condition'],
            'pricing_information': ['가격', '비용', '견적', 'price', 'cost', 'estimate'],
            'technical_specs': ['사양', '규격', '치수', 'specification', 'dimension'],
            'market_information': ['시장', '트렌드', '경쟁', 'market', 'trend', 'competition']
        }

    def merge_multiple_contents(self, 
                              content_items: List[ContentItem],
                              merge_strategy: str = 'comprehensive') -> Dict:
        """
        다중 내용 지능형 병합
        
        Args:
            content_items: 병합할 내용 아이템들
            merge_strategy: 병합 전략 ('comprehensive', 'temporal', 'importance_based')
            
        Returns:
            Dict: 병합 결과
        """
        try:
            self.logger.info(f"내용 병합 시작: {len(content_items)}개 아이템, 전략: {merge_strategy}")
            
            # 1. 전처리
            processed_items = self._preprocess_content_items(content_items)
            
            # 2. 중복 내용 감지 및 제거
            deduplicated_items = self._remove_duplicates(processed_items)
            
            # 3. 유사 내용 그룹핑
            similarity_groups = self._group_similar_content(deduplicated_items)
            
            # 4. 시간순 정렬
            temporal_sorted = self._sort_by_temporal_logic(similarity_groups)
            
            # 5. 내용 중요도 계산
            importance_scored = self._calculate_importance_scores(temporal_sorted)
            
            # 6. 병합 전략 적용
            merged_content = self._apply_merge_strategy(importance_scored, merge_strategy)
            
            # 7. 최종 정리 및 구조화
            final_result = self._finalize_merged_content(merged_content)
            
            # 8. 병합 메타데이터 생성
            merge_metadata = self._generate_merge_metadata(content_items, final_result)
            
            return {
                'merged_content': final_result,
                'metadata': merge_metadata,
                'original_count': len(content_items),
                'final_count': len(final_result.get('sections', [])),
                'merge_strategy': merge_strategy,
                'quality_score': self._calculate_merge_quality_score(final_result),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"내용 병합 오류: {str(e)}")
            return {
                'error': str(e),
                'merged_content': {},
                'metadata': {}
            }

    def merge_realtime_content(self, 
                             existing_content: Dict,
                             new_content: ContentItem) -> Dict:
        """
        실시간 내용 병합 (기존 내용에 새 내용 추가)
        
        Args:
            existing_content: 기존 병합된 내용
            new_content: 새로 추가할 내용
            
        Returns:
            Dict: 업데이트된 병합 결과
        """
        try:
            # 새 내용 전처리
            processed_new = self._preprocess_single_content(new_content)
            
            # 기존 내용과 유사도 검사
            similarity_results = self._check_similarity_with_existing(
                existing_content, processed_new
            )
            
            # 병합 결정
            if similarity_results['should_merge']:
                # 기존 섹션에 병합
                updated_content = self._merge_with_existing_section(
                    existing_content, processed_new, similarity_results
                )
            else:
                # 새 섹션으로 추가
                updated_content = self._add_as_new_section(
                    existing_content, processed_new
                )
            
            # 메타데이터 업데이트
            self._update_realtime_metadata(updated_content, new_content)
            
            return updated_content
            
        except Exception as e:
            self.logger.error(f"실시간 내용 병합 오류: {str(e)}")
            return existing_content

    def extract_key_insights(self, merged_content: Dict) -> Dict:
        """병합된 내용에서 핵심 인사이트 추출"""
        try:
            insights = {
                'key_topics': [],
                'important_facts': [],
                'jewelry_specific_info': {},
                'action_items': [],
                'summary': '',
                'confidence_score': 0.0
            }
            
            if not merged_content.get('sections'):
                return insights
            
            # 전체 텍스트 결합
            all_text = ' '.join([
                section.get('content', '') 
                for section in merged_content.get('sections', [])
            ])
            
            # 주요 토픽 추출
            insights['key_topics'] = self._extract_key_topics(all_text)
            
            # 중요한 사실 추출
            insights['important_facts'] = self._extract_important_facts(all_text)
            
            # 주얼리 특화 정보 추출
            if self.jewelry_mode:
                insights['jewelry_specific_info'] = self._extract_jewelry_info(all_text)
            
            # 액션 아이템 추출
            insights['action_items'] = self._extract_action_items(all_text)
            
            # 요약 생성
            insights['summary'] = self._generate_smart_summary(merged_content)
            
            # 신뢰도 점수 계산
            insights['confidence_score'] = self._calculate_insight_confidence(
                merged_content, insights
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"인사이트 추출 오류: {str(e)}")
            return {
                'key_topics': [],
                'important_facts': [],
                'jewelry_specific_info': {},
                'action_items': [],
                'summary': '인사이트 추출 중 오류가 발생했습니다.',
                'confidence_score': 0.0
            }

    # === 전처리 메서드들 ===
    
    def _preprocess_content_items(self, items: List[ContentItem]) -> List[ContentItem]:
        """내용 아이템들 전처리"""
        processed_items = []
        
        for item in items:
            # 내용 정리
            cleaned_content = self._clean_content(item.content)
            
            # 최소 길이 체크
            if len(cleaned_content) < self.merge_config['min_content_length']:
                continue
            
            # 처리된 내용 저장
            item.processed_content = cleaned_content
            
            # 기본 중요도 점수 계산
            item.importance_score = self._calculate_basic_importance(item)
            
            processed_items.append(item)
        
        return processed_items
    
    def _preprocess_single_content(self, item: ContentItem) -> ContentItem:
        """단일 내용 아이템 전처리"""
        item.processed_content = self._clean_content(item.content)
        item.importance_score = self._calculate_basic_importance(item)
        return item
    
    def _clean_content(self, content: str) -> str:
        """내용 정리"""
        if not content:
            return ""
        
        # 중복 제거
        for pattern in self.duplicate_patterns:
            content = re.sub(pattern, r'\1', content, flags=re.IGNORECASE)
        
        # 여러 공백을 하나로
        content = re.sub(r'\s+', ' ', content)
        
        # 앞뒤 공백 제거
        content = content.strip()
        
        return content
    
    def _calculate_basic_importance(self, item: ContentItem) -> float:
        """기본 중요도 점수 계산"""
        score = 0.0
        content = item.processed_content or item.content
        
        # 소스 타입별 기본 가중치
        source_weights = {
            'audio': self.merge_config['importance_weight_audio'],
            'image': self.merge_config['importance_weight_image'],
            'document': self.merge_config['importance_weight_document']
        }
        score += source_weights.get(item.source_type, 0.2)
        
        # 내용 길이 보너스
        length_bonus = min(0.3, len(content) / 1000)
        score += length_bonus
        
        # 주얼리 용어 보너스
        if self.jewelry_mode:
            jewelry_bonus = self._calculate_jewelry_term_bonus(content)
            score += jewelry_bonus
        
        # 숫자/데이터 포함 보너스
        if re.search(r'\d+(?:\.\d+)?', content):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_jewelry_term_bonus(self, content: str) -> float:
        """주얼리 용어 보너스 계산"""
        bonus = 0.0
        content_lower = content.lower()
        
        for category, terms in self.jewelry_terms.items():
            found_terms = sum(1 for term in terms if term.lower() in content_lower)
            bonus += found_terms * 0.02  # 용어당 2% 보너스
        
        return min(0.2, bonus)  # 최대 20% 보너스
    
    # === 중복 제거 메서드들 ===
    
    def _remove_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """중복 내용 제거"""
        unique_items = []
        seen_contents = []
        
        for item in items:
            content = item.processed_content
            is_duplicate = False
            
            for seen_content in seen_contents:
                similarity = self._calculate_text_similarity(content, seen_content)
                if similarity >= self.merge_config['duplicate_threshold']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_items.append(item)
                seen_contents.append(content)
            else:
                self.logger.debug(f"중복 내용 제거: {content[:50]}...")
        
        return unique_items
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        
        # 기본 문자열 유사도
        basic_similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # 단어 기반 유사도
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        word_similarity = intersection / union if union > 0 else 0.0
        
        # 가중 평균
        return basic_similarity * 0.6 + word_similarity * 0.4
    
    # === 유사 내용 그룹핑 메서드들 ===
    
    def _group_similar_content(self, items: List[ContentItem]) -> List[List[ContentItem]]:
        """유사한 내용끼리 그룹핑"""
        groups = []
        ungrouped_items = items.copy()
        
        while ungrouped_items:
            current_item = ungrouped_items.pop(0)
            current_group = [current_item]
            
            # 유사한 아이템 찾기
            remaining_items = []
            for item in ungrouped_items:
                similarity = self._calculate_text_similarity(
                    current_item.processed_content,
                    item.processed_content
                )
                
                if similarity >= self.merge_config['similarity_threshold']:
                    current_group.append(item)
                    item.similarity_groups.append(len(groups))
                else:
                    remaining_items.append(item)
            
            groups.append(current_group)
            ungrouped_items = remaining_items
        
        return groups
    
    # === 시간순 정렬 메서드들 ===
    
    def _sort_by_temporal_logic(self, groups: List[List[ContentItem]]) -> List[List[ContentItem]]:
        """시간 논리에 따른 정렬"""
        # 각 그룹 내에서 시간순 정렬
        for group in groups:
            group.sort(key=lambda x: x.timestamp)
        
        # 그룹들을 대표 시간순으로 정렬
        groups.sort(key=lambda group: min(item.timestamp for item in group))
        
        return groups
    
    # === 중요도 계산 메서드들 ===
    
    def _calculate_importance_scores(self, groups: List[List[ContentItem]]) -> List[List[ContentItem]]:
        """그룹별 중요도 점수 재계산"""
        for group in groups:
            # 그룹 내 상호 참조 보너스
            if len(group) > 1:
                for item in group:
                    item.importance_score += 0.1 * (len(group) - 1)
            
            # 시간적 연속성 보너스
            if len(group) > 1:
                for i in range(1, len(group)):
                    time_diff = group[i].timestamp - group[i-1].timestamp
                    if time_diff <= self.merge_config['time_window_seconds']:
                        group[i].importance_score += 0.05
        
        return groups
    
    # === 병합 전략 적용 메서드들 ===
    
    def _apply_merge_strategy(self, groups: List[List[ContentItem]], strategy: str) -> Dict:
        """병합 전략 적용"""
        if strategy == 'comprehensive':
            return self._comprehensive_merge(groups)
        elif strategy == 'temporal':
            return self._temporal_merge(groups)
        elif strategy == 'importance_based':
            return self._importance_based_merge(groups)
        else:
            return self._comprehensive_merge(groups)
    
    def _comprehensive_merge(self, groups: List[List[ContentItem]]) -> Dict:
        """종합적 병합"""
        merged_sections = []
        
        for i, group in enumerate(groups):
            if not group:
                continue
            
            # 그룹의 대표 내용 결정
            primary_item = max(group, key=lambda x: x.importance_score)
            
            # 그룹 내용 통합
            merged_content = self._merge_group_content(group, primary_item)
            
            # 카테고리 분류
            category = self._classify_content_category(merged_content)
            
            section = {
                'section_id': f"section_{i+1}",
                'category': category,
                'content': merged_content,
                'source_types': list(set(item.source_type for item in group)),
                'timestamp_range': {
                    'start': min(item.timestamp for item in group),
                    'end': max(item.timestamp for item in group)
                },
                'importance_score': max(item.importance_score for item in group),
                'item_count': len(group)
            }
            
            merged_sections.append(section)
        
        return {
            'sections': merged_sections,
            'merge_type': 'comprehensive',
            'total_sections': len(merged_sections)
        }
    
    def _temporal_merge(self, groups: List[List[ContentItem]]) -> Dict:
        """시간 기반 병합"""
        # 모든 아이템을 시간순으로 정렬
        all_items = []
        for group in groups:
            all_items.extend(group)
        
        all_items.sort(key=lambda x: x.timestamp)
        
        # 시간 윈도우 기반 섹션 생성
        sections = []
        current_section_items = []
        section_start_time = None
        
        for item in all_items:
            if not current_section_items:
                current_section_items = [item]
                section_start_time = item.timestamp
            else:
                time_diff = item.timestamp - section_start_time
                if time_diff <= self.merge_config['time_window_seconds'] * 2:
                    current_section_items.append(item)
                else:
                    # 현재 섹션 완료
                    if current_section_items:
                        section = self._create_temporal_section(current_section_items)
                        sections.append(section)
                    
                    # 새 섹션 시작
                    current_section_items = [item]
                    section_start_time = item.timestamp
        
        # 마지막 섹션 추가
        if current_section_items:
            section = self._create_temporal_section(current_section_items)
            sections.append(section)
        
        return {
            'sections': sections,
            'merge_type': 'temporal',
            'total_sections': len(sections)
        }
    
    def _importance_based_merge(self, groups: List[List[ContentItem]]) -> Dict:
        """중요도 기반 병합"""
        # 모든 아이템을 중요도순으로 정렬
        all_items = []
        for group in groups:
            all_items.extend(group)
        
        all_items.sort(key=lambda x: x.importance_score, reverse=True)
        
        # 중요도별 섹션 생성
        high_importance = [item for item in all_items if item.importance_score >= 0.7]
        medium_importance = [item for item in all_items if 0.4 <= item.importance_score < 0.7]
        low_importance = [item for item in all_items if item.importance_score < 0.4]
        
        sections = []
        
        if high_importance:
            sections.append({
                'section_id': 'high_importance',
                'category': 'critical_information',
                'content': self._merge_items_content(high_importance),
                'importance_level': 'high',
                'item_count': len(high_importance)
            })
        
        if medium_importance:
            sections.append({
                'section_id': 'medium_importance',
                'category': 'supporting_information',
                'content': self._merge_items_content(medium_importance),
                'importance_level': 'medium',
                'item_count': len(medium_importance)
            })
        
        if low_importance:
            sections.append({
                'section_id': 'low_importance',
                'category': 'additional_information',
                'content': self._merge_items_content(low_importance),
                'importance_level': 'low',
                'item_count': len(low_importance)
            })
        
        return {
            'sections': sections,
            'merge_type': 'importance_based',
            'total_sections': len(sections)
        }
    
    def _merge_group_content(self, group: List[ContentItem], primary_item: ContentItem) -> str:
        """그룹 내용 병합"""
        if len(group) == 1:
            return group[0].processed_content
        
        # 주요 내용을 기반으로 보조 내용 추가
        merged_content = primary_item.processed_content
        
        for item in group:
            if item == primary_item:
                continue
            
            # 새로운 정보가 있는지 확인
            similarity = self._calculate_text_similarity(
                merged_content, item.processed_content
            )
            
            if similarity < 0.8:  # 충분히 다른 내용이면 추가
                merged_content += f" {item.processed_content}"
        
        return self._clean_content(merged_content)
    
    def _merge_items_content(self, items: List[ContentItem]) -> str:
        """아이템들의 내용 병합"""
        if not items:
            return ""
        
        if len(items) == 1:
            return items[0].processed_content
        
        # 중요도 순으로 정렬
        sorted_items = sorted(items, key=lambda x: x.importance_score, reverse=True)
        
        merged_content = sorted_items[0].processed_content
        
        for item in sorted_items[1:]:
            # 유사도 체크
            similarity = self._calculate_text_similarity(merged_content, item.processed_content)
            if similarity < 0.8:
                merged_content += f" {item.processed_content}"
        
        return self._clean_content(merged_content)
    
    def _create_temporal_section(self, items: List[ContentItem]) -> Dict:
        """시간 기반 섹션 생성"""
        if not items:
            return {}
        
        merged_content = self._merge_items_content(items)
        category = self._classify_content_category(merged_content)
        
        return {
            'section_id': f"temporal_{int(items[0].timestamp)}",
            'category': category,
            'content': merged_content,
            'source_types': list(set(item.source_type for item in items)),
            'timestamp_range': {
                'start': min(item.timestamp for item in items),
                'end': max(item.timestamp for item in items)
            },
            'item_count': len(items)
        }
    
    def _classify_content_category(self, content: str) -> str:
        """내용 카테고리 분류"""
        content_lower = content.lower()
        
        category_scores = {}
        for category, keywords in self.content_categories.items():
            score = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general_information'
    
    # === 최종 정리 메서드들 ===
    
    def _finalize_merged_content(self, merged_content: Dict) -> Dict:
        """병합된 내용 최종 정리"""
        # 섹션 재정렬 (중요도 및 시간순)
        sections = merged_content.get('sections', [])
        
        # 각 섹션의 품질 점수 계산
        for section in sections:
            section['quality_score'] = self._calculate_section_quality(section)
        
        # 중요도와 품질을 고려한 정렬
        sections.sort(key=lambda x: (
            x.get('importance_score', 0) * 0.6 + 
            x.get('quality_score', 0) * 0.4
        ), reverse=True)
        
        # 섹션 ID 재할당
        for i, section in enumerate(sections):
            section['section_id'] = f"final_section_{i+1}"
            section['order'] = i + 1
        
        merged_content['sections'] = sections
        merged_content['final_section_count'] = len(sections)
        
        return merged_content
    
    def _calculate_section_quality(self, section: Dict) -> float:
        """섹션 품질 점수 계산"""
        content = section.get('content', '')
        
        if not content:
            return 0.0
        
        quality_score = 0.0
        
        # 내용 길이 점수
        length_score = min(1.0, len(content) / 500)
        quality_score += length_score * 0.3
        
        # 아이템 수 점수
        item_count = section.get('item_count', 1)
        count_score = min(1.0, item_count / 5)
        quality_score += count_score * 0.2
        
        # 소스 다양성 점수
        source_types = section.get('source_types', [])
        diversity_score = len(source_types) / 3  # 최대 3개 소스
        quality_score += diversity_score * 0.2
        
        # 주얼리 용어 점수 (주얼리 모드인 경우)
        if self.jewelry_mode:
            jewelry_score = self._calculate_jewelry_term_bonus(content)
            quality_score += jewelry_score * 0.3
        else:
            quality_score += 0.3  # 기본 점수
        
        return min(1.0, quality_score)
    
    # === 메타데이터 생성 메서드들 ===
    
    def _generate_merge_metadata(self, original_items: List[ContentItem], final_result: Dict) -> Dict:
        """병합 메타데이터 생성"""
        metadata = {
            'merge_timestamp': datetime.now().isoformat(),
            'original_items': {
                'total_count': len(original_items),
                'audio_count': len([item for item in original_items if item.source_type == 'audio']),
                'image_count': len([item for item in original_items if item.source_type == 'image']),
                'document_count': len([item for item in original_items if item.source_type == 'document'])
            },
            'final_sections': {
                'total_count': len(final_result.get('sections', [])),
                'average_quality': np.mean([
                    section.get('quality_score', 0) 
                    for section in final_result.get('sections', [])
                ]) if final_result.get('sections') else 0
            },
            'compression_ratio': len(final_result.get('sections', [])) / len(original_items) if original_items else 0,
            'time_span': {
                'start': min(item.timestamp for item in original_items) if original_items else 0,
                'end': max(item.timestamp for item in original_items) if original_items else 0
            },
            'processing_statistics': {
                'duplicates_removed': len(original_items) - len(final_result.get('sections', [])),
                'categories_identified': len(set(
                    section.get('category', 'unknown') 
                    for section in final_result.get('sections', [])
                ))
            }
        }
        
        return metadata
    
    def _calculate_merge_quality_score(self, final_result: Dict) -> float:
        """병합 품질 점수 계산"""
        sections = final_result.get('sections', [])
        
        if not sections:
            return 0.0
        
        # 섹션별 품질 점수 평균
        quality_scores = [section.get('quality_score', 0) for section in sections]
        average_quality = np.mean(quality_scores) if quality_scores else 0
        
        # 추가 품질 요소들
        category_diversity = len(set(section.get('category', 'unknown') for section in sections))
        diversity_bonus = min(0.2, category_diversity / 5)  # 최대 20% 보너스
        
        total_score = average_quality + diversity_bonus
        
        return min(1.0, total_score)
    
    # === 실시간 병합 메서드들 ===
    
    def _check_similarity_with_existing(self, existing_content: Dict, new_item: ContentItem) -> Dict:
        """기존 내용과의 유사도 검사"""
        sections = existing_content.get('sections', [])
        
        if not sections:
            return {'should_merge': False, 'best_match': None, 'similarity': 0.0}
        
        best_similarity = 0.0
        best_match_section = None
        
        for section in sections:
            similarity = self._calculate_text_similarity(
                section.get('content', ''),
                new_item.processed_content
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_section = section
        
        should_merge = best_similarity >= self.merge_config['similarity_threshold']
        
        return {
            'should_merge': should_merge,
            'best_match': best_match_section,
            'similarity': best_similarity
        }
    
    def _merge_with_existing_section(self, existing_content: Dict, new_item: ContentItem, similarity_results: Dict) -> Dict:
        """기존 섹션에 병합"""
        best_match = similarity_results['best_match']
        
        if not best_match:
            return existing_content
        
        # 기존 내용과 새 내용 병합
        existing_text = best_match.get('content', '')
        new_text = new_item.processed_content
        
        # 유사도가 높지만 완전히 같지 않다면 추가 정보로 간주
        if similarity_results['similarity'] < 0.95:
            merged_text = f"{existing_text} {new_text}"
            best_match['content'] = self._clean_content(merged_text)
        
        # 메타데이터 업데이트
        if 'source_types' not in best_match:
            best_match['source_types'] = []
        
        if new_item.source_type not in best_match['source_types']:
            best_match['source_types'].append(new_item.source_type)
        
        # 시간 범위 업데이트
        if 'timestamp_range' not in best_match:
            best_match['timestamp_range'] = {'start': new_item.timestamp, 'end': new_item.timestamp}
        else:
            best_match['timestamp_range']['end'] = max(
                best_match['timestamp_range']['end'], 
                new_item.timestamp
            )
        
        return existing_content
    
    def _add_as_new_section(self, existing_content: Dict, new_item: ContentItem) -> Dict:
        """새 섹션으로 추가"""
        if 'sections' not in existing_content:
            existing_content['sections'] = []
        
        new_section = {
            'section_id': f"realtime_section_{len(existing_content['sections']) + 1}",
            'category': self._classify_content_category(new_item.processed_content),
            'content': new_item.processed_content,
            'source_types': [new_item.source_type],
            'timestamp_range': {
                'start': new_item.timestamp,
                'end': new_item.timestamp
            },
            'importance_score': new_item.importance_score,
            'item_count': 1,
            'quality_score': self._calculate_basic_importance(new_item)
        }
        
        existing_content['sections'].append(new_section)
        
        return existing_content
    
    def _update_realtime_metadata(self, content: Dict, new_item: ContentItem) -> None:
        """실시간 메타데이터 업데이트"""
        if 'metadata' not in content:
            content['metadata'] = {}
        
        metadata = content['metadata']
        
        # 마지막 업데이트 시간
        metadata['last_update'] = datetime.now().isoformat()
        
        # 총 아이템 수
        metadata['total_items'] = metadata.get('total_items', 0) + 1
        
        # 소스별 카운트
        source_key = f"{new_item.source_type}_count"
        metadata[source_key] = metadata.get(source_key, 0) + 1
    
    # === 인사이트 추출 메서드들 ===
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """주요 토픽 추출"""
        topics = []
        
        # 주얼리 관련 토픽 추출
        if self.jewelry_mode:
            for category, terms in self.jewelry_terms.items():
                found_terms = [term for term in terms if term.lower() in text.lower()]
                if found_terms:
                    topics.extend(found_terms[:3])  # 카테고리별 최대 3개
        
        # 일반 키워드 추출 (빈도 기반)
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(5) if count > 1]
        topics.extend(common_words)
        
        return list(set(topics))[:10]  # 최대 10개 토픽
    
    def _extract_important_facts(self, text: str) -> List[str]:
        """중요한 사실 추출"""
        facts = []
        
        # 숫자가 포함된 문장 (가격, 사양 등)
        number_sentences = re.findall(r'[^.!?]*\d+[^.!?]*[.!?]', text)
        facts.extend(number_sentences[:3])
        
        # 대문자로 시작하는 중요 문장
        important_sentences = re.findall(r'[A-Z][^.!?]{20,}[.!?]', text)
        facts.extend(important_sentences[:3])
        
        return facts[:5]  # 최대 5개 사실
    
    def _extract_jewelry_info(self, text: str) -> Dict:
        """주얼리 특화 정보 추출"""
        jewelry_info = {
            'materials': [],
            'quality_info': [],
            'products': [],
            'prices': [],
            'certifications': []
        }
        
        text_lower = text.lower()
        
        # 재료 정보
        for term in self.jewelry_terms['material_terms']:
            if term.lower() in text_lower:
                jewelry_info['materials'].append(term)
        
        # 품질 정보
        for term in self.jewelry_terms['quality_terms']:
            if term.lower() in text_lower:
                jewelry_info['quality_info'].append(term)
        
        # 제품 정보
        for term in self.jewelry_terms['product_terms']:
            if term.lower() in text_lower:
                jewelry_info['products'].append(term)
        
        # 가격 정보 추출
        price_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',
            r'₩[\d,]+',
            r'\d+\s*원',
            r'\d+\s*달러'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            jewelry_info['prices'].extend(matches)
        
        # 인증서 정보
        cert_keywords = ['GIA', 'AGS', 'certificate', 'certification', '감정서', '인증서']
        for keyword in cert_keywords:
            if keyword.lower() in text_lower:
                jewelry_info['certifications'].append(keyword)
        
        return jewelry_info
    
    def _extract_action_items(self, text: str) -> List[str]:
        """액션 아이템 추출"""
        action_items = []
        
        # 명령형 동사로 시작하는 문장
        action_patterns = [
            r'[Cc]heck [^.!?]*[.!?]',
            r'[Cc]onfirm [^.!?]*[.!?]',
            r'[Cc]ontact [^.!?]*[.!?]',
            r'[Ss]end [^.!?]*[.!?]',
            r'확인[^.!?]*[.!?]',
            r'연락[^.!?]*[.!?]',
            r'준비[^.!?]*[.!?]'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            action_items.extend(matches)
        
        # 질문 형태 (답변이 필요한 것들)
        question_patterns = [
            r'[^.!?]*\?',
            r'[^.!?]*인가\?',
            r'[^.!?]*입니까\?'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, text)
            action_items.extend(matches[:2])  # 최대 2개 질문
        
        return action_items[:5]  # 최대 5개 액션 아이템
    
    def _generate_smart_summary(self, merged_content: Dict) -> str:
        """스마트 요약 생성"""
        sections = merged_content.get('sections', [])
        
        if not sections:
            return "병합된 내용이 없습니다."
        
        # 가장 중요한 섹션들의 내용 추출
        important_sections = sorted(
            sections, 
            key=lambda x: x.get('importance_score', 0), 
            reverse=True
        )[:3]
        
        summary_parts = []
        
        for i, section in enumerate(important_sections):
            content = section.get('content', '')
            
            # 첫 번째 문장 또는 처음 100자 추출
            first_sentence = re.split(r'[.!?]', content)[0]
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:100] + "..."
            elif len(first_sentence) < len(content):
                first_sentence += "."
            
            summary_parts.append(f"{i+1}. {first_sentence}")
        
        return " ".join(summary_parts)
    
    def _calculate_insight_confidence(self, merged_content: Dict, insights: Dict) -> float:
        """인사이트 신뢰도 계산"""
        confidence = 0.0
        
        # 섹션 수 기반 신뢰도
        section_count = len(merged_content.get('sections', []))
        confidence += min(0.3, section_count / 10)
        
        # 소스 다양성 기반 신뢰도
        all_source_types = set()
        for section in merged_content.get('sections', []):
            all_source_types.update(section.get('source_types', []))
        
        source_diversity = len(all_source_types)
        confidence += min(0.3, source_diversity / 3)
        
        # 추출된 정보 풍부성 기반 신뢰도
        info_richness = (
            len(insights.get('key_topics', [])) +
            len(insights.get('important_facts', [])) +
            len(insights.get('action_items', []))
        )
        confidence += min(0.4, info_richness / 15)
        
        return min(1.0, confidence)


# 사용 예제
if __name__ == "__main__":
    merger = SmartContentMerger(jewelry_mode=True)
    
    print("🧠 Smart Content Merger v2.1 - 테스트 시작")
    print("=" * 50)
    
    # 테스트 데이터 생성
    test_items = [
        ContentItem(
            content="다이아몬드 반지의 품질은 4C로 평가됩니다. 캐럿, 컷, 투명도, 색상이 중요합니다.",
            source_type="audio",
            timestamp=1625097600,
            metadata={"confidence": 85.0}
        ),
        ContentItem(
            content="Diamond ring quality assessment: 4C evaluation - Carat, Cut, Clarity, Color",
            source_type="image",
            timestamp=1625097605,
            metadata={"ocr_confidence": 92.0}
        ),
        ContentItem(
            content="GIA 감정서에 따르면 이 다이아몬드는 1.5캐럿, VVS1 등급입니다.",
            source_type="document",
            timestamp=1625097610,
            metadata={"document_type": "certificate"}
        )
    ]
    
    # 병합 실행
    result = merger.merge_multiple_contents(test_items, merge_strategy='comprehensive')
    
    print(f"병합 결과:")
    print(f"- 원본 아이템 수: {result['original_count']}")
    print(f"- 최종 섹션 수: {result['final_count']}")
    print(f"- 병합 품질 점수: {result['quality_score']:.3f}")
    
    # 인사이트 추출
    insights = merger.extract_key_insights(result['merged_content'])
    print(f"- 추출된 키워드: {len(insights['key_topics'])}개")
    print(f"- 신뢰도: {insights['confidence_score']:.3f}")
    
    print("모듈 로드 완료 ✅")
