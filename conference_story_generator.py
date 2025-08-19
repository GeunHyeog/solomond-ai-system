#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📖 컨퍼런스 스토리 생성 엔진
Conference Story Generator for Holistic Analysis

핵심 목표: 분산된 정보를 하나의 일관된 컨퍼런스 스토리로 재구성
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from dataclasses import dataclass, asdict
import re
from collections import defaultdict, Counter
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    import spacy
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

@dataclass
class StorySegment:
    """스토리 세그먼트"""
    segment_id: str
    segment_type: str  # opening, main_topic, discussion, conclusion, action_items
    title: str
    content: str
    participants: List[str]
    key_points: List[str]
    source_fragments: List[str]
    timeline_position: int
    importance_score: float

@dataclass
class ConferenceNarrative:
    """컨퍼런스 내러티브 전체 구조"""
    conference_name: str
    narrative_summary: str
    key_takeaways: List[str]
    story_segments: List[StorySegment]
    participant_journey: Dict[str, List[str]]
    decision_points: List[str]
    unresolved_issues: List[str]
    next_steps: List[str]
    confidence_score: float

class ConferenceStoryGenerator:
    """컨퍼런스 스토리 생성 엔진"""
    
    def __init__(self, conference_name: str = "default"):
        self.conference_name = conference_name
        self.db_path = f"conference_analysis_{conference_name}.db"
        
        # NLP 모델 초기화
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                
                # SpaCy 모델
                try:
                    self.nlp = spacy.load("ko_core_news_sm")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        self.nlp = spacy.blank("ko")
                
                self.use_advanced_nlp = True
            except Exception as e:
                st.warning(f"고급 NLP 모델 초기화 실패: {e}")
                self.use_advanced_nlp = False
        else:
            self.use_advanced_nlp = False
        
        # 스토리 구성 요소
        self.fragments = []
        self.connections = []
        self.speaker_profiles = {}
        self.topic_clusters = []
        self.story_segments = []
    
    def load_analysis_data(self) -> bool:
        """분석 데이터 로드"""
        try:
            # 조각 데이터 로드
            self.fragments = self._load_fragments()
            if not self.fragments:
                return False
            
            # 연결 데이터 로드 (의미적 연결 엔진에서 생성)
            # 실제로는 데이터베이스에서 로드하지만, 여기서는 기본 연결 생성
            self._generate_basic_connections()
            
            return True
            
        except Exception as e:
            st.error(f"분석 데이터 로드 실패: {e}")
            return False
    
    def _load_fragments(self) -> List[Dict]:
        """조각 데이터 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM fragments ORDER BY created_at')
            rows = cursor.fetchall()
            
            fragments = []
            for row in rows:
                fragment = {
                    'fragment_id': row[0],
                    'file_source': row[1],
                    'file_type': row[2],
                    'timestamp': row[3],
                    'speaker': row[4],
                    'content': row[5],
                    'confidence': row[6],
                    'keywords': json.loads(row[7]) if row[7] else []
                }
                fragments.append(fragment)
            
            return fragments
            
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
    
    def _generate_basic_connections(self):
        """기본 연결 생성 (의미적 연결 엔진 결과가 없는 경우)"""
        # 발표자별 그룹핑
        speaker_groups = defaultdict(list)
        for fragment in self.fragments:
            speaker = fragment.get('speaker', 'Unknown')
            speaker_groups[speaker].append(fragment)
        
        self.speaker_profiles = speaker_groups
        
        # 주제별 간단한 클러스터링 (키워드 기반)
        self._cluster_by_keywords()
    
    def _cluster_by_keywords(self):
        """키워드 기반 간단한 클러스터링"""
        # 모든 키워드 수집
        all_keywords = []
        for fragment in self.fragments:
            all_keywords.extend(fragment.get('keywords', []))
        
        # 주요 키워드 추출
        keyword_freq = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_freq.most_common(10)]
        
        # 키워드별 조각 그룹핑
        for keyword in top_keywords:
            cluster_fragments = []
            for fragment in self.fragments:
                if keyword in fragment.get('keywords', []):
                    cluster_fragments.append(fragment['fragment_id'])
            
            if len(cluster_fragments) >= 2:
                self.topic_clusters.append({
                    'cluster_name': keyword,
                    'fragments': cluster_fragments,
                    'keyword': keyword
                })
    
    def generate_conference_story(self) -> ConferenceNarrative:
        """컨퍼런스 스토리 생성"""
        if not self.load_analysis_data():
            raise ValueError("분석 데이터를 로드할 수 없습니다.")
        
        # 1. 스토리 세그먼트 생성
        story_segments = self._create_story_segments()
        
        # 2. 전체 내러티브 구성
        narrative_summary = self._generate_narrative_summary()
        
        # 3. 핵심 내용 추출
        key_takeaways = self._extract_key_takeaways()
        
        # 4. 참여자 여정 추적
        participant_journey = self._trace_participant_journey()
        
        # 5. 의사결정 포인트 식별
        decision_points = self._identify_decision_points()
        
        # 6. 미해결 이슈 추출
        unresolved_issues = self._extract_unresolved_issues()
        
        # 7. 다음 단계 제안
        next_steps = self._generate_next_steps()
        
        # 8. 신뢰도 계산
        confidence_score = self._calculate_confidence_score()
        
        narrative = ConferenceNarrative(
            conference_name=self.conference_name,
            narrative_summary=narrative_summary,
            key_takeaways=key_takeaways,
            story_segments=story_segments,
            participant_journey=participant_journey,
            decision_points=decision_points,
            unresolved_issues=unresolved_issues,
            next_steps=next_steps,
            confidence_score=confidence_score
        )
        
        return narrative
    
    def _create_story_segments(self) -> List[StorySegment]:
        """스토리 세그먼트 생성"""
        segments = []
        
        # 1. 오프닝 세그먼트
        opening_fragments = self._identify_opening_fragments()
        if opening_fragments:
            opening_segment = StorySegment(
                segment_id="opening",
                segment_type="opening",
                title="컨퍼런스 시작",
                content=self._summarize_fragments(opening_fragments),
                participants=self._extract_participants(opening_fragments),
                key_points=self._extract_key_points(opening_fragments),
                source_fragments=[f['fragment_id'] for f in opening_fragments],
                timeline_position=1,
                importance_score=0.8
            )
            segments.append(opening_segment)
        
        # 2. 주요 주제별 세그먼트
        for i, topic_cluster in enumerate(self.topic_clusters):
            topic_fragments = [f for f in self.fragments if f['fragment_id'] in topic_cluster['fragments']]
            
            if topic_fragments:
                topic_segment = StorySegment(
                    segment_id=f"topic_{i}",
                    segment_type="main_topic",
                    title=f"주제 논의: {topic_cluster['cluster_name']}",
                    content=self._summarize_fragments(topic_fragments),
                    participants=self._extract_participants(topic_fragments),
                    key_points=self._extract_key_points(topic_fragments),
                    source_fragments=[f['fragment_id'] for f in topic_fragments],
                    timeline_position=i + 2,
                    importance_score=len(topic_fragments) / len(self.fragments)
                )
                segments.append(topic_segment)
        
        # 3. 토론/질의응답 세그먼트
        discussion_fragments = self._identify_discussion_fragments()
        if discussion_fragments:
            discussion_segment = StorySegment(
                segment_id="discussion",
                segment_type="discussion",
                title="토론 및 질의응답",
                content=self._summarize_fragments(discussion_fragments),
                participants=self._extract_participants(discussion_fragments),
                key_points=self._extract_key_points(discussion_fragments),
                source_fragments=[f['fragment_id'] for f in discussion_fragments],
                timeline_position=len(segments) + 1,
                importance_score=0.9
            )
            segments.append(discussion_segment)
        
        # 4. 결론/정리 세그먼트
        conclusion_fragments = self._identify_conclusion_fragments()
        if conclusion_fragments:
            conclusion_segment = StorySegment(
                segment_id="conclusion",
                segment_type="conclusion",
                title="결론 및 정리",
                content=self._summarize_fragments(conclusion_fragments),
                participants=self._extract_participants(conclusion_fragments),
                key_points=self._extract_key_points(conclusion_fragments),
                source_fragments=[f['fragment_id'] for f in conclusion_fragments],
                timeline_position=len(segments) + 1,
                importance_score=1.0
            )
            segments.append(conclusion_segment)
        
        return segments
    
    def _identify_opening_fragments(self) -> List[Dict]:
        """오프닝 조각 식별"""
        opening_keywords = ['시작', '소개', '안녕', '환영', 'welcome', 'introduction', '개회']
        opening_fragments = []
        
        for fragment in self.fragments[:3]:  # 처음 3개 조각에서 찾기
            content = fragment['content'].lower()
            if any(keyword in content for keyword in opening_keywords):
                opening_fragments.append(fragment)
        
        return opening_fragments if opening_fragments else self.fragments[:1]
    
    def _identify_discussion_fragments(self) -> List[Dict]:
        """토론 조각 식별"""
        discussion_keywords = ['질문', '답변', '토론', '의견', 'question', 'discussion', '어떻게', '왜']
        discussion_fragments = []
        
        for fragment in self.fragments:
            content = fragment['content'].lower()
            if any(keyword in content for keyword in discussion_keywords):
                discussion_fragments.append(fragment)
        
        return discussion_fragments
    
    def _identify_conclusion_fragments(self) -> List[Dict]:
        """결론 조각 식별"""
        conclusion_keywords = ['결론', '정리', '마무리', '끝', 'conclusion', 'summary', '다음', '앞으로']
        conclusion_fragments = []
        
        for fragment in self.fragments[-3:]:  # 마지막 3개 조각에서 찾기
            content = fragment['content'].lower()
            if any(keyword in content for keyword in conclusion_keywords):
                conclusion_fragments.append(fragment)
        
        return conclusion_fragments if conclusion_fragments else self.fragments[-1:]
    
    def _summarize_fragments(self, fragments: List[Dict]) -> str:
        """조각들을 요약"""
        if not fragments:
            return "관련 내용이 없습니다."
        
        # 모든 내용 결합
        all_content = " ".join([f['content'] for f in fragments if f['content']])
        
        # 길이 제한 (1000자)
        if len(all_content) > 1000:
            all_content = all_content[:1000] + "..."
        
        return all_content
    
    def _extract_participants(self, fragments: List[Dict]) -> List[str]:
        """참여자 추출"""
        participants = set()
        for fragment in fragments:
            if fragment.get('speaker') and fragment['speaker'].strip():
                participants.add(fragment['speaker'])
        
        return list(participants)
    
    def _extract_key_points(self, fragments: List[Dict]) -> List[str]:
        """핵심 포인트 추출"""
        all_keywords = []
        for fragment in fragments:
            all_keywords.extend(fragment.get('keywords', []))
        
        # 빈도수 기반 상위 키워드
        keyword_freq = Counter(all_keywords)
        key_points = [f"• {kw}" for kw, _ in keyword_freq.most_common(5)]
        
        return key_points
    
    def _generate_narrative_summary(self) -> str:
        """전체 내러티브 요약 생성"""
        summary_parts = []
        
        # 기본 정보
        summary_parts.append(f"📅 **{self.conference_name} 컨퍼런스**")
        summary_parts.append(f"총 {len(self.fragments)}개의 자료가 분석되었으며, {len(self.speaker_profiles)}명의 참여자와 {len(self.topic_clusters)}개의 주요 주제가 논의되었습니다.")
        summary_parts.append("")
        
        # 주요 흐름
        summary_parts.append("**📖 컨퍼런스 흐름:**")
        
        if self.story_segments:
            for segment in sorted(self.story_segments, key=lambda s: s.timeline_position):
                summary_parts.append(f"{segment.timeline_position}. **{segment.title}** - {len(segment.participants)}명 참여")
        
        summary_parts.append("")
        
        # 전체적인 특징
        if self.topic_clusters:
            top_topic = max(self.topic_clusters, key=lambda t: len(t['fragments']))
            summary_parts.append(f"가장 집중적으로 논의된 주제는 '{top_topic['cluster_name']}'였으며, ")
        
        if self.speaker_profiles:
            most_active_speaker = max(self.speaker_profiles.keys(), key=lambda s: len(self.speaker_profiles[s]))
            summary_parts.append(f"'{most_active_speaker}'가 가장 활발하게 발언했습니다.")
        
        return "\n".join(summary_parts)
    
    def _extract_key_takeaways(self) -> List[str]:
        """핵심 결과 추출"""
        takeaways = []
        
        # 가장 중요한 주제들
        if self.topic_clusters:
            top_topics = sorted(self.topic_clusters, key=lambda t: len(t['fragments']), reverse=True)[:3]
            for i, topic in enumerate(top_topics, 1):
                takeaways.append(f"{i}. {topic['cluster_name']}에 대한 심도 있는 논의가 이루어졌습니다.")
        
        # 참여자 활동도
        if self.speaker_profiles:
            active_speakers = sorted(self.speaker_profiles.keys(), key=lambda s: len(self.speaker_profiles[s]), reverse=True)[:2]
            takeaways.append(f"주요 발표자는 {', '.join(active_speakers)}였습니다.")
        
        # 전체적인 성과
        avg_confidence = np.mean([f['confidence'] for f in self.fragments]) if self.fragments else 0
        takeaways.append(f"분석 품질이 {avg_confidence:.1%}로 높은 신뢰도를 보입니다.")
        
        return takeaways
    
    def _trace_participant_journey(self) -> Dict[str, List[str]]:
        """참여자별 여정 추적"""
        journey = {}
        
        for speaker, fragments in self.speaker_profiles.items():
            speaker_journey = []
            
            # 발언 순서대로 정렬
            sorted_fragments = sorted(fragments, key=lambda f: f.get('timestamp', ''))
            
            for fragment in sorted_fragments[:5]:  # 최대 5개 발언
                content_preview = fragment['content'][:50] + "..." if len(fragment['content']) > 50 else fragment['content']
                speaker_journey.append(content_preview)
            
            journey[speaker] = speaker_journey
        
        return journey
    
    def _identify_decision_points(self) -> List[str]:
        """의사결정 포인트 식별"""
        decision_keywords = ['결정', '선택', '채택', '승인', '합의', 'decision', 'agree', '정하']
        decision_points = []
        
        for fragment in self.fragments:
            content = fragment['content'].lower()
            if any(keyword in content for keyword in decision_keywords):
                decision_preview = fragment['content'][:100] + "..." if len(fragment['content']) > 100 else fragment['content']
                decision_points.append(f"• {decision_preview}")
        
        return decision_points[:5]  # 최대 5개
    
    def _extract_unresolved_issues(self) -> List[str]:
        """미해결 이슈 추출"""
        issue_keywords = ['문제', '이슈', '해결', '검토', '고민', 'issue', 'problem', '추후', '나중']
        unresolved_issues = []
        
        for fragment in self.fragments:
            content = fragment['content'].lower()
            if any(keyword in content for keyword in issue_keywords):
                issue_preview = fragment['content'][:100] + "..." if len(fragment['content']) > 100 else fragment['content']
                unresolved_issues.append(f"• {issue_preview}")
        
        return unresolved_issues[:5]  # 최대 5개
    
    def _generate_next_steps(self) -> List[str]:
        """다음 단계 제안"""
        next_steps = []
        
        # 주요 주제별 후속 조치
        for topic in self.topic_clusters[:3]:
            next_steps.append(f"📋 {topic['cluster_name']} 관련 상세 계획 수립")
        
        # 참여자별 후속 조치
        if self.speaker_profiles:
            next_steps.append(f"📞 주요 참여자 {len(self.speaker_profiles)}명과 개별 후속 미팅")
        
        # 일반적인 후속 조치
        next_steps.extend([
            "📝 회의록 정리 및 배포",
            "📅 다음 미팅 일정 조율",
            "🎯 실행 계획 구체화"
        ])
        
        return next_steps[:5]  # 최대 5개
    
    def _calculate_confidence_score(self) -> float:
        """전체 신뢰도 계산"""
        if not self.fragments:
            return 0.0
        
        # 개별 조각 신뢰도 평균
        fragment_confidence = np.mean([f['confidence'] for f in self.fragments])
        
        # 연결성 보너스 (더 많은 연결이 있으면 더 신뢰할 만함)
        connection_bonus = min(0.2, len(self.topic_clusters) * 0.05)
        
        # 참여자 다양성 보너스
        diversity_bonus = min(0.1, len(self.speaker_profiles) * 0.02)
        
        total_confidence = fragment_confidence + connection_bonus + diversity_bonus
        
        return min(1.0, total_confidence)

# Streamlit UI
def main():
    st.title("📖 컨퍼런스 스토리 생성 엔진")
    st.markdown("**분산된 정보를 하나의 일관된 컨퍼런스 스토리로 재구성합니다**")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "my_conference")
    
    # 생성기 초기화
    generator = ConferenceStoryGenerator(conference_name)
    
    # 스토리 생성
    if st.button("📖 컨퍼런스 스토리 생성"):
        with st.spinner("컨퍼런스 스토리를 생성하고 있습니다..."):
            try:
                narrative = generator.generate_conference_story()
                
                st.success("✅ 컨퍼런스 스토리 생성 완료!")
                
                # 신뢰도 표시
                st.metric("📊 스토리 신뢰도", f"{narrative.confidence_score:.1%}")
                
                # 전체 내러티브
                st.markdown("## 📖 컨퍼런스 전체 스토리")
                st.markdown(narrative.narrative_summary)
                
                # 핵심 결과
                st.markdown("## 🎯 핵심 결과")
                for takeaway in narrative.key_takeaways:
                    st.markdown(f"- {takeaway}")
                
                # 스토리 세그먼트
                st.markdown("## 📚 상세 스토리 구성")
                for segment in sorted(narrative.story_segments, key=lambda s: s.timeline_position):
                    with st.expander(f"🔸 {segment.title} (중요도: {segment.importance_score:.1%})"):
                        st.markdown(f"**참여자:** {', '.join(segment.participants) if segment.participants else '없음'}")
                        st.markdown(f"**핵심 포인트:**")
                        for point in segment.key_points:
                            st.markdown(point)
                        st.markdown(f"**내용:**")
                        st.markdown(segment.content)
                
                # 참여자 여정
                if narrative.participant_journey:
                    st.markdown("## 👥 참여자별 여정")
                    for participant, journey in narrative.participant_journey.items():
                        with st.expander(f"🎤 {participant}"):
                            for i, step in enumerate(journey, 1):
                                st.markdown(f"{i}. {step}")
                
                # 의사결정 포인트
                if narrative.decision_points:
                    st.markdown("## ⚖️ 주요 의사결정")
                    for decision in narrative.decision_points:
                        st.markdown(decision)
                
                # 미해결 이슈
                if narrative.unresolved_issues:
                    st.markdown("## ❓ 미해결 이슈")
                    for issue in narrative.unresolved_issues:
                        st.markdown(issue)
                
                # 다음 단계
                st.markdown("## ➡️ 다음 단계")
                for step in narrative.next_steps:
                    st.markdown(f"- {step}")
                
                # 상세 정보
                with st.expander("📊 상세 분석 정보"):
                    st.json(asdict(narrative))
                
            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ 스토리 생성 중 오류 발생: {e}")
    
    st.markdown("---")
    st.markdown("**💡 사용법:** 먼저 홀리스틱 컨퍼런스 분석기와 의미적 연결 엔진을 실행한 후 이 시스템을 사용하세요.")

if __name__ == "__main__":
    main()