#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 실행 가능한 인사이트 추출기
Actionable Insights Extractor for Conference Analysis

핵심 목표: 복잡한 컨퍼런스 내용을 3줄 요약 + 5가지 구체적 액션 아이템으로 압축
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

try:
    from sentence_transformers import SentenceTransformer
    import spacy
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

@dataclass
class ThreeLineSummary:
    """3줄 요약 구조"""
    line1_what: str      # 무엇을 논의했는가?
    line2_why: str       # 왜 중요한가?
    line3_outcome: str   # 결과/결론은 무엇인가?
    confidence: float

@dataclass
class ActionItem:
    """액션 아이템 구조"""
    action_id: str
    title: str
    description: str
    priority: str        # high, medium, low
    owner: Optional[str]
    deadline: Optional[str]
    dependencies: List[str]
    success_criteria: str
    evidence_source: List[str]  # 근거가 된 fragment_ids

@dataclass
class ConferenceInsights:
    """컨퍼런스 인사이트 전체 구조"""
    conference_name: str
    analysis_date: str
    three_line_summary: ThreeLineSummary
    action_items: List[ActionItem]
    key_metrics: Dict[str, Any]
    risk_factors: List[str]
    success_indicators: List[str]
    stakeholder_map: Dict[str, List[str]]

class ActionableInsightsExtractor:
    """실행 가능한 인사이트 추출기"""
    
    def __init__(self, conference_name: str = "default"):
        self.conference_name = conference_name
        self.db_path = f"conference_analysis_{conference_name}.db"
        
        # NLP 모델 초기화
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_advanced_nlp = True
            except Exception as e:
                st.warning(f"고급 NLP 모델 초기화 실패: {e}")
                self.use_advanced_nlp = False
        else:
            self.use_advanced_nlp = False
        
        # 분석 데이터
        self.fragments = []
        self.speaker_profiles = {}
        self.topic_clusters = []
    
    def load_conference_data(self) -> bool:
        """컨퍼런스 데이터 로드"""
        try:
            self.fragments = self._load_fragments()
            if not self.fragments:
                return False
            
            self._build_speaker_profiles()
            self._build_topic_clusters()
            
            return True
            
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
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
    
    def _build_speaker_profiles(self):
        """발표자 프로필 구축"""
        speaker_groups = defaultdict(list)
        for fragment in self.fragments:
            speaker = fragment.get('speaker', 'Unknown')
            if speaker and speaker.strip():
                speaker_groups[speaker].append(fragment)
        
        self.speaker_profiles = speaker_groups
    
    def _build_topic_clusters(self):
        """주제 클러스터 구축"""
        # 키워드 기반 클러스터링
        all_keywords = []
        for fragment in self.fragments:
            all_keywords.extend(fragment.get('keywords', []))
        
        keyword_freq = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_freq.most_common(8)]
        
        self.topic_clusters = []
        for keyword in top_keywords:
            cluster_fragments = []
            for fragment in self.fragments:
                if keyword in fragment.get('keywords', []):
                    cluster_fragments.append(fragment)
            
            if len(cluster_fragments) >= 2:
                self.topic_clusters.append({
                    'cluster_name': keyword,
                    'fragments': cluster_fragments,
                    'importance': len(cluster_fragments) / len(self.fragments)
                })
        
        # 중요도순 정렬
        self.topic_clusters.sort(key=lambda t: t['importance'], reverse=True)
    
    def extract_actionable_insights(self) -> ConferenceInsights:
        """실행 가능한 인사이트 추출"""
        if not self.load_conference_data():
            raise ValueError("컨퍼런스 데이터를 로드할 수 없습니다.")
        
        # 1. 3줄 요약 생성
        three_line_summary = self._generate_three_line_summary()
        
        # 2. 5가지 액션 아이템 추출
        action_items = self._extract_action_items()
        
        # 3. 핵심 메트릭 계산
        key_metrics = self._calculate_key_metrics()
        
        # 4. 리스크 요소 식별
        risk_factors = self._identify_risk_factors()
        
        # 5. 성공 지표 정의
        success_indicators = self._define_success_indicators()
        
        # 6. 이해관계자 맵핑
        stakeholder_map = self._build_stakeholder_map()
        
        insights = ConferenceInsights(
            conference_name=self.conference_name,
            analysis_date=datetime.now().isoformat(),
            three_line_summary=three_line_summary,
            action_items=action_items,
            key_metrics=key_metrics,
            risk_factors=risk_factors,
            success_indicators=success_indicators,
            stakeholder_map=stakeholder_map
        )
        
        return insights
    
    def _generate_three_line_summary(self) -> ThreeLineSummary:
        """3줄 요약 생성"""
        # Line 1: 무엇을 논의했는가?
        if self.topic_clusters:
            top_topics = [cluster['cluster_name'] for cluster in self.topic_clusters[:3]]
            line1_what = f"{', '.join(top_topics)} 등 {len(self.topic_clusters)}개 주제에 대해 {len(self.fragments)}개 자료로 논의했습니다."
        else:
            line1_what = f"총 {len(self.fragments)}개의 자료를 통해 다양한 주제를 논의했습니다."
        
        # Line 2: 왜 중요한가?
        if self.speaker_profiles:
            participant_count = len(self.speaker_profiles)
            most_active_speaker = max(self.speaker_profiles.keys(), key=lambda s: len(self.speaker_profiles[s]))
            line2_why = f"{participant_count}명의 참여자가 활발히 의견을 교환했으며, {most_active_speaker} 등이 핵심 아젠다를 주도했습니다."
        else:
            line2_why = "다양한 관점에서 심도 있는 분석과 토론이 이루어졌습니다."
        
        # Line 3: 결과/결론은 무엇인가?
        avg_confidence = np.mean([f['confidence'] for f in self.fragments])
        high_confidence_count = sum(1 for f in self.fragments if f['confidence'] > 0.8)
        line3_outcome = f"분석 신뢰도 {avg_confidence:.1%}, 고품질 자료 {high_confidence_count}개 확보로 구체적인 실행 계획 수립이 가능합니다."
        
        return ThreeLineSummary(
            line1_what=line1_what,
            line2_why=line2_why,
            line3_outcome=line3_outcome,
            confidence=avg_confidence
        )
    
    def _extract_action_items(self) -> List[ActionItem]:
        """5가지 액션 아이템 추출"""
        action_items = []
        
        # 1. 주요 주제별 후속 조치
        for i, topic_cluster in enumerate(self.topic_clusters[:2]):
            action = ActionItem(
                action_id=f"topic_action_{i+1}",
                title=f"{topic_cluster['cluster_name']} 심화 분석",
                description=f"{topic_cluster['cluster_name']} 관련 논의 내용을 바탕으로 구체적인 실행 계획을 수립하고 세부 일정을 확정합니다.",
                priority="high" if topic_cluster['importance'] > 0.3 else "medium",
                owner=self._suggest_topic_owner(topic_cluster),
                deadline=self._suggest_deadline(7),  # 1주일
                dependencies=[],
                success_criteria=f"{topic_cluster['cluster_name']} 관련 실행 계획서 완성 및 이해관계자 승인",
                evidence_source=[f['fragment_id'] for f in topic_cluster['fragments']]
            )
            action_items.append(action)
        
        # 2. 이해관계자 후속 미팅
        if self.speaker_profiles:
            key_speakers = sorted(self.speaker_profiles.keys(), key=lambda s: len(self.speaker_profiles[s]), reverse=True)[:3]
            action = ActionItem(
                action_id="stakeholder_followup",
                title="핵심 참여자 후속 미팅",
                description=f"{', '.join(key_speakers)} 등 주요 참여자와 개별 후속 미팅을 진행하여 세부 사항을 논의합니다.",
                priority="high",
                owner=key_speakers[0] if key_speakers else None,
                deadline=self._suggest_deadline(5),  # 5일
                dependencies=[],
                success_criteria="모든 핵심 참여자와의 개별 미팅 완료 및 합의 사항 정리",
                evidence_source=[]
            )
            action_items.append(action)
        
        # 3. 문서화 및 정리
        action = ActionItem(
            action_id="documentation",
            title="회의 결과 문서화",
            description="논의된 모든 내용을 체계적으로 정리하고, 향후 참조 가능한 문서로 작성합니다.",
            priority="medium",
            owner=None,
            deadline=self._suggest_deadline(3),  # 3일
            dependencies=["stakeholder_followup"],
            success_criteria="완전한 회의록 및 실행 계획서 작성",
            evidence_source=[f['fragment_id'] for f in self.fragments]
        )
        action_items.append(action)
        
        # 4. 리스크 관리
        action = ActionItem(
            action_id="risk_management",
            title="잠재 리스크 관리 방안 수립",
            description="논의 과정에서 식별된 잠재적 위험 요소들에 대한 대응 방안을 마련합니다.",
            priority="medium",
            owner=None,
            deadline=self._suggest_deadline(10),  # 10일
            dependencies=["documentation"],
            success_criteria="리스크 관리 계획서 완성 및 대응 체계 구축",
            evidence_source=self._find_risk_related_fragments()
        )
        action_items.append(action)
        
        # 5. 성과 모니터링 체계 구축
        action = ActionItem(
            action_id="monitoring_system",
            title="성과 모니터링 체계 구축",
            description="합의된 사항들의 이행 상황을 지속적으로 모니터링할 수 있는 체계를 구축합니다.",
            priority="low",
            owner=None,
            deadline=self._suggest_deadline(14),  # 2주
            dependencies=["topic_action_1", "documentation"],
            success_criteria="KPI 정의 및 모니터링 대시보드 구축",
            evidence_source=[]
        )
        action_items.append(action)
        
        return action_items
    
    def _suggest_topic_owner(self, topic_cluster: Dict) -> Optional[str]:
        """주제별 담당자 추천"""
        # 해당 주제에서 가장 많이 발언한 사람
        speaker_counts = defaultdict(int)
        
        for fragment in topic_cluster['fragments']:
            speaker = fragment.get('speaker')
            if speaker:
                speaker_counts[speaker] += 1
        
        if speaker_counts:
            return max(speaker_counts, key=speaker_counts.get)
        return None
    
    def _suggest_deadline(self, days: int) -> str:
        """마감일 제안"""
        deadline = datetime.now() + timedelta(days=days)
        return deadline.strftime("%Y-%m-%d")
    
    def _find_risk_related_fragments(self) -> List[str]:
        """리스크 관련 조각 찾기"""
        risk_keywords = ['위험', '문제', '우려', '리스크', 'risk', 'issue', 'concern', '장애', '어려움']
        risk_fragments = []
        
        for fragment in self.fragments:
            content = fragment['content'].lower()
            if any(keyword in content for keyword in risk_keywords):
                risk_fragments.append(fragment['fragment_id'])
        
        return risk_fragments
    
    def _calculate_key_metrics(self) -> Dict[str, Any]:
        """핵심 메트릭 계산"""
        return {
            "total_fragments": len(self.fragments),
            "total_participants": len(self.speaker_profiles),
            "total_topics": len(self.topic_clusters),
            "average_confidence": np.mean([f['confidence'] for f in self.fragments]) if self.fragments else 0,
            "high_quality_fragments": sum(1 for f in self.fragments if f['confidence'] > 0.8),
            "analysis_completeness": min(1.0, len(self.fragments) / 10),  # 10개 이상이면 완전함
            "stakeholder_engagement": len(self.speaker_profiles) / max(1, len(self.fragments) / 5)  # 참여도
        }
    
    def _identify_risk_factors(self) -> List[str]:
        """리스크 요소 식별"""
        risks = []
        
        # 데이터 품질 리스크
        low_confidence_count = sum(1 for f in self.fragments if f['confidence'] < 0.5)
        if low_confidence_count > len(self.fragments) * 0.3:
            risks.append(f"⚠️ 분석 품질이 낮은 자료가 {low_confidence_count}개 있어 결론의 신뢰성에 영향을 줄 수 있습니다.")
        
        # 참여자 집중도 리스크
        if self.speaker_profiles:
            max_speaker_ratio = max(len(fragments) for fragments in self.speaker_profiles.values()) / len(self.fragments)
            if max_speaker_ratio > 0.6:
                risks.append("⚠️ 특정 인물의 발언 비중이 높아 다양한 관점이 부족할 수 있습니다.")
        
        # 주제 편중 리스크
        if self.topic_clusters:
            max_topic_ratio = self.topic_clusters[0]['importance']
            if max_topic_ratio > 0.5:
                risks.append("⚠️ 특정 주제에 논의가 집중되어 다른 중요 사안이 소홀히 다뤄졌을 수 있습니다.")
        
        # 소통 리스크
        if len(self.speaker_profiles) < 3:
            risks.append("⚠️ 참여자 수가 적어 충분한 토론과 검증이 이루어지지 않았을 가능성이 있습니다.")
        
        return risks if risks else ["✅ 특별한 위험 요소가 발견되지 않았습니다."]
    
    def _define_success_indicators(self) -> List[str]:
        """성공 지표 정의"""
        indicators = []
        
        # 실행률 지표
        indicators.append("📈 액션 아이템 완료율 80% 이상 달성")
        
        # 참여도 지표
        if self.speaker_profiles:
            indicators.append(f"👥 {len(self.speaker_profiles)}명 참여자 만족도 4.0/5.0 이상")
        
        # 성과 지표
        if self.topic_clusters:
            indicators.append(f"🎯 주요 {len(self.topic_clusters[:3])}개 주제별 구체적 성과 창출")
        
        # 프로세스 지표
        indicators.append("⏰ 합의된 일정 준수율 90% 이상")
        
        # 품질 지표
        indicators.append("📊 후속 미팅에서 참조 자료로 활용률 70% 이상")
        
        return indicators
    
    def _build_stakeholder_map(self) -> Dict[str, List[str]]:
        """이해관계자 맵핑"""
        stakeholder_map = {
            "핵심 의사결정자": [],
            "실행 담당자": [],
            "검토/승인자": [],
            "정보 공유 대상": []
        }
        
        if self.speaker_profiles:
            speakers = sorted(self.speaker_profiles.keys(), key=lambda s: len(self.speaker_profiles[s]), reverse=True)
            
            # 발언량 기준으로 역할 분류
            total_speakers = len(speakers)
            
            if total_speakers >= 4:
                stakeholder_map["핵심 의사결정자"] = speakers[:2]
                stakeholder_map["실행 담당자"] = speakers[2:4]
                stakeholder_map["검토/승인자"] = speakers[4:6] if len(speakers) > 4 else []
                stakeholder_map["정보 공유 대상"] = speakers[6:] if len(speakers) > 6 else []
            elif total_speakers >= 2:
                stakeholder_map["핵심 의사결정자"] = speakers[:1]
                stakeholder_map["실행 담당자"] = speakers[1:]
            else:
                stakeholder_map["핵심 의사결정자"] = speakers
        
        return stakeholder_map

# Streamlit UI
def main():
    st.title("🎯 실행 가능한 인사이트 추출기")
    st.markdown("**복잡한 컨퍼런스 내용을 3줄 요약과 5가지 구체적 액션 아이템으로 압축합니다**")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "my_conference")
    
    # 추출기 초기화
    extractor = ActionableInsightsExtractor(conference_name)
    
    # 인사이트 추출
    if st.button("🎯 실행 가능한 인사이트 추출"):
        with st.spinner("실행 가능한 인사이트를 추출하고 있습니다..."):
            try:
                insights = extractor.extract_actionable_insights()
                
                st.success("✅ 실행 가능한 인사이트 추출 완료!")
                
                # 3줄 요약
                st.markdown("## 📋 3줄 요약")
                summary = insights.three_line_summary
                
                st.markdown("### 1️⃣ 무엇을 논의했는가?")
                st.info(summary.line1_what)
                
                st.markdown("### 2️⃣ 왜 중요한가?")
                st.info(summary.line2_why)
                
                st.markdown("### 3️⃣ 결과/결론은 무엇인가?")
                st.info(summary.line3_outcome)
                
                st.metric("📊 요약 신뢰도", f"{summary.confidence:.1%}")
                
                # 5가지 액션 아이템
                st.markdown("## ✅ 5가지 액션 아이템")
                
                for i, action in enumerate(insights.action_items, 1):
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    priority_icon = priority_color.get(action.priority, "⚪")
                    
                    with st.expander(f"{priority_icon} {i}. {action.title} ({action.priority} 우선순위)"):
                        st.markdown(f"**설명:** {action.description}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**담당자:** {action.owner or '미정'}")
                            st.markdown(f"**마감일:** {action.deadline or '미정'}")
                        
                        with col2:
                            st.markdown(f"**의존성:** {', '.join(action.dependencies) if action.dependencies else '없음'}")
                            st.markdown(f"**근거 자료:** {len(action.evidence_source)}개")
                        
                        st.markdown(f"**성공 기준:** {action.success_criteria}")
                
                # 핵심 메트릭
                st.markdown("## 📊 핵심 메트릭")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("총 분석 자료", insights.key_metrics["total_fragments"])
                    st.metric("참여자 수", insights.key_metrics["total_participants"])
                
                with col2:
                    st.metric("논의 주제", insights.key_metrics["total_topics"])
                    st.metric("고품질 자료", insights.key_metrics["high_quality_fragments"])
                
                with col3:
                    st.metric("평균 신뢰도", f"{insights.key_metrics['average_confidence']:.1%}")
                    st.metric("분석 완성도", f"{insights.key_metrics['analysis_completeness']:.1%}")
                
                # 리스크 요소
                st.markdown("## ⚠️ 주의사항")
                for risk in insights.risk_factors:
                    st.markdown(f"- {risk}")
                
                # 성공 지표
                st.markdown("## 🎯 성공 지표")
                for indicator in insights.success_indicators:
                    st.markdown(f"- {indicator}")
                
                # 이해관계자 맵
                st.markdown("## 👥 이해관계자 맵")
                for role, members in insights.stakeholder_map.items():
                    if members:
                        st.markdown(f"**{role}:** {', '.join(members)}")
                
                # 상세 정보
                with st.expander("📊 상세 분석 정보"):
                    st.json(asdict(insights))
                
            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ 인사이트 추출 중 오류 발생: {e}")
    
    st.markdown("---")
    st.markdown("**💡 사용법:** 컨퍼런스 분석이 완료된 후 이 시스템을 실행하여 실행 가능한 인사이트를 확인하세요.")

if __name__ == "__main__":
    main()