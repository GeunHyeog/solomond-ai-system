#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗃️ 홀리스틱 컨퍼런스 분석기 (Supabase 지원)
Holistic Conference Analyzer with Supabase Support
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from dataclasses import dataclass, asdict
import re
from collections import defaultdict, Counter

# 데이터베이스 어댑터 임포트
from database_adapter import DatabaseFactory, DatabaseInterface

try:
    from sentence_transformers import SentenceTransformer
    import spacy
    # 🛡️ 안전한 모델 로딩 시스템
    from defensive_model_loader import safe_sentence_transformer_load, enable_defensive_mode
    enable_defensive_mode()
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    safe_sentence_transformer_load = None

@dataclass
class ConferenceFragment:
    """컨퍼런스 조각"""
    fragment_id: str
    file_source: str
    file_type: str
    timestamp: str
    speaker: Optional[str]
    content: str
    confidence: float
    keywords: List[str]
    embedding: Optional[np.ndarray] = None

@dataclass
class ConferenceEntity:
    """컨퍼런스 개체"""
    entity_id: str
    entity_type: str  # person, topic, decision, action_item
    name: str
    mentions: List[str]  # fragment_ids
    importance_score: float
    relationships: List[str]  # 연관 entity_ids

class HolisticConferenceAnalyzerSupabase:
    """Supabase 지원 홀리스틱 컨퍼런스 분석기"""
    
    def __init__(self, conference_name: str = "default", db_type: str = "auto"):
        self.conference_name = conference_name
        
        # 데이터베이스 초기화
        self.db: DatabaseInterface = DatabaseFactory.create_database(db_type, conference_name)
        
        # NLP 모델 초기화
        if ADVANCED_NLP_AVAILABLE:
            try:
                # 🛡️ 안전한 모델 로딩으로 meta tensor 문제 완전 방지
                self.embedder = safe_sentence_transformer_load('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_advanced_nlp = True
            except Exception as e:
                st.warning(f"고급 NLP 모델 초기화 실패: {e}")
                self.use_advanced_nlp = False
        else:
            self.use_advanced_nlp = False
        
        # 분석 데이터
        self.fragments: List[ConferenceFragment] = []
        self.entities: List[ConferenceEntity] = []
        self.topics: List[Dict[str, Any]] = []
    
    def check_database_connection(self) -> Dict[str, Any]:
        """데이터베이스 연결 상태 확인"""
        db_type = type(self.db).__name__
        is_connected = self.db.is_connected()
        
        if is_connected:
            count = self.db.get_fragment_count(self.conference_name)
            return {
                "connected": True,
                "database_type": db_type,
                "fragment_count": count,
                "message": f"{db_type}에서 {count}개 조각 발견"
            }
        else:
            return {
                "connected": False,
                "database_type": db_type,
                "fragment_count": 0,
                "message": f"{db_type} 연결 실패 또는 데이터 없음"
            }
    
    def load_fragments_from_database(self) -> bool:
        """데이터베이스에서 조각 로드"""
        try:
            fragment_data = self.db.get_fragments(self.conference_name)
            
            self.fragments = []
            for data in fragment_data:
                fragment = ConferenceFragment(
                    fragment_id=data['fragment_id'],
                    file_source=data['file_source'],
                    file_type=data['file_type'],
                    timestamp=data['timestamp'],
                    speaker=data['speaker'],
                    content=data['content'],
                    confidence=data['confidence'],
                    keywords=data['keywords'] if isinstance(data['keywords'], list) else []
                )
                self.fragments.append(fragment)
            
            return len(self.fragments) > 0
            
        except Exception as e:
            st.error(f"조각 로드 실패: {e}")
            return False
    
    def create_sample_data_if_empty(self) -> bool:
        """데이터가 없으면 샘플 데이터 생성"""
        if self.db.get_fragment_count(self.conference_name) > 0:
            return True
        
        # 테이블 생성
        self.db.create_fragments_table()
        
        # 샘플 데이터 생성
        sample_fragments = [
            {
                'fragment_id': f'{self.conference_name}_001',
                'file_source': 'presentation_intro.jpg',
                'file_type': 'image',
                'timestamp': datetime.now().isoformat(),
                'speaker': '김대표',
                'content': 'AI 기술 동향 컨퍼런스에 오신 여러분을 환영합니다. 오늘은 인공지능의 최신 발전사항과 비즈니스 적용 방안에 대해 논의하겠습니다.',
                'confidence': 0.92,
                'keywords': ["AI", "기술", "동향", "컨퍼런스", "인공지능", "비즈니스"]
            },
            {
                'fragment_id': f'{self.conference_name}_002',
                'file_source': 'discussion_audio.m4a',
                'file_type': 'audio',
                'timestamp': datetime.now().isoformat(),
                'speaker': '박연구원',
                'content': 'ChatGPT와 GPT-4의 등장으로 자연어 처리 분야가 혁신되고 있습니다. 특히 대화형 AI의 성능이 놀라울 정도로 향상되었죠.',
                'confidence': 0.88,
                'keywords': ["ChatGPT", "GPT-4", "자연어", "처리", "대화형", "AI", "성능"]
            },
            {
                'fragment_id': f'{self.conference_name}_003',
                'file_source': 'technical_slide.png',
                'file_type': 'image',
                'timestamp': datetime.now().isoformat(),
                'speaker': '이개발자',
                'content': '머신러닝 모델 개발에서 가장 중요한 것은 데이터 품질입니다. 좋은 데이터 없이는 좋은 모델을 만들 수 없어요.',
                'confidence': 0.85,
                'keywords': ["머신러닝", "모델", "개발", "데이터", "품질"]
            },
            {
                'fragment_id': f'{self.conference_name}_004',
                'file_source': 'qa_session.wav',
                'file_type': 'audio',
                'timestamp': datetime.now().isoformat(),
                'speaker': '최질문자',
                'content': '실제 서비스에 AI를 도입할 때 어떤 점들을 주의해야 할까요? 특히 확장성과 비용 측면에서 고려사항이 궁금합니다.',
                'confidence': 0.90,
                'keywords': ["서비스", "AI", "도입", "확장성", "비용", "고려사항"]
            },
            {
                'fragment_id': f'{self.conference_name}_005',
                'file_source': 'business_discussion.mp4',
                'file_type': 'video',
                'timestamp': datetime.now().isoformat(),
                'speaker': '정매니저',
                'content': 'AI 도입 ROI 분석 결과, 초기 비용은 높지만 장기적으로는 30% 이상의 효율성 증대를 기대할 수 있습니다.',
                'confidence': 0.87,
                'keywords': ["AI", "도입", "ROI", "분석", "비용", "효율성"]
            }
        ]
        
        if self.db.insert_fragments_batch(sample_fragments):
            st.success(f"✅ {len(sample_fragments)}개 샘플 조각 생성 완료")
            return True
        else:
            st.error("❌ 샘플 데이터 생성 실패")
            return False
    
    def analyze_conference_holistically(self) -> Dict[str, Any]:
        """홀리스틱 컨퍼런스 분석 실행"""
        try:
            # 1. 데이터베이스에서 조각 로드
            if not self.load_fragments_from_database():
                # 데이터가 없으면 샘플 데이터 생성
                if not self.create_sample_data_if_empty():
                    return {"error": "데이터 로드 및 생성 실패"}
                # 다시 로드
                if not self.load_fragments_from_database():
                    return {"error": "샘플 데이터 로드 실패"}
            
            # 2. 개체 추출
            self._extract_entities()
            
            # 3. 주제 클러스터링
            self._perform_topic_clustering()
            
            # 4. 전체 통계 계산
            total_fragments = len(self.fragments)
            total_entities = len(self.entities)
            total_topics = len(self.topics)
            
            # 5. 발표자별 분석
            speaker_analysis = self._analyze_speakers()
            
            # 6. 신뢰도 계산
            avg_confidence = np.mean([f.confidence for f in self.fragments]) if self.fragments else 0
            
            # 7. 시간대별 분석
            temporal_analysis = self._analyze_temporal_patterns()
            
            results = {
                "conference_name": self.conference_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "database_type": type(self.db).__name__,
                "total_fragments": total_fragments,
                "total_entities": total_entities,
                "total_topics": total_topics,
                "average_confidence": avg_confidence,
                "speaker_analysis": speaker_analysis,
                "temporal_analysis": temporal_analysis,
                "topic_distribution": self.topics,
                "key_insights": self._generate_key_insights()
            }
            
            return results
            
        except Exception as e:
            return {"error": f"홀리스틱 분석 실패: {e}"}
    
    def _extract_entities(self):
        """개체 추출"""
        # 발표자 개체
        speakers = {}
        for fragment in self.fragments:
            if fragment.speaker:
                if fragment.speaker not in speakers:
                    speakers[fragment.speaker] = []
                speakers[fragment.speaker].append(fragment.fragment_id)
        
        # 발표자를 개체로 변환
        for speaker, mentions in speakers.items():
            entity = ConferenceEntity(
                entity_id=f"speaker_{speaker}",
                entity_type="person",
                name=speaker,
                mentions=mentions,
                importance_score=len(mentions) / len(self.fragments),
                relationships=[]
            )
            self.entities.append(entity)
        
        # 키워드 기반 주제 개체
        all_keywords = []
        for fragment in self.fragments:
            all_keywords.extend(fragment.keywords)
        
        keyword_freq = Counter(all_keywords)
        for keyword, freq in keyword_freq.most_common(10):
            if freq >= 2:  # 2번 이상 언급된 키워드만
                mentions = []
                for fragment in self.fragments:
                    if keyword in fragment.keywords:
                        mentions.append(fragment.fragment_id)
                
                entity = ConferenceEntity(
                    entity_id=f"topic_{keyword}",
                    entity_type="topic",
                    name=keyword,
                    mentions=mentions,
                    importance_score=freq / len(self.fragments),
                    relationships=[]
                )
                self.entities.append(entity)
    
    def _perform_topic_clustering(self):
        """주제 클러스터링"""
        # 키워드 기반 간단한 클러스터링
        all_keywords = []
        for fragment in self.fragments:
            all_keywords.extend(fragment.keywords)
        
        keyword_freq = Counter(all_keywords)
        
        for keyword, freq in keyword_freq.most_common(8):
            if freq >= 2:
                cluster_fragments = []
                for fragment in self.fragments:
                    if keyword in fragment.keywords:
                        cluster_fragments.append({
                            'fragment_id': fragment.fragment_id,
                            'content': fragment.content[:100] + "..." if len(fragment.content) > 100 else fragment.content,
                            'speaker': fragment.speaker,
                            'confidence': fragment.confidence
                        })
                
                topic = {
                    'topic_name': keyword,
                    'frequency': freq,
                    'importance': freq / len(self.fragments),
                    'fragments': cluster_fragments,
                    'related_speakers': list(set([f['speaker'] for f in cluster_fragments if f['speaker']]))
                }
                self.topics.append(topic)
    
    def _analyze_speakers(self) -> Dict[str, Any]:
        """발표자 분석"""
        speaker_stats = {}
        
        for fragment in self.fragments:
            if fragment.speaker:
                if fragment.speaker not in speaker_stats:
                    speaker_stats[fragment.speaker] = {
                        'fragment_count': 0,
                        'total_confidence': 0,
                        'keywords': [],
                        'content_length': 0
                    }
                
                speaker_stats[fragment.speaker]['fragment_count'] += 1
                speaker_stats[fragment.speaker]['total_confidence'] += fragment.confidence
                speaker_stats[fragment.speaker]['keywords'].extend(fragment.keywords)
                speaker_stats[fragment.speaker]['content_length'] += len(fragment.content)
        
        # 통계 계산
        for speaker, stats in speaker_stats.items():
            stats['avg_confidence'] = stats['total_confidence'] / stats['fragment_count']
            stats['top_keywords'] = [kw for kw, _ in Counter(stats['keywords']).most_common(5)]
            stats['avg_content_length'] = stats['content_length'] / stats['fragment_count']
            stats['engagement_score'] = stats['fragment_count'] / len(self.fragments)
        
        return speaker_stats
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """시간대별 패턴 분석"""
        if not self.fragments:
            return {}
        
        # 단순 순서 기반 분석 (실제 타임스탬프 파싱 대신)
        total_fragments = len(self.fragments)
        
        # 시작, 중간, 끝 구간 분석
        start_section = self.fragments[:total_fragments//3] if total_fragments >= 3 else self.fragments[:1]
        middle_section = self.fragments[total_fragments//3:2*total_fragments//3] if total_fragments >= 3 else []
        end_section = self.fragments[2*total_fragments//3:] if total_fragments >= 3 else []
        
        def analyze_section(fragments, section_name):
            if not fragments:
                return None
            
            keywords = []
            speakers = []
            avg_confidence = 0
            
            for f in fragments:
                keywords.extend(f.keywords)
                if f.speaker:
                    speakers.append(f.speaker)
                avg_confidence += f.confidence
            
            return {
                'section': section_name,
                'fragment_count': len(fragments),
                'top_keywords': [kw for kw, _ in Counter(keywords).most_common(3)],
                'active_speakers': list(set(speakers)),
                'avg_confidence': avg_confidence / len(fragments) if fragments else 0
            }
        
        return {
            'total_duration_fragments': total_fragments,
            'start_section': analyze_section(start_section, '시작'),
            'middle_section': analyze_section(middle_section, '중간'),
            'end_section': analyze_section(end_section, '끝')
        }
    
    def _generate_key_insights(self) -> List[str]:
        """핵심 인사이트 생성"""
        insights = []
        
        if not self.fragments:
            return ["분석할 데이터가 없습니다."]
        
        # 가장 활발한 발표자
        speaker_counts = Counter([f.speaker for f in self.fragments if f.speaker])
        if speaker_counts:
            most_active = speaker_counts.most_common(1)[0]
            insights.append(f"가장 활발한 발표자는 '{most_active[0]}'로 {most_active[1]}번 발언했습니다.")
        
        # 주요 주제
        if self.topics:
            top_topic = max(self.topics, key=lambda t: t['importance'])
            insights.append(f"핵심 주제는 '{top_topic['topic_name']}'로 전체의 {top_topic['importance']:.1%}를 차지합니다.")
        
        # 전체 신뢰도
        avg_confidence = np.mean([f.confidence for f in self.fragments])
        insights.append(f"전체 분석 신뢰도는 {avg_confidence:.1%}로 {'높은' if avg_confidence > 0.8 else '보통' if avg_confidence > 0.6 else '낮은'} 수준입니다.")
        
        # 다양성 분석
        unique_speakers = len(set([f.speaker for f in self.fragments if f.speaker]))
        if unique_speakers > 1:
            insights.append(f"{unique_speakers}명의 다양한 참여자가 균형있게 참여했습니다.")
        
        return insights

# Streamlit UI
def main():
    st.title("🗃️ 홀리스틱 컨퍼런스 분석기 (Supabase 지원)")
    st.markdown("**멀티모달 데이터를 완전히 통합하여 전체적인 컨퍼런스 이해를 제공합니다**")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 데이터베이스 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "my_conference")
    db_type = st.sidebar.selectbox("데이터베이스 타입", ["auto", "sqlite", "supabase"])
    
    # 분석기 초기화
    analyzer = HolisticConferenceAnalyzerSupabase(conference_name, db_type)
    
    # 데이터베이스 상태 확인
    db_status = analyzer.check_database_connection()
    
    if db_status["connected"]:
        st.success(f"✅ {db_status['message']}")
    else:
        st.warning(f"⚠️ {db_status['message']}")
        if db_type == "supabase":
            st.info("💡 Supabase 환경변수를 설정하거나 SQLite 모드를 사용하세요.")
    
    # 홀리스틱 분석 실행
    if st.button("🗃️ 홀리스틱 데이터베이스 분석 시작", type="primary"):
        with st.spinner("홀리스틱 분석을 수행하고 있습니다..."):
            results = analyzer.analyze_conference_holistically()
            
            if "error" in results:
                st.error(f"❌ 분석 실패: {results['error']}")
                return
            
            st.success("✅ 홀리스틱 분석 완료!")
            
            # 결과 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 조각 수", results["total_fragments"])
            
            with col2:
                st.metric("발견된 개체", results["total_entities"])
            
            with col3:
                st.metric("주요 주제", results["total_topics"])
            
            with col4:
                st.metric("평균 신뢰도", f"{results['average_confidence']:.1%}")
            
            # 상세 결과 탭
            tab1, tab2, tab3, tab4 = st.tabs(["📊 전체 개요", "👥 발표자 분석", "📋 주제 분석", "🕐 시간대 분석"])
            
            with tab1:
                st.markdown("### 🎯 핵심 인사이트")
                for insight in results["key_insights"]:
                    st.markdown(f"- {insight}")
                
                st.markdown("### 📊 데이터베이스 정보")
                st.markdown(f"**데이터베이스**: {results['database_type']}")
                st.markdown(f"**분석 시간**: {results['analysis_timestamp']}")
            
            with tab2:
                st.markdown("### 👥 발표자별 상세 분석")
                speaker_analysis = results.get("speaker_analysis", {})
                
                for speaker, stats in speaker_analysis.items():
                    with st.expander(f"🎤 {speaker}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("발언 횟수", stats["fragment_count"])
                            st.metric("평균 신뢰도", f"{stats['avg_confidence']:.1%}")
                        
                        with col2:
                            st.metric("참여도", f"{stats['engagement_score']:.1%}")
                            st.metric("평균 발언 길이", f"{stats['avg_content_length']:.0f}자")
                        
                        st.markdown("**주요 키워드**")
                        st.write(", ".join(stats["top_keywords"]))
            
            with tab3:
                st.markdown("### 📋 주제별 상세 분석")
                topics = results.get("topic_distribution", [])
                
                for topic in topics:
                    with st.expander(f"📌 {topic['topic_name']} (중요도: {topic['importance']:.1%})"):
                        st.markdown(f"**언급 빈도**: {topic['frequency']}회")
                        st.markdown(f"**관련 발표자**: {', '.join(topic['related_speakers'])}")
                        
                        st.markdown("**관련 내용**")
                        for fragment in topic['fragments'][:3]:  # 상위 3개만 표시
                            st.markdown(f"- *{fragment['speaker']}*: {fragment['content']}")
            
            with tab4:
                st.markdown("### 🕐 시간대별 패턴 분석")
                temporal = results.get("temporal_analysis", {})
                
                if temporal:
                    sections = [temporal.get('start_section'), temporal.get('middle_section'), temporal.get('end_section')]
                    
                    for section in sections:
                        if section:
                            st.markdown(f"#### {section['section']} 구간")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("조각 수", section['fragment_count'])
                                st.metric("평균 신뢰도", f"{section['avg_confidence']:.1%}")
                            
                            with col2:
                                st.markdown("**주요 키워드**")
                                st.write(", ".join(section['top_keywords']))
                                st.markdown("**활발한 발표자**")
                                st.write(", ".join(section['active_speakers']))
            
            # 상세 정보
            with st.expander("📊 전체 분석 결과 (JSON)"):
                st.json(results)

if __name__ == "__main__":
    main()