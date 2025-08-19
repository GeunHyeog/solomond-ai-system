#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 의미적 연결 엔진
Semantic Connection Engine for Holistic Conference Analysis

핵심 목표: 파일 간 의미적 연관성 자동 탐지 및 연결
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from dataclasses import dataclass
import sqlite3
import json
from pathlib import Path
import networkx as nx
from collections import defaultdict, Counter
import re

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import spacy
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    st.warning("⚠️ 고급 NLP 라이브러리가 필요합니다. pip install sentence-transformers faiss-cpu spacy scikit-learn")

@dataclass
class SemanticConnection:
    """의미적 연결 데이터 구조"""
    connection_id: str
    source_fragment: str
    target_fragment: str
    connection_type: str  # semantic, speaker, temporal, topical
    confidence: float
    evidence: str
    relationship: str  # similar, continues, contradicts, supports

@dataclass
class SpeakerProfile:
    """발표자 프로필"""
    speaker_id: str
    name: str
    fragments: List[str]
    key_topics: List[str]
    speaking_style: Dict[str, Any]
    total_mentions: int

@dataclass
class TopicCluster:
    """주제 클러스터"""
    cluster_id: str
    cluster_name: str
    fragments: List[str]
    key_keywords: List[str]
    centrality_score: float
    internal_coherence: float

class SemanticConnectionEngine:
    """의미적 연결 엔진"""
    
    def __init__(self, conference_name: str = "default"):
        self.conference_name = conference_name
        self.db_path = f"conference_analysis_{conference_name}.db"
        
        # 고급 NLP 모델 초기화
        if ADVANCED_NLP_AVAILABLE:
            try:
                # 문장 임베딩 모델 (다국어 지원)
                self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                
                # FAISS 벡터 인덱스
                self.vector_index = None
                self.fragment_embeddings = {}
                
                # SpaCy NLP 파이프라인 (한국어/영어)
                try:
                    # 한국어 모델 시도
                    self.nlp = spacy.load("ko_core_news_sm")
                except OSError:
                    try:
                        # 영어 모델 시도
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        # 기본 빈 모델
                        self.nlp = spacy.blank("ko")
                        st.warning("⚠️ SpaCy 언어 모델이 설치되지 않았습니다. 기본 기능으로 작동합니다.")
                
                self.use_advanced_nlp = True
                
            except Exception as e:
                st.error(f"고급 NLP 모델 초기화 실패: {e}")
                self.use_advanced_nlp = False
        else:
            self.use_advanced_nlp = False
        
        # 연결 그래프
        self.connection_graph = nx.Graph()
        self.connections: List[SemanticConnection] = []
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        self.topic_clusters: List[TopicCluster] = []
    
    def load_fragments_from_db(self) -> List[Dict]:
        """데이터베이스에서 조각들 로드"""
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
                    'keywords': json.loads(row[7]) if row[7] else [],
                    'embedding': json.loads(row[8]) if row[8] else None
                }
                fragments.append(fragment)
            
            return fragments
            
        except sqlite3.OperationalError:
            st.warning("⚠️ 분석된 컨퍼런스 데이터가 없습니다. 먼저 컨퍼런스 분석을 실행해주세요.")
            return []
        finally:
            conn.close()
    
    def analyze_semantic_connections(self) -> Dict[str, Any]:
        """의미적 연결 분석 실행"""
        fragments = self.load_fragments_from_db()
        
        if not fragments:
            return {"error": "분석할 조각이 없습니다."}
        
        if len(fragments) < 2:
            return {"error": "연결 분석을 위해서는 최소 2개의 조각이 필요합니다."}
        
        # 1. 임베딩 생성 및 벡터 인덱스 구축
        self._build_vector_index(fragments)
        
        # 2. 의미적 유사성 연결
        semantic_connections = self._find_semantic_connections(fragments)
        
        # 3. 발표자별 연결
        speaker_connections = self._find_speaker_connections(fragments)
        
        # 4. 주제별 클러스터링
        topic_clusters = self._cluster_by_topics(fragments)
        
        # 5. 시간적 연결 (순서 기반)
        temporal_connections = self._find_temporal_connections(fragments)
        
        # 6. 연결 그래프 구축
        self._build_connection_graph(fragments)
        
        # 7. 핵심 인사이트 추출
        insights = self._extract_connection_insights()
        
        return {
            "total_fragments": len(fragments),
            "semantic_connections": len(semantic_connections),
            "speaker_connections": len(speaker_connections),
            "topic_clusters": len(topic_clusters),
            "temporal_connections": len(temporal_connections),
            "connection_insights": insights,
            "speaker_profiles": {k: {
                "name": v.name,
                "fragments_count": len(v.fragments),
                "key_topics": v.key_topics[:5],
                "total_mentions": v.total_mentions
            } for k, v in self.speaker_profiles.items()},
            "topic_clusters_summary": [{
                "cluster_name": tc.cluster_name,
                "fragments_count": len(tc.fragments),
                "key_keywords": tc.key_keywords[:5],
                "coherence": tc.internal_coherence
            } for tc in topic_clusters]
        }
    
    def _build_vector_index(self, fragments: List[Dict]):
        """벡터 인덱스 구축"""
        if not self.use_advanced_nlp:
            return
        
        contents = []
        fragment_ids = []
        
        for fragment in fragments:
            if fragment['content'] and len(fragment['content'].strip()) > 0:
                contents.append(fragment['content'])
                fragment_ids.append(fragment['fragment_id'])
        
        if not contents:
            return
        
        try:
            # 임베딩 생성
            embeddings = self.embedder.encode(contents, show_progress_bar=True)
            
            # 차원 확인
            if len(embeddings.shape) != 2:
                st.error("임베딩 차원이 올바르지 않습니다.")
                return
            
            # FAISS 인덱스 구축
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # L2 정규화 (cosine similarity를 위해) - 안전한 복사본 사용
            embeddings_copy = embeddings.copy().astype('float32')
            
            # 벡터가 유효한지 확인
            if not np.all(np.isfinite(embeddings_copy)):
                st.warning("일부 임베딩 벡터에 무한값이나 NaN이 포함되어 있습니다. 정리 중...")
                embeddings_copy = np.nan_to_num(embeddings_copy, nan=0.0, posinf=1.0, neginf=-1.0)
            
            faiss.normalize_L2(embeddings_copy)
            self.vector_index.add(embeddings_copy)
            
            # 임베딩 저장
            for i, fragment_id in enumerate(fragment_ids):
                self.fragment_embeddings[fragment_id] = embeddings[i].tolist()
            
            st.success(f"✅ {len(contents)}개 조각의 벡터 인덱스 구축 완료")
            
        except Exception as e:
            st.error(f"벡터 인덱스 구축 실패: {e}")
            # 디버그 정보 추가
            st.error(f"디버그 정보: contents={len(contents)}, embeddings shape={embeddings.shape if 'embeddings' in locals() else 'N/A'}")
    
    def _find_semantic_connections(self, fragments: List[Dict]) -> List[SemanticConnection]:
        """의미적 유사성 기반 연결 탐지"""
        connections = []
        
        if not self.use_advanced_nlp or not self.vector_index:
            return connections
        
        try:
            # 각 조각에 대해 유사한 조각들 찾기
            for i, fragment in enumerate(fragments):
                if fragment['fragment_id'] not in self.fragment_embeddings:
                    continue
                
                try:
                    embedding = np.array(self.fragment_embeddings[fragment['fragment_id']]).reshape(1, -1).astype('float32')
                    
                    # 벡터 유효성 검사
                    if not np.all(np.isfinite(embedding)):
                        st.warning(f"조각 {fragment['fragment_id']}의 임베딩에 유효하지 않은 값이 있습니다. 건너뜀.")
                        continue
                    
                    embedding_copy = embedding.copy()
                    faiss.normalize_L2(embedding_copy)
                    
                    # 유사한 조각들 검색 (자기 자신 제외)
                    k = min(5, len(fragments))  # 상위 5개
                    scores, indices = self.vector_index.search(embedding_copy, k)
                    
                    for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        if idx != i and score > 0.7:  # 높은 유사도만
                            target_fragment = fragments[idx]
                            
                            connection = SemanticConnection(
                                connection_id=f"sem_{fragment['fragment_id']}_{target_fragment['fragment_id']}",
                                source_fragment=fragment['fragment_id'],
                                target_fragment=target_fragment['fragment_id'],
                                connection_type="semantic",
                                confidence=float(score),
                                evidence=f"의미적 유사도: {score:.3f}",
                                relationship="similar"
                            )
                            connections.append(connection)
                            
                except Exception as inner_e:
                    st.warning(f"조각 {fragment['fragment_id']} 처리 중 오류: {inner_e}")
                    continue
            
            self.connections.extend(connections)
            return connections
            
        except Exception as e:
            st.error(f"의미적 연결 탐지 실패: {e}")
            return []
    
    def _find_speaker_connections(self, fragments: List[Dict]) -> List[SemanticConnection]:
        """발표자 기반 연결 탐지"""
        connections = []
        speaker_fragments = defaultdict(list)
        
        # 발표자별 조각 그룹핑
        for fragment in fragments:
            speaker = fragment.get('speaker')
            if speaker and speaker.strip():
                speaker_fragments[speaker].append(fragment)
        
        # 발표자 프로필 생성
        for speaker, speaker_frags in speaker_fragments.items():
            if len(speaker_frags) < 2:
                continue
            
            # 발표자의 주요 키워드 추출
            all_keywords = []
            for frag in speaker_frags:
                all_keywords.extend(frag.get('keywords', []))
            
            key_topics = [kw for kw, count in Counter(all_keywords).most_common(5)]
            
            # 발표자 프로필 생성
            speaker_id = f"speaker_{speaker.replace(' ', '_')}"
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                name=speaker,
                fragments=[f['fragment_id'] for f in speaker_frags],
                key_topics=key_topics,
                speaking_style={},  # 추후 확장
                total_mentions=len(speaker_frags)
            )
            self.speaker_profiles[speaker_id] = profile
            
            # 같은 발표자의 조각들 간 연결
            for i in range(len(speaker_frags)):
                for j in range(i + 1, len(speaker_frags)):
                    frag1, frag2 = speaker_frags[i], speaker_frags[j]
                    
                    connection = SemanticConnection(
                        connection_id=f"spk_{frag1['fragment_id']}_{frag2['fragment_id']}",
                        source_fragment=frag1['fragment_id'],
                        target_fragment=frag2['fragment_id'],
                        connection_type="speaker",
                        confidence=0.9,
                        evidence=f"동일 발표자: {speaker}",
                        relationship="continues"
                    )
                    connections.append(connection)
        
        self.connections.extend(connections)
        return connections
    
    def _cluster_by_topics(self, fragments: List[Dict]) -> List[TopicCluster]:
        """주제별 클러스터링"""
        if not self.use_advanced_nlp or not self.fragment_embeddings:
            return []
        
        try:
            # 임베딩 매트릭스 구성
            embeddings = []
            fragment_ids = []
            
            for fragment in fragments:
                if fragment['fragment_id'] in self.fragment_embeddings:
                    embeddings.append(self.fragment_embeddings[fragment['fragment_id']])
                    fragment_ids.append(fragment['fragment_id'])
            
            if len(embeddings) < 3:
                return []
            
            embeddings = np.array(embeddings)
            
            # 최적 클러스터 수 결정 (3~8개)
            max_clusters = min(8, len(embeddings) // 2)
            n_clusters = max(3, max_clusters)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # 클러스터별 조각 그룹핑
            clusters = defaultdict(list)
            for fragment_id, label in zip(fragment_ids, cluster_labels):
                clusters[label].append(fragment_id)
            
            # TopicCluster 객체 생성
            topic_clusters = []
            for cluster_id, cluster_fragments in clusters.items():
                if len(cluster_fragments) < 2:
                    continue
                
                # 클러스터의 주요 키워드 추출
                cluster_keywords = []
                for frag_id in cluster_fragments:
                    fragment = next(f for f in fragments if f['fragment_id'] == frag_id)
                    cluster_keywords.extend(fragment.get('keywords', []))
                
                key_keywords = [kw for kw, _ in Counter(cluster_keywords).most_common(5)]
                
                # 클러스터 내부 일관성 계산
                cluster_embeddings = [self.fragment_embeddings[fid] for fid in cluster_fragments]
                coherence = self._calculate_cluster_coherence(cluster_embeddings)
                
                cluster_name = f"주제클러스터_{cluster_id}_{key_keywords[0] if key_keywords else 'unknown'}"
                
                topic_cluster = TopicCluster(
                    cluster_id=f"topic_{cluster_id}",
                    cluster_name=cluster_name,
                    fragments=cluster_fragments,
                    key_keywords=key_keywords,
                    centrality_score=0.0,  # 추후 계산
                    internal_coherence=coherence
                )
                topic_clusters.append(topic_cluster)
            
            # 일관성 순으로 정렬
            topic_clusters.sort(key=lambda tc: tc.internal_coherence, reverse=True)
            self.topic_clusters = topic_clusters
            
            return topic_clusters
            
        except Exception as e:
            st.error(f"주제 클러스터링 실패: {e}")
            return []
    
    def _calculate_cluster_coherence(self, embeddings: List[List[float]]) -> float:
        """클러스터 내부 일관성 계산"""
        if len(embeddings) < 2:
            return 0.0
        
        try:
            embeddings_array = np.array(embeddings)
            similarities = cosine_similarity(embeddings_array)
            
            # 자기 자신과의 유사도 제외하고 평균 계산
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            mean_similarity = similarities[mask].mean()
            
            return float(mean_similarity)
            
        except Exception:
            return 0.0
    
    def _find_temporal_connections(self, fragments: List[Dict]) -> List[SemanticConnection]:
        """시간적/순서적 연결 탐지"""
        connections = []
        
        # 파일 소스별로 그룹핑 (같은 파일에서 나온 조각들은 순서가 있음)
        file_groups = defaultdict(list)
        for fragment in fragments:
            file_groups[fragment['file_source']].append(fragment)
        
        for file_source, file_fragments in file_groups.items():
            if len(file_fragments) < 2:
                continue
            
            # 파일 내 조각들을 순서대로 연결
            for i in range(len(file_fragments) - 1):
                current_frag = file_fragments[i]
                next_frag = file_fragments[i + 1]
                
                connection = SemanticConnection(
                    connection_id=f"temp_{current_frag['fragment_id']}_{next_frag['fragment_id']}",
                    source_fragment=current_frag['fragment_id'],
                    target_fragment=next_frag['fragment_id'],
                    connection_type="temporal",
                    confidence=0.8,
                    evidence=f"동일 파일 순서: {file_source}",
                    relationship="continues"
                )
                connections.append(connection)
        
        self.connections.extend(connections)
        return connections
    
    def _build_connection_graph(self, fragments: List[Dict]):
        """연결 그래프 구축"""
        # 노드 추가 (각 조각)
        for fragment in fragments:
            self.connection_graph.add_node(
                fragment['fragment_id'],
                content=fragment['content'][:100],  # 미리보기
                speaker=fragment.get('speaker', 'Unknown'),
                file_type=fragment['file_type'],
                confidence=fragment['confidence']
            )
        
        # 엣지 추가 (연결)
        for connection in self.connections:
            self.connection_graph.add_edge(
                connection.source_fragment,
                connection.target_fragment,
                connection_type=connection.connection_type,
                confidence=connection.confidence,
                relationship=connection.relationship
            )
    
    def _extract_connection_insights(self) -> List[str]:
        """연결 분석 인사이트 추출"""
        insights = []
        
        if not self.connections:
            insights.append("📭 발견된 연결이 없습니다.")
            return insights
        
        # 연결 타입별 통계
        connection_types = Counter(conn.connection_type for conn in self.connections)
        insights.append(f"🔗 총 {len(self.connections)}개의 연결이 발견되었습니다")
        
        for conn_type, count in connection_types.most_common():
            type_name = {
                'semantic': '의미적 연결',
                'speaker': '발표자 연결', 
                'temporal': '시간적 연결',
                'topical': '주제별 연결'
            }.get(conn_type, conn_type)
            insights.append(f"  - {type_name}: {count}개")
        
        # 가장 연결이 많은 조각
        if self.connection_graph.nodes():
            degrees = dict(self.connection_graph.degree())
            max_degree_node = max(degrees, key=degrees.get)
            max_degree = degrees[max_degree_node]
            
            if max_degree > 0:
                insights.append(f"🎯 가장 많이 연결된 조각: {max_degree_node} ({max_degree}개 연결)")
        
        # 발표자 분석
        if self.speaker_profiles:
            most_active_speaker = max(self.speaker_profiles.values(), key=lambda sp: sp.total_mentions)
            insights.append(f"👤 가장 활발한 발표자: {most_active_speaker.name} ({most_active_speaker.total_mentions}개 조각)")
        
        # 주제 클러스터 분석
        if self.topic_clusters:
            best_cluster = max(self.topic_clusters, key=lambda tc: tc.internal_coherence)
            insights.append(f"💡 가장 일관성 있는 주제: {best_cluster.cluster_name} (일관성: {best_cluster.internal_coherence:.3f})")
        
        return insights

# Streamlit UI
def main():
    st.title("🧠 의미적 연결 엔진")
    st.markdown("**파일 간 의미적 연관성을 자동으로 탐지하고 연결합니다**")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "my_conference")
    
    # 엔진 초기화
    engine = SemanticConnectionEngine(conference_name)
    
    if not ADVANCED_NLP_AVAILABLE:
        st.error("⚠️ 고급 NLP 라이브러리가 필요합니다.")
        st.code("pip install sentence-transformers faiss-cpu spacy scikit-learn")
        return
    
    st.info("🚀 고급 NLP 시스템이 활성화되었습니다!")
    
    # 분석 실행
    if st.button("🔍 의미적 연결 분석 실행"):
        with st.spinner("의미적 연결을 분석하고 있습니다... (시간이 걸릴 수 있습니다)"):
            result = engine.analyze_semantic_connections()
            
            if "error" not in result:
                st.success("✅ 의미적 연결 분석 완료!")
                
                # 결과 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("전체 조각", result["total_fragments"])
                    st.metric("의미적 연결", result["semantic_connections"])
                
                with col2:
                    st.metric("발표자 연결", result["speaker_connections"])
                    st.metric("주제 클러스터", result["topic_clusters"])
                
                with col3:
                    st.metric("시간적 연결", result["temporal_connections"])
                
                # 인사이트
                st.markdown("## 💡 연결 분석 인사이트")
                for insight in result["connection_insights"]:
                    st.markdown(f"- {insight}")
                
                # 발표자 프로필
                if result["speaker_profiles"]:
                    st.markdown("## 👥 발표자 프로필")
                    for speaker_id, profile in result["speaker_profiles"].items():
                        with st.expander(f"🎤 {profile['name']}"):
                            st.write(f"**발언 조각:** {profile['fragments_count']}개")
                            st.write(f"**언급 횟수:** {profile['total_mentions']}회")
                            st.write(f"**주요 주제:** {', '.join(profile['key_topics'])}")
                
                # 주제 클러스터
                if result["topic_clusters_summary"]:
                    st.markdown("## 🎯 주제 클러스터")
                    for cluster in result["topic_clusters_summary"]:
                        with st.expander(f"📊 {cluster['cluster_name']}"):
                            st.write(f"**관련 조각:** {cluster['fragments_count']}개")
                            st.write(f"**주요 키워드:** {', '.join(cluster['key_keywords'])}")
                            st.write(f"**내부 일관성:** {cluster['coherence']:.3f}")
                
                # 상세 정보
                with st.expander("📊 상세 분석 정보"):
                    st.json(result)
            else:
                st.error(result["error"])
    
    st.markdown("---")
    st.markdown("**💡 사용법:** 먼저 홀리스틱 컨퍼런스 분석기에서 데이터를 분석한 후 이 시스템을 실행하세요.")

if __name__ == "__main__":
    main()