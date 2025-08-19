#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 홀리스틱 컨퍼런스 분석 엔진
Holistic Conference Analysis Engine for SOLOMOND AI

핵심 목표: 개별 파일 분석 → 전체 컨퍼런스 상황의 입체적 이해
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
import hashlib
import re
from collections import defaultdict, Counter

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    st.warning("⚠️ 고급 NLP 라이브러리가 설치되지 않았습니다. 기본 분석 모드로 실행됩니다.")

@dataclass
class ConferenceFragment:
    """컨퍼런스 조각 데이터 구조"""
    fragment_id: str
    file_source: str
    file_type: str  # image, audio, video, text
    timestamp: Optional[str]
    speaker: Optional[str]
    content: str
    confidence: float
    keywords: List[str]
    embedding: Optional[List[float]] = None

@dataclass
class ConferenceEntity:
    """컨퍼런스 개체 (사람, 회사, 제품 등)"""
    entity_id: str
    entity_type: str  # person, company, product, concept
    name: str
    mentions: List[str]  # fragment_ids where mentioned
    relations: Dict[str, List[str]]  # relations to other entities

@dataclass
class ConferenceTopic:
    """컨퍼런스 주제/테마"""
    topic_id: str
    topic_name: str
    keywords: List[str]
    fragments: List[str]  # fragment_ids
    importance_score: float
    sentiment: str  # positive, negative, neutral

class HolisticConferenceAnalyzer:
    """홀리스틱 컨퍼런스 분석 엔진"""
    
    def __init__(self, conference_name: str = "default"):
        self.conference_name = conference_name
        self.db_path = f"conference_analysis_{conference_name}.db"
        self.fragments: List[ConferenceFragment] = []
        self.entities: List[ConferenceEntity] = []
        self.topics: List[ConferenceTopic] = []
        
        # 고급 NLP 모델 초기화
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_embeddings = True
            except Exception as e:
                st.warning(f"임베딩 모델 로드 실패: {e}")
                self.use_embeddings = False
        else:
            self.use_embeddings = False
        
        self._init_database()
    
    def _init_database(self):
        """SQLite 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 컨퍼런스 조각 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fragments (
                fragment_id TEXT PRIMARY KEY,
                file_source TEXT,
                file_type TEXT,
                timestamp TEXT,
                speaker TEXT,
                content TEXT,
                confidence REAL,
                keywords TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 개체 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT,
                name TEXT,
                mentions TEXT,
                relations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 주제 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                topic_id TEXT PRIMARY KEY,
                topic_name TEXT,
                keywords TEXT,
                fragments TEXT,
                importance_score REAL,
                sentiment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 컨퍼런스 메타데이터 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conference_metadata (
                conference_name TEXT PRIMARY KEY,
                total_fragments INTEGER,
                total_entities INTEGER,
                total_topics INTEGER,
                analysis_completed BOOLEAN,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_fragment_from_analysis(self, file_path: str, analysis_result: Dict[str, Any]):
        """기존 분석 결과에서 조각 추가"""
        file_type = self._detect_file_type(file_path)
        
        # 분석 결과에서 내용 추출
        content = ""
        confidence = 0.0
        speaker = None
        keywords = []
        
        if 'extracted_text' in analysis_result:
            content = analysis_result['extracted_text']
            confidence = analysis_result.get('confidence', 0.8)
        elif 'transcribed_text' in analysis_result:
            content = analysis_result['transcribed_text']
            confidence = analysis_result.get('confidence', 0.8)
        elif 'summary' in analysis_result:
            content = analysis_result['summary']
            confidence = 0.7
        
        # 키워드 추출
        keywords = self._extract_keywords(content)
        
        # 발표자 추출 (기본적인 패턴 매칭)
        speaker = self._extract_speaker(content)
        
        # 임베딩 생성
        embedding = None
        if self.use_embeddings and content:
            try:
                embedding = self.embedder.encode(content).tolist()
            except Exception as e:
                st.warning(f"임베딩 생성 실패: {e}")
        
        # 조각 생성
        fragment_id = self._generate_fragment_id(file_path, content)
        fragment = ConferenceFragment(
            fragment_id=fragment_id,
            file_source=file_path,
            file_type=file_type,
            timestamp=datetime.now().isoformat(),
            speaker=speaker,
            content=content,
            confidence=confidence,
            keywords=keywords,
            embedding=embedding
        )
        
        self.fragments.append(fragment)
        self._save_fragment_to_db(fragment)
        
        return fragment_id
    
    def _detect_file_type(self, file_path: str) -> str:
        """파일 타입 감지"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return 'image'
        elif ext in ['.mp3', '.wav', '.m4a', '.flac']:
            return 'audio'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
        elif ext in ['.txt', '.md', '.doc', '.docx']:
            return 'text'
        else:
            return 'unknown'
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출 (간단한 TF-IDF 기반)"""
        if not content:
            return []
        
        # 기본적인 전처리
        content = re.sub(r'[^\w\s]', ' ', content.lower())
        words = content.split()
        
        # 불용어 제거 (기본적인 한국어/영어)
        stopwords = {
            '그', '이', '저', '것', '수', '있', '하', '되', '의', '가', '을', '를', '에', '에서', '으로',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        words = [w for w in words if len(w) > 2 and w not in stopwords]
        
        # 빈도 계산
        word_freq = Counter(words)
        
        # 상위 키워드 반환
        return [word for word, _ in word_freq.most_common(max_keywords)]
    
    def _extract_speaker(self, content: str) -> Optional[str]:
        """발표자 추출 (기본 패턴 매칭)"""
        if not content:
            return None
        
        # 일반적인 발표자 패턴
        patterns = [
            r'발표자[:\s]*([가-힣A-Za-z\s]+)',
            r'speaker[:\s]*([A-Za-z\s]+)',
            r'([가-힣]+)\s*님이?\s*말씀',
            r'([A-Za-z]+)\s*said',
            r'질문자[:\s]*([가-힣A-Za-z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                speaker = match.group(1).strip()
                if len(speaker) > 1 and len(speaker) < 30:
                    return speaker
        
        return None
    
    def _generate_fragment_id(self, file_path: str, content: str) -> str:
        """조각 ID 생성"""
        unique_string = f"{file_path}_{content[:100]}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _save_fragment_to_db(self, fragment: ConferenceFragment):
        """조각을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 임베딩을 JSON으로 직렬화
        embedding_json = json.dumps(fragment.embedding) if fragment.embedding else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO fragments 
            (fragment_id, file_source, file_type, timestamp, speaker, content, confidence, keywords, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fragment.fragment_id,
            fragment.file_source,
            fragment.file_type,
            fragment.timestamp,
            fragment.speaker,
            fragment.content,
            fragment.confidence,
            json.dumps(fragment.keywords),
            embedding_json
        ))
        
        conn.commit()
        conn.close()
    
    def load_fragments_from_db(self) -> List[ConferenceFragment]:
        """데이터베이스에서 조각들 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM fragments ORDER BY created_at')
        rows = cursor.fetchall()
        
        fragments = []
        for row in rows:
            embedding = json.loads(row[8]) if row[8] else None
            fragment = ConferenceFragment(
                fragment_id=row[0],
                file_source=row[1],
                file_type=row[2],
                timestamp=row[3],
                speaker=row[4],
                content=row[5],
                confidence=row[6],
                keywords=json.loads(row[7]) if row[7] else [],
                embedding=embedding
            )
            fragments.append(fragment)
        
        conn.close()
        self.fragments = fragments
        return fragments
    
    def analyze_conference_holistically(self) -> Dict[str, Any]:
        """홀리스틱 컨퍼런스 분석 실행"""
        if not self.fragments:
            self.load_fragments_from_db()
        
        if not self.fragments:
            return {"error": "분석할 조각이 없습니다."}
        
        # 1. 개체 추출
        self._extract_entities()
        
        # 2. 주제 분석
        self._analyze_topics()
        
        # 3. 전체 스토리 구성
        story = self._generate_conference_story()
        
        # 4. 핵심 인사이트 추출
        insights = self._extract_key_insights()
        
        # 5. 액션 아이템 생성
        action_items = self._generate_action_items()
        
        return {
            "conference_name": self.conference_name,
            "total_fragments": len(self.fragments),
            "total_entities": len(self.entities),
            "total_topics": len(self.topics),
            "conference_story": story,
            "key_insights": insights,
            "action_items": action_items,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _extract_entities(self):
        """개체 추출 (사람, 회사, 제품 등)"""
        entity_mentions = defaultdict(list)
        
        for fragment in self.fragments:
            content = fragment.content
            
            # 기본적인 개체 추출 패턴
            patterns = {
                'person': [
                    r'([가-힣]{2,4})\s*님',
                    r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # John Smith
                    r'([가-힣]{2,4})\s*대표',
                    r'([가-힣]{2,4})\s*교수'
                ],
                'company': [
                    r'([가-힣A-Za-z]+)\s*회사',
                    r'([A-Z][a-z]+)\s*Inc',
                    r'([A-Z][a-z]+)\s*Corp',
                    r'([가-힣]+)\s*기업'
                ],
                'product': [
                    r'([A-Z][a-z]+\s*[0-9]+)',  # iPhone 15
                    r'([가-힣]+\s*[0-9]+)',
                ]
            }
            
            for entity_type, type_patterns in patterns.items():
                for pattern in type_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        entity_name = match.group(1).strip()
                        if len(entity_name) > 1:
                            entity_mentions[f"{entity_type}_{entity_name}"].append(fragment.fragment_id)
        
        # 개체 객체 생성
        self.entities = []
        for entity_key, mentions in entity_mentions.items():
            if len(mentions) >= 1:  # 최소 1번 언급된 개체만
                entity_type, entity_name = entity_key.split('_', 1)
                entity_id = hashlib.md5(entity_key.encode()).hexdigest()[:8]
                
                entity = ConferenceEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_name,
                    mentions=mentions,
                    relations={}
                )
                self.entities.append(entity)
    
    def _analyze_topics(self):
        """주제 분석"""
        # 모든 키워드 수집
        all_keywords = []
        for fragment in self.fragments:
            all_keywords.extend(fragment.keywords)
        
        # 키워드 빈도 분석
        keyword_freq = Counter(all_keywords)
        
        # 주요 키워드 그룹핑 (간단한 클러스터링)
        topic_groups = defaultdict(list)
        processed_keywords = set()
        
        for keyword, freq in keyword_freq.most_common(20):
            if keyword in processed_keywords:
                continue
            
            # 관련 키워드 찾기
            related_keywords = [keyword]
            for other_keyword, _ in keyword_freq.items():
                if other_keyword != keyword and other_keyword not in processed_keywords:
                    # 간단한 유사성 체크 (편집 거리 기반)
                    if self._is_similar_keyword(keyword, other_keyword):
                        related_keywords.append(other_keyword)
                        processed_keywords.add(other_keyword)
            
            if len(related_keywords) >= 1:
                topic_name = f"주제_{len(topic_groups) + 1}_{keyword}"
                topic_groups[topic_name] = related_keywords
                processed_keywords.add(keyword)
        
        # 주제 객체 생성
        self.topics = []
        for topic_name, keywords in topic_groups.items():
            # 해당 주제와 관련된 조각들 찾기
            related_fragments = []
            for fragment in self.fragments:
                if any(kw in fragment.keywords for kw in keywords):
                    related_fragments.append(fragment.fragment_id)
            
            if related_fragments:
                topic_id = hashlib.md5(topic_name.encode()).hexdigest()[:8]
                topic = ConferenceTopic(
                    topic_id=topic_id,
                    topic_name=topic_name.split('_', 2)[-1],  # 실제 키워드만
                    keywords=keywords,
                    fragments=related_fragments,
                    importance_score=len(related_fragments) * sum(keyword_freq[kw] for kw in keywords),
                    sentiment="neutral"  # 기본값
                )
                self.topics.append(topic)
        
        # 중요도순 정렬
        self.topics.sort(key=lambda t: t.importance_score, reverse=True)
    
    def _is_similar_keyword(self, kw1: str, kw2: str, threshold: float = 0.6) -> bool:
        """키워드 유사성 체크 (간단한 Jaccard 유사도)"""
        if len(kw1) < 2 or len(kw2) < 2:
            return False
        
        set1 = set(kw1)
        set2 = set(kw2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard = intersection / union if union > 0 else 0
        return jaccard >= threshold
    
    def _generate_conference_story(self) -> str:
        """컨퍼런스 전체 스토리 생성"""
        if not self.fragments:
            return "분석할 내용이 없습니다."
        
        story_parts = []
        
        # 1. 개요
        story_parts.append(f"📊 **컨퍼런스 개요**")
        story_parts.append(f"- 총 {len(self.fragments)}개의 자료가 분석되었습니다")
        story_parts.append(f"- {len(self.entities)}개의 주요 개체(인물/회사/제품)가 식별되었습니다")
        story_parts.append(f"- {len(self.topics)}개의 핵심 주제가 논의되었습니다")
        story_parts.append("")
        
        # 2. 주요 주제들
        if self.topics:
            story_parts.append("🎯 **주요 논의 주제**")
            for i, topic in enumerate(self.topics[:5], 1):
                story_parts.append(f"{i}. **{topic.topic_name}**")
                story_parts.append(f"   - 관련 키워드: {', '.join(topic.keywords[:5])}")
                story_parts.append(f"   - 언급 빈도: {len(topic.fragments)}회")
            story_parts.append("")
        
        # 3. 주요 참여자
        if self.entities:
            people = [e for e in self.entities if e.entity_type == 'person']
            companies = [e for e in self.entities if e.entity_type == 'company']
            
            if people:
                story_parts.append("👥 **주요 참여자**")
                for person in people[:5]:
                    story_parts.append(f"- **{person.name}**: {len(person.mentions)}회 언급")
                story_parts.append("")
            
            if companies:
                story_parts.append("🏢 **관련 기업/조직**")
                for company in companies[:5]:
                    story_parts.append(f"- **{company.name}**: {len(company.mentions)}회 언급")
                story_parts.append("")
        
        # 4. 전체 흐름
        story_parts.append("📋 **컨퍼런스 흐름**")
        
        # 파일 타입별 분류
        file_types = defaultdict(list)
        for fragment in self.fragments:
            file_types[fragment.file_type].append(fragment)
        
        for file_type, fragments in file_types.items():
            type_name = {
                'image': '📸 이미지 자료',
                'audio': '🎵 음성 기록',
                'video': '🎬 영상 자료',
                'text': '📝 텍스트 자료'
            }.get(file_type, f'📁 {file_type} 자료')
            
            story_parts.append(f"**{type_name}** ({len(fragments)}개)")
            
            # 각 타입별 주요 내용 요약
            for fragment in fragments[:3]:  # 상위 3개만
                content_preview = fragment.content[:100] + "..." if len(fragment.content) > 100 else fragment.content
                story_parts.append(f"  - {content_preview}")
            
            if len(fragments) > 3:
                story_parts.append(f"  - ... 외 {len(fragments) - 3}개 더")
            story_parts.append("")
        
        return "\n".join(story_parts)
    
    def _extract_key_insights(self) -> List[str]:
        """핵심 인사이트 추출 (3줄 요약)"""
        insights = []
        
        if self.topics:
            # 가장 중요한 주제
            top_topic = self.topics[0]
            insights.append(f"💡 가장 핵심적으로 다뤄진 주제는 '{top_topic.topic_name}'입니다 ({len(top_topic.fragments)}회 언급)")
        
        if self.entities:
            # 가장 많이 언급된 인물
            people = [e for e in self.entities if e.entity_type == 'person']
            if people:
                top_person = max(people, key=lambda p: len(p.mentions))
                insights.append(f"👤 '{top_person.name}'이(가) 가장 많이 언급되었습니다 ({len(top_person.mentions)}회)")
        
        # 전체적인 분석 결과
        total_confidence = np.mean([f.confidence for f in self.fragments]) if self.fragments else 0
        insights.append(f"📊 전체 자료의 분석 신뢰도는 {total_confidence:.1%}입니다")
        
        return insights[:3]  # 최대 3개
    
    def _generate_action_items(self) -> List[str]:
        """액션 아이템 생성 (5가지)"""
        action_items = []
        
        # 1. 후속 미팅 관련
        if len(self.fragments) > 5:
            action_items.append("📅 주요 논의사항에 대한 후속 미팅 일정 조율")
        
        # 2. 자료 정리
        if len(self.topics) > 3:
            action_items.append(f"📋 {len(self.topics)}개 주제별로 세부 자료 정리 및 문서화")
        
        # 3. 이해관계자 연락
        people = [e for e in self.entities if e.entity_type == 'person']
        if len(people) > 2:
            action_items.append(f"📞 주요 참여자 {len(people)}명과 개별 후속 논의")
        
        # 4. 기술적 검토
        tech_keywords = [f for f in self.fragments if any(kw in f.content.lower() for kw in ['기술', '개발', 'technology', 'development'])]
        if tech_keywords:
            action_items.append("🔧 기술적 이슈에 대한 전문가 검토 및 피드백 수집")
        
        # 5. 다음 단계 계획
        action_items.append("🎯 논의된 내용을 바탕으로 구체적인 실행 계획 수립")
        
        return action_items[:5]  # 최대 5개

# Streamlit UI 부분
def main():
    st.title("🎯 홀리스틱 컨퍼런스 분석 엔진")
    st.markdown("**개별 파일 분석을 넘어서 전체 컨퍼런스 상황의 입체적 이해**")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 설정")
    conference_name = st.sidebar.text_input("컨퍼런스 이름", "my_conference")
    
    # 분석기 초기화
    analyzer = HolisticConferenceAnalyzer(conference_name)
    
    # 기존 분석 결과 로드
    existing_fragments = analyzer.load_fragments_from_db()
    
    if existing_fragments:
        st.info(f"📊 기존 분석 결과: {len(existing_fragments)}개 조각이 발견되었습니다")
        
        if st.button("🔍 홀리스틱 분석 실행"):
            with st.spinner("전체 컨퍼런스 상황을 분석하고 있습니다..."):
                result = analyzer.analyze_conference_holistically()
                
                if "error" not in result:
                    st.success("✅ 홀리스틱 분석 완료!")
                    
                    # 결과 표시
                    st.markdown("## 📖 컨퍼런스 전체 스토리")
                    st.markdown(result["conference_story"])
                    
                    st.markdown("## 💡 핵심 인사이트")
                    for insight in result["key_insights"]:
                        st.markdown(f"- {insight}")
                    
                    st.markdown("## ✅ 액션 아이템")
                    for item in result["action_items"]:
                        st.markdown(f"- {item}")
                    
                    # 상세 정보
                    with st.expander("📊 상세 분석 정보"):
                        st.json(result)
                else:
                    st.error(result["error"])
    else:
        st.warning("📁 분석할 데이터가 없습니다. 먼저 컨퍼런스 자료를 분석해주세요.")
        st.markdown("**사용법:**")
        st.markdown("1. 기존 컨퍼런스 분석 시스템(8501)에서 자료를 분석하세요")
        st.markdown("2. 분석이 완료되면 이 시스템에서 홀리스틱 분석을 실행하세요")

if __name__ == "__main__":
    main()