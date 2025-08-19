#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 컨퍼런스 분석 브릿지
Bridge between basic analysis (8501) and holistic analysis (8600)
"""

import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import streamlit as st

def create_sample_data_for_testing():
    """테스트용 샘플 데이터 생성"""
    db_path = "conference_analysis_my_conference.db"
    
    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 테이블이 없으면 생성
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
    
    # 샘플 데이터 삽입
    sample_fragments = [
        {
            'fragment_id': 'frag_001',
            'file_source': 'presentation_intro.jpg',
            'file_type': 'image',
            'timestamp': datetime.now().isoformat(),
            'speaker': '김대표',
            'content': '오늘 컨퍼런스에 참석해주신 여러분 감사합니다. 새로운 기술 동향에 대해 논의하겠습니다.',
            'confidence': 0.92,
            'keywords': '["컨퍼런스", "기술", "동향", "새로운", "논의"]'
        },
        {
            'fragment_id': 'frag_002',
            'file_source': 'discussion_audio.m4a',
            'file_type': 'audio',
            'timestamp': datetime.now().isoformat(),
            'speaker': '박연구원',
            'content': 'AI 기술의 발전 속도가 매우 빠릅니다. 특히 자연어 처리 분야에서 혁신적인 성과가 나타나고 있습니다.',
            'confidence': 0.88,
            'keywords': '["AI", "기술", "발전", "자연어", "처리", "혁신"]'
        },
        {
            'fragment_id': 'frag_003',
            'file_source': 'technical_slide.png',
            'file_type': 'image',
            'timestamp': datetime.now().isoformat(),
            'speaker': '이개발자',
            'content': '머신러닝 모델의 성능 향상을 위해서는 데이터 품질이 가장 중요합니다. 전처리 과정에 충분한 시간을 투자해야 합니다.',
            'confidence': 0.85,
            'keywords': '["머신러닝", "모델", "성능", "데이터", "품질", "전처리"]'
        },
        {
            'fragment_id': 'frag_004',
            'file_source': 'qa_session.wav',
            'file_type': 'audio',
            'timestamp': datetime.now().isoformat(),
            'speaker': '질문자',
            'content': '실제 프로덕션 환경에서는 어떤 문제들이 발생할 수 있나요? 특히 확장성 측면에서 고려해야 할 사항이 있다면 알려주세요.',
            'confidence': 0.90,
            'keywords': '["프로덕션", "환경", "문제", "확장성", "고려사항"]'
        },
        {
            'fragment_id': 'frag_005',
            'file_source': 'conclusion_video.mp4',
            'file_type': 'video',
            'timestamp': datetime.now().isoformat(),
            'speaker': '김대표',
            'content': '오늘 논의한 내용들을 정리하면, AI 기술 도입 시 고려해야 할 핵심 요소들을 파악할 수 있었습니다. 다음 미팅에서는 구체적인 실행 계획을 수립하겠습니다.',
            'confidence': 0.94,
            'keywords': '["정리", "AI", "도입", "핵심요소", "실행계획", "다음미팅"]'
        }
    ]
    
    # 데이터 삽입
    for fragment in sample_fragments:
        cursor.execute('''
            INSERT OR REPLACE INTO fragments 
            (fragment_id, file_source, file_type, timestamp, speaker, content, confidence, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fragment['fragment_id'],
            fragment['file_source'],
            fragment['file_type'],
            fragment['timestamp'],
            fragment['speaker'],
            fragment['content'],
            fragment['confidence'],
            fragment['keywords']
        ))
    
    conn.commit()
    conn.close()
    
    return len(sample_fragments)

def check_and_fix_data():
    """데이터 상태 확인 및 수정"""
    db_path = "conference_analysis_my_conference.db"
    
    if not os.path.exists(db_path):
        st.warning("데이터베이스 파일이 없습니다. 샘플 데이터를 생성합니다.")
        count = create_sample_data_for_testing()
        return f"샘플 데이터 {count}개가 생성되었습니다."
    
    # 기존 데이터 확인
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM fragments')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            st.warning("데이터베이스가 비어있습니다. 샘플 데이터를 생성합니다.")
            sample_count = create_sample_data_for_testing()
            return f"샘플 데이터 {sample_count}개가 생성되었습니다."
        else:
            return f"기존 데이터 {count}개가 확인되었습니다."
            
    except Exception as e:
        st.error(f"데이터베이스 오류: {e}")
        return f"오류 발생: {e}"

def main():
    st.title("🔗 컨퍼런스 분석 브릿지")
    st.markdown("**기본 분석과 홀리스틱 분석 간의 데이터 브릿지**")
    
    st.markdown("## 📊 데이터 상태 확인")
    
    if st.button("🔍 데이터 상태 확인 및 수정"):
        with st.spinner("데이터 상태를 확인하고 있습니다..."):
            result = check_and_fix_data()
            st.success(result)
    
    st.markdown("## 🎯 홀리스틱 분석 시스템 연결")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 기본 분석 시스템 (8501)"):
            st.markdown("👉 [기본 컨퍼런스 분석](http://localhost:8501)")
    
    with col2:
        if st.button("🎯 홀리스틱 분석 시스템 (8600)"):
            st.markdown("👉 [홀리스틱 컨퍼런스 분석](http://localhost:8600)")
    
    st.markdown("---")
    st.markdown("### 💡 사용 순서")
    st.markdown("""
    1. **데이터 준비**: 위의 "데이터 상태 확인 및 수정" 버튼을 클릭하여 테스트 데이터를 생성하세요
    2. **홀리스틱 분석**: http://localhost:8600 에서 홀리스틱 분석을 실행하세요
    3. **결과 확인**: 5개 탭에서 다양한 관점의 분석 결과를 확인하세요
    """)

if __name__ == "__main__":
    main()