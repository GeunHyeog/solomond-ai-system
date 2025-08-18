#!/usr/bin/env python3
"""
브라우저에서 CLI 수준의 화자 분리 결과 표시
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def load_existing_analysis_results():
    """기존 분석 결과 파일들 로드"""
    results_dir = Path(".")
    
    # 분석 결과 파일들 찾기
    json_files = list(results_dir.glob("*analysis*.json"))
    json_files.extend(list(results_dir.glob("*transcript*.json")))
    json_files.extend(list(results_dir.glob("*stt*.json")))
    
    # 파일 크기로 필터링 (너무 작은 파일 제외)
    valid_files = []
    for file in json_files:
        try:
            file_size = file.stat().st_size
            if file_size > 1000:  # 1KB 이상
                valid_files.append({
                    'file': file,
                    'name': file.name,
                    'size_mb': file_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                })
        except:
            continue
    
    # 수정일 기준 정렬 (최신순)
    valid_files.sort(key=lambda x: x['modified'], reverse=True)
    
    return valid_files

def display_speaker_timeline(segments, title="화자별 대화 타임라인"):
    """화자별 타임라인 시각화"""
    if not segments:
        st.warning("세그먼트 데이터가 없습니다.")
        return
    
    # 타임라인 데이터 준비
    timeline_data = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, segment in enumerate(segments[:50]):  # 처음 50개만 표시
        start = segment.get('start', 0)
        end = segment.get('end', start + 1)
        text = segment.get('text', '').strip()
        speaker_id = segment.get('speaker', i % 4)  # 기본 4명으로 가정
        
        timeline_data.append({
            'Speaker': f'화자 {speaker_id + 1}',
            'Start': start,
            'End': end,
            'Duration': end - start,
            'Text': text[:100] + '...' if len(text) > 100 else text,
            'Index': i
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        
        # Gantt 차트 생성
        fig = px.timeline(
            df, 
            x_start='Start', 
            x_end='End', 
            y='Speaker',
            color='Speaker',
            hover_data=['Text', 'Duration'],
            title=title
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="시간 (초)",
            yaxis_title="화자"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_speaker_statistics(segments):
    """화자별 통계 표시"""
    if not segments:
        return
    
    # 화자별 통계 계산
    speaker_stats = {}
    
    for segment in segments:
        speaker_id = segment.get('speaker', 0)
        speaker_name = f'화자 {speaker_id + 1}'
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '')
        
        if speaker_name not in speaker_stats:
            speaker_stats[speaker_name] = {
                'total_time': 0,
                'word_count': 0,
                'segments_count': 0,
                'texts': []
            }
        
        speaker_stats[speaker_name]['total_time'] += (end - start)
        speaker_stats[speaker_name]['word_count'] += len(text.split())
        speaker_stats[speaker_name]['segments_count'] += 1
        speaker_stats[speaker_name]['texts'].append(text.strip())
    
    # 통계 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 화자별 발화 시간")
        
        if speaker_stats:
            speakers = list(speaker_stats.keys())
            times = [speaker_stats[s]['total_time'] for s in speakers]
            
            fig = px.bar(
                x=speakers,
                y=times,
                title="화자별 총 발화 시간 (초)",
                color=speakers
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 화자별 단어 수")
        
        if speaker_stats:
            words = [speaker_stats[s]['word_count'] for s in speakers]
            
            fig = px.pie(
                values=words,
                names=speakers,
                title="화자별 단어 수 비율"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # 상세 통계 테이블
    st.markdown("#### 📋 화자별 상세 통계")
    
    stats_data = []
    for speaker, stats in speaker_stats.items():
        stats_data.append({
            '화자': speaker,
            '발화 시간(초)': f"{stats['total_time']:.1f}",
            '단어 수': stats['word_count'],
            '발화 횟수': stats['segments_count'],
            '평균 발화 길이': f"{stats['total_time'] / stats['segments_count']:.1f}초" if stats['segments_count'] > 0 else "0초"
        })
    
    if stats_data:
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True)

def display_full_transcript_with_speakers(segments):
    """전체 전사 텍스트를 화자별로 표시"""
    st.markdown("#### 📝 화자별 전체 대화 내용")
    
    if not segments:
        st.warning("전사 데이터가 없습니다.")
        return
    
    # 화자별 색상 정의
    speaker_colors = {
        '화자 1': '#FF6B6B',
        '화자 2': '#4ECDC4', 
        '화자 3': '#45B7D1',
        '화자 4': '#96CEB4'
    }
    
    # 검색 기능
    search_term = st.text_input("🔍 대화 내용 검색", placeholder="검색할 키워드를 입력하세요")
    
    # 화자별 필터
    available_speakers = list(set([f'화자 {seg.get("speaker", 0) + 1}' for seg in segments]))
    selected_speakers = st.multiselect(
        "👥 표시할 화자 선택", 
        available_speakers, 
        default=available_speakers
    )
    
    # 시간 범위 필터
    if segments:
        max_time = max([seg.get('end', 0) for seg in segments])
        time_range = st.slider(
            "⏰ 시간 범위 선택 (초)",
            0.0, max_time, (0.0, max_time),
            step=10.0
        )
    
    st.markdown("---")
    
    # 대화 내용 표시
    filtered_segments = []
    
    for i, segment in enumerate(segments):
        speaker_id = segment.get('speaker', 0)
        speaker_name = f'화자 {speaker_id + 1}'
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        # 필터링 조건 확인
        if speaker_name not in selected_speakers:
            continue
        
        if start_time < time_range[0] or end_time > time_range[1]:
            continue
        
        if search_term and search_term.lower() not in text.lower():
            continue
        
        filtered_segments.append({
            'speaker': speaker_name,
            'start': start_time,
            'end': end_time,
            'text': text,
            'index': i
        })
    
    # 결과 표시
    st.markdown(f"**필터링된 대화**: {len(filtered_segments)}개 세그먼트")
    
    for segment in filtered_segments[:100]:  # 최대 100개만 표시
        speaker = segment['speaker']
        start = segment['start']
        text = segment['text']
        
        # 화자별 색상 적용
        color = speaker_colors.get(speaker, '#CCCCCC')
        
        # 검색어 하이라이트
        display_text = text
        if search_term and search_term.lower() in text.lower():
            display_text = text.replace(
                search_term, 
                f"**:red[{search_term}]**"
            )
        
        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {color}; 
                padding: 10px; 
                margin: 5px 0; 
                background-color: rgba(255,255,255,0.1);
                border-radius: 5px;
            ">
                <strong style="color: {color};">{speaker}</strong> 
                <span style="color: #888; font-size: 0.9em;">({start:.1f}초)</span><br>
                {display_text}
            </div>
            """, 
            unsafe_allow_html=True
        )

def main():
    st.set_page_config(
        page_title="솔로몬드 AI - CLI 수준 화자 분리 결과",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 솔로몬드 AI - CLI 수준 화자 분리 결과 브라우저")
    st.markdown("CLI에서 분석한 결과를 브라우저에서 상세히 확인할 수 있습니다.")
    
    # 사이드바 - 파일 선택
    st.sidebar.title("📁 분석 결과 선택")
    
    # 기존 분석 결과 로드
    available_files = load_existing_analysis_results()
    
    if not available_files:
        st.warning("분석 결과 파일을 찾을 수 없습니다.")
        st.info("먼저 CLI 또는 브라우저에서 분석을 실행해주세요.")
        return
    
    # 파일 선택 드롭다운
    file_options = []
    for file_info in available_files[:10]:  # 최신 10개만
        name = file_info['name']
        size = file_info['size_mb']
        modified = file_info['modified']
        file_options.append(f"{name} ({size:.1f}MB, {modified.strftime('%m-%d %H:%M')})")
    
    selected_file_display = st.sidebar.selectbox(
        "분석 결과 파일 선택",
        file_options
    )
    
    if not selected_file_display:
        st.info("분석 결과 파일을 선택해주세요.")
        return
    
    # 선택된 파일 로드
    selected_index = file_options.index(selected_file_display)
    selected_file = available_files[selected_index]['file']
    
    st.sidebar.markdown(f"**선택된 파일:** {selected_file.name}")
    st.sidebar.markdown(f"**파일 크기:** {available_files[selected_index]['size_mb']:.1f}MB")
    
    # JSON 파일 로드
    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        st.success(f"✅ 분석 결과 로드 완료: {selected_file.name}")
        
    except Exception as e:
        st.error(f"❌ 파일 로드 실패: {str(e)}")
        return
    
    # 데이터 구조 확인 및 표시
    st.markdown("---")
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 분석 개요", 
        "🎭 화자별 대화", 
        "📈 통계 및 차트", 
        "📝 전체 전사문"
    ])
    
    with tab1:
        st.markdown("### 📊 분석 개요")
        
        # 기본 정보 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'start_time' in analysis_data:
                st.metric("분석 시작", analysis_data['start_time'])
        
        with col2:
            if 'target_files' in analysis_data:
                st.metric("분석 파일 수", len(analysis_data['target_files']))
        
        with col3:
            if 'session_id' in analysis_data:
                st.metric("세션 ID", analysis_data['session_id'])
        
        # 대상 파일 정보
        if 'target_files' in analysis_data:
            st.markdown("#### 📁 분석 대상 파일")
            for file_info in analysis_data['target_files']:
                with st.expander(f"📄 {file_info.get('file_name', 'Unknown')}"):
                    st.json(file_info)
    
    with tab2:
        st.markdown("### 🎭 화자별 대화 내용")
        
        # STT 결과에서 세그먼트 추출
        segments = []
        
        if 'stt_results' in analysis_data:
            for stt_result in analysis_data['stt_results']:
                if 'stt_result' in stt_result:
                    stt_data = stt_result['stt_result']
                    if 'segments' in stt_data:
                        segments.extend(stt_data['segments'])
        
        if segments:
            display_full_transcript_with_speakers(segments)
        else:
            st.warning("화자별 세그먼트 데이터를 찾을 수 없습니다.")
    
    with tab3:
        st.markdown("### 📈 통계 및 차트")
        
        if segments:
            # 화자별 통계 표시
            display_speaker_statistics(segments)
            
            # 타임라인 차트
            st.markdown("---")
            display_speaker_timeline(segments)
        else:
            st.warning("통계를 생성할 데이터가 없습니다.")
    
    with tab4:
        st.markdown("### 📝 전체 전사문")
        
        # 전체 전사 텍스트 표시
        if 'stt_results' in analysis_data:
            for i, stt_result in enumerate(analysis_data['stt_results']):
                if 'stt_result' in stt_result and 'text' in stt_result['stt_result']:
                    file_name = stt_result.get('file_info', {}).get('file_name', f'파일 {i+1}')
                    
                    with st.expander(f"📄 {file_name} - 전체 전사문", expanded=i==0):
                        full_text = stt_result['stt_result']['text']
                        
                        # 텍스트 통계
                        col1, col2, col3 = st.columns(3)
                        col1.metric("총 글자 수", len(full_text))
                        col2.metric("단어 수", len(full_text.split()))
                        col3.metric("문장 수", full_text.count('.') + full_text.count('!') + full_text.count('?'))
                        
                        # 전사 텍스트
                        st.text_area(
                            "전체 전사 내용",
                            full_text,
                            height=400,
                            key=f"full_text_{i}"
                        )
                        
                        # 다운로드 버튼
                        st.download_button(
                            "📥 전사문 다운로드",
                            full_text,
                            file_name=f"{file_name}_transcript.txt",
                            mime="text/plain"
                        )
        else:
            st.warning("전사 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()