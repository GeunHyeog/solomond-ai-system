#!/usr/bin/env python3
"""
🎨 모듈1 UI 컴포넌트 - 실시간 진행률 및 UX 개선
사용자 경험 최적화를 위한 인터페이스 컴포넌트들

업데이트: 2025-01-30 - 실시간 진행률 + 미리보기 기능
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go

class RealTimeProgressUI:
    """실시간 진행률 표시 UI"""
    
    def __init__(self):
        self.progress_container = None
        self.status_container = None
        self.stats_container = None
        
    def initialize_progress_display(self, total_items: int, task_name: str = "처리"):
        """진행률 표시 초기화"""
        self.progress_container = st.container()
        
        with self.progress_container:
            st.markdown(f"### 📊 {task_name} 진행 상황")
            
            # 메인 진행률 바
            self.main_progress = st.progress(0)
            self.main_status = st.empty()
            
            # 상세 통계
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                self.processed_metric = st.empty()
            with col2:
                self.remaining_metric = st.empty()
            with col3:
                self.speed_metric = st.empty()
            with col4:
                self.eta_metric = st.empty()
            
            # 실시간 로그
            self.log_container = st.empty()
            
        return self
    
    def update_progress(self, current: int, total: int, current_item: str = "", 
                       processing_time: float = 0, logs: List[str] = None):
        """진행률 업데이트"""
        progress = current / total if total > 0 else 0
        remaining = total - current
        
        # 메인 진행률
        self.main_progress.progress(progress)
        self.main_status.text(f"처리 중: {current_item} ({current}/{total})")
        
        # 통계 업데이트
        self.processed_metric.metric("처리 완료", f"{current}/{total}")
        self.remaining_metric.metric("남은 항목", remaining)
        
        if processing_time > 0:
            items_per_sec = current / processing_time
            self.speed_metric.metric("처리 속도", f"{items_per_sec:.1f}/초")
            
            if items_per_sec > 0:
                eta_seconds = remaining / items_per_sec
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                self.eta_metric.metric("예상 완료", f"{eta_min}:{eta_sec:02d}")
        
        # 로그 표시
        if logs:
            with self.log_container.container():
                with st.expander("📝 처리 로그", expanded=False):
                    for log in logs[-10:]:  # 최근 10개만 표시
                        st.text(log)

class ResultPreviewUI:
    """결과 미리보기 UI"""
    
    def __init__(self):
        self.preview_container = None
        
    def initialize_preview_display(self):
        """미리보기 표시 초기화"""
        self.preview_container = st.container()
        return self
    
    def show_audio_preview(self, result: Dict, file_name: str):
        """음성 분석 결과 미리보기"""
        with self.preview_container:
            with st.expander(f"🎵 {file_name} - 음성 분석 결과", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # 텍스트 미리보기 (처음 200자)
                    text = result.get('text', '')
                    preview_text = text[:200] + "..." if len(text) > 200 else text
                    st.text_area("인식된 텍스트 (미리보기)", preview_text, height=100)
                    
                with col2:
                    # 통계 정보
                    st.metric("처리 시간", f"{result.get('processing_time', 0):.2f}초")
                    st.metric("텍스트 길이", f"{len(text)} 문자")
                    st.metric("언어", result.get('language', 'unknown'))
                    
                # 세그먼트 정보 (있는 경우)
                segments = result.get('segments', [])
                if segments and len(segments) > 0:
                    st.markdown("**시간별 세그먼트:**")
                    for i, segment in enumerate(segments[:3]):  # 처음 3개만
                        start = segment.get('start', 0)
                        end = segment.get('end', 0)
                        text_seg = segment.get('text', '')
                        st.text(f"{start:.1f}s - {end:.1f}s: {text_seg[:50]}...")
    
    def show_gemstone_preview(self, results: List[Dict]):
        """보석 분석 결과 미리보기"""
        if not results:
            return
            
        with self.preview_container:
            st.markdown("### 💎 보석 분석 결과 미리보기")
            
            # 전체 통계
            total_files = len(results)
            successful = len([r for r in results if 'error' not in r])
            total_colors = sum(len(r.get('dominant_colors', [])) for r in results if 'error' not in r)
            avg_processing_time = np.mean([r.get('processing_time', 0) for r in results if 'error' not in r])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("분석된 보석", f"{successful}/{total_files}")
            with col2:
                st.metric("추출된 색상", total_colors)
            with col3:
                st.metric("평균 처리 시간", f"{avg_processing_time:.2f}초")
            with col4:
                origins_found = len([r for r in results if 'error' not in r and r.get('origin_analysis', {}).get('estimated_origins')])
                st.metric("산지 추정 완료", origins_found)
            
            # 샘플 결과 표시 (처음 3개)
            for i, result in enumerate(results[:3]):
                if 'error' not in result:
                    with st.expander(f"💎 {result.get('filename', f'보석_{i+1}')} - 분석 결과"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 주요 색상 표시
                            colors = result.get('dominant_colors', [])
                            if colors:
                                st.markdown("**주요 색상:**")
                                for color in colors[:3]:
                                    color_info = f"🎨 {color.get('description', 'Unknown')} ({color.get('percentage', 0):.1f}%)"
                                    st.markdown(f"• {color_info}")
                            
                            # 산지 추정 표시
                            origin = result.get('origin_analysis', {})
                            origins = origin.get('estimated_origins', [])
                            if origins:
                                st.markdown("**추정 산지:**")
                                for orig, score in origins[:2]:
                                    st.markdown(f"• 🌍 {orig} (점수: {score:.1f})")
                        
                        with col2:
                            st.metric("처리 시간", f"{result.get('processing_time', 0):.2f}초")
                            st.metric("색상 수", len(colors))
                            confidence = origin.get('confidence_level', 'Unknown')
                            st.metric("신뢰도", confidence)
    
    def show_image_preview(self, results: List[Dict]):
        """이미지 분석 결과 미리보기"""
        if not results:
            return
            
        with self.preview_container:
            st.markdown("### 🖼️ 이미지 분석 결과 미리보기")
            
            # 전체 통계
            total_files = len(results)
            successful = len([r for r in results if 'error' not in r])
            total_blocks = sum(r.get('total_blocks', 0) for r in results if 'error' not in r)
            avg_confidence = np.mean([
                np.mean([block['confidence'] for block in r.get('text_blocks', [])])
                for r in results if 'error' not in r and r.get('text_blocks')
            ]) if any(r.get('text_blocks') for r in results if 'error' not in r) else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("처리된 이미지", f"{successful}/{total_files}")
            with col2:
                st.metric("추출된 텍스트 블록", total_blocks)
            with col3:
                st.metric("평균 신뢰도", f"{avg_confidence:.2%}")
            with col4:
                avg_time = np.mean([r.get('processing_time', 0) for r in results if 'error' not in r])
                st.metric("평균 처리 시간", f"{avg_time:.2f}초")
            
            # 상위 결과 미리보기
            successful_results = [r for r in results if 'error' not in r and r.get('text_blocks')]
            if successful_results:
                st.markdown("**상위 결과 미리보기:**")
                for i, result in enumerate(successful_results[:3]):  # 상위 3개
                    with st.expander(f"📄 {result.get('filename', f'이미지 {i+1}')}", expanded=False):
                        blocks = result.get('text_blocks', [])
                        for j, block in enumerate(blocks[:5]):  # 상위 5개 블록
                            confidence_color = "🟢" if block['confidence'] > 0.8 else "🟡" if block['confidence'] > 0.5 else "🔴"
                            st.text(f"{confidence_color} [{block['confidence']:.2f}] {block['text']}")

class AnalyticsUI:
    """분석 통계 및 차트 UI"""
    
    def __init__(self):
        pass
    
    def show_processing_analytics(self, audio_results: List[Dict], image_results: List[Dict]):
        """처리 분석 통계 표시"""
        st.markdown("### 📈 처리 분석 통계")
        
        # 처리 시간 분석
        if audio_results or image_results:
            col1, col2 = st.columns(2)
            
            with col1:
                if audio_results:
                    audio_times = [r.get('processing_time', 0) for r in audio_results if 'error' not in r]
                    if audio_times:
                        fig_audio = px.bar(
                            x=[f"파일 {i+1}" for i in range(len(audio_times))],
                            y=audio_times,
                            title="🎵 음성 파일 처리 시간",
                            labels={'x': '파일', 'y': '처리 시간 (초)'}
                        )
                        fig_audio.update_layout(height=300)
                        st.plotly_chart(fig_audio, use_container_width=True)
            
            with col2:
                if image_results:
                    image_times = [r.get('processing_time', 0) for r in image_results if 'error' not in r]
                    if image_times:
                        fig_image = px.histogram(
                            x=image_times,
                            title="🖼️ 이미지 처리 시간 분포",
                            labels={'x': '처리 시간 (초)', 'y': '파일 수'},
                            nbins=10
                        )
                        fig_image.update_layout(height=300)
                        st.plotly_chart(fig_image, use_container_width=True)
        
        # 텍스트 추출 품질 분석
        if image_results:
            confidence_data = []
            for result in image_results:
                if 'error' not in result and result.get('text_blocks'):
                    for block in result['text_blocks']:
                        confidence_data.append({
                            'filename': result.get('filename', ''),
                            'confidence': block['confidence'],
                            'text_length': len(block['text'])
                        })
            
            if confidence_data:
                df = pd.DataFrame(confidence_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_conf = px.scatter(
                        df, x='text_length', y='confidence',
                        title="📊 텍스트 길이 vs 신뢰도",
                        labels={'text_length': '텍스트 길이', 'confidence': '신뢰도'}
                    )
                    fig_conf.update_layout(height=300)
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                with col2:
                    # 신뢰도 분포
                    fig_dist = px.histogram(
                        df, x='confidence',
                        title="📈 신뢰도 분포",
                        labels={'confidence': '신뢰도', 'count': '블록 수'},
                        nbins=20
                    )
                    fig_dist.update_layout(height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)
    
    def show_gemstone_analytics(self, gemstone_results: List[Dict]):
        """보석 분석 통계 표시"""
        st.markdown("### 📈 보석 분석 통계")
        
        if not gemstone_results:
            st.info("보석 분석 데이터가 없습니다.")
            return
        
        # 성공한 분석 결과만 필터링
        successful_results = [r for r in gemstone_results if 'error' not in r]
        
        if not successful_results:
            st.warning("성공한 보석 분석 결과가 없습니다.")
            return
        
        # 처리 시간 분석
        col1, col2 = st.columns(2)
        
        with col1:
            processing_times = [r.get('processing_time', 0) for r in successful_results]
            if processing_times:
                fig_times = px.bar(
                    x=[f"보석 {i+1}" for i in range(len(processing_times))],
                    y=processing_times,
                    title="💎 보석별 처리 시간",
                    labels={'x': '보석', 'y': '처리 시간 (초)'}
                )
                fig_times.update_layout(height=300)
                st.plotly_chart(fig_times, use_container_width=True)
        
        with col2:
            # 색상 다양성 분석
            all_colors = []
            for result in successful_results:
                colors = result.get('dominant_colors', [])
                for color in colors:
                    all_colors.append(color.get('description', 'Unknown'))
            
            if all_colors:
                from collections import Counter
                color_counts = Counter(all_colors)
                top_colors = dict(color_counts.most_common(8))
                
                fig_colors = px.pie(
                    values=list(top_colors.values()),
                    names=list(top_colors.keys()),
                    title="🎨 주요 색상 분포"
                )
                fig_colors.update_layout(height=300)
                st.plotly_chart(fig_colors, use_container_width=True)
        
        # 산지 추정 분석
        col1, col2 = st.columns(2)
        
        with col1:
            # 산지별 통계
            origin_data = []
            for result in successful_results:
                origins = result.get('origin_analysis', {}).get('estimated_origins', [])
                for origin, score in origins:
                    origin_data.append({
                        'origin': origin,
                        'score': score,
                        'filename': result.get('filename', '')
                    })
            
            if origin_data:
                df_origins = pd.DataFrame(origin_data)
                origin_avg = df_origins.groupby('origin')['score'].mean().sort_values(ascending=False)
                
                fig_origins = px.bar(
                    x=origin_avg.index,
                    y=origin_avg.values,
                    title="🌍 산지별 평균 추정 점수",
                    labels={'x': '산지', 'y': '평균 점수'}
                )
                fig_origins.update_layout(height=300)
                st.plotly_chart(fig_origins, use_container_width=True)
        
        with col2:
            # 신뢰도 분포
            confidence_levels = []
            for result in successful_results:
                confidence = result.get('origin_analysis', {}).get('confidence_level', 'Unknown')
                confidence_levels.append(confidence)
            
            if confidence_levels:
                from collections import Counter
                conf_counts = Counter(confidence_levels)
                
                fig_conf = px.pie(
                    values=list(conf_counts.values()),
                    names=list(conf_counts.keys()),
                    title="🎯 신뢰도 수준 분포"
                )
                fig_conf.update_layout(height=300)
                st.plotly_chart(fig_conf, use_container_width=True)
        
        # 성능 메트릭 요약
        st.markdown("### 📊 성능 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_time = np.mean([r.get('processing_time', 0) for r in successful_results])
            st.metric("평균 처리 시간", f"{avg_time:.2f}초")
        
        with col2:
            total_colors = sum(len(r.get('dominant_colors', [])) for r in successful_results)
            st.metric("총 추출 색상", f"{total_colors}개")
        
        with col3:
            high_conf = len([r for r in successful_results if r.get('origin_analysis', {}).get('confidence_level') == 'High'])
            st.metric("높은 신뢰도", f"{high_conf}개")
        
        with col4:
            success_rate = len(successful_results) / len(gemstone_results) * 100
            st.metric("분석 성공률", f"{success_rate:.1f}%")

class EnhancedResultDisplay:
    """향상된 결과 표시 UI"""
    
    def __init__(self):
        self.preview_ui = ResultPreviewUI()
        self.analytics_ui = AnalyticsUI()
    
    def show_comprehensive_results(self, analysis_results: Dict):
        """종합 결과 표시"""
        audio_results = analysis_results.get('audio', [])
        image_results = analysis_results.get('images', [])
        summary = analysis_results.get('summary')
        
        # 탭으로 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📋 요약", "🎵 음성", "🖼️ 이미지", "📊 분석"])
        
        with tab1:
            self.show_executive_summary(audio_results, image_results, summary)
        
        with tab2:
            if audio_results:
                # 미리보기 컨테이너 초기화
                if self.preview_ui:
                    self.preview_ui.initialize_preview_display()
                    for i, result in enumerate(audio_results):
                        self.preview_ui.show_audio_preview(result, f"음성파일_{i+1}")
            else:
                st.info("음성 분석 결과가 없습니다.")
        
        with tab3:
            if image_results:
                # 미리보기 컨테이너 초기화
                if self.preview_ui:
                    self.preview_ui.initialize_preview_display()
                    self.preview_ui.show_image_preview(image_results)
            else:
                st.info("이미지 분석 결과가 없습니다.")
        
        with tab4:
            self.analytics_ui.show_processing_analytics(audio_results, image_results)
    
    def show_gemstone_comprehensive_results(self, results: Dict):
        """보석 분석 종합 결과 표시"""
        gemstone_results = results.get('gemstone_analysis', [])
        summary = results.get('summary', '')
        
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📋 요약", "💎 보석 분석", "🌍 산지 추정", "📊 통계"])
        
        with tab1:
            self.show_gemstone_executive_summary(gemstone_results, summary)
        
        with tab2:
            if gemstone_results:
                # 미리보기 컨테이너 초기화
                if self.preview_ui:
                    self.preview_ui.initialize_preview_display()
                    self.preview_ui.show_gemstone_preview(gemstone_results)
            else:
                st.info("보석 분석 결과가 없습니다.")
        
        with tab3:
            self.show_origin_analysis(gemstone_results)
        
        with tab4:
            self.analytics_ui.show_gemstone_analytics(gemstone_results)
    
    def show_gemstone_executive_summary(self, gemstone_results: List[Dict], summary: str):
        """보석 분석 임원 요약 표시"""
        st.markdown("### 💎 보석 분석 요약")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 분석 결과")
            if gemstone_results:
                successful = len([r for r in gemstone_results if 'error' not in r])
                total_colors = sum(len(r.get('dominant_colors', [])) for r in gemstone_results if 'error' not in r)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("분석된 보석", f"{successful}개")
                with col_b:
                    st.metric("추출된 색상", f"{total_colors}개")
            else:
                st.info("보석 분석 데이터 없음")
        
        with col2:
            st.markdown("#### ⚡ 성능")
            if gemstone_results:
                avg_time = np.mean([r.get('processing_time', 0) for r in gemstone_results if 'error' not in r])
                origins_estimated = len([r for r in gemstone_results if 'error' not in r and r.get('origin_analysis', {}).get('estimated_origins')])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("평균 처리시간", f"{avg_time:.2f}초")
                with col_b:
                    st.metric("산지 추정 완료", f"{origins_estimated}개")
        
        if summary:
            st.markdown("#### 🤖 AI 종합 분석")
            st.markdown(summary)
        
        st.markdown("#### 💡 추천 액션")
        if gemstone_results:
            high_confidence_count = len([r for r in gemstone_results if 'error' not in r and r.get('origin_analysis', {}).get('confidence_level') == 'High'])
            if high_confidence_count > 0:
                st.markdown(f"• 💎 {high_confidence_count}개 보석의 산지 추정 신뢰도가 높습니다.")
            else:
                st.markdown("• 🔍 추가적인 전문 감정을 권장합니다.")
            
            # 색상 다양성 분석
            all_colors = []
            for result in gemstone_results:
                if 'error' not in result:
                    colors = result.get('dominant_colors', [])
                    all_colors.extend([c.get('description', '') for c in colors])
            
            unique_colors = len(set(all_colors))
            if unique_colors > 10:
                st.markdown(f"• 🌈 다양한 색상 ({unique_colors}종)이 발견되어 컬렉션의 다양성이 우수합니다.")
    
    def show_origin_analysis(self, gemstone_results: List[Dict]):
        """산지 분석 결과 표시"""
        st.markdown("### 🌍 산지 추정 분석")
        
        if not gemstone_results:
            st.info("산지 분석 데이터가 없습니다.")
            return
        
        # 산지별 통계
        origin_stats = {}
        for result in gemstone_results:
            if 'error' not in result:
                origins = result.get('origin_analysis', {}).get('estimated_origins', [])
                for origin, score in origins:
                    if origin not in origin_stats:
                        origin_stats[origin] = {'count': 0, 'total_score': 0, 'files': []}
                    origin_stats[origin]['count'] += 1
                    origin_stats[origin]['total_score'] += score
                    origin_stats[origin]['files'].append(result.get('filename', ''))
        
        if origin_stats:
            st.markdown("#### 📊 산지별 통계")
            
            # 상위 산지들
            sorted_origins = sorted(origin_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for origin, stats in sorted_origins[:5]:  # 상위 5개만 표시
                avg_score = stats['total_score'] / stats['count']
                with st.expander(f"🌍 {origin} ({stats['count']}개 보석, 평균 점수: {avg_score:.1f})"):
                    st.markdown(f"**해당 보석들:** {', '.join(stats['files'][:5])}{'...' if len(stats['files']) > 5 else ''}")
                    
                    # 신뢰도 표시
                    if avg_score > 50:
                        st.success("🔥 높은 신뢰도")
                    elif avg_score > 30:
                        st.warning("⚠️ 중간 신뢰도")
                    else:
                        st.info("ℹ️ 낮은 신뢰도 - 추가 검증 필요")
        else:
            st.info("산지 추정 결과가 없습니다.")
    
    def show_executive_summary(self, audio_results: List[Dict], image_results: List[Dict], summary: str):
        """임원 요약 표시"""
        st.markdown("### 📋 분석 요약")
        
        # 핵심 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎵 음성 분석")
            if audio_results:
                total_text = sum(len(r.get('text', '')) for r in audio_results if 'error' not in r)
                avg_time = np.mean([r.get('processing_time', 0) for r in audio_results if 'error' not in r])
                st.metric("총 텍스트", f"{total_text:,} 문자")
                st.metric("평균 처리시간", f"{avg_time:.2f}초")
            else:
                st.info("음성 데이터 없음")
        
        with col2:
            st.markdown("#### 🖼️ 이미지 분석")
            if image_results:
                total_blocks = sum(r.get('total_blocks', 0) for r in image_results if 'error' not in r)
                successful = len([r for r in image_results if 'error' not in r])
                st.metric("추출된 블록", f"{total_blocks:,}개")
                st.metric("성공률", f"{successful}/{len(image_results)}")
            else:
                st.info("이미지 데이터 없음")
        
        with col3:
            st.markdown("#### ⚡ 성능")
            total_files = len(audio_results) + len(image_results)
            if total_files > 0:
                total_time = sum(r.get('processing_time', 0) for r in audio_results + image_results if 'error' not in r)
                st.metric("총 파일", f"{total_files}개")
                st.metric("총 처리시간", f"{total_time:.2f}초")
        
        # AI 요약
        if summary:
            st.markdown("### 🤖 AI 종합 요약")
            with st.container():
                st.markdown(f"""
                <div style="
                    padding: 20px; 
                    background-color: #f0f2f6; 
                    border-radius: 10px; 
                    border-left: 5px solid #ff6b6b;
                    margin: 10px 0;
                ">
                {summary}
                </div>
                """, unsafe_allow_html=True)
        
        # 추천 액션
        st.markdown("### 💡 추천 액션")
        recommendations = self.generate_recommendations(audio_results, image_results)
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    def generate_recommendations(self, audio_results: List[Dict], image_results: List[Dict]) -> List[str]:
        """추천 액션 생성"""
        recommendations = []
        
        # 음성 분석 기반 추천
        if audio_results:
            avg_confidence = np.mean([
                len(r.get('text', '')) > 100 for r in audio_results if 'error' not in r
            ])
            if avg_confidence > 0.8:
                recommendations.append("🎯 음성 인식 품질이 우수합니다. 상세 분석을 권장합니다.")
            else:
                recommendations.append("⚠️ 음성 품질이 낮습니다. 노이즈 제거 후 재분석을 권장합니다.")
        
        # 이미지 분석 기반 추천
        if image_results:
            successful_rate = len([r for r in image_results if 'error' not in r]) / len(image_results)
            if successful_rate > 0.9:
                recommendations.append("✅ 이미지 처리 성공률이 높습니다. 배치 처리 활용을 권장합니다.")
            else:
                recommendations.append("📷 일부 이미지 처리에 실패했습니다. 이미지 품질을 확인해주세요.")
        
        # 성능 기반 추천
        total_files = len(audio_results) + len(image_results)
        if total_files > 10:
            recommendations.append("🚀 대용량 처리에 적합합니다. GPU 가속 활용을 권장합니다.")
        
        return recommendations or ["📊 추가 데이터로 더 정확한 분석이 가능합니다."]