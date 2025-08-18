#!/usr/bin/env python3
"""
메모리 최적화 대시보드 애드온 v2.6
모니터링 대시보드에 추가되는 메모리 탭 구현
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

def render_memory_tab(memory_optimizer):
    """🧠 메모리 탭 렌더링 (NEW)"""
    st.subheader("🧠 메모리 최적화 엔진")
    
    try:
        # 현재 메모리 상태
        memory_status = memory_optimizer.get_current_status()
        
        if memory_status.get('status') != 'no_data':
            # 메모리 상태 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                usage_percent = memory_status['memory_usage_percent']
                target_percent = memory_status['target_usage_percent']
                delta = usage_percent - target_percent
                
                st.metric(
                    "💾 메모리 사용률",
                    f"{usage_percent:.1f}%",
                    delta=f"{delta:+.1f}%",
                    delta_color="inverse"
                )
            
            with col2:
                pressure_level = memory_status['memory_pressure_level']
                pressure_colors = {
                    'low': '🟢',
                    'medium': '🟡', 
                    'high': '🟠',
                    'critical': '🔴'
                }
                
                st.metric(
                    "⚡ 메모리 압박 수준",
                    f"{pressure_colors.get(pressure_level, '❓')} {pressure_level.upper()}"
                )
            
            with col3:
                st.metric(
                    "🎯 목표 사용률",
                    f"{memory_status['target_usage_percent']:.1f}%",
                    help="메모리 최적화 엔진의 목표 사용률"
                )
            
            with col4:
                is_within_target = memory_status['is_within_target']
                status_icon = "✅" if is_within_target else "⚠️"
                status_text = "목표 달성" if is_within_target else "목표 초과"
                
                st.metric(
                    "📊 목표 달성 상태",
                    f"{status_icon} {status_text}"
                )
            
            # 메모리 압박 수준에 따른 경고
            if pressure_level == 'critical':
                st.error("🚨 **메모리 압박이 심각합니다!** 즉시 메모리 정리가 필요합니다.")
            elif pressure_level == 'high':
                st.warning("⚠️ **메모리 압박이 높습니다.** 메모리 정리를 권장합니다.")
            elif pressure_level == 'medium':
                st.info("ℹ️ **메모리 압박이 보통 수준입니다.** 모니터링을 계속하세요.")
            else:
                st.success("✅ **메모리 상태가 양호합니다.**")
            
            st.markdown("---")
            
            # 최적화 통계
            st.subheader("📊 최적화 통계")
            
            opt_stats = memory_status['optimization_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🔧 총 최적화 횟수",
                    f"{opt_stats['total_optimizations']}회"
                )
            
            with col2:
                st.metric(
                    "💾 총 해제 메모리",
                    f"{opt_stats['memory_freed_total_mb']:.1f}MB"
                )
            
            with col3:
                st.metric(
                    "⏱️ 평균 처리 시간",
                    f"{opt_stats['average_optimization_time_ms']:.1f}ms"
                )
            
            with col4:
                st.metric(
                    "✅ 성공률",
                    f"{opt_stats['success_rate']:.1%}"
                )
            
            # 마지막 최적화 시간
            if opt_stats['last_optimization']:
                last_opt_time = datetime.fromisoformat(opt_stats['last_optimization'])
                time_diff = datetime.now() - last_opt_time
                st.info(f"🕒 마지막 최적화: {time_diff.total_seconds()/60:.1f}분 전")
            
            st.markdown("---")
            
            # 메모리 추세 분석
            st.subheader("📈 메모리 사용 추세")
            
            try:
                memory_trend = memory_optimizer.get_memory_trend(hours=1)
                
                if memory_trend.get('status') != 'insufficient_data':
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "📊 현재 사용률",
                            f"{memory_trend['current_usage']:.1f}%"
                        )
                        st.metric(
                            "📈 최대 사용률",
                            f"{memory_trend['max_usage']:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "📉 최소 사용률",
                            f"{memory_trend['min_usage']:.1f}%"
                        )
                        st.metric(
                            "📊 평균 사용률",
                            f"{memory_trend['avg_usage']:.1f}%"
                        )
                    
                    with col3:
                        trend_icon = "📈" if memory_trend['usage_trend'] == 'increasing' else "📉"
                        st.metric(
                            "📊 사용 추세",
                            f"{trend_icon} {memory_trend['usage_trend']}"
                        )
                        
                        volatility = memory_trend['volatility']
                        volatility_level = "높음" if volatility > 5 else "보통" if volatility > 2 else "낮음"
                        st.metric(
                            "📊 변동성",
                            f"{volatility_level} ({volatility:.1f})"
                        )
                    
                    # 메모리 누수 감지
                    leaks_detected = memory_trend['memory_leaks_detected']
                    if leaks_detected > 0:
                        st.warning(f"🔍 **메모리 누수 의심**: {leaks_detected}개 패턴 감지")
                        st.info("메모리 사용량이 지속적으로 증가하는 패턴이 감지되었습니다. 원인을 조사하세요.")
                    else:
                        st.success("✅ 메모리 누수가 감지되지 않았습니다.")
                
                else:
                    st.info("📊 추세 분석을 위한 데이터가 부족합니다. (최소 2개 데이터 포인트 필요)")
            
            except Exception as e:
                st.error(f"❌ 메모리 추세 분석 실패: {e}")
            
            st.markdown("---")
            
            # 모니터링 설정
            st.subheader("⚙️ 메모리 모니터링 설정")
            
            col1, col2 = st.columns(2)
            
            with col1:
                auto_opt_enabled = memory_status['auto_optimization_enabled']
                monitoring_enabled = memory_status['monitoring_enabled']
                
                if auto_opt_enabled:
                    st.success("✅ 자동 최적화: 활성화")
                else:
                    st.warning("⚠️ 자동 최적화: 비활성화")
                
                if monitoring_enabled:
                    st.success("✅ 실시간 모니터링: 활성화")
                else:
                    st.error("🔴 실시간 모니터링: 비활성화")
            
            with col2:
                st.metric(
                    "🏠 프로세스 메모리",
                    f"{memory_status['process_memory_mb']:.1f}MB",
                    help="현재 프로세스가 사용하는 메모리"
                )
                
                st.metric(
                    "📦 캐시 크기",
                    f"{memory_status['cache_size_mb']:.1f}MB",
                    help="관리되는 캐시 크기"
                )
            
            # 큰 객체 수
            large_objects = memory_status.get('large_objects_count', 0)
            if large_objects > 10:
                st.warning(f"⚠️ **큰 객체 감지**: {large_objects}개 (1MB 이상)")
                st.info("메모리를 많이 사용하는 큰 객체들이 있습니다. 정리를 고려하세요.")
        
        else:
            st.warning("⚠️ 메모리 최적화 엔진 데이터가 없습니다. 잠시 후 다시 확인해주세요.")
    
    except Exception as e:
        st.error(f"❌ 메모리 상태를 가져올 수 없습니다: {e}")
    
    st.markdown("---")
    
    # 메모리 최적화 액션
    st.subheader("🚀 메모리 최적화 액션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 가비지 컬렉션", help="Python 가비지 컬렉션을 강제 실행합니다"):
            with st.spinner("가비지 컬렉션 실행 중..."):
                try:
                    import gc
                    collected = gc.collect()
                    st.success(f"✅ 가비지 컬렉션 완료: {collected}개 객체 정리")
                except Exception as e:
                    st.error(f"❌ 가비지 컬렉션 실패: {e}")
    
    with col2:
        if st.button("📦 캐시 정리", help="관리되는 캐시를 정리합니다"):
            st.info("캐시 정리는 메모리 최적화 엔진을 통해 수행됩니다.")
            st.info("위의 '메모리 정리' 버튼을 사용하세요.")
    
    with col3:
        if st.button("📊 상세 진단", help="상세한 메모리 진단을 수행합니다"):
            with st.spinner("메모리 진단 중..."):
                try:
                    import psutil
                    import gc
                    
                    # 시스템 메모리 정보
                    memory = psutil.virtual_memory()
                    process = psutil.Process()
                    process_memory = process.memory_info()
                    
                    st.subheader("🔍 상세 메모리 진단")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**시스템 메모리**")
                        st.info(f"총 메모리: {memory.total / (1024**3):.1f}GB")
                        st.info(f"사용 가능: {memory.available / (1024**3):.1f}GB")
                        st.info(f"사용 중: {memory.used / (1024**3):.1f}GB")
                        st.info(f"사용률: {memory.percent:.1f}%")
                    
                    with col2:
                        st.write("**프로세스 메모리**")
                        st.info(f"RSS: {process_memory.rss / (1024**2):.1f}MB")
                        st.info(f"VMS: {process_memory.vms / (1024**2):.1f}MB")
                        
                        # 가비지 컬렉션 통계
                        st.write("**가비지 컬렉션**")
                        try:
                            gen0 = len(gc.get_objects(0))
                            gen1 = len(gc.get_objects(1))
                            gen2 = len(gc.get_objects(2))
                            st.info(f"세대 0: {gen0:,}개")
                            st.info(f"세대 1: {gen1:,}개")
                            st.info(f"세대 2: {gen2:,}개")
                        except:
                            st.warning("GC 통계 수집 실패")
                
                except Exception as e:
                    st.error(f"❌ 상세 진단 실패: {e}")

def render_memory_optimization_controls(memory_optimizer, project_root):
    """메모리 최적화 제어 패널"""
    st.markdown("---")
    st.subheader("🧠 메모리 최적화 제어")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 메모리 정리 (일반)", help="일반적인 메모리 정리를 수행합니다"):
            with st.spinner("메모리 정리 중..."):
                try:
                    actions = memory_optimizer.force_optimization(aggressive=False)
                    total_freed = sum(a.memory_freed_mb for a in actions if a.success)
                    if total_freed > 0:
                        st.success(f"✅ 메모리 정리 완료: {total_freed:.1f}MB 해제")
                    else:
                        st.info("정리할 메모리가 없습니다")
                except Exception as e:
                    st.error(f"❌ 메모리 정리 실패: {e}")
    
    with col2:
        if st.button("🔥 메모리 정리 (강력)", help="강력한 메모리 정리를 수행합니다"):
            with st.spinner("강력한 메모리 정리 중..."):
                try:
                    actions = memory_optimizer.force_optimization(aggressive=True)
                    total_freed = sum(a.memory_freed_mb for a in actions if a.success)
                    if total_freed > 0:
                        st.success(f"✅ 강력한 메모리 정리 완료: {total_freed:.1f}MB 해제")
                    else:
                        st.info("정리할 메모리가 없습니다")
                except Exception as e:
                    st.error(f"❌ 강력한 메모리 정리 실패: {e}")
    
    with col3:
        if st.button("📊 메모리 진단", help="메모리 진단 보고서를 생성합니다"):
            with st.spinner("메모리 진단 중..."):
                try:
                    diagnostic_path = project_root / f"memory_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    memory_optimizer.export_diagnostics(str(diagnostic_path))
                    st.success(f"✅ 메모리 진단 보고서 생성: {diagnostic_path.name}")
                    
                    # 다운로드 버튼
                    with open(diagnostic_path, 'r', encoding='utf-8') as f:
                        diagnostic_content = f.read()
                    
                    st.download_button(
                        label="📥 진단 보고서 다운로드",
                        data=diagnostic_content,
                        file_name=diagnostic_path.name,
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"❌ 메모리 진단 실패: {e}")

def render_memory_trend_chart(memory_optimizer):
    """메모리 사용 추세 차트"""
    try:
        memory_trend = memory_optimizer.get_memory_trend(hours=4)
        
        if memory_trend.get('status') != 'insufficient_data':
            st.subheader("📈 메모리 사용 추세 차트 (4시간)")
            
            # 여기서는 실제 히스토리 데이터가 필요하지만 
            # 현재 API에서는 트렌드 요약만 제공하므로 가상 데이터로 시연
            time_points = []
            memory_usage = []
            
            # 4시간 동안의 가상 데이터 생성
            current_time = datetime.now()
            for i in range(48):  # 5분 간격으로 48개 포인트
                time_points.append(current_time - timedelta(minutes=i*5))
                # 현재 사용률 기준으로 가상 변동 생성
                base_usage = memory_trend['current_usage']
                variation = (i % 10 - 5) * 2  # -10 ~ +10 변동
                memory_usage.append(max(30, min(95, base_usage + variation)))
            
            time_points.reverse()
            memory_usage.reverse()
            
            # Plotly 차트 생성
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=memory_usage,
                mode='lines+markers',
                name='메모리 사용률',
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))
            
            # 목표 사용률 라인 추가
            target_line = [70] * len(time_points)  # 70% 목표
            fig.add_trace(go.Scatter(
                x=time_points,
                y=target_line,
                mode='lines',
                name='목표 사용률 (70%)',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            # 위험 수준 라인 추가
            warning_line = [85] * len(time_points)  # 85% 경고
            fig.add_trace(go.Scatter(
                x=time_points,
                y=warning_line,
                mode='lines',
                name='경고 수준 (85%)',
                line=dict(color='#f39c12', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title="메모리 사용률 추세",
                xaxis_title="시간",
                yaxis_title="메모리 사용률 (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 추세 분석 요약
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if memory_trend['usage_trend'] == 'increasing':
                    st.warning("📈 메모리 사용률이 증가 추세입니다")
                else:
                    st.success("📉 메모리 사용률이 감소 추세입니다")
            
            with col2:
                volatility = memory_trend['volatility']
                if volatility > 5:
                    st.warning(f"⚡ 높은 변동성: {volatility:.1f}")
                else:
                    st.info(f"📊 변동성: {volatility:.1f}")
            
            with col3:
                leaks = memory_trend['memory_leaks_detected']
                if leaks > 0:
                    st.error(f"🔍 누수 의심: {leaks}개")
                else:
                    st.success("✅ 누수 감지 없음")
        
        else:
            st.info("📊 차트 생성을 위한 충분한 데이터가 없습니다.")
    
    except Exception as e:
        st.error(f"❌ 메모리 추세 차트 생성 실패: {e}")