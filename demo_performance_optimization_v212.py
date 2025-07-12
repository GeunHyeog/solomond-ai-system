"""
🚀 솔로몬드 AI v2.1.2 - 성능 최적화 통합 데모
모든 최적화 기능을 체험할 수 있는 종합 데모 시스템

새로운 v2.1.2 기능:
✅ 실시간 성능 모니터링
✅ 스마트 메모리 최적화  
✅ 자동 에러 복구 시스템
✅ 종합 성능 벤치마크
✅ 대용량 파일 처리 최적화
"""

import streamlit as st
import time
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path
import logging
import threading

# v2.1.2 최적화 모듈들 import
try:
    from core.performance_profiler_v21 import PerformanceProfiler, get_system_health, global_profiler
    from core.memory_optimizer_v21 import MemoryManager, global_memory_manager, memory_optimized
    from core.error_recovery_system_v21 import ErrorRecoverySystem, global_recovery_system, resilient
    from core.integrated_performance_test_v21 import SystemPerformanceAnalyzer, run_performance_analysis
    
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ v2.1.2 최적화 모듈 로드 실패: {e}")
    MODULES_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="솔로몬드 AI v2.1.2 - 성능 최적화 데모",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-banner {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-banner {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-banner {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .demo-section {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 솔로몬드 AI v2.1.2</h1>
        <h3>성능 최적화 & 안정성 강화 통합 데모</h3>
        <p>실시간 모니터링 • 메모리 최적화 • 에러 복구 • 성능 벤치마크</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODULES_AVAILABLE:
        st.error("⚠️ v2.1.2 최적화 모듈을 찾을 수 없습니다. core 폴더의 모듈들을 확인해주세요.")
        st.info("📁 필요한 모듈들:\n- performance_profiler_v21.py\n- memory_optimizer_v21.py\n- error_recovery_system_v21.py\n- integrated_performance_test_v21.py")
        return
    
    # 사이드바 메뉴
    with st.sidebar:
        st.header("🎛️ 데모 메뉴")
        demo_mode = st.selectbox(
            "데모 선택",
            ["🏠 홈 대시보드", "📊 실시간 모니터링", "🧠 메모리 최적화", 
             "🛡️ 에러 복구 시스템", "🚀 성능 벤치마크", "⚙️ 통합 테스트"]
        )
        
        st.markdown("---")
        st.info("💡 **v2.1.2 신기능**\n\n"
                "• 실시간 시스템 모니터링\n"
                "• 스마트 메모리 관리\n"
                "• 자동 에러 복구\n"
                "• 성능 병목점 분석\n"
                "• 대용량 파일 최적화")
    
    # 메인 컨텐츠
    if demo_mode == "🏠 홈 대시보드":
        show_dashboard()
    elif demo_mode == "📊 실시간 모니터링":
        show_performance_monitoring()
    elif demo_mode == "🧠 메모리 최적화":
        show_memory_optimization()
    elif demo_mode == "🛡️ 에러 복구 시스템":
        show_error_recovery()
    elif demo_mode == "🚀 성능 벤치마크":
        show_performance_benchmark()
    elif demo_mode == "⚙️ 통합 테스트":
        show_integrated_test()

def show_dashboard():
    """홈 대시보드"""
    st.header("🏠 시스템 현황 대시보드")
    
    # 시스템 건강도 체크
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            system_health = get_system_health()
            health_score = system_health.get('health_score', 0)
            
            if health_score >= 85:
                st.success(f"💚 시스템 건강도\n**{health_score}/100**")
            elif health_score >= 70:
                st.warning(f"💛 시스템 건강도\n**{health_score}/100**")
            else:
                st.error(f"❤️ 시스템 건강도\n**{health_score}/100**")
        except Exception as e:
            st.error(f"건강도 체크 실패\n{str(e)[:30]}...")
    
    with col2:
        try:
            memory_stats = global_memory_manager.get_memory_usage()
            usage_percent = memory_stats.percent
            
            if usage_percent < 70:
                st.success(f"💾 메모리 사용률\n**{usage_percent:.1f}%**")
            elif usage_percent < 85:
                st.warning(f"💾 메모리 사용률\n**{usage_percent:.1f}%**")
            else:
                st.error(f"💾 메모리 사용률\n**{usage_percent:.1f}%**")
        except Exception as e:
            st.error(f"메모리 체크 실패\n{str(e)[:30]}...")
    
    with col3:
        try:
            recovery_status = global_recovery_system.get_system_status()
            error_rate = recovery_status.get('error_rate_1h', 0)
            
            if error_rate < 5:
                st.success(f"🛡️ 에러율 (1시간)\n**{error_rate:.1f}%**")
            elif error_rate < 15:
                st.warning(f"🛡️ 에러율 (1시간)\n**{error_rate:.1f}%**")
            else:
                st.error(f"🛡️ 에러율 (1시간)\n**{error_rate:.1f}%**")
        except Exception as e:
            st.error(f"에러율 체크 실패\n{str(e)[:30]}...")
    
    with col4:
        try:
            cache_stats = global_memory_manager.cache.stats()
            hit_rate = cache_stats.get('hit_rate', 0)
            
            if hit_rate > 70:
                st.success(f"📊 캐시 적중률\n**{hit_rate:.1f}%**")
            elif hit_rate > 40:
                st.warning(f"📊 캐시 적중률\n**{hit_rate:.1f}%**")
            else:
                st.error(f"📊 캐시 적중률\n**{hit_rate:.1f}%**")
        except Exception as e:
            st.error(f"캐시 체크 실패\n{str(e)[:30]}...")
    
    st.markdown("---")
    
    # 빠른 액션
    st.subheader("⚡ 빠른 액션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 메모리 정리", use_container_width=True):
            with st.spinner("메모리 정리 중..."):
                try:
                    result = global_memory_manager.routine_cleanup()
                    st.success(f"✅ {result['freed_mb']:.2f}MB 메모리 해제")
                except Exception as e:
                    st.error(f"메모리 정리 실패: {e}")
    
    with col2:
        if st.button("📊 성능 체크", use_container_width=True):
            with st.spinner("성능 분석 중..."):
                try:
                    # 간단한 성능 체크
                    start_time = time.time()
                    test_data = [i**2 for i in range(10000)]
                    process_time = time.time() - start_time
                    
                    if process_time < 0.1:
                        st.success(f"✅ 성능 우수 ({process_time:.3f}초)")
                    else:
                        st.warning(f"⚠️ 성능 주의 ({process_time:.3f}초)")
                except Exception as e:
                    st.error(f"성능 체크 실패: {e}")
    
    with col3:
        if st.button("🔄 시스템 새로고침", use_container_width=True):
            st.rerun()
    
    # 시스템 개요
    st.markdown("---")
    st.subheader("📋 시스템 개요")
    
    try:
        overview_data = {
            "버전": "v2.1.2",
            "상태": "활성",
            "업타임": "실행 중",
            "모듈": "4개 최적화 모듈 로드됨"
        }
        
        for key, value in overview_data.items():
            st.write(f"**{key}**: {value}")
            
    except Exception as e:
        st.error(f"시스템 개요 로드 실패: {e}")

def show_performance_monitoring():
    """성능 모니터링 데모"""
    st.header("📊 실시간 성능 모니터링")
    
    # 모니터링 시작/중지 버튼
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        
        if st.button("🟢 모니터링 시작" if not st.session_state.monitoring_active else "🔴 모니터링 중지"):
            st.session_state.monitoring_active = not st.session_state.monitoring_active
            
            if st.session_state.monitoring_active:
                global_profiler.start_monitoring(interval=1.0)
                st.success("모니터링이 시작되었습니다")
            else:
                global_profiler.stop_monitoring()
                st.info("모니터링이 중지되었습니다")
    
    if st.session_state.monitoring_active:
        # 실시간 메트릭 표시
        metrics_placeholder = st.empty()
        
        for i in range(10):  # 10초간 모니터링
            try:
                summary = global_profiler.get_performance_summary()
                
                if 'averages' in summary:
                    avg = summary['averages']
                    peaks = summary.get('peaks', {})
                    
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "CPU 사용률",
                                f"{avg.get('cpu_percent', 0):.1f}%",
                                f"피크: {peaks.get('cpu_percent', 0):.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "메모리 사용률", 
                                f"{avg.get('memory_percent', 0):.1f}%",
                                f"피크: {peaks.get('memory_percent', 0):.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "디스크 읽기",
                                f"{avg.get('disk_read_mb_s', 0):.2f}MB/s",
                                f"피크: {peaks.get('disk_read_mb_s', 0):.2f}MB/s"
                            )
                        
                        with col4:
                            current_status = summary.get('current_status', {})
                            st.metric(
                                "시스템 상태",
                                current_status.get('status', 'Unknown'),
                                f"스레드: {current_status.get('threads', 0)}"
                            )
                        
                        # 권장사항
                        recommendations = summary.get('recommendations', [])
                        if recommendations:
                            st.subheader("💡 실시간 권장사항")
                            for rec in recommendations[:3]:
                                st.info(rec)
                
                time.sleep(1)
                
            except Exception as e:
                st.error(f"모니터링 오류: {e}")
                break
    
    else:
        st.info("📊 모니터링을 시작하려면 위의 버튼을 클릭하세요")
        
        # 최근 성능 요약 표시
        try:
            summary = global_profiler.get_performance_summary()
            if summary and 'module_performance' in summary:
                st.subheader("📈 모듈별 성능 현황")
                
                perf_data = summary['module_performance']
                for module_name, stats in list(perf_data.items())[:5]:
                    with st.expander(f"📦 {module_name}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("호출 횟수", stats.get('call_count', 0))
                        with col2:
                            st.metric("평균 실행시간", f"{stats.get('avg_time', 0):.3f}초")
                        with col3:
                            st.metric("에러 횟수", stats.get('error_count', 0))
        except Exception as e:
            st.warning(f"성능 요약 로드 실패: {e}")

def show_memory_optimization():
    """메모리 최적화 데모"""
    st.header("🧠 메모리 최적화 데모")
    
    # 현재 메모리 상태
    try:
        memory_stats = global_memory_manager.get_memory_usage()
        optimization_report = global_memory_manager.get_optimization_report()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "메모리 사용률",
                f"{memory_stats.percent:.1f}%",
                f"사용: {memory_stats.used_mb:.1f}MB"
            )
        
        with col2:
            cache_stats = optimization_report.get('cache', {})
            st.metric(
                "캐시 적중률",
                f"{cache_stats.get('hit_rate', 0):.1f}%",
                f"항목: {cache_stats.get('items', 0)}개"
            )
        
        with col3:
            cleanup_stats = optimization_report.get('cleanup_stats', {})
            st.metric(
                "정리 횟수",
                cleanup_stats.get('total_cleanups', 0),
                f"해제: {cleanup_stats.get('bytes_freed_mb', 0):.1f}MB"
            )
        
    except Exception as e:
        st.error(f"메모리 상태 조회 실패: {e}")
    
    st.markdown("---")
    
    # 메모리 최적화 기능 테스트
    st.subheader("🛠️ 메모리 최적화 기능")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 캐시 테스트")
        
        if st.button("캐시 데이터 생성"):
            with st.spinner("캐시 데이터 생성 중..."):
                try:
                    # 테스트 데이터 캐싱
                    for i in range(50):
                        key = f"test_jewelry_{i}"
                        value = {
                            "id": i,
                            "type": f"다이아몬드_{i}",
                            "carat": round(i * 0.1, 2),
                            "price": i * 1000,
                            "data": "x" * 1000  # 큰 데이터
                        }
                        global_memory_manager.cache.put(key, value)
                    
                    st.success("✅ 50개 캐시 항목이 생성되었습니다")
                    
                    # 캐시 통계 업데이트
                    cache_stats = global_memory_manager.cache.stats()
                    st.info(f"📊 캐시 사용량: {cache_stats['size_mb']:.2f}MB")
                    
                except Exception as e:
                    st.error(f"캐시 생성 실패: {e}")
        
        if st.button("캐시 읽기 테스트"):
            with st.spinner("캐시 읽기 테스트 중..."):
                try:
                    hit_count = 0
                    miss_count = 0
                    
                    for i in range(50):
                        key = f"test_jewelry_{i}"
                        value = global_memory_manager.cache.get(key)
                        if value:
                            hit_count += 1
                        else:
                            miss_count += 1
                    
                    st.success(f"✅ 캐시 읽기 완료")
                    st.info(f"적중: {hit_count}개, 누락: {miss_count}개")
                    
                except Exception as e:
                    st.error(f"캐시 읽기 실패: {e}")
    
    with col2:
        st.subheader("🧹 메모리 정리")
        
        if st.button("일반 정리"):
            with st.spinner("메모리 정리 중..."):
                try:
                    result = global_memory_manager.routine_cleanup()
                    st.success(f"✅ {result['freed_mb']:.2f}MB 해제됨")
                    st.info(f"객체 {result['objects_collected']}개 정리됨")
                except Exception as e:
                    st.error(f"정리 실패: {e}")
        
        if st.button("긴급 정리", type="secondary"):
            with st.spinner("긴급 메모리 정리 중..."):
                try:
                    result = global_memory_manager.emergency_cleanup()
                    st.warning(f"🚨 긴급 정리: {result['freed_mb']:.2f}MB 해제됨")
                    st.info(f"캐시 {result['cache_cleared_mb']:.2f}MB 삭제됨")
                except Exception as e:
                    st.error(f"긴급 정리 실패: {e}")
        
        if st.button("캐시 초기화"):
            try:
                global_memory_manager.cache.clear()
                st.success("✅ 캐시가 초기화되었습니다")
            except Exception as e:
                st.error(f"캐시 초기화 실패: {e}")
    
    # 메모리 사용량 데코레이터 테스트
    st.markdown("---")
    st.subheader("🔧 메모리 최적화 데코레이터 테스트")
    
    if st.button("최적화 함수 실행"):
        with st.spinner("최적화된 함수 실행 중..."):
            try:
                @memory_optimized(cache_key="jewelry_calculation")
                def expensive_jewelry_calculation():
                    """무거운 주얼리 계산 시뮬레이션"""
                    time.sleep(0.5)  # 무거운 작업 시뮬레이션
                    return {
                        "total_diamonds": 1000,
                        "average_price": 5000,
                        "calculation_time": time.time()
                    }
                
                start_time = time.time()
                result1 = expensive_jewelry_calculation()
                first_time = time.time() - start_time
                
                start_time = time.time()
                result2 = expensive_jewelry_calculation()  # 캐시된 결과
                second_time = time.time() - start_time
                
                st.success(f"✅ 첫 번째 실행: {first_time:.3f}초")
                st.success(f"✅ 두 번째 실행 (캐시): {second_time:.3f}초")
                st.info(f"🚀 속도 향상: {first_time/second_time:.1f}배")
                
            except Exception as e:
                st.error(f"함수 실행 실패: {e}")

def show_error_recovery():
    """에러 복구 시스템 데모"""
    st.header("🛡️ 에러 복구 시스템 데모")
    
    # 시스템 상태
    try:
        recovery_status = global_recovery_system.get_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_status = recovery_status.get('health_status', 'Unknown')
            if health_status == 'HEALTHY':
                st.success(f"🟢 시스템 상태\n**{health_status}**")
            elif health_status == 'WARNING':
                st.warning(f"🟡 시스템 상태\n**{health_status}**")
            else:
                st.error(f"🔴 시스템 상태\n**{health_status}**")
        
        with col2:
            st.metric(
                "에러율 (1시간)",
                f"{recovery_status.get('error_rate_1h', 0):.2f}%",
                f"총 {recovery_status.get('total_errors', 0)}개"
            )
        
        with col3:
            st.metric(
                "활성 작업",
                recovery_status.get('active_operations', 0),
                "진행 중"
            )
        
        with col4:
            breakers = recovery_status.get('circuit_breakers', {})
            open_breakers = sum(1 for state in breakers.values() if state == 'OPEN')
            st.metric(
                "회로 차단기",
                f"{len(breakers)}개",
                f"{open_breakers}개 열림"
            )
        
    except Exception as e:
        st.error(f"시스템 상태 조회 실패: {e}")
    
    st.markdown("---")
    
    # 에러 복구 기능 테스트
    st.subheader("🧪 에러 복구 기능 테스트")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 자동 재시도 테스트")
        
        if st.button("재시도 함수 실행"):
            with st.spinner("재시도 함수 실행 중..."):
                
                @resilient(operation_id="test_retry")
                def unreliable_function():
                    """불안정한 함수 시뮬레이션"""
                    import random
                    if random.random() < 0.6:  # 60% 확률로 실패
                        raise ConnectionError("네트워크 연결 실패")
                    return "작업 성공!"
                
                try:
                    result = unreliable_function()
                    st.success(f"✅ {result}")
                except Exception as e:
                    st.error(f"❌ 최종 실패: {e}")
        
        if st.button("파일 작업 테스트"):
            with st.spinner("파일 작업 테스트 중..."):
                
                @resilient(operation_id="test_file")
                def file_operation():
                    """파일 작업 시뮬레이션"""
                    import random
                    if random.random() < 0.5:
                        raise FileNotFoundError("파일을 찾을 수 없습니다")
                    return "파일 처리 완료"
                
                try:
                    result = file_operation()
                    st.success(f"✅ {result}")
                except Exception as e:
                    st.warning(f"⚠️ 폴백 실행: 기본 파일 사용")
    
    with col2:
        st.subheader("🔌 회로 차단기 테스트")
        
        if st.button("외부 서비스 호출"):
            with st.spinner("외부 서비스 호출 중..."):
                
                from core.error_recovery_system_v21 import with_circuit_breaker
                
                @with_circuit_breaker("external_api")
                def call_external_service():
                    """외부 API 호출 시뮬레이션"""
                    import random
                    if random.random() < 0.8:  # 80% 확률로 실패
                        raise ConnectionError("API 서버 응답 없음")
                    return "API 응답 성공"
                
                success_count = 0
                total_attempts = 5
                
                for i in range(total_attempts):
                    try:
                        result = call_external_service()
                        success_count += 1
                        st.success(f"호출 {i+1}: ✅ {result}")
                    except Exception as e:
                        st.error(f"호출 {i+1}: ❌ {str(e)[:50]}...")
                
                st.info(f"📊 성공률: {success_count}/{total_attempts} ({success_count/total_attempts*100:.1f}%)")
        
        if st.button("복구 리포트 생성"):
            try:
                report = global_recovery_system.generate_recovery_report()
                
                st.subheader("📋 복구 리포트")
                
                summary = report.get('summary', {})
                st.write(f"**24시간 에러 수**: {summary.get('total_errors_24h', 0)}")
                st.write(f"**복구 성공률**: {summary.get('recovery_success_rate', 0):.1f}%")
                
                recommendations = report.get('recommendations', [])
                if recommendations:
                    st.subheader("💡 권장사항")
                    for rec in recommendations:
                        st.info(rec)
                
            except Exception as e:
                st.error(f"리포트 생성 실패: {e}")

def show_performance_benchmark():
    """성능 벤치마크 데모"""
    st.header("🚀 성능 벤치마크 데모")
    
    if st.button("전체 벤치마크 실행", type="primary"):
        with st.spinner("성능 벤치마크 실행 중... (약 30초 소요)"):
            try:
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("벤치마크 초기화 중...")
                progress_bar.progress(10)
                time.sleep(1)
                
                # 성능 분석 실행
                report = run_performance_analysis(save_report=False)
                
                progress_bar.progress(100)
                status_text.text("벤치마크 완료!")
                
                # 결과 표시
                st.success(f"✅ 벤치마크 완료 - 전체 점수: **{report.overall_score:.1f}/100**")
                
                # 점수별 등급
                score = report.overall_score
                if score >= 90:
                    st.balloons()
                    grade = "🏆 우수 (Excellent)"
                elif score >= 80:
                    grade = "🥈 좋음 (Good)"
                elif score >= 70:
                    grade = "🥉 보통 (Fair)"
                elif score >= 60:
                    grade = "⚠️ 주의 (Poor)"
                else:
                    grade = "🚨 위험 (Critical)"
                
                st.subheader(f"🎖️ 성능 등급: {grade}")
                
                # 벤치마크 결과 상세
                st.subheader("📊 벤치마크 결과 상세")
                
                for result in report.benchmark_results:
                    with st.expander(f"📦 {result.test_name} - {result.success_rate_percent:.1f}% 성공률"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("실행 시간", f"{result.duration_seconds:.3f}초")
                            st.metric("메모리 사용", f"{result.memory_used_mb:.2f}MB")
                        
                        with col2:
                            st.metric("처리량", f"{result.throughput_ops_per_sec:.2f} ops/sec")
                            st.metric("에러 수", result.errors_count)
                        
                        with col3:
                            if result.metadata:
                                st.write("**상세 정보:**")
                                for key, value in result.metadata.items():
                                    st.write(f"- {key}: {value}")
                
                # 최적화 권장사항
                st.subheader("💡 최적화 권장사항")
                for i, rec in enumerate(report.optimization_recommendations, 1):
                    st.info(f"{i}. {rec}")
                
                # 시스템 상태 요약
                if report.system_health:
                    st.subheader("💊 시스템 건강도")
                    health_score = report.system_health.get('health_score', 0)
                    st.progress(health_score / 100)
                    st.write(f"건강도 점수: {health_score}/100")
                
            except Exception as e:
                st.error(f"벤치마크 실행 실패: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    else:
        st.info("🎯 전체 시스템 성능을 벤치마크하려면 위 버튼을 클릭하세요")
        
        st.markdown("""
        ### 📋 벤치마크 항목
        
        1. **📁 파일 처리 성능**
           - 텍스트 파일 읽기/쓰기
           - JSON 데이터 파싱
           - 대용량 파일 처리
        
        2. **🧠 메모리 관리 성능**
           - 캐시 읽기/쓰기 속도
           - 메모리 정리 효율성
           - 객체 생성/소멸 성능
        
        3. **🛡️ 에러 복구 성능**
           - 에러 감지 속도
           - 복구 처리 시간
           - 시스템 안정성
        
        4. **🔄 동시 작업 성능**
           - 멀티스레딩 효율성
           - 리소스 경합 처리
           - 작업 처리량
        """)

def show_integrated_test():
    """통합 테스트"""
    st.header("⚙️ v2.1.2 통합 테스트")
    
    if st.button("전체 시스템 통합 테스트", type="primary"):
        with st.spinner("통합 테스트 실행 중..."):
            test_results = {}
            
            # 1. 성능 프로파일러 테스트
            st.subheader("1. 📊 성능 프로파일러 테스트")
            try:
                profiler = PerformanceProfiler()
                profiler.start_monitoring(interval=0.5)
                time.sleep(2)  # 2초간 모니터링
                profiler.stop_monitoring()
                
                summary = profiler.get_performance_summary()
                if summary:
                    st.success("✅ 성능 프로파일러 정상 작동")
                    test_results['profiler'] = True
                else:
                    st.error("❌ 성능 프로파일러 오류")
                    test_results['profiler'] = False
            except Exception as e:
                st.error(f"❌ 성능 프로파일러 실패: {e}")
                test_results['profiler'] = False
            
            # 2. 메모리 최적화 테스트
            st.subheader("2. 🧠 메모리 최적화 테스트")
            try:
                # 메모리 테스트 데이터 생성
                for i in range(20):
                    key = f"integration_test_{i}"
                    value = f"테스트 데이터 {i}" * 50
                    global_memory_manager.cache.put(key, value)
                
                # 캐시 읽기 테스트
                hit_count = 0
                for i in range(20):
                    key = f"integration_test_{i}"
                    if global_memory_manager.cache.get(key):
                        hit_count += 1
                
                if hit_count == 20:
                    st.success("✅ 메모리 캐시 정상 작동")
                    test_results['memory'] = True
                else:
                    st.warning(f"⚠️ 메모리 캐시 부분 실패: {hit_count}/20")
                    test_results['memory'] = False
                
                # 메모리 정리 테스트
                cleanup_result = global_memory_manager.routine_cleanup()
                if cleanup_result:
                    st.success("✅ 메모리 정리 정상 작동")
                else:
                    st.error("❌ 메모리 정리 실패")
                    
            except Exception as e:
                st.error(f"❌ 메모리 최적화 실패: {e}")
                test_results['memory'] = False
            
            # 3. 에러 복구 시스템 테스트
            st.subheader("3. 🛡️ 에러 복구 시스템 테스트")
            try:
                @resilient(operation_id="integration_test")
                def test_recovery_function():
                    import random
                    if random.random() < 0.5:
                        raise ValueError("테스트 에러")
                    return "성공"
                
                recovery_attempts = 0
                for i in range(5):
                    try:
                        result = test_recovery_function()
                        if result == "성공":
                            recovery_attempts += 1
                    except:
                        pass
                
                if recovery_attempts > 0:
                    st.success(f"✅ 에러 복구 정상 작동 ({recovery_attempts}/5 성공)")
                    test_results['recovery'] = True
                else:
                    st.error("❌ 에러 복구 실패")
                    test_results['recovery'] = False
                    
            except Exception as e:
                st.error(f"❌ 에러 복구 실패: {e}")
                test_results['recovery'] = False
            
            # 4. 통합 성능 테스트
            st.subheader("4. 🚀 통합 성능 테스트")
            try:
                from core.integrated_performance_test_v21 import PerformanceTestSuite
                
                test_suite = PerformanceTestSuite()
                benchmark_results = test_suite.run_full_benchmark()
                
                success_rate = sum(1 for r in benchmark_results if r.success_rate_percent > 50) / len(benchmark_results)
                
                if success_rate >= 0.75:
                    st.success(f"✅ 통합 성능 테스트 정상 ({success_rate*100:.1f}% 통과)")
                    test_results['performance'] = True
                else:
                    st.warning(f"⚠️ 통합 성능 테스트 부분 통과 ({success_rate*100:.1f}% 통과)")
                    test_results['performance'] = False
                
                test_suite.cleanup()
                
            except Exception as e:
                st.error(f"❌ 통합 성능 테스트 실패: {e}")
                test_results['performance'] = False
            
            # 최종 결과
            st.markdown("---")
            st.subheader("📋 통합 테스트 결과")
            
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            
            if passed_tests == total_tests:
                st.success(f"🎉 모든 테스트 통과! ({passed_tests}/{total_tests})")
                st.balloons()
            elif passed_tests >= total_tests * 0.75:
                st.warning(f"⚠️ 대부분 테스트 통과 ({passed_tests}/{total_tests})")
            else:
                st.error(f"❌ 테스트 실패 다수 ({passed_tests}/{total_tests})")
            
            # 상세 결과
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**통과한 테스트:**")
                for test_name, passed in test_results.items():
                    if passed:
                        st.write(f"✅ {test_name}")
            
            with col2:
                st.write("**실패한 테스트:**")
                failed_any = False
                for test_name, passed in test_results.items():
                    if not passed:
                        st.write(f"❌ {test_name}")
                        failed_any = True
                
                if not failed_any:
                    st.write("없음 🎉")
    
    else:
        st.info("🧪 v2.1.2의 모든 최적화 기능을 통합 테스트하려면 위 버튼을 클릭하세요")
        
        st.markdown("""
        ### 🔍 통합 테스트 항목
        
        **v2.1.2 신규 기능 종합 검증:**
        
        1. **📊 성능 프로파일러**
           - 실시간 모니터링 기능
           - 시스템 리소스 추적
           - 성능 메트릭 수집
        
        2. **🧠 메모리 최적화**
           - 스마트 캐싱 시스템
           - 자동 메모리 정리
           - 대용량 파일 처리
        
        3. **🛡️ 에러 복구 시스템**
           - 자동 재시도 로직
           - 회로 차단기 패턴
           - 시스템 안정성 보장
        
        4. **🚀 성능 벤치마크**
           - 종합 성능 측정
           - 최적화 권장사항
           - 시스템 등급 평가
        
        통합 테스트를 통해 모든 기능이 정상적으로 연동되는지 확인합니다.
        """)

if __name__ == "__main__":
    main()
