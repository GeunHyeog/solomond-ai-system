import streamlit as st
import time
from functools import wraps

def cached_ui_component(ttl=300):
    """UI 컴포넌트 캐싱 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            if cache_key in st.session_state.get('ui_cache', {}):
                cache_data = st.session_state.ui_cache[cache_key]
                if time.time() - cache_data['timestamp'] < ttl:
                    return cache_data['result']
            
            # 캐시 미스시 실행
            result = func(*args, **kwargs)
            
            if 'ui_cache' not in st.session_state:
                st.session_state.ui_cache = {}
            
            st.session_state.ui_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            return result
        return wrapper
    return decorator

def with_loading_spinner(message="처리 중..."):
    """로딩 스피너 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def optimized_file_uploader(label, **kwargs):
    """최적화된 파일 업로더"""
    # 파일 크기 제한 체크
    if 'accept_multiple_files' in kwargs and kwargs['accept_multiple_files']:
        kwargs.setdefault('help', '여러 파일 선택 시 총 용량 200MB 이하 권장')
    
    return st.file_uploader(label, **kwargs)

def progress_tracker(total_steps):
    """진행률 추적기"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class ProgressTracker:
        def __init__(self):
            self.current_step = 0
            self.total = total_steps
        
        def update(self, step_name=""):
            self.current_step += 1
            progress = self.current_step / self.total
            progress_bar.progress(progress)
            
            if step_name:
                status_text.text(f"진행률: {progress:.1%} - {step_name}")
            else:
                status_text.text(f"진행률: {progress:.1%}")
        
        def complete(self, message="완료!"):
            progress_bar.progress(1.0)
            status_text.success(message)
    
    return ProgressTracker()

def session_state_cleanup():
    """세션 상태 정리"""
    # 오래된 캐시 데이터 정리
    if 'ui_cache' in st.session_state:
        current_time = time.time()
        cache = st.session_state.ui_cache
        
        expired_keys = [
            key for key, data in cache.items()
            if current_time - data['timestamp'] > 600  # 10분 이상된 캐시
        ]
        
        for key in expired_keys:
            del cache[key]
        
        if expired_keys:
            st.success(f"캐시 정리: {len(expired_keys)}개 항목 삭제")

def error_boundary(func):
    """에러 경계 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            with st.expander("🔍 오류 상세 정보"):
                st.code(f"함수: {func.__name__}")
                st.code(f"오류 유형: {type(e).__name__}")
                st.code(f"오류 메시지: {str(e)}")
            
            if st.button("🔄 다시 시도"):
                st.rerun()
            
            return None
    return wrapper
