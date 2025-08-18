#!/usr/bin/env python3
"""
브라우저 실행 중 발생하는 에러 자동 수정 시스템
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def fix_streamlit_config_issues():
    """Streamlit 설정 관련 이슈 수정"""
    print("=== Streamlit 설정 이슈 수정 ===")
    
    # .streamlit 디렉토리 생성
    streamlit_dir = project_root / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    
    # config.toml 생성/수정
    config_path = streamlit_dir / "config.toml"
    config_content = """[server]
port = 8503
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[logger]
level = "info"
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Streamlit 설정 파일 생성: {config_path}")

def fix_import_errors():
    """Import 에러 수정"""
    print("\n=== Import 에러 수정 ===")
    
    # __init__.py 파일 확인/생성
    core_dir = project_root / "core"
    if core_dir.exists():
        init_file = core_dir / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('# Core module\n')
            print(f"✅ __init__.py 생성: {init_file}")
        else:
            print("✅ __init__.py 존재함")
    
    # 필수 디렉토리 생성
    required_dirs = ["core", "data", "logs", "temp"]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 디렉토리 확인: {dir_name}")

def fix_memory_issues():
    """메모리 관련 이슈 수정"""
    print("\n=== 메모리 이슈 수정 ===")
    
    # 가비지 컬렉션 강제 실행
    import gc
    gc.collect()
    
    # 환경 변수 설정 (CPU 모드 강제)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    
    print("✅ 메모리 최적화 설정 적용")

def fix_encoding_issues():
    """인코딩 이슈 수정"""
    print("\n=== 인코딩 이슈 수정 ===")
    
    # 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    print("✅ UTF-8 인코딩 설정 적용")

def fix_path_issues():
    """경로 관련 이슈 수정"""
    print("\n=== 경로 이슈 수정 ===")
    
    # Python 경로에 프로젝트 루트 추가
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 작업 디렉토리 설정
    os.chdir(project_root)
    
    print(f"✅ 작업 디렉토리 설정: {project_root}")

def create_error_handling_wrapper():
    """에러 핸들링 래퍼 생성"""
    print("\n=== 에러 핸들링 시스템 생성 ===")
    
    wrapper_content = '''import streamlit as st
import sys
import traceback
from pathlib import Path

def safe_import(module_name, fallback=None):
    """안전한 모듈 import"""
    try:
        return __import__(module_name)
    except ImportError as e:
        st.error(f"모듈 {module_name} 로드 실패: {e}")
        return fallback

def error_handler(func):
    """에러 핸들러 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            with st.expander("상세 오류 정보"):
                st.code(traceback.format_exc())
            return None
    return wrapper

# 전역 에러 핸들러 설정
def setup_global_error_handling():
    """전역 에러 핸들링 설정"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print(f"예상치 못한 오류: {exc_type.__name__}: {exc_value}")
    
    sys.excepthook = handle_exception
'''
    
    wrapper_path = project_root / "error_handling.py"
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    print(f"✅ 에러 핸들링 래퍼 생성: {wrapper_path}")

def fix_session_state_issues():
    """세션 상태 관련 이슈 수정"""
    print("\n=== 세션 상태 이슈 수정 ===")
    
    session_fix_content = '''import streamlit as st

def init_session_state():
    """세션 상태 초기화"""
    default_states = {
        'analysis_step': 1,
        'uploaded_files': [],
        'analysis_results': None,
        'user_settings': {},
        'error_count': 0,
        'last_error': None
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_session_state():
    """세션 상태 리셋"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def safe_session_get(key, default=None):
    """안전한 세션 상태 접근"""
    return st.session_state.get(key, default)

def safe_session_set(key, value):
    """안전한 세션 상태 설정"""
    try:
        st.session_state[key] = value
        return True
    except Exception as e:
        st.error(f"세션 상태 설정 실패: {e}")
        return False
'''
    
    session_path = project_root / "session_state_helper.py"
    with open(session_path, 'w', encoding='utf-8') as f:
        f.write(session_fix_content)
    
    print(f"✅ 세션 상태 헬퍼 생성: {session_path}")

def create_browser_compatibility_check():
    """브라우저 호환성 체크 생성"""
    print("\n=== 브라우저 호환성 체크 생성 ===")
    
    browser_check_content = '''import streamlit as st

def check_browser_compatibility():
    """브라우저 호환성 검사"""
    st.markdown("""
    <script>
    // JavaScript 활성화 확인
    if (typeof window !== 'undefined') {
        window.parent.postMessage({type: 'js_enabled', value: true}, '*');
    }
    
    // WebSocket 지원 확인
    if (typeof WebSocket !== 'undefined') {
        window.parent.postMessage({type: 'websocket_supported', value: true}, '*');
    }
    
    // 브라우저 정보
    const browserInfo = {
        userAgent: navigator.userAgent,
        language: navigator.language,
        cookieEnabled: navigator.cookieEnabled,
        onLine: navigator.onLine
    };
    window.parent.postMessage({type: 'browser_info', value: browserInfo}, '*');
    </script>
    """, unsafe_allow_html=True)

def display_browser_requirements():
    """브라우저 요구사항 표시"""
    with st.expander("🌐 브라우저 요구사항"):
        st.markdown("""
        **지원되는 브라우저:**
        - Chrome 90+
        - Firefox 88+
        - Safari 14+
        - Edge 90+
        
        **필요한 설정:**
        - JavaScript 활성화
        - 쿠키 허용
        - WebSocket 지원
        """)
'''
    
    browser_path = project_root / "browser_compatibility.py"
    with open(browser_path, 'w', encoding='utf-8') as f:
        f.write(browser_check_content)
    
    print(f"✅ 브라우저 호환성 체크 생성: {browser_path}")

def apply_all_fixes():
    """모든 수정사항 적용"""
    print("🔧 브라우저 에러 자동 수정 시작")
    print("=" * 50)
    
    try:
        fix_streamlit_config_issues()
        fix_import_errors()
        fix_memory_issues()
        fix_encoding_issues()
        fix_path_issues()
        create_error_handling_wrapper()
        fix_session_state_issues()
        create_browser_compatibility_check()
        
        print("\n" + "=" * 50)
        print("✅ 모든 수정사항 적용 완료!")
        print("\n권장 다음 단계:")
        print("1. 브라우저에서 Ctrl+F5로 강력 새로고침")
        print("2. 브라우저 캐시 및 쿠키 삭제")
        print("3. 개발자 도구(F12)로 JavaScript 에러 확인")
        print("4. 필요시 Streamlit 서버 재시작")
        
    except Exception as e:
        print(f"❌ 수정 중 오류 발생: {e}")

if __name__ == "__main__":
    apply_all_fixes()