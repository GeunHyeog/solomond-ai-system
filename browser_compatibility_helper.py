import streamlit as st
import streamlit.components.v1 as components

def browser_compatibility_check():
    """브라우저 호환성 체크"""
    
    # JavaScript를 통한 브라우저 정보 수집
    browser_check_js = """
    <script>
    function checkBrowserCompatibility() {
        const compatibility = {
            userAgent: navigator.userAgent,
            language: navigator.language,
            cookieEnabled: navigator.cookieEnabled,
            onLine: navigator.onLine,
            platform: navigator.platform,
            webSocket: typeof WebSocket !== 'undefined',
            localStorage: typeof Storage !== 'undefined',
            fetch: typeof fetch !== 'undefined'
        };
        
        // Streamlit으로 데이터 전송
        window.parent.postMessage({
            type: 'browser_compatibility',
            data: compatibility
        }, '*');
        
        return compatibility;
    }
    
    // 페이지 로드 후 실행
    document.addEventListener('DOMContentLoaded', checkBrowserCompatibility);
    
    // 즉시 실행도 추가
    checkBrowserCompatibility();
    </script>
    """
    
    components.html(browser_check_js, height=0)

def show_browser_requirements():
    """브라우저 요구사항 표시"""
    with st.expander("🌐 브라우저 호환성 정보"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ 지원되는 브라우저:**")
            st.markdown("""
            - Chrome 90+
            - Firefox 88+
            - Safari 14+
            - Edge 90+
            """)
        
        with col2:
            st.markdown("**⚙️ 필요한 설정:**")
            st.markdown("""
            - JavaScript 활성화
            - 쿠키 허용
            - WebSocket 지원
            - localStorage 사용 가능
            """)

def add_performance_monitoring():
    """성능 모니터링 추가"""
    if 'performance_start' not in st.session_state:
        st.session_state.performance_start = time.time()
    
    current_time = time.time()
    session_duration = current_time - st.session_state.performance_start
    
    if session_duration > 300:  # 5분 이상
        st.sidebar.warning(f"⏱️ 세션 시간: {session_duration/60:.1f}분\n새로고침을 권장합니다.")

def optimize_websocket_connection():
    """WebSocket 연결 최적화"""
    websocket_optimization_js = """
    <script>
    // WebSocket 연결 최적화
    if (typeof window.streamlitWebSocket !== 'undefined') {
        const originalWebSocket = window.streamlitWebSocket;
        
        // 연결 재시도 로직
        const reconnectInterval = 5000;
        let reconnectTimer;
        
        originalWebSocket.addEventListener('close', function(event) {
            console.log('WebSocket connection closed, attempting to reconnect...');
            
            reconnectTimer = setTimeout(() => {
                if (window.location.reload) {
                    window.location.reload();
                }
            }, reconnectInterval);
        });
        
        originalWebSocket.addEventListener('open', function(event) {
            console.log('WebSocket connection established');
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
            }
        });
    }
    </script>
    """
    
    components.html(websocket_optimization_js, height=0)
