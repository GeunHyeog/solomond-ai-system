"""
솔로몬드 AI 시스템 - UI 컴포넌트
재사용 가능한 UI 컴포넌트 및 헬퍼 함수
"""

def get_upload_component(file_types: list = None) -> str:
    """
    파일 업로드 컴포넌트 HTML 생성
    
    Args:
        file_types: 허용할 파일 확장자 리스트
        
    Returns:
        HTML 문자열
    """
    if file_types is None:
        file_types = [".mp3", ".wav", ".m4a"]
    
    accept_attr = ",".join(file_types)
    
    return f"""
    <div class="upload-area">
        <h3>🎵 음성 파일 업로드</h3>
        <input type="file" name="audio_file" accept="{accept_attr}" required>
        <br>
        <button type="submit">🚀 음성 인식 시작</button>
    </div>
    """

def get_progress_component() -> str:
    """진행률 표시 컴포넌트"""
    return """
    <div id="progress" style="display:none;">
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>
        <p id="progressText">처리 중...</p>
    </div>
    """

def get_result_component() -> str:
    """결과 표시 컴포넌트"""
    return """
    <div id="result" class="result" style="display:none;">
        <h3>📄 처리 결과:</h3>
        <div id="resultContent"></div>
    </div>
    """

def get_status_badge(status: str, message: str) -> str:
    """
    상태 배지 컴포넌트
    
    Args:
        status: success, error, info, warning
        message: 표시할 메시지
    """
    return f"""
    <div class="status {status}">
        <strong>{message}</strong>
    </div>
    """

def get_feature_card(title: str, description: str, icon: str = "🔧") -> str:
    """
    기능 카드 컴포넌트
    
    Args:
        title: 카드 제목
        description: 카드 설명
        icon: 아이콘 (이모지)
    """
    return f"""
    <div class="feature-card">
        <h4>{icon} {title}</h4>
        <p>{description}</p>
    </div>
    """

def get_navigation_component() -> str:
    """네비게이션 컴포넌트"""
    return """
    <nav class="navigation">
        <div class="nav-links">
            <a href="/" class="nav-link">홈</a>
            <a href="/docs" class="nav-link" target="_blank">API 문서</a>
            <a href="/test" class="nav-link" target="_blank">시스템 테스트</a>
        </div>
    </nav>
    """

# 편의 함수들
def wrap_in_container(content: str, title: str = "") -> str:
    """콘텐츠를 컨테이너로 감싸기"""
    title_html = f"<h1>{title}</h1>" if title else ""
    return f"""
    <div class="container">
        {title_html}
        {content}
    </div>
    """

def create_modal(content: str, modal_id: str = "modal") -> str:
    """모달 창 생성"""
    return f"""
    <div id="{modal_id}" class="modal" style="display:none;">
        <div class="modal-content">
            <span class="close">&times;</span>
            {content}
        </div>
    </div>
    """
