"""
🔧 서버 설정 통합 모듈
솔로몬드 AI 시스템 - 중복 코드 제거 (3/3단계)

목적: 15개 파일에서 반복되는 서버 주소 하드코딩을 중앙화
효과: 배포 환경별 설정 관리 개선 및 하드코딩 제거
"""

import os
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class Environment(Enum):
    """배포 환경 구분"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ServerEndpoint:
    """서버 엔드포인트 정보"""
    host: str
    port: int
    protocol: str = "http"
    
    @property
    def url(self) -> str:
        """완전한 URL 반환"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def address(self) -> Tuple[str, int]:
        """(host, port) 튜플 반환"""
        return (self.host, self.port)

class ServerConfig:
    """솔로몬드 AI 시스템 서버 설정 관리자"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.endpoints = self._load_endpoints()
    
    def _detect_environment(self) -> Environment:
        """현재 실행 환경 감지"""
        env_name = os.getenv('SOLOMOND_ENV', 'development').lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def _load_endpoints(self) -> Dict[str, ServerEndpoint]:
        """환경별 서버 엔드포인트 설정 로드"""
        
        # 기본 설정 (개발 환경)
        base_config = {
            'main_server': ServerEndpoint('localhost', 8080),
            'streamlit_ui': ServerEndpoint('localhost', 8503),
            'websocket_stt': ServerEndpoint('localhost', 8765),
            'api_server': ServerEndpoint('localhost', 8000),
            'monitoring': ServerEndpoint('localhost', 8888),
        }
        
        # 환경별 설정 오버라이드
        if self.environment == Environment.DEVELOPMENT:
            # 개발 환경: 모든 포트를 localhost에서 사용
            pass  # 기본 설정 그대로 사용
            
        elif self.environment == Environment.TESTING:
            # 테스트 환경: 다른 포트 범위 사용
            base_config.update({
                'main_server': ServerEndpoint('localhost', 18080),
                'streamlit_ui': ServerEndpoint('localhost', 18503),
                'websocket_stt': ServerEndpoint('localhost', 18765),
                'api_server': ServerEndpoint('localhost', 18000),
                'monitoring': ServerEndpoint('localhost', 18888),
            })
            
        elif self.environment == Environment.STAGING:
            # 스테이징 환경: 실제 서버 주소 사용
            base_config.update({
                'main_server': ServerEndpoint('0.0.0.0', 8080),
                'streamlit_ui': ServerEndpoint('0.0.0.0', 8503),
                'websocket_stt': ServerEndpoint('0.0.0.0', 8765),
                'api_server': ServerEndpoint('0.0.0.0', 8000),
                'monitoring': ServerEndpoint('0.0.0.0', 8888),
            })
            
        elif self.environment == Environment.PRODUCTION:
            # 프로덕션 환경: 보안 설정 적용
            base_config.update({
                'main_server': ServerEndpoint('0.0.0.0', 443, 'https'),
                'streamlit_ui': ServerEndpoint('0.0.0.0', 8503, 'https'),
                'websocket_stt': ServerEndpoint('0.0.0.0', 8765, 'wss'),
                'api_server': ServerEndpoint('0.0.0.0', 8000, 'https'),
                'monitoring': ServerEndpoint('127.0.0.1', 8888),  # 내부 접근만
            })
        
        # 환경 변수로 개별 설정 오버라이드
        self._apply_env_overrides(base_config)
        
        return base_config
    
    def _apply_env_overrides(self, config: Dict[str, ServerEndpoint]):
        """환경 변수를 통한 개별 설정 오버라이드"""
        
        env_mappings = {
            'SOLOMOND_MAIN_HOST': ('main_server', 'host'),
            'SOLOMOND_MAIN_PORT': ('main_server', 'port'),
            'SOLOMOND_STREAMLIT_HOST': ('streamlit_ui', 'host'),
            'SOLOMOND_STREAMLIT_PORT': ('streamlit_ui', 'port'),
            'SOLOMOND_STT_HOST': ('websocket_stt', 'host'),
            'SOLOMOND_STT_PORT': ('websocket_stt', 'port'),
            'SOLOMOND_API_HOST': ('api_server', 'host'),
            'SOLOMOND_API_PORT': ('api_server', 'port'),
        }
        
        for env_key, (service, attr) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value and service in config:
                if attr == 'port':
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        continue
                
                setattr(config[service], attr, env_value)
    
    def get_endpoint(self, service: str) -> Optional[ServerEndpoint]:
        """서비스별 엔드포인트 반환
        
        Args:
            service: 서비스 이름 ('main_server', 'streamlit_ui', 등)
            
        Returns:
            ServerEndpoint: 서버 엔드포인트 정보
        """
        return self.endpoints.get(service)
    
    def get_host_port(self, service: str) -> Optional[Tuple[str, int]]:
        """서비스별 (host, port) 튜플 반환
        
        기존 코드 호환용:
        ```python
        host, port = config.get_host_port('streamlit_ui')
        ```
        """
        endpoint = self.get_endpoint(service)
        return endpoint.address if endpoint else None
    
    def get_url(self, service: str) -> Optional[str]:
        """서비스별 완전한 URL 반환"""
        endpoint = self.get_endpoint(service)
        return endpoint.url if endpoint else None
    
    def get_websocket_url(self, service: str = 'websocket_stt') -> Optional[str]:
        """WebSocket URL 반환"""
        endpoint = self.get_endpoint(service)
        if endpoint:
            protocol = 'wss' if endpoint.protocol == 'https' else 'ws'
            return f"{protocol}://{endpoint.host}:{endpoint.port}"
        return None
    
    def list_all_endpoints(self) -> Dict[str, str]:
        """모든 엔드포인트 목록 반환 (디버깅용)"""
        return {
            service: endpoint.url 
            for service, endpoint in self.endpoints.items()
        }
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment == Environment.DEVELOPMENT

# 전역 설정 인스턴스
_server_config = None

def get_server_config() -> ServerConfig:
    """전역 ServerConfig 인스턴스 반환"""
    global _server_config
    if _server_config is None:
        _server_config = ServerConfig()
    return _server_config

# 편의 함수들 (기존 코드 호환성)

def get_streamlit_config() -> Tuple[str, int]:
    """Streamlit 서버 설정 반환
    
    기존 코드에서 이렇게 사용:
    ```python
    from config.server_config import get_streamlit_config
    host, port = get_streamlit_config()
    ```
    """
    config = get_server_config()
    endpoint = config.get_endpoint('streamlit_ui')
    return (endpoint.host, endpoint.port) if endpoint else ('localhost', 8503)

def get_main_server_config() -> Tuple[str, int]:
    """메인 서버 설정 반환"""
    config = get_server_config()
    endpoint = config.get_endpoint('main_server')
    return (endpoint.host, endpoint.port) if endpoint else ('localhost', 8080)

def get_websocket_config() -> Tuple[str, int]:
    """WebSocket 서버 설정 반환"""
    config = get_server_config()
    endpoint = config.get_endpoint('websocket_stt')
    return (endpoint.host, endpoint.port) if endpoint else ('localhost', 8765)

def get_api_server_config() -> Tuple[str, int]:
    """API 서버 설정 반환"""
    config = get_server_config()
    endpoint = config.get_endpoint('api_server')
    return (endpoint.host, endpoint.port) if endpoint else ('localhost', 8000)

# URL 생성 편의 함수들

def get_streamlit_url() -> str:
    """Streamlit UI URL 반환"""
    config = get_server_config()
    return config.get_url('streamlit_ui') or 'http://localhost:8503'

def get_websocket_url() -> str:
    """WebSocket URL 반환"""
    config = get_server_config()
    return config.get_websocket_url() or 'ws://localhost:8765'

def get_api_url() -> str:
    """API 서버 URL 반환"""
    config = get_server_config()
    return config.get_url('api_server') or 'http://localhost:8000'

# 레거시 호환성 함수들

def get_default_host() -> str:
    """기본 호스트 주소 반환 (레거시 호환용)"""
    config = get_server_config()
    if config.is_development():
        return 'localhost'
    else:
        return '0.0.0.0'

def get_default_port(service: str = 'main') -> int:
    """기본 포트 반환 (레거시 호환용)"""
    port_map = {
        'main': 8080,
        'streamlit': 8503,
        'websocket': 8765,
        'api': 8000
    }
    return port_map.get(service, 8080)

# 환경 변수 기반 설정 함수들

def setup_server_environment():
    """서버 환경 설정 (환경 변수 기반)
    
    다음 환경 변수들을 지원:
    - SOLOMOND_ENV: development, testing, staging, production
    - SOLOMOND_MAIN_HOST, SOLOMOND_MAIN_PORT: 메인 서버
    - SOLOMOND_STREAMLIT_HOST, SOLOMOND_STREAMLIT_PORT: Streamlit UI
    - SOLOMOND_STT_HOST, SOLOMOND_STT_PORT: WebSocket STT 서버
    - SOLOMOND_API_HOST, SOLOMOND_API_PORT: API 서버
    """
    config = get_server_config()
    return config

# 서버 헬스 체크 함수들

def check_server_availability(service: str) -> bool:
    """서버 가용성 확인"""
    import socket
    
    config = get_server_config()
    endpoint = config.get_endpoint(service)
    
    if not endpoint:
        return False
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((endpoint.host, endpoint.port))
        sock.close()
        return result == 0
    except:
        return False

def get_server_status() -> Dict[str, bool]:
    """모든 서버 상태 확인"""
    config = get_server_config()
    status = {}
    
    for service in config.endpoints.keys():
        status[service] = check_server_availability(service)
    
    return status

# 모듈 초기화
_config = get_server_config()  # 모듈 import 시 자동 초기화