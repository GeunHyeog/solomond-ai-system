"""
솔로몬드 AI 시스템 설정 모듈
"""

from .compute_config import (
    ComputeConfig, 
    ComputeMode, 
    get_compute_config, 
    setup_compute_environment, 
    force_cpu_mode, 
    get_device_string
)

from .server_config import (
    ServerConfig,
    Environment,
    ServerEndpoint,
    get_server_config,
    get_streamlit_config,
    get_main_server_config,
    get_websocket_config,
    get_api_server_config,
    get_streamlit_url,
    get_websocket_url,
    get_api_url
)

__all__ = [
    # Compute Config
    'ComputeConfig', 'ComputeMode', 'get_compute_config', 
    'setup_compute_environment', 'force_cpu_mode', 'get_device_string',
    
    # Server Config
    'ServerConfig', 'Environment', 'ServerEndpoint', 'get_server_config',
    'get_streamlit_config', 'get_main_server_config', 'get_websocket_config',
    'get_api_server_config', 'get_streamlit_url', 'get_websocket_url', 'get_api_url'
]