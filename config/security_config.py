#!/usr/bin/env python3
"""
🔐 솔로몬드 AI 보안 설정 관리자
- 환경변수 기반 안전한 설정 로딩
- API 키 마스킹 및 로깅 보안
- 설정 검증 및 오류 처리
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SecurityConfig:
    """안전한 환경변수 기반 설정 관리"""
    
    def __init__(self):
        self.config_loaded = False
        self._load_environment()
    
    def _load_environment(self):
        """환경변수 로딩 (dotenv 지원)"""
        try:
            from dotenv import load_dotenv
            
            # .env 파일 찾기
            env_path = Path.cwd() / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                logger.info("✅ .env 파일에서 환경변수 로딩 완료")
            else:
                logger.warning("⚠️ .env 파일 없음 - 시스템 환경변수 사용")
                
        except ImportError:
            logger.warning("⚠️ python-dotenv 없음 - 시스템 환경변수만 사용")
        
        self.config_loaded = True
    
    def get_github_token(self) -> Optional[str]:
        """GitHub 토큰 안전 조회"""
        token = os.getenv('GITHUB_TOKEN')
        if token:
            logger.info(f"✅ GitHub 토큰 로딩됨: {self._mask_token(token)}")
        else:
            logger.warning("⚠️ GITHUB_TOKEN 환경변수 없음")
        return token
    
    def get_supabase_config(self) -> Dict[str, Optional[str]]:
        """Supabase 설정 안전 조회"""
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        
        if url and key:
            logger.info(f"✅ Supabase 설정 로딩됨: {self._mask_url(url)}")
        else:
            logger.warning("⚠️ Supabase 환경변수 없음")
        
        return {'url': url, 'key': key}
    
    def get_notion_api_key(self) -> Optional[str]:
        """Notion API 키 안전 조회"""
        key = os.getenv('NOTION_API_KEY')
        if key:
            logger.info(f"✅ Notion API 키 로딩됨: {self._mask_token(key)}")
        else:
            logger.warning("⚠️ NOTION_API_KEY 환경변수 없음")
        return key
    
    def get_openai_api_key(self) -> Optional[str]:
        """OpenAI API 키 안전 조회"""
        key = os.getenv('OPENAI_API_KEY')
        if key:
            logger.info(f"✅ OpenAI API 키 로딩됨: {self._mask_token(key)}")
        return key
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Anthropic API 키 안전 조회"""
        key = os.getenv('ANTHROPIC_API_KEY')
        if key:
            logger.info(f"✅ Anthropic API 키 로딩됨: {self._mask_token(key)}")
        return key
    
    def get_system_config(self) -> Dict[str, Any]:
        """시스템 설정 조회"""
        return {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', '500')),
            'enable_debug': os.getenv('ENABLE_DEBUG', 'false').lower() == 'true',
            'user_files_path': os.getenv('USER_FILES_PATH', './user_files'),
            'cache_path': os.getenv('CACHE_PATH', './cache'),
            'logs_path': os.getenv('LOGS_PATH', './logs')
        }
    
    def get_ai_config(self) -> Dict[str, str]:
        """AI 모델 설정 조회"""
        return {
            'ollama_base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'default_stt_model': os.getenv('DEFAULT_STT_MODEL', 'whisper-large-v3'),
            'default_ocr_engine': os.getenv('DEFAULT_OCR_ENGINE', 'easyocr'),
            'default_llm_model': os.getenv('DEFAULT_LLM_MODEL', 'qwen2.5:7b')
        }
    
    def validate_critical_config(self) -> Dict[str, bool]:
        """핵심 설정 검증"""
        validation = {
            'github_token': bool(self.get_github_token()),
            'supabase_config': all(self.get_supabase_config().values()),
            'notion_api_key': bool(self.get_notion_api_key()),
            'env_loaded': self.config_loaded
        }
        
        missing = [k for k, v in validation.items() if not v]
        if missing:
            logger.error(f"❌ 누락된 설정: {missing}")
        else:
            logger.info("✅ 모든 핵심 설정 검증 완료")
        
        return validation
    
    def _mask_token(self, token: str) -> str:
        """토큰 마스킹 (처음 4글자 + *** + 끝 4글자)"""
        if len(token) <= 8:
            return "***"
        return f"{token[:4]}***{token[-4:]}"
    
    def _mask_url(self, url: str) -> str:
        """URL 마스킹"""
        if "://" in url:
            protocol, rest = url.split("://", 1)
            if "." in rest:
                domain_parts = rest.split(".")
                masked_domain = f"{domain_parts[0][:3]}***{domain_parts[-1]}"
                return f"{protocol}://{masked_domain}"
        return "***"
    
    def export_masked_config(self) -> Dict[str, Any]:
        """마스킹된 설정 정보 출력 (디버깅용)"""
        github_token = self.get_github_token()
        supabase = self.get_supabase_config()
        notion_key = self.get_notion_api_key()
        
        return {
            'github_token': self._mask_token(github_token) if github_token else None,
            'supabase_url': self._mask_url(supabase['url']) if supabase['url'] else None,
            'supabase_key': self._mask_token(supabase['key']) if supabase['key'] else None,
            'notion_api_key': self._mask_token(notion_key) if notion_key else None,
            'system_config': self.get_system_config(),
            'ai_config': self.get_ai_config(),
            'validation': self.validate_critical_config()
        }

# 글로벌 인스턴스
security_config = SecurityConfig()

# 편의 함수들
def get_github_token() -> Optional[str]:
    return security_config.get_github_token()

def get_supabase_config() -> Dict[str, Optional[str]]:
    return security_config.get_supabase_config()

def get_notion_api_key() -> Optional[str]:
    return security_config.get_notion_api_key()

def get_system_config() -> Dict[str, Any]:
    return security_config.get_system_config()

def validate_config() -> Dict[str, bool]:
    return security_config.validate_critical_config()

if __name__ == "__main__":
    # 설정 테스트
    print("🔐 보안 설정 테스트")
    config = security_config.export_masked_config()
    print(json.dumps(config, indent=2, ensure_ascii=False))