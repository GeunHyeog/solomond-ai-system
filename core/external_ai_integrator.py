#!/usr/bin/env python3
"""
외부 AI 모델 통합 시스템 v2.6
OpenAI GPT, Anthropic Claude, Google Gemini API 통합
"""

import os
import time
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading

class AIProvider(Enum):
    """AI 공급자"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"

class TaskType(Enum):
    """작업 타입"""
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    QA = "question_answering"
    ENHANCEMENT = "enhancement"
    CLASSIFICATION = "classification"
    GENERATION = "generation"

@dataclass
class AIModelConfig:
    """AI 모델 설정"""
    provider: AIProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout_seconds: int = 30
    rate_limit_rpm: int = 60  # requests per minute
    cost_per_1k_tokens: float = 0.0
    supports_streaming: bool = False
    context_window: int = 4000
    is_enabled: bool = True

@dataclass
class AIRequest:
    """AI 요청"""
    task_type: TaskType
    prompt: str
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")

@dataclass
class AIResponse:
    """AI 응답"""
    request_id: str
    provider: AIProvider
    model_name: str
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    cost_estimate: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExternalAIIntegrator:
    """외부 AI 모델 통합 관리자"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".solomond_ai" / "ai_configs.json"
        self.lock = threading.RLock()
        self.logger = self._setup_logging()
        
        # AI 모델 설정
        self.models: Dict[str, AIModelConfig] = {}
        self.rate_limiters: Dict[str, Dict] = {}
        
        # 사용 통계
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0,
            'requests_by_provider': {},
            'session_start': datetime.now()
        }
        
        # 설정 로드
        self._load_configurations()
        self._initialize_default_models()
        
        # 사용자 설정 연동
        try:
            from .user_settings_manager import get_global_settings_manager
            self.settings_manager = get_global_settings_manager()
            self._load_api_keys_from_settings()
        except ImportError:
            self.settings_manager = None
        
        self.logger.info("🤖 외부 AI 모델 통합 시스템 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_configurations(self) -> None:
        """AI 모델 설정 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                for model_id, model_data in config_data.get('models', {}).items():
                    model_data['provider'] = AIProvider(model_data['provider'])
                    self.models[model_id] = AIModelConfig(**model_data)
                
                self.logger.info(f"📥 AI 모델 설정 로드: {len(self.models)}개 모델")
            
        except Exception as e:
            self.logger.warning(f"⚠️ AI 모델 설정 로드 실패: {e}")
    
    def _save_configurations(self) -> None:
        """AI 모델 설정 저장"""
        try:
            # 디렉토리 생성
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'models': {},
                'updated_at': datetime.now().isoformat(),
                'version': '2.6'
            }
            
            for model_id, model_config in self.models.items():
                model_data = {
                    'provider': model_config.provider.value,
                    'model_name': model_config.model_name,
                    'api_key': model_config.api_key,
                    'base_url': model_config.base_url,
                    'max_tokens': model_config.max_tokens,
                    'temperature': model_config.temperature,
                    'timeout_seconds': model_config.timeout_seconds,
                    'rate_limit_rpm': model_config.rate_limit_rpm,
                    'cost_per_1k_tokens': model_config.cost_per_1k_tokens,
                    'supports_streaming': model_config.supports_streaming,
                    'context_window': model_config.context_window,
                    'is_enabled': model_config.is_enabled
                }
                config_data['models'][model_id] = model_data
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("💾 AI 모델 설정 저장 완료")
            
        except Exception as e:
            self.logger.error(f"❌ AI 모델 설정 저장 실패: {e}")
    
    def _initialize_default_models(self) -> None:
        """기본 AI 모델 초기화"""
        default_models = [
            # OpenAI GPT 모델들
            AIModelConfig(
                provider=AIProvider.OPENAI,
                model_name="gpt-4o-mini",
                max_tokens=4000,
                temperature=0.7,
                rate_limit_rpm=500,
                cost_per_1k_tokens=0.00015,
                supports_streaming=True,
                context_window=128000,
                is_enabled=True
            ),
            AIModelConfig(
                provider=AIProvider.OPENAI,
                model_name="gpt-4o",
                max_tokens=4000,
                temperature=0.7,
                rate_limit_rpm=500,
                cost_per_1k_tokens=0.005,
                supports_streaming=True,
                context_window=128000,
                is_enabled=True
            ),
            AIModelConfig(
                provider=AIProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                max_tokens=4000,
                temperature=0.7,
                rate_limit_rpm=3500,
                cost_per_1k_tokens=0.0005,
                supports_streaming=True,
                context_window=16385,
                is_enabled=True
            ),
            
            # Anthropic Claude 모델들
            AIModelConfig(
                provider=AIProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                base_url="https://api.anthropic.com",
                max_tokens=4000,
                temperature=0.7,
                rate_limit_rpm=50,
                cost_per_1k_tokens=0.003,
                supports_streaming=True,
                context_window=200000,
                is_enabled=True
            ),
            AIModelConfig(
                provider=AIProvider.ANTHROPIC,
                model_name="claude-3-5-haiku-20241022",
                base_url="https://api.anthropic.com",
                max_tokens=4000,
                temperature=0.7,
                rate_limit_rpm=50,
                cost_per_1k_tokens=0.00025,
                supports_streaming=True,
                context_window=200000,
                is_enabled=True
            ),
            
            # Google Gemini 모델들
            AIModelConfig(
                provider=AIProvider.GOOGLE,
                model_name="gemini-1.5-flash",
                base_url="https://generativelanguage.googleapis.com",
                max_tokens=8192,
                temperature=0.7,
                rate_limit_rpm=15,
                cost_per_1k_tokens=0.00015,
                supports_streaming=True,
                context_window=1000000,
                is_enabled=True
            ),
            AIModelConfig(
                provider=AIProvider.GOOGLE,
                model_name="gemini-1.5-pro",
                base_url="https://generativelanguage.googleapis.com",
                max_tokens=8192,
                temperature=0.7,
                rate_limit_rpm=2,
                cost_per_1k_tokens=0.0035,
                supports_streaming=True,
                context_window=2000000,
                is_enabled=True
            )
        ]
        
        # 기존에 없는 모델만 추가
        for model_config in default_models:
            model_id = f"{model_config.provider.value}_{model_config.model_name.replace('-', '_').replace('.', '_')}"
            if model_id not in self.models:
                self.models[model_id] = model_config
                
        self.logger.info(f"🔧 기본 AI 모델 초기화: {len(self.models)}개 모델 등록")
    
    def _load_api_keys_from_settings(self) -> None:
        """사용자 설정에서 API 키 로드"""
        if not self.settings_manager:
            return
        
        try:
            # OpenAI API 키
            openai_key = self.settings_manager.get_setting("api.openai_key")
            if openai_key:
                for model_id, model_config in self.models.items():
                    if model_config.provider == AIProvider.OPENAI:
                        model_config.api_key = openai_key
            
            # Anthropic API 키
            anthropic_key = self.settings_manager.get_setting("api.anthropic_key")
            if anthropic_key:
                for model_id, model_config in self.models.items():
                    if model_config.provider == AIProvider.ANTHROPIC:
                        model_config.api_key = anthropic_key
            
            # Google API 키
            google_key = self.settings_manager.get_setting("api.google_key")
            if google_key:
                for model_id, model_config in self.models.items():
                    if model_config.provider == AIProvider.GOOGLE:
                        model_config.api_key = google_key
            
            self.logger.debug("🔑 사용자 설정에서 API 키 로드 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ API 키 로드 실패: {e}")
    
    def add_model(self, model_id: str, model_config: AIModelConfig) -> bool:
        """AI 모델 추가"""
        try:
            with self.lock:
                self.models[model_id] = model_config
                self._save_configurations()
                
                self.logger.info(f"➕ AI 모델 추가: {model_id} ({model_config.provider.value})")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ AI 모델 추가 실패 {model_id}: {e}")
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """AI 모델 제거"""
        try:
            with self.lock:
                if model_id in self.models:
                    del self.models[model_id]
                    self._save_configurations()
                    
                    self.logger.info(f"➖ AI 모델 제거: {model_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"❌ AI 모델 제거 실패 {model_id}: {e}")
            return False
    
    def set_api_key(self, provider: AIProvider, api_key: str) -> bool:
        """API 키 설정"""
        try:
            with self.lock:
                # 해당 공급자의 모든 모델에 API 키 적용
                updated_count = 0
                for model_config in self.models.values():
                    if model_config.provider == provider:
                        model_config.api_key = api_key
                        updated_count += 1
                
                # 사용자 설정에도 저장
                if self.settings_manager:
                    key_name = f"api.{provider.value}_key"
                    from .user_settings_manager import SettingType, SettingScope
                    self.settings_manager.set_setting(
                        key_name, 
                        api_key,
                        SettingType.SYSTEM,
                        SettingScope.GLOBAL,
                        f"{provider.value.upper()} API 키"
                    )
                
                self._save_configurations()
                
                self.logger.info(f"🔑 {provider.value.upper()} API 키 설정: {updated_count}개 모델 업데이트")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ API 키 설정 실패 {provider.value}: {e}")
            return False
    
    def get_available_models(self, task_type: Optional[TaskType] = None) -> List[str]:
        """사용 가능한 모델 목록"""
        with self.lock:
            available = []
            
            for model_id, model_config in self.models.items():
                if not model_config.is_enabled or not model_config.api_key:
                    continue
                
                # 작업 타입별 필터링 (필요시 확장)
                if task_type and task_type == TaskType.SUMMARIZATION:
                    # 요약 작업에 적합한 모델만
                    if model_config.max_tokens < 1000:
                        continue
                
                available.append(model_id)
            
            return sorted(available)
    
    def _check_rate_limit(self, model_id: str) -> bool:
        """요청 속도 제한 확인"""
        if model_id not in self.rate_limiters:
            self.rate_limiters[model_id] = {
                'requests': [],
                'last_reset': time.time()
            }
        
        limiter = self.rate_limiters[model_id]
        model_config = self.models[model_id]
        current_time = time.time()
        
        # 1분 이전 요청 제거
        limiter['requests'] = [
            req_time for req_time in limiter['requests']
            if current_time - req_time < 60
        ]
        
        # 요청 수 확인
        if len(limiter['requests']) >= model_config.rate_limit_rpm:
            return False
        
        return True
    
    def _record_request(self, model_id: str) -> None:
        """요청 기록"""
        if model_id not in self.rate_limiters:
            self.rate_limiters[model_id] = {
                'requests': [],
                'last_reset': time.time()
            }
        
        self.rate_limiters[model_id]['requests'].append(time.time())
    
    async def generate_response(self, request: AIRequest, model_id: Optional[str] = None) -> AIResponse:
        """AI 응답 생성"""
        start_time = time.time()
        
        try:
            # 모델 선택
            if model_id is None:
                available_models = self.get_available_models(request.task_type)
                if not available_models:
                    return AIResponse(
                        request_id=request.request_id,
                        provider=AIProvider.OPENAI,  # 기본값
                        model_name="unavailable",
                        content="",
                        success=False,
                        error_message="사용 가능한 AI 모델이 없습니다. API 키를 설정해주세요."
                    )
                
                # 기본적으로 첫 번째 사용 가능한 모델 선택
                model_id = available_models[0]
            
            if model_id not in self.models:
                return AIResponse(
                    request_id=request.request_id,
                    provider=AIProvider.OPENAI,
                    model_name=model_id,
                    content="",
                    success=False,
                    error_message=f"모델을 찾을 수 없습니다: {model_id}"
                )
            
            model_config = self.models[model_id]
            
            # API 키 확인
            if not model_config.api_key:
                return AIResponse(
                    request_id=request.request_id,
                    provider=model_config.provider,
                    model_name=model_config.model_name,
                    content="",
                    success=False,
                    error_message=f"{model_config.provider.value.upper()} API 키가 설정되지 않았습니다."
                )
            
            # 요청 속도 제한 확인
            if not self._check_rate_limit(model_id):
                return AIResponse(
                    request_id=request.request_id,
                    provider=model_config.provider,
                    model_name=model_config.model_name,
                    content="",
                    success=False,
                    error_message="요청 속도 제한에 도달했습니다. 잠시 후 다시 시도해주세요."
                )
            
            # 요청 기록
            self._record_request(model_id)
            
            # 공급자별 API 호출
            if model_config.provider == AIProvider.OPENAI:
                response = await self._call_openai_api(request, model_config)
            elif model_config.provider == AIProvider.ANTHROPIC:
                response = await self._call_anthropic_api(request, model_config)
            elif model_config.provider == AIProvider.GOOGLE:
                response = await self._call_google_api(request, model_config)
            else:
                return AIResponse(
                    request_id=request.request_id,
                    provider=model_config.provider,
                    model_name=model_config.model_name,
                    content="",
                    success=False,
                    error_message=f"지원하지 않는 AI 공급자: {model_config.provider.value}"
                )
            
            # 처리 시간 및 비용 계산
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            
            if response.success and response.usage:
                total_tokens = response.usage.get('total_tokens', 0)
                response.cost_estimate = (total_tokens / 1000) * model_config.cost_per_1k_tokens
            
            # 통계 업데이트
            self._update_usage_stats(response, model_config)
            
            return response
            
        except Exception as e:
            error_response = AIResponse(
                request_id=request.request_id,
                provider=AIProvider.OPENAI,
                model_name="error",
                content="",
                success=False,
                error_message=f"AI 요청 처리 중 오류: {str(e)}"
            )
            
            self.logger.error(f"❌ AI 요청 실패 {request.request_id}: {e}")
            return error_response
    
    async def _call_openai_api(self, request: AIRequest, model_config: AIModelConfig) -> AIResponse:
        """OpenAI API 호출"""
        try:
            # OpenAI 라이브러리 사용
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=model_config.api_key)
            except ImportError:
                # HTTP 요청으로 fallback
                return await self._call_openai_http(request, model_config)
            
            # 메시지 구성
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # API 호출
            response = await client.chat.completions.create(
                model=model_config.model_name,
                messages=messages,
                max_tokens=request.max_tokens or model_config.max_tokens,
                temperature=request.temperature or model_config.temperature,
                timeout=model_config.timeout_seconds
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return AIResponse(
                request_id=request.request_id,
                provider=model_config.provider,
                model_name=model_config.model_name,
                content=content,
                usage=usage,
                success=True
            )
            
        except Exception as e:
            return AIResponse(
                request_id=request.request_id,
                provider=model_config.provider,
                model_name=model_config.model_name,
                content="",
                success=False,
                error_message=f"OpenAI API 오류: {str(e)}"
            )
    
    async def _call_openai_http(self, request: AIRequest, model_config: AIModelConfig) -> AIResponse:
        """OpenAI HTTP API 호출"""
        try:
            headers = {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            payload = {
                "model": model_config.model_name,
                "messages": messages,
                "max_tokens": request.max_tokens or model_config.max_tokens,
                "temperature": request.temperature or model_config.temperature
            }
            
            timeout = aiohttp.ClientTimeout(total=model_config.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        usage = data.get('usage', {})
                        
                        return AIResponse(
                            request_id=request.request_id,
                            provider=model_config.provider,
                            model_name=model_config.model_name,
                            content=content,
                            usage=usage,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        return AIResponse(
                            request_id=request.request_id,
                            provider=model_config.provider,
                            model_name=model_config.model_name,
                            content="",
                            success=False,
                            error_message=f"OpenAI HTTP 오류 {response.status}: {error_text}"
                        )
            
        except Exception as e:
            return AIResponse(
                request_id=request.request_id,
                provider=model_config.provider,
                model_name=model_config.model_name,
                content="",
                success=False,
                error_message=f"OpenAI HTTP 요청 오류: {str(e)}"
            )
    
    async def _call_anthropic_api(self, request: AIRequest, model_config: AIModelConfig) -> AIResponse:
        """Anthropic Claude API 호출"""
        try:
            headers = {
                "x-api-key": model_config.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": model_config.model_name,
                "max_tokens": request.max_tokens or model_config.max_tokens,
                "temperature": request.temperature or model_config.temperature,
                "messages": [{"role": "user", "content": request.prompt}]
            }
            
            if request.system_prompt:
                payload["system"] = request.system_prompt
            
            timeout = aiohttp.ClientTimeout(total=model_config.timeout_seconds)
            base_url = model_config.base_url or "https://api.anthropic.com"
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{base_url}/v1/messages",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        usage = data.get('usage', {})
                        
                        return AIResponse(
                            request_id=request.request_id,
                            provider=model_config.provider,
                            model_name=model_config.model_name,
                            content=content,
                            usage=usage,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        return AIResponse(
                            request_id=request.request_id,
                            provider=model_config.provider,
                            model_name=model_config.model_name,
                            content="",
                            success=False,
                            error_message=f"Anthropic API 오류 {response.status}: {error_text}"
                        )
            
        except Exception as e:
            return AIResponse(
                request_id=request.request_id,
                provider=model_config.provider,
                model_name=model_config.model_name,
                content="",
                success=False,
                error_message=f"Anthropic API 요청 오류: {str(e)}"
            )
    
    async def _call_google_api(self, request: AIRequest, model_config: AIModelConfig) -> AIResponse:
        """Google Gemini API 호출"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # 프롬프트 구성
            prompt_text = request.prompt
            if request.system_prompt:
                prompt_text = f"{request.system_prompt}\n\n{prompt_text}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt_text}]
                }],
                "generationConfig": {
                    "maxOutputTokens": request.max_tokens or model_config.max_tokens,
                    "temperature": request.temperature or model_config.temperature
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=model_config.timeout_seconds)
            base_url = model_config.base_url or "https://generativelanguage.googleapis.com"
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{base_url}/v1beta/models/{model_config.model_name}:generateContent?key={model_config.api_key}",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        usage = data.get('usageMetadata', {})
                        
                        return AIResponse(
                            request_id=request.request_id,
                            provider=model_config.provider,
                            model_name=model_config.model_name,
                            content=content,
                            usage=usage,
                            success=True
                        )
                    else:
                        error_text = await response.text()
                        return AIResponse(
                            request_id=request.request_id,
                            provider=model_config.provider,
                            model_name=model_config.model_name,
                            content="",
                            success=False,
                            error_message=f"Google API 오류 {response.status}: {error_text}"
                        )
            
        except Exception as e:
            return AIResponse(
                request_id=request.request_id,
                provider=model_config.provider,
                model_name=model_config.model_name,
                content="",
                success=False,
                error_message=f"Google API 요청 오류: {str(e)}"
            )
    
    def _update_usage_stats(self, response: AIResponse, model_config: AIModelConfig) -> None:
        """사용 통계 업데이트"""
        with self.lock:
            self.usage_stats['total_requests'] += 1
            
            if response.success:
                self.usage_stats['successful_requests'] += 1
                
                # 토큰 사용량
                total_tokens = response.usage.get('total_tokens', 0)
                self.usage_stats['total_tokens_used'] += total_tokens
                
                # 비용
                self.usage_stats['total_cost'] += response.cost_estimate
                
                # 공급자별 통계
                provider_name = response.provider.value
                if provider_name not in self.usage_stats['requests_by_provider']:
                    self.usage_stats['requests_by_provider'][provider_name] = 0
                self.usage_stats['requests_by_provider'][provider_name] += 1
                
            else:
                self.usage_stats['failed_requests'] += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """사용 통계 반환"""
        with self.lock:
            stats = self.usage_stats.copy()
            stats['session_duration_minutes'] = (datetime.now() - stats['session_start']).total_seconds() / 60
            stats['success_rate'] = (
                stats['successful_requests'] / stats['total_requests'] * 100
                if stats['total_requests'] > 0 else 0
            )
            return stats
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """모델 정보 반환"""
        if model_id not in self.models:
            return None
        
        model_config = self.models[model_id]
        
        return {
            'model_id': model_id,
            'provider': model_config.provider.value,
            'model_name': model_config.model_name,
            'max_tokens': model_config.max_tokens,
            'context_window': model_config.context_window,
            'cost_per_1k_tokens': model_config.cost_per_1k_tokens,
            'rate_limit_rpm': model_config.rate_limit_rpm,
            'supports_streaming': model_config.supports_streaming,
            'is_enabled': model_config.is_enabled,
            'has_api_key': bool(model_config.api_key)
        }
    
    def test_model_connection(self, model_id: str) -> Dict[str, Any]:
        """모델 연결 테스트"""
        if model_id not in self.models:
            return {
                'success': False,
                'error': f'모델을 찾을 수 없습니다: {model_id}'
            }
        
        try:
            # 간단한 테스트 요청
            test_request = AIRequest(
                task_type=TaskType.GENERATION,
                prompt="안녕하세요! 간단히 인사말로 답변해주세요.",
                max_tokens=50,
                temperature=0.7
            )
            
            # 동기 방식으로 테스트 (asyncio 이벤트 루프 확인)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.generate_response(test_request, model_id))
                        response = future.result(timeout=30)
                else:
                    response = asyncio.run(self.generate_response(test_request, model_id))
            except RuntimeError:
                # 이벤트 루프가 없으면 새로 생성
                response = asyncio.run(self.generate_response(test_request, model_id))
            
            if response.success:
                return {
                    'success': True,
                    'model_name': response.model_name,
                    'response_preview': response.content[:100] + "..." if len(response.content) > 100 else response.content,
                    'processing_time_ms': response.processing_time_ms,
                    'tokens_used': response.usage.get('total_tokens', 0),
                    'cost_estimate': response.cost_estimate
                }
            else:
                return {
                    'success': False,
                    'error': response.error_message
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'연결 테스트 중 오류: {str(e)}'
            }
    
    def cleanup(self) -> None:
        """리소스 정리"""
        with self.lock:
            # 설정 저장
            self._save_configurations()
            
            # 통계 리셋
            self.usage_stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_tokens_used': 0,
                'total_cost': 0.0,
                'requests_by_provider': {},
                'session_start': datetime.now()
            }
            
            self.logger.info("🧹 외부 AI 모델 통합 시스템 정리 완료")

# 전역 외부 AI 통합 관리자
_global_ai_integrator = None
_global_integrator_lock = threading.Lock()

def get_global_ai_integrator() -> ExternalAIIntegrator:
    """전역 외부 AI 통합 관리자 가져오기"""
    global _global_ai_integrator
    
    with _global_integrator_lock:
        if _global_ai_integrator is None:
            _global_ai_integrator = ExternalAIIntegrator()
        return _global_ai_integrator

# 편의 함수들
async def generate_ai_response(prompt: str, task_type: TaskType = TaskType.GENERATION,
                              system_prompt: Optional[str] = None, model_id: Optional[str] = None) -> AIResponse:
    """AI 응답 생성 (편의 함수)"""
    integrator = get_global_ai_integrator()
    
    request = AIRequest(
        task_type=task_type,
        prompt=prompt,
        system_prompt=system_prompt
    )
    
    return await integrator.generate_response(request, model_id)

def set_ai_api_key(provider: str, api_key: str) -> bool:
    """AI API 키 설정 (편의 함수)"""
    integrator = get_global_ai_integrator()
    
    try:
        provider_enum = AIProvider(provider.lower())
        return integrator.set_api_key(provider_enum, api_key)
    except ValueError:
        return False

def get_ai_usage_stats() -> Dict[str, Any]:
    """AI 사용 통계 조회 (편의 함수)"""
    integrator = get_global_ai_integrator()
    return integrator.get_usage_stats()

# 사용 예시
if __name__ == "__main__":
    async def test_ai_integrator():
        integrator = ExternalAIIntegrator()
        
        # API 키 설정 (실제 사용 시 환경변수나 설정 파일에서 로드)
        # integrator.set_api_key(AIProvider.OPENAI, "your-openai-api-key")
        
        # 사용 가능한 모델 확인
        models = integrator.get_available_models()
        print(f"사용 가능한 모델: {models}")
        
        if models:
            # 테스트 요청
            request = AIRequest(
                task_type=TaskType.SUMMARIZATION,
                prompt="다음 텍스트를 요약해주세요: 인공지능은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 문제 해결 등의 작업을 수행할 수 있도록 하는 기술입니다.",
                system_prompt="당신은 전문적인 요약 어시스턴트입니다. 핵심 내용을 간결하게 정리해주세요."
            )
            
            # 응답 생성
            response = await integrator.generate_response(request, models[0])
            
            print(f"\n응답 성공: {response.success}")
            if response.success:
                print(f"모델: {response.model_name}")
                print(f"응답: {response.content}")
                print(f"토큰 사용: {response.usage}")
                print(f"처리 시간: {response.processing_time_ms:.1f}ms")
                print(f"예상 비용: ${response.cost_estimate:.6f}")
            else:
                print(f"오류: {response.error_message}")
        
        # 사용 통계
        stats = integrator.get_usage_stats()
        print(f"\n사용 통계: {stats}")
        
        # 정리
        integrator.cleanup()
    
    # 테스트 실행
    print("🤖 외부 AI 모델 통합 시스템 테스트 시작")
    asyncio.run(test_ai_integrator())
    print("✅ 테스트 완료!")