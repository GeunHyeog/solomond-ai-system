#!/usr/bin/env python3
"""
보안 강화 API 서버
FastAPI 기반 보안 API 서버화
"""

import asyncio
import time
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from utils.logger import get_logger

# 데이터 모델들
class AnalysisRequest(BaseModel):
    """분석 요청 모델"""
    session_id: str = Field(..., description="세션 ID")
    audio_files: List[str] = Field(default=[], description="오디오 파일 경로들")
    image_files: List[str] = Field(default=[], description="이미지 파일 경로들")
    basic_info: Dict[str, Any] = Field(default={}, description="기본 정보")
    context: Dict[str, Any] = Field(default={}, description="컨텍스트 정보")

class StreamingRequest(BaseModel):
    """스트리밍 요청 모델"""
    session_id: str = Field(..., description="세션 ID")
    duration: int = Field(default=30, description="스트리밍 지속시간(초)")
    participants: str = Field(default="실시간 입력", description="참가자 정보")

class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="검색 쿼리")
    context: Dict[str, Any] = Field(default={}, description="검색 컨텍스트")
    sites: List[str] = Field(default=[], description="대상 사이트 목록")

class APIResponse(BaseModel):
    """API 응답 모델"""
    status: str = Field(..., description="응답 상태")
    data: Dict[str, Any] = Field(default={}, description="응답 데이터")
    message: str = Field(default="", description="응답 메시지")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class SecurityConfig:
    """보안 설정"""
    api_key_length: int = 32
    session_timeout: int = 3600  # 1시간
    rate_limit_requests: int = 100  # 시간당 요청 수
    rate_limit_window: int = 3600  # 1시간
    allowed_hosts: List[str] = None
    cors_origins: List[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    def __post_init__(self):
        if self.allowed_hosts is None:
            self.allowed_hosts = ["localhost", "127.0.0.1"]
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:8503", "http://127.0.0.1:8503"]

class SecurityAPIServer:
    """보안 강화 API 서버"""
    
    def __init__(self, config: SecurityConfig = None):
        self.logger = get_logger(f'{__name__}.SecurityAPIServer')
        self.config = config or SecurityConfig()
        
        # 보안 관련 저장소
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # FastAPI 앱 초기화
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            self.logger.error("FastAPI 모듈이 설치되지 않음")
        
        self.logger.info("보안 API 서버 초기화 완료")
    
    def _create_fastapi_app(self) -> FastAPI:
        """FastAPI 앱 생성"""
        
        app = FastAPI(
            title="솔로몬드 AI 보안 API",
            description="보안 강화된 AI 분석 API 서버",
            version="2.4.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS 미들웨어 추가
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # 신뢰할 수 있는 호스트 미들웨어
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.allowed_hosts
        )
        
        # 미들웨어 등록
        app.middleware("http")(self._security_middleware)
        
        # 라우트 등록
        self._register_routes(app)
        
        return app
    
    async def _security_middleware(self, request: Request, call_next):
        """보안 미들웨어"""
        
        start_time = time.time()
        
        # 1. Rate Limiting
        client_ip = request.client.host
        if not await self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # 2. 요청 크기 제한
        if hasattr(request, 'content_length') and request.content_length:
            if request.content_length > self.config.max_file_size:
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request entity too large"}
                )
        
        # 3. 요청 처리
        try:
            response = await call_next(request)
            
            # 4. 응답 헤더 추가
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            # 5. 처리 시간 로깅
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            self.logger.info(f"요청 처리: {request.method} {request.url.path} - {process_time:.3f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"요청 처리 오류: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Rate Limit 체크"""
        
        current_time = time.time()
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = {
                "requests": [],
                "blocked_until": 0
            }
        
        client_data = self.rate_limits[client_ip]
        
        # 차단 시간 확인
        if current_time < client_data["blocked_until"]:
            return False
        
        # 오래된 요청 기록 제거
        window_start = current_time - self.config.rate_limit_window
        client_data["requests"] = [
            req_time for req_time in client_data["requests"] 
            if req_time > window_start
        ]
        
        # 요청 수 체크
        if len(client_data["requests"]) >= self.config.rate_limit_requests:
            # 1시간 차단
            client_data["blocked_until"] = current_time + 3600
            self.logger.warning(f"Rate limit exceeded for {client_ip}")
            return False
        
        # 현재 요청 기록
        client_data["requests"].append(current_time)
        return True
    
    def _register_routes(self, app: FastAPI):
        """라우트 등록"""
        
        security = HTTPBearer()
        
        @app.get("/", response_model=APIResponse)
        async def root():
            """API 정보"""
            return APIResponse(
                status="success",
                data={
                    "service": "솔로몬드 AI 보안 API",
                    "version": "2.4.0",
                    "endpoints": [
                        "/auth/create-key",
                        "/analysis/batch",
                        "/streaming/start",
                        "/search/jewelry",
                        "/health"
                    ]
                },
                message="API 서버 정상 작동"
            )
        
        @app.post("/auth/create-key", response_model=APIResponse)
        async def create_api_key(
            request: Dict[str, str],
            background_tasks: BackgroundTasks
        ):
            """API 키 생성"""
            
            try:
                # API 키 생성
                api_key = secrets.token_urlsafe(self.config.api_key_length)
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                
                # API 키 정보 저장
                self.api_keys[key_hash] = {
                    "created_at": datetime.now().isoformat(),
                    "description": request.get("description", ""),
                    "permissions": request.get("permissions", ["read", "write"]),
                    "usage_count": 0,
                    "last_used": None
                }
                
                # 백그라운드에서 키 정리 작업 스케줄링
                background_tasks.add_task(self._cleanup_expired_keys)
                
                return APIResponse(
                    status="success",
                    data={
                        "api_key": api_key,
                        "expires_in": "30 days",
                        "permissions": self.api_keys[key_hash]["permissions"]
                    },
                    message="API 키 생성 완료"
                )
                
            except Exception as e:
                self.logger.error(f"API 키 생성 실패: {str(e)}")
                raise HTTPException(status_code=500, detail="API 키 생성 실패")
        
        @app.post("/analysis/batch", response_model=APIResponse)
        async def batch_analysis(
            request: AnalysisRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """배치 분석 (인증 필요)"""
            
            # 인증 확인
            if not await self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                # 실제 분석 엔진 호출 (시뮬레이션)
                analysis_result = await self._run_batch_analysis(request)
                
                return APIResponse(
                    status="success",
                    data=analysis_result,
                    message="배치 분석 완료"
                )
                
            except Exception as e:
                self.logger.error(f"배치 분석 실패: {str(e)}")
                raise HTTPException(status_code=500, detail="분석 처리 실패")
        
        @app.post("/streaming/start", response_model=APIResponse)
        async def start_streaming(
            request: StreamingRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """실시간 스트리밍 시작"""
            
            if not await self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                # 스트리밍 세션 생성
                session_result = await self._create_streaming_session(request)
                
                return APIResponse(
                    status="success",
                    data=session_result,
                    message="스트리밍 세션 시작"
                )
                
            except Exception as e:
                self.logger.error(f"스트리밍 시작 실패: {str(e)}")
                raise HTTPException(status_code=500, detail="스트리밍 시작 실패")
        
        @app.post("/search/jewelry", response_model=APIResponse)
        async def jewelry_search(
            request: SearchRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """주얼리 검색"""
            
            if not await self._verify_api_key(credentials.credentials):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                # 브라우저 자동화 검색 실행
                search_result = await self._run_jewelry_search(request)
                
                return APIResponse(
                    status="success",
                    data=search_result,
                    message="주얼리 검색 완료"
                )
                
            except Exception as e:
                self.logger.error(f"주얼리 검색 실패: {str(e)}")
                raise HTTPException(status_code=500, detail="검색 처리 실패")
        
        @app.get("/health", response_model=APIResponse)
        async def health_check():
            """헬스 체크"""
            
            health_data = {
                "server_status": "running",
                "uptime": time.time(),
                "active_sessions": len(self.sessions),
                "api_keys": len(self.api_keys),
                "memory_usage": "검사 필요"
            }
            
            return APIResponse(
                status="success",
                data=health_data,
                message="서버 정상 작동"
            )
    
    async def _verify_api_key(self, api_key: str) -> bool:
        """API 키 검증"""
        
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            if key_hash in self.api_keys:
                # 사용 통계 업데이트
                self.api_keys[key_hash]["usage_count"] += 1
                self.api_keys[key_hash]["last_used"] = datetime.now().isoformat()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"API 키 검증 실패: {str(e)}")
            return False
    
    async def _run_batch_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """배치 분석 실행 (시뮬레이션)"""
        
        # 실제 환경에서는 real_analysis_engine 호출
        await asyncio.sleep(1)  # 분석 시간 시뮬레이션
        
        return {
            "session_id": request.session_id,
            "processed_files": {
                "audio": len(request.audio_files),
                "image": len(request.image_files)
            },
            "analysis_results": {
                "summary": "주얼리 구매 상담 분석 완료",
                "sentiment": "positive",
                "keywords": ["결혼반지", "예산", "상담"],
                "recommendations": ["제품 카탈로그 준비", "가격 비교 자료"]
            },
            "processing_time": 1.0,
            "confidence": 0.87
        }
    
    async def _create_streaming_session(self, request: StreamingRequest) -> Dict[str, Any]:
        """스트리밍 세션 생성"""
        
        session_id = request.session_id
        
        # 세션 정보 저장
        self.sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "duration": request.duration,
            "participants": request.participants,
            "status": "active",
            "expires_at": (datetime.now() + timedelta(seconds=request.duration)).isoformat()
        }
        
        return {
            "session_id": session_id,
            "status": "started",
            "duration": request.duration,
            "websocket_url": f"ws://localhost:8000/streaming/{session_id}",
            "expires_at": self.sessions[session_id]["expires_at"]
        }
    
    async def _run_jewelry_search(self, request: SearchRequest) -> Dict[str, Any]:
        """주얼리 검색 실행 (시뮬레이션)"""
        
        # 실제 환경에서는 browser_automation_engine 호출
        await asyncio.sleep(0.5)  # 검색 시간 시뮬레이션
        
        return {
            "query": request.query,
            "search_results": {
                "total_sites": 3,
                "successful_searches": 2,
                "products_found": 15,
                "price_range": "150만원 - 300만원"
            },
            "recommendations": [
                "브라이달 전문 매장 방문 권장",
                "예산 범위 내 다양한 옵션 확인 가능"
            ],
            "processing_time": 0.5
        }
    
    async def _cleanup_expired_keys(self):
        """만료된 API 키 정리"""
        
        current_time = datetime.now()
        expired_keys = []
        
        for key_hash, key_info in self.api_keys.items():
            created_at = datetime.fromisoformat(key_info["created_at"])
            if (current_time - created_at).days > 30:  # 30일 만료
                expired_keys.append(key_hash)
        
        for key_hash in expired_keys:
            del self.api_keys[key_hash]
        
        if expired_keys:
            self.logger.info(f"만료된 API 키 {len(expired_keys)}개 정리 완료")
    
    async def start_server(self, host: str = "127.0.0.1", port: int = 8000):
        """API 서버 시작"""
        
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI가 설치되지 않아 서버를 시작할 수 없습니다")
            return False
        
        try:
            self.logger.info(f"보안 API 서버 시작: http://{host}:{port}")
            
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"서버 시작 실패: {str(e)}")
            return False

# 사용 예시
async def demo_security_api():
    """보안 API 서버 데모"""
    
    print("=== 보안 API 서버 데모 ===")
    
    # 보안 설정
    config = SecurityConfig(
        rate_limit_requests=50,
        allowed_hosts=["localhost", "127.0.0.1", "192.168.1.100"],
        cors_origins=["http://localhost:8503"]
    )
    
    # 서버 생성
    server = SecurityAPIServer(config)
    
    if server.app:
        print("FastAPI 서버 설정 완료")
        print("서버 시작 방법:")
        print("  python -m uvicorn core.security_api_server:app --host 127.0.0.1 --port 8000")
        print("  또는 server.start_server() 호출")
        
        # 데모용으로 짧은 시간만 실행
        print("\n데모 서버를 3초간 실행합니다...")
        
        # 실제 서버 시작 (백그라운드)
        import asyncio
        server_task = asyncio.create_task(server.start_server(port=8001))
        
        # 3초 대기 후 종료
        await asyncio.sleep(3)
        server_task.cancel()
        
        print("데모 완료")
    else:
        print("FastAPI 설치 필요: pip install fastapi uvicorn")

if __name__ == "__main__":
    asyncio.run(demo_security_api())