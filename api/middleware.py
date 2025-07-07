"""
솔로몬드 AI 시스템 - 미들웨어
FastAPI 미들웨어 및 요청 처리 로직
"""

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청 로깅 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 요청 정보 로깅
        print(f"📥 {request.method} {request.url.path}")
        
        # 요청 처리
        response = await call_next(request)
        
        # 응답 시간 계산
        process_time = time.time() - start_time
        
        # 응답 정보 로깅  
        print(f"📤 {response.status_code} ({process_time:.2f}s)")
        
        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class SecurityMiddleware(BaseHTTPMiddleware):
    """보안 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        # 보안 헤더 추가
        response = await call_next(request)
        
        # 보안 관련 헤더 설정
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response

class FileUploadMiddleware(BaseHTTPMiddleware):
    """파일 업로드 미들웨어"""
    
    def __init__(self, app, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        super().__init__(app)
        self.max_file_size = max_file_size
    
    async def dispatch(self, request: Request, call_next):
        # 파일 업로드 크기 제한 체크
        if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_file_size:
                return Response(
                    content="파일 크기가 너무 큽니다. 100MB 이하로 업로드해주세요.",
                    status_code=413
                )
        
        return await call_next(request)

# 미들웨어 설정 함수들
def add_logging_middleware(app):
    """로깅 미들웨어 추가"""
    app.add_middleware(RequestLoggingMiddleware)

def add_security_middleware(app):
    """보안 미들웨어 추가"""
    app.add_middleware(SecurityMiddleware)

def add_file_upload_middleware(app, max_size: int = 100 * 1024 * 1024):
    """파일 업로드 미들웨어 추가"""
    app.add_middleware(FileUploadMiddleware, max_file_size=max_size)

def setup_all_middleware(app):
    """모든 미들웨어 설정"""
    add_file_upload_middleware(app)
    add_security_middleware(app)
    add_logging_middleware(app)
    
    print("✅ 모든 미들웨어 설정 완료")
