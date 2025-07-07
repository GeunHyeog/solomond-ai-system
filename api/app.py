"""
솔로몬드 AI 시스템 - FastAPI 앱 팩토리
FastAPI 애플리케이션 생성 및 설정
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Import 처리 (개발 환경 호환)
try:
    # 상대 import 시도
    from .routes import router
    from ..ui.templates import get_main_template
except ImportError:
    try:
        # 절대 import 시도
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from api.routes import router
        from ui.templates import get_main_template
    except ImportError:
        print("⚠️ 모듈 import 실패. 기본 라우터를 사용합니다.")
        from fastapi import APIRouter
        router = APIRouter()
        
        @router.get("/test")
        async def fallback_test():
            return {"status": "fallback_mode", "message": "모듈화 시스템을 로드할 수 없습니다."}
        
        def get_main_template():
            return """
            <!DOCTYPE html>
            <html>
            <head><title>시스템 로딩 중</title></head>
            <body>
                <h1>🔄 시스템 초기화 중...</h1>
                <p>모듈화된 시스템을 로드하는 중입니다.</p>
                <p>잠시 후 다시 시도하거나 <a href="/api/test">시스템 상태</a>를 확인해주세요.</p>
            </body>
            </html>
            """

def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 팩토리
    
    Returns:
        설정된 FastAPI 앱 인스턴스
    """
    
    # FastAPI 앱 생성
    app = FastAPI(
        title="솔로몬드 AI 시스템",
        description="실제 내용을 읽고 분석하는 차세대 AI 플랫폼",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발용, 프로덕션에서는 제한 필요
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API 라우터 포함
    app.include_router(router, prefix="/api", tags=["STT API"])
    
    # 메인 페이지 라우트
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """메인 페이지 - HTML 템플릿 반환"""
        return get_main_template()
    
    # 레거시 호환성을 위한 라우트 (기존 minimal_stt_test.py와 동일)
    @app.post("/process_audio")
    async def legacy_process_audio(audio_file):
        """레거시 호환성을 위한 라우트"""
        # API 라우터의 process_audio를 직접 호출
        from api.routes import process_audio
        return await process_audio(audio_file)
    
    @app.get("/test")
    async def legacy_test():
        """레거시 호환성을 위한 테스트 라우트"""
        from api.routes import system_test
        return await system_test()
    
    # 헬스체크 라우트
    @app.get("/health")
    async def health():
        """간단한 헬스체크"""
        return {"status": "ok", "version": "3.0"}
    
    return app

def run_app(host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
    """
    앱 실행 헬퍼 함수
    
    Args:
        host: 바인딩할 호스트 주소
        port: 포트 번호
        debug: 디버그 모드 여부
    """
    app = create_app()
    
    print("=" * 60)
    print("🚀 솔로몬드 AI 시스템 v3.0 시작")
    print("=" * 60)
    print(f"📍 주소: http://{host}:{port}")
    print(f"📖 API 문서: http://{host}:{port}/docs")
    print(f"🧪 테스트: http://{host}:{port}/test")
    print("=" * 60)
    
    try:
        uvicorn.run(
            app, 
            host=host, 
            port=port, 
            reload=debug,
            log_level="info" if debug else "warning"
        )
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")
        print("🔄 레거시 모드로 전환을 고려해주세요.")

if __name__ == "__main__":
    run_app(debug=True)
