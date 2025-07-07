"""
솔로몬드 AI 시스템 - API 라우트
FastAPI 기반 REST API 엔드포인트
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time
from typing import Optional

# Import 처리 (개발 환경 호환)
try:
    # 상대 import 시도
    from ..core.analyzer import get_analyzer, check_whisper_status
except ImportError:
    try:
        # 절대 import 시도
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from core.analyzer import get_analyzer, check_whisper_status
    except ImportError:
        # 최후의 수단: 함수 정의
        print("⚠️ core.analyzer를 import할 수 없습니다. 대체 함수를 사용합니다.")
        
        def get_analyzer(model_size="base"):
            class DummyAnalyzer:
                def get_model_info(self):
                    return {"model_size": "base", "model_loaded": False, "whisper_available": False, "supported_formats": [".mp3", ".wav", ".m4a"]}
                async def analyze_uploaded_file(self, file_content, filename, language="ko"):
                    return {"success": False, "error": "Analyzer not available"}
            return DummyAnalyzer()
        
        def check_whisper_status():
            return {"whisper_available": False, "import_error": "Module not found"}

# API 라우터 생성
router = APIRouter()

@router.post("/process_audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """
    음성 파일 처리 API
    
    업로드된 음성 파일을 STT 분석하여 텍스트로 변환
    """
    start_time = time.time()
    
    try:
        # 파일 기본 정보 수집
        filename = audio_file.filename
        file_content = await audio_file.read()
        file_size_mb = round(len(file_content) / (1024 * 1024), 2)
        
        print(f"📁 파일 수신: {filename} ({file_size_mb} MB)")
        
        # Whisper 상태 확인
        whisper_status = check_whisper_status()
        if not whisper_status["whisper_available"]:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 실행하세요."
            })
        
        # 분석기 가져오기
        analyzer = get_analyzer("base")
        
        # 음성 분석 실행
        result = await analyzer.analyze_uploaded_file(
            file_content=file_content,
            filename=filename,
            language="ko"
        )
        
        # 응답 형식을 기존 minimal_stt_test.py와 동일하게 맞춤
        if result["success"]:
            return JSONResponse({
                "success": True,
                "filename": result["filename"],
                "file_size": result["file_size"],
                "transcribed_text": result["transcribed_text"],
                "processing_time": result["processing_time"],
                "detected_language": result["detected_language"]
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result["error"],
                "processing_time": result.get("processing_time", round(time.time() - start_time, 2))
            })
            
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ API 오류: {error_msg}")
        
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "processing_time": processing_time
        })

@router.get("/test")
async def system_test():
    """
    시스템 상태 테스트 API
    
    시스템의 현재 상태와 사용 가능한 기능들을 확인
    """
    whisper_status = check_whisper_status()
    analyzer = get_analyzer()
    model_info = analyzer.get_model_info()
    
    return JSONResponse({
        "status": "OK",
        "version": "3.0",
        "python_version": "3.13+",
        "whisper_available": whisper_status["whisper_available"],
        "model_info": model_info,
        "supported_formats": model_info["supported_formats"],
        "features": {
            "stt": whisper_status["whisper_available"],
            "translation": True,
            "file_upload": True,
            "modular_architecture": True
        }
    })

@router.get("/health")
async def health_check():
    """
    헬스 체크 API
    
    서버가 정상적으로 동작하는지 확인
    """
    return {"status": "healthy", "timestamp": time.time()}

@router.post("/analyze_batch")
async def analyze_batch_files(
    files: list[UploadFile] = File(...),
    language: Optional[str] = "ko"
):
    """
    다중 파일 배치 분석 API (확장 기능)
    
    여러 음성 파일을 한 번에 처리
    """
    results = []
    analyzer = get_analyzer()
    
    for file in files:
        try:
            file_content = await file.read()
            result = await analyzer.analyze_uploaded_file(
                file_content=file_content,
                filename=file.filename,
                language=language
            )
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "result": {
                    "success": False,
                    "error": str(e)
                }
            })
    
    successful_count = sum(1 for r in results if r["result"]["success"])
    
    return JSONResponse({
        "batch_success": True,
        "total_files": len(files),
        "successful_files": successful_count,
        "failed_files": len(files) - successful_count,
        "results": results
    })

@router.get("/models")
async def list_available_models():
    """
    사용 가능한 Whisper 모델 목록 API
    """
    models = [
        {"name": "tiny", "size": "~39 MB", "speed": "매우 빠름", "accuracy": "낮음"},
        {"name": "base", "size": "~74 MB", "speed": "빠름", "accuracy": "중간"},
        {"name": "small", "size": "~244 MB", "speed": "보통", "accuracy": "좋음"},
        {"name": "medium", "size": "~769 MB", "speed": "느림", "accuracy": "매우 좋음"},
        {"name": "large", "size": "~1550 MB", "speed": "매우 느림", "accuracy": "최고"}
    ]
    
    return JSONResponse({
        "available_models": models,
        "current_model": get_analyzer().get_model_info().get("model_size", "base"),
        "recommendation": "base 모델이 속도와 정확도의 균형이 좋습니다."
    })

# 에러 핸들러
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "내부 서버 오류가 발생했습니다."}
    )
