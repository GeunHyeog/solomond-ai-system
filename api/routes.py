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
    from ..core.video_processor import get_video_processor, check_video_support
except ImportError:
    try:
        # 절대 import 시도
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from core.analyzer import get_analyzer, check_whisper_status
        from core.video_processor import get_video_processor, check_video_support
    except ImportError:
        # 최후의 수단: 함수 정의
        print("⚠️ core 모듈들을 import할 수 없습니다. 대체 함수를 사용합니다.")
        
        def get_analyzer(model_size="base"):
            class DummyAnalyzer:
                def get_model_info(self):
                    return {"model_size": "base", "model_loaded": False, "whisper_available": False, "supported_formats": [".mp3", ".wav", ".m4a"]}
                async def analyze_uploaded_file(self, file_content, filename, language="ko"):
                    return {"success": False, "error": "Analyzer not available"}
            return DummyAnalyzer()
        
        def check_whisper_status():
            return {"whisper_available": False, "import_error": "Module not found"}
        
        def get_video_processor():
            class DummyVideoProcessor:
                def is_video_file(self, filename):
                    return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                async def extract_audio_from_video(self, video_content, filename):
                    return {"success": False, "error": "Video processor not available"}
                async def get_video_info(self, video_content, filename):
                    return {"error": "Video processor not available"}
            return DummyVideoProcessor()
        
        def check_video_support():
            return {"supported_formats": [".mp4", ".avi", ".mov"], "ffmpeg_available": False}

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

@router.post("/process_video")
async def process_video(video_file: UploadFile = File(...)):
    """
    🎥 동영상 파일 처리 API (Phase 3 신규 기능)
    
    업로드된 동영상 파일에서 음성을 추출하고 STT 분석하여 텍스트로 변환
    """
    start_time = time.time()
    
    try:
        # 파일 기본 정보 수집
        filename = video_file.filename
        file_content = await video_file.read()
        file_size_mb = round(len(file_content) / (1024 * 1024), 2)
        
        print(f"🎬 동영상 파일 수신: {filename} ({file_size_mb} MB)")
        
        # 동영상 프로세서 가져오기
        video_processor = get_video_processor()
        
        # 동영상 파일 형식 확인
        if not video_processor.is_video_file(filename):
            return JSONResponse({
                "success": False,
                "error": f"지원하지 않는 동영상 형식입니다. 지원 형식: {check_video_support()['supported_formats']}"
            })
        
        # 동영상에서 음성 추출
        extraction_result = await video_processor.extract_audio_from_video(
            video_content=file_content,
            original_filename=filename
        )
        
        if not extraction_result["success"]:
            return JSONResponse({
                "success": False,
                "error": f"음성 추출 실패: {extraction_result['error']}",
                "processing_time": round(time.time() - start_time, 2),
                "install_guide": extraction_result.get("install_guide")
            })
        
        # Whisper 상태 확인
        whisper_status = check_whisper_status()
        if not whisper_status["whisper_available"]:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 실행하세요."
            })
        
        # 추출된 음성을 STT 분석
        analyzer = get_analyzer("base")
        stt_result = await analyzer.analyze_uploaded_file(
            file_content=extraction_result["audio_content"],
            filename=extraction_result["extracted_filename"],
            language="ko"
        )
        
        # 결과 통합
        if stt_result["success"]:
            return JSONResponse({
                "success": True,
                "original_filename": filename,
                "original_file_size": file_size_mb,
                "extracted_audio_filename": extraction_result["extracted_filename"],
                "transcribed_text": stt_result["transcribed_text"],
                "processing_time": round(time.time() - start_time, 2),
                "detected_language": stt_result["detected_language"],
                "extraction_method": extraction_result["extraction_method"],
                "file_type": "video"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": f"STT 분석 실패: {stt_result['error']}",
                "processing_time": round(time.time() - start_time, 2),
                "extraction_success": True,
                "extraction_method": extraction_result["extraction_method"]
            })
            
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ 동영상 처리 API 오류: {error_msg}")
        
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "processing_time": processing_time,
            "file_type": "video"
        })

@router.post("/video_info")
async def get_video_info(video_file: UploadFile = File(...)):
    """
    🎬 동영상 파일 정보 분석 API
    
    동영상 파일의 메타데이터와 호환성을 확인
    """
    try:
        filename = video_file.filename
        file_content = await video_file.read()
        
        video_processor = get_video_processor()
        info = await video_processor.get_video_info(file_content, filename)
        
        return JSONResponse({
            "success": True,
            "video_info": info
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@router.get("/video_support")
async def check_video_support_status():
    """
    🔧 동영상 지원 상태 확인 API
    
    시스템의 동영상 처리 지원 상태를 확인
    """
    support_info = check_video_support()
    video_processor = get_video_processor()
    
    return JSONResponse({
        "video_support": {
            "supported_formats": support_info["supported_formats"],
            "ffmpeg_available": support_info["ffmpeg_available"],
            "python_version": support_info.get("python_version", "3.13+"),
            "status": "available" if support_info["ffmpeg_available"] else "ffmpeg_required"
        },
        "recommendations": {
            "ffmpeg_required": not support_info["ffmpeg_available"],
            "install_guide": {
                "windows": "https://ffmpeg.org/download.html에서 다운로드 후 PATH 설정",
                "mac": "brew install ffmpeg",
                "ubuntu": "sudo apt update && sudo apt install ffmpeg"
            } if not support_info["ffmpeg_available"] else None
        }
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
    video_support = check_video_support()
    
    return JSONResponse({
        "status": "OK",
        "version": "3.0",
        "python_version": "3.13+",
        "whisper_available": whisper_status["whisper_available"],
        "model_info": model_info,
        "supported_formats": {
            "audio": model_info["supported_formats"],
            "video": video_support["supported_formats"]
        },
        "features": {
            "stt": whisper_status["whisper_available"],
            "translation": True,
            "file_upload": True,
            "modular_architecture": True,
            "video_processing": video_support["ffmpeg_available"],  # 🆕 동영상 지원 상태
            "batch_processing": True
        },
        "phase_3_status": {
            "video_support": "completed",
            "next_goals": ["multi_language", "ai_enhancement", "cloud_deployment"]
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
    
    여러 음성/동영상 파일을 한 번에 처리
    """
    results = []
    analyzer = get_analyzer()
    video_processor = get_video_processor()
    
    for file in files:
        try:
            file_content = await file.read()
            
            # 동영상 파일인지 확인
            if video_processor.is_video_file(file.filename):
                # 동영상 처리
                extraction_result = await video_processor.extract_audio_from_video(
                    video_content=file_content,
                    original_filename=file.filename
                )
                
                if extraction_result["success"]:
                    # 추출된 음성으로 STT 실행
                    result = await analyzer.analyze_uploaded_file(
                        file_content=extraction_result["audio_content"],
                        filename=extraction_result["extracted_filename"],
                        language=language
                    )
                    result["file_type"] = "video"
                    result["extraction_method"] = extraction_result["extraction_method"]
                else:
                    result = extraction_result
                    result["file_type"] = "video"
            else:
                # 일반 음성 파일 처리
                result = await analyzer.analyze_uploaded_file(
                    file_content=file_content,
                    filename=file.filename,
                    language=language
                )
                result["file_type"] = "audio"
            
            results.append({
                "filename": file.filename,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "result": {
                    "success": False,
                    "error": str(e),
                    "file_type": "unknown"
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

# 오류 처리는 FastAPI 앱 레벨에서 처리하도록 변경
# (APIRouter에서는 exception_handler 사용 불가)
