"""
솔로몬드 AI 시스템 - API 라우트
FastAPI 기반 REST API 엔드포인트
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time
from typing import Optional

# 상대 import (모듈화된 구조)
try:
    from ..core.analyzer import get_analyzer, check_whisper_status
except ImportError:
    # 개발 중 절대 import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.analyzer import get_analyzer, check_whisper_status

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
            return JSONResponse({\n                \"success\": False,\n                \"error\": \"Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 실행하세요.\"\n            })\n        \n        # 분석기 가져오기\n        analyzer = get_analyzer(\"base\")\n        \n        # 음성 분석 실행\n        result = await analyzer.analyze_uploaded_file(\n            file_content=file_content,\n            filename=filename,\n            language=\"ko\"\n        )\n        \n        # 응답 형식을 기존 minimal_stt_test.py와 동일하게 맞춤\n        if result[\"success\"]:\n            return JSONResponse({\n                \"success\": True,\n                \"filename\": result[\"filename\"],\n                \"file_size\": result[\"file_size\"],\n                \"transcribed_text\": result[\"transcribed_text\"],\n                \"processing_time\": result[\"processing_time\"],\n                \"detected_language\": result[\"detected_language\"]\n            })\n        else:\n            return JSONResponse({\n                \"success\": False,\n                \"error\": result[\"error\"],\n                \"processing_time\": result.get(\"processing_time\", round(time.time() - start_time, 2))\n            })\n            \n    except Exception as e:\n        processing_time = round(time.time() - start_time, 2)\n        error_msg = str(e)\n        \n        print(f\"❌ API 오류: {error_msg}\")\n        \n        return JSONResponse({\n            \"success\": False,\n            \"error\": error_msg,\n            \"processing_time\": processing_time\n        })\n\n@router.get(\"/test\")\nasync def system_test():\n    \"\"\"\n    시스템 상태 테스트 API\n    \n    시스템의 현재 상태와 사용 가능한 기능들을 확인\n    \"\"\"\n    whisper_status = check_whisper_status()\n    analyzer = get_analyzer()\n    model_info = analyzer.get_model_info()\n    \n    return JSONResponse({\n        \"status\": \"OK\",\n        \"version\": \"3.0\",\n        \"python_version\": \"3.13+\",\n        \"whisper_available\": whisper_status[\"whisper_available\"],\n        \"model_info\": model_info,\n        \"supported_formats\": model_info[\"supported_formats\"],\n        \"features\": {\n            \"stt\": whisper_status[\"whisper_available\"],\n            \"translation\": True,\n            \"file_upload\": True,\n            \"modular_architecture\": True\n        }\n    })\n\n@router.get(\"/health\")\nasync def health_check():\n    \"\"\"\n    헬스 체크 API\n    \n    서버가 정상적으로 동작하는지 확인\n    \"\"\"\n    return {\"status\": \"healthy\", \"timestamp\": time.time()}\n\n@router.post(\"/analyze_batch\")\nasync def analyze_batch_files(\n    files: list[UploadFile] = File(...),\n    language: Optional[str] = \"ko\"\n):\n    \"\"\"\n    다중 파일 배치 분석 API (확장 기능)\n    \n    여러 음성 파일을 한 번에 처리\n    \"\"\"\n    results = []\n    analyzer = get_analyzer()\n    \n    for file in files:\n        try:\n            file_content = await file.read()\n            result = await analyzer.analyze_uploaded_file(\n                file_content=file_content,\n                filename=file.filename,\n                language=language\n            )\n            results.append({\n                \"filename\": file.filename,\n                \"result\": result\n            })\n        except Exception as e:\n            results.append({\n                \"filename\": file.filename,\n                \"result\": {\n                    \"success\": False,\n                    \"error\": str(e)\n                }\n            })\n    \n    successful_count = sum(1 for r in results if r[\"result\"][\"success\"])\n    \n    return JSONResponse({\n        \"batch_success\": True,\n        \"total_files\": len(files),\n        \"successful_files\": successful_count,\n        \"failed_files\": len(files) - successful_count,\n        \"results\": results\n    })\n\n@router.get(\"/models\")\nasync def list_available_models():\n    \"\"\"\n    사용 가능한 Whisper 모델 목록 API\n    \"\"\"\n    models = [\n        {\"name\": \"tiny\", \"size\": \"~39 MB\", \"speed\": \"매우 빠름\", \"accuracy\": \"낮음\"},\n        {\"name\": \"base\", \"size\": \"~74 MB\", \"speed\": \"빠름\", \"accuracy\": \"중간\"},\n        {\"name\": \"small\", \"size\": \"~244 MB\", \"speed\": \"보통\", \"accuracy\": \"좋음\"},\n        {\"name\": \"medium\", \"size\": \"~769 MB\", \"speed\": \"느림\", \"accuracy\": \"매우 좋음\"},\n        {\"name\": \"large\", \"size\": \"~1550 MB\", \"speed\": \"매우 느림\", \"accuracy\": \"최고\"}\n    ]\n    \n    return JSONResponse({\n        \"available_models\": models,\n        \"current_model\": get_analyzer().model_size,\n        \"recommendation\": \"base 모델이 속도와 정확도의 균형이 좋습니다.\"\n    })\n\n# 에러 핸들러\n@router.exception_handler(HTTPException)\nasync def http_exception_handler(request, exc):\n    return JSONResponse(\n        status_code=exc.status_code,\n        content={\"success\": False, \"error\": exc.detail}\n    )\n\n@router.exception_handler(Exception)\nasync def general_exception_handler(request, exc):\n    return JSONResponse(\n        status_code=500,\n        content={\"success\": False, \"error\": \"내부 서버 오류가 발생했습니다.\"}\n    )\n