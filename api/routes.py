"""
솔로몬드 AI 시스템 - API 라우트
FastAPI 기반 REST API 엔드포인트 (Phase 3.3 AI 고도화)
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import time
from typing import Optional

# Import 처리 (개발 환경 호환)
try:
    # 상대 import 시도
    from ..core.analyzer import get_analyzer, check_whisper_status, get_language_support
    from ..core.video_processor import get_video_processor, check_video_support
    from ..core.speaker_analyzer import get_speaker_analyzer, check_speaker_analysis_support
except ImportError:
    try:
        # 절대 import 시도
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from core.analyzer import get_analyzer, check_whisper_status, get_language_support
        from core.video_processor import get_video_processor, check_video_support
        from core.speaker_analyzer import get_speaker_analyzer, check_speaker_analysis_support
    except ImportError:
        # 최후의 수단: 함수 정의
        print("⚠️ core 모듈들을 import할 수 없습니다. 대체 함수를 사용합니다.")
        
        def get_analyzer(model_size="base"):
            class DummyAnalyzer:
                def get_model_info(self):
                    return {"model_size": "base", "model_loaded": False, "whisper_available": False, "supported_formats": [".mp3", ".wav", ".m4a"]}
                async def analyze_uploaded_file(self, file_content, filename, language="auto"):
                    return {"success": False, "error": "Analyzer not available"}
            return DummyAnalyzer()
        
        def check_whisper_status():
            return {"whisper_available": False, "import_error": "Module not found"}
        
        def get_language_support():
            return {"supported_languages": {}, "auto_detection": False}
        
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
        
        def get_speaker_analyzer():
            class DummySpeakerAnalyzer:
                async def analyze_uploaded_file(self, file_content, filename, use_advanced=True):
                    return {"success": False, "error": "Speaker analyzer not available"}
                def get_capabilities(self):
                    return {"supported_formats": [".mp3", ".wav"], "pyannote_available": False}
            return DummySpeakerAnalyzer()
        
        def check_speaker_analysis_support():
            return {"pyannote_available": False, "algorithms": {}}

# API 라우터 생성
router = APIRouter()

@router.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    language: str = Query("auto", description="언어 코드 (auto, ko, en, zh, ja 등)")
):
    """
    음성 파일 처리 API (다국어 지원)
    
    업로드된 음성 파일을 STT 분석하여 텍스트로 변환
    
    Args:
        audio_file: 업로드할 음성 파일
        language: 인식할 언어 (auto=자동감지, ko=한국어, en=영어, zh=중국어, ja=일본어 등)
    """
    start_time = time.time()
    
    try:
        # 파일 기본 정보 수집
        filename = audio_file.filename
        file_content = await audio_file.read()
        file_size_mb = round(len(file_content) / (1024 * 1024), 2)
        
        print(f"📁 파일 수신: {filename} ({file_size_mb} MB), 언어: {language}")
        
        # Whisper 상태 확인
        whisper_status = check_whisper_status()
        if not whisper_status["whisper_available"]:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 실행하세요."
            })
        
        # 분석기 가져오기
        analyzer = get_analyzer("base")
        
        # 🆕 다국어 음성 분석 실행
        result = await analyzer.analyze_uploaded_file(
            file_content=file_content,
            filename=filename,
            language=language
        )
        
        # 응답 형식 (다국어 정보 포함)
        if result["success"]:
            return JSONResponse({
                "success": True,
                "filename": result["filename"],
                "file_size": result["file_size"],
                "transcribed_text": result["transcribed_text"],
                "processing_time": result["processing_time"],
                "detected_language": result["detected_language"],
                "language_info": result.get("language_info", {}),
                "requested_language": result.get("requested_language", language),
                "confidence": result.get("confidence", 0.0)
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result["error"],
                "processing_time": result.get("processing_time", round(time.time() - start_time, 2)),
                "requested_language": language
            })
            
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ API 오류: {error_msg}")
        
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "processing_time": processing_time,
            "requested_language": language
        })

@router.post("/analyze_speakers")
async def analyze_speakers(
    audio_file: UploadFile = File(...),
    use_advanced: bool = Query(True, description="고급 분석 사용 여부 (PyAnnote AI)")
):
    """
    🎭 화자 구분 분석 API (Phase 3.3 신규)
    
    업로드된 음성 파일에서 여러 화자를 구분하고 발언 시간을 분석
    
    Args:
        audio_file: 업로드할 음성 파일
        use_advanced: 고급 AI 분석 사용 여부 (기본: True)
    """
    start_time = time.time()
    
    try:
        # 파일 기본 정보 수집
        filename = audio_file.filename
        file_content = await audio_file.read()
        file_size_mb = round(len(file_content) / (1024 * 1024), 2)
        
        print(f"🎭 화자 구분 파일 수신: {filename} ({file_size_mb} MB)")
        
        # 화자 분석기 가져오기
        speaker_analyzer = get_speaker_analyzer()
        
        # 화자 구분 분석 실행
        result = await speaker_analyzer.analyze_uploaded_file(
            file_content=file_content,
            filename=filename,
            use_advanced=use_advanced
        )
        
        # 응답 형식
        if result["success"]:
            return JSONResponse({
                "success": True,
                "filename": result["filename"],
                "file_size": result["file_size"],
                "processing_time": result["processing_time"],
                "analysis_method": result["method"],
                "total_duration": result["total_duration"],
                "num_speakers": result["num_speakers"],
                "segments": result["segments"],
                "speaker_statistics": result["speaker_statistics"],
                "analysis_info": result.get("analysis_info", {}),
                "feature_type": "speaker_diarization"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result["error"],
                "processing_time": result.get("processing_time", round(time.time() - start_time, 2)),
                "fallback_available": result.get("fallback") is not None,
                "feature_type": "speaker_diarization"
            })
            
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ 화자 구분 API 오류: {error_msg}")
        
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "processing_time": processing_time,
            "feature_type": "speaker_diarization"
        })

@router.get("/speaker_support")
async def get_speaker_support_info():
    """
    🎭 화자 구분 지원 정보 API (Phase 3.3 신규)
    
    시스템의 화자 구분 기능 지원 상태를 확인
    """
    try:
        support_info = check_speaker_analysis_support()
        speaker_analyzer = get_speaker_analyzer()
        capabilities = speaker_analyzer.get_capabilities()
        
        return JSONResponse({
            "success": True,
            "speaker_diarization": {
                "supported_formats": capabilities["supported_formats"],
                "pyannote_available": capabilities["pyannote_available"],
                "max_speakers": capabilities["max_speakers"],
                "min_segment_duration": capabilities["min_segment_duration"],
                "algorithms": capabilities["algorithms"]
            },
            "recommendations": {
                "pyannote_installation": not capabilities["pyannote_available"],
                "install_guide": {
                    "command": "pip install pyannote.audio torch",
                    "note": "고급 AI 기반 화자 구분을 위해 권장"
                } if not capabilities["pyannote_available"] else None
            },
            "phase": capabilities["phase"]
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@router.post("/process_video")
async def process_video(
    video_file: UploadFile = File(...),
    language: str = Query("auto", description="언어 코드 (auto, ko, en, zh, ja 등)")
):
    """
    🎥 동영상 파일 처리 API (다국어 지원)
    
    업로드된 동영상 파일에서 음성을 추출하고 STT 분석하여 텍스트로 변환
    
    Args:
        video_file: 업로드할 동영상 파일
        language: 인식할 언어 (auto=자동감지, ko=한국어, en=영어, zh=중국어, ja=일본어 등)
    """
    start_time = time.time()
    
    try:
        # 파일 기본 정보 수집
        filename = video_file.filename
        file_content = await video_file.read()
        file_size_mb = round(len(file_content) / (1024 * 1024), 2)
        
        print(f"🎬 동영상 파일 수신: {filename} ({file_size_mb} MB), 언어: {language}")
        
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
                "install_guide": extraction_result.get("install_guide"),
                "requested_language": language
            })
        
        # Whisper 상태 확인
        whisper_status = check_whisper_status()
        if not whisper_status["whisper_available"]:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 실행하세요."
            })
        
        # 🆕 추출된 음성을 다국어 STT 분석
        analyzer = get_analyzer("base")
        stt_result = await analyzer.analyze_uploaded_file(
            file_content=extraction_result["audio_content"],
            filename=extraction_result["extracted_filename"],
            language=language
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
                "language_info": stt_result.get("language_info", {}),
                "requested_language": language,
                "confidence": stt_result.get("confidence", 0.0),
                "extraction_method": extraction_result["extraction_method"],
                "file_type": "video"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": f"STT 분석 실패: {stt_result['error']}",
                "processing_time": round(time.time() - start_time, 2),
                "extraction_success": True,
                "extraction_method": extraction_result["extraction_method"],
                "requested_language": language
            })
            
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ 동영상 처리 API 오류: {error_msg}")
        
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "processing_time": processing_time,
            "file_type": "video",
            "requested_language": language
        })

@router.get("/language_support")
async def get_language_support_info():
    """
    🌍 언어 지원 정보 API (Phase 3.2)
    
    시스템에서 지원하는 언어 목록과 기능을 확인
    """
    try:
        language_info = get_language_support()
        analyzer = get_analyzer()
        
        return JSONResponse({
            "success": True,
            "supported_languages": language_info["supported_languages"],
            "auto_detection": language_info["auto_detection"],
            "default_language": language_info["default_language"],
            "total_languages": len(language_info["supported_languages"]),
            "phase": language_info["phase"],
            "features": {
                "auto_language_detection": True,
                "confidence_scoring": True,
                "multi_language_batch": True,
                "real_time_detection": True
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@router.post("/detect_language")
async def detect_language(audio_file: UploadFile = File(...)):
    """
    🔍 언어 감지 전용 API (Phase 3.2)
    
    음성 파일의 언어만 감지하고 STT는 실행하지 않음
    """
    try:
        filename = audio_file.filename
        file_content = await audio_file.read()
        
        # 임시 파일 생성하여 언어 감지
        import tempfile
        import os
        
        file_ext = filename.split('.')[-1] if '.' in filename else 'tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            analyzer = get_analyzer()
            detection_result = analyzer.detect_language(temp_path)
            
            return JSONResponse({
                "success": detection_result["success"],
                "filename": filename,
                "detected_language": detection_result.get("detected_language"),
                "confidence": detection_result.get("confidence", 0.0),
                "language_info": detection_result.get("language_info", {}),
                "all_probabilities": detection_result.get("all_probabilities", {}),
                "error": detection_result.get("error")
            })
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
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
    시스템 상태 테스트 API (Phase 3.3 업데이트)
    
    시스템의 현재 상태와 사용 가능한 기능들을 확인
    """
    whisper_status = check_whisper_status()
    analyzer = get_analyzer()
    model_info = analyzer.get_model_info()
    video_support = check_video_support()
    language_support = get_language_support()
    speaker_support = check_speaker_analysis_support()
    
    return JSONResponse({
        "status": "OK",
        "version": "3.3",
        "python_version": "3.13+",
        "whisper_available": whisper_status["whisper_available"],
        "model_info": model_info,
        "supported_formats": {
            "audio": model_info["supported_formats"],
            "video": video_support["supported_formats"]
        },
        "language_support": {
            "total_languages": len(language_support["supported_languages"]),
            "auto_detection": language_support["auto_detection"],
            "supported_languages": list(language_support["supported_languages"].keys())
        },
        "ai_features": {
            "speaker_diarization": speaker_support.get("pyannote_available", False),
            "advanced_algorithms": len(speaker_support.get("algorithms", {}))
        },
        "features": {
            "stt": whisper_status["whisper_available"],
            "translation": True,
            "file_upload": True,
            "modular_architecture": True,
            "video_processing": video_support["ffmpeg_available"],
            "batch_processing": True,
            "multilingual_support": True,
            "auto_language_detection": True,
            "confidence_scoring": True,
            "speaker_diarization": True,  # 🆕 화자 구분
            "ai_enhancement": True  # 🆕 AI 고도화
        },
        "phase_status": {
            "phase_3_1": "completed - 동영상 지원",
            "phase_3_2": "completed - 다국어 확장",
            "phase_3_3": "in_progress - AI 고도화",
            "current_milestone": "화자 구분 기능 개발",
            "next_goals": ["감정 분석", "자동 요약", "주얼리 특화"]
        }
    })

@router.get("/health")
async def health_check():
    """
    헬스 체크 API
    
    서버가 정상적으로 동작하는지 확인
    """
    return {"status": "healthy", "timestamp": time.time(), "phase": "3.3"}

@router.post("/analyze_batch")
async def analyze_batch_files(
    files: list[UploadFile] = File(...),
    language: str = Query("auto", description="언어 코드 (auto, ko, en, zh, ja 등)")
):
    """
    다중 파일 배치 분석 API (다국어 지원)
    
    여러 음성/동영상 파일을 한 번에 처리
    
    Args:
        files: 업로드할 파일들
        language: 인식할 언어 (auto=자동감지, ko=한국어, en=영어 등)
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
                    # 추출된 음성으로 다국어 STT 실행
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
                # 일반 음성 파일 처리 (다국어 지원)
                result = await analyzer.analyze_uploaded_file(
                    file_content=file_content,
                    filename=file.filename,
                    language=language
                )
                result["file_type"] = "audio"
            
            # 요청된 언어 정보 추가
            result["requested_language"] = language
            
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
                    "file_type": "unknown",
                    "requested_language": language
                }
            })
    
    successful_count = sum(1 for r in results if r["result"]["success"])
    
    # 언어별 통계
    language_stats = {}
    for r in results:
        if r["result"]["success"]:
            detected_lang = r["result"].get("detected_language", "unknown")
            language_stats[detected_lang] = language_stats.get(detected_lang, 0) + 1
    
    return JSONResponse({
        "batch_success": True,
        "total_files": len(files),
        "successful_files": successful_count,
        "failed_files": len(files) - successful_count,
        "requested_language": language,
        "language_statistics": language_stats,
        "results": results
    })

@router.get("/models")
async def list_available_models():
    """
    사용 가능한 Whisper 모델 목록 API
    """
    models = [
        {"name": "tiny", "size": "~39 MB", "speed": "매우 빠름", "accuracy": "낮음", "languages": "99개 언어"},
        {"name": "base", "size": "~74 MB", "speed": "빠름", "accuracy": "중간", "languages": "99개 언어"},
        {"name": "small", "size": "~244 MB", "speed": "보통", "accuracy": "좋음", "languages": "99개 언어"},
        {"name": "medium", "size": "~769 MB", "speed": "느림", "accuracy": "매우 좋음", "languages": "99개 언어"},
        {"name": "large", "size": "~1550 MB", "speed": "매우 느림", "accuracy": "최고", "languages": "99개 언어"}
    ]
    
    return JSONResponse({
        "available_models": models,
        "current_model": get_analyzer().get_model_info().get("model_size", "base"),
        "recommendation": "base 모델이 속도와 정확도의 균형이 좋습니다.",
        "multilingual_support": "모든 모델이 99개 언어를 지원합니다.",
        "auto_detection": "언어 자동 감지 기능 포함"
    })

# 오류 처리는 FastAPI 앱 레벨에서 처리하도록 변경
# (APIRouter에서는 exception_handler 사용 불가)
