#!/usr/bin/env python3
"""
🎯 Module1 API 서비스
기존 conference_analysis.py를 FastAPI로 래핑
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import sys
import os
import tempfile
import json
from datetime import datetime

# 현재 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Module1: Conference Analysis API",
    description="컨퍼런스 분석 모듈 API",
    version="1.0.0"
)

# Ollama 인터페이스 로드
try:
    from shared.ollama_interface import OllamaInterface
    ollama = OllamaInterface()
    OLLAMA_AVAILABLE = True
    print(f"✅ Ollama 연결 성공: {len(ollama.available_models)}개 모델")
except Exception as e:
    OLLAMA_AVAILABLE = False
    print(f"❌ Ollama 연결 실패: {e}")

@app.get("/")
async def root():
    """서비스 정보"""
    return {
        "service": "Module1 - Conference Analysis",
        "version": "1.0.0",
        "status": "online",
        "ollama_available": OLLAMA_AVAILABLE,
        "models": ollama.available_models if OLLAMA_AVAILABLE else [],
        "endpoints": [
            "POST /analyze - 파일 분석",
            "GET /health - 상태 확인"
        ]
    }

@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "ollama": "available" if OLLAMA_AVAILABLE else "unavailable",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_files(files: List[UploadFile] = File(...)):
    """파일 분석 - 기존 로직 활용"""
    
    if not OLLAMA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ollama AI 서비스를 사용할 수 없습니다")
    
    if not files:
        raise HTTPException(status_code=400, detail="분석할 파일이 없습니다")
    
    try:
        results = {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "files_count": len(files),
            "files_analyzed": [],
            "comprehensive_summary": {},
            "ai_analysis": {}
        }
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = []
            
            # 파일 저장
            for file in files:
                temp_path = os.path.join(temp_dir, file.filename)
                content = await file.read()
                with open(temp_path, 'wb') as f:
                    f.write(content)
                saved_files.append({
                    "filename": file.filename,
                    "size": len(content),
                    "content_type": file.content_type,
                    "path": temp_path
                })
            
            # 파일별 분석
            for file_info in saved_files:
                file_result = await analyze_single_file(file_info)
                results["files_analyzed"].append(file_result)
            
            # 종합 분석 (AI 활용)
            if results["files_analyzed"]:
                comprehensive = await generate_comprehensive_analysis(results["files_analyzed"])
                results["comprehensive_summary"] = comprehensive
                
                # AI 분석 추가
                ai_analysis = await generate_ai_analysis(results["files_analyzed"])
                results["ai_analysis"] = ai_analysis
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

async def analyze_single_file(file_info: dict) -> dict:
    """단일 파일 분석"""
    filename = file_info["filename"]
    file_path = file_info["path"]
    
    result = {
        "filename": filename,
        "content_type": file_info["content_type"],
        "size": file_info["size"],
        "analysis_type": "unknown",
        "content": "",
        "metadata": {}
    }
    
    try:
        # 파일 확장자에 따른 분석
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.txt', '.md']:
            # 텍스트 파일 분석
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result.update({
                "analysis_type": "text",
                "content": content[:1000] + "..." if len(content) > 1000 else content,
                "character_count": len(content),
                "line_count": len(content.split('\n'))
            })
            
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 이미지 파일 분석 (OCR 시뮬레이션)
            result.update({
                "analysis_type": "image",
                "content": f"이미지 파일 감지: {filename}",
                "metadata": {"width": "unknown", "height": "unknown"}
            })
            
        elif ext in ['.mp3', '.wav', '.m4a']:
            # 오디오 파일 분석 (STT 시뮬레이션)
            result.update({
                "analysis_type": "audio", 
                "content": f"오디오 파일 감지: {filename}",
                "metadata": {"duration": "unknown", "format": ext}
            })
            
        elif ext in ['.mp4', '.avi', '.mov']:
            # 비디오 파일 분석
            result.update({
                "analysis_type": "video",
                "content": f"비디오 파일 감지: {filename}",
                "metadata": {"duration": "unknown", "resolution": "unknown"}
            })
        
        else:
            result["content"] = f"지원되지 않는 파일 형식: {ext}"
    
    except Exception as e:
        result["error"] = str(e)
    
    return result

async def generate_comprehensive_analysis(file_results: List[dict]) -> dict:
    """종합 분석 생성"""
    
    total_files = len(file_results)
    file_types = {}
    total_content = []
    
    for result in file_results:
        analysis_type = result.get("analysis_type", "unknown")
        file_types[analysis_type] = file_types.get(analysis_type, 0) + 1
        
        if result.get("content"):
            total_content.append(f"{result['filename']}: {result['content'][:200]}")
    
    return {
        "total_files": total_files,
        "file_types_distribution": file_types,
        "content_preview": total_content[:5],  # 최대 5개 파일 미리보기
        "analysis_timestamp": datetime.now().isoformat()
    }

async def generate_ai_analysis(file_results: List[dict]) -> dict:
    """AI 기반 종합 분석"""
    
    if not OLLAMA_AVAILABLE:
        return {"error": "AI 분석 불가 - Ollama 연결 없음"}
    
    try:
        # 텍스트 내용 통합
        combined_content = ""
        for result in file_results:
            if result.get("content") and result.get("analysis_type") == "text":
                combined_content += f"\n{result['filename']}:\n{result['content']}\n"
        
        if not combined_content.strip():
            return {"message": "텍스트 내용이 없어 AI 분석을 수행할 수 없습니다"}
        
        # AI 분석 실행
        analysis_prompt = f"""다음 파일 내용들을 분석해주세요:

{combined_content[:2000]}

분석 항목:
1. 주요 주제
2. 핵심 내용 요약
3. 중요 키워드
4. 전체적인 맥락"""

        ai_response = ollama.generate_response(
            prompt=analysis_prompt,
            model="qwen2.5:7b"
        )
        
        return {
            "ai_summary": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
            "model_used": "qwen2.5:7b",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"AI 분석 실패: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("🎯 Module1 API 서비스 시작...")
    print("📍 서비스: http://localhost:8001")
    print("📚 API 문서: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)