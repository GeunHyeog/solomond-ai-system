#!/usr/bin/env python3
"""
🚀 SOLOMOND AI API Gateway
통합 진입점 - 모든 모듈 API를 하나로 관리
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  
from typing import List, Optional
import httpx
import asyncio
import json
from datetime import datetime

app = FastAPI(
    title="SOLOMOND AI API Gateway",
    description="4개 모듈 통합 API 게이트웨이",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 정보
SERVICES = {
    "module1": {"host": "localhost", "port": 8001, "name": "컨퍼런스 분석"},
    "module2": {"host": "localhost", "port": 8002, "name": "웹 크롤러"},
    "module3": {"host": "localhost", "port": 8003, "name": "보석 분석"},
    "module4": {"host": "localhost", "port": 8004, "name": "3D CAD 변환"}
}

@app.get("/")
async def root():
    """메인 대시보드 - 모든 서비스 상태 표시"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SOLOMOND AI 통합 대시보드</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .services {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
            .service {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .service h3 {{ color: #333; margin-top: 0; }}
            .status {{ display: inline-block; padding: 5px 10px; border-radius: 5px; color: white; font-size: 12px; }}
            .online {{ background: #4CAF50; }}
            .offline {{ background: #f44336; }}
            .btn {{ background: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; }}
            .btn:hover {{ background: #1976D2; }}
            .api-section {{ margin-top: 40px; background: white; padding: 20px; border-radius: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 SOLOMOND AI 통합 시스템</h1>
                <p>4개 모듈을 하나의 API로 통합 관리</p>
                <p><strong>현재 시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="services">
                <div class="service">
                    <h3>📊 모듈1: 컨퍼런스 분석</h3>
                    <span class="status online">온라인</span>
                    <p>음성, 영상, 이미지 파일을 AI로 종합 분석</p>
                    <a href="http://localhost:8001/docs" class="btn" target="_blank">API 문서</a>
                    <a href="http://localhost:8501" class="btn" target="_blank">Streamlit UI</a>
                </div>
                
                <div class="service">
                    <h3>🕷️ 모듈2: 웹 크롤러</h3>
                    <span class="status offline">준비중</span>
                    <p>웹사이트 자동 수집 및 데이터 분석</p>
                    <a href="http://localhost:8002/docs" class="btn" target="_blank">API 문서</a>
                    <a href="http://localhost:8502" class="btn" target="_blank">Streamlit UI</a>
                </div>
                
                <div class="service">
                    <h3>💎 모듈3: 보석 분석</h3>
                    <span class="status offline">준비중</span>
                    <p>보석 이미지 분석 및 산지 추정</p>
                    <a href="http://localhost:8003/docs" class="btn" target="_blank">API 문서</a>
                    <a href="http://localhost:8503" class="btn" target="_blank">Streamlit UI</a>
                </div>
                
                <div class="service">
                    <h3>🏗️ 모듈4: 3D CAD</h3>
                    <span class="status offline">준비중</span>
                    <p>이미지를 3D CAD 모델로 자동 변환</p>
                    <a href="http://localhost:8004/docs" class="btn" target="_blank">API 문서</a>
                    <a href="http://localhost:8504" class="btn" target="_blank">Streamlit UI</a>
                </div>
            </div>
            
            <div class="api-section">
                <h2>🔌 통합 API 사용법</h2>
                <h3>1. 컨퍼런스 분석 API</h3>
                <pre><code>POST /api/module1/analyze
Content-Type: multipart/form-data
Files: 업로드할 파일들</code></pre>
                
                <h3>2. 서비스 상태 확인</h3>
                <pre><code>GET /health</code></pre>
                
                <p><a href="/docs" class="btn">📚 전체 API 문서 보기</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """모든 서비스 상태 체크"""
    status = {}
    
    for service_id, config in SERVICES.items():
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://{config['host']}:{config['port']}/health")
                status[service_id] = {
                    "name": config["name"],
                    "status": "online" if response.status_code == 200 else "error",
                    "port": config["port"]
                }
        except:
            status[service_id] = {
                "name": config["name"], 
                "status": "offline",
                "port": config["port"]
            }
    
    return {"gateway": "online", "services": status, "timestamp": datetime.now().isoformat()}

@app.post("/api/module1/analyze")
async def module1_analyze(files: List[UploadFile] = File(...)):
    """모듈1 컨퍼런스 분석 프록시"""
    try:
        # 모듈1 서비스로 파일 전달
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append(("files", (file.filename, content, file.content_type)))
            await file.seek(0)  # 파일 포인터 리셋
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "http://localhost:8001/analyze",
                files=files_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Module1 분석 실패: {str(e)}")

@app.get("/api/services")
async def list_services():
    """사용 가능한 서비스 목록"""
    return {
        "services": SERVICES,
        "gateway_info": {
            "version": "1.0.0",
            "description": "SOLOMOND AI 통합 API Gateway",
            "endpoints": [
                "GET /health - 서비스 상태 확인",
                "POST /api/module1/analyze - 컨퍼런스 분석",
                "GET /api/services - 서비스 목록"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 SOLOMOND AI API Gateway 시작...")
    print("📍 대시보드: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)