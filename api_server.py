"""
솔로몬드 AI 시스템 - 고용량 다중분석 API 서버
FastAPI 기반 REST API + WebSocket 실시간 모니터링

주요 엔드포인트:
- POST /api/v1/analyze/batch - 배치 분석
- POST /api/v1/analyze/streaming - 스트리밍 분석  
- GET /api/v1/status/{session_id} - 처리 상태 조회
- WebSocket /ws/progress/{session_id} - 실시간 진행률
- GET /api/v1/health - 시스템 상태
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import time
import uuid
from datetime import datetime
import tempfile
import os
from pathlib import Path
import logging

# 커스텀 모듈
try:
    from core.advanced_llm_summarizer_complete import EnhancedLLMSummarizer
    from core.large_file_streaming_engine import LargeFileStreamingEngine, StreamingProgress
    from core.multimodal_integrator import get_multimodal_integrator
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# FastAPI 앱 초기화
app = FastAPI(
    title="솔로몬드 AI - 고용량 다중분석 API",
    description="주얼리 업계 특화 멀티모달 AI 분석 시스템",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
active_sessions: Dict[str, Dict] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}
llm_summarizer: Optional[EnhancedLLMSummarizer] = None
streaming_engine: Optional[LargeFileStreamingEngine] = None

# 데이터 모델
class AnalysisRequest(BaseModel):
    session_name: str = Field(..., description="세션 이름")
    analysis_type: str = Field("comprehensive", description="분석 타입 (comprehensive, executive, technical, business)")
    max_memory_mb: int = Field(150, description="최대 메모리 사용량 (MB)")
    enable_streaming: bool = Field(True, description="스트리밍 처리 활성화")
    priority_sources: List[str] = Field(default_factory=list, description="우선순위 소스 타입")

class AnalysisResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    processing_started: bool = False
    estimated_time: Optional[float] = None

class StatusResponse(BaseModel):
    session_id: str
    status: str  # pending, processing, completed, error
    progress: float  # 0-100
    files_processed: int
    total_files: int
    processing_time: float
    estimated_remaining: Optional[float] = None
    memory_usage_mb: float
    current_stage: str
    error_message: Optional[str] = None

class ResultResponse(BaseModel):
    session_id: str
    success: bool
    final_summary: str
    quality_score: float
    processing_time: float
    files_processed: int
    recommendations: List[str]
    source_summaries: Dict[str, Any]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    modules_available: bool
    active_sessions: int
    system_info: Dict[str, Any]

# 시스템 초기화
@app.on_event("startup")
async def startup_event():
    """시스템 시작시 초기화"""
    global llm_summarizer, streaming_engine
    
    logging.info("솔로몬드 AI 시스템 시작 중...")
    
    if MODULES_AVAILABLE:
        try:
            llm_summarizer = EnhancedLLMSummarizer()
            streaming_engine = LargeFileStreamingEngine(max_memory_mb=200)
            logging.info("고급 모듈 초기화 완료")
        except Exception as e:
            logging.error(f"모듈 초기화 오류: {e}")
    else:
        logging.warning("고급 모듈 없음. 기본 모드로 실행")

@app.on_event("shutdown")
async def shutdown_event():
    """시스템 종료시 정리"""
    global streaming_engine
    
    logging.info("솔로몬드 AI 시스템 종료 중...")
    
    if streaming_engine:
        streaming_engine.cleanup()

# API 엔드포인트
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """시스템 상태 확인"""
    import psutil
    
    return HealthResponse(
        status="healthy" if MODULES_AVAILABLE else "limited",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        modules_available=MODULES_AVAILABLE,
        active_sessions=len(active_sessions),
        system_info={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "python_version": "3.11+"
        }
    )

@app.post("/api/v1/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(
    background_tasks: BackgroundTasks,
    request: AnalysisRequest,
    files: List[UploadFile] = File(...)
):
    """배치 분석 시작"""
    
    # 파일 수 제한 검증
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="파일 수는 50개를 초과할 수 없습니다")
    
    # 총 크기 검증
    total_size = sum(file.size for file in files if file.size)
    if total_size > 5 * 1024 * 1024 * 1024:  # 5GB
        raise HTTPException(status_code=400, detail="총 파일 크기는 5GB를 초과할 수 없습니다")
    
    # 세션 생성
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "session_name": request.session_name,
        "status": "pending",
        "progress": 0.0,
        "files": [],
        "created_at": time.time(),
        "analysis_type": request.analysis_type,
        "total_files": len(files),
        "files_processed": 0,
        "processing_time": 0.0,
        "memory_usage_mb": 0.0,
        "current_stage": "파일 업로드 중",
        "result": None,
        "error_message": None
    }
    
    active_sessions[session_id] = session_data
    
    # 파일 데이터 준비
    files_data = []
    temp_dir = tempfile.mkdtemp(prefix=f"solomond_{session_id}_")
    
    try:
        for i, file in enumerate(files):
            # 파일 저장
            file_path = os.path.join(temp_dir, file.filename)
            content = await file.read()
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            files_data.append({
                "filename": file.filename,
                "file_path": file_path,
                "size_mb": len(content) / (1024 * 1024),
                "content": content,
                "processed_text": ""  # STT/OCR 결과가 들어갈 자리
            })
            
            session_data["files"].append({
                "filename": file.filename,
                "size_mb": len(content) / (1024 * 1024),
                "status": "uploaded"
            })
        
        # 백그라운드에서 처리 시작
        background_tasks.add_task(
            process_batch_analysis,
            session_id,
            files_data,
            request,
            temp_dir
        )
        
        # 처리 시간 추정
        estimated_time = estimate_processing_time(files_data, request.analysis_type)
        
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            message=f"{len(files)}개 파일 분석이 시작되었습니다",
            processing_started=True,
            estimated_time=estimated_time
        )
        
    except Exception as e:
        if session_id in active_sessions:
            del active_sessions[session_id]
        raise HTTPException(status_code=500, detail=f"분석 시작 오류: {str(e)}")

@app.post("/api/v1/analyze/streaming", response_model=AnalysisResponse)
async def analyze_streaming(
    background_tasks: BackgroundTasks,
    request: AnalysisRequest,
    files: List[UploadFile] = File(...)
):
    """스트리밍 분석 시작"""
    
    if not MODULES_AVAILABLE or not streaming_engine:
        raise HTTPException(status_code=503, detail="스트리밍 처리가 지원되지 않습니다")
    
    # 배치 분석과 동일한 유효성 검사
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="파일 수는 50개를 초과할 수 없습니다")
    
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "session_name": request.session_name,
        "status": "pending",
        "progress": 0.0,
        "files": [],
        "created_at": time.time(),
        "analysis_type": request.analysis_type,
        "total_files": len(files),
        "files_processed": 0,
        "processing_time": 0.0,
        "memory_usage_mb": 0.0,
        "current_stage": "스트리밍 준비 중",
        "result": None,
        "error_message": None,
        "streaming_mode": True
    }
    
    active_sessions[session_id] = session_data
    
    # 파일 준비 및 스트리밍 처리 시작
    files_data = []
    temp_dir = tempfile.mkdtemp(prefix=f"solomond_stream_{session_id}_")
    
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            content = await file.read()
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            files_data.append({
                "filename": file.filename,
                "file_path": file_path,
                "size_mb": len(content) / (1024 * 1024),
                "file_type": detect_file_type(file.filename)
            })
        
        # 백그라운드에서 스트리밍 처리 시작
        background_tasks.add_task(
            process_streaming_analysis,
            session_id,
            files_data,
            request,
            temp_dir
        )
        
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            message=f"스트리밍 분석이 시작되었습니다 ({len(files)}개 파일)",
            processing_started=True,
            estimated_time=estimate_processing_time(files_data, request.analysis_type) * 0.8  # 스트리밍이 더 빠름
        )
        
    except Exception as e:
        if session_id in active_sessions:
            del active_sessions[session_id]
        raise HTTPException(status_code=500, detail=f"스트리밍 분석 시작 오류: {str(e)}")

@app.get("/api/v1/status/{session_id}", response_model=StatusResponse)
async def get_analysis_status(session_id: str):
    """분석 상태 조회"""
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    session = active_sessions[session_id]
    current_time = time.time()
    processing_time = current_time - session["created_at"]
    
    # 남은 시간 추정
    estimated_remaining = None
    if session["progress"] > 0 and session["status"] == "processing":
        estimated_total = processing_time / (session["progress"] / 100)
        estimated_remaining = max(0, estimated_total - processing_time)
    
    return StatusResponse(
        session_id=session_id,
        status=session["status"],
        progress=session["progress"],
        files_processed=session["files_processed"],
        total_files=session["total_files"],
        processing_time=processing_time,
        estimated_remaining=estimated_remaining,
        memory_usage_mb=session["memory_usage_mb"],
        current_stage=session["current_stage"],
        error_message=session.get("error_message")
    )

@app.get("/api/v1/result/{session_id}", response_model=ResultResponse)
async def get_analysis_result(session_id: str):
    """분석 결과 조회"""
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    session = active_sessions[session_id]
    
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="분석이 아직 완료되지 않았습니다")
    
    result = session.get("result", {})
    
    return ResultResponse(
        session_id=session_id,
        success=result.get("success", False),
        final_summary=result.get("hierarchical_summary", {}).get("final_summary", ""),
        quality_score=result.get("quality_assessment", {}).get("quality_score", 0),
        processing_time=session["processing_time"],
        files_processed=session["files_processed"],
        recommendations=result.get("recommendations", []),
        source_summaries=result.get("hierarchical_summary", {}).get("source_summaries", {}),
        metadata={
            "session_name": session.get("session_name", ""),
            "analysis_type": session.get("analysis_type", ""),
            "total_files": session["total_files"],
            "created_at": session["created_at"],
            "completed_at": session.get("completed_at", time.time())
        }
    )

@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    # WebSocket 연결 정리
    if session_id in websocket_connections:
        for ws in websocket_connections[session_id]:
            try:
                await ws.close()
            except:
                pass
        del websocket_connections[session_id]
    
    # 세션 데이터 삭제
    del active_sessions[session_id]
    
    return {"message": "세션이 삭제되었습니다", "session_id": session_id}

@app.get("/api/v1/sessions")
async def list_sessions():
    """활성 세션 목록 조회"""
    
    sessions_info = []
    for session_id, session in active_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "session_name": session.get("session_name", ""),
            "status": session["status"],
            "progress": session["progress"],
            "created_at": session["created_at"],
            "total_files": session["total_files"],
            "files_processed": session["files_processed"]
        })
    
    return {"sessions": sessions_info, "total": len(sessions_info)}

# WebSocket 엔드포인트
@app.websocket("/ws/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    """실시간 진행률 WebSocket"""
    
    await websocket.accept()
    
    # 연결 관리
    if session_id not in websocket_connections:
        websocket_connections[session_id] = []
    websocket_connections[session_id].append(websocket)
    
    try:
        while True:
            # 세션 상태 확인
            if session_id in active_sessions:
                session = active_sessions[session_id]
                
                progress_data = {
                    "session_id": session_id,
                    "status": session["status"],
                    "progress": session["progress"],
                    "current_stage": session["current_stage"],
                    "files_processed": session["files_processed"],
                    "total_files": session["total_files"],
                    "memory_usage_mb": session["memory_usage_mb"],
                    "timestamp": time.time()
                }
                
                await websocket.send_text(json.dumps(progress_data))
                
                # 완료되면 연결 종료
                if session["status"] in ["completed", "error"]:
                    break
            else:
                # 세션이 없으면 연결 종료
                await websocket.send_text(json.dumps({"error": "Session not found"}))
                break
            
            await asyncio.sleep(1)  # 1초마다 업데이트
            
    except WebSocketDisconnect:
        pass
    finally:
        # 연결 정리
        if session_id in websocket_connections:
            websocket_connections[session_id].remove(websocket)
            if not websocket_connections[session_id]:
                del websocket_connections[session_id]

# 백그라운드 처리 함수들
async def process_batch_analysis(session_id: str, files_data: List[Dict], request: AnalysisRequest, temp_dir: str):
    """배치 분석 백그라운드 처리"""
    
    session = active_sessions[session_id]
    session["status"] = "processing"
    session["current_stage"] = "파일 분석 중"
    
    start_time = time.time()
    
    try:
        # LLM 처리
        if llm_summarizer:
            # 진행률 업데이트
            session["progress"] = 10
            await notify_websocket_clients(session_id, session)
            
            result = await llm_summarizer.process_large_batch(files_data)
            
            session["progress"] = 90
            session["current_stage"] = "결과 생성 중"
            await notify_websocket_clients(session_id, session)
            
        else:
            # 모의 처리
            for i in range(len(files_data)):
                await asyncio.sleep(0.5)
                session["files_processed"] = i + 1
                session["progress"] = (i + 1) / len(files_data) * 90
                await notify_websocket_clients(session_id, session)
            
            result = create_mock_result(files_data)
        
        # 완료 처리
        session["status"] = "completed"
        session["progress"] = 100
        session["current_stage"] = "완료"
        session["result"] = result
        session["processing_time"] = time.time() - start_time
        session["completed_at"] = time.time()
        session["files_processed"] = len(files_data)
        
        await notify_websocket_clients(session_id, session)
        
    except Exception as e:
        session["status"] = "error"
        session["error_message"] = str(e)
        session["processing_time"] = time.time() - start_time
        
        await notify_websocket_clients(session_id, session)
        
        logging.error(f"배치 분석 오류 (세션 {session_id}): {e}")
    
    finally:
        # 임시 파일 정리
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

async def process_streaming_analysis(session_id: str, files_data: List[Dict], request: AnalysisRequest, temp_dir: str):
    """스트리밍 분석 백그라운드 처리"""
    
    session = active_sessions[session_id]
    session["status"] = "processing"
    session["current_stage"] = "스트리밍 처리 중"
    
    start_time = time.time()
    
    try:
        processed_results = []
        
        for i, file_data in enumerate(files_data):
            session["current_stage"] = f"파일 처리 중: {file_data['filename']}"
            
            if streaming_engine:
                # 실제 스트리밍 처리
                result = await streaming_engine.process_large_file(
                    file_data["file_path"],
                    file_data["file_type"]
                )
                processed_results.append(result)
            else:
                # 모의 스트리밍 처리
                await asyncio.sleep(0.3)
                processed_results.append({"success": True, "filename": file_data["filename"]})
            
            # 진행률 업데이트
            session["files_processed"] = i + 1
            session["progress"] = (i + 1) / len(files_data) * 80
            await notify_websocket_clients(session_id, session)
        
        # 결과 통합
        session["current_stage"] = "결과 통합 중"
        session["progress"] = 85
        await notify_websocket_clients(session_id, session)
        
        # LLM 요약
        if llm_summarizer:
            # 처리된 결과를 LLM에 전달할 형태로 변환
            llm_input = []
            for result in processed_results:
                if result.get("success"):
                    llm_input.append({
                        "filename": result.get("filename", "unknown"),
                        "processed_text": result.get("result", {}).get("processed_text", ""),
                        "size_mb": 1.0  # 예시
                    })
            
            final_result = await llm_summarizer.process_large_batch(llm_input)
        else:
            final_result = create_mock_result(files_data)
        
        # 완료 처리
        session["status"] = "completed"
        session["progress"] = 100
        session["current_stage"] = "완료"
        session["result"] = final_result
        session["processing_time"] = time.time() - start_time
        session["completed_at"] = time.time()
        
        await notify_websocket_clients(session_id, session)
        
    except Exception as e:
        session["status"] = "error"
        session["error_message"] = str(e)
        session["processing_time"] = time.time() - start_time
        
        await notify_websocket_clients(session_id, session)
        
        logging.error(f"스트리밍 분석 오류 (세션 {session_id}): {e}")
    
    finally:
        # 임시 파일 정리
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

async def notify_websocket_clients(session_id: str, session_data: Dict):
    """WebSocket 클라이언트들에게 업데이트 전송"""
    
    if session_id in websocket_connections:
        progress_data = {
            "session_id": session_id,
            "status": session_data["status"],
            "progress": session_data["progress"],
            "current_stage": session_data["current_stage"],
            "files_processed": session_data["files_processed"],
            "total_files": session_data["total_files"],
            "timestamp": time.time()
        }
        
        # 연결된 모든 클라이언트에게 전송
        disconnected_clients = []
        for websocket in websocket_connections[session_id]:
            try:
                await websocket.send_text(json.dumps(progress_data))
            except:
                disconnected_clients.append(websocket)
        
        # 끊어진 연결 정리
        for ws in disconnected_clients:
            websocket_connections[session_id].remove(ws)

# 유틸리티 함수들
def detect_file_type(filename: str) -> str:
    """파일 타입 감지"""
    ext = Path(filename).suffix.lower()
    
    if ext in ['.mp3', '.wav', '.m4a', '.aac']:
        return 'audio'
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return 'video'
    elif ext in ['.pdf', '.docx', '.txt']:
        return 'document'
    elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
        return 'image'
    else:
        return 'unknown'

def estimate_processing_time(files_data: List[Dict], analysis_type: str) -> float:
    """처리 시간 추정"""
    base_time_per_mb = 2.0  # 초/MB
    
    total_mb = sum(f.get("size_mb", 0) for f in files_data)
    base_time = total_mb * base_time_per_mb
    
    # 분석 타입별 가중치
    type_multiplier = {
        "comprehensive": 1.5,
        "executive": 1.0,
        "technical": 1.2,
        "business": 1.1
    }
    
    multiplier = type_multiplier.get(analysis_type, 1.0)
    return base_time * multiplier

def create_mock_result(files_data: List[Dict]) -> Dict:
    """모의 결과 생성"""
    return {
        "success": True,
        "session_id": str(uuid.uuid4()),
        "processing_time": 15.5,
        "files_processed": len(files_data),
        "chunks_processed": len(files_data) * 3,
        "hierarchical_summary": {
            "final_summary": "주얼리 시장 분석 결과, 다이아몬드 가격 상승세와 GIA 인증서 중요성 증대가 확인되었습니다.",
            "source_summaries": {
                "audio": {"summary": "음성 분석 요약", "chunk_count": 5},
                "documents": {"summary": "문서 분석 요약", "chunk_count": 7}
            }
        },
        "quality_assessment": {
            "quality_score": 87.5,
            "coverage_ratio": 0.82,
            "compression_ratio": 0.15
        },
        "recommendations": [
            "우수한 품질의 요약이 생성되었습니다.",
            "주얼리 전문 용어 커버리지가 양호합니다."
        ]
    }

# 예외 처리
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"API 오류: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "내부 서버 오류가 발생했습니다", "status_code": 500}
    )

# 개발용 메인 함수
if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 솔로몬드 AI API 서버 시작")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🌐 WebSocket: ws://localhost:8000/ws/progress/{session_id}")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
