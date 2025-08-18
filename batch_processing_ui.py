#!/usr/bin/env python3
"""
Phase 2: 배치 처리 통합 웹 UI
주얼리 AI 플랫폼 - 다중 파일 현장 처리 시스템

Features:
- 다중 파일 드래그 앤 드롭 업로드
- 실시간 처리 진행률 모니터링  
- 크로스 검증 결과 시각화
- 주얼리 현장 특화 세션 설정
- 모바일 최적화 (전시회/세미나 현장)
- WebSocket 실시간 연동
- 결과 내보내기 기능
"""

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import uuid
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Core 모듈 임포트
from core.batch_processing_engine import BatchProcessingEngine, SessionConfig, FileItem, FileType
from core.cross_validation_visualizer import CrossValidationVisualizer, CrossValidationResult, ValidationMetrics

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="💎 솔로몬드 배치 처리 시스템",
    description="주얼리 업계 특화 다중 파일 분석 플랫폼",
    version="2.0.0"
)

# 전역 변수
batch_engine = BatchProcessingEngine()
visualizer = CrossValidationVisualizer()
active_connections: Dict[str, WebSocket] = {}
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# 메인 웹 인터페이스
@app.get("/", response_class=HTMLResponse)
async def get_batch_processing_ui():
    """배치 처리 메인 UI"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>💎 솔로몬드 배치 처리 시스템</title>
        <style>
            {get_main_ui_styles()}
        </style>
    </head>
    <body>
        <div class="app-container">
            {get_header_section()}
            {get_session_setup_section()}
            {get_file_upload_section()}
            {get_processing_monitor_section()}
            {get_results_section()}
        </div>
        
        <script>
            {get_main_ui_javascript()}
        </script>
    </body>
    </html>
    """
    
    return html_content

def get_main_ui_styles():
    """메인 UI 스타일 - 대형 함수 (리팩토링 고려 대상 - 677줄)"""
    return """
    :root {
        --primary: #2C5282;
        --secondary: #E53E3E;
        --success: #38A169;
        --warning: #D69E2E;
        --gold: #B7791F;
        --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --surface: #FFFFFF;
        --background: #F7FAFC;
        --text-primary: #2D3748;
        --text-secondary: #4A5568;
        --border: #E2E8F0;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--background);
        color: var(--text-primary);
        line-height: 1.6;
        overflow-x: hidden;
    }
    
    .app-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* 헤더 섹션 */
    .header-section {
        background: var(--gradient);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        animation: float 20s infinite linear;
        pointer-events: none;
    }
    
    @keyframes float {
        0% { transform: translateX(-50px) translateY(-50px); }
        100% { transform: translateX(50px) translateY(50px); }
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 20px;
    }
    
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    
    .feature-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* 섹션 공통 스타일 */
    .section {
        background: var(--surface);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* 세션 설정 섹션 */
    .session-setup {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
    }
    
    .form-group {
        display: flex;
        flex-direction: column;
    }
    
    .form-label {
        font-weight: 500;
        margin-bottom: 8px;
        color: var(--text-primary);
    }
    
    .form-input, .form-select, .form-textarea {
        padding: 12px 16px;
        border: 2px solid var(--border);
        border-radius: var(--border-radius);
        font-size: 1rem;
        transition: all 0.3s ease;
        background: var(--surface);
    }
    
    .form-input:focus, .form-select:focus, .form-textarea:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(44, 82, 130, 0.1);
    }
    
    .form-textarea {
        min-height: 80px;
        resize: vertical;
    }
    
    .participants-input {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    
    .participant-tag {
        background: var(--primary);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .remove-participant {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        font-size: 1.2rem;
        padding: 0;
        line-height: 1;
    }
    
    .create-session-btn {
        background: var(--gradient);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: var(--border-radius);
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 20px;
    }
    
    .create-session-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.3);
    }
    
    /* 파일 업로드 섹션 */
    .upload-area {
        border: 3px dashed var(--border);
        border-radius: var(--border-radius);
        padding: 60px 40px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        background: linear-gradient(45deg, transparent 25%, rgba(44, 82, 130, 0.02) 25%, rgba(44, 82, 130, 0.02) 75%, transparent 75%);
        background-size: 20px 20px;
    }
    
    .upload-area:hover, .upload-area.dragover {
        border-color: var(--primary);
        background-color: rgba(44, 82, 130, 0.05);
        transform: scale(1.02);
    }
    
    .upload-icon {
        font-size: 4rem;
        color: var(--primary);
        margin-bottom: 20px;
        display: block;
    }
    
    .upload-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 10px;
    }
    
    .upload-subtext {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .file-list {
        margin-top: 30px;
        display: grid;
        gap: 15px;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px;
        background: var(--background);
        border-radius: var(--border-radius);
        border: 1px solid var(--border);
    }
    
    .file-info {
        display: flex;
        align-items: center;
        gap: 15px;
        flex: 1;
    }
    
    .file-icon {
        font-size: 2rem;
        width: 50px;
        text-align: center;
    }
    
    .file-details h4 {
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .file-details p {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .file-actions {
        display: flex;
        gap: 10px;
    }
    
    .btn {
        padding: 8px 16px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .btn-sm {
        padding: 6px 12px;
        font-size: 0.8rem;
    }
    
    .btn-primary {
        background: var(--primary);
        color: white;
    }
    
    .btn-danger {
        background: var(--secondary);
        color: white;
    }
    
    .btn-success {
        background: var(--success);
        color: white;
    }
    
    .btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .start-processing-btn {
        background: var(--success);
        color: white;
        border: none;
        padding: 20px 40px;
        border-radius: var(--border-radius);
        font-size: 1.1rem;
        font-weight: 700;
        cursor: pointer;
        margin-top: 30px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .start-processing-btn:hover {
        background: #2F855A;
        transform: translateY(-2px);
    }
    
    .start-processing-btn:disabled {
        background: #CBD5E0;
        cursor: not-allowed;
        transform: none;
    }
    
    /* 처리 모니터 섹션 */
    .processing-monitor {
        display: none;
    }
    
    .processing-monitor.active {
        display: block;
    }
    
    .progress-overview {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .progress-stat {
        text-align: center;
        padding: 20px;
        background: var(--background);
        border-radius: var(--border-radius);
        border: 1px solid var(--border);
    }
    
    .progress-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary);
        display: block;
    }
    
    .progress-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 5px;
    }
    
    .overall-progress {
        margin-bottom: 30px;
    }
    
    .progress-bar {
        width: 100%;
        height: 20px;
        background: var(--border);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--gradient);
        width: 0%;
        transition: width 0.5s ease;
        position: relative;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .file-progress-list {
        display: grid;
        gap: 15px;
    }
    
    .file-progress-item {
        padding: 20px;
        background: var(--background);
        border-radius: var(--border-radius);
        border: 1px solid var(--border);
    }
    
    .file-progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .file-status {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-pending {
        background: rgba(214, 158, 46, 0.1);
        color: var(--warning);
    }
    
    .status-processing {
        background: rgba(44, 82, 130, 0.1);
        color: var(--primary);
    }
    
    .status-completed {
        background: rgba(56, 161, 105, 0.1);
        color: var(--success);
    }
    
    .status-failed {
        background: rgba(229, 62, 62, 0.1);
        color: var(--secondary);
    }
    
    /* 결과 섹션 */
    .results-section {
        display: none;
    }
    
    .results-section.active {
        display: block;
    }
    
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
    }
    
    .export-buttons {
        display: flex;
        gap: 10px;
    }
    
    .cross-validation-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .validation-metric {
        text-align: center;
        padding: 25px;
        background: var(--background);
        border-radius: var(--border-radius);
        border: 1px solid var(--border);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .metric-high { color: var(--success); }
    .metric-medium { color: var(--warning); }
    .metric-low { color: var(--secondary); }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-secondary);
    }
    
    .visualization-container {
        background: var(--surface);
        border-radius: var(--border-radius);
        padding: 30px;
        border: 1px solid var(--border);
        margin-bottom: 30px;
    }
    
    .jewelry-insights {
        background: linear-gradient(135deg, rgba(183, 121, 31, 0.05) 0%, rgba(183, 121, 31, 0.1) 100%);
        border: 2px solid var(--gold);
        border-radius: var(--border-radius);
        padding: 30px;
    }
    
    .insights-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--gold);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .insights-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
    }
    
    .insight-category {
        background: white;
        padding: 20px;
        border-radius: var(--border-radius);
        border-left: 4px solid var(--gold);
    }
    
    .insight-category h4 {
        color: var(--gold);
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .insight-list {
        list-style: none;
    }
    
    .insight-item {
        padding: 8px 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.9rem;
    }
    
    .insight-item:last-child {
        border-bottom: none;
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .app-container {
            padding: 10px;
        }
        
        .header-title {
            font-size: 1.8rem;
        }
        
        .session-setup {
            grid-template-columns: 1fr;
        }
        
        .progress-overview {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .cross-validation-summary {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .insights-grid {
            grid-template-columns: 1fr;
        }
        
        .upload-area {
            padding: 40px 20px;
        }
        
        .upload-icon {
            font-size: 3rem;
        }
    }
    
    /* 로딩 애니메이션 */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* 성공/에러 메시지 */
    .message {
        padding: 15px 20px;
        border-radius: var(--border-radius);
        margin-bottom: 20px;
        font-weight: 500;
    }
    
    .message-success {
        background: rgba(56, 161, 105, 0.1);
        color: var(--success);
        border: 1px solid rgba(56, 161, 105, 0.3);
    }
    
    .message-error {
        background: rgba(229, 62, 62, 0.1);
        color: var(--secondary);
        border: 1px solid rgba(229, 62, 62, 0.3);
    }
    
    .message-warning {
        background: rgba(214, 158, 46, 0.1);
        color: var(--warning);
        border: 1px solid rgba(214, 158, 46, 0.3);
    }
    
    /* 특수 효과 */
    .sparkle {
        position: relative;
        overflow: hidden;
    }
    
    .sparkle::before {
        content: '✨';
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 1.2rem;
        animation: sparkle 3s infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 0; transform: scale(0.5); }
        50% { opacity: 1; transform: scale(1); }
    }
    """

def get_header_section():
    """헤더 섹션 HTML"""
    return """
    <div class="header-section">
        <div class="header-content">
            <h1 class="header-title">💎 솔로몬드 배치 처리 시스템</h1>
            <p class="header-subtitle">주얼리 업계 특화 다중 파일 분석 플랫폼</p>
            <div class="feature-badges">
                <div class="feature-badge">🔄 다중 파일 동시 처리</div>
                <div class="feature-badge">✅ 크로스 검증</div>
                <div class="feature-badge">📊 실시간 모니터링</div>
                <div class="feature-badge">💎 주얼리 특화</div>
                <div class="feature-badge">📱 모바일 최적화</div>
            </div>
        </div>
    </div>
    """

def get_session_setup_section():
    """세션 설정 섹션 HTML"""
    return """
    <div class="section session-setup-section">
        <h2 class="section-title">
            <span>⚙️</span>
            <span>세션 설정</span>
        </h2>
        
        <div id="sessionMessage"></div>
        
        <div class="session-setup">
            <div class="form-group">
                <label class="form-label" for="sessionName">세션 이름</label>
                <input type="text" id="sessionName" class="form-input" 
                       placeholder="예: 2025 홍콩주얼리쇼 다이아몬드 세미나">
            </div>
            
            <div class="form-group">
                <label class="form-label" for="eventType">이벤트 유형</label>
                <select id="eventType" class="form-select">
                    <option value="">이벤트 유형 선택</option>
                    <option value="주얼리 전시회">주얼리 전시회</option>
                    <option value="업계 세미나">업계 세미나</option>
                    <option value="고객 상담">고객 상담</option>
                    <option value="제품 교육">제품 교육</option>
                    <option value="시장 분석">시장 분석 회의</option>
                    <option value="거래 협상">거래 협상</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label" for="topic">주제</label>
                <input type="text" id="topic" class="form-input" 
                       placeholder="예: 2025년 다이아몬드 시장 전망">
            </div>
            
            <div class="form-group">
                <label class="form-label" for="expectedDuration">예상 소요시간 (분)</label>
                <input type="number" id="expectedDuration" class="form-input" 
                       placeholder="60" min="1" max="480">
            </div>
            
            <div class="form-group">
                <label class="form-label" for="participantInput">참석자</label>
                <input type="text" id="participantInput" class="form-input" 
                       placeholder="참석자 이름 입력 후 Enter">
                <div id="participantsList" class="participants-input"></div>
            </div>
            
            <div class="form-group">
                <label class="form-label" for="priorityTypes">우선 파일 유형</label>
                <select id="priorityTypes" class="form-select" multiple>
                    <option value="audio">음성 파일</option>
                    <option value="video">영상 파일</option>
                    <option value="document">문서 파일</option>
                    <option value="image">이미지 파일</option>
                </select>
            </div>
        </div>
        
        <button id="createSessionBtn" class="create-session-btn">
            📝 세션 생성
        </button>
    </div>
    """

def get_file_upload_section():
    """파일 업로드 섹션 HTML"""
    return """
    <div class="section file-upload-section" style="display: none;">
        <h2 class="section-title">
            <span>📁</span>
            <span>파일 업로드</span>
        </h2>
        
        <div id="uploadMessage"></div>
        
        <div class="upload-area" id="uploadArea">
            <span class="upload-icon">☁️</span>
            <div class="upload-text">파일을 드래그하여 놓거나 클릭하여 선택</div>
            <div class="upload-subtext">
                지원 형식: MP3, WAV, M4A, MP4, AVI, PDF, DOC, JPG, PNG<br>
                최대 파일 크기: 500MB | 최대 파일 개수: 20개
            </div>
            <input type="file" id="fileInput" multiple accept=".mp3,.wav,.m4a,.mp4,.avi,.pdf,.doc,.docx,.jpg,.jpeg,.png" style="display: none;">
        </div>
        
        <div id="fileList" class="file-list"></div>
        
        <button id="startProcessingBtn" class="start-processing-btn" disabled>
            🚀 배치 처리 시작
        </button>
    </div>
    """

def get_processing_monitor_section():
    """처리 모니터 섹션 HTML"""
    return """
    <div class="section processing-monitor" id="processingMonitor">
        <h2 class="section-title">
            <span>⚡</span>
            <span>실시간 처리 모니터링</span>
        </h2>
        
        <div class="progress-overview">
            <div class="progress-stat">
                <span id="totalFiles" class="progress-value">0</span>
                <div class="progress-label">전체 파일</div>
            </div>
            <div class="progress-stat">
                <span id="completedFiles" class="progress-value">0</span>
                <div class="progress-label">완료된 파일</div>
            </div>
            <div class="progress-stat">
                <span id="overallConfidence" class="progress-value">0%</span>
                <div class="progress-label">전체 신뢰도</div>
            </div>
            <div class="progress-stat">
                <span id="processingTime" class="progress-value">0s</span>
                <div class="progress-label">처리 시간</div>
            </div>
        </div>
        
        <div class="overall-progress">
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill"></div>
            </div>
            <div id="progressText" style="text-align: center; margin-top: 10px;">
                처리 준비 중...
            </div>
        </div>
        
        <div id="fileProgressList" class="file-progress-list"></div>
    </div>
    """

def get_results_section():
    """결과 섹션 HTML"""
    return """
    <div class="section results-section" id="resultsSection">
        <div class="results-header">
            <h2 class="section-title">
                <span>📊</span>
                <span>분석 결과</span>
            </h2>
            <div class="export-buttons">
                <button class="btn btn-primary" onclick="exportResults('pdf')">
                    📄 PDF 내보내기
                </button>
                <button class="btn btn-primary" onclick="exportResults('excel')">
                    📊 Excel 내보내기
                </button>
                <button class="btn btn-primary" onclick="exportResults('json')">
                    💾 JSON 내보내기
                </button>
            </div>
        </div>
        
        <div class="cross-validation-summary">
            <div class="validation-metric">
                <div id="finalConfidence" class="metric-value metric-high">0%</div>
                <div class="metric-label">전체 신뢰도</div>
            </div>
            <div class="validation-metric">
                <div id="contentOverlap" class="metric-value metric-medium">0%</div>
                <div class="metric-label">내용 중복도</div>
            </div>
            <div class="validation-metric">
                <div id="qualityImprovement" class="metric-value metric-high">0%</div>
                <div class="metric-label">품질 개선</div>
            </div>
            <div class="validation-metric">
                <div id="jeweryTermsTotal" class="metric-value metric-high">0</div>
                <div class="metric-label">주얼리 전문용어</div>
            </div>
        </div>
        
        <div class="visualization-container">
            <h3 style="margin-bottom: 20px;">📈 크로스 검증 시각화</h3>
            <div id="visualizationContent">
                <!-- 크로스 검증 시각화가 여기에 로드됩니다 -->
            </div>
        </div>
        
        <div class="jewelry-insights">
            <div class="insights-title">
                <span>💎</span>
                <span>주얼리 특화 인사이트</span>
            </div>
            <div id="jewelryInsights" class="insights-grid">
                <!-- 주얼리 인사이트가 여기에 로드됩니다 -->
            </div>
        </div>
    </div>
    """

def get_main_ui_javascript():
    """메인 UI JavaScript - 대형 함수 (리팩토링 고려 대상 - 534줄)"""
    return """
    // 전역 변수
    let currentSessionId = null;
    let uploadedFiles = [];
    let participants = [];
    let socket = null;
    let processingStartTime = null;
    
    // 페이지 로드시 초기화
    document.addEventListener('DOMContentLoaded', function() {
        initializeUI();
        setupWebSocket();
    });
    
    function initializeUI() {
        // 세션 생성 버튼
        document.getElementById('createSessionBtn').addEventListener('click', createSession);
        
        // 참석자 입력
        document.getElementById('participantInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                addParticipant(this.value.trim());
                this.value = '';
            }
        });
        
        // 파일 업로드 영역
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('drop', handleDrop);
        fileInput.addEventListener('change', handleFileSelect);
        
        // 처리 시작 버튼
        document.getElementById('startProcessingBtn').addEventListener('click', startBatchProcessing);
    }
    
    function setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = function(event) {
            console.log('WebSocket 연결 성공');
        };
        
        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket 연결 종료');
            // 재연결 시도
            setTimeout(setupWebSocket, 3000);
        };
        
        socket.onerror = function(error) {
            console.error('WebSocket 오류:', error);
        };
    }
    
    function handleWebSocketMessage(data) {
        switch(data.type) {
            case 'processing_progress':
                updateProcessingProgress(data.data);
                break;
            case 'file_completed':
                updateFileProgress(data.data);
                break;
            case 'processing_completed':
                handleProcessingCompleted(data.data);
                break;
            case 'error':
                showMessage(data.message, 'error');
                break;
        }
    }
    
    async function createSession() {
        const sessionName = document.getElementById('sessionName').value.trim();
        const eventType = document.getElementById('eventType').value;
        const topic = document.getElementById('topic').value.trim();
        const expectedDuration = parseInt(document.getElementById('expectedDuration').value) || 60;
        const priorityTypes = Array.from(document.getElementById('priorityTypes').selectedOptions).map(opt => opt.value);
        
        if (!sessionName || !eventType || !topic) {
            showMessage('모든 필드를 입력해주세요.', 'error', 'sessionMessage');
            return;
        }
        
        const sessionConfig = {
            session_name: sessionName,
            event_type: eventType,
            topic: topic,
            participants: participants,
            expected_duration: expectedDuration,
            priority_file_types: priorityTypes
        };
        
        try {
            const response = await fetch('/api/create-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sessionConfig)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                currentSessionId = result.session_id;
                showMessage(`세션 생성 완료: ${result.session_id}`, 'success', 'sessionMessage');
                
                // 파일 업로드 섹션 표시
                document.querySelector('.file-upload-section').style.display = 'block';
                
                // WebSocket에 세션 ID 전송
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({
                        type: 'join_session',
                        session_id: currentSessionId
                    }));
                }
            } else {
                showMessage(result.detail || '세션 생성에 실패했습니다.', 'error', 'sessionMessage');
            }
        } catch (error) {
            console.error('세션 생성 오류:', error);
            showMessage('세션 생성 중 오류가 발생했습니다.', 'error', 'sessionMessage');
        }
    }
    
    function addParticipant(name) {
        if (name && !participants.includes(name)) {
            participants.push(name);
            updateParticipantsList();
        }
    }
    
    function removeParticipant(name) {
        participants = participants.filter(p => p !== name);
        updateParticipantsList();
    }
    
    function updateParticipantsList() {
        const container = document.getElementById('participantsList');
        container.innerHTML = participants.map(name => `
            <div class="participant-tag">
                <span>${name}</span>
                <button class="remove-participant" onclick="removeParticipant('${name}')">×</button>
            </div>
        `).join('');
    }
    
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('dragover');
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        addFiles(files);
    }
    
    function handleFileSelect(e) {
        const files = Array.from(e.target.files);
        addFiles(files);
    }
    
    function addFiles(files) {
        const validFiles = files.filter(file => {
            const validTypes = [
                'audio/mpeg', 'audio/wav', 'audio/m4a',
                'video/mp4', 'video/avi', 'video/quicktime',
                'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'image/jpeg', 'image/png', 'image/gif'
            ];
            const maxSize = 500 * 1024 * 1024; // 500MB
            
            return validTypes.includes(file.type) && file.size <= maxSize;
        });
        
        if (validFiles.length !== files.length) {
            showMessage(`${files.length - validFiles.length}개 파일이 지원되지 않는 형식이거나 크기가 너무 큽니다.`, 'warning', 'uploadMessage');
        }
        
        if (uploadedFiles.length + validFiles.length > 20) {
            showMessage('최대 20개 파일까지 업로드할 수 있습니다.', 'error', 'uploadMessage');
            return;
        }
        
        validFiles.forEach(file => {
            const fileId = generateFileId();
            uploadedFiles.push({
                id: fileId,
                file: file,
                name: file.name,
                size: file.size,
                type: getFileType(file.type),
                status: 'pending'
            });
        });
        
        updateFileList();
        updateStartButton();
    }
    
    function getFileType(mimeType) {
        if (mimeType.startsWith('audio/')) return 'audio';
        if (mimeType.startsWith('video/')) return 'video';
        if (mimeType.startsWith('image/')) return 'image';
        return 'document';
    }
    
    function generateFileId() {
        return Math.random().toString(36).substr(2, 9);
    }
    
    function updateFileList() {
        const container = document.getElementById('fileList');
        
        if (uploadedFiles.length === 0) {
            container.innerHTML = '';
            return;
        }
        
        container.innerHTML = uploadedFiles.map(file => {
            const icon = getFileIcon(file.type);
            const sizeText = formatFileSize(file.size);
            
            return `
                <div class="file-item" data-file-id="${file.id}">
                    <div class="file-info">
                        <div class="file-icon">${icon}</div>
                        <div class="file-details">
                            <h4>${file.name}</h4>
                            <p>${file.type.toUpperCase()} • ${sizeText}</p>
                        </div>
                    </div>
                    <div class="file-actions">
                        <button class="btn btn-sm btn-danger" onclick="removeFile('${file.id}')">
                            🗑️ 제거
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    }
    
    function getFileIcon(type) {
        const icons = {
            'audio': '🎵',
            'video': '🎬',
            'image': '🖼️',
            'document': '📄'
        };
        return icons[type] || '📁';
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    function removeFile(fileId) {
        uploadedFiles = uploadedFiles.filter(file => file.id !== fileId);
        updateFileList();
        updateStartButton();
    }
    
    function updateStartButton() {
        const button = document.getElementById('startProcessingBtn');
        button.disabled = uploadedFiles.length === 0 || !currentSessionId;
    }
    
    async function startBatchProcessing() {
        if (!currentSessionId || uploadedFiles.length === 0) {
            showMessage('세션 생성 및 파일 업로드를 먼저 완료해주세요.', 'error', 'uploadMessage');
            return;
        }
        
        // UI 전환
        document.querySelector('.file-upload-section').style.display = 'none';
        document.getElementById('processingMonitor').classList.add('active');
        
        // 처리 시작 시간 기록
        processingStartTime = Date.now();
        
        // FormData 생성
        const formData = new FormData();
        formData.append('session_id', currentSessionId);
        
        uploadedFiles.forEach(fileData => {
            formData.append('files', fileData.file);
        });
        
        try {
            // 파일 업로드 및 처리 시작
            const response = await fetch('/api/start-batch-processing', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                showMessage('배치 처리가 시작되었습니다.', 'success', 'uploadMessage');
                initializeProcessingMonitor();
            } else {
                showMessage(result.detail || '배치 처리 시작에 실패했습니다.', 'error', 'uploadMessage');
            }
        } catch (error) {
            console.error('배치 처리 시작 오류:', error);
            showMessage('배치 처리 시작 중 오류가 발생했습니다.', 'error', 'uploadMessage');
        }
    }
    
    function initializeProcessingMonitor() {
        // 초기 통계 설정
        document.getElementById('totalFiles').textContent = uploadedFiles.length;
        document.getElementById('completedFiles').textContent = '0';
        document.getElementById('overallConfidence').textContent = '0%';
        document.getElementById('processingTime').textContent = '0s';
        
        // 파일 진행률 목록 초기화
        const container = document.getElementById('fileProgressList');
        container.innerHTML = uploadedFiles.map(file => `
            <div class="file-progress-item" data-file-id="${file.id}">
                <div class="file-progress-header">
                    <span>${file.name}</span>
                    <span class="file-status status-pending">대기중</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        `).join('');
        
        // 처리 시간 업데이트 시작
        startTimeUpdater();
    }
    
    function startTimeUpdater() {
        setInterval(() => {
            if (processingStartTime) {
                const elapsed = Math.floor((Date.now() - processingStartTime) / 1000);
                document.getElementById('processingTime').textContent = `${elapsed}s`;
            }
        }, 1000);
    }
    
    function updateProcessingProgress(data) {
        // 전체 진행률 업데이트
        document.getElementById('progressFill').style.width = `${data.progress}%`;
        document.getElementById('progressText').textContent = `진행률: ${data.progress.toFixed(1)}%`;
        
        // 완료된 파일 수 업데이트
        document.getElementById('completedFiles').textContent = data.completed_files;
        
        // 전체 신뢰도 업데이트
        document.getElementById('overallConfidence').textContent = `${(data.overall_confidence * 100).toFixed(1)}%`;
    }
    
    function updateFileProgress(data) {
        const fileElement = document.querySelector(`[data-file-id="${data.file_id}"]`);
        if (fileElement) {
            const statusElement = fileElement.querySelector('.file-status');
            const progressFill = fileElement.querySelector('.progress-fill');
            
            // 상태 업데이트
            statusElement.className = `file-status status-${data.status}`;
            statusElement.textContent = getStatusText(data.status);
            
            // 진행률 업데이트
            const progress = data.status === 'completed' ? 100 : data.status === 'processing' ? 50 : 0;
            progressFill.style.width = `${progress}%`;
        }
    }
    
    function getStatusText(status) {
        const statusTexts = {
            'pending': '대기중',
            'processing': '처리중',
            'completed': '완료',
            'failed': '실패'
        };
        return statusTexts[status] || status;
    }
    
    function handleProcessingCompleted(data) {
        // 처리 완료 UI로 전환
        document.getElementById('processingMonitor').classList.remove('active');
        document.getElementById('resultsSection').classList.add('active');
        
        // 최종 결과 표시
        displayFinalResults(data);
        
        showMessage('모든 파일 처리가 완료되었습니다!', 'success');
    }
    
    function displayFinalResults(data) {
        // 최종 지표 업데이트
        document.getElementById('finalConfidence').textContent = `${(data.cross_validation.confidence_score * 100).toFixed(1)}%`;
        document.getElementById('contentOverlap').textContent = `${data.cross_validation.content_overlap_percentage.toFixed(1)}%`;
        document.getElementById('qualityImprovement').textContent = `${(data.quality_improvement * 100).toFixed(1)}%`;
        document.getElementById('jeweryTermsTotal').textContent = data.jewelry_insights.total_terms || 0;
        
        // 시각화 로드
        loadVisualization(data);
        
        // 주얼리 인사이트 표시
        displayJewelryInsights(data.jewelry_insights);
    }
    
    async function loadVisualization(data) {
        try {
            const response = await fetch('/api/generate-visualization', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    session_id: currentSessionId,
                    results: data 
                })
            });
            
            const htmlContent = await response.text();
            document.getElementById('visualizationContent').innerHTML = htmlContent;
        } catch (error) {
            console.error('시각화 로드 오류:', error);
            document.getElementById('visualizationContent').innerHTML = '<p>시각화를 로드할 수 없습니다.</p>';
        }
    }
    
    function displayJewelryInsights(insights) {
        const container = document.getElementById('jewelryInsights');
        
        const categories = [
            { key: 'price_mentions', title: '💰 가격 정보', icon: '💰' },
            { key: 'quality_grades', title: '⭐ 품질 등급', icon: '⭐' },
            { key: 'technical_terms', title: '🔧 기술 용어', icon: '🔧' },
            { key: 'market_trends', title: '📈 시장 트렌드', icon: '📈' }
        ];
        
        container.innerHTML = categories.map(category => {
            const items = insights[category.key] || [];
            const itemsHtml = items.length > 0 
                ? items.map(item => `<li class="insight-item">${item}</li>`).join('')
                : '<li class="insight-item">정보가 발견되지 않았습니다.</li>';
            
            return `
                <div class="insight-category">
                    <h4>${category.icon} ${category.title}</h4>
                    <ul class="insight-list">${itemsHtml}</ul>
                </div>
            `;
        }).join('');
    }
    
    async function exportResults(format) {
        if (!currentSessionId) {
            showMessage('결과를 내보낼 세션이 없습니다.', 'error');
            return;
        }
        
        try {
            const response = await fetch(`/api/export-results/${format}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `batch_results_${currentSessionId}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                showMessage(`${format.toUpperCase()} 파일이 다운로드되었습니다.`, 'success');
            } else {
                showMessage('내보내기에 실패했습니다.', 'error');
            }
        } catch (error) {
            console.error('내보내기 오류:', error);
            showMessage('내보내기 중 오류가 발생했습니다.', 'error');
        }
    }
    
    function showMessage(message, type = 'info', containerId = null) {
        const messageHtml = `
            <div class="message message-${type}">
                ${message}
            </div>
        `;
        
        const container = containerId 
            ? document.getElementById(containerId)
            : document.body;
        
        if (containerId) {
            container.innerHTML = messageHtml;
        } else {
            container.insertAdjacentHTML('afterbegin', messageHtml);
            
            // 5초 후 메시지 제거
            setTimeout(() => {
                const messageElement = container.querySelector('.message');
                if (messageElement) {
                    messageElement.remove();
                }
            }, 5000);
        }
    }
    """

# API 엔드포인트들
@app.post("/api/create-session")
async def create_session_api(session_data: dict):
    """세션 생성 API"""
    try:
        session_config = SessionConfig(
            session_id=str(uuid.uuid4()),
            session_name=session_data["session_name"],
            event_type=session_data["event_type"],
            topic=session_data["topic"],
            participants=session_data.get("participants", []),
            expected_duration=session_data.get("expected_duration", 60),
            priority_file_types=[FileType(ft) for ft in session_data.get("priority_file_types", [])]
        )
        
        session_id = await batch_engine.create_session(session_config)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "세션이 성공적으로 생성되었습니다."
        }
        
    except Exception as e:
        logger.error(f"세션 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-batch-processing")
async def start_batch_processing_api(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """배치 처리 시작 API"""
    try:
        # 파일 저장 및 FileItem 생성
        file_items = []
        for file in files:
            # 파일 저장
            file_path = upload_dir / f"{session_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # FileItem 생성
            file_type = get_file_type_from_filename(file.filename)
            file_item = FileItem(
                file_id="",
                filename=file.filename,
                file_type=file_type,
                file_path=str(file_path),
                size_mb=len(content) / (1024 * 1024)
            )
            file_items.append(file_item)
        
        # 세션에 파일 추가
        await batch_engine.add_files_to_session(session_id, file_items)
        
        # 배치 처리 시작 (비동기)
        asyncio.create_task(process_batch_with_updates(session_id))
        
        return {
            "success": True,
            "message": "배치 처리가 시작되었습니다.",
            "files_count": len(file_items)
        }
        
    except Exception as e:
        logger.error(f"배치 처리 시작 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_with_updates(session_id: str):
    """WebSocket으로 실시간 업데이트를 보내며 배치 처리"""
    try:
        # 배치 처리 시작
        result = await batch_engine.start_batch_processing(session_id)
        
        # 완료 메시지 전송
        await broadcast_to_session(session_id, {
            "type": "processing_completed",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"배치 처리 오류: {e}")
        await broadcast_to_session(session_id, {
            "type": "error",
            "message": f"처리 중 오류가 발생했습니다: {str(e)}"
        })

def get_file_type_from_filename(filename: str) -> FileType:
    """파일명에서 파일 타입 추출"""
    ext = filename.lower().split('.')[-1]
    
    if ext in ['mp3', 'wav', 'm4a', 'aac']:
        return FileType.AUDIO
    elif ext in ['mp4', 'avi', 'mov', 'mkv']:
        return FileType.VIDEO
    elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
        return FileType.IMAGE
    else:
        return FileType.DOCUMENT

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 연결 관리"""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "join_session":
                session_id = message["session_id"]
                # 세션별 연결 관리는 여기서 구현
                await websocket.send_text(json.dumps({
                    "type": "session_joined",
                    "session_id": session_id
                }))
                
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
    finally:
        if connection_id in active_connections:
            del active_connections[connection_id]

async def broadcast_to_session(session_id: str, message: dict):
    """특정 세션의 모든 연결에 메시지 브로드캐스트"""
    for connection in active_connections.values():
        try:
            await connection.send_text(json.dumps(message))
        except:
            pass

@app.post("/api/generate-visualization")
async def generate_visualization_api(data: dict):
    """시각화 생성 API"""
    try:
        # 모의 ValidationMetrics 생성 (실제로는 batch_engine 결과 사용)
        sample_metrics = [
            ValidationMetrics(
                file_id=f"file_{i}",
                filename=f"file_{i}.mp3",
                file_type="audio",
                content_completeness=85 + i * 5,
                keyword_accuracy=80 + i * 3,
                audio_quality=75 + i * 4,
                time_accuracy=90 + i * 2,
                confidence_score=0.8 + i * 0.05,
                cross_match_score=0.75 + i * 0.05,
                jewelry_terms_found=10 + i * 2,
                price_accuracy=85.0,
                technical_terms=5 + i
            ) for i in range(3)
        ]
        
        # CrossValidationResult 생성
        validation_result = CrossValidationResult(
            session_id=data["session_id"],
            session_name="배치 처리 결과",
            overall_confidence=0.89,
            content_overlap=0.87,
            quality_improvement=0.23,
            file_metrics=sample_metrics,
            cross_matrix=[[1.0, 0.89, 0.86], [0.89, 1.0, 0.84], [0.86, 0.84, 1.0]],
            common_keywords=["다이아몬드", "4C", "GIA", "캐럿"],
            unique_keywords={},
            jewelry_insights={
                "price_mentions": ["$8,500", "$25,000"],
                "quality_grades": ["4C", "GIA"],
                "technical_terms": ["프린세스 컷"]
            },
            validation_timestamp=datetime.now().isoformat(),
            processing_time=45.2
        )
        
        # 시각화 HTML 생성
        html_content = visualizer.generate_visualization_html(validation_result)
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"시각화 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """세션 상태 조회 API"""
    try:
        status = batch_engine.get_session_status(session_id)
        return status
    except Exception as e:
        logger.error(f"세션 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 솔로몬드 배치 처리 시스템 시작")
    logger.info("💎 주얼리 업계 특화 다중 파일 분석 플랫폼")
    logger.info("📊 Phase 2: 현장 친화적 배치 처리 UI")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )
