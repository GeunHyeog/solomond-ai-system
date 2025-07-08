# Phase 2: 배치 업로드 인터페이스 - 현장 친화적 웹 UI
# 주얼리 AI 플랫폼 - 전시회/세미나 현장 특화

from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import asyncio
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
import shutil
import os

# 기존 배치 처리 엔진 import
from core.batch_processing_engine import (
    BatchProcessingEngine, SessionConfig, FileItem, FileType, ProcessingStatus
)

app = FastAPI(title="주얼리 AI 플랫폼 - Phase 2", version="2.0.0")

# 전역 변수
batch_engine = BatchProcessingEngine()
active_sessions = {}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 현장 친화적 스타일
MODERN_UI_STYLE = """
<style>
:root {
    --primary-color: #2C5282;
    --secondary-color: #E53E3E;
    --success-color: #38A169;
    --warning-color: #D69E2E;
    --gold-color: #B7791F;
    --jewelry-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --surface-color: #FFFFFF;
    --text-primary: #2D3748;
    --text-secondary: #4A5568;
    --border-color: #E2E8F0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    background: var(--jewelry-gradient);
    color: white;
    padding: 30px 0;
    text-align: center;
    box-shadow: var(--shadow);
    margin-bottom: 30px;
}

.header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header .subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 300;
}

.session-creator {
    background: var(--surface-color);
    border-radius: 20px;
    padding: 30px;
    box-shadow: var(--shadow);
    margin-bottom: 30px;
    border: 1px solid var(--border-color);
}

.session-title {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 25px;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--primary-color);
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 25px;
}

.form-group {
    display: flex;
    flex-direction: column;
}

.form-group label {
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text-primary);
    font-size: 0.95rem;
}

.form-group input, .form-group select, .form-group textarea {
    padding: 12px 16px;
    border: 2px solid var(--border-color);
    border-radius: 12px;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: #FAFAFA;
}

.form-group input:focus, .form-group select:focus, .form-group textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    background: white;
    box-shadow: 0 0 0 3px rgba(44, 82, 130, 0.1);
}

.upload-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.upload-zone {
    background: var(--surface-color);
    border: 3px dashed var(--border-color);
    border-radius: 20px;
    padding: 30px 20px;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.upload-zone:hover {
    border-color: var(--primary-color);
    background: rgba(44, 82, 130, 0.02);
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}

.upload-zone.dragover {
    border-color: var(--success-color);
    background: rgba(56, 161, 105, 0.05);
}

.upload-icon {
    font-size: 3rem;
    margin-bottom: 15px;
    display: block;
}

.upload-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.upload-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 15px;
}

.upload-zone input[type="file"] {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
}

.upload-limit {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 10px;
    font-style: italic;
}

.file-preview {
    background: #F7FAFC;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.priority-selector {
    background: rgba(183, 121, 31, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
}

.priority-title {
    font-weight: 600;
    color: var(--gold-color);
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.priority-options {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.priority-chip {
    background: white;
    border: 2px solid var(--gold-color);
    border-radius: 20px;
    padding: 8px 16px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 0.9rem;
    user-select: none;
}

.priority-chip:hover {
    background: var(--gold-color);
    color: white;
}

.priority-chip.selected {
    background: var(--gold-color);
    color: white;
}

.action-buttons {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin-top: 30px;
    flex-wrap: wrap;
}

.btn {
    padding: 15px 30px;
    border: none;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-width: 160px;
    justify-content: center;
}

.btn-primary {
    background: var(--jewelry-gradient);
    color: white;
    box-shadow: var(--shadow);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px -8px rgba(44, 82, 130, 0.6);
}

.btn-secondary {
    background: white;
    color: var(--primary-color);
    border: 2px solid var(--primary-color);
}

.btn-secondary:hover {
    background: var(--primary-color);
    color: white;
}

.progress-container {
    background: var(--surface-color);
    border-radius: 20px;
    padding: 30px;
    box-shadow: var(--shadow);
    margin: 30px 0;
    display: none;
}

.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.progress-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--primary-color);
}

.progress-status {
    background: var(--success-color);
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.overall-progress {
    margin-bottom: 25px;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-weight: 500;
}

.progress-bar {
    width: 100%;
    height: 12px;
    background: var(--border-color);
    border-radius: 6px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--jewelry-gradient);
    border-radius: 6px;
    transition: width 0.3s ease;
    width: 0%;
}

.file-progress-list {
    display: grid;
    gap: 15px;
}

.file-progress-item {
    background: #F7FAFC;
    border-radius: 12px;
    padding: 15px;
    border-left: 4px solid var(--border-color);
}

.file-progress-item.processing {
    border-left-color: var(--warning-color);
}

.file-progress-item.completed {
    border-left-color: var(--success-color);
}

.file-progress-item.failed {
    border-left-color: var(--secondary-color);
}

.file-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.file-name {
    font-weight: 600;
    color: var(--text-primary);
}

.file-status {
    font-size: 0.85rem;
    padding: 3px 8px;
    border-radius: 12px;
    font-weight: 500;
}

.status-pending { background: #E2E8F0; color: #4A5568; }
.status-processing { background: #FED7D7; color: #C53030; }
.status-completed { background: #C6F6D5; color: #22543D; }
.status-failed { background: #FED7D7; color: #C53030; }

.results-container {
    background: var(--surface-color);
    border-radius: 20px;
    padding: 30px;
    box-shadow: var(--shadow);
    margin: 30px 0;
    display: none;
}

.results-header {
    text-align: center;
    margin-bottom: 30px;
}

.results-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
}

.confidence-score {
    font-size: 3rem;
    font-weight: 800;
    background: var(--jewelry-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 20px 0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.stat-card {
    background: #F7FAFC;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    border: 1px solid var(--border-color);
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    display: block;
}

.stat-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 5px;
}

.content-sections {
    display: grid;
    gap: 25px;
}

.content-section {
    background: #FAFAFA;
    border-radius: 15px;
    padding: 25px;
    border: 1px solid var(--border-color);
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.keywords-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.keyword-tag {
    background: var(--jewelry-gradient);
    color: white;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.jewelry-insights {
    background: rgba(183, 121, 31, 0.05);
    border: 2px solid var(--gold-color);
    border-radius: 15px;
    padding: 25px;
    margin-top: 20px;
}

.insights-title {
    color: var(--gold-color);
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.insight-item {
    background: white;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    border-left: 4px solid var(--gold-color);
}

.alert {
    padding: 15px 20px;
    border-radius: 12px;
    margin: 20px 0;
    font-weight: 500;
}

.alert-info {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #1E40AF;
}

.alert-success {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #166534;
}

.alert-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #92400E;
}

.mobile-optimized {
    @media (max-width: 768px) {
        .container { padding: 10px; }
        .header h1 { font-size: 1.8rem; }
        .upload-grid { grid-template-columns: 1fr; }
        .form-grid { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
        .action-buttons { flex-direction: column; }
        .btn { min-width: 100%; }
    }
}

/* 애니메이션 효과 */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeInUp 0.6s ease-out;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.processing-indicator {
    animation: pulse 2s infinite;
}

/* 터치 친화적 요소 */
@media (hover: none) {
    .upload-zone:hover {
        transform: none;
    }
    
    .btn:hover {
        transform: none;
    }
}
</style>
"""

BATCH_UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>주얼리 AI 플랫폼 - Phase 2 배치 분석</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    {style}
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>💎 주얼리 AI 플랫폼 Phase 2</h1>
            <p class="subtitle">다중 파일 배치 분석 시스템 - 전시회/세미나 현장 특화</p>
        </div>
    </div>

    <div class="container">
        {alert_html}
        
        <!-- 세션 생성 -->
        <div class="session-creator fade-in">
            <div class="session-title">
                <span>📅</span>
                <span>새 분석 세션 생성</span>
            </div>
            
            <form id="sessionForm" method="post" enctype="multipart/form-data">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="session_name">🎯 세션명</label>
                        <input type="text" id="session_name" name="session_name" 
                               placeholder="예: 2025 홍콩주얼리쇼 다이아몬드 세미나" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="event_type">🏢 이벤트 타입</label>
                        <select id="event_type" name="event_type" required>
                            <option value="">선택하세요</option>
                            <option value="주얼리 전시회">🏆 주얼리 전시회</option>
                            <option value="업계 세미나">📚 업계 세미나</option>
                            <option value="고객 상담">💬 고객 상담</option>
                            <option value="제품 교육">🎓 제품 교육</option>
                            <option value="시장 분석">📊 시장 분석</option>
                            <option value="비즈니스 미팅">🤝 비즈니스 미팅</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="topic">📋 주제/내용</label>
                        <input type="text" id="topic" name="topic" 
                               placeholder="예: 2025년 다이아몬드 시장 전망">
                    </div>
                    
                    <div class="form-group">
                        <label for="participants">👥 참석자</label>
                        <input type="text" id="participants" name="participants" 
                               placeholder="예: 전근혁 대표, 업계 전문가들">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="description">📝 추가 설명</label>
                    <textarea id="description" name="description" rows="3" 
                              placeholder="분석 목적이나 특별한 요구사항이 있으면 입력해주세요"></textarea>
                </div>
                
                <!-- 우선순위 선택 -->
                <div class="priority-selector">
                    <div class="priority-title">
                        <span>⭐</span>
                        <span>파일 우선순위 설정 (중요한 파일 타입 선택)</span>
                    </div>
                    <div class="priority-options">
                        <div class="priority-chip" data-type="audio">🎤 음성 파일</div>
                        <div class="priority-chip" data-type="video">🎥 영상 파일</div>
                        <div class="priority-chip" data-type="document">📄 문서 파일</div>
                        <div class="priority-chip" data-type="image">🖼️ 이미지 파일</div>
                    </div>
                    <input type="hidden" id="priority_types" name="priority_types" value="">
                </div>
                
                <!-- 파일 업로드 영역 -->
                <div class="upload-grid">
                    <div class="upload-zone" data-type="audio">
                        <div class="upload-icon">🎤</div>
                        <div class="upload-title">메인 녹음 파일</div>
                        <div class="upload-subtitle">MP3, WAV, M4A</div>
                        <input type="file" name="audio_files" multiple accept="audio/*">
                        <div class="upload-limit">최대 500MB, 고품질 권장</div>
                        <div class="file-preview" id="audio-preview"></div>
                    </div>
                    
                    <div class="upload-zone" data-type="video">
                        <div class="upload-icon">🎥</div>
                        <div class="upload-title">영상 파일</div>
                        <div class="upload-subtitle">MP4, MOV, AVI</div>
                        <input type="file" name="video_files" multiple accept="video/*">
                        <div class="upload-limit">최대 2GB, 음성 추출 후 분석</div>
                        <div class="file-preview" id="video-preview"></div>
                    </div>
                    
                    <div class="upload-zone" data-type="document">
                        <div class="upload-icon">📄</div>
                        <div class="upload-title">관련 문서</div>
                        <div class="upload-subtitle">PDF, DOCX, PPT</div>
                        <input type="file" name="document_files" multiple accept=".pdf,.doc,.docx,.ppt,.pptx">
                        <div class="upload-limit">최대 100MB, handout/자료</div>
                        <div class="file-preview" id="document-preview"></div>
                    </div>
                    
                    <div class="upload-zone" data-type="image">
                        <div class="upload-icon">🖼️</div>
                        <div class="upload-title">이미지 파일</div>
                        <div class="upload-subtitle">JPG, PNG, GIF</div>
                        <input type="file" name="image_files" multiple accept="image/*">
                        <div class="upload-limit">최대 50MB, OCR 텍스트 추출</div>
                        <div class="file-preview" id="image-preview"></div>
                    </div>
                </div>
                
                <div class="action-buttons">
                    <button type="button" class="btn btn-secondary" onclick="resetForm()">
                        🔄 초기화
                    </button>
                    <button type="submit" class="btn btn-primary" id="startAnalysis">
                        🚀 배치 분석 시작
                    </button>
                </div>
            </form>
        </div>
        
        <!-- 진행률 표시 -->
        <div id="progressContainer" class="progress-container">
            <div class="progress-header">
                <div class="progress-title">🔄 배치 분석 진행 중...</div>
                <div class="progress-status">진행 중</div>
            </div>
            
            <div class="overall-progress">
                <div class="progress-label">
                    <span>전체 진행률</span>
                    <span id="overallPercent">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="overallProgress"></div>
                </div>
            </div>
            
            <div id="fileProgressList" class="file-progress-list">
                <!-- 파일별 진행률이 여기에 동적으로 추가됩니다 -->
            </div>
        </div>
        
        <!-- 결과 표시 -->
        <div id="resultsContainer" class="results-container">
            <div class="results-header">
                <div class="results-title">
                    <span>🎉</span>
                    <span>배치 분석 완료!</span>
                    <span>✨</span>
                </div>
                <div class="confidence-score" id="confidenceScore">95.2%</div>
                <p>크로스 검증 신뢰도</p>
            </div>
            
            <div class="stats-grid" id="statsGrid">
                <!-- 통계가 동적으로 추가됩니다 -->
            </div>
            
            <div class="content-sections" id="contentSections">
                <!-- 분석 결과가 동적으로 추가됩니다 -->
            </div>
            
            <div class="jewelry-insights" id="jewelryInsights">
                <div class="insights-title">
                    <span>💎</span>
                    <span>주얼리 특화 인사이트</span>
                </div>
                <div id="insightsContent">
                    <!-- 인사이트가 동적으로 추가됩니다 -->
                </div>
            </div>
            
            <div class="action-buttons">
                <button type="button" class="btn btn-secondary" onclick="downloadResults()">
                    📥 결과 다운로드
                </button>
                <button type="button" class="btn btn-primary" onclick="startNewSession()">
                    ➕ 새 세션 시작
                </button>
            </div>
        </div>
    </div>

    <script>
        let currentSessionId = null;
        let selectedPriorities = [];
        
        // 우선순위 선택 기능
        document.querySelectorAll('.priority-chip').forEach(chip => {
            chip.addEventListener('click', function() {
                const type = this.dataset.type;
                if (selectedPriorities.includes(type)) {
                    selectedPriorities = selectedPriorities.filter(p => p !== type);
                    this.classList.remove('selected');
                } else {
                    selectedPriorities.push(type);
                    this.classList.add('selected');
                }
                document.getElementById('priority_types').value = selectedPriorities.join(',');
            });
        });
        
        // 드래그 앤 드롭 기능
        document.querySelectorAll('.upload-zone').forEach(zone => {
            const input = zone.querySelector('input[type="file"]');
            const preview = zone.querySelector('.file-preview');
            
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            
            zone.addEventListener('dragleave', () => {
                zone.classList.remove('dragover');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                input.files = e.dataTransfer.files;
                updateFilePreview(input, preview);
            });
            
            input.addEventListener('change', () => {
                updateFilePreview(input, preview);
            });
        });
        
        function updateFilePreview(input, preview) {
            const files = Array.from(input.files);
            if (files.length > 0) {
                const fileNames = files.map(f => `${f.name} (${(f.size/1024/1024).toFixed(1)}MB)`);
                preview.innerHTML = `선택된 파일: ${fileNames.join(', ')}`;
                preview.style.display = 'block';
            } else {
                preview.style.display = 'none';
            }
        }
        
        // 폼 제출 처리
        document.getElementById('sessionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const startButton = document.getElementById('startAnalysis');
            
            startButton.disabled = true;
            startButton.innerHTML = '🔄 세션 생성 중...';
            
            try {
                const response = await fetch('/batch-analysis', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentSessionId = result.session_id;
                    showProgress();
                    pollProgress();
                } else {
                    alert('오류: ' + result.error);
                    startButton.disabled = false;
                    startButton.innerHTML = '🚀 배치 분석 시작';
                }
            } catch (error) {
                alert('네트워크 오류: ' + error.message);
                startButton.disabled = false;
                startButton.innerHTML = '🚀 배치 분석 시작';
            }
        });
        
        function showProgress() {
            document.querySelector('.session-creator').style.display = 'none';
            document.getElementById('progressContainer').style.display = 'block';
        }
        
        async function pollProgress() {
            if (!currentSessionId) return;
            
            try {
                const response = await fetch(`/session-status/${currentSessionId}`);
                const status = await response.json();
                
                updateProgressUI(status);
                
                if (status.status === 'completed') {
                    const resultResponse = await fetch(`/session-result/${currentSessionId}`);
                    const result = await resultResponse.json();
                    showResults(result);
                } else if (status.status !== 'failed') {
                    setTimeout(pollProgress, 1000); // 1초마다 체크
                }
            } catch (error) {
                console.error('Progress polling error:', error);
                setTimeout(pollProgress, 2000); // 오류 시 2초 후 재시도
            }
        }
        
        function updateProgressUI(status) {
            const overallProgress = document.getElementById('overallProgress');
            const overallPercent = document.getElementById('overallPercent');
            const fileProgressList = document.getElementById('fileProgressList');
            
            // 전체 진행률 업데이트
            const progress = status.progress || 0;
            overallProgress.style.width = progress + '%';
            overallPercent.textContent = Math.round(progress) + '%';
            
            // 파일별 진행률 업데이트 (실제 구현에서는 서버에서 파일별 상태 전송)
            if (status.files) {
                fileProgressList.innerHTML = '';
                status.files.forEach(file => {
                    const item = createFileProgressItem(file);
                    fileProgressList.appendChild(item);
                });
            }
        }
        
        function createFileProgressItem(file) {
            const item = document.createElement('div');
            item.className = `file-progress-item ${file.status}`;
            
            item.innerHTML = `
                <div class="file-info">
                    <div class="file-name">${file.filename}</div>
                    <div class="file-status status-${file.status}">${getStatusText(file.status)}</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${file.progress || 0}%"></div>
                </div>
            `;
            
            return item;
        }
        
        function getStatusText(status) {
            const statusMap = {
                'pending': '대기 중',
                'processing': '처리 중',
                'completed': '완료',
                'failed': '실패'
            };
            return statusMap[status] || status;
        }
        
        function showResults(result) {
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'block';
            
            // 신뢰도 점수
            const confidenceScore = Math.round(result.cross_validation.confidence_score * 100);
            document.getElementById('confidenceScore').textContent = confidenceScore + '%';
            
            // 통계 그리드
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <span class="stat-number">${result.final_result.summary.total_files}</span>
                    <span class="stat-label">전체 파일</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">${result.final_result.summary.successfully_processed}</span>
                    <span class="stat-label">성공 처리</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">${Math.round(result.final_result.summary.average_confidence * 100)}%</span>
                    <span class="stat-label">평균 신뢰도</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">${result.final_result.summary.total_processing_time.toFixed(1)}s</span>
                    <span class="stat-label">처리 시간</span>
                </div>
            `;
            
            // 공통 키워드
            const contentSections = document.getElementById('contentSections');
            const keywordsHtml = result.cross_validation.common_keywords.map(keyword => 
                `<span class="keyword-tag">${keyword}</span>`
            ).join('');
            
            contentSections.innerHTML = `
                <div class="content-section">
                    <div class="section-title">
                        <span>🎯</span>
                        <span>핵심 내용 (${confidenceScore}% 신뢰도)</span>
                    </div>
                    <p>${result.cross_validation.verified_content.substring(0, 300)}...</p>
                </div>
                <div class="content-section">
                    <div class="section-title">
                        <span>🏷️</span>
                        <span>공통 키워드</span>
                    </div>
                    <div class="keywords-container">
                        ${keywordsHtml}
                    </div>
                </div>
            `;
            
            // 주얼리 인사이트
            const insights = result.final_result.jewelry_insights;
            const insightsContent = document.getElementById('insightsContent');
            let insightsHtml = '';
            
            if (insights.price_mentions && insights.price_mentions.length > 0) {
                insightsHtml += `
                    <div class="insight-item">
                        <strong>💰 가격 정보:</strong> ${insights.price_mentions.join(', ')}
                    </div>
                `;
            }
            
            if (insights.quality_grades && insights.quality_grades.length > 0) {
                insightsHtml += `
                    <div class="insight-item">
                        <strong>💎 품질 등급:</strong> ${insights.quality_grades.join(', ')}
                    </div>
                `;
            }
            
            if (insights.technical_terms && insights.technical_terms.length > 0) {
                insightsHtml += `
                    <div class="insight-item">
                        <strong>🔧 기술 용어:</strong> ${insights.technical_terms.join(', ')}
                    </div>
                `;
            }
            
            insightsContent.innerHTML = insightsHtml || '<p>추가적인 주얼리 인사이트를 분석 중입니다.</p>';
        }
        
        function resetForm() {
            document.getElementById('sessionForm').reset();
            selectedPriorities = [];
            document.querySelectorAll('.priority-chip').forEach(chip => {
                chip.classList.remove('selected');
            });
            document.querySelectorAll('.file-preview').forEach(preview => {
                preview.style.display = 'none';
            });
        }
        
        function downloadResults() {
            if (currentSessionId) {
                window.open(`/download-results/${currentSessionId}`, '_blank');
            }
        }
        
        function startNewSession() {
            currentSessionId = null;
            document.getElementById('resultsContainer').style.display = 'none';
            document.getElementById('progressContainer').style.display = 'none';
            document.querySelector('.session-creator').style.display = 'block';
            resetForm();
            window.scrollTo(0, 0);
        }
        
        // 페이지 로드 시 애니메이션
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelector('.session-creator').classList.add('fade-in');
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request, message: str = None, message_type: str = "info"):
    """메인 페이지 - 배치 업로드 인터페이스"""
    
    # 메시지 HTML 생성
    alert_html = ""
    if message:
        alert_class = f"alert-{message_type}"
        emoji = {"info": "💡", "success": "✅", "warning": "⚠️", "error": "❌"}.get(message_type, "ℹ️")
        alert_html = f'<div class="alert {alert_class}"><strong>{emoji} 알림</strong>: {message}</div>'
    else:
        alert_html = f'''
        <div class="alert alert-info">
            <strong>🚀 Phase 2 시작!</strong> 다중 파일 배치 분석으로 세미나/전시회 현장에서 여러 파일을 동시에 처리하여 
            95% 이상 신뢰도의 완벽한 분석 결과를 얻으세요.
        </div>
        '''
    
    return BATCH_UPLOAD_HTML.format(
        style=MODERN_UI_STYLE,
        alert_html=alert_html
    )

@app.post("/batch-analysis")
async def start_batch_analysis(
    background_tasks: BackgroundTasks,
    session_name: str = Form(...),
    event_type: str = Form(...),
    topic: str = Form(""),
    participants: str = Form(""),
    description: str = Form(""),
    priority_types: str = Form(""),
    audio_files: List[UploadFile] = File([]),
    video_files: List[UploadFile] = File([]),
    document_files: List[UploadFile] = File([]),
    image_files: List[UploadFile] = File([])
):
    """배치 분석 시작"""
    
    try:
        # 세션 ID 생성
        session_id = str(uuid.uuid4())
        
        # 우선순위 파일 타입 파싱
        priority_file_types = []
        if priority_types:
            type_mapping = {
                "audio": FileType.AUDIO,
                "video": FileType.VIDEO, 
                "document": FileType.DOCUMENT,
                "image": FileType.IMAGE
            }
            priority_file_types = [type_mapping[t] for t in priority_types.split(',') if t in type_mapping]
        
        # 세션 설정 생성
        config = SessionConfig(
            session_id=session_id,
            session_name=session_name,
            event_type=event_type,
            topic=topic,
            participants=participants.split(',') if participants else [],
            priority_file_types=priority_file_types
        )
        
        # 세션 생성
        await batch_engine.create_session(config)
        
        # 파일 처리 및 저장
        file_items = []
        
        # 각 파일 타입별 처리
        file_groups = [
            (audio_files, FileType.AUDIO),
            (video_files, FileType.VIDEO),
            (document_files, FileType.DOCUMENT),
            (image_files, FileType.IMAGE)
        ]
        
        for files, file_type in file_groups:
            for uploaded_file in files:
                if uploaded_file.filename:
                    # 파일 저장
                    file_path = UPLOAD_DIR / f"{session_id}_{uploaded_file.filename}"
                    
                    with open(file_path, "wb") as buffer:
                        content = await uploaded_file.read()
                        buffer.write(content)
                    
                    # FileItem 생성
                    file_item = FileItem(
                        file_id="",  # 엔진에서 자동 생성
                        filename=uploaded_file.filename,
                        file_type=file_type,
                        file_path=str(file_path),
                        size_mb=len(content) / 1024 / 1024
                    )
                    
                    file_items.append(file_item)
        
        if not file_items:
            return JSONResponse({
                "success": False,
                "error": "최소 1개 이상의 파일을 업로드해주세요."
            })
        
        # 세션에 파일 추가
        await batch_engine.add_files_to_session(session_id, file_items)
        
        # 세션 정보 저장
        active_sessions[session_id] = {
            "config": config,
            "files": file_items,
            "started_at": time.time(),
            "status": "created"
        }
        
        # 백그라운드에서 배치 처리 시작
        background_tasks.add_task(process_batch_in_background, session_id)
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": f"세션 '{session_name}' 생성 완료. {len(file_items)}개 파일 배치 처리를 시작합니다."
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"배치 분석 시작 중 오류 발생: {str(e)}"
        })

async def process_batch_in_background(session_id: str):
    """백그라운드에서 배치 처리 실행"""
    try:
        # 세션 상태 업데이트
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "processing"
        
        # 배치 처리 실행
        result = await batch_engine.start_batch_processing(session_id)
        
        # 결과 저장
        if session_id in active_sessions:
            active_sessions[session_id]["result"] = result
            active_sessions[session_id]["status"] = "completed"
            active_sessions[session_id]["completed_at"] = time.time()
            
    except Exception as e:
        # 오류 처리
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "failed"
            active_sessions[session_id]["error"] = str(e)

@app.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """세션 상태 조회"""
    
    if session_id not in active_sessions:
        return JSONResponse({"error": "세션을 찾을 수 없습니다"})
    
    session = active_sessions[session_id]
    engine_status = batch_engine.get_session_status(session_id)
    
    # 진행률 계산
    progress = 0
    if session["status"] == "created":
        progress = 5
    elif session["status"] == "processing":
        progress = 50  # 실제로는 더 정교한 계산 필요
    elif session["status"] == "completed":
        progress = 100
    elif session["status"] == "failed":
        progress = 0
    
    return JSONResponse({
        "session_id": session_id,
        "status": session["status"],
        "progress": progress,
        "files_count": len(session["files"]),
        "created_at": session["started_at"],
        "config": {
            "session_name": session["config"].session_name,
            "event_type": session["config"].event_type,
            "topic": session["config"].topic
        }
    })

@app.get("/session-result/{session_id}")
async def get_session_result(session_id: str):
    """세션 결과 조회"""
    
    if session_id not in active_sessions:
        return JSONResponse({"error": "세션을 찾을 수 없습니다"})
    
    session = active_sessions[session_id]
    
    if session["status"] != "completed":
        return JSONResponse({"error": "아직 처리가 완료되지 않았습니다"})
    
    if "result" not in session:
        return JSONResponse({"error": "결과를 찾을 수 없습니다"})
    
    return JSONResponse(session["result"])

@app.get("/download-results/{session_id}")
async def download_results(session_id: str):
    """결과 다운로드"""
    
    if session_id not in active_sessions:
        return JSONResponse({"error": "세션을 찾을 수 없습니다"})
    
    session = active_sessions[session_id]
    
    if "result" not in session:
        return JSONResponse({"error": "다운로드할 결과가 없습니다"})
    
    # JSON 파일로 결과 생성
    result_data = {
        "session_info": {
            "session_name": session["config"].session_name,
            "event_type": session["config"].event_type,
            "topic": session["config"].topic,
            "participants": session["config"].participants,
            "generated_at": datetime.now().isoformat()
        },
        "analysis_result": session["result"]
    }
    
    from fastapi.responses import StreamingResponse
    import io
    
    json_str = json.dumps(result_data, ensure_ascii=False, indent=2)
    json_bytes = json_str.encode('utf-8')
    
    return StreamingResponse(
        io.BytesIO(json_bytes),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=jewelry_analysis_{session_id[:8]}.json"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 주얼리 AI 플랫폼 Phase 2 - 배치 업로드 인터페이스 시작")
    print("📱 현장 친화적 UI/UX - 전시회/세미나 특화")
    print("🌐 서버 주소: http://localhost:8080")
    print("💎 다중 파일 배치 분석 시스템 준비 완료")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)