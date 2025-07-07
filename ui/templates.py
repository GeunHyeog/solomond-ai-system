"""
솔로몬드 AI 시스템 - UI 템플릿
HTML 템플릿 및 UI 컴포넌트 관리 (Phase 3.3 AI 고도화)
"""

def get_main_template() -> str:
    """
    메인 페이지 HTML 템플릿 반환 (Phase 3.3 AI 고도화 추가)
    
    기존 기능 + 화자 구분 기능 추가
    """
    
    return """
<!DOCTYPE html>
<html>
<head>
    <title>솔로몬드 AI 시스템 v3.3 - AI 고도화</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #333; 
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-style: italic;
        }
        .upload-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .language-selector, .analysis-options {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 2px solid #e9ecef;
        }
        input[type="file"] { 
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
            width: 100%;
            margin: 10px 0;
        }
        select {
            padding: 12px 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
            width: 100%;
            margin: 10px 0;
            cursor: pointer;
        }
        button { 
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 18px;
            margin: 10px 5px;
            transition: all 0.3s ease;
            min-width: 150px;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }
        .detect-btn {
            background: linear-gradient(45deg, #17a2b8, #138496);
        }
        .speaker-btn {
            background: linear-gradient(45deg, #6f42c1, #5a32a3);
        }
        .button-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
        }
        .result { 
            margin-top: 30px; 
            padding: 25px; 
            background: #f8f9fa; 
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
        }
        .speaker-timeline {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 1px solid #dee2e6;
        }
        .speaker-segment {
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            border-radius: 6px;
            border-left: 4px solid;
        }
        .speaker-a { border-left-color: #007bff; background: #e7f3ff; }
        .speaker-b { border-left-color: #28a745; background: #e8f5e8; }
        .speaker-c { border-left-color: #ffc107; background: #fff9e6; }
        .speaker-d { border-left-color: #dc3545; background: #ffe6e6; }
        .speaker-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
        }
        .status { 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            font-weight: 500;
        }
        .success { 
            background: #d4edda; 
            color: #155724; 
            border: 1px solid #c3e6cb;
        }
        .error { 
            background: #f8d7da; 
            color: #721c24; 
            border: 1px solid #f5c6cb;
        }
        .info { 
            background: #d1ecf1; 
            color: #0c5460; 
            border: 1px solid #bee5eb;
        }
        .warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }
        .version-badge {
            background: #6f42c1;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .feature-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .feature-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        .feature-card h4 {
            margin: 0 0 10px 0;
            color: #495057;
        }
        .format-support {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .format-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
        }
        .format-card h4 {
            margin-top: 0;
            color: #495057;
        }
        .format-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .format-tag {
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 500;
        }
        .video-tag {
            background: #ffeaa7;
            color: #d63031;
        }
        .audio-tag {
            background: #a8e6cf;
            color: #00b894;
        }
        .language-tag {
            background: #ddd6fe;
            color: #7c3aed;
        }
        .ai-tag {
            background: #fecaca;
            color: #dc2626;
        }
        .confidence-bar {
            background: #e9ecef;
            height: 6px;
            border-radius: 3px;
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-fill {
            background: linear-gradient(45deg, #28a745, #20c997);
            height: 100%;
            transition: width 0.3s ease;
        }
        .time-badge {
            background: #6c757d;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 솔로몬드 AI 시스템</h1>
        <div class="subtitle">
            실제 내용을 읽고 분석하는 차세대 AI 플랫폼
            <span class="version-badge">v3.3 AI 고도화</span>
        </div>
        
        <div class="status info">
            <strong>🎯 시스템 상태:</strong><br>
            ✅ Python 3.13 완전 호환<br>
            ✅ 모듈화 구조 완성<br>
            ✅ Whisper STT 엔진<br>
            🎥 동영상 지원 완료<br>
            🌍 다국어 지원 완료<br>
            🎭 <strong>새기능: AI 고도화 추가!</strong>
        </div>
        
        <div class="format-support">
            <div class="format-card">
                <h4>🎵 지원하는 음성 형식</h4>
                <div class="format-list">
                    <span class="format-tag audio-tag">MP3</span>
                    <span class="format-tag audio-tag">WAV</span>
                    <span class="format-tag audio-tag">M4A</span>
                </div>
            </div>
            <div class="format-card">
                <h4>🎥 지원하는 동영상 형식</h4>
                <div class="format-list">
                    <span class="format-tag video-tag">MP4</span>
                    <span class="format-tag video-tag">AVI</span>
                    <span class="format-tag video-tag">MOV</span>
                    <span class="format-tag video-tag">MKV</span>
                    <span class="format-tag video-tag">WEBM</span>
                    <span class="format-tag video-tag">FLV</span>
                </div>
            </div>
            <div class="format-card">
                <h4>🌍 지원하는 언어</h4>
                <div class="format-list">
                    <span class="format-tag language-tag">🌐 자동감지</span>
                    <span class="format-tag language-tag">🇰🇷 한국어</span>
                    <span class="format-tag language-tag">🇺🇸 English</span>
                    <span class="format-tag language-tag">🇨🇳 中文</span>
                    <span class="format-tag language-tag">🇯🇵 日本語</span>
                </div>
            </div>
            <div class="format-card">
                <h4>🎭 AI 고도화 기능</h4>
                <div class="format-list">
                    <span class="format-tag ai-tag">화자 구분</span>
                    <span class="format-tag ai-tag">감정 분석</span>
                    <span class="format-tag ai-tag">자동 요약</span>
                    <span class="format-tag ai-tag">특화 분석</span>
                </div>
            </div>
        </div>
        
        <div class="feature-list">
            <div class="feature-card">
                <h4>🎤 고급 STT</h4>
                <p>OpenAI Whisper 기반 정확한 음성 인식</p>
            </div>
            <div class="feature-card">
                <h4>🎬 동영상 처리</h4>
                <p>FFmpeg 기반 음성 추출</p>
            </div>
            <div class="feature-card">
                <h4>🌐 다국어</h4>
                <p>11개 언어 + 자동 감지</p>
            </div>
            <div class="feature-card">
                <h4>🎭 화자 구분</h4>
                <p>AI 기반 다중 화자 분석</p>
            </div>
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="language-selector">
                <h3>🌍 언어 선택</h3>
                <select id="languageSelect" name="language">
                    <option value="auto">🌐 자동 감지 (권장)</option>
                    <option value="ko">🇰🇷 한국어</option>
                    <option value="en">🇺🇸 English</option>
                    <option value="zh">🇨🇳 中文 (중국어)</option>
                    <option value="ja">🇯🇵 日本語 (일본어)</option>
                    <option value="es">🇪🇸 Español (스페인어)</option>
                    <option value="fr">🇫🇷 Français (프랑스어)</option>
                    <option value="de">🇩🇪 Deutsch (독일어)</option>
                    <option value="ru">🇷🇺 Русский (러시아어)</option>
                    <option value="pt">🇵🇹 Português (포르투갈어)</option>
                    <option value="it">🇮🇹 Italiano (이탈리아어)</option>
                </select>
            </div>
            
            <div class="analysis-options">
                <h3>🎭 분석 옵션</h3>
                <div style="margin: 15px 0;">
                    <label>
                        <input type="checkbox" id="enableSpeakerAnalysis" checked>
                        화자 구분 분석 포함 (회의/상담 분석에 유용)
                    </label>
                </div>
                <div style="margin: 15px 0;">
                    <label>
                        <input type="checkbox" id="useAdvancedAI">
                        고급 AI 분석 사용 (더 정확하지만 처리 시간 증가)
                    </label>
                </div>
            </div>
            
            <div class="upload-section">
                <h3>📁 파일 업로드</h3>
                <p>음성 파일 또는 동영상 파일을 선택하세요</p>
                <input type="file" 
                       name="media_file" 
                       accept=".mp3,.wav,.m4a,.mp4,.avi,.mov,.mkv,.webm,.flv" 
                       required>
                
                <div class="button-group">
                    <button type="button" class="detect-btn" onclick="detectLanguageOnly()">
                        🔍 언어만 감지
                    </button>
                    <button type="button" class="speaker-btn" onclick="analyzeSpeakersOnly()">
                        🎭 화자만 구분
                    </button>
                    <button type="submit">🚀 전체 분석</button>
                </div>
                
                <div style="margin-top: 15px; font-size: 14px; color: #666;">
                    <strong>💡 사용 팁:</strong><br>
                    • <strong>전체 분석</strong>: STT + 언어감지 + 화자구분 (종합 분석)<br>
                    • <strong>화자만 구분</strong>: 누가 언제 말했는지만 확인<br>
                    • <strong>언어만 감지</strong>: 파일의 언어만 확인<br>
                    • 최대 파일 크기: 100MB 권장
                </div>
            </div>
        </form>
        
        <div id="ffmpegWarning" class="status warning" style="display:none;">
            <strong>⚠️ FFmpeg 설치 필요</strong><br>
            동영상 처리를 위해 FFmpeg가 필요합니다. 설치 방법을 확인하세요.
        </div>
        
        <div id="result" class="result" style="display:none;">
            <h3>📄 분석 결과:</h3>
            <div id="resultContent"></div>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666; font-size: 14px;">
            개발자: 전근혁 (솔로몬드 대표) | 
            <a href="/docs" target="_blank">API 문서</a> | 
            <a href="/test" target="_blank">시스템 테스트</a> |
            <a href="/video_support" target="_blank">동영상 지원 상태</a> |
            <a href="/language_support" target="_blank">언어 지원 상태</a> |
            <a href="/speaker_support" target="_blank">화자 구분 상태</a>
        </div>
    </div>

    <script>
        // 시스템 상태 확인
        Promise.all([
            fetch('/api/video_support').then(r => r.json()).catch(() => ({})),
            fetch('/api/language_support').then(r => r.json()).catch(() => ({})),
            fetch('/api/speaker_support').then(r => r.json()).catch(() => ({}))
        ]).then(([videoData, langData, speakerData]) => {
            if (videoData.video_support && !videoData.video_support.ffmpeg_available) {
                document.getElementById('ffmpegWarning').style.display = 'block';
            }
            
            console.log('🌍 지원 언어:', langData.total_languages || '확인 중...');
            console.log('🎭 화자 구분:', speakerData.speaker_diarization?.pyannote_available ? 'AI 모델' : '기본 알고리즘');
        }).catch(err => console.warn('상태 확인 실패:', err));

        // 화자 구분 전용 분석
        async function analyzeSpeakersOnly() {
            const fileInput = document.querySelector('input[type="file"]');
            
            if (!fileInput.files[0]) {
                alert('먼저 파일을 선택해주세요');
                return;
            }
            
            const formData = new FormData();
            const useAdvanced = document.getElementById('useAdvancedAI').checked;
            formData.append('audio_file', fileInput.files[0]);
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultDiv.style.display = 'block';
            resultContent.innerHTML = '<div class="status info">🎭 화자 구분 분석 중... 잠시만 기다려주세요.</div>';
            
            try {
                const response = await fetch(`/api/analyze_speakers?use_advanced=${useAdvanced}`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displaySpeakerResults(result);
                } else {
                    resultContent.innerHTML = `
                        <div class="status error">
                            <strong>❌ 화자 구분 실패</strong><br>
                            오류: ${result.error}
                        </div>
                    `;
                }
            } catch (error) {
                resultContent.innerHTML = `
                    <div class="status error">
                        <strong>❌ 네트워크 오류</strong><br>
                        ${error.message}
                    </div>
                `;
            }
        }

        // 화자 구분 결과 표시
        function displaySpeakerResults(result) {
            const resultContent = document.getElementById('resultContent');
            
            // 화자별 통계
            let statsHtml = '<div class="speaker-stats">';
            Object.entries(result.speaker_statistics).forEach(([speaker, stats]) => {
                statsHtml += `
                    <div class="stat-card">
                        <h4>${speaker}</h4>
                        <div><strong>${stats.total_duration}초</strong></div>
                        <div>${stats.percentage}%</div>
                        <div>신뢰도: ${Math.round(stats.avg_confidence * 100)}%</div>
                    </div>
                `;
            });
            statsHtml += '</div>';
            
            // 타임라인
            let timelineHtml = '<div class="speaker-timeline"><h4>🕐 화자별 타임라인</h4>';
            result.segments.forEach((segment, index) => {
                const speakerClass = `speaker-${String.fromCharCode(97 + (index % 4))}`;
                timelineHtml += `
                    <div class="speaker-segment ${speakerClass}">
                        <span class="time-badge">${segment.start}s - ${segment.end}s</span>
                        <strong>${segment.speaker}</strong>
                        <span style="margin-left: auto;">신뢰도: ${Math.round(segment.confidence * 100)}%</span>
                    </div>
                `;
            });
            timelineHtml += '</div>';
            
            resultContent.innerHTML = `
                <div class="status success">
                    <strong>✅ 화자 구분 완료!</strong>
                </div>
                <h4>📊 화자별 통계</h4>
                ${statsHtml}
                ${timelineHtml}
                <h4>📋 분석 정보</h4>
                <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <li><strong>총 길이:</strong> ${result.total_duration}초</li>
                    <li><strong>감지된 화자:</strong> ${result.num_speakers}명</li>
                    <li><strong>분석 방법:</strong> ${result.analysis_info.algorithm}</li>
                    <li><strong>처리 시간:</strong> ${result.processing_time}초</li>
                </div>
            `;
        }

        // 언어만 감지하는 함수 (기존)
        async function detectLanguageOnly() {
            const fileInput = document.querySelector('input[type="file"]');
            
            if (!fileInput.files[0]) {
                alert('먼저 파일을 선택해주세요');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio_file', fileInput.files[0]);
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultDiv.style.display = 'block';
            resultContent.innerHTML = '<div class="status info">🔍 언어 감지 중... 잠시만 기다려주세요.</div>';
            
            try {
                const response = await fetch('/api/detect_language', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const confidence = Math.round(result.confidence * 100);
                    const langInfo = result.language_info || {};
                    
                    let probsHtml = '';
                    if (result.all_probabilities) {
                        probsHtml = '<h4>🎯 상위 언어 후보:</h4><div class="language-grid">';
                        Object.entries(result.all_probabilities).slice(0, 5).forEach(([lang, prob]) => {
                            const percent = Math.round(prob * 100);
                            probsHtml += `
                                <div class="language-item">
                                    <span>${lang}</span>
                                    <span>${percent}%</span>
                                </div>
                            `;
                        });
                        probsHtml += '</div>';
                    }
                    
                    resultContent.innerHTML = `
                        <div class="status success">
                            <strong>✅ 언어 감지 완료!</strong>
                        </div>
                        <h4>🌍 감지된 언어:</h4>
                        <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0;">
                            <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
                                ${langInfo.flag || '🌐'} ${langInfo.name || result.detected_language}
                            </div>
                            <div>신뢰도: ${confidence}%</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                        ${probsHtml}
                    `;
                    
                    // 감지된 언어로 드롭다운 설정
                    const languageSelect = document.getElementById('languageSelect');
                    if (result.detected_language && languageSelect) {
                        languageSelect.value = result.detected_language;
                    }
                    
                } else {
                    resultContent.innerHTML = `
                        <div class="status error">
                            <strong>❌ 언어 감지 실패</strong><br>
                            오류: ${result.error}
                        </div>
                    `;
                }
            } catch (error) {
                resultContent.innerHTML = `
                    <div class="status error">
                        <strong>❌ 네트워크 오류</strong><br>
                        ${error.message}
                    </div>
                `;
            }
        }

        // 전체 분석 함수 (STT + 화자 구분)
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.querySelector('input[type="file"]');
            const languageSelect = document.getElementById('languageSelect');
            const enableSpeakerAnalysis = document.getElementById('enableSpeakerAnalysis').checked;
            
            if (!fileInput.files[0]) {
                alert('파일을 선택해주세요');
                return;
            }
            
            const file = fileInput.files[0];
            const fileName = file.name.toLowerCase();
            const selectedLanguage = languageSelect.value;
            
            // 파일 형식에 따라 API 엔드포인트 결정
            const videoFormats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'];
            const isVideoFile = videoFormats.some(format => fileName.endsWith(format));
            
            const apiEndpoint = isVideoFile ? '/api/process_video' : '/api/process_audio';
            const fileParamName = isVideoFile ? 'video_file' : 'audio_file';
            
            formData.append(fileParamName, file);
            
            // 언어 파라미터 추가
            const apiUrl = `${apiEndpoint}?language=${selectedLanguage}`;
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultDiv.style.display = 'block';
            
            const languageInfo = languageSelect.options[languageSelect.selectedIndex].text;
            
            if (isVideoFile) {
                resultContent.innerHTML = `
                    <div class="status info">
                        🎬 동영상 전체 분석 중...<br>
                        📋 선택된 언어: ${languageInfo}<br>
                        🎭 화자 구분: ${enableSpeakerAnalysis ? '포함' : '제외'}<br>
                        1️⃣ 음성 추출 중...<br>
                        2️⃣ STT 분석 대기 중...
                    </div>
                `;
            } else {
                resultContent.innerHTML = `
                    <div class="status info">
                        🎵 음성 전체 분석 중...<br>
                        📋 선택된 언어: ${languageInfo}<br>
                        🎭 화자 구분: ${enableSpeakerAnalysis ? '포함' : '제외'}<br>
                        잠시만 기다려주세요.
                    </div>
                `;
            }
            
            try {
                // STT 분석 실행
                const sttResponse = await fetch(apiUrl, {
                    method: 'POST',
                    body: formData
                });
                
                const sttResult = await sttResponse.json();
                
                if (sttResult.success) {
                    // STT 결과 표시
                    displaySTTResults(sttResult, isVideoFile);
                    
                    // 화자 구분 분석도 실행 (선택된 경우)
                    if (enableSpeakerAnalysis) {
                        await performSpeakerAnalysis(file);
                    }
                } else {
                    displayErrorResult(sttResult, isVideoFile);
                }
            } catch (error) {
                resultContent.innerHTML = `
                    <div class="status error">
                        <strong>❌ 네트워크 오류</strong><br>
                        ${error.message}
                    </div>
                `;
            }
        };

        // STT 결과 표시 함수
        function displaySTTResults(result, isVideoFile) {
            const resultContent = document.getElementById('resultContent');
            const fileTypeIcon = isVideoFile ? '🎬' : '🎵';
            
            const processingInfo = isVideoFile ? 
                `<li><strong>원본 파일:</strong> ${result.original_filename}</li>
                 <li><strong>추출된 음성:</strong> ${result.extracted_audio_filename}</li>
                 <li><strong>추출 방법:</strong> ${result.extraction_method}</li>` :
                `<li><strong>파일명:</strong> ${result.filename}</li>`;
            
            const langInfo = result.language_info || {};
            const confidence = result.confidence ? Math.round(result.confidence * 100) : 0;
            
            let sttHtml = `
                <div class="status success">
                    <strong>✅ ${isVideoFile ? '동영상' : '음성'} STT 완료!</strong>
                </div>
                <h4>📝 인식된 텍스트:</h4>
                <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0; font-size: 16px; line-height: 1.6;">
                    ${result.transcribed_text || '(텍스트 없음)'}
                </div>
                <h4>🌍 언어 정보:</h4>
                <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 24px;">${langInfo.flag || '🌐'}</span>
                        <div>
                            <strong>감지된 언어:</strong> ${langInfo.name || result.detected_language}<br>
                            <strong>요청한 언어:</strong> ${document.getElementById('languageSelect').options[document.getElementById('languageSelect').selectedIndex].text}
                        </div>
                    </div>
                    ${confidence > 0 ? `
                        <div>신뢰도: ${confidence}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    ` : ''}
                </div>
                <h4>📊 파일 정보:</h4>
                <ul style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    ${processingInfo}
                    <li><strong>파일 크기:</strong> ${result.file_size || result.original_file_size} MB</li>
                    <li><strong>처리 시간:</strong> ${result.processing_time}초</li>
                    <li><strong>파일 타입:</strong> ${result.file_type || (isVideoFile ? 'video' : 'audio')} ${fileTypeIcon}</li>
                </ul>
            `;
            
            resultContent.innerHTML = sttHtml;
        }

        // 화자 구분 분석 실행 (전체 분석의 일부)
        async function performSpeakerAnalysis(file) {
            const resultContent = document.getElementById('resultContent');
            
            // 현재 내용에 로딩 추가
            resultContent.innerHTML += '<div class="status info" id="speakerLoading">🎭 화자 구분 분석 중...</div>';
            
            try {
                const formData = new FormData();
                formData.append('audio_file', file);
                
                const response = await fetch('/api/analyze_speakers', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // 로딩 메시지 제거
                document.getElementById('speakerLoading').remove();
                
                if (result.success) {
                    // 화자 구분 결과 추가
                    appendSpeakerResults(result);
                } else {
                    resultContent.innerHTML += `
                        <div class="status warning">
                            <strong>⚠️ 화자 구분 분석 실패</strong><br>
                            ${result.error}
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('speakerLoading').remove();
                resultContent.innerHTML += `
                    <div class="status warning">
                        <strong>⚠️ 화자 구분 네트워크 오류</strong><br>
                        ${error.message}
                    </div>
                `;
            }
        }

        // 화자 구분 결과를 기존 결과에 추가
        function appendSpeakerResults(result) {
            const resultContent = document.getElementById('resultContent');
            
            // 화자별 통계
            let statsHtml = '<div class="speaker-stats">';
            Object.entries(result.speaker_statistics).forEach(([speaker, stats]) => {
                statsHtml += `
                    <div class="stat-card">
                        <h4>${speaker}</h4>
                        <div><strong>${stats.total_duration}초</strong></div>
                        <div>${stats.percentage}%</div>
                        <div>신뢰도: ${Math.round(stats.avg_confidence * 100)}%</div>
                    </div>
                `;
            });
            statsHtml += '</div>';
            
            // 타임라인
            let timelineHtml = '<div class="speaker-timeline"><h4>🕐 화자별 타임라인</h4>';
            result.segments.forEach((segment, index) => {
                const speakerClass = `speaker-${String.fromCharCode(97 + (index % 4))}`;
                timelineHtml += `
                    <div class="speaker-segment ${speakerClass}">
                        <span class="time-badge">${segment.start}s - ${segment.end}s</span>
                        <strong>${segment.speaker}</strong>
                        <span style="margin-left: auto;">신뢰도: ${Math.round(segment.confidence * 100)}%</span>
                    </div>
                `;
            });
            timelineHtml += '</div>';
            
            // 화자 구분 결과 추가
            resultContent.innerHTML += `
                <h4>🎭 화자 구분 결과:</h4>
                <div class="status success">
                    <strong>✅ 화자 구분 완료! ${result.num_speakers}명 감지</strong>
                </div>
                <h4>📊 화자별 통계</h4>
                ${statsHtml}
                ${timelineHtml}
            `;
        }

        function displayErrorResult(result, isVideoFile) {
            const resultContent = document.getElementById('resultContent');
            let errorContent = `
                <div class="status error">
                    <strong>❌ 처리 실패</strong><br>
                    오류: ${result.error}
                </div>
            `;
            
            // FFmpeg 설치 가이드 표시
            if (result.install_guide) {
                errorContent += `
                    <div class="status warning">
                        <strong>💡 해결 방법:</strong><br>
                        • Windows: <a href="https://ffmpeg.org/download.html" target="_blank">FFmpeg 다운로드</a> 후 PATH 설정<br>
                        • Mac: <code>brew install ffmpeg</code><br>
                        • Ubuntu: <code>sudo apt update && sudo apt install ffmpeg</code>
                    </div>
                `;
            }
            
            resultContent.innerHTML = errorContent;
        }
    </script>
</body>
</html>
    """

def get_error_template(error_title: str, error_message: str) -> str:
    """
    에러 페이지 템플릿
    
    Args:
        error_title: 에러 제목
        error_message: 에러 메시지
        
    Returns:
        에러 페이지 HTML
    """
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>오류 - 솔로몬드 AI 시스템</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        .error {{ background: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ {error_title}</h1>
        <div class="error">
            <p>{error_message}</p>
        </div>
        <p><a href="/">← 메인으로 돌아가기</a></p>
    </div>
</body>
</html>
    """
