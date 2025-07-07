"""
솔로몬드 AI 시스템 - UI 템플릿
HTML 템플릿 및 UI 컴포넌트 관리
"""

def get_main_template() -> str:
    """
    메인 페이지 HTML 템플릿 반환
    
    기존 minimal_stt_test.py와 동일한 인터페이스 제공
    """
    
    return """
<!DOCTYPE html>
<html>
<head>
    <title>솔로몬드 AI 시스템 v3.0</title>
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
            max-width: 700px; 
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
        .upload-area { 
            border: 3px dashed #ddd; 
            padding: 50px; 
            text-align: center; 
            margin: 30px 0;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #4CAF50;
            background: #f8f9fa;
        }
        input[type="file"] { 
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
        }
        button { 
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 18px;
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }
        .result { 
            margin-top: 30px; 
            padding: 25px; 
            background: #f8f9fa; 
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 솔로몬드 AI 시스템</h1>
        <div class="subtitle">
            실제 내용을 읽고 분석하는 차세대 AI 플랫폼
            <span class="version-badge">v3.0 모듈화</span>
        </div>
        
        <div class="status info">
            <strong>🎯 시스템 상태:</strong><br>
            ✅ Python 3.13 완전 호환<br>
            ✅ 모듈화 구조 완성<br>
            ✅ Whisper STT 엔진<br>
            🎵 지원 형식: MP3, WAV, M4A
        </div>
        
        <div class="feature-list">
            <div class="feature-card">
                <h4>🎤 고급 STT</h4>
                <p>OpenAI Whisper 기반 정확한 음성 인식</p>
            </div>
            <div class="feature-card">
                <h4>🌐 다국어</h4>
                <p>한국어, 영어 등 다양한 언어 지원</p>
            </div>
            <div class="feature-card">
                <h4>📱 반응형</h4>
                <p>모바일 친화적 인터페이스</p>
            </div>
            <div class="feature-card">
                <h4>⚡ 빠른 처리</h4>
                <p>최적화된 처리 성능</p>
            </div>
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <h3>🎵 음성 파일 업로드</h3>
                <input type="file" name="audio_file" accept=".mp3,.wav,.m4a" required>
                <br>
                <button type="submit">🚀 음성 인식 시작</button>
            </div>
        </form>
        
        <div id="result" class="result" style="display:none;">
            <h3>📄 처리 결과:</h3>
            <div id="resultContent"></div>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666; font-size: 14px;">
            개발자: 전근혁 (솔로몬드 대표) | 
            <a href="/docs" target="_blank">API 문서</a> | 
            <a href="/test" target="_blank">시스템 테스트</a>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.querySelector('input[type="file"]');
            
            if (!fileInput.files[0]) {
                alert('파일을 선택해주세요');
                return;
            }
            
            formData.append('audio_file', fileInput.files[0]);
            
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            resultDiv.style.display = 'block';
            resultContent.innerHTML = '<div class="status info">🔄 처리 중... 잠시만 기다려주세요.</div>';
            
            try {
                // 모듈화된 API 엔드포인트 사용
                const response = await fetch('/api/process_audio', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultContent.innerHTML = `
                        <div class="status success">
                            <strong>✅ 음성 인식 성공!</strong>
                        </div>
                        <h4>📝 인식된 텍스트:</h4>
                        <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0; font-size: 16px; line-height: 1.6;">
                            ${result.transcribed_text || '(텍스트 없음)'}
                        </div>
                        <h4>📊 파일 정보:</h4>
                        <ul style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                            <li><strong>파일명:</strong> ${result.filename}</li>
                            <li><strong>크기:</strong> ${result.file_size}</li>
                            <li><strong>처리 시간:</strong> ${result.processing_time}초</li>
                            <li><strong>언어:</strong> ${result.detected_language}</li>
                        </ul>
                    `;
                } else {
                    resultContent.innerHTML = `
                        <div class="status error">
                            <strong>❌ 처리 실패</strong><br>
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
        };
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
