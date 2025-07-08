# 개선된 최소 기능 STT 테스트 - 파일 업로드 문제 해결판
# 파일명: minimal_stt_test_fixed.py
# 업데이트: 2025.07.08 - 파일 업로드 안정성 강화

import os
import sys
import tempfile
import traceback
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

# 시스템 정보 출력
print(f"🐍 Python 버전: {sys.version}")
print(f"📁 현재 디렉토리: {os.getcwd()}")

# Whisper 설치 확인 및 자세한 진단
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper 라이브러리 로드 성공")
    
    # Whisper 모델 사전 로드 테스트
    try:
        print("🔄 Whisper 'base' 모델 로드 테스트 중...")
        test_model = whisper.load_model("base")
        print("✅ Whisper 모델 로드 성공")
        del test_model  # 메모리 해제
    except Exception as model_error:
        print(f"⚠️ Whisper 모델 로드 경고: {model_error}")
        
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"❌ Whisper 라이브러리 없음: {e}")
    print("설치 명령어: pip install openai-whisper")

# FastAPI 앱 생성
app = FastAPI(
    title="파일 업로드 문제 해결 테스트",
    description="솔로몬드 AI 시스템 - 파일 업로드 안정성 강화 버전",
    version="2.0.1"
)

# CORS 미들웨어 추가 (브라우저 호환성 향상)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파일 크기 제한 설정 (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# 개선된 HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>파일 업로드 문제 해결 테스트</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 40px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container { 
            max-width: 700px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .upload-area { 
            border: 3px dashed #3498db; 
            padding: 50px; 
            text-align: center; 
            margin: 30px 0; 
            border-radius: 15px;
            background: #f8fafc;
            transition: all 0.3s ease;
        }
        .upload-area:hover { 
            border-color: #2980b9; 
            background: #e3f2fd; 
        }
        input[type="file"] { 
            padding: 15px; 
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            width: 100%;
            margin: 10px 0;
        }
        button { 
            background: linear-gradient(45deg, #3498db, #2980b9); 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        button:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        .result { 
            margin-top: 30px; 
            padding: 25px; 
            background: #f8f9fa; 
            border-radius: 12px; 
            border-left: 5px solid #3498db;
        }
        .status { 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            font-weight: bold;
        }
        .success { background: #d4edda; color: #155724; border-left: 5px solid #28a745; }
        .error { background: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
        .info { background: #d1ecf1; color: #0c5460; border-left: 5px solid #17a2b8; }
        .warning { background: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }
        
        .system-info {
            background: #e8f5e8;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .progress {
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
            display: none;
        }
        
        .progress-bar {
            width: 0%;
            height: 30px;
            background: linear-gradient(45deg, #3498db, #2980b9);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛠️ 파일 업로드 문제 해결 테스트</h1>
        
        <div class="system-info">
            <h3>📊 시스템 진단 상태</h3>
            <strong>Python 버전:</strong> 3.13+ ✅<br>
            <strong>Whisper STT:</strong> """ + ("✅ 사용 가능" if WHISPER_AVAILABLE else "❌ 설치 필요") + """<br>
            <strong>FastAPI:</strong> ✅ 실행 중<br>
            <strong>지원 형식:</strong> MP3, WAV, M4A (최대 100MB)<br>
            <strong>CORS 설정:</strong> ✅ 활성화<br>
            <strong>시간:</strong> <span id="currentTime"></span>
        </div>
        
        """ + ("""
        <div class="status warning">
            <strong>⚠️ Whisper 설치 필요</strong><br>
            음성 인식을 위해 다음 명령어를 실행하세요:<br>
            <code>pip install openai-whisper</code>
        </div>
        """ if not WHISPER_AVAILABLE else "") + """
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <div style="font-size: 48px; margin-bottom: 20px;">🎤</div>
                <h3>음성 파일을 선택하세요</h3>
                <input type="file" name="audio_file" accept=".mp3,.wav,.m4a,.aac,.flac" required>
                <br><br>
                <button type="submit" id="submitBtn">🚀 음성 인식 시작</button>
            </div>
        </form>
        
        <div class="progress" id="progressContainer">
            <div class="progress-bar" id="progressBar">0%</div>
        </div>
        
        <div id="result" class="result" style="display:none;">
            <h3>📋 처리 결과</h3>
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        // 현재 시간 표시
        function updateTime() {
            const now = new Date();
            document.getElementById('currentTime').textContent = now.toLocaleString('ko-KR');
        }
        updateTime();
        setInterval(updateTime, 1000);
        
        // 진행률 시뮬레이션
        function startProgress() {
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            
            progressContainer.style.display = 'block';
            let progress = 0;
            
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                
                progressBar.style.width = progress + '%';
                progressBar.textContent = Math.round(progress) + '%';
            }, 500);
            
            return interval;
        }
        
        // 파일 업로드 처리
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const fileInput = document.querySelector('input[type="file"]');
            const submitBtn = document.getElementById('submitBtn');
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            // 파일 선택 확인
            if (!fileInput.files[0]) {
                alert('파일을 선택해주세요');
                return;
            }
            
            const file = fileInput.files[0];
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);
            
            // 파일 크기 검사
            if (file.size > 100 * 1024 * 1024) {
                alert('파일 크기가 100MB를 초과합니다.');
                return;
            }
            
            // UI 상태 변경
            submitBtn.disabled = true;
            submitBtn.textContent = '🔄 처리 중...';
            resultDiv.style.display = 'block';
            
            // 진행률 시작
            const progressInterval = startProgress();
            
            resultContent.innerHTML = `
                <div class="status info">
                    <strong>📁 파일 정보</strong><br>
                    파일명: ${file.name}<br>
                    크기: ${fileSize} MB<br>
                    형식: ${file.type || '자동 감지'}
                </div>
                <div class="status info">
                    🔄 서버로 업로드 중... 잠시만 기다려주세요.
                </div>
            `;
            
            try {
                // FormData 생성
                const formData = new FormData();
                formData.append('audio_file', file);
                
                console.log('📤 파일 업로드 시작:', file.name);
                
                // 서버로 전송
                const response = await fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                });
                
                console.log('📡 서버 응답 수신:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const result = await response.json();
                console.log('📋 처리 결과:', result);
                
                // 진행률 완료
                clearInterval(progressInterval);
                document.getElementById('progressBar').style.width = '100%';
                document.getElementById('progressBar').textContent = '완료!';
                
                if (result.success) {
                    resultContent.innerHTML = `
                        <div class="status success">
                            <strong>✅ 음성 인식 성공!</strong><br>
                            처리 시간: ${result.processing_time}초
                        </div>
                        <div style="background: white; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #28a745;">
                            <h4>🎯 인식된 텍스트:</h4>
                            <p style="font-size: 16px; line-height: 1.6; margin: 0;">
                                ${result.transcribed_text || '(텍스트가 감지되지 않았습니다)'}
                            </p>
                        </div>
                        <div class="status info">
                            <strong>📊 상세 정보</strong><br>
                            언어: ${result.detected_language || '자동 감지'}<br>
                            파일 크기: ${result.file_size}<br>
                            처리 방식: Whisper ${result.model || 'base'} 모델
                        </div>
                    `;
                } else {
                    resultContent.innerHTML = `
                        <div class="status error">
                            <strong>❌ 처리 실패</strong><br>
                            오류: ${result.error}<br>
                            처리 시간: ${result.processing_time || 0}초
                        </div>
                        <div class="status warning">
                            <strong>💡 해결 방법:</strong><br>
                            • 파일 형식 확인 (MP3, WAV, M4A)<br>
                            • 파일 크기 확인 (100MB 이하)<br>
                            • Whisper 설치 확인<br>
                            • 네트워크 연결 확인
                        </div>
                    `;
                }
            } catch (error) {
                console.error('❌ 업로드 오류:', error);
                
                clearInterval(progressInterval);
                
                resultContent.innerHTML = `
                    <div class="status error">
                        <strong>❌ 네트워크 오류</strong><br>
                        ${error.message}
                    </div>
                    <div class="status warning">
                        <strong>🔧 진단 단계:</strong><br>
                        1. 서버가 실행 중인지 확인<br>
                        2. 방화벽 설정 확인<br>
                        3. 파일 권한 확인<br>
                        4. 브라우저 콘솔(F12) 확인
                    </div>
                `;
            } finally {
                // UI 복원
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 음성 인식 시작';
                
                setTimeout(() => {
                    document.getElementById('progressContainer').style.display = 'none';
                }, 2000);
            }
        };
        
        // 파일 선택 시 정보 표시
        document.querySelector('input[type="file"]').onchange = function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileSize = (file.size / (1024 * 1024)).toFixed(2);
                console.log(`📁 파일 선택됨: ${file.name} (${fileSize} MB)`);
            }
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return HTML_TEMPLATE

@app.post("/process_audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """음성 파일 처리 - 강화된 오류 처리"""
    import time
    start_time = time.time()
    
    try:
        # 1. 파일 기본 정보 확인
        filename = audio_file.filename or "unknown_file"
        print(f"📁 파일 수신 시작: {filename}")
        
        # 2. 파일 내용 읽기
        try:
            content = await audio_file.read()
            file_size_mb = len(content) / (1024 * 1024)
            file_size_str = f"{file_size_mb:.2f} MB"
            
            print(f"📊 파일 크기: {file_size_str}")
            
            # 파일 크기 제한 확인
            if len(content) > MAX_FILE_SIZE:
                return JSONResponse({
                    "success": False,
                    "error": f"파일 크기가 {MAX_FILE_SIZE/(1024*1024):.0f}MB를 초과합니다. (현재: {file_size_str})",
                    "processing_time": round(time.time() - start_time, 2)
                })
                
        except Exception as read_error:
            print(f"❌ 파일 읽기 오류: {read_error}")
            return JSONResponse({
                "success": False,
                "error": f"파일 읽기 실패: {str(read_error)}",
                "processing_time": round(time.time() - start_time, 2)
            })
        
        # 3. Whisper 사용 가능 여부 확인
        if not WHISPER_AVAILABLE:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 명령어로 설치하세요.",
                "processing_time": round(time.time() - start_time, 2)
            })
        
        # 4. 파일 형식 확인
        file_ext = Path(filename).suffix.lower()
        supported_formats = ['.mp3', '.wav', '.m4a', '.aac', '.flac']
        
        if file_ext not in supported_formats:
            return JSONResponse({
                "success": False,
                "error": f"지원하지 않는 파일 형식: {file_ext}. 지원 형식: {', '.join(supported_formats)}",
                "processing_time": round(time.time() - start_time, 2)
            })
        
        # 5. 임시 파일 생성 및 저장
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
                
            print(f"💾 임시 파일 생성: {temp_path}")
            
            # 6. Whisper 모델 로딩
            print("🔄 Whisper 모델 로딩...")
            try:
                model = whisper.load_model("base")
                print("✅ Whisper 모델 로드 완료")
            except Exception as model_error:
                print(f"❌ Whisper 모델 로드 실패: {model_error}")
                return JSONResponse({
                    "success": False,
                    "error": f"Whisper 모델 로드 실패: {str(model_error)}",
                    "processing_time": round(time.time() - start_time, 2)
                })
            
            # 7. 음성 인식 실행
            print("🎤 음성 인식 시작...")
            try:
                result = model.transcribe(temp_path, language="ko")
                transcribed_text = result["text"].strip()
                detected_language = result.get("language", "unknown")
                
                print(f"✅ 음성 인식 완료")
                print(f"📝 결과 미리보기: {transcribed_text[:100]}...")
                
            except Exception as transcribe_error:
                print(f"❌ 음성 인식 실패: {transcribe_error}")
                return JSONResponse({
                    "success": False,
                    "error": f"음성 인식 처리 실패: {str(transcribe_error)}",
                    "processing_time": round(time.time() - start_time, 2)
                })
            
            # 8. 성공 응답
            processing_time = round(time.time() - start_time, 2)
            
            return JSONResponse({
                "success": True,
                "filename": filename,
                "file_size": file_size_str,
                "transcribed_text": transcribed_text,
                "detected_language": detected_language,
                "processing_time": processing_time,
                "model": "base",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        finally:
            # 9. 임시 파일 정리
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"🗑️ 임시 파일 삭제: {temp_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ 임시 파일 삭제 실패: {cleanup_error}")
                    
    except Exception as e:
        # 예상치 못한 오류 처리
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ 예상치 못한 오류: {error_msg}")
        print(f"🔍 오류 상세:\n{traceback.format_exc()}")
        
        return JSONResponse({
            "success": False,
            "error": f"서버 내부 오류: {error_msg}",
            "processing_time": processing_time,
            "debug_info": "자세한 내용은 서버 콘솔을 확인하세요."
        })

@app.get("/test")
async def test():
    """시스템 진단 테스트"""
    import platform
    
    try:
        # Whisper 모델 테스트
        whisper_status = "설치됨"
        if WHISPER_AVAILABLE:
            try:
                test_model = whisper.load_model("base")
                whisper_status = "정상 작동"
                del test_model
            except:
                whisper_status = "설치됨 (모델 로드 오류)"
        else:
            whisper_status = "미설치"
            
        return JSONResponse({
            "status": "OK",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": platform.python_version(),
                "platform": platform.system(),
                "architecture": platform.machine()
            },
            "dependencies": {
                "whisper": whisper_status,
                "fastapi": "정상",
                "uvicorn": "정상"
            },
            "settings": {
                "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
                "supported_formats": ["MP3", "WAV", "M4A", "AAC", "FLAC"],
                "cors_enabled": True
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "ERROR",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

if __name__ == "__main__":
    print("=" * 80)
    print("🛠️  파일 업로드 문제 해결 테스트 서버")
    print("=" * 80)
    print(f"🐍 Python: {sys.version}")
    print(f"🎤 Whisper: {'✅ 사용 가능' if WHISPER_AVAILABLE else '❌ 설치 필요'}")
    print(f"📁 최대 파일 크기: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"🌐 접속 주소: http://localhost:8080")
    print(f"🔧 진단 URL: http://localhost:8080/test")
    print("=" * 80)
    
    if not WHISPER_AVAILABLE:
        print("⚠️  Whisper 설치 필요:")
        print("   pip install openai-whisper")
        print("   설치 후 서버를 다시 시작하세요.")
        print("=" * 80)
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8080,
            reload=False,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 서버가 안전하게 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 서버 시작 오류: {e}")
        print("해결 방법:")
        print("1. 포트 8080이 사용 중인지 확인")
        print("2. 방화벽 설정 확인")
        print("3. 관리자 권한으로 실행")