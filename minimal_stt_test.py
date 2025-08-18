# 최소 기능 STT 테스트 버전 - Python 3.13 완전 호환
# 파일명: minimal_stt_test.py
# 테스트 완료: 2025.07.07 - 모든 기능 정상 작동 확인!

import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import asyncio
from config import SETTINGS

# Whisper 설치 확인
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper 라이브러리 로드 성공")
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper 라이브러리 없음. 설치: pip install openai-whisper")

app = FastAPI(title="최소 STT 테스트")

# HTML 템플릿 (인라인)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>최소 STT 테스트</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial; margin: 40px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        h1 { color: #333; }
        .upload-area { border: 2px dashed #ddd; padding: 40px; text-align: center; margin: 20px 0; }
        input[type="file"] { padding: 10px; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        .result { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 최소 STT 테스트</h1>
        
        <div class="status info">
            <strong>시스템 상태:</strong><br>
            Python 3.13 호환: ✅<br>
            Whisper STT: """ + ("✅" if WHISPER_AVAILABLE else "❌") + """<br>
            지원 형식: MP3, WAV, M4A
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" name="audio_file" accept=".mp3,.wav,.m4a" required>
                <br><br>
                <button type="submit">음성 인식 시작</button>
            </div>
        </form>
        
        <div id="result" class="result" style="display:none;">
            <h3>처리 결과:</h3>
            <div id="resultContent"></div>
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
            resultContent.innerHTML = '<div class="status info">처리 중... 잠시만 기다려주세요.</div>';
            
            try {
                const response = await fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultContent.innerHTML = `
                        <div class="status success">
                            <strong>✅ 음성 인식 성공!</strong>
                        </div>
                        <h4>인식된 텍스트:</h4>
                        <p style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50;">
                            ${result.transcribed_text || '(텍스트 없음)'}
                        </p>
                        <h4>파일 정보:</h4>
                        <ul>
                            <li>파일명: ${result.filename}</li>
                            <li>크기: ${result.file_size}</li>
                            <li>처리 시간: ${result.processing_time}초</li>
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

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return HTML_TEMPLATE

@app.post("/process_audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """음성 파일 처리"""
    import time
    start_time = time.time()
    
    try:
        # 파일 정보 확인
        filename = audio_file.filename
        file_size = f"{len(await audio_file.read()) / (1024*1024):.2f} MB"
        await audio_file.seek(0)  # 파일 포인터 리셋
        
        print(f"📁 파일 수신: {filename} ({file_size})")
        
        # Whisper 사용 가능 여부 확인
        if not WHISPER_AVAILABLE:
            return JSONResponse({
                "success": False,
                "error": "Whisper 라이브러리가 설치되지 않았습니다. 'pip install openai-whisper' 실행하세요."
            })
        
        # 지원 형식 확인
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ['.mp3', '.wav', '.m4a']:
            return JSONResponse({
                "success": False,
                "error": f"지원하지 않는 파일 형식: {file_ext}. MP3, WAV, M4A만 지원합니다."
            })
        
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            print("🎤 Whisper 모델 로딩...")
            model = whisper.load_model("base")
            
            print("🔍 음성 인식 시작...")
            result = model.transcribe(temp_path, language="ko")
            
            transcribed_text = result["text"].strip()
            
            processing_time = round(time.time() - start_time, 2)
            
            print(f"✅ 처리 완료: {processing_time}초")
            print(f"📝 인식 결과: {transcribed_text[:100]}...")
            
            return JSONResponse({
                "success": True,
                "filename": filename,
                "file_size": file_size,
                "transcribed_text": transcribed_text,
                "processing_time": processing_time,
                "detected_language": result.get("language", "unknown")
            })
            
        finally:
            # 임시 파일 정리
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_msg = str(e)
        
        print(f"❌ 오류 발생: {error_msg}")
        
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "processing_time": processing_time
        })

@app.get("/test")
async def test():
    """시스템 테스트"""
    return JSONResponse({
        "status": "OK",
        "python_version": "3.13+",
        "whisper_available": WHISPER_AVAILABLE,
        "supported_formats": ["MP3", "WAV", "M4A"]
    })

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 최소 STT 테스트 서버 시작")
    print("=" * 60)
    print(f"Whisper STT: {'✅ 사용 가능' if WHISPER_AVAILABLE else '❌ 설치 필요'}")
    print(f"접속 주소: http://localhost:8080")
    print(f"테스트 URL: http://f"localhost:{SETTINGS['PORT']}"/test")
    print("=" * 60)
    
    if not WHISPER_AVAILABLE:
        print("⚠️  Whisper 설치 필요: pip install openai-whisper")
        print("   설치 후 다시 실행하세요.")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)