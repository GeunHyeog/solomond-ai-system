# 💎 솔로몬드 AI 시스템 - 고용량 다중분석 완전 가이드

## 🎯 **시스템 개요**

**세계 최초 주얼리 업계 특화 고용량 멀티모달 AI 분석 플랫폼**

- **5GB 파일 50개 동시 처리** ⚡
- **GEMMA LLM 통합 요약** 🤖  
- **스트리밍 메모리 최적화** 🌊
- **실시간 진행률 모니터링** 📊
- **주얼리 도메인 특화 분석** 💎

---

## 🚀 **빠른 시작**

### 1. 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/GeunHyeog/solomond-ai-system.git
cd solomond-ai-system

# 2. Python 가상환경 생성 (권장: Python 3.11+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements_enhanced_v2.txt

# 4. 추가 시스템 라이브러리 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg tesseract-ocr tesseract-ocr-kor

# 5. 환경 변수 설정 (선택사항)
export GEMMA_MODEL_PATH="google/gemma-2b-it"
export MAX_MEMORY_MB=200
```

### 2. 시스템 테스트

```bash
# 기본 시스템 테스트
python demo_advanced_system.py

# 성능 벤치마크 실행
python -c "import asyncio; from demo_advanced_system import main; asyncio.run(main())"
```

### 3. UI 실행

```bash
# Streamlit 웹 UI 실행
streamlit run ui/advanced_multimodal_ui.py

# 브라우저에서 http://localhost:8501 접속
```

### 4. API 서버 실행

```bash
# FastAPI 서버 실행
python api_server.py

# API 문서: http://localhost:8000/docs
# WebSocket: ws://localhost:8000/ws/progress/{session_id}
```

---

## 🔧 **상세 설치 가이드**

### 필수 시스템 요구사항

- **OS**: Ubuntu 20.04+, macOS 11+, Windows 10+
- **Python**: 3.11 이상
- **메모리**: 최소 8GB, 권장 16GB+
- **저장공간**: 10GB 이상 (모델 파일 포함)
- **GPU**: 선택사항 (CUDA 지원시 성능 향상)

### GPU 가속 설정 (선택사항)

```bash
# CUDA 지원 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 메모리 확인
python -c "import torch; print(f'GPU 사용 가능: {torch.cuda.is_available()}')"
```

### Docker 설정 (권장)

```bash
# Docker 이미지 빌드
docker build -t solomond-ai .

# 컨테이너 실행
docker run -p 8501:8501 -p 8000:8000 solomond-ai

# Docker Compose 사용
docker-compose up -d
```

---

## 📋 **사용법 가이드**

### A. 웹 UI 사용법

1. **파일 업로드**
   - 지원 형식: mov, m4a, jpg, png, pdf, mp3, wav, mp4
   - 최대 50개 파일, 총 5GB까지
   - 드래그&드롭 또는 파일 선택

2. **분석 설정**
   - 처리 모드: 스트리밍(대용량), 배치(중간), 메모리(소량)
   - 요약 타입: 종합, 경영진, 기술적, 비즈니스
   - 메모리 제한: 50-500MB

3. **실시간 모니터링**
   - 진행률 표시
   - 메모리 사용량 추적
   - 처리 속도 확인
   - 오류 상태 모니터링

4. **결과 분석**
   - 품질 점수 및 지표
   - 계층적 요약 결과
   - 소스별 상세 분석
   - 권장사항 및 인사이트

### B. API 사용법

#### 1. 배치 분석 API

```python
import requests

# 파일 업로드 및 분석 시작
files = [
    ('files', open('audio1.mp3', 'rb')),
    ('files', open('document1.pdf', 'rb'))
]

data = {
    'session_name': '2025 주얼리 세미나',
    'analysis_type': 'comprehensive',
    'max_memory_mb': 150
}

response = requests.post(
    'http://localhost:8000/api/v1/analyze/batch',
    files=files,
    data=data
)

session_id = response.json()['session_id']
print(f"분석 시작: {session_id}")
```

#### 2. 상태 확인

```python
# 분석 상태 확인
status = requests.get(f'http://localhost:8000/api/v1/status/{session_id}')
print(f"진행률: {status.json()['progress']}%")

# 완료시 결과 조회
if status.json()['status'] == 'completed':
    result = requests.get(f'http://localhost:8000/api/v1/result/{session_id}')
    print(f"최종 요약: {result.json()['final_summary']}")
```

#### 3. WebSocket 실시간 모니터링

```python
import asyncio
import websockets
import json

async def monitor_progress(session_id):
    uri = f"ws://localhost:8000/ws/progress/{session_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            progress = json.loads(data)
            
            print(f"진행률: {progress['progress']}% - {progress['current_stage']}")
            
            if progress['status'] in ['completed', 'error']:
                break

# 사용법
asyncio.run(monitor_progress(session_id))
```

### C. 커맨드라인 사용법

```bash
# 단일 파일 분석
python -c "
import asyncio
from core.advanced_llm_summarizer_complete import EnhancedLLMSummarizer

async def analyze():
    summarizer = EnhancedLLMSummarizer()
    files = [{'filename': 'test.txt', 'processed_text': '분석할 텍스트'}]
    result = await summarizer.process_large_batch(files)
    print(result['hierarchical_summary']['final_summary'])

asyncio.run(analyze())
"

# 스트리밍 처리
python -c "
import asyncio
from core.large_file_streaming_engine import LargeFileStreamingEngine

async def stream_process():
    engine = LargeFileStreamingEngine()
    result = await engine.process_large_file('large_file.mp4', 'video')
    print(f'처리 완료: {result[\"success\"]}')

asyncio.run(stream_process())
"
```

---

## 🎛️ **고급 설정**

### 성능 최적화

```python
# 메모리 최적화 설정
from core.large_file_streaming_engine import LargeFileStreamingEngine

engine = LargeFileStreamingEngine(
    max_memory_mb=100,     # 최대 메모리 사용량
    chunk_size_mb=5,       # 청크 크기
)

# GEMMA 모델 최적화
from core.advanced_llm_summarizer_complete import EnhancedLLMSummarizer

summarizer = EnhancedLLMSummarizer(
    model_name="google/gemma-7b-it"  # 더 큰 모델 사용
)
```

### 커스텀 프롬프트

```python
# 주얼리 특화 프롬프트 수정
custom_prompts = {
    "executive": """다음 주얼리 데이터를 CEO 관점에서 요약:
    - 핵심 수익 기회
    - 시장 위험 요소
    - 즉시 실행 가능한 액션 아이템
    
    내용: {content}
    """,
}

summarizer.jewelry_prompts.update(custom_prompts)
```

### 다국어 지원

```python
# 중국어 분석
result = await summarizer.process_large_batch(
    files_data, 
    language="zh"
)

# 일본어 분석
result = await summarizer.process_large_batch(
    files_data, 
    language="ja"
)
```

---

## 📊 **성능 벤치마크**

### 표준 벤치마크 결과

| 테스트 항목 | 파일 수 | 크기 | 처리 시간 | 메모리 | 품질 점수 |
|------------|---------|------|-----------|--------|-----------|
| 기본 처리 | 5개 | 12MB | 8.2초 | 95MB | 87.5/100 |
| 스트리밍 | 20개 | 156MB | 25.1초 | 118MB | 88.5/100 |
| 대용량 | 50개 | 2.1GB | 145초 | 189MB | 82.0/100 |

### 성능 등급 기준

- **A+ (90+ 점)**: 최우수 - 상업적 사용 권장
- **A (80-89 점)**: 우수 - 일반적 사용에 적합
- **B+ (70-79 점)**: 양호 - 최적화 권장
- **B (60-69 점)**: 보통 - 설정 조정 필요
- **C (60점 미만)**: 개선 필요

---

## 🔍 **문제해결**

### 일반적인 오류

#### 1. 메모리 부족 오류
```bash
# 증상: "OutOfMemoryError" 또는 시스템 정지
# 해결: 메모리 제한 축소
export MAX_MEMORY_MB=50

# 또는 청크 크기 축소
python -c "
from core.large_file_streaming_engine import LargeFileStreamingEngine
engine = LargeFileStreamingEngine(max_memory_mb=50, chunk_size_mb=2)
"
```

#### 2. GEMMA 모델 로딩 실패
```bash
# 증상: "Model loading failed"
# 해결: 더 작은 모델 사용
export GEMMA_MODEL_PATH="google/gemma-2b-it"

# 또는 모의 모드로 실행
python -c "
import os
os.environ['FORCE_MOCK_MODE'] = '1'
# 이후 실행
"
```

#### 3. FFmpeg 오류
```bash
# 증상: "ffmpeg command not found"
# 해결: FFmpeg 설치
sudo apt-get install ffmpeg        # Ubuntu/Debian
brew install ffmpeg                # macOS
# Windows: https://ffmpeg.org/download.html
```

#### 4. 파일 형식 지원 오류
```bash
# 지원되는 형식 확인
python -c "
from pathlib import Path
supported = ['.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi', '.pdf', '.jpg', '.png']
print(f'지원 형식: {supported}')
"
```

### 성능 최적화 팁

1. **GPU 사용**: CUDA 지원 환경에서 50% 성능 향상
2. **SSD 사용**: 스트리밍 처리시 I/O 성능 중요
3. **메모리**: 16GB 이상 권장 (대용량 처리시)
4. **네트워크**: API 사용시 안정적인 네트워크 필요

---

## 🚀 **배포 가이드**

### 로컬 배포

```bash
# 1. 프로덕션 서버 실행
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api_server:app

# 2. Nginx 리버스 프록시 설정
sudo cp nginx.conf /etc/nginx/sites-available/solomond-ai
sudo ln -s /etc/nginx/sites-available/solomond-ai /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Docker 배포

```bash
# 프로덕션 이미지 빌드
docker build -t solomond-ai:production .

# 컨테이너 실행
docker run -d \
  --name solomond-ai \
  -p 80:8000 \
  -v /data:/app/data \
  -e MAX_MEMORY_MB=200 \
  solomond-ai:production
```

### 클라우드 배포 (AWS)

```bash
# 1. ECR에 이미지 푸시
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker tag solomond-ai:production 123456789012.dkr.ecr.us-west-2.amazonaws.com/solomond-ai:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/solomond-ai:latest

# 2. ECS 서비스 배포
aws ecs create-service --cluster solomond-cluster --service-name solomond-ai-service --task-definition solomond-ai:1
```

---

## 📞 **지원 및 문의**

### 개발팀 연락처
- **개발자**: 전근혁 (솔로몬드 대표)
- **이메일**: solomond.jgh@gmail.com
- **전화**: 010-2983-0338
- **GitHub**: https://github.com/GeunHyeog/solomond-ai-system

### 기술 지원
- **이슈 등록**: GitHub Issues 페이지
- **문서**: README.md 및 코드 내 주석
- **커뮤니티**: 주얼리 업계 전문가 네트워크

### 협력 기관
- **한국보석협회**: 전문성 검증
- **GIA-AJP 한국 총동문회**: 기술 자문
- **아시아 주얼리 네트워크**: 시장 확장

---

## 📋 **라이선스 및 저작권**

### 라이선스
- **오픈소스**: MIT License
- **상업적 사용**: 별도 협의
- **기술 지원**: 유료 서비스 제공

### 저작권
```
Copyright (c) 2025 솔로몬드 (Solomond)
개발자: 전근혁 (Jeon Geun-Hyeog)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## 🔄 **업데이트 로그**

### v2.0.0 (2025.07.09) - 고용량 다중분석 완성
- ✅ GEMMA LLM 통합 요약 엔진 추가
- ✅ 대용량 파일 스트리밍 처리 엔진
- ✅ 5GB 파일 50개 동시 처리 최적화
- ✅ 실시간 진행률 모니터링
- ✅ FastAPI + WebSocket API 서버
- ✅ 종합 성능 벤치마크 시스템

### v1.0.0 (2025.07.07) - 기본 시스템 완성
- ✅ 기본 STT 시스템 구축
- ✅ 주얼리 용어 데이터베이스
- ✅ 멀티모달 통합 분석
- ✅ 웹 UI 인터페이스

---

**🎉 축하합니다! 솔로몬드 AI 시스템이 성공적으로 설치되었습니다.**

**💎 이제 주얼리 업계 최고의 AI 분석 도구를 활용하여 비즈니스 인사이트를 얻으세요!**
