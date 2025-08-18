# 🚀 SOLOMOND AI - n8n 워크플로우 자동화 시스템 완전 통합 가이드

## 🎯 개요
SOLOMOND AI의 컨퍼런스 분석 시스템이 완료되면 자동으로 n8n 워크플로우가 실행되어 다음과 같은 자동화가 수행됩니다:

1. **듀얼 브레인 AI 인사이트** 자동 생성
2. **Google Calendar 이벤트** 자동 생성 
3. **시스템 모니터링** 및 알림 발송
4. **데이터 파이프라인** 후속 처리

---

## 🔧 1단계: 시스템 설치 및 설정

### A. 자동 설치 (권장)
```bash
# 완전 자동 설치 실행
python setup_n8n_automation.py
```

### B. 수동 설치
1. **Node.js 설치**: https://nodejs.org (필수)
2. **n8n 설치**: `npm install -g n8n`
3. **Python 의존성**: `pip install httpx asyncio requests`

---

## 📋 2단계: 사용자 제공 필요 정보

### 🔑 필수 정보 (사용자가 준비해야 할 것들)

#### A. Google Calendar API 인증
```
📍 위치: Google Cloud Console (https://console.cloud.google.com/)
📋 준비물:
  1. Google 계정
  2. 새 프로젝트 생성 (또는 기존 프로젝트 사용)
  3. Google Calendar API 활성화
  4. OAuth 2.0 인증 정보 생성
  5. credentials.json 파일 다운로드
  
📁 파일 위치: C:\Users\PC_58410\solomond-ai-system\credentials.json
```

#### B. n8n 초기 설정
```
🌐 n8n 대시보드: http://localhost:5678
📋 최초 실행 시 필요:
  1. 관리자 계정 생성 (이메일 + 비밀번호)
  2. 워크플로우 자동 import 허용
  3. Google Calendar credential 연결
```

#### C. SOLOMOND AI 시스템 실행
```
🎯 필수 서비스:
  1. 포트 8501: 컨퍼런스 분석 시스템
  2. 포트 8500: 메인 대시보드
  
🚀 시작 명령어:
  streamlit run conference_analysis_COMPLETE_WORKING.py --server.port 8501
```

---

## 🚀 3단계: 시스템 시작 방법

### 원클릭 시작 (권장)
```bash
# Windows
start_n8n_system.bat

# 또는 Python 직접 실행
python setup_n8n_automation.py
```

### 수동 시작
```bash
# 1. n8n 서버 시작
n8n start

# 2. SOLOMOND AI 시작
streamlit run conference_analysis_COMPLETE_WORKING.py --server.port 8501

# 3. 메인 대시보드 시작 (선택사항)
streamlit run solomond_ai_main_dashboard.py --server.port 8500
```

---

## ✅ 4단계: 작동 확인 및 테스트

### A. 시스템 테스트 실행
```bash
python test_n8n_integration.py
```

### B. 수동 확인 체크리스트
- [ ] n8n 대시보드 접속: http://localhost:5678
- [ ] SOLOMOND AI 접속: http://localhost:8501
- [ ] 컨퍼런스 분석 기능 정상 작동
- [ ] 분석 완료 후 n8n 워크플로우 자동 실행 확인

---

## 🔗 5단계: 자동화 워크플로우 사용법

### 실제 사용 과정
1. **파일 업로드**: 컨퍼런스 관련 이미지/음성/비디오 파일
2. **분석 실행**: SOLOMOND AI가 자동으로 분석 수행
3. **분석 완료**: 자동으로 n8n 웹훅 트리거 전송
4. **자동화 실행**:
   - ✅ AI 인사이트 생성
   - ✅ 구글 캘린더 이벤트 생성
   - ✅ 시스템 모니터링 알림
   - ✅ 후속 데이터 처리

### 결과 확인 방법
- **n8n 대시보드**: 워크플로우 실행 이력 확인
- **Google Calendar**: 자동 생성된 이벤트 확인  
- **SOLOMOND AI**: 듀얼 브레인 인사이트 확인

---

## 🛠️ 6단계: 문제 해결

### 일반적인 문제들

#### A. n8n 연결 실패
```bash
# 해결 방법
1. Node.js 재설치
2. n8n 재설치: npm install -g n8n
3. 방화벽 5678 포트 허용
4. start_n8n_system.bat 관리자 권한으로 실행
```

#### B. Google Calendar 인증 실패
```bash
# 해결 방법
1. credentials.json 파일 경로 확인
2. Google Cloud Console에서 API 활성화 재확인
3. OAuth 동의 화면 설정 완료
4. 리다이렉트 URI 정확히 설정
```

#### C. SOLOMOND AI 포트 충돌
```bash
# 해결 방법
1. 포트 상태 확인: netstat -an | findstr 8501
2. 프로세스 종료: taskkill /f /im streamlit.exe
3. 다른 포트 사용: --server.port 8502
```

---

## 📚 추가 자료

### 관련 파일들
- `setup_n8n_automation.py`: 완전 자동 설치 스크립트
- `test_n8n_integration.py`: 종합 테스트 스크립트
- `start_n8n_system.bat`: 시스템 시작 스크립트
- `GOOGLE_CALENDAR_SETUP_GUIDE.md`: 상세 캘린더 설정 가이드
- `N8N_INTEGRATION_TEST_REPORT.md`: 테스트 결과 보고서

### 시스템 URL 정리
- **n8n 대시보드**: http://localhost:5678
- **SOLOMOND AI 분석**: http://localhost:8501  
- **메인 대시보드**: http://localhost:8500
- **웹 크롤러**: http://localhost:8502
- **보석 분석**: http://localhost:8503
- **3D CAD 변환**: http://localhost:8504

---

## 🎉 완료!

이제 SOLOMOND AI에서 컨퍼런스 분석이 완료되면 자동으로:
- 🧠 AI가 인사이트를 생성하고
- 📅 구글 캘린더에 이벤트를 추가하며  
- 📊 시스템 모니터링까지 자동으로 수행됩니다!

**문제가 발생하면**: `python test_n8n_integration.py` 실행하여 문제점을 진단하세요.

---
*SOLOMOND AI Project Manager가 완전 자동화 시스템을 구축했습니다. 🤖*