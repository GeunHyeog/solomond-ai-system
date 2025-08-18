# 🔗 SOLOMOND AI n8n 워크플로우 설정 가이드

## 📋 현재 상황
- ✅ n8n 서버 정상 작동 (포트 5678)
- ⏳ 워크플로우 생성 필요
- ⏳ 구글 캘린더 연동 설정 필요

## 🛠️ **단계별 설정 방법**

### 1️⃣ **n8n 웹 인터페이스 접속**
1. 브라우저에서 접속: http://localhost:5678
2. 최초 접속 시 계정 생성 (이메일/비밀번호)
3. 워크플로우 에디터 접속

### 2️⃣ **SOLOMOND Dual Brain Pipeline 생성**

#### 새 워크플로우 생성:
1. "New Workflow" 클릭
2. 워크플로우 이름: "SOLOMOND Dual Brain Pipeline"

#### 노드 구성:
1. **Webhook 노드 추가**:
   - 노드 타입: "Webhook"
   - HTTP Method: POST
   - Path: `analysis-complete`
   - Response Mode: "Respond to Webhook"

2. **IF 노드 추가** (분석 상태 확인):
   - 조건: `{{ $json.status }} equals "completed"`

3. **HTTP Request 노드** (AI 인사이트):
   - URL: `http://localhost:8580/api/generate-insights`
   - Method: POST
   - Send Body: JSON
   - Body: `{{ $json }}`

4. **Google Calendar 노드**:
   - Resource: Event
   - Operation: Create
   - Calendar ID: primary
   - Summary: `{{ $json.analysis_title || '컨퍼런스 분석 완료' }}`
   - Description: `{{ $json.summary }}\\n\\n생성된 인사이트:\\n{{ $json.insights }}`

5. **Respond to Webhook 노드**:
   - Response: JSON
   - Body: `{ "status": "success", "message": "듀얼 브레인 워크플로우 완료" }`

#### 노드 연결:
- Webhook → IF → [HTTP Request + Google Calendar] → Respond to Webhook

### 3️⃣ **SOLOMOND File Analysis Pipeline 생성**

#### 새 워크플로우 생성:
1. "New Workflow" 클릭  
2. 워크플로우 이름: "SOLOMOND File Analysis Pipeline"

#### 노드 구성:
1. **Webhook 노드**:
   - Path: `file-upload`
   - Method: POST

2. **IF 노드** (파일 타입 확인):
   - 조건: `{{ $json.file_type }} contains "audio"`

3. **HTTP Request 노드** (오디오 분석):
   - URL: `http://localhost:8501/api/process-audio`

4. **HTTP Request 노드** (이미지 분석):
   - URL: `http://localhost:8501/api/process-image`

5. **HTTP Request 노드** (듀얼 브레인 트리거):
   - URL: `http://localhost:5678/webhook/analysis-complete`

### 4️⃣ **구글 캘린더 연동 설정**

#### Google Calendar 노드 인증:
1. Google Calendar 노드 클릭
2. "Create New Credential" 선택
3. Google OAuth2 API 설정:
   - Client ID: 구글 클라우드 콘솔에서 생성
   - Client Secret: 구글 클라우드 콘솔에서 생성
   - Scope: `https://www.googleapis.com/auth/calendar`

#### 구글 클라우드 콘솔 설정:
1. https://console.cloud.google.com 접속
2. "APIs & Services" → "Credentials" 
3. "Create Credentials" → "OAuth 2.0 Client IDs"
4. Application Type: Web Application
5. Authorized redirect URIs: `http://localhost:5678/rest/oauth2-credential/callback`

### 5️⃣ **워크플로우 활성화**
1. 각 워크플로우에서 "Active" 토글 활성화
2. "Save" 클릭하여 저장
3. 웹훅 URL 확인:
   - 분석 완료: `http://localhost:5678/webhook/analysis-complete`
   - 파일 업로드: `http://localhost:5678/webhook/file-upload`

## 🧪 **테스트 방법**

### 웹훅 테스트:
```bash
# 듀얼 브레인 파이프라인 테스트
curl -X POST "http://localhost:5678/webhook/analysis-complete" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "completed",
    "analysis_title": "테스트 분석",
    "summary": "n8n 워크플로우 테스트입니다.",
    "insights": "자동 생성된 인사이트"
  }'

# 파일 분석 파이프라인 테스트  
curl -X POST "http://localhost:5678/webhook/file-upload" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "image",
    "filename": "test.jpg",
    "file_size": 1024000
  }'
```

## ✅ **설정 완료 체크리스트**
- [ ] n8n 웹 인터페이스 접속 완료
- [ ] SOLOMOND Dual Brain Pipeline 생성 완료
- [ ] SOLOMOND File Analysis Pipeline 생성 완료  
- [ ] 구글 캘린더 OAuth 인증 완료
- [ ] 워크플로우 활성화 완료
- [ ] 웹훅 URL 테스트 성공

## 🔄 **다음 단계**
설정 완료 후 `python test_n8n_setup.py`를 다시 실행하여 모든 기능이 정상 작동하는지 확인하세요.