# 🔗 SOLOMOND AI n8n 워크플로우 다음 단계

## 📋 현재 상황
- ✅ n8n 서버 정상 작동 (포트 5678)
- ✅ 첫 번째 웹훅 노드 생성 완료 (POST /analysis-complete)
- ⏳ 추가 노드들 수동 추가 필요

## 🚀 우선순위별 다음 단계

### 1️⃣ **즉시 완료해야 할 작업** (예상 10분)

#### n8n 웹 인터페이스에서 노드 추가:
1. **현재 브라우저**: http://localhost:5678 접속 중
2. **웹훅 노드 우측** + 버튼 클릭
3. **IF 노드 추가**:
   - 조건: `{{ $json.status }}` equals `completed`
4. **HTTP Request 노드 2개 추가**:
   - AI 인사이트: `http://localhost:8580/api/generate-insights`
   - 구글 캘린더: Google Calendar 노드 사용
5. **Respond to Webhook 노드 추가**

### 2️⃣ **워크플로우 저장 및 활성화** (예상 5분)

1. 워크플로우 이름 변경: "SOLOMOND Dual Brain Pipeline"
2. **Save** 버튼 클릭
3. **Active** 토글 활성화
4. 웹훅 URL 확인: `http://localhost:5678/webhook/analysis-complete`

### 3️⃣ **즉시 테스트 가능** (예상 5분)

```bash
# 터미널에서 실행
curl -X POST "http://localhost:5678/webhook/analysis-complete" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "completed",
    "analysis_title": "테스트 분석",
    "summary": "n8n 워크플로우 테스트",
    "insights": "첫 번째 성공적인 테스트"
  }'
```

### 4️⃣ **SOLOMOND 시스템 연동** (예상 10분)

#### conference_analysis_COMPLETE_WORKING.py 수정:
```python
# 분석 완료 시 n8n 웹훅 호출
import requests

def trigger_dual_brain_workflow(analysis_data):
    webhook_url = "http://localhost:5678/webhook/analysis-complete"
    try:
        response = requests.post(webhook_url, json=analysis_data)
        return response.json()
    except Exception as e:
        print(f"n8n 워크플로우 트리거 실패: {e}")
        return None
```

### 5️⃣ **구글 캘린더 연동** (예상 15분)

1. **Google Cloud Console 설정**:
   - https://console.cloud.google.com 접속
   - APIs & Services → Credentials
   - OAuth 2.0 Client ID 생성
   - Redirect URI: `http://localhost:5678/rest/oauth2-credential/callback`

2. **n8n에서 인증 설정**:
   - Google Calendar 노드 클릭
   - Create New Credential 선택
   - Client ID, Secret 입력
   - 인증 완료

## 📊 완료 예상 시간

| 단계 | 작업 | 예상 시간 | 우선순위 |
|------|------|-----------|----------|
| 1 | 노드 추가 | 10분 | 🔥 높음 |
| 2 | 저장/활성화 | 5분 | 🔥 높음 |
| 3 | 기본 테스트 | 5분 | 🔥 높음 |
| 4 | SOLOMOND 연동 | 10분 | 📊 중간 |
| 5 | 구글 캘린더 | 15분 | 📊 중간 |
| **총합** | **전체 완성** | **45분** | **완료 가능** |

## 🎯 즉시 시작하세요!

**다음 명령어로 현재 상태 확인:**
```bash
# n8n 상태 확인
curl http://localhost:5678/healthz

# 테스트 스크립트 실행
python test_n8n_setup.py
```

**브라우저에서 즉시 진행:**
- http://localhost:5678 → 워크플로우 에디터에서 노드 추가 시작

---
**이 파일 위치**: `C:\Users\PC_58410\solomond-ai-system\N8N_NEXT_STEPS.md`