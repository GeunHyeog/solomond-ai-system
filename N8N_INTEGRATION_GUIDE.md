# 🔗 SOLOMOND AI - n8n 완전 통합 가이드

## 🚀 현재 상황
- ✅ n8n 서버 실행 중 (http://localhost:5678)
- ✅ Google Calendar OAuth 인증 완료
- ✅ 워크플로우 JSON 파일 준비 완료
- 🔄 **현재 단계**: 워크플로우 수동 생성 및 활성화

## 📋 즉시 실행 단계

### 1️⃣ **n8n 대시보드에서 워크플로우 생성**

1. **n8n 접속**: http://localhost:5678
2. **새 워크플로우 생성**: "New workflow" 클릭
3. **JSON 가져오기**:
   - 우상단 "..." 메뉴 → "Import from file"
   - `n8n_dual_brain_workflow.json` 파일 선택
   - 워크플로우가 자동으로 생성됨

### 2️⃣ **Google Calendar 자격증명 연결**

1. **Google Calendar 노드 선택**
2. **Credential 설정**:
   - 기존에 설정한 Google Calendar credential 선택
   - 또는 "Create New" 선택하여 재설정
3. **테스트 연결**: "Test step" 버튼으로 연결 확인

### 3️⃣ **워크플로우 활성화**

1. **워크플로우 저장**: Ctrl+S 또는 "Save" 버튼
2. **활성화 토글**: 우상단 활성화 스위치 ON
3. **웹훅 URL 확인**: `http://localhost:5678/webhook/analysis-complete`

## 🧪 테스트 방법

### **A. 수동 테스트 (웹훅 직접 호출)**
```bash
curl -X POST http://localhost:5678/webhook/analysis-complete \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "test_001",
    "timestamp": "2025-08-14T10:30:00Z",
    "pre_info": {
      "conference_name": "SOLOMOND AI 테스트 컨퍼런스",
      "conference_location": "서울 테스트 센터"
    },
    "total_files": 5,
    "success_count": 4,
    "status": "completed"
  }'
```

### **B. 자동 테스트 (스크립트 실행)**
```bash
python test_workflow_integration.py
```

## 🔧 SOLOMOND AI 시스템 연동

### **컨퍼런스 분석 시스템에 웹훅 추가**

`conference_analysis_COMPLETE_WORKING.py` 파일에 다음 코드 추가:

```python
import requests

def trigger_dual_brain_workflow(analysis_data):
    """분석 완료 후 듀얼 브레인 워크플로우 트리거"""
    webhook_url = "http://localhost:5678/webhook/analysis-complete"
    
    try:
        response = requests.post(webhook_url, json=analysis_data, timeout=30)
        if response.status_code == 200:
            st.success("🧠 듀얼 브레인 워크플로우 트리거 성공!")
            return response.json()
        else:
            st.warning(f"워크플로우 트리거 실패: {response.status_code}")
    except Exception as e:
        st.error(f"워크플로우 트리거 오류: {e}")
    
    return None

# 분석 완료 시점에 추가
if analysis_complete:
    # ... 기존 분석 코드 ...
    
    # 듀얼 브레인 워크플로우 트리거
    workflow_data = {
        "analysis_id": f"solomond_{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "pre_info": pre_info,
        "total_files": len(uploaded_files),
        "success_count": successful_analyses,
        "status": "completed"
    }
    
    dual_brain_result = trigger_dual_brain_workflow(workflow_data)
    if dual_brain_result:
        st.info(f"📅 캘린더 이벤트 ID: {dual_brain_result.get('calendar_event', 'none')}")
```

## 🎯 최종 워크플로우

```
📁 컨퍼런스 파일 분석 완료
    ↓ (웹훅 트리거)
🔗 n8n 듀얼 브레인 워크플로우 시작
    ↓
📊 분석 데이터 처리 및 검증
    ↓
📅 구글 캘린더 이벤트 자동 생성
    ↓
💬 SOLOMOND AI 시스템에 결과 응답
```

## ✅ 완료 체크리스트

- [ ] n8n에서 워크플로우 JSON 가져오기
- [ ] Google Calendar 자격증명 연결
- [ ] 워크플로우 활성화
- [ ] 웹훅 URL 테스트
- [ ] 컨퍼런스 분석 시스템에 웹훅 추가
- [ ] 전체 통합 테스트 실행

## 🚨 문제 해결

### **워크플로우 활성화 실패 시**
- Google Calendar 자격증명 재확인
- n8n 로그 확인 (개발자 도구 Console)
- 워크플로우 노드별 개별 테스트

### **웹훅 응답 없음 시**
- n8n 서버 상태 확인 (http://localhost:5678/healthz)
- 워크플로우 활성화 상태 확인
- 웹훅 URL 정확성 확인

---

🎉 **완료되면**: 컨퍼런스 분석 → 구글 캘린더 자동 이벤트 생성 → 듀얼 브레인 시스템 완성!