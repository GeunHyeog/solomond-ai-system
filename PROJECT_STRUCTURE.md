# 📁 프로젝트 구조 설계서

## 🏗️ 모듈화 아키텍처

### `/core/` - 핵심 AI 엔진
**목적**: 주얼리 특화 AI 처리의 중심  
**책임**: STT, NLP, 번역, 도메인 특화 분석

```
/core/
├── /stt/
│   ├── whisper_engine.py      # OpenAI Whisper 최적화
│   ├── jewelry_stt.py         # 주얼리 용어 특화 STT
│   └── stream_processor.py    # 실시간 스트리밍 처리
├── /nlp/
│   ├── analyzer.py            # 텍스트 분석 엔진
│   ├── keyword_extractor.py   # 키워드 추출
│   └── sentiment_analyzer.py  # 감정 분석
├── /jewelry/
│   ├── terms_database.py      # 주얼리 용어 DB
│   ├── grading_system.py      # 4C 등급 분석
│   └── brand_recognition.py   # 브랜드 인식
└── /translation/
    ├── multilingual.py        # 다국어 번역
    └── context_translator.py  # 문맥 기반 번역
```

### `/api/` - FastAPI 서버
**목적**: RESTful API 제공 및 비즈니스 로직 처리  
**책임**: 요청 처리, 인증, 데이터 검증

```
/api/
├── main.py                    # FastAPI 애플리케이션
├── /routers/
│   ├── stt_router.py         # STT 관련 API
│   ├── analysis_router.py    # 분석 API
│   └── auth_router.py        # 인증 API
├── /models/
│   ├── request_models.py     # 요청 데이터 모델
│   └── response_models.py    # 응답 데이터 모델
└── /middleware/
    ├── auth_middleware.py    # 인증 미들웨어
    └── cors_middleware.py    # CORS 처리
```

### `/ui/` - 사용자 인터페이스
**목적**: 사용자 경험 최적화  
**책임**: 화면 렌더링, 사용자 인터랙션

```
/ui/
├── /streamlit/               # MVP 프로토타입
│   ├── app.py               # 메인 Streamlit 앱
│   ├── /pages/              # 페이지별 구성
│   └── /components/         # 재사용 컴포넌트
└── /react/                  # 프로덕션 UI (향후)
    ├── /src/
    ├── /public/
    └── package.json
```

### `/data/` - 데이터 저장소
**목적**: 구조화된 데이터 관리  
**책임**: 용어집, 설정, 캐시

```
/data/
├── /jewelry_terms/          # 주얼리 용어 데이터
│   ├── gemstones.json       # 보석 종류
│   ├── grading_4c.json      # 4C 등급 시스템
│   └── brands.json          # 브랜드 정보
├── /languages/              # 다국어 데이터
│   ├── korean.json
│   ├── english.json
│   └── chinese.json
└── /cache/                  # 임시 저장소
    ├── processed_audio/
    └── analysis_results/
```

### `/tests/` - 자동 테스트
**목적**: 품질 보증 및 안정성 확보  
**책임**: 단위 테스트, 통합 테스트, 성능 테스트

```
/tests/
├── /unit/                   # 단위 테스트
│   ├── test_stt.py
│   ├── test_nlp.py
│   └── test_api.py
├── /integration/            # 통합 테스트
│   ├── test_workflow.py
│   └── test_performance.py
└── /fixtures/               # 테스트 데이터
    ├── sample_audio/
    └── expected_results/
```

### `/docs/` - 문서화
**목적**: 프로젝트 이해도 및 유지보수성 향상  
**책임**: 기술 문서, 사용자 가이드, API 문서

```
/docs/
├── /api/                    # API 문서
├── /user_guide/             # 사용자 가이드
├── /development/            # 개발 가이드
└── /deployment/             # 배포 가이드
```

### `/deploy/` - 배포 스크립트
**목적**: 일관된 배포 환경 제공  
**책임**: 컨테이너화, 환경 설정, 자동화

```
/deploy/
├── Dockerfile               # 컨테이너 이미지
├── docker-compose.yml       # 개발 환경
├── kubernetes/              # K8s 배포 (향후)
└── /scripts/                # 배포 스크립트
    ├── build.sh
    └── deploy.sh
```

## 🔄 연속성 보장 설계

### **상태 추적 시스템**
```python
# 각 모듈은 상태 정보를 자동으로 기록
class StateManager:
    def save_progress(self, module, progress, context):
        # Memory 시스템에 자동 저장
        pass
    
    def restore_session(self, session_id):
        # 이전 세션 상태 복원
        pass
```

### **자동 백업 프로토콜**
```yaml
# .github/workflows/auto-backup.yml
auto_backup:
  triggers:
    - push
    - schedule: "0 */2 * * *"  # 2시간마다
  actions:
    - Memory 상태 저장
    - 코드 버전 태깅
    - Notion 진행률 업데이트
```

## 📊 모듈별 책임 매트릭스

| 모듈 | 입력 | 출력 | 의존성 | 테스트 커버리지 |
|------|------|------|--------|------------------|
| `/core/stt/` | 오디오 파일 | 텍스트 | Whisper | 95%+ |
| `/core/nlp/` | 텍스트 | 분석 결과 | spaCy | 90%+ |
| `/api/` | HTTP 요청 | JSON 응답 | FastAPI | 85%+ |
| `/ui/` | 사용자 입력 | 화면 출력 | Streamlit | 80%+ |

---
> 🏗️ **설계 원칙**: 각 모듈은 독립적으로 개발, 테스트, 배포 가능  
> 🔄 **확장성**: 새로운 기능은 기존 구조를 해치지 않고 추가 가능  
> 📊 **측정 가능**: 모든 모듈의 성능과 품질을 정량적으로 측정