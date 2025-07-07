# 🚀 솔로몬드 AI 시스템 v3.0 - 모듈화 구조

## 📁 디렉토리 구조

```
solomond-ai-system/
├── config/              # 시스템 설정
│   ├── __init__.py
│   ├── settings.py      # 중앙 설정 관리
│   └── constants.py     # 상수 정의
├── core/                # 핵심 비즈니스 로직
│   ├── __init__.py
│   ├── analyzer.py      # AI 분석 엔진
│   ├── file_processor.py # 파일 처리
│   └── workflow.py      # 4단계 워크플로우
├── api/                 # API 계층
│   ├── __init__.py
│   ├── app.py          # FastAPI 앱 팩토리
│   ├── routes.py       # API 라우트
│   └── middleware.py   # 미들웨어
├── ui/                  # 사용자 인터페이스
│   ├── __init__.py
│   ├── templates.py    # HTML 템플릿
│   └── components.py   # UI 컴포넌트
├── utils/               # 유틸리티
│   ├── __init__.py
│   ├── memory.py       # 메모리 관리
│   └── logger.py       # 로깅 시스템
├── tests/               # 테스트
├── logs/                # 로그 파일
├── temp/                # 임시 파일
├── main.py             # 메인 진입점
└── requirements.txt    # 의존성
```

## 🎯 모듈화 장점

### 1. 개발 효율성
- **모듈별 독립 개발**: 각 기능을 별도로 개발 및 테스트
- **코드 재사용성**: 다른 프로젝트에서 모듈 재활용 가능
- **유지보수성**: 특정 기능 수정 시 해당 모듈만 변경

### 2. 확장성
- **새 기능 추가**: 새 모듈로 기능 확장 용이
- **플러그인 구조**: 선택적 기능 활성화/비활성화
- **API 확장**: RESTful API 쉽게 확장 가능

### 3. 품질 보증
- **단위 테스트**: 모듈별 독립 테스트
- **의존성 관리**: 명확한 모듈 간 의존성
- **에러 격리**: 한 모듈 오류가 전체에 영향 최소화

## 🚀 빠른 시작

```bash
# 1. 저장소 클론
git clone https://github.com/GeunHyeog/solomond-ai-system.git
cd solomond-ai-system

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 시스템 시작
python main.py
```

## 🔧 개발 가이드

### 새 모듈 추가
1. 적절한 디렉토리에 모듈 파일 생성
2. `__init__.py`에 exports 추가
3. 필요한 의존성 설정
4. 테스트 코드 작성

### 설정 변경
- `config/settings.py`에서 모든 설정 관리
- 환경별 설정은 환경변수로 오버라이드

### 로깅
```python
from utils import get_logger

logger = get_logger(__name__)
logger.info("로그 메시지")
```

### 메모리 관리
```python
from utils import get_memory_manager

memory_manager = get_memory_manager()
memory_manager.cleanup()
```

## 📊 성능 최적화

- **메모리 효율성**: 자동 메모리 정리 및 모니터링
- **비동기 처리**: AsyncIO 기반 파일 처리
- **캐싱**: 결과 캐싱으로 중복 처리 방지
- **모듈화**: 필요한 기능만 로드

## 🔐 보안

- **파일 검증**: 업로드 파일 형식 및 크기 검증
- **메모리 격리**: 프로세스별 메모리 제한
- **로그 보안**: 민감 정보 로깅 방지
- **CORS 설정**: API 접근 제한

---

**개발자**: 전근혁 (솔로몬드 대표)  
**버전**: v3.0 (MCP 통합)  
**라이선스**: MIT License