# 레거시 파일 백업 폴더

이 폴더에는 솔로몬드 AI 시스템 개발 과정에서 생성된 구 버전 파일들이 백업되어 있습니다.

## 📁 폴더 구조

### ui_versions/
- **jewelry_stt_ui.py** - 초기 버전
- **jewelry_stt_ui_v2.1.1.py** - v2.1.1 버전
- **jewelry_stt_ui_v213.py** - v2.13 버전  
- **jewelry_stt_ui_v214_*.py** - v2.14 시리즈
- **jewelry_stt_ui_v23_hotfix.py** - v2.3 핫픽스
- **jewelry_stt_ui_v23_hybrid*.py** - v2.3 하이브리드 버전들

**현재 사용**: `jewelry_stt_ui_v23_real.py` (메인 디렉토리)

### requirements/
다양한 개발 단계에서 생성된 의존성 파일들:
- **requirements.txt** - 기본 버전
- **requirements_enhanced*.txt** - 강화 버전들
- **requirements_v2*.txt** - v2 시리즈
- **requirements_windows*.txt** - 윈도우 전용 버전들

**현재 사용**: `requirements_v23_windows.txt` (메인 디렉토리)

### test_files/
개발 과정에서 생성된 다양한 테스트 파일들:
- **test_real_ocr.py** - OCR 테스트
- **test_api.py** - API 테스트
- **test_hybrid_integration.py** - 통합 테스트
- **test_document_*.py** - 문서 처리 테스트
- **test_youtube_*.py** - YouTube 처리 테스트
- 기타 특수 목적 테스트들

**현재 사용 중인 테스트들** (메인 디렉토리):
- `test_real_analysis.py` - 실제 분석 테스트
- `test_message_extraction.py` - 메시지 추출 테스트
- `test_real.py` - 실제 시스템 테스트
- `test_whisper_stt.py` - Whisper STT 테스트
- `test_easyocr.py` - EasyOCR 테스트
- `test_enhanced_m4a.py` - 강화된 M4A 처리 테스트
- `test_m4a_conversion.py` - M4A 변환 테스트

## ⚠️ 중요 사항

- 이 파일들은 **안전을 위해 백업**된 것입니다
- 필요시 언제든지 메인 디렉토리로 복원 가능합니다
- **삭제하지 마세요** - 디버깅이나 기능 복원시 필요할 수 있습니다

## 🗑️ 정리 기준

**백업된 파일들**:
- 더 이상 사용되지 않는 구 버전 UI
- 중복되는 requirements 파일들
- 특수 목적이 끝난 일회성 테스트들
- 개발 중 생성된 실험적 코드들

**메인에 유지된 파일들**:
- 현재 운영 중인 메인 UI (`jewelry_stt_ui_v23_real.py`)
- 현재 사용 중인 의존성 (`requirements_v23_windows.txt`)
- 핵심 기능 테스트 파일들
- 지속적으로 사용될 유틸리티들

---

**백업 일시**: 2025-07-22  
**백업 이유**: 시스템 정리 및 유지보수성 향상  
**복원 방법**: 필요한 파일을 메인 디렉토리로 `mv` 또는 `cp` 명령어로 이동