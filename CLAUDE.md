# 🎯 솔로몬드 AI 시스템 - Claude Code 컨텍스트

## 📋 프로젝트 개요
**솔로몬드 AI v2.3** - 실제 AI 분석 시스템 (주얼리 전문)
- **주요 기능**: 음성(Whisper STT) + 이미지(EasyOCR) + AI 요약(Transformers)
- **현재 상태**: 실제 분석 시스템 완성, 브라우저 모니터링 통합
- **최신 업데이트**: 2025-07-20

## 🚀 현재 개발 단계

### ✅ **완료된 단계**
1. **가짜 → 실제 분석 전환 완료** (2025-07-20)
   - Whisper STT, EasyOCR, Transformers 통합
   - CPU 모드 강제 설정으로 GPU 메모리 문제 해결
   - 멀티파일 배치 분석 기능 구현

2. **윈도우 브라우저 모니터링 시스템** (2025-07-20)
   - `windows_demo_monitor.py` - 윈도우 실행 모니터링
   - `demo_capture_system.py` - Playwright 기반 캐쳐
   - WSL-윈도우 데이터 동기화 구현

3. **디버깅 및 에러 관리 시스템** (2025-07-20)
   - `collect_debug_info.py` - 자동 디버깅 정보 수집
   - `DEBUG_REPORT.md` - 에러 보고 템플릿
   - 실시간 시스템 상태 모니터링

4. **MCP 통합 확장** (2025-07-20)
   - Playwright MCP 서버 설정 완료
   - Memory, Filesystem, GitHub MCP 활용 중
   - 브라우저 자동화 준비 완료

### 🔄 **현재 진행 중**
- **Playwright MCP 활성화**: Claude Code 재시작 후 브라우저 자동화
- **음성 분석 최적화**: m4a 파일 호환성 문제 해결
- **지속적 메모리 관리**: 개발 단계 자동 추적

### 📅 **다음 예정 단계**
1. **Playwright MCP 실전 활용** - 완전 자동 브라우저 모니터링
2. **음성 파일 포맷 변환** - m4a → wav 자동 변환
3. **성능 최적화** - 대용량 파일 처리 개선

## 🧪 **최근 테스트 결과** (2025-07-20)

### ✅ **성공 사례**
- **이미지 분석**: 23개 파일 → EasyOCR 완벽 텍스트 블록 추출
- **브라우저 모니터링**: 50개 스크린샷 + JSON 리포트 생성
- **시스템 안정성**: CPU 모드에서 안정적 작동

### ❌ **해결 필요**
- **음성 분석**: m4a 파일 2개 → Whisper STT 처리 실패
- **원인**: 파일 포맷 호환성 (FFmpeg 정상, Whisper 테스트 성공)
- **해결책**: m4a → wav 변환 또는 코덱 설정 조정

## 🛠️ **핵심 시스템 구성**

### 📁 **주요 파일들**
```
solomond-ai-system/
├── jewelry_stt_ui_v23_real.py          # 메인 Streamlit UI
├── core/real_analysis_engine.py        # 실제 분석 엔진
├── windows_demo_monitor.py             # 윈도우 모니터링
├── demo_capture_system.py              # Playwright 캐쳐
├── collect_debug_info.py               # 디버깅 수집
└── CLAUDE.md                           # 이 파일
```

### 🔧 **설정 파일들**
- **MCP 설정**: `~/.config/claude/claude_desktop_config.json`
- **의존성**: `requirements_v23_windows.txt`
- **환경 변수**: `CUDA_VISIBLE_DEVICES=''` (CPU 모드)

## 🎯 **개발 패턴 및 워크플로우**

### 📋 **표준 개발 순서**
1. **문제 식별** → TodoWrite로 작업 계획
2. **코드 구현** → 점진적 개발 및 테스트
3. **에러 디버깅** → collect_debug_info.py 활용
4. **메모리 저장** → MCP Memory에 컨텍스트 기록
5. **Git 커밋** → 의미있는 커밋 메시지로 저장
6. **문서 업데이트** → CLAUDE.md 컨텍스트 갱신

### 🧠 **메모리 관리 전략**
- **엔티티 생성**: 각 개발 세션을 별도 엔티티로 관리
- **관계 매핑**: 시스템 간 의존성 관계 추적
- **상태 추적**: 성공/실패 상태 및 해결 방안 기록

## 🔍 **디버깅 정보 수집 방법**

### 🚨 **에러 발생 시 실행 순서**
```bash
# 1. 자동 디버깅 정보 수집
python3 collect_debug_info.py

# 2. Streamlit 상태 확인
ps aux | grep streamlit

# 3. 윈도우 모니터링 (윈도우에서)
start_windows_monitor.bat

# 4. 메모리에 상황 기록
# MCP Memory 자동 활용
```

## 📊 **성능 메트릭 및 벤치마크**

### 📈 **현재 성능 지표**
- **이미지 OCR**: ~3초/파일, 평균 신뢰도 85%+
- **음성 STT**: WAV 파일 기준 ~15초/분
- **시스템 메모리**: 2.2GB 사용 (6.7GB 중)
- **GPU 상태**: CPU 모드 강제 (CUDA_VISIBLE_DEVICES='')

## 🔗 **외부 통합 상태**

### ✅ **활성 MCP 서버들**
- **Memory**: 지식 그래프 관리
- **Filesystem**: 파일 시스템 접근
- **GitHub**: 코드 저장소 관리
- **Playwright**: 브라우저 자동화 (설정 완료)
- **Perplexity**: 외부 정보 검색
- **Sequential Thinking**: 구조화된 문제 해결

### 🌐 **GitHub 저장소**
- **URL**: https://github.com/GeunHyeog/solomond-ai-system
- **최신 커밋**: `73521a1` - 브라우저 모니터링 시스템 완성
- **브랜치**: `main`
- **상태**: 최신 동기화 완료

## 💡 **재접속 시 확인사항**

### 🔍 **상태 점검 체크리스트**
1. **Streamlit 실행 상태**: `ps aux | grep streamlit`
2. **MCP 서버 연결**: 사용 가능한 mcp__* 함수 확인
3. **GitHub 동기화**: `git status && git log --oneline -3`
4. **메모리 상태**: MCP Memory 검색으로 최신 컨텍스트 확인
5. **시스템 리소스**: `free -h && nvidia-smi` (선택적)

### 🚀 **즉시 재개 명령어**
```bash
# Streamlit 재시작 (필요시)
python3 -m streamlit run jewelry_stt_ui_v23_real.py --server.port 8503

# 최신 메모리 검색
# MCP Memory: "솔로몬드 AI 시스템" 검색

# 개발 상태 확인
git log --oneline -5
```

## 📚 **문서 및 가이드**
- **사용자 가이드**: `README_Windows_Monitor.md`
- **디버깅 가이드**: `DEBUG_REPORT.md`
- **설치 가이드**: `README.md`

## 🔄 **자동 업데이트 메커니즘**
이 파일은 주요 개발 단계마다 자동으로 업데이트되어 항상 최신 상태를 유지합니다.


---
**Last Updated**: 2025-07-20 02:18 KST  
**Version**: v2.3-dev  
**Status**: 브라우저 모니터링 시스템 완성 단계  
**Session ID**: 20250720_021808
**Git Status**: 92 uncommitted changes
**Next Session Goal**: 현재 개발 상태 재평가, 미완성 기능 식별