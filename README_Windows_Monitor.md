# 🖥️ 윈도우 데모 모니터링 시스템

윈도우에서 직접 실행하여 브라우저 활동을 모니터링하는 시스템입니다.

## 🚀 빠른 시작

### 1. 윈도우에서 실행
```cmd
# WSL 경로로 이동
cd \\wsl$\Ubuntu\home\solomond\claude\solomond-ai-system

# 배치 파일 실행 (권장)
start_windows_monitor.bat

# 또는 직접 Python 실행
python windows_demo_monitor.py
```

### 2. 의존성 설치 (최초 1회)
```cmd
pip install psutil pyautogui requests pillow
```

## 🎯 사용 방법

1. **Streamlit 앱 실행** (WSL에서)
   ```bash
   cd /home/solomond/claude/solomond-ai-system
   python -m streamlit run jewelry_stt_ui_v23_real.py --server.port 8503
   ```

2. **윈도우 모니터 실행**
   - `start_windows_monitor.bat` 더블클릭
   - 또는 CMD에서 `python windows_demo_monitor.py` 실행

3. **시연 진행**
   - 윈도우 브라우저에서 `http://localhost:8503` 접속
   - 📁 멀티파일 분석 탭에서 m4a, mov, jpg 파일 업로드
   - 분석 과정 시연

4. **결과 확인**
   - `windows_captures/` 폴더에 스크린샷과 리포트 저장
   - JSON 리포트로 상세 활동 분석

## 📊 모니터링 기능

### 자동 감지 항목
- ✅ **화면 캐쳐**: 3초마다 자동 스크린샷
- ✅ **브라우저 감지**: Chrome, Edge, Firefox 실행 상태
- ✅ **Streamlit 활동**: localhost:8503 접속 감지
- ✅ **윈도우 전환**: 활성 프로그램 추적
- ✅ **시스템 리소스**: CPU, 메모리 사용량

### 생성 파일
```
windows_captures/
├── screenshot_20250720_010203_001.png
├── screenshot_20250720_010203_002.png
├── ...
└── windows_session_report_20250720_010203.json
```

## 🔧 고급 설정

### 모니터링 시간 변경
```python
# windows_demo_monitor.py 실행 시 입력 프롬프트에서 설정
모니터링 시간 (분, 엔터시 기본 5분): 10
```

### 캐쳐 간격 조정
```python
# windows_demo_monitor.py 파일 수정
self.capture_interval = 3.0  # 초 단위
```

## 📁 WSL 연동

### 결과를 WSL로 복사
```cmd
# 윈도우 CMD에서
copy windows_captures\\*.* \\\\wsl$\\Ubuntu\\home\\solomond\\claude\\solomond-ai-system\\demo_captures\\
```

### 자동 동기화 (선택적)
```python
# 향후 구현 예정: 실시간 WSL 동기화
```

## 🛠️ 트러블슈팅

### 의존성 오류
```cmd
# 패키지 재설치
pip uninstall psutil pyautogui requests pillow
pip install psutil pyautogui requests pillow
```

### 권한 오류
```cmd
# 관리자 권한으로 CMD 실행
```

### Streamlit 접속 실패
```bash
# WSL에서 Streamlit 상태 확인
netstat -tulpn | grep 8503
```

## 📱 모바일/태블릿 지원

현재는 윈도우 전용입니다. 다른 플랫폼 지원:
- **Android**: Termux + Python
- **iOS**: Pythonista 앱
- **macOS**: 유사한 스크립트 수정

## 🎉 시연 결과 예시

```json
{
  "session_info": {
    "total_captures": 67,
    "duration": "201.3초",
    "platform": "windows"
  },
  "activity_summary": {
    "streamlit_interactions": 45,
    "streamlit_interaction_rate": "67.2%",
    "browser_usage": {"chrome.exe": 67},
    "unique_windows": 8
  }
}
```

## 💡 다음 단계

1. **실제 파일 시연**: m4a, mov, jpg 업로드
2. **결과 분석**: JSON 리포트 확인
3. **성능 최적화**: 캐쳐 간격 조정
4. **WSL 동기화**: 자동 결과 전송