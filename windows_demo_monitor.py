#!/usr/bin/env python3
"""
윈도우용 데모 모니터링 시스템
윈도우에서 직접 실행하여 브라우저 활동을 모니터링
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
import base64
from config import SETTINGS

try:
    import psutil
    import pyautogui
    import requests
    WINDOWS_DEPS_AVAILABLE = True
except ImportError:
    WINDOWS_DEPS_AVAILABLE = False

class WindowsDemoMonitor:
    """윈도우용 데모 모니터링 시스템"""
    
    def __init__(self):
        self.streamlit_url = "http://f"localhost:{SETTINGS['PORT']}""
        self.capture_dir = Path("windows_captures")
        self.capture_dir.mkdir(exist_ok=True)
        
        # 캐쳐 설정
        self.capture_interval = 3.0  # 3초마다
        self.max_captures = 100
        
        # 데이터 저장
        self.captures = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🖥️ 윈도우 데모 모니터 초기화")
        print(f"📁 저장 위치: {self.capture_dir}")
        
        if not WINDOWS_DEPS_AVAILABLE:
            print("❌ 윈도우 의존성 설치 필요:")
            print("pip install psutil pyautogui requests pillow")
    
    def check_dependencies(self):
        """의존성 확인"""
        missing = []
        
        try:
            import psutil
        except ImportError:
            missing.append("psutil")
            
        try:
            import pyautogui
        except ImportError:
            missing.append("pyautogui")
            
        try:
            import requests
        except ImportError:
            missing.append("requests")
            
        if missing:
            print(f"❌ 누락된 패키지: {', '.join(missing)}")
            print("설치 명령어:")
            print(f"pip install {' '.join(missing)}")
            return False
        
        return True
    
    def get_browser_processes(self):
        """실행 중인 브라우저 프로세스 찾기"""
        browsers = []
        browser_names = ['chrome.exe', 'msedge.exe', 'firefox.exe', 'opera.exe']
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'].lower() in [b.lower() for b in browser_names]:
                    browsers.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': ' '.join(proc.info['cmdline'] or [])
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return browsers
    
    def check_streamlit_access(self):
        """Streamlit 앱 접근 확인"""
        try:
            response = requests.get(self.streamlit_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def capture_screen(self):
        """화면 캐쳐"""
        try:
            screenshot = pyautogui.screenshot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            screenshot_path = self.capture_dir / f"screenshot_{self.session_id}_{len(self.captures):03d}_{timestamp}.png"
            screenshot.save(str(screenshot_path))
            return str(screenshot_path)
        except Exception as e:
            print(f"⚠️ 스크린샷 실패: {e}")
            return None
    
    def get_system_info(self):
        """시스템 정보 수집"""
        try:
            # 활성 윈도우 정보
            try:
                import win32gui
                active_window = win32gui.GetForegroundWindow()
                window_title = win32gui.GetWindowText(active_window)
            except ImportError:
                window_title = "알 수 없음 (win32gui 미설치)"
            
            # 메모리 사용량
            memory = psutil.virtual_memory()
            
            # CPU 사용량
            cpu_percent = psutil.cpu_percent()
            
            return {
                'active_window': window_title,
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_streamlit_activity(self):
        """Streamlit 활동 감지"""
        try:
            # 간단한 접근성 확인
            streamlit_accessible = self.check_streamlit_access()
            
            # 브라우저 프로세스 확인
            browsers = self.get_browser_processes()
            
            # 활성 윈도우에서 localhost 확인
            system_info = self.get_system_info()
            window_title = system_info.get('active_window', '')
            
            streamlit_active = any([
                'localhost:8503' in window_title.lower(),
                'streamlit' in window_title.lower(),
                '솔로몬드' in window_title
            ])
            
            return {
                'streamlit_accessible': streamlit_accessible,
                'browsers_running': len(browsers),
                'browser_processes': browsers,
                'streamlit_window_active': streamlit_active,
                'active_window_title': window_title
            }
        except Exception as e:
            return {'error': str(e)}
    
    def create_capture_data(self):
        """캐쳐 데이터 생성"""
        timestamp = datetime.now().isoformat()
        
        # 스크린샷
        screenshot_path = self.capture_screen()
        
        # 시스템 정보
        system_info = self.get_system_info()
        
        # Streamlit 활동
        streamlit_activity = self.detect_streamlit_activity()
        
        capture_data = {
            'timestamp': timestamp,
            'capture_index': len(self.captures),
            'session_id': self.session_id,
            'screenshot_path': screenshot_path,
            'system_info': system_info,
            'streamlit_activity': streamlit_activity,
            'platform': 'windows'
        }
        
        return capture_data
    
    def start_monitoring(self, duration_minutes=10):
        """모니터링 시작"""
        if not self.check_dependencies():
            return None
        
        print(f"🚀 윈도우 데모 모니터링 시작 ({duration_minutes}분)")
        print("=" * 60)
        print(f"📍 Target: {self.streamlit_url}")
        print(f"📸 간격: {self.capture_interval}초")
        print(f"💾 저장: {self.capture_dir}")
        print()
        
        # 초기 상태 확인
        streamlit_ok = self.check_streamlit_access()
        browsers = self.get_browser_processes()
        
        print(f"✅ Streamlit 접근: {'가능' if streamlit_ok else '불가능'}")
        print(f"🌐 브라우저 실행: {len(browsers)}개")
        for browser in browsers[:3]:  # 처음 3개만 표시
            print(f"   - {browser['name']} (PID: {browser['pid']})")
        print()
        
        if not streamlit_ok:
            print("⚠️ Streamlit 앱에 접근할 수 없습니다.")
            print("   Streamlit이 실행 중인지 확인해주세요: http://localhost:8503")
            print()
        
        print("📸 모니터링 시작... (Ctrl+C로 중단)")
        print("💡 윈도우에서 자유롭게 Streamlit 시연을 진행하세요!")
        print()
        
        # 모니터링 루프
        end_time = time.time() + (duration_minutes * 60)
        
        try:
            while time.time() < end_time and len(self.captures) < self.max_captures:
                # 캐쳐 실행
                capture_data = self.create_capture_data()
                self.captures.append(capture_data)
                
                # 진행 상황 출력
                activity = capture_data['streamlit_activity']
                active_window = capture_data['system_info'].get('active_window', 'N/A')
                
                status_icon = "🎯" if activity.get('streamlit_window_active') else "⏸️"
                
                print(f"{status_icon} 캐쳐 {len(self.captures):03d}: {datetime.now().strftime('%H:%M:%S')} - {active_window[:50]}")
                
                # 대기
                time.sleep(self.capture_interval)
        
        except KeyboardInterrupt:
            print(f"\n⏹️ 사용자에 의해 중단됨 (캐쳐: {len(self.captures)}개)")
        
        # 세션 리포트 생성
        session_report = self.generate_session_report()
        
        print("\n" + "=" * 60)
        print("📊 윈도우 데모 모니터링 완료!")
        print("=" * 60)
        
        self.display_summary(session_report)
        
        return session_report
    
    def generate_session_report(self):
        """세션 리포트 생성"""
        if not self.captures:
            return {"error": "캐쳐된 데이터가 없습니다"}
        
        # 세션 기본 정보
        session_info = {
            'session_id': self.session_id,
            'total_captures': len(self.captures),
            'start_time': self.captures[0]['timestamp'],
            'end_time': self.captures[-1]['timestamp'],
            'platform': 'windows'
        }
        
        # 시간 계산
        if len(self.captures) >= 2:
            start = datetime.fromisoformat(self.captures[0]['timestamp'])
            end = datetime.fromisoformat(self.captures[-1]['timestamp'])
            duration = (end - start).total_seconds()
            session_info['duration'] = f"{duration:.1f}초"
        
        # 활동 분석
        streamlit_interactions = 0
        browser_usage = {}
        active_windows = []
        
        for capture in self.captures:
            # Streamlit 활동 카운트
            if capture['streamlit_activity'].get('streamlit_window_active'):
                streamlit_interactions += 1
            
            # 브라우저 사용량
            browsers = capture['streamlit_activity'].get('browser_processes', [])
            for browser in browsers:
                name = browser['name']
                browser_usage[name] = browser_usage.get(name, 0) + 1
            
            # 활성 윈도우 추적
            window = capture['system_info'].get('active_window', '')
            if window and window not in active_windows:
                active_windows.append(window)
        
        activity_summary = {
            'streamlit_interactions': streamlit_interactions,
            'streamlit_interaction_rate': f"{(streamlit_interactions/len(self.captures)*100):.1f}%",
            'browser_usage': browser_usage,
            'unique_windows': len(active_windows),
            'window_list': active_windows[:10]  # 상위 10개만
        }
        
        # 추천사항 생성
        recommendations = []
        if streamlit_interactions > len(self.captures) * 0.7:
            recommendations.append("✅ Streamlit 앱을 집중적으로 사용하셨습니다!")
        elif streamlit_interactions > 0:
            recommendations.append("🔶 Streamlit 사용이 감지되었습니다.")
        else:
            recommendations.append("💡 Streamlit 앱 사용이 감지되지 않았습니다. 브라우저에서 http://localhost:8503 접속을 확인해주세요.")
        
        if len(browser_usage) > 1:
            recommendations.append(f"🌐 {len(browser_usage)}개 브라우저 사용이 감지되었습니다.")
        
        # 전체 리포트
        session_report = {
            'session_info': session_info,
            'activity_summary': activity_summary,
            'captures': self.captures,
            'recommendations': recommendations,
            'platform_specific': {
                'windows_version': 'detected',
                'monitoring_method': 'screen_capture + process_monitoring'
            }
        }
        
        # JSON 저장
        report_path = self.capture_dir / f"windows_session_report_{self.session_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(session_report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 세션 리포트 저장: {report_path}")
        
        return session_report
    
    def display_summary(self, session_report):
        """요약 정보 표시"""
        session_info = session_report['session_info']
        activity = session_report['activity_summary']
        
        print(f"📈 캐쳐 통계:")
        print(f"   • 총 캐쳐: {session_info['total_captures']}개")
        print(f"   • 소요 시간: {session_info.get('duration', 'N/A')}")
        print(f"   • 세션 ID: {session_info['session_id']}")
        print()
        
        print(f"🎯 활동 분석:")
        print(f"   • Streamlit 상호작용: {activity['streamlit_interactions']}회 ({activity['streamlit_interaction_rate']})")
        print(f"   • 사용된 브라우저: {', '.join(activity['browser_usage'].keys()) if activity['browser_usage'] else '없음'}")
        print(f"   • 활성 윈도우 수: {activity['unique_windows']}개")
        print()
        
        print(f"💡 추천사항:")
        for rec in session_report['recommendations']:
            print(f"   {rec}")
        print()
        
        print(f"📁 저장된 파일:")
        print(f"   📊 리포트: windows_captures/windows_session_report_{self.session_id}.json")
        print(f"   📸 스크린샷: windows_captures/screenshot_{self.session_id}_*.png")
        print()
        
        # WSL 전송용 요약
        wsl_summary = f"""
🖥️ 윈도우 데모 모니터링 결과

📊 기본 정보:
- 세션 ID: {session_info['session_id']}
- 총 캐쳐: {session_info['total_captures']}개
- 시간: {session_info.get('duration', 'N/A')}
- 플랫폼: Windows

🎯 활동 요약:
- Streamlit 상호작용: {activity['streamlit_interactions']}회 ({activity['streamlit_interaction_rate']})
- 브라우저: {', '.join(activity['browser_usage'].keys()) if activity['browser_usage'] else '없음'}
- 윈도우 전환: {activity['unique_windows']}개

💡 평가: {' | '.join(session_report['recommendations'])}
"""
        
        print("🤖 WSL/Claude 전달용 요약:")
        print("-" * 50)
        print(wsl_summary.strip())
        print("-" * 50)

def main():
    """메인 실행 함수"""
    print("🖥️ 윈도우 데모 모니터링 시스템")
    print("=" * 50)
    print("💎 윈도우에서 브라우저 활동을 모니터링합니다!")
    print()
    
    # 기본 설정
    duration = 5  # 기본 5분
    
    # 사용자 입력 (선택적)
    try:
        user_input = input(f"모니터링 시간 (분, 엔터시 기본 {duration}분): ").strip()
        if user_input.isdigit():
            duration = int(user_input)
    except:
        pass
    
    print(f"⚙️ 설정: {duration}분간 모니터링")
    print()
    
    # 모니터 시작
    monitor = WindowsDemoMonitor()
    session_report = monitor.start_monitoring(duration_minutes=duration)
    
    if session_report:
        print("\n🎉 모니터링 완료!")
        print("📋 WSL로 데이터를 전송하려면 다음 명령어를 사용하세요:")
        print(f"   copy windows_captures\\*.* \\\\wsl$\\Ubuntu\\home\\solomond\\claude\\solomond-ai-system\\demo_captures\\")
    else:
        print("\n❌ 모니터링 실패!")

if __name__ == "__main__":
    main()