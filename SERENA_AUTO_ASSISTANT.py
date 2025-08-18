#!/usr/bin/env python3
"""
자동화된 Serena 어시스턴트
사용자가 호출하지 않아도 알아서 SOLOMOND AI 시스템을 모니터링하고 관리
"""

import os
import time
import json
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess
import requests
import psutil

class SerenaAutoAssistant:
    """자동화된 Serena 어시스턴트"""
    
    def __init__(self):
        self.is_running = False
        self.monitoring_thread = None
        self.log_file = "serena_auto_log.json"
        self.config_file = "serena_auto_config.json"
        
        # 기본 설정
        self.config = {
            "monitoring_interval": 30,  # 30초마다 체크
            "auto_fix_enabled": True,
            "health_check_interval": 300,  # 5분마다 건강도 체크
            "performance_monitoring": True,
            "auto_optimization": True,
            "notification_enabled": True
        }
        
        self.load_config()
        self.activities = []
        
    def load_config(self):
        """설정 로드"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except:
                pass
    
    def save_config(self):
        """설정 저장"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def log_activity(self, activity_type: str, message: str, data: Dict = None):
        """활동 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "message": message,
            "data": data or {}
        }
        
        self.activities.append(log_entry)
        
        # 로그 파일에 저장
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            # 최근 1000개만 유지
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
        except:
            pass
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {activity_type}: {message}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 건강도 자동 체크"""
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('.').percent,
            "streamlit_processes": 0,
            "ports_status": {},
            "issues_found": []
        }
        
        # Streamlit 프로세스 확인
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'streamlit' in proc.info['name'].lower():
                    health_data["streamlit_processes"] += 1
            except:
                continue
        
        # 포트 상태 확인
        ports_to_check = [8500, 8501, 8502, 8503, 8504, 8520]
        for port in ports_to_check:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=3)
                health_data["ports_status"][str(port)] = {
                    "accessible": True,
                    "status_code": response.status_code
                }
            except:
                health_data["ports_status"][str(port)] = {
                    "accessible": False
                }
        
        # 이슈 탐지
        if health_data["cpu_usage"] > 90:
            health_data["issues_found"].append("HIGH_CPU_USAGE")
        
        if health_data["memory_usage"] > 85:
            health_data["issues_found"].append("HIGH_MEMORY_USAGE")
        
        if health_data["streamlit_processes"] == 0:
            health_data["issues_found"].append("NO_STREAMLIT_PROCESSES")
        
        accessible_ports = sum(1 for status in health_data["ports_status"].values() if status["accessible"])
        if accessible_ports < len(ports_to_check) / 2:
            health_data["issues_found"].append("MULTIPLE_PORT_FAILURES")
        
        return health_data
    
    def auto_fix_issues(self, health_data: Dict[str, Any]) -> List[str]:
        """이슈 자동 수정"""
        if not self.config["auto_fix_enabled"]:
            return []
        
        fixes_applied = []
        issues = health_data.get("issues_found", [])
        
        for issue in issues:
            if issue == "NO_STREAMLIT_PROCESSES":
                # 메인 대시보드 자동 시작
                try:
                    cmd = 'start cmd /k "streamlit run solomond_ai_main_dashboard.py --server.port 8500"'
                    subprocess.Popen(cmd, shell=True)
                    fixes_applied.append("Started main dashboard")
                    time.sleep(5)
                except Exception as e:
                    self.log_activity("ERROR", f"Failed to start dashboard: {e}")
            
            elif issue == "MULTIPLE_PORT_FAILURES":
                # 여러 모듈 재시작
                try:
                    subprocess.Popen("START_ALL_MODULES.bat", shell=True)
                    fixes_applied.append("Restarted all modules")
                except Exception as e:
                    self.log_activity("ERROR", f"Failed to restart modules: {e}")
            
            elif issue == "HIGH_MEMORY_USAGE":
                # 메모리 정리
                try:
                    # Python 가비지 컬렉션 강제 실행
                    import gc
                    gc.collect()
                    fixes_applied.append("Forced garbage collection")
                except:
                    pass
        
        return fixes_applied
    
    def run_serena_analysis(self) -> Dict[str, Any]:
        """Serena 분석 자동 실행"""
        try:
            # serena_quick_test.py 실행
            result = subprocess.run(
                ["python", "serena_quick_test.py"], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            analysis_result = {
                "success": result.returncode == 0,
                "output": result.stdout if result.returncode == 0 else result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_system_performance(self) -> List[str]:
        """시스템 성능 자동 최적화"""
        if not self.config["auto_optimization"]:
            return []
        
        optimizations = []
        
        try:
            # 임시 파일 정리
            temp_dirs = ["temp", "cache", "__pycache__"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                if os.path.getmtime(file_path) < time.time() - 3600:  # 1시간 이상 된 파일
                                    os.remove(file_path)
                            except:
                                continue
            optimizations.append("Cleaned temporary files")
            
        except:
            pass
        
        return optimizations
    
    def generate_smart_notifications(self, health_data: Dict[str, Any]) -> List[str]:
        """똑똑한 알림 생성"""
        notifications = []
        
        # 성능 기반 알림
        if health_data["cpu_usage"] > 80:
            notifications.append(f"⚠️ CPU 사용률 높음: {health_data['cpu_usage']:.1f}%")
        
        if health_data["memory_usage"] > 80:
            notifications.append(f"⚠️ 메모리 사용률 높음: {health_data['memory_usage']:.1f}%")
        
        # 포트 상태 알림
        failed_ports = [port for port, status in health_data["ports_status"].items() 
                       if not status["accessible"]]
        
        if failed_ports:
            notifications.append(f"🔌 접근 불가 포트: {', '.join(failed_ports)}")
        
        # 긍정적 알림
        if len(health_data["issues_found"]) == 0:
            notifications.append("✅ 모든 시스템이 정상 작동 중입니다")
        
        return notifications
    
    def monitoring_cycle(self):
        """모니터링 사이클"""
        self.log_activity("START", "Serena 자동 어시스턴트 모니터링 시작")
        
        while self.is_running:
            try:
                # 1. 시스템 건강도 체크
                health_data = self.check_system_health()
                
                # 2. 이슈가 있으면 자동 수정
                if health_data["issues_found"]:
                    self.log_activity("ISSUES", f"발견된 이슈: {', '.join(health_data['issues_found'])}")
                    
                    fixes = self.auto_fix_issues(health_data)
                    if fixes:
                        self.log_activity("AUTO_FIX", f"자동 수정 적용: {', '.join(fixes)}")
                
                # 3. 성능 최적화
                if self.config["performance_monitoring"]:
                    optimizations = self.optimize_system_performance()
                    if optimizations:
                        self.log_activity("OPTIMIZATION", f"성능 최적화: {', '.join(optimizations)}")
                
                # 4. 알림 생성
                if self.config["notification_enabled"]:
                    notifications = self.generate_smart_notifications(health_data)
                    for notification in notifications:
                        if "⚠️" in notification or "🔌" in notification:
                            self.log_activity("ALERT", notification)
                
                # 5. 주기적으로 Serena 분석 실행
                current_time = datetime.now()
                if current_time.minute % 10 == 0:  # 10분마다
                    analysis_result = self.run_serena_analysis()
                    if analysis_result["success"]:
                        self.log_activity("ANALYSIS", "Serena 자동 분석 완료")
                    else:
                        self.log_activity("ERROR", f"Serena 분석 실패: {analysis_result.get('error', 'Unknown')}")
                
            except Exception as e:
                self.log_activity("ERROR", f"모니터링 사이클 오류: {e}")
            
            # 다음 사이클까지 대기
            time.sleep(self.config["monitoring_interval"])
    
    def start(self):
        """자동 어시스턴트 시작"""
        if self.is_running:
            print("이미 실행 중입니다.")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_cycle, daemon=True)
        self.monitoring_thread.start()
        
        print("🤖 Serena 자동 어시스턴트가 시작되었습니다!")
        print(f"   - 모니터링 간격: {self.config['monitoring_interval']}초")
        print(f"   - 자동 수정: {'활성화' if self.config['auto_fix_enabled'] else '비활성화'}")
        print(f"   - 성능 모니터링: {'활성화' if self.config['performance_monitoring'] else '비활성화'}")
        print("   - 로그: serena_auto_log.json에서 확인 가능")
    
    def stop(self):
        """자동 어시스턴트 중지"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.log_activity("STOP", "Serena 자동 어시스턴트 중지")
        print("🛑 Serena 자동 어시스턴트가 중지되었습니다.")
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "is_running": self.is_running,
            "config": self.config,
            "recent_activities": self.activities[-10:] if self.activities else [],
            "uptime": "계산 필요" if self.is_running else "중지됨"
        }

def main():
    """메인 실행"""
    assistant = SerenaAutoAssistant()
    
    print("🤖 Serena 자동 어시스턴트")
    print("=" * 50)
    print("1. start - 자동 모니터링 시작")
    print("2. stop - 중지")
    print("3. status - 상태 확인")
    print("4. config - 설정 변경")
    print("5. exit - 종료")
    print("=" * 50)
    
    try:
        while True:
            command = input("\\n명령어 입력: ").strip().lower()
            
            if command == "start":
                assistant.start()
            elif command == "stop":
                assistant.stop()
            elif command == "status":
                status = assistant.get_status()
                print(f"실행 상태: {'실행 중' if status['is_running'] else '중지됨'}")
                if status['recent_activities']:
                    print("최근 활동:")
                    for activity in status['recent_activities'][-3:]:
                        print(f"  - {activity['timestamp'][:19]} | {activity['type']}: {activity['message']}")
            elif command == "config":
                print("현재 설정:")
                for key, value in assistant.config.items():
                    print(f"  {key}: {value}")
            elif command == "exit":
                assistant.stop()
                break
            elif command == "demo":
                # 데모 모드: 자동 시작하고 30초 후 종료
                assistant.start()
                print("데모 모드: 30초 동안 실행...")
                time.sleep(30)
                assistant.stop()
            else:
                print("알 수 없는 명령어입니다.")
    
    except KeyboardInterrupt:
        assistant.stop()
        print("\\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()