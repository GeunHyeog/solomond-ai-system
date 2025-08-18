#!/usr/bin/env python3
"""
🔄 자동 프로세스 관리 시스템
프로세스 자동 재시작, 헬스체크, 장애 복구 시스템
"""

import psutil
import subprocess
import time
import json
import threading
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

class ProcessStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

@dataclass
class ProcessInfo:
    """프로세스 정보 클래스"""
    name: str
    module_type: str
    file_path: str
    port: int
    pid: Optional[int] = None
    status: ProcessStatus = ProcessStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    last_restart: Optional[datetime] = None
    restart_count: int = 0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

class ProcessManager:
    """자동 프로세스 관리 클래스"""
    
    def __init__(self, health_check_interval: int = 60):
        self.health_check_interval = health_check_interval
        self.config_file = Path(__file__).parent / "process_config.json"
        self.log_file = Path(__file__).parent / "process_manager.log"
        self.status_file = Path(__file__).parent / "process_status.json"
        
        self.processes: Dict[str, ProcessInfo] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 설정
        self.settings = {
            'max_restart_attempts': 3,
            'restart_cooldown_minutes': 5,
            'health_check_timeout': 10,
            'memory_limit_mb': 1000,
            'cpu_limit_percent': 80,
            'auto_restart_on_crash': True,
            'auto_restart_on_high_memory': True
        }
        
        self.load_config()
        self.discover_processes()
    
    def load_config(self):
        """설정 로드"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.settings.update(data.get('settings', {}))
                    
                    # 프로세스 정보 로드
                    for name, proc_data in data.get('processes', {}).items():
                        self.processes[name] = ProcessInfo(
                            name=proc_data['name'],
                            module_type=proc_data['module_type'],
                            file_path=proc_data['file_path'],
                            port=proc_data['port'],
                            status=ProcessStatus(proc_data.get('status', 'unknown')),
                            restart_count=proc_data.get('restart_count', 0)
                        )
            except Exception as e:
                self.log(f"설정 로드 실패: {e}")
    
    def save_config(self):
        """설정 저장"""
        try:
            data = {
                'settings': self.settings,
                'processes': {},
                'last_updated': datetime.now().isoformat()
            }
            
            for name, proc in self.processes.items():
                data['processes'][name] = {
                    'name': proc.name,
                    'module_type': proc.module_type,
                    'file_path': proc.file_path,
                    'port': proc.port,
                    'status': proc.status.value,
                    'restart_count': proc.restart_count,
                    'last_restart': proc.last_restart.isoformat() if proc.last_restart else None
                }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"설정 저장 실패: {e}")
    
    def save_status(self):
        """현재 상태 저장"""
        try:
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'processes': {},
                'summary': {
                    'total_processes': len(self.processes),
                    'running_processes': sum(1 for p in self.processes.values() if p.status == ProcessStatus.RUNNING),
                    'crashed_processes': sum(1 for p in self.processes.values() if p.status == ProcessStatus.CRASHED),
                    'total_memory_mb': sum(p.memory_mb for p in self.processes.values()),
                    'monitoring_active': self.is_monitoring
                }
            }
            
            for name, proc in self.processes.items():
                status_data['processes'][name] = {
                    'status': proc.status.value,
                    'pid': proc.pid,
                    'port': proc.port,
                    'memory_mb': proc.memory_mb,
                    'cpu_percent': proc.cpu_percent,
                    'response_time_ms': proc.response_time_ms,
                    'last_health_check': proc.last_health_check.isoformat() if proc.last_health_check else None,
                    'restart_count': proc.restart_count,
                    'error_message': proc.error_message
                }
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"상태 저장 실패: {e}")
    
    def log(self, message: str, level: str = "INFO"):
        """로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] [{level}] {message}\n")
            print(f"[{level}] {message}")  # 콘솔에도 출력
        except Exception:
            pass
    
    def discover_processes(self):
        """실행 중인 Streamlit 프로세스 자동 발견"""
        # 표준 모듈 정의
        standard_modules = {
            8500: ("main_dashboard", "메인 대시보드", "solomond_ai_main_dashboard.py"),
            8501: ("conference_analysis", "컨퍼런스 분석", "modules/module1_conference/conference_analysis_enhanced.py"),
            8502: ("web_crawler", "웹 크롤러", "modules/module2_crawler/web_crawler_main.py"),
            8503: ("gemstone_analyzer", "보석 분석", "modules/module3_gemstone/gemstone_analyzer.py"),
            8504: ("image_to_cad", "3D CAD 변환", "modules/module4_3d_cad/image_to_cad.py"),
            8505: ("performance_optimized", "성능 최적화", "modules/module1_conference/conference_analysis_performance_optimized.py")
        }
        
        # 현재 실행 중인 프로세스 스캔
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
                
                if 'streamlit' in cmdline.lower():
                    # 포트 찾기
                    port = None
                    try:
                        for conn in proc.connections():
                            if (conn.status == psutil.CONN_LISTEN and 
                                conn.laddr and 
                                8500 <= conn.laddr.port <= 8600):
                                port = conn.laddr.port
                                break
                    except Exception:
                        continue
                    
                    if port:
                        # 표준 모듈과 매칭
                        if port in standard_modules:
                            module_type, name, file_path = standard_modules[port]
                        else:
                            module_type = f"unknown_{port}"
                            name = f"Unknown Module (:{port})"
                            file_path = ""
                        
                        process_key = f"{module_type}_{port}"
                        
                        if process_key not in self.processes:
                            self.processes[process_key] = ProcessInfo(
                                name=name,
                                module_type=module_type,
                                file_path=file_path,
                                port=port,
                                pid=proc.info['pid'],
                                status=ProcessStatus.RUNNING
                            )
                            self.log(f"프로세스 발견: {name} (PID: {proc.info['pid']}, Port: {port})")
                        else:
                            # 기존 프로세스 정보 업데이트
                            self.processes[process_key].pid = proc.info['pid']
                            self.processes[process_key].status = ProcessStatus.RUNNING
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def check_process_health(self, process_info: ProcessInfo) -> bool:
        """프로세스 헬스체크"""
        try:
            # 1. PID 존재 확인
            if process_info.pid:
                if not psutil.pid_exists(process_info.pid):
                    process_info.status = ProcessStatus.CRASHED
                    process_info.error_message = "프로세스가 종료됨"
                    return False
                
                # 2. 프로세스 정보 업데이트
                try:
                    proc = psutil.Process(process_info.pid)
                    process_info.memory_mb = round(proc.memory_info().rss / 1024 / 1024, 1)
                    process_info.cpu_percent = proc.cpu_percent()
                except Exception:
                    pass
            
            # 3. HTTP 응답 확인
            start_time = time.time()
            try:
                response = requests.get(
                    f"http://localhost:{process_info.port}",
                    timeout=self.settings['health_check_timeout']
                )
                response_time = (time.time() - start_time) * 1000
                process_info.response_time_ms = round(response_time, 1)
                
                if response.status_code == 200:
                    process_info.status = ProcessStatus.RUNNING
                    process_info.error_message = None
                    process_info.last_health_check = datetime.now()
                    return True
                else:
                    process_info.status = ProcessStatus.CRASHED
                    process_info.error_message = f"HTTP {response.status_code}"
                    return False
                    
            except requests.exceptions.RequestException as e:
                process_info.status = ProcessStatus.CRASHED
                process_info.error_message = f"HTTP 요청 실패: {str(e)[:100]}"
                return False
        
        except Exception as e:
            process_info.status = ProcessStatus.UNKNOWN
            process_info.error_message = f"헬스체크 오류: {str(e)[:100]}"
            return False
    
    def restart_process(self, process_key: str) -> bool:
        """프로세스 재시작"""
        if process_key not in self.processes:
            return False
        
        process_info = self.processes[process_key]
        
        # 재시작 횟수 제한 확인
        if process_info.restart_count >= self.settings['max_restart_attempts']:
            self.log(f"재시작 한도 초과: {process_info.name} ({process_info.restart_count}회)")
            return False
        
        # 쿨다운 시간 확인
        if (process_info.last_restart and 
            datetime.now() - process_info.last_restart < timedelta(minutes=self.settings['restart_cooldown_minutes'])):
            self.log(f"재시작 쿨다운 중: {process_info.name}")
            return False
        
        self.log(f"프로세스 재시작 시작: {process_info.name}")
        
        try:
            # 1. 기존 프로세스 종료
            if process_info.pid and psutil.pid_exists(process_info.pid):
                try:
                    proc = psutil.Process(process_info.pid)
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    pass
            
            process_info.status = ProcessStatus.STARTING
            
            # 2. 잠시 대기
            time.sleep(3)
            
            # 3. 새 프로세스 시작
            project_root = Path(__file__).parent.parent
            file_path = project_root / process_info.file_path
            
            if not file_path.exists():
                process_info.status = ProcessStatus.CRASHED
                process_info.error_message = f"파일 없음: {process_info.file_path}"
                return False
            
            command = [
                "streamlit", "run",
                str(file_path),
                "--server.port", str(process_info.port),
                "--server.headless", "true"
            ]
            
            subprocess_proc = subprocess.Popen(command, cwd=str(project_root))
            
            # 4. 시작 확인
            time.sleep(5)
            success = self.check_process_health(process_info)
            
            if success:
                process_info.pid = subprocess_proc.pid
                process_info.restart_count += 1
                process_info.last_restart = datetime.now()
                self.log(f"프로세스 재시작 성공: {process_info.name} (PID: {subprocess_proc.pid})")
                return True
            else:
                process_info.status = ProcessStatus.CRASHED
                self.log(f"프로세스 재시작 실패: {process_info.name}")
                return False
                
        except Exception as e:
            process_info.status = ProcessStatus.CRASHED
            process_info.error_message = f"재시작 오류: {str(e)[:100]}"
            self.log(f"프로세스 재시작 오류 {process_info.name}: {e}")
            return False
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.log("프로세스 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.log("프로세스 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                self.log("헬스체크 실행 중...")
                
                for process_key, process_info in self.processes.items():
                    # 헬스체크
                    is_healthy = self.check_process_health(process_info)
                    
                    # 자동 재시작 조건 확인
                    needs_restart = False
                    restart_reason = ""
                    
                    if not is_healthy and self.settings['auto_restart_on_crash']:
                        needs_restart = True
                        restart_reason = "크래시 감지"
                    
                    elif (self.settings['auto_restart_on_high_memory'] and 
                          process_info.memory_mb > self.settings['memory_limit_mb']):
                        needs_restart = True
                        restart_reason = f"높은 메모리 사용량 ({process_info.memory_mb}MB)"
                    
                    elif process_info.cpu_percent > self.settings['cpu_limit_percent']:
                        needs_restart = True
                        restart_reason = f"높은 CPU 사용량 ({process_info.cpu_percent}%)"
                    
                    # 재시작 실행
                    if needs_restart:
                        self.log(f"자동 재시작 트리거: {process_info.name} - {restart_reason}")
                        self.restart_process(process_key)
                
                # 상태 저장
                self.save_status()
                self.save_config()
                
                # 대기
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.log(f"모니터링 루프 오류: {e}", "ERROR")
                time.sleep(60)  # 오류시 1분 대기
    
    def get_status_report(self) -> Dict:
        """상태 리포트 생성"""
        running_count = sum(1 for p in self.processes.values() if p.status == ProcessStatus.RUNNING)
        crashed_count = sum(1 for p in self.processes.values() if p.status == ProcessStatus.CRASHED)
        total_memory = sum(p.memory_mb for p in self.processes.values())
        total_restarts = sum(p.restart_count for p in self.processes.values())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.is_monitoring,
            'summary': {
                'total_processes': len(self.processes),
                'running_processes': running_count,
                'crashed_processes': crashed_count,
                'total_memory_mb': round(total_memory, 1),
                'total_restarts': total_restarts
            },
            'processes': {
                name: {
                    'name': proc.name,
                    'status': proc.status.value,
                    'port': proc.port,
                    'pid': proc.pid,
                    'memory_mb': proc.memory_mb,
                    'cpu_percent': proc.cpu_percent,
                    'response_time_ms': proc.response_time_ms,
                    'restart_count': proc.restart_count,
                    'last_health_check': proc.last_health_check.isoformat() if proc.last_health_check else None,
                    'error_message': proc.error_message
                }
                for name, proc in self.processes.items()
            },
            'settings': self.settings
        }
    
    def manual_restart_all_crashed(self) -> Dict[str, bool]:
        """크래시된 모든 프로세스 수동 재시작"""
        results = {}
        
        crashed_processes = [
            (name, proc) for name, proc in self.processes.items()
            if proc.status == ProcessStatus.CRASHED
        ]
        
        if not crashed_processes:
            return {'message': '크래시된 프로세스 없음'}
        
        for name, proc in crashed_processes:
            self.log(f"수동 재시작: {proc.name}")
            success = self.restart_process(name)
            results[name] = success
        
        return results

def main():
    """테스트 실행"""
    manager = ProcessManager()
    
    print("Process Manager Test")
    print("=" * 50)
    
    # 프로세스 발견
    manager.discover_processes()
    
    # 상태 리포트
    report = manager.get_status_report()
    
    print(f"Total Processes: {report['summary']['total_processes']}")
    print(f"Running: {report['summary']['running_processes']}")
    print(f"Crashed: {report['summary']['crashed_processes']}")
    print(f"Total Memory: {report['summary']['total_memory_mb']} MB")
    
    print(f"\nProcesses:")
    for name, proc in report['processes'].items():
        status_icon = "🟢" if proc['status'] == 'running' else "🔴"
        print(f"  {status_icon} {proc['name']} (:{proc['port']}) - {proc['memory_mb']} MB - {proc['status']}")
    
    # 모니터링 시작 (테스트용으로 짧게)
    print(f"\nStarting monitoring for 30 seconds...")
    manager.start_monitoring()
    time.sleep(30)
    manager.stop_monitoring()
    
    print(f"Monitoring test completed")

if __name__ == "__main__":
    main()