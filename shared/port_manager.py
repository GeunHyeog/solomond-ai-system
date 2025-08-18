#!/usr/bin/env python3
"""
🔧 포트 표준화 및 관리 시스템
모든 모듈의 포트를 표준화하고 자동으로 관리하는 시스템
"""

import json
import socket
import subprocess
import psutil
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModuleConfig:
    """모듈 설정 클래스"""
    name: str
    type: str
    file_path: str
    preferred_port: int
    icon: str
    description: str
    status: str = "inactive"
    actual_port: Optional[int] = None
    pid: Optional[int] = None
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    last_started: Optional[datetime] = None

class PortManager:
    """포트 표준화 및 관리 클래스"""
    
    def __init__(self):
        self.config_file = Path(__file__).parent / "port_config.json"
        self.log_file = Path(__file__).parent / "port_manager.log"
        self.standard_modules = self._load_standard_config()
        self.load_config()
    
    def _load_standard_config(self) -> Dict[str, ModuleConfig]:
        """표준 모듈 설정 로드"""
        return {
            "main_dashboard": ModuleConfig(
                name="메인 대시보드",
                type="main_dashboard", 
                file_path="solomond_ai_main_dashboard.py",
                preferred_port=8500,
                icon="🎯",
                description="통합 관리 대시보드"
            ),
            "conference_analysis": ModuleConfig(
                name="컨퍼런스 분석",
                type="conference_analysis",
                file_path="modules/module1_conference/conference_analysis_enhanced.py", 
                preferred_port=8501,
                icon="🎤",
                description="음성 기반 화자 분리 시스템"
            ),
            "web_crawler": ModuleConfig(
                name="웹 크롤러",
                type="web_crawler",
                file_path="modules/module2_crawler/web_crawler_main.py",
                preferred_port=8502,
                icon="🕷️", 
                description="웹 크롤링 및 블로그 자동화"
            ),
            "gemstone_analyzer": ModuleConfig(
                name="보석 분석",
                type="gemstone_analyzer",
                file_path="modules/module3_gemstone/gemstone_analyzer.py",
                preferred_port=8503,
                icon="💎",
                description="AI 보석 산지 분석"
            ),
            "image_to_cad": ModuleConfig(
                name="3D CAD 변환",
                type="image_to_cad", 
                file_path="modules/module4_3d_cad/image_to_cad.py",
                preferred_port=8504,
                icon="🏗️",
                description="이미지→3D CAD 변환"
            ),
            "performance_optimized": ModuleConfig(
                name="성능 최적화 컨퍼런스",
                type="conference_performance",
                file_path="modules/module1_conference/conference_analysis_performance_optimized.py",
                preferred_port=8505,
                icon="🚀", 
                description="75% 성능 향상 버전"
            )
        }
    
    def load_config(self):
        """설정 파일 로드"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSON에서 ModuleConfig 객체로 변환
                    for key, config_data in data.items():
                        if key in self.standard_modules:
                            module = self.standard_modules[key]
                            module.actual_port = config_data.get('actual_port')
                            module.status = config_data.get('status', 'inactive')
                            module.pid = config_data.get('pid')
            except Exception as e:
                self.log(f"설정 로드 실패: {e}")
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            data = {}
            for key, module in self.standard_modules.items():
                data[key] = {
                    'name': module.name,
                    'type': module.type,
                    'preferred_port': module.preferred_port,
                    'actual_port': module.actual_port,
                    'status': module.status,
                    'pid': module.pid,
                    'last_updated': datetime.now().isoformat()
                }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"설정 저장 실패: {e}")
    
    def log(self, message: str):
        """로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass
    
    def is_port_available(self, port: int) -> bool:
        """포트 사용 가능 여부 확인"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0
        except Exception:
            return True
    
    def find_available_port(self, preferred_port: int) -> int:
        """사용 가능한 포트 찾기"""
        if self.is_port_available(preferred_port):
            return preferred_port
        
        # 선호 포트 +10까지 시도
        for offset in range(1, 11):
            port = preferred_port + offset
            if self.is_port_available(port):
                return port
        
        # 8600번대에서 찾기
        for port in range(8600, 8700):
            if self.is_port_available(port):
                return port
        
        raise Exception(f"사용 가능한 포트를 찾을 수 없습니다 (기준: {preferred_port})")
    
    def get_process_info(self, port: int) -> Optional[Dict]:
        """포트를 사용하는 프로세스 정보"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                try:
                    for conn in proc.connections():
                        if (conn.status == psutil.CONN_LISTEN and 
                            conn.laddr and 
                            conn.laddr.port == port):
                            
                            memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                            cpu_percent = proc.info['cpu_percent'] or 0.0
                            
                            return {
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': " ".join(proc.info['cmdline']) if proc.info['cmdline'] else "",
                                'memory_mb': memory_mb,
                                'cpu_percent': cpu_percent,
                                'port': port
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return None
    
    def scan_active_modules(self) -> Dict[str, ModuleConfig]:
        """현재 실행 중인 모듈 스캔"""
        active_modules = {}
        
        # 8500-8600 범위 스캔
        for port in range(8500, 8601):
            proc_info = self.get_process_info(port)
            if proc_info and 'streamlit' in proc_info['cmdline'].lower():
                
                # 표준 모듈과 매칭 시도
                module_key = None
                for key, module in self.standard_modules.items():
                    if (port == module.preferred_port or 
                        module.file_path in proc_info['cmdline']):
                        module_key = key
                        break
                
                if module_key:
                    module = self.standard_modules[module_key]
                    module.actual_port = port
                    module.status = "active"
                    module.pid = proc_info['pid']
                    module.memory_mb = proc_info['memory_mb']
                    module.cpu_percent = proc_info['cpu_percent']
                    active_modules[module_key] = module
                else:
                    # 알 수 없는 모듈
                    unknown_key = f"unknown_{port}"
                    active_modules[unknown_key] = ModuleConfig(
                        name=f"알 수 없는 모듈 (:{port})",
                        type="unknown",
                        file_path="",
                        preferred_port=port,
                        icon="❓",
                        description="식별되지 않은 Streamlit 앱",
                        status="active",
                        actual_port=port,
                        pid=proc_info['pid'],
                        memory_mb=proc_info['memory_mb'],
                        cpu_percent=proc_info['cpu_percent']
                    )
        
        return active_modules
    
    def standardize_ports(self) -> Dict[str, str]:
        """포트 표준화 실행"""
        results = {}
        active_modules = self.scan_active_modules()
        
        self.log("=== 포트 표준화 시작 ===")
        
        for module_key, module in active_modules.items():
            if module_key.startswith("unknown_"):
                continue
                
            preferred_port = module.preferred_port
            actual_port = module.actual_port
            
            if actual_port == preferred_port:
                results[module_key] = f"✅ {module.name}: 포트 {actual_port} (표준 준수)"
                self.log(f"표준 준수: {module.name} - 포트 {actual_port}")
            else:
                # 포트 변경 필요
                if self.is_port_available(preferred_port):
                    # 프로세스 재시작으로 포트 변경
                    success = self.restart_module_on_port(module_key, preferred_port)
                    if success:
                        results[module_key] = f"🔄 {module.name}: 포트 {actual_port} → {preferred_port} (표준화 완료)"
                        self.log(f"포트 변경 성공: {module.name} - {actual_port} → {preferred_port}")
                    else:
                        results[module_key] = f"❌ {module.name}: 포트 변경 실패"
                        self.log(f"포트 변경 실패: {module.name}")
                else:
                    results[module_key] = f"⚠️ {module.name}: 선호 포트 {preferred_port} 사용 중, 현재 {actual_port} 유지"
                    self.log(f"선호 포트 사용 중: {module.name} - {preferred_port}")
        
        # 설정 저장
        self.save_config()
        self.log("=== 포트 표준화 완료 ===")
        
        return results
    
    def restart_module_on_port(self, module_key: str, target_port: int) -> bool:
        """모듈을 특정 포트로 재시작"""
        try:
            module = self.standard_modules[module_key]
            
            # 기존 프로세스 종료
            if module.pid:
                try:
                    proc = psutil.Process(module.pid)
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    pass
            
            # 잠시 대기
            time.sleep(2)
            
            # 새 포트로 재시작
            project_root = Path(__file__).parent.parent
            file_path = project_root / module.file_path
            
            if not file_path.exists():
                return False
            
            command = [
                "streamlit", "run",
                str(file_path),
                "--server.port", str(target_port),
                "--server.headless", "true"
            ]
            
            process = subprocess.Popen(command, cwd=str(project_root))
            
            # 시작 확인
            time.sleep(3)
            if self.get_process_info(target_port):
                module.actual_port = target_port
                module.pid = process.pid
                module.status = "active"
                module.last_started = datetime.now()
                return True
            
        except Exception as e:
            self.log(f"모듈 재시작 실패 {module_key}: {e}")
        
        return False
    
    def get_port_status_report(self) -> Dict:
        """포트 상태 리포트 생성"""
        active_modules = self.scan_active_modules()
        
        total_memory = sum(module.memory_mb for module in active_modules.values())
        total_cpu = sum(module.cpu_percent for module in active_modules.values())
        
        standardized_count = 0
        non_standard_count = 0
        
        for module_key, module in active_modules.items():
            if module_key.startswith("unknown_"):
                continue
            if module.actual_port == module.preferred_port:
                standardized_count += 1
            else:
                non_standard_count += 1
        
        return {
            "active_modules": len(active_modules),
            "standardized_ports": standardized_count,
            "non_standard_ports": non_standard_count,
            "unknown_modules": len([k for k in active_modules.keys() if k.startswith("unknown_")]),
            "total_memory_mb": round(total_memory, 1),
            "total_cpu_percent": round(total_cpu, 1),
            "modules": active_modules,
            "scan_time": datetime.now().isoformat()
        }
    
    def terminate_module(self, module_key: str) -> bool:
        """특정 모듈 종료"""
        try:
            if module_key in self.standard_modules:
                module = self.standard_modules[module_key]
                if module.pid:
                    proc = psutil.Process(module.pid)
                    proc.terminate()
                    proc.wait(timeout=10)
                    
                    module.status = "inactive"
                    module.actual_port = None
                    module.pid = None
                    
                    self.save_config()
                    self.log(f"모듈 종료: {module.name}")
                    return True
        except Exception as e:
            self.log(f"모듈 종료 실패 {module_key}: {e}")
        
        return False

def main():
    """테스트 실행"""
    manager = PortManager()
    
    print("Port Manager Test")
    print("=" * 50)
    
    # 현재 상태 확인
    report = manager.get_port_status_report()
    print(f"Active modules: {report['active_modules']}")
    print(f"Standardized ports: {report['standardized_ports']}")
    print(f"Non-standard ports: {report['non_standard_ports']}")
    print(f"Total memory: {report['total_memory_mb']} MB")
    
    # 포트 표준화 실행
    print("\nStandardizing ports...")
    results = manager.standardize_ports()
    
    for module, result in results.items():
        print(f"  {result}")

if __name__ == "__main__":
    main()