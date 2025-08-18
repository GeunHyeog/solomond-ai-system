#!/usr/bin/env python3
"""
🔍 동적 포트 감지 시스템
메인 대시보드가 각 모듈의 실제 실행 포트를 자동으로 감지하는 시스템
"""

import socket
import requests
import subprocess
import json
import os
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import psutil

class PortDetector:
    """동적 포트 감지 및 모듈 상태 관리 클래스"""
    
    def __init__(self):
        self.base_port_range = (8500, 8600)  # 검색할 포트 범위
        self.timeout = 3  # HTTP 요청 타임아웃
        self.cache_file = Path(__file__).parent / "port_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """포트 캐시 로드"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.port_cache = json.load(f)
            else:
                self.port_cache = {}
        except Exception:
            self.port_cache = {}
    
    def save_cache(self):
        """포트 캐시 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.port_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def get_active_ports(self) -> List[int]:
        """현재 활성화된 포트 목록 반환"""
        active_ports = []
        
        try:
            # netstat를 사용해서 LISTENING 포트 확인
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            for line in result.stdout.split('\n'):
                if 'LISTENING' in line and ':85' in line:  # 8500번대 포트만
                    try:
                        parts = line.split()
                        address = parts[1]
                        port = int(address.split(':')[-1])
                        if self.base_port_range[0] <= port <= self.base_port_range[1]:
                            active_ports.append(port)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception:
            # netstat 실패 시 psutil 사용
            try:
                for conn in psutil.net_connections():
                    if (conn.status == psutil.CONN_LISTEN and 
                        conn.laddr and 
                        self.base_port_range[0] <= conn.laddr.port <= self.base_port_range[1]):
                        active_ports.append(conn.laddr.port)
            except Exception:
                pass
        
        return sorted(list(set(active_ports)))
    
    def check_streamlit_app(self, port: int) -> Optional[Dict]:
        """특정 포트에서 실행 중인 Streamlit 앱 정보 확인"""
        try:
            # HTTP 요청으로 앱 상태 확인
            response = requests.get(
                f"http://localhost:{port}",
                timeout=self.timeout,
                allow_redirects=False
            )
            
            if response.status_code in [200, 302]:
                # 앱이 실행 중임을 확인
                app_info = {
                    "port": port,
                    "status": "active",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                    "last_check": time.time(),
                    "url": f"http://localhost:{port}"
                }
                
                # HTML 내용에서 앱 정보 추출 시도
                try:
                    if response.status_code == 200:
                        html_content = response.text.lower()
                        
                        # 앱 타입 추측
                        if any(keyword in html_content for keyword in ['conference', '컨퍼런스', '회의']):
                            app_info["type"] = "conference_analysis"
                            app_info["name"] = "컨퍼런스 분석"
                            app_info["icon"] = "🎯"
                        elif any(keyword in html_content for keyword in ['crawler', '크롤러', 'blog', '블로그']):
                            app_info["type"] = "web_crawler"
                            app_info["name"] = "웹 크롤러"
                            app_info["icon"] = "🕷️"
                        elif any(keyword in html_content for keyword in ['gemstone', '보석', 'diamond', 'ruby']):
                            app_info["type"] = "gemstone_analyzer"
                            app_info["name"] = "보석 분석"
                            app_info["icon"] = "💎"
                        elif any(keyword in html_content for keyword in ['cad', '3d', 'rhino', '라이노']):
                            app_info["type"] = "image_to_cad"
                            app_info["name"] = "3D CAD 변환"
                            app_info["icon"] = "🏗️"
                        elif any(keyword in html_content for keyword in ['dashboard', '대시보드', 'main']):
                            app_info["type"] = "main_dashboard"
                            app_info["name"] = "메인 대시보드"
                            app_info["icon"] = "🎯"
                        else:
                            app_info["type"] = "unknown"
                            app_info["name"] = f"포트 {port} 앱"
                            app_info["icon"] = "🔹"
                
                except Exception:
                    app_info["type"] = "streamlit_app"
                    app_info["name"] = f"Streamlit 앱 (:{port})"
                    app_info["icon"] = "🔹"
                
                return app_info
                
        except requests.exceptions.RequestException:
            pass
        
        return None
    
    def detect_all_modules(self) -> Dict[int, Dict]:
        """모든 활성 모듈 감지"""
        active_ports = self.get_active_ports()
        detected_modules = {}
        
        for port in active_ports:
            app_info = self.check_streamlit_app(port)
            if app_info:
                detected_modules[port] = app_info
                
                # 캐시 업데이트
                self.port_cache[str(port)] = {
                    "type": app_info.get("type"),
                    "name": app_info.get("name"),
                    "icon": app_info.get("icon"),
                    "last_seen": time.time()
                }
        
        # 캐시 저장
        self.save_cache()
        
        return detected_modules
    
    def get_module_by_type(self, module_type: str) -> Optional[Dict]:
        """특정 타입의 모듈 찾기"""
        modules = self.detect_all_modules()
        for port, module_info in modules.items():
            if module_info.get("type") == module_type:
                return module_info
        return None
    
    def get_preferred_ports(self) -> Dict[str, int]:
        """모듈 타입별 기본 설정 포트"""
        return {
            "main_dashboard": 8511,
            "conference_analysis": 8510,
            "web_crawler": 8502,
            "gemstone_analyzer": 8503,
            "image_to_cad": 8504
        }
    
    def find_available_port(self, preferred_port: int = None) -> int:
        """사용 가능한 포트 찾기"""
        if preferred_port and self.is_port_available(preferred_port):
            return preferred_port
        
        # 8500번대에서 사용 가능한 포트 찾기
        for port in range(self.base_port_range[0], self.base_port_range[1]):
            if self.is_port_available(port):
                return port
        
        raise Exception("사용 가능한 포트를 찾을 수 없습니다")
    
    def is_port_available(self, port: int) -> bool:
        """포트 사용 가능 여부 확인"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0  # 연결 실패 = 포트 사용 가능
        except Exception:
            return True
    
    def get_process_info(self, port: int) -> Optional[Dict]:
        """특정 포트를 사용하는 프로세스 정보"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    for conn in proc.connections():
                        if (conn.status == psutil.CONN_LISTEN and 
                            conn.laddr and 
                            conn.laddr.port == port):
                            return {
                                "pid": proc.info['pid'],
                                "name": proc.info['name'],
                                "cmdline": " ".join(proc.info['cmdline']) if proc.info['cmdline'] else "",
                                "port": port
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return None
    
    def generate_module_mapping(self) -> Dict:
        """동적 모듈 매핑 생성"""
        detected_modules = self.detect_all_modules()
        preferred_ports = self.get_preferred_ports()
        
        mapping = {
            "detected_modules": detected_modules,
            "preferred_ports": preferred_ports,
            "active_ports": list(detected_modules.keys()),
            "last_scan": time.time(),
            "total_active": len(detected_modules)
        }
        
        # 타입별 매핑
        type_mapping = {}
        for port, module_info in detected_modules.items():
            module_type = module_info.get("type", "unknown")
            type_mapping[module_type] = {
                "port": port,
                "name": module_info.get("name"),
                "icon": module_info.get("icon"),
                "url": f"http://localhost:{port}",
                "status": "active"
            }
        
        mapping["by_type"] = type_mapping
        
        return mapping

def main():
    """테스트 실행"""
    detector = PortDetector()
    
    print("Port Detection System Test")
    print("=" * 50)
    
    # 활성 포트 감지
    active_ports = detector.get_active_ports()
    print(f"Active ports: {active_ports}")
    
    # 모든 모듈 감지
    modules = detector.detect_all_modules()
    print(f"\nDetected modules ({len(modules)}):")
    for port, info in modules.items():
        print(f"  {info.get('name', 'Unknown')} - http://localhost:{port}")
    
    # 매핑 정보
    mapping = detector.generate_module_mapping()
    print(f"\nMapping info:")
    print(json.dumps(mapping, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()