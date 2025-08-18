#!/usr/bin/env python3
"""
SOLOMOND AI 4개 모듈 통합 테스트 시스템
NEXT_SESSION_GOALS.md의 우선순위 1 작업 수행
"""

import subprocess
import time
import requests
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any

class MultiModuleIntegrationTester:
    """4개 모듈 통합 테스트"""
    
    def __init__(self):
        self.modules = {
            "main_dashboard": {
                "port": 8500,
                "file": "solomond_ai_main_dashboard.py",
                "name": "메인 대시보드"
            },
            "module1_conference": {
                "port": 8501,
                "file": "modules/module1_conference/conference_analysis.py",
                "name": "컨퍼런스 분석"
            },
            "module2_crawler": {
                "port": 8502,
                "file": "modules/module2_crawler/web_crawler_main.py",
                "name": "웹 크롤러"
            },
            "module3_gemstone": {
                "port": 8503,
                "file": "modules/module3_gemstone/gemstone_analyzer.py",
                "name": "보석 분석"
            },
            "module4_3d_cad": {
                "port": 8504,
                "file": "modules/module4_3d_cad/image_to_cad.py",
                "name": "3D CAD 변환"
            }
        }
        
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "module_status": {},
            "performance_metrics": {},
            "integration_tests": {},
            "recommendations": []
        }
    
    def check_file_exists(self) -> Dict[str, bool]:
        """모듈 파일 존재 확인"""
        import os
        
        file_status = {}
        print("📁 모듈 파일 존재 확인...")
        
        for module_id, module_info in self.modules.items():
            exists = os.path.exists(module_info["file"])
            file_status[module_id] = exists
            status = "✅" if exists else "❌"
            print(f"  {status} {module_info['name']}: {module_info['file']}")
        
        return file_status
    
    def check_running_processes(self) -> Dict[str, Any]:
        """실행 중인 Streamlit 프로세스 확인"""
        print("\\n🔍 실행 중인 프로세스 확인...")
        
        running_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'streamlit' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    running_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline
                    })
            except:
                continue
        
        print(f"  실행 중인 Streamlit 프로세스: {len(running_processes)}개")
        for proc in running_processes:
            print(f"    PID {proc['pid']}: {proc['cmdline'][:80]}...")
        
        return {"count": len(running_processes), "processes": running_processes}
    
    def test_port_connectivity(self) -> Dict[str, Any]:
        """포트 연결성 테스트"""
        print("\\n🌐 포트 연결성 테스트...")
        
        port_status = {}
        for module_id, module_info in self.modules.items():
            port = module_info["port"]
            try:
                response = requests.get(f"http://localhost:{port}", timeout=5)
                status = "✅ 정상" if response.status_code == 200 else f"⚠️ 응답 {response.status_code}"
                port_status[module_id] = {
                    "accessible": True,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except requests.exceptions.ConnectionError:
                status = "❌ 연결 실패"
                port_status[module_id] = {
                    "accessible": False,
                    "error": "Connection refused"
                }
            except requests.exceptions.Timeout:
                status = "⏱️ 타임아웃"
                port_status[module_id] = {
                    "accessible": False,
                    "error": "Timeout"
                }
            except Exception as e:
                status = f"❌ 오류: {str(e)[:30]}"
                port_status[module_id] = {
                    "accessible": False,
                    "error": str(e)
                }
            
            print(f"  포트 {port} ({module_info['name']}): {status}")
        
        return port_status
    
    def start_missing_modules(self, port_status: Dict) -> Dict[str, Any]:
        """누락된 모듈 시작"""
        print("\\n🚀 누락된 모듈 시작...")
        
        started_modules = {}
        for module_id, module_info in self.modules.items():
            if not port_status.get(module_id, {}).get("accessible", False):
                print(f"  시작 중: {module_info['name']} (포트 {module_info['port']})")
                
                try:
                    # Windows에서 백그라운드 실행
                    cmd = f'start cmd /k "streamlit run {module_info["file"]} --server.port {module_info["port"]}"'
                    subprocess.Popen(cmd, shell=True)
                    
                    started_modules[module_id] = {
                        "attempted": True,
                        "command": cmd
                    }
                    
                    # 시작 대기
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"    ❌ 시작 실패: {e}")
                    started_modules[module_id] = {
                        "attempted": True,
                        "error": str(e)
                    }
            else:
                started_modules[module_id] = {"already_running": True}
        
        return started_modules
    
    def wait_for_startup(self, timeout=60) -> Dict[str, Any]:
        """모든 모듈 시작 대기"""
        print(f"\\n⏳ 모듈 시작 대기 (최대 {timeout}초)...")
        
        start_time = time.time()
        startup_status = {}
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for module_id, module_info in self.modules.items():
                try:
                    response = requests.get(f"http://localhost:{module_info['port']}", timeout=2)
                    if response.status_code == 200:
                        if module_id not in startup_status:
                            print(f"  ✅ {module_info['name']} 준비 완료")
                            startup_status[module_id] = {
                                "ready": True,
                                "startup_time": time.time() - start_time
                            }
                    else:
                        all_ready = False
                except:
                    all_ready = False
            
            if all_ready:
                print("  🎉 모든 모듈 준비 완료!")
                break
            
            time.sleep(2)
        
        return startup_status
    
    def test_cross_module_workflow(self) -> Dict[str, Any]:
        """크로스 모듈 워크플로우 테스트"""
        print("\\n🔗 크로스 모듈 워크플로우 테스트...")
        
        workflow_tests = {
            "main_to_modules": self.test_main_dashboard_navigation(),
            "module_to_module": self.test_module_interconnection(),
            "data_sharing": self.test_data_sharing_capability()
        }
        
        return workflow_tests
    
    def test_main_dashboard_navigation(self) -> Dict[str, Any]:
        """메인 대시보드 네비게이션 테스트"""
        try:
            response = requests.get("http://localhost:8500", timeout=10)
            
            # 간단한 HTML 내용 확인
            content = response.text.lower()
            has_navigation = any(keyword in content for keyword in [
                "module", "conference", "crawler", "gemstone", "cad"
            ])
            
            return {
                "accessible": response.status_code == 200,
                "has_navigation": has_navigation,
                "response_size": len(response.text)
            }
        except Exception as e:
            return {"accessible": False, "error": str(e)}
    
    def test_module_interconnection(self) -> Dict[str, Any]:
        """모듈 간 연결 테스트"""
        interconnection_status = {}
        
        for module_id, module_info in self.modules.items():
            if module_id == "main_dashboard":
                continue
                
            try:
                response = requests.get(f"http://localhost:{module_info['port']}", timeout=5)
                interconnection_status[module_id] = {
                    "accessible": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds()
                }
            except:
                interconnection_status[module_id] = {"accessible": False}
        
        return interconnection_status
    
    def test_data_sharing_capability(self) -> Dict[str, Any]:
        """데이터 공유 능력 테스트"""
        import os
        
        # 공유 가능한 데이터 디렉토리 확인
        shared_dirs = ["uploads", "outputs", "temp", "cache", "analysis_history"]
        data_sharing = {}
        
        for dir_name in shared_dirs:
            exists = os.path.exists(dir_name)
            if exists:
                try:
                    files = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
                    data_sharing[dir_name] = {"exists": True, "file_count": files}
                except:
                    data_sharing[dir_name] = {"exists": True, "file_count": 0}
            else:
                data_sharing[dir_name] = {"exists": False}
        
        return data_sharing
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 수집"""
        print("\\n📊 성능 메트릭 수집...")
        
        # CPU, 메모리 사용량
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # 프로세스별 메모리 사용량
        streamlit_memory = 0
        streamlit_count = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'streamlit' in proc.info['name'].lower():
                    streamlit_memory += proc.info['memory_info'].rss
                    streamlit_count += 1
            except:
                continue
        
        metrics = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent
            },
            "streamlit": {
                "process_count": streamlit_count,
                "memory_usage_mb": streamlit_memory / (1024**2),
                "memory_per_process_mb": (streamlit_memory / streamlit_count / (1024**2)) if streamlit_count > 0 else 0
            }
        }
        
        print(f"  CPU 사용률: {cpu_percent:.1f}%")
        print(f"  메모리 사용률: {memory.percent:.1f}%")
        print(f"  Streamlit 프로세스: {streamlit_count}개")
        print(f"  Streamlit 메모리: {streamlit_memory / (1024**2):.1f}MB")
        
        return metrics
    
    def generate_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        # 성능 기반 권장사항
        metrics = self.test_results.get("performance_metrics", {})
        system_metrics = metrics.get("system", {})
        
        if system_metrics.get("cpu_percent", 0) > 80:
            recommendations.append("CPU 사용률이 높습니다. 일부 모듈을 순차적으로 실행을 고려하세요.")
        
        if system_metrics.get("memory_percent", 0) > 85:
            recommendations.append("메모리 사용률이 높습니다. 캐시 정리나 메모리 최적화가 필요합니다.")
        
        # 모듈 상태 기반 권장사항
        port_status = self.test_results.get("module_status", {}).get("port_connectivity", {})
        failed_modules = [k for k, v in port_status.items() if not v.get("accessible", False)]
        
        if failed_modules:
            recommendations.append(f"다음 모듈들의 재시작이 필요합니다: {', '.join(failed_modules)}")
        
        # 일반적 권장사항
        if len(recommendations) == 0:
            recommendations.append("모든 시스템이 정상 작동 중입니다. 정기적인 모니터링을 권장합니다.")
        
        return recommendations
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        print("🎯 SOLOMOND AI 4개 모듈 통합 테스트 시작")
        print("=" * 60)
        
        # 1. 파일 존재 확인
        file_status = self.check_file_exists()
        self.test_results["module_status"]["file_exists"] = file_status
        
        # 2. 실행 중인 프로세스 확인
        process_status = self.check_running_processes()
        self.test_results["module_status"]["running_processes"] = process_status
        
        # 3. 포트 연결성 테스트
        port_status = self.test_port_connectivity()
        self.test_results["module_status"]["port_connectivity"] = port_status
        
        # 4. 누락된 모듈 시작
        started_modules = self.start_missing_modules(port_status)
        self.test_results["module_status"]["started_modules"] = started_modules
        
        # 5. 시작 대기
        startup_status = self.wait_for_startup()
        self.test_results["module_status"]["startup_status"] = startup_status
        
        # 6. 크로스 모듈 워크플로우 테스트
        workflow_tests = self.test_cross_module_workflow()
        self.test_results["integration_tests"] = workflow_tests
        
        # 7. 성능 메트릭 수집
        performance_metrics = self.collect_performance_metrics()
        self.test_results["performance_metrics"] = performance_metrics
        
        # 8. 권장사항 생성
        recommendations = self.generate_recommendations()
        self.test_results["recommendations"] = recommendations
        
        # 9. 완료 시간 기록
        self.test_results["end_time"] = datetime.now().isoformat()
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_module_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\\n💾 테스트 결과 저장: {filename}")
        return filename
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\\n" + "=" * 60)
        print("📋 테스트 결과 요약")
        print("=" * 60)
        
        # 모듈 상태 요약
        port_status = self.test_results.get("module_status", {}).get("port_connectivity", {})
        accessible_count = sum(1 for v in port_status.values() if v.get("accessible", False))
        total_modules = len(self.modules)
        
        print(f"🎯 모듈 접근성: {accessible_count}/{total_modules} ({accessible_count/total_modules*100:.1f}%)")
        
        # 성능 요약
        metrics = self.test_results.get("performance_metrics", {})
        if metrics:
            system = metrics.get("system", {})
            streamlit = metrics.get("streamlit", {})
            
            print(f"💻 시스템 성능:")
            print(f"   CPU: {system.get('cpu_percent', 0):.1f}%")
            print(f"   메모리: {system.get('memory_percent', 0):.1f}%")
            print(f"   Streamlit 프로세스: {streamlit.get('process_count', 0)}개")
        
        # 권장사항
        recommendations = self.test_results.get("recommendations", [])
        if recommendations:
            print(f"\\n💡 권장사항:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

def main():
    """메인 실행"""
    tester = MultiModuleIntegrationTester()
    
    # 종합 테스트 실행
    results = tester.run_comprehensive_test()
    
    # 결과 저장
    result_file = tester.save_results()
    
    # 요약 출력
    tester.print_summary()
    
    print(f"\\n🎉 통합 테스트 완료!")
    print(f"📄 상세 결과: {result_file}")
    
    return results

if __name__ == "__main__":
    main()