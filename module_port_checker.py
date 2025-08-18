#!/usr/bin/env python3
"""
🔍 모듈 포트 및 상태 점검기
각 모듈의 포트 설정과 실행 상태를 자동으로 점검하는 스크립트
"""

import subprocess
import json
import time
import requests
from pathlib import Path
from datetime import datetime

class ModulePortChecker:
    def __init__(self):
        self.modules = {
            'module1_conference': {
                'name': '컨퍼런스 분석',
                'port': 8501,
                'main_file': 'modules/module1_conference/conference_analysis.py',
                'optimized_file': 'modules/module1_conference/conference_analysis_optimized.py'
            },
            'module2_crawler': {
                'name': '웹 크롤러',
                'port': 8502,
                'main_file': 'modules/module2_crawler/web_crawler_main.py',
                'optimized_file': 'modules/module2_crawler/web_crawler_optimized.py'
            },
            'module3_gemstone': {
                'name': '보석 분석',
                'port': 8503,
                'main_file': 'modules/module3_gemstone/gemstone_analyzer.py',
                'optimized_file': 'modules/module3_gemstone/gemstone_analyzer_optimized.py'
            },
            'module4_3d_cad': {
                'name': '3D CAD 변환',
                'port': 8504,
                'main_file': 'modules/module4_3d_cad/image_to_cad.py',
                'optimized_file': 'modules/module4_3d_cad/image_to_cad_optimized.py'
            }
        }
        
    def check_port_availability(self, port):
        """포트 사용 가능 여부 확인"""
        try:
            response = requests.get(f'http://localhost:{port}', timeout=2)
            return True, response.status_code
        except requests.exceptions.ConnectionError:
            return False, None
        except requests.exceptions.Timeout:
            return True, 'timeout'
        except Exception as e:
            return False, str(e)
    
    def check_file_exists(self, filepath):
        """파일 존재 여부 확인"""
        return Path(filepath).exists()
    
    def get_running_streamlit_processes(self):
        """실행 중인 Streamlit 프로세스 확인"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            streamlit_processes = []
            for line in result.stdout.split('\n'):
                if 'streamlit' in line and 'run' in line:
                    streamlit_processes.append(line.strip())
            return streamlit_processes
        except Exception as e:
            try:
                # Windows에서는 tasklist 사용
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                      capture_output=True, text=True)
                return result.stdout.split('\n')
            except:
                return [f"Process check failed: {e}"]
    
    def check_module_status(self):
        """전체 모듈 상태 점검"""
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'running_processes': self.get_running_streamlit_processes(),
            'summary': {
                'total_modules': len(self.modules),
                'files_exist': 0,
                'ports_available': 0,
                'ports_in_use': 0
            }
        }
        
        print("모듈 포트 및 상태 점검 시작...")
        print("=" * 60)
        
        for module_id, module_info in self.modules.items():
            print(f"\n[{module_info['name']}] (포트 {module_info['port']})")
            
            module_status = {
                'name': module_info['name'],
                'port': module_info['port'],
                'main_file_exists': self.check_file_exists(module_info['main_file']),
                'optimized_file_exists': self.check_file_exists(module_info['optimized_file']),
                'port_in_use': False,
                'port_response': None,
                'recommended_action': ''
            }
            
            # 파일 존재 확인
            if module_status['main_file_exists']:
                print(f"   [OK] 메인 파일 존재: {module_info['main_file']}")
                status_report['summary']['files_exist'] += 1
            else:
                print(f"   [NO] 메인 파일 없음: {module_info['main_file']}")
                
            if module_status['optimized_file_exists']:
                print(f"   [OPT] 최적화 파일 존재: {module_info['optimized_file']}")
            else:
                print(f"   [WARN] 최적화 파일 없음: {module_info['optimized_file']}")
            
            # 포트 상태 확인
            port_available, response = self.check_port_availability(module_info['port'])
            module_status['port_in_use'] = port_available
            module_status['port_response'] = response
            
            if port_available:
                print(f"   [RUNNING] 포트 {module_info['port']} 사용 중 (응답: {response})")
                status_report['summary']['ports_in_use'] += 1
                module_status['recommended_action'] = '이미 실행 중'
            else:
                print(f"   [AVAILABLE] 포트 {module_info['port']} 사용 가능")
                status_report['summary']['ports_available'] += 1
                if module_status['main_file_exists']:
                    module_status['recommended_action'] = '실행 가능'
                else:
                    module_status['recommended_action'] = '파일 확인 필요'
            
            status_report['modules'][module_id] = module_status
        
        # 실행 중인 프로세스 정보
        print(f"\n[PROCESSES] 실행 중인 Streamlit 프로세스:")
        for process in status_report['running_processes'][:5]:  # 최대 5개만 표시
            if process.strip():
                print(f"   - {process}")
        
        # 요약 정보
        print(f"\n[SUMMARY] 요약:")
        print(f"   총 모듈 수: {status_report['summary']['total_modules']}")
        print(f"   파일 존재: {status_report['summary']['files_exist']}")
        print(f"   사용 가능한 포트: {status_report['summary']['ports_available']}")
        print(f"   사용 중인 포트: {status_report['summary']['ports_in_use']}")
        
        return status_report
    
    def generate_startup_commands(self, status_report):
        """각 모듈 실행 명령어 생성"""
        print(f"\n[COMMANDS] 모듈 실행 명령어:")
        print("=" * 40)
        
        commands = []
        for module_id, module_status in status_report['modules'].items():
            module_info = self.modules[module_id]
            
            if not module_status['port_in_use'] and module_status['main_file_exists']:
                # 최적화 파일이 있으면 우선 사용
                file_to_use = (module_info['optimized_file'] 
                             if module_status['optimized_file_exists'] 
                             else module_info['main_file'])
                
                cmd = f"streamlit run {file_to_use} --server.port {module_info['port']}"
                commands.append(cmd)
                print(f"# {module_info['name']}")
                print(f"{cmd}")
                print("")
        
        return commands
    
    def save_report(self, status_report, filename=None):
        """상태 보고서를 JSON 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"module_status_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(status_report, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVED] 상태 보고서 저장: {filename}")
        return filename

def main():
    checker = ModulePortChecker()
    
    # 상태 점검 실행
    status_report = checker.check_module_status()
    
    # 실행 명령어 생성
    commands = checker.generate_startup_commands(status_report)
    
    # 보고서 저장
    report_file = checker.save_report(status_report)
    
    # 추천 작업 제안
    print(f"\n[RECOMMENDATIONS] 추천 작업:")
    if status_report['summary']['ports_in_use'] == 0:
        print("   1. 메인 대시보드가 실행 중이므로 개별 모듈을 차례로 실행해보세요")
        print("   2. 먼저 모듈 1(컨퍼런스 분석)부터 시작하는 것을 권장합니다")
    elif status_report['summary']['ports_in_use'] < 4:
        print("   일부 모듈이 실행 중입니다. 나머지 모듈들을 실행해보세요")
    else:
        print("   모든 모듈이 실행 중입니다. 개발/테스트를 진행하세요")

if __name__ == "__main__":
    main()