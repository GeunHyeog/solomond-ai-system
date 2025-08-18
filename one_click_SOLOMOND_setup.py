#!/usr/bin/env python3
"""
솔로몬드 AI v3.0 - 원클릭 완전 자동 설정
모든 설정을 한 번에 자동으로 처리하는 통합 시스템
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

class OneClickSolomonSetup:
    """원클릭 솔로몬드 AI 완전 설정"""
    
    def __init__(self):
        self.setup_start_time = datetime.now()
        self.results = {}
        self.current_dir = os.getcwd()
        
    def print_banner(self):
        """설정 시작 배너"""
        print("=" * 60)
        print("솔로몬드 AI v3.0 - 원클릭 완전 자동 설정")
        print("=" * 60)
        print(f"시작 시간: {self.setup_start_time}")
        print(f"작업 디렉토리: {self.current_dir}")
        print()
    
    def check_config_files(self):
        """필수 설정 파일 확인"""
        print("1. 설정 파일 확인...")
        
        required_files = [
            "supabase_config.json",
            "CLAUDE.md"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"ERROR: 필수 파일 누락 - {missing_files}")
            return False
        
        # Supabase 설정 검증
        try:
            with open("supabase_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            required_keys = ["supabase.url", "supabase.anon_key", "supabase.service_role_key"]
            for key_path in required_keys:
                keys = key_path.split(".")
                current = config
                for key in keys:
                    if key not in current:
                        print(f"ERROR: 설정 키 누락 - {key_path}")
                        return False
                    current = current[key]
            
            print("SUCCESS: 모든 설정 파일 확인 완료")
            return True
            
        except Exception as e:
            print(f"ERROR: 설정 파일 검증 실패 - {e}")
            return False
    
    def install_all_dependencies(self):
        """모든 필요한 라이브러리 일괄 설치"""
        print("\n2. 의존성 라이브러리 일괄 설치...")
        
        # requirements 파일에서 의존성 읽기
        requirements_files = [
            "requirements_v23_windows.txt",
            "requirements.txt"
        ]
        
        requirements_found = False
        for req_file in requirements_files:
            if os.path.exists(req_file):
                print(f"사용할 requirements 파일: {req_file}")
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "-r", req_file
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print("SUCCESS: requirements 설치 완료")
                        requirements_found = True
                        break
                    else:
                        print(f"WARNING: {req_file} 설치 중 일부 오류")
                        
                except Exception as e:
                    print(f"WARNING: {req_file} 설치 실패 - {e}")
        
        # 필수 라이브러리 개별 설치
        essential_libs = [
            "streamlit",
            "supabase", 
            "requests",
            "psutil",
            "python-dotenv"
        ]
        
        print("필수 라이브러리 개별 확인...")
        for lib in essential_libs:
            try:
                __import__(lib.replace("-", "_"))
                print(f"SUCCESS: {lib} - 이미 설치됨")
            except ImportError:
                print(f"설치 중: {lib}...")
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", lib
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print(f"SUCCESS: {lib} - 설치 완료")
                    else:
                        print(f"ERROR: {lib} - 설치 실패")
                        
                except Exception as e:
                    print(f"ERROR: {lib} - 설치 오류: {e}")
        
        return True
    
    def setup_supabase_automatically(self):
        """Supabase 완전 자동 설정"""
        print("\n3. Supabase 자동 설정...")
        
        try:
            # auto_supabase_setup.py 실행
            result = subprocess.run([
                sys.executable, "auto_supabase_setup.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("SUCCESS: Supabase 자동 설정 완료")
                return True
            else:
                print("INFO: Supabase 수동 스키마 실행 필요")
                print("자동 설정 출력:")
                print(result.stdout)
                return "manual_needed"
                
        except Exception as e:
            print(f"ERROR: Supabase 자동 설정 실패 - {e}")
            return False
    
    def test_all_integrations(self):
        """모든 통합 기능 테스트"""
        print("\n4. 통합 기능 전체 테스트...")
        
        try:
            # simple_final_test.py 실행
            result = subprocess.run([
                sys.executable, "simple_final_test.py"
            ], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
            
            if "SUCCESS" in result.stdout and "100.0%" in result.stdout:
                print("SUCCESS: 모든 통합 테스트 통과")
                return True
            else:
                print("INFO: 통합 테스트 결과 확인 필요")
                # 유니코드 에러 무시하고 핵심 정보만 출력
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if any(keyword in line for keyword in ['SUCCESS', 'ERROR', '성공:', '실패:']):
                        try:
                            print(line)
                        except UnicodeEncodeError:
                            print(line.encode('ascii', 'ignore').decode('ascii'))
                return "partial"
                
        except Exception as e:
            print(f"ERROR: 통합 테스트 실패 - {e}")
            return False
    
    def start_all_modules(self):
        """4개 모듈 자동 시작"""
        print("\n5. 솔로몬드 AI 모듈 자동 시작...")
        
        modules = [
            {
                "name": "메인 대시보드",
                "file": "SOLOMONDd_ai_main_dashboard.py",
                "port": 8500
            },
            {
                "name": "컨퍼런스 분석",
                "file": "modules/module1_conference/conference_analysis.py", 
                "port": 8501
            },
            {
                "name": "웹 크롤러",
                "file": "modules/module2_crawler/web_crawler_main.py",
                "port": 8502
            },
            {
                "name": "보석 분석",
                "file": "modules/module3_gemstone/gemstone_analyzer.py",
                "port": 8503
            },
            {
                "name": "3D CAD 변환",
                "file": "modules/module4_3d_cad/image_to_cad.py",
                "port": 8504
            }
        ]
        
        available_modules = []
        for module in modules:
            if os.path.exists(module["file"]):
                available_modules.append(module)
                print(f"SUCCESS: {module['name']} - {module['file']} 확인")
            else:
                print(f"ERROR: {module['name']} - {module['file']} 파일 없음")
        
        if available_modules:
            print(f"\n사용 가능한 모듈: {len(available_modules)}개")
            print("\n모듈 시작 명령어:")
            for module in available_modules:
                print(f"streamlit run {module['file']} --server.port {module['port']}")
            
            print("\n브라우저에서 접속:")
            for module in available_modules:
                print(f"- {module['name']}: http://localhost:{module['port']}")
                
            return True
        else:
            print("ERROR: 사용 가능한 모듈이 없습니다")
            return False
    
    def create_startup_script(self):
        """시작 스크립트 생성"""
        print("\n6. 시작 스크립트 생성...")
        
        # Windows 배치 파일
        batch_content = """@echo off
echo 솔로몬드 AI v3.0 시작 중...

echo 메인 대시보드 시작...
start cmd /k "streamlit run SOLOMONDd_ai_main_dashboard.py --server.port 8500"

timeout /t 3

echo 컨퍼런스 분석 모듈 시작...
if exist "modules\\module1_conference\\conference_analysis.py" (
    start cmd /k "streamlit run modules\\module1_conference\\conference_analysis.py --server.port 8501"
)

timeout /t 2

echo 웹 크롤러 모듈 시작...
if exist "modules\\module2_crawler\\web_crawler_main.py" (
    start cmd /k "streamlit run modules\\module2_crawler\\web_crawler_main.py --server.port 8502"
)

timeout /t 2

echo 보석 분석 모듈 시작...
if exist "modules\\module3_gemstone\\gemstone_analyzer.py" (
    start cmd /k "streamlit run modules\\module3_gemstone\\gemstone_analyzer.py --server.port 8503"
)

timeout /t 2

echo 3D CAD 변환 모듈 시작...
if exist "modules\\module4_3d_cad\\image_to_cad.py" (
    start cmd /k "streamlit run modules\\module4_3d_cad\\image_to_cad.py --server.port 8504"
)

echo.
echo 솔로몬드 AI v3.0 시작 완료!
echo 메인 대시보드: http://localhost:8500
echo.
pause
"""
        
        with open("start_SOLOMOND_ai.bat", "w", encoding="utf-8") as f:
            f.write(batch_content)
        
        print("SUCCESS: start_SOLOMOND_ai.bat 생성 완료")
        
        # Python 스크립트
        python_content = """#!/usr/bin/env python3
import subprocess
import time
import webbrowser

def start_SOLOMOND_ai():
    print("솔로몬드 AI v3.0 자동 시작...")
    
    modules = [
        ("메인 대시보드", "SOLOMONDd_ai_main_dashboard.py", 8500),
        ("컨퍼런스 분석", "modules/module1_conference/conference_analysis.py", 8501),
        ("웹 크롤러", "modules/module2_crawler/web_crawler_main.py", 8502),
        ("보석 분석", "modules/module3_gemstone/gemstone_analyzer.py", 8503),
        ("3D CAD", "modules/module4_3d_cad/image_to_cad.py", 8504)
    ]
    
    for name, file, port in modules:
        try:
            subprocess.Popen([
                "streamlit", "run", file, "--server.port", str(port)
            ])
            print(f"✓ {name} 시작됨 (포트 {port})")
            time.sleep(2)
        except:
            print(f"✗ {name} 시작 실패")
    
    time.sleep(5)
    webbrowser.open("http://localhost:8500")

if __name__ == "__main__":
    start_SOLOMOND_ai()
"""
        
        with open("start_SOLOMOND_ai.py", "w", encoding="utf-8") as f:
            f.write(python_content)
        
        print("SUCCESS: start_SOLOMOND_ai.py 생성 완료")
        return True
    
    def run_complete_setup(self):
        """완전 자동 설정 실행"""
        self.print_banner()
        
        setup_steps = [
            ("설정 파일 확인", self.check_config_files),
            ("의존성 설치", self.install_all_dependencies),
            ("Supabase 설정", self.setup_supabase_automatically),
            ("통합 테스트", self.test_all_integrations),
            ("모듈 확인", self.start_all_modules),
            ("시작 스크립트", self.create_startup_script)
        ]
        
        success_count = 0
        manual_steps = []
        
        for step_name, step_func in setup_steps:
            print(f"\n[{len(self.results) + 1}/6] {step_name}...")
            
            try:
                result = step_func()
                self.results[step_name] = result
                
                if result is True:
                    success_count += 1
                    print(f"SUCCESS: {step_name}")
                elif result == "manual_needed":
                    print(f"WARNING: {step_name} - 수동 작업 필요")
                    manual_steps.append("Supabase SQL 스키마 실행")
                elif result == "partial":
                    success_count += 0.5
                    print(f"PARTIAL: {step_name} - 부분 성공")
                else:
                    print(f"FAILED: {step_name}")
                    
            except Exception as e:
                print(f"ERROR: {step_name} - {e}")
                self.results[step_name] = False
        
        # 최종 결과
        total_time = (datetime.now() - self.setup_start_time).total_seconds()
        success_rate = (success_count / len(setup_steps)) * 100
        
        print(f"\n" + "=" * 60)
        print("원클릭 설정 완료")
        print("=" * 60)
        print(f"성공률: {success_count}/{len(setup_steps)} ({success_rate:.1f}%)")
        print(f"소요 시간: {total_time:.1f}초")
        
        if success_count >= 4:
            print("\n솔로몬드 AI v3.0 자동 설정 성공!")
            print("\n사용 방법:")
            print("1. start_SOLOMOND_ai.bat 실행 (Windows)")
            print("2. 또는 python start_SOLOMOND_ai.py 실행")
            print("3. 브라우저에서 http://localhost:8500 접속")
            
            # 자동으로 메인 대시보드 시작
            try:
                import webbrowser
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    "SOLOMONDd_ai_main_dashboard.py", "--server.port", "8500"
                ])
                time.sleep(3)
                webbrowser.open("http://localhost:8500")
                print("\n메인 대시보드가 자동으로 시작되었습니다!")
            except:
                print("\n수동으로 메인 대시보드를 시작해주세요:")
                print("streamlit run SOLOMONDd_ai_main_dashboard.py --server.port 8500")
            
        else:
            print("\n추가 설정 필요:")
            if manual_steps:
                for i, step in enumerate(manual_steps, 1):
                    print(f"{i}. {step}")
            print(f"{len(manual_steps) + 1}. python one_click_SOLOMOND_setup.py 재실행")
        
        # 설정 보고서 저장
        report = {
            "timestamp": datetime.now().isoformat(),
            "setup_results": self.results,
            "success_rate": success_rate,
            "total_time_seconds": total_time,
            "manual_steps_needed": manual_steps,
            "system_ready": success_count >= 4
        }
        
        with open("one_click_setup_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

if __name__ == "__main__":
    print("솔로몬드 AI v3.0 원클릭 설정을 시작합니다...")
    
    setup = OneClickSolomonSetup() 
    result = setup.run_complete_setup()
    
    if result["system_ready"]:
        print("\n설정 완료! 솔로몬드 AI v3.0을 사용할 수 있습니다.")
    else:
        print(f"\n설정 보고서: one_click_setup_report.json")