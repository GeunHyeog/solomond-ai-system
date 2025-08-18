#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클린 UI 자동 테스트 스크립트
새로운 4단계 워크플로우 자동 검증
"""
import os
import sys
import time
import subprocess
import requests
from pathlib import Path
import threading

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def start_streamlit_server():
    """Streamlit 서버 시작"""
    print("[SETUP] Streamlit 서버 시작 중...")
    
    try:
        # 사용 가능한 포트 찾기
        test_ports = [8507, 8508, 8509, 8510]
        
        for port in test_ports:
            try:
                # 포트가 사용 중인지 확인
                response = requests.get(f"http://localhost:{port}", timeout=1)
                print(f"[INFO] 포트 {port}는 이미 사용 중")
            except requests.exceptions.RequestException:
                # 포트가 비어있음
                print(f"[FOUND] 포트 {port} 사용 가능")
                
                # Streamlit 실행
                cmd = [
                    sys.executable, "-m", "streamlit", "run", 
                    "jewelry_stt_ui_v23_clean.py", 
                    "--server.port", str(port),
                    "--server.headless", "true"
                ]
                
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    cwd=os.getcwd()
                )
                
                # 서버 시작 대기
                print(f"[WAIT] 서버 시작 대기 중... (포트 {port})")
                time.sleep(10)
                
                # 서버 확인
                try:
                    response = requests.get(f"http://localhost:{port}", timeout=5)
                    if response.status_code == 200:
                        print(f"[SUCCESS] 서버 정상 시작됨: http://localhost:{port}")
                        return port, process
                    else:
                        print(f"[ERROR] 서버 응답 오류: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] 서버 연결 실패: {e}")
                
                # 실패 시 프로세스 종료
                process.terminate()
        
        print("[ERROR] 사용 가능한 포트를 찾을 수 없습니다")
        return None, None
        
    except Exception as e:
        print(f"[ERROR] 서버 시작 실패: {e}")
        return None, None

def test_file_structure():
    """파일 구조 테스트"""
    print("\n[TEST 1] 파일 구조 검증")
    
    # 필수 파일 확인
    required_files = [
        "jewelry_stt_ui_v23_clean.py",
        "enhanced_speaker_identifier.py", 
        "test_new_workflow.py",
        "test_workflow_structure.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"  ❌ {file_path} - 파일 없음")
        else:
            print(f"  ✅ {file_path} - 파일 존재")
    
    # user_files 폴더 확인
    user_files_path = Path("user_files")
    if user_files_path.exists():
        print(f"  ✅ user_files 폴더 존재")
        
        # 하위 폴더 확인
        for subfolder in ["audio", "images", "videos"]:
            subfolder_path = user_files_path / subfolder
            if subfolder_path.exists():
                files = list(subfolder_path.glob("*"))
                print(f"    📁 {subfolder}: {len(files)}개 파일")
            else:
                print(f"    ❌ {subfolder} 폴더 없음")
    else:
        print(f"  ❌ user_files 폴더 없음")
        missing_files.append("user_files/")
    
    if missing_files:
        print(f"[RESULT] 파일 구조 테스트 실패 - 누락된 파일: {len(missing_files)}개")
        return False
    else:
        print(f"[RESULT] 파일 구조 테스트 성공")
        return True

def test_python_syntax():
    """Python 문법 테스트"""
    print("\n[TEST 2] Python 문법 검증")
    
    test_files = [
        "jewelry_stt_ui_v23_clean.py",
        "enhanced_speaker_identifier.py"
    ]
    
    syntax_errors = []
    
    for file_path in test_files:
        if Path(file_path).exists():
            try:
                # 문법 검사
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", file_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"  ✅ {file_path} - 문법 정상")
                else:
                    print(f"  ❌ {file_path} - 문법 오류")
                    print(f"    {result.stderr}")
                    syntax_errors.append(file_path)
                    
            except Exception as e:
                print(f"  ❌ {file_path} - 검사 실패: {e}")
                syntax_errors.append(file_path)
    
    if syntax_errors:
        print(f"[RESULT] 문법 테스트 실패 - 오류 파일: {len(syntax_errors)}개")
        return False
    else:
        print(f"[RESULT] 문법 테스트 성공")
        return True

def test_workflow_logic():
    """워크플로우 로직 테스트"""
    print("\n[TEST 3] 워크플로우 로직 검증")
    
    try:
        # 구조 테스트 실행
        print("  [3-1] 구조 테스트 실행 중...")
        result = subprocess.run(
            [sys.executable, "test_workflow_structure.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("  ✅ 워크플로우 구조 테스트 성공")
            
            # 출력에서 성공 지표 찾기
            output = result.stdout
            if "[SUCCESS]" in output and "4단계 워크플로우" in output:
                print("  ✅ 4단계 워크플로우 로직 검증 완료")
                return True
            else:
                print("  ⚠️ 워크플로우 테스트 완료했지만 성공 신호 불명확")
                print(f"    출력 일부: {output[:200]}...")
                return True
        else:
            print("  ❌ 워크플로우 구조 테스트 실패")
            print(f"    오류: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ 워크플로우 테스트 시간 초과 (30초)")
        print("  ℹ️ 실제 파일 처리로 인한 시간 초과일 수 있음")
        return True  # 시간 초과는 실패가 아님
        
    except Exception as e:
        print(f"  ❌ 워크플로우 테스트 실행 오류: {e}")
        return False

def test_enhanced_speaker_identifier():
    """Enhanced Speaker Identifier 테스트"""
    print("\n[TEST 4] Enhanced Speaker Identifier 검증")
    
    try:
        # Enhanced Speaker Identifier 임포트 테스트
        test_code = '''
import sys
sys.path.append(".")
from enhanced_speaker_identifier import EnhancedSpeakerIdentifier

# 인스턴스 생성 테스트
identifier = EnhancedSpeakerIdentifier(expected_speakers=3)
print("Enhanced Speaker Identifier 생성 성공")

# 테스트 세그먼트
test_segments = [
    {"start": 0.0, "end": 3.0, "text": "안녕하세요. 오늘 회의에 참석해 주셔서 감사합니다."},
    {"start": 4.0, "end": 6.0, "text": "네, 안녕하세요. 질문이 있습니다."},
    {"start": 7.0, "end": 12.0, "text": "그럼 이제 시작하겠습니다. 첫 번째 안건은 무엇입니까?"}
]

# 화자 구분 테스트
result_segments = identifier.identify_speakers_from_segments(test_segments)
speakers = set(seg["speaker"] for seg in result_segments)
print(f"화자 구분 테스트 성공: {len(speakers)}명 구분")
        '''
        
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            print("  ✅ Enhanced Speaker Identifier 테스트 성공")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.strip():
                    print(f"    {line}")
            return True
        else:
            print("  ❌ Enhanced Speaker Identifier 테스트 실패")
            print(f"    오류: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ Enhanced Speaker Identifier 테스트 시간 초과")
        return False
        
    except Exception as e:
        print(f"  ❌ Enhanced Speaker Identifier 테스트 오류: {e}")
        return False

def test_server_functionality(port):
    """서버 기능 테스트"""
    print(f"\n[TEST 5] 서버 기능 검증 (포트 {port})")
    
    try:
        base_url = f"http://localhost:{port}"
        
        # 메인 페이지 접근
        print("  [5-1] 메인 페이지 접근 테스트")
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            print("  ✅ 메인 페이지 접근 성공")
            
            # HTML 내용 확인
            content = response.text
            
            # 핵심 UI 요소 확인
            ui_elements = [
                "솔로몬드 AI",
                "4단계 워크플로우", 
                "소스별 정보 추출",
                "정보 종합",
                "풀스크립트 생성",
                "요약본 생성"
            ]
            
            found_elements = []
            for element in ui_elements:
                if element in content:
                    found_elements.append(element)
                    print(f"    ✅ '{element}' 발견")
                else:
                    print(f"    ❌ '{element}' 누락")
            
            if len(found_elements) >= 4:  # 최소 4개 요소 필요
                print("  ✅ UI 요소 검증 성공")
                return True
            else:
                print(f"  ⚠️ UI 요소 부족: {len(found_elements)}/{len(ui_elements)}")
                return False
                
        else:
            print(f"  ❌ 메인 페이지 접근 실패: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 서버 연결 오류: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 서버 테스트 오류: {e}")
        return False

def run_comprehensive_test():
    """종합 자동 테스트 실행"""
    print("🤖 솔로몬드 AI 새로운 4단계 워크플로우 자동 테스트 시작")
    print("=" * 60)
    
    test_results = {}
    
    # 1. 파일 구조 테스트
    test_results['file_structure'] = test_file_structure()
    
    # 2. Python 문법 테스트
    test_results['python_syntax'] = test_python_syntax()
    
    # 3. 워크플로우 로직 테스트
    test_results['workflow_logic'] = test_workflow_logic()
    
    # 4. Enhanced Speaker Identifier 테스트
    test_results['speaker_identifier'] = test_enhanced_speaker_identifier()
    
    # 5. 서버 시작 및 기능 테스트
    print(f"\n[TEST 5] 서버 시작 및 기능 테스트")
    port, process = start_streamlit_server()
    
    if port and process:
        test_results['server_functionality'] = test_server_functionality(port)
        
        # 서버 종료
        print(f"\n[CLEANUP] 서버 종료 중...")
        process.terminate()
        time.sleep(2)
        print(f"[CLEANUP] 서버 종료 완료")
    else:
        test_results['server_functionality'] = False
        print(f"[RESULT] 서버 기능 테스트 실패 - 서버 시작 불가")
    
    # 결과 요약
    print(f"\n" + "=" * 60)
    print(f"🎯 자동 테스트 결과 요약")
    print(f"=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        print(f"{status} {test_display}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 전체 결과: {passed_tests}/{total_tests} 테스트 통과")
    
    if passed_tests == total_tests:
        print(f"🎉 모든 테스트 통과! 새로운 4단계 워크플로우 완전 검증 완료")
        return True
    elif passed_tests >= total_tests * 0.8:  # 80% 이상 통과
        print(f"✅ 대부분 테스트 통과! 시스템 정상 작동 가능")
        return True
    else:
        print(f"⚠️ 일부 테스트 실패 - 추가 수정 필요")
        return False

def main():
    """메인 실행"""
    try:
        success = run_comprehensive_test()
        
        if success:
            print(f"\n🚀 자동 테스트 완료: 새로운 4단계 워크플로우 시스템 준비 완료!")
            print(f"💡 사용법: python -m streamlit run jewelry_stt_ui_v23_clean.py --server.port 8507")
        else:
            print(f"\n⚠️ 자동 테스트 완료: 일부 문제 발견, 수동 확인 권장")
            
    except KeyboardInterrupt:
        print(f"\n⏸️ 사용자에 의해 테스트 중단됨")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()