#!/usr/bin/env python3
"""
Windows에서 Ollama 자동 설치
"""

import subprocess
import time
import urllib.request
import os
import sys

def install_ollama_winget():
    """winget으로 Ollama 설치"""
    
    print("=== winget으로 Ollama 설치 ===")
    
    try:
        # winget 명령어 확인
        result = subprocess.run(["winget", "--version"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("winget을 사용할 수 없습니다.")
            return False
        
        print(f"winget 버전: {result.stdout.strip()}")
        
        # Ollama 설치
        print("Ollama 설치 중... (몇 분 소요)")
        
        result = subprocess.run([
            "winget", "install", "--id=Ollama.Ollama", 
            "--silent", "--accept-package-agreements", "--accept-source-agreements"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("Ollama 설치 성공!")
            return True
        else:
            print(f"설치 실패: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("설치 시간 초과")
        return False
    except Exception as e:
        print(f"설치 오류: {str(e)}")
        return False

def download_and_install_ollama():
    """직접 다운로드 후 설치"""
    
    print("\n=== 직접 다운로드 설치 ===")
    
    try:
        # Ollama Windows 설치 파일 URL
        url = "https://ollama.ai/download/OllamaSetup.exe"
        filename = "OllamaSetup.exe"
        
        print(f"다운로드 중: {url}")
        
        # 다운로드
        urllib.request.urlretrieve(url, filename)
        print(f"다운로드 완료: {filename}")
        
        # 자동 설치 실행
        print("자동 설치 실행 중...")
        
        result = subprocess.run([filename, "/S"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("설치 완료!")
            
            # 설치 파일 삭제
            try:
                os.remove(filename)
                print("설치 파일 정리 완료")
            except:
                pass
            
            return True
        else:
            print(f"설치 실패: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"다운로드/설치 오류: {str(e)}")
        return False

def check_installation():
    """설치 확인"""
    
    print("\n=== 설치 확인 ===")
    
    # PATH 새로고침을 위해 잠시 대기
    time.sleep(5)
    
    try:
        # 직접 실행 경로들 확인
        possible_paths = [
            r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME")),
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Ollama 발견: {path}")
                
                # 버전 확인
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print(f"버전: {result.stdout.strip()}")
                    
                    # 환경변수에 추가
                    ollama_dir = os.path.dirname(path)
                    current_path = os.environ.get('PATH', '')
                    
                    if ollama_dir not in current_path:
                        os.environ['PATH'] = f"{ollama_dir};{current_path}"
                        print(f"PATH에 추가: {ollama_dir}")
                    
                    return True
        
        # 일반 명령어로 확인
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"명령어로 확인 성공: {result.stdout.strip()}")
            return True
        
        print("설치 확인 실패")
        return False
        
    except Exception as e:
        print(f"확인 오류: {str(e)}")
        return False

def start_ollama_service():
    """Ollama 서비스 시작"""
    
    print("\n=== Ollama 서비스 시작 ===")
    
    try:
        # ollama serve 백그라운드 실행
        print("서비스 시작 중...")
        
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW)
        
        # 서비스 시작 대기
        print("서비스 시작 대기... (15초)")
        time.sleep(15)
        
        # 연결 테스트
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            
            if response.status_code == 200:
                print("서비스 시작 성공!")
                return True
            else:
                print(f"서비스 응답 오류: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"서비스 연결 실패: {str(e)}")
            return False
            
    except Exception as e:
        print(f"서비스 시작 오류: {str(e)}")
        return False

def install_gemma2_models():
    """GEMMA2 모델 설치"""
    
    print("\n=== GEMMA2 모델 설치 ===")
    
    models = [
        ("gemma2:2b", "경량 버전 (2B 파라미터)"),
        ("gemma2:9b", "권장 버전 (9B 파라미터)")
    ]
    
    installed = []
    
    for model_id, description in models:
        print(f"\n{description} 설치 중: {model_id}")
        print("이 작업은 몇 분 소요될 수 있습니다...")
        
        try:
            result = subprocess.run(["ollama", "pull", model_id],
                                  capture_output=True, text=True, timeout=900)  # 15분 대기
            
            if result.returncode == 0:
                print(f"설치 완료: {model_id}")
                installed.append(model_id)
            else:
                print(f"설치 실패: {result.stderr}")
                # 실패해도 계속 진행
                
        except subprocess.TimeoutExpired:
            print(f"설치 시간 초과: {model_id}")
        except Exception as e:
            print(f"설치 오류: {str(e)}")
    
    return installed

def main():
    """메인 설치 프로세스"""
    
    print("Ollama + GEMMA2 자동 설치")
    print("="*40)
    
    success = False
    
    # 1. winget으로 설치 시도
    if install_ollama_winget():
        success = True
    
    # 2. 직접 다운로드로 설치 시도  
    if not success:
        if download_and_install_ollama():
            success = True
    
    if not success:
        print("\n자동 설치 실패")
        print("수동 설치 필요:")
        print("1. https://ollama.ai/download 방문")
        print("2. Windows용 설치 파일 다운로드")
        print("3. 설치 후 다시 실행")
        return False
    
    # 3. 설치 확인
    if not check_installation():
        print("\n설치 확인 실패")
        return False
    
    # 4. 서비스 시작
    if not start_ollama_service():
        print("\n서비스 시작 실패")
        print("수동으로 'ollama serve' 실행 후 다시 시도하세요.")
        return False
    
    # 5. GEMMA2 모델 설치
    installed_models = install_gemma2_models()
    
    if installed_models:
        print(f"\n성공! 다음 모델이 설치되었습니다:")
        for model in installed_models:
            print(f"  - {model}")
        
        print(f"\n다음 단계:")
        print(f"1. python simple_gemma3_test.py - 성능 테스트")
        print(f"2. 솔로몬드 AI 시스템에 통합")
        
        return True
    else:
        print(f"\n모델 설치 실패")
        print(f"수동으로 다음 명령어 실행:")
        print(f"  ollama pull gemma2:2b")
        print(f"  ollama pull gemma2:9b")
        
        return False

if __name__ == "__main__":
    main()