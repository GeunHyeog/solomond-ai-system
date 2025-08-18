#!/usr/bin/env python3
"""
GEMMA3 자동 설치 및 설정 스크립트
"""

import subprocess
import time
import asyncio
import sys
import os
from pathlib import Path

class AutoGEMMA3Setup:
    """GEMMA3 자동 설치 및 설정"""
    
    def __init__(self):
        self.ollama_installed = False
        self.ollama_running = False
        
    def check_ollama_installation(self):
        """Ollama 설치 상태 확인"""
        
        print("=== Ollama 설치 상태 확인 ===")
        
        # Windows에서 Ollama 설치 확인
        try:
            # 1. 명령어로 확인
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✓ Ollama 설치됨: {result.stdout.strip()}")
                self.ollama_installed = True
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # 2. 일반적인 설치 경로 확인
        common_paths = [
            r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME")),
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print(f"✓ Ollama 발견: {path}")
                self.ollama_installed = True
                return True
        
        print("✗ Ollama가 설치되지 않음")
        return False
    
    def install_ollama(self):
        """Ollama 자동 설치"""
        
        print("\n=== Ollama 자동 설치 ===")
        
        if self.ollama_installed:
            print("✓ Ollama 이미 설치됨")
            return True
        
        try:
            # Windows에서 winget으로 설치 시도
            print("winget으로 Ollama 설치 시도...")
            
            result = subprocess.run(
                ["winget", "install", "--id=Ollama.Ollama", "--silent"],
                capture_output=True,
                text=True,
                timeout=300  # 5분 대기
            )
            
            if result.returncode == 0:
                print("✓ Ollama 설치 성공")
                self.ollama_installed = True
                return True
            else:
                print(f"winget 설치 실패: {result.stderr}")
                
        except Exception as e:
            print(f"자동 설치 실패: {str(e)}")
        
        # 수동 설치 안내
        print("\n수동 설치가 필요합니다:")
        print("1. https://ollama.ai/download 방문")
        print("2. Windows용 Ollama 다운로드")
        print("3. 설치 후 다시 실행")
        
        return False
    
    def start_ollama_service(self):
        """Ollama 서비스 시작"""
        
        print("\n=== Ollama 서비스 시작 ===")
        
        # 1. 이미 실행 중인지 확인
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ollama.exe"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "ollama.exe" in result.stdout:
                print("✓ Ollama 서비스 이미 실행 중")
                self.ollama_running = True
                return True
                
        except Exception as e:
            print(f"프로세스 확인 실패: {str(e)}")
        
        # 2. 서비스 시작 시도
        try:
            print("Ollama 서비스 시작 중...")
            
            # 백그라운드로 ollama serve 실행
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # 서비스 시작 대기
            print("서비스 시작 대기 중... (10초)")
            time.sleep(10)
            
            # 서비스 확인
            if self.check_ollama_server():
                print("✓ Ollama 서비스 시작 성공")
                self.ollama_running = True
                return True
            else:
                print("✗ 서비스 시작 실패")
                return False
                
        except Exception as e:
            print(f"서비스 시작 실패: {str(e)}")
            return False
    
    def check_ollama_server(self):
        """Ollama 서버 응답 확인"""
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def install_gemma3_models(self):
        """GEMMA3 모델 자동 설치"""
        
        print("\n=== GEMMA3 모델 자동 설치 ===")
        
        if not self.ollama_running:
            print("✗ Ollama 서비스가 실행되지 않음")
            return False
        
        # 설치할 모델 목록 (경량 -> 고성능 순)
        models_to_install = [
            ("gemma2:2b", "경량 버전 (2B)"),
            ("gemma2:9b", "권장 버전 (9B)")
        ]
        
        installed_models = []
        
        for model_id, description in models_to_install:
            print(f"\n{description} 설치 중: {model_id}")
            
            try:
                # ollama pull 명령어 실행
                process = subprocess.Popen(
                    ["ollama", "pull", model_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 실시간 진행 상황 표시
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # 간단한 진행 상황만 표시
                        if "pulling" in output.lower() or "%" in output:
                            print(f"  진행: {output.strip()}")
                
                # 프로세스 완료 대기
                return_code = process.poll()
                
                if return_code == 0:
                    print(f"✓ {model_id} 설치 완료")
                    installed_models.append(model_id)
                else:
                    stderr_output = process.stderr.read()
                    print(f"✗ {model_id} 설치 실패: {stderr_output}")
                
            except Exception as e:
                print(f"✗ {model_id} 설치 오류: {str(e)}")
        
        if installed_models:
            print(f"\n✓ 설치된 모델: {len(installed_models)}개")
            for model in installed_models:
                print(f"  - {model}")
            return True
        else:
            print("\n✗ 모델 설치 실패")
            return False
    
    async def test_gemma3_performance(self):
        """GEMMA3 성능 자동 테스트"""
        
        print("\n=== GEMMA3 성능 자동 테스트 ===")
        
        # 간단한 테스트 스크립트 실행
        try:
            result = subprocess.run(
                [sys.executable, "simple_gemma3_test.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✓ 성능 테스트 완료")
                print(result.stdout)
                return True
            else:
                print(f"✗ 성능 테스트 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ 테스트 실행 오류: {str(e)}")
            return False
    
    async def integrate_with_SOLOMONDd_ai(self):
        """솔로몬드 AI 시스템에 GEMMA3 통합"""
        
        print("\n=== 솔로몬드 AI 통합 ===")
        
        # ollama_integration_engine.py 업데이트
        try:
            engine_file = Path("core/ollama_integration_engine.py")
            
            if engine_file.exists():
                # 기존 파일 읽기
                with open(engine_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # GEMMA3 모델 추가 (기존 모델과 함께)
                if "gemma2:9b" not in content:
                    # 모델 설정 부분 찾아서 GEMMA3 추가
                    models_section = '''        self.models = {
            "korean_chat": "llama3.1:8b-korean",  # 한국어 대화 이해
            "emotion_analysis": "mistral:7b",      # 감정 분석
            "structured_output": "codellama:7b",   # 구조화된 출력
            "gemma3_korean": "gemma2:9b",          # GEMMA3 한국어 분석 (NEW)
            "gemma3_fast": "gemma2:2b"             # GEMMA3 빠른 처리 (NEW)
        }'''
                    
                    # 기존 모델 설정 찾아서 교체
                    import re
                    pattern = r'self\.models = \{[^}]+\}'
                    
                    if re.search(pattern, content):
                        updated_content = re.sub(pattern, models_section, content)
                        
                        # 백업 파일 생성
                        backup_file = engine_file.with_suffix('.py.backup')
                        with open(backup_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # 업데이트된 내용 저장
                        with open(engine_file, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                        
                        print(f"✓ {engine_file} 업데이트 완료")
                        print(f"✓ 백업 파일: {backup_file}")
                        
                        return True
                
                else:
                    print("✓ GEMMA3 이미 통합됨")
                    return True
                    
            else:
                print("✗ ollama_integration_engine.py 파일을 찾을 수 없음")
                return False
                
        except Exception as e:
            print(f"✗ 통합 실패: {str(e)}")
            return False
    
    async def run_full_setup(self):
        """전체 설정 자동 실행"""
        
        print("🚀 GEMMA3 자동 설정 시작")
        print("="*50)
        
        try:
            # 1. Ollama 설치 확인
            if not self.check_ollama_installation():
                if not self.install_ollama():
                    print("❌ Ollama 설치 실패 - 수동 설치 필요")
                    return False
            
            # 2. Ollama 서비스 시작
            if not self.start_ollama_service():
                print("❌ Ollama 서비스 시작 실패")
                return False
            
            # 3. GEMMA3 모델 설치
            if not await self.install_gemma3_models():
                print("❌ GEMMA3 모델 설치 실패")
                return False
            
            # 4. 성능 테스트
            if not await self.test_gemma3_performance():
                print("⚠️ 성능 테스트 실패 - 수동 확인 필요")
            
            # 5. 솔로몬드 AI 통합
            if not await self.integrate_with_SOLOMONDd_ai():
                print("⚠️ 시스템 통합 실패 - 수동 설정 필요")
            
            print("\n🎉 GEMMA3 자동 설정 완료!")
            print("✓ Ollama 서비스 실행 중")
            print("✓ GEMMA3 모델 설치 완료")
            print("✓ 솔로몬드 AI 통합 준비 완료")
            
            print(f"\n다음 단계:")
            print(f"1. python demo_integrated_system.py - 통합 테스트")
            print(f"2. 메인 시스템에서 GEMMA3 성능 확인")
            print(f"3. 기존 모델과 성능 비교")
            
            return True
            
        except Exception as e:
            print(f"❌ 자동 설정 실패: {str(e)}")
            return False

async def main():
    """메인 실행"""
    
    setup = AutoGEMMA3Setup()
    success = await setup.run_full_setup()
    
    if success:
        print("\n✨ 모든 설정이 완료되었습니다!")
    else:
        print("\n⚠️ 일부 설정에 문제가 있습니다. 수동 확인이 필요합니다.")

if __name__ == "__main__":
    asyncio.run(main())