#!/usr/bin/env python3
"""
Ollama 자동 설치 및 설정 스크립트
솔로몬드 AI v2.4 통합용
"""

import os
import sys
import subprocess
import platform
import time
import requests
import json
from pathlib import Path
from typing import Dict, List, Any
import asyncio
import aiohttp

class OllamaAutoSetup:
    """Ollama 자동 설치 및 설정"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.arch = platform.machine().lower()
        
        # 필요한 모델들
        self.required_models = [
            "llama3.2:3b",          # 경량 한국어 모델 (4GB)
            "mistral:7b",           # 감정 분석 (4.1GB)
            "codellama:7b"          # 구조화된 출력 (3.8GB)
        ]
        
        # 선택 모델들 (고성능)
        self.optional_models = [
            "llama3.1:8b",          # 고성능 한국어 (4.7GB)
            "qwen2.5:7b"            # 다국어 지원 (4.4GB)
        ]
        
        print("Ollama 자동 설치 시스템 초기화")
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """시스템 요구사항 확인"""
        
        requirements = {
            "platform_supported": True,
            "memory_gb": 0,
            "disk_space_gb": 0,
            "recommendations": [],
            "warnings": []
        }
        
        try:
            # 메모리 확인
            if self.platform == "windows":
                import psutil
                memory_bytes = psutil.virtual_memory().total
                requirements["memory_gb"] = memory_bytes / (1024**3)
            
            # 디스크 공간 확인
            disk_usage = os.statvfs('.') if hasattr(os, 'statvfs') else None
            if disk_usage:
                requirements["disk_space_gb"] = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
            
            # 권장사항 생성
            if requirements["memory_gb"] < 8:
                requirements["warnings"].append("메모리 8GB 미만: 성능 저하 가능")
                requirements["recommendations"].append("경량 모델만 설치 권장")
            elif requirements["memory_gb"] >= 16:
                requirements["recommendations"].append("고성능 모델 설치 가능")
            
            if requirements["disk_space_gb"] < 20:
                requirements["warnings"].append("디스크 공간 부족: 20GB 이상 권장")
                
        except Exception as e:
            requirements["warnings"].append(f"시스템 정보 확인 실패: {str(e)}")
            
        return requirements
    
    def install_ollama(self) -> bool:
        """Ollama 설치"""
        
        print("Ollama 설치 시작...")
        
        try:
            if self.platform == "windows":
                return self._install_windows()
            elif self.platform == "linux":
                return self._install_linux()
            elif self.platform == "darwin":
                return self._install_macos()
            else:
                print(f"❌ 지원하지 않는 플랫폼: {self.platform}")
                return False
                
        except Exception as e:
            print(f"❌ 설치 실패: {str(e)}")
            return False
    
    def _install_windows(self) -> bool:
        """Windows에 Ollama 설치"""
        
        try:
            # winget 사용 시도
            result = subprocess.run(
                ["winget", "install", "ollama"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅ winget으로 설치 완료")
                return True
            else:
                print("⚠️ winget 설치 실패, 수동 다운로드 안내")
                self._manual_install_guide()
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️ 설치 시간 초과")
            return False
        except FileNotFoundError:
            print("⚠️ winget을 찾을 수 없음, 수동 설치 필요")
            self._manual_install_guide()
            return False
    
    def _install_linux(self) -> bool:
        """Linux에 Ollama 설치"""
        
        try:
            # 공식 설치 스크립트 사용
            result = subprocess.run(
                ["curl", "-fsSL", "https://ollama.ai/install.sh"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # 스크립트 실행
                install_result = subprocess.run(
                    ["sh", "-c", result.stdout],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if install_result.returncode == 0:
                    print("✅ Linux 설치 완료")
                    return True
                    
        except Exception as e:
            print(f"❌ Linux 설치 실패: {str(e)}")
            
        return False
    
    def _install_macos(self) -> bool:
        """macOS에 Ollama 설치"""
        
        try:
            # Homebrew 사용 시도
            result = subprocess.run(
                ["brew", "install", "ollama"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅ Homebrew로 설치 완료")
                return True
                
        except Exception as e:
            print(f"❌ macOS 설치 실패: {str(e)}")
            
        return False
    
    def _manual_install_guide(self):
        """수동 설치 가이드"""
        
        print("""
        📋 수동 설치 가이드:
        
        1. https://ollama.ai/download 방문
        2. 운영체제에 맞는 설치파일 다운로드
        3. 설치파일 실행
        4. 설치 완료 후 'ollama serve' 명령어 실행
        5. 다시 이 스크립트 실행
        """)
    
    def check_ollama_status(self) -> bool:
        """Ollama 서버 상태 확인"""
        
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ Ollama 설치됨: {result.stdout.strip()}")
                return True
            else:
                print("❌ Ollama 설치되지 않음")
                return False
                
        except FileNotFoundError:
            print("❌ Ollama 명령어를 찾을 수 없음")
            return False
        except Exception as e:
            print(f"❌ 상태 확인 실패: {str(e)}")
            return False
    
    def start_ollama_server(self) -> bool:
        """Ollama 서버 시작"""
        
        print("🚀 Ollama 서버 시작...")
        
        try:
            # 백그라운드에서 서버 시작
            if self.platform == "windows":
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # 서버 시작 대기
            for i in range(30):  # 30초 대기
                time.sleep(1)
                if self._check_server_running():
                    print("✅ Ollama 서버 시작됨")
                    return True
                print(f"⏳ 서버 시작 대기... ({i+1}/30)")
            
            print("❌ 서버 시작 시간 초과")
            return False
            
        except Exception as e:
            print(f"❌ 서버 시작 실패: {str(e)}")
            return False
    
    def _check_server_running(self) -> bool:
        """서버 실행 상태 확인"""
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def install_models(self, install_optional: bool = False) -> Dict[str, bool]:
        """모델 설치"""
        
        print("🧠 AI 모델 설치 시작...")
        
        models_to_install = self.required_models.copy()
        if install_optional:
            models_to_install.extend(self.optional_models)
        
        results = {}
        
        for model in models_to_install:
            print(f"📥 {model} 설치 중...")
            success = await self._install_single_model(model)
            results[model] = success
            
            if success:
                print(f"✅ {model} 설치 완료")
            else:
                print(f"❌ {model} 설치 실패")
        
        return results
    
    async def _install_single_model(self, model_name: str) -> bool:
        """개별 모델 설치"""
        
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True
            else:
                print(f"⚠️ {model_name} 설치 오류: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ {model_name} 설치 예외: {str(e)}")
            return False
    
    async def test_integration(self) -> Dict[str, Any]:
        """솔로몬드 AI 통합 테스트"""
        
        print("🧪 솔로몬드 AI 통합 테스트...")
        
        try:
            from core.ollama_integration_engine import OllamaIntegrationEngine
            
            engine = OllamaIntegrationEngine()
            status = await engine.check_ollama_availability()
            
            if status["server_available"]:
                # 간단한 한국어 테스트
                test_result = await engine.analyze_korean_conversation(
                    "고객: 안녕하세요. 상담사: 네, 무엇을 도와드릴까요?"
                )
                
                return {
                    "integration_success": True,
                    "server_status": status,
                    "test_result": test_result
                }
            else:
                return {
                    "integration_success": False,
                    "server_status": status
                }
                
        except Exception as e:
            return {
                "integration_success": False,
                "error": str(e)
            }
    
    def generate_config(self) -> str:
        """설정 파일 생성"""
        
        config = {
            "ollama": {
                "base_url": "http://localhost:11434",
                "models": {
                    "korean_chat": "llama3.2:3b",
                    "emotion_analysis": "mistral:7b",
                    "structured_output": "codellama:7b"
                },
                "timeout": 60,
                "max_retries": 3
            },
            "integration": {
                "fallback_enabled": True,
                "cache_responses": True,
                "log_level": "INFO"
            }
        }
        
        config_path = "config/ollama_config.json"
        os.makedirs("config", exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"📋 설정 파일 생성: {config_path}")
        return config_path

async def main():
    """메인 설치 프로세스"""
    
    setup = OllamaAutoSetup()
    
    print("=" * 60)
    print("🦙 Ollama + 솔로몬드 AI 통합 설치")
    print("=" * 60)
    
    # 1. 시스템 요구사항 확인
    print("\n1️⃣ 시스템 요구사항 확인")
    requirements = setup.check_system_requirements()
    
    print(f"💻 메모리: {requirements['memory_gb']:.1f}GB")
    print(f"💾 디스크: {requirements['disk_space_gb']:.1f}GB")
    
    if requirements["warnings"]:
        print("⚠️ 경고사항:")
        for warning in requirements["warnings"]:
            print(f"  • {warning}")
    
    # 2. Ollama 설치
    print("\n2️⃣ Ollama 설치")
    if not setup.check_ollama_status():
        install_success = setup.install_ollama()
        if not install_success:
            print("❌ 설치 실패. 수동 설치 후 다시 실행하세요.")
            return
    
    # 3. 서버 시작
    print("\n3️⃣ Ollama 서버 시작")
    if not setup._check_server_running():
        server_success = setup.start_ollama_server()
        if not server_success:
            print("❌ 서버 시작 실패")
            return
    
    # 4. 모델 설치
    print("\n4️⃣ AI 모델 설치")
    install_optional = input("고성능 모델도 설치하시겠습니까? (y/N): ").lower() == 'y'
    
    model_results = await setup.install_models(install_optional)
    
    successful_models = [model for model, success in model_results.items() if success]
    print(f"✅ 설치 완료: {len(successful_models)}/{len(model_results)} 모델")
    
    # 5. 통합 테스트
    print("\n5️⃣ 솔로몬드 AI 통합 테스트")
    test_result = await setup.test_integration()
    
    if test_result["integration_success"]:
        print("✅ 통합 테스트 성공!")
        print("🎉 Ollama + 솔로몬드 AI 통합 완료!")
    else:
        print("❌ 통합 테스트 실패")
        print(f"오류: {test_result.get('error', '알 수 없는 오류')}")
    
    # 6. 설정 파일 생성
    print("\n6️⃣ 설정 파일 생성")
    config_path = setup.generate_config()
    
    print("\n" + "=" * 60)
    print("🎯 설치 완료!")
    print("=" * 60)
    print("다음 명령어로 솔로몬드 AI를 시작하세요:")
    print("python -m streamlit run jewelry_stt_ui_v23_real.py --server.port 8503")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())