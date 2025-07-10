#!/usr/bin/env python3
"""
🚀 솔로몬드 AI v2.1.1 - Windows 자동 설치 스크립트
Windows 환경에서 발생하는 모든 패키지 문제를 자동으로 해결합니다.

Author: 전근혁 (Solomond)
Created: 2025.07.11
Version: 2.1.1
"""

import os
import sys
import subprocess
import platform
import importlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class Color:
    """컬러 출력을 위한 ANSI 코드"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class WindowsInstaller:
    """Windows 환경 최적화 설치 관리자"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.is_windows = platform.system().lower() == "windows"
        self.install_log = []
        self.failed_packages = []
        self.success_packages = []
        
        print(f"{Color.CYAN}{Color.BOLD}🚀 솔로몬드 AI v2.1.1 Windows 설치 마법사{Color.END}")
        print(f"{Color.WHITE}=" * 60 + Color.END)
        print(f"{Color.GREEN}🖥️  시스템: {platform.system()} {platform.release()}{Color.END}")
        print(f"{Color.GREEN}🐍 Python: {sys.version.split()[0]}{Color.END}")
        print(f"{Color.WHITE}=" * 60 + Color.END)
    
    def check_python_version(self) -> bool:
        """Python 버전 호환성 확인"""
        print(f"\\n{Color.BLUE}🔍 Python 버전 확인 중...{Color.END}")
        
        if self.python_version < (3, 9):
            print(f"{Color.RED}❌ Python 3.9 이상이 필요합니다. 현재: {sys.version.split()[0]}{Color.END}")
            print(f"{Color.YELLOW}💡 Python 3.9-3.11 설치를 권장합니다.{Color.END}")
            return False
        elif self.python_version >= (3, 12):
            print(f"{Color.YELLOW}⚠️  Python 3.12+는 일부 패키지 호환성 문제가 있을 수 있습니다.{Color.END}")
            print(f"{Color.YELLOW}   Python 3.9-3.11 사용을 권장합니다.{Color.END}")
        
        print(f"{Color.GREEN}✅ Python 버전 호환성 확인 완료{Color.END}")
        return True
    
    def check_pip_and_upgrade(self) -> bool:
        """pip 업그레이드"""
        print(f"\\n{Color.BLUE}📦 pip 업그레이드 중...{Color.END}")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            
            print(f"{Color.GREEN}✅ pip 업그레이드 완료{Color.END}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Color.YELLOW}⚠️  pip 업그레이드 실패, 계속 진행합니다.{Color.END}")
            return False
    
    def install_package_safe(self, package: str, alternatives: List[str] = None) -> bool:
        """안전한 패키지 설치"""
        packages_to_try = [package] + (alternatives or [])
        
        for pkg in packages_to_try:
            try:
                print(f"   📦 {pkg} 설치 중...")
                
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", pkg, "--no-cache-dir"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"   {Color.GREEN}✅ {pkg} 설치 성공{Color.END}")
                    self.success_packages.append(pkg)
                    self.install_log.append(f"SUCCESS: {pkg}")
                    return True
                else:
                    print(f"   {Color.YELLOW}⚠️  {pkg} 설치 실패: {result.stderr[:100]}...{Color.END}")
                    
            except subprocess.TimeoutExpired:
                print(f"   {Color.YELLOW}⏰ {pkg} 설치 시간 초과{Color.END}")
            except Exception as e:
                print(f"   {Color.YELLOW}⚠️  {pkg} 설치 오류: {str(e)[:100]}...{Color.END}")
        
        print(f"   {Color.RED}❌ {package} 및 대안 패키지 설치 실패{Color.END}")
        self.failed_packages.append(package)
        self.install_log.append(f"FAILED: {package}")
        return False
    
    def install_core_packages(self) -> Dict[str, bool]:
        """핵심 패키지 설치"""
        print(f"\\n{Color.BLUE}🔥 Phase 1: 핵심 패키지 설치 중...{Color.END}")
        
        core_packages = {
            "streamlit": ["streamlit>=1.28.0,<2.0.0"],
            "fastapi": ["fastapi>=0.104.0,<1.0.0"],
            "uvicorn": ["uvicorn[standard]>=0.24.0,<1.0.0"],
            "pandas": ["pandas>=2.0.0,<3.0.0"],
            "numpy": ["numpy>=1.24.0,<2.0.0"],
            "Pillow": ["Pillow>=10.0.0,<11.0.0"],
        }
        
        results = {}
        for name, packages in core_packages.items():
            print(f"\\n📋 {name} 설치 중...")
            results[name] = self.install_package_safe(packages[0])
        
        return results
    
    def install_ai_packages(self) -> Dict[str, bool]:
        """AI/ML 패키지 설치"""
        print(f"\\n{Color.BLUE}🔥 Phase 2: AI/ML 패키지 설치 중...{Color.END}")
        
        ai_packages = {
            "openai": ["openai>=1.3.0,<2.0.0"],
            "whisper": ["openai-whisper>=20231117"],
            "torch": ["torch>=2.0.0,<3.0.0", "torch --index-url https://download.pytorch.org/whl/cpu"],
            "torchvision": ["torchvision>=0.15.0,<1.0.0"],
            "torchaudio": ["torchaudio>=2.0.0,<3.0.0"],
        }
        
        results = {}
        for name, packages in ai_packages.items():
            print(f"\\n🧠 {name} 설치 중...")
            results[name] = self.install_package_safe(packages[0], packages[1:])
        
        return results
    
    def install_audio_packages(self) -> Dict[str, bool]:
        """음성 처리 패키지 설치 (Windows 최적화)"""
        print(f"\\n{Color.BLUE}🔥 Phase 3: 음성 처리 패키지 설치 중...{Color.END}")
        
        audio_packages = {
            "soundfile": ["soundfile>=0.12.0,<1.0.0"],
            "pydub": ["pydub>=0.25.0,<1.0.0"],
            "librosa": ["librosa>=0.10.0,<1.0.0"],  # 문제가 될 수 있음
        }
        
        results = {}
        for name, packages in audio_packages.items():
            print(f"\\n🎵 {name} 설치 중...")
            if name == "librosa":
                # librosa는 특별 처리
                result = self.install_librosa_windows()
                results[name] = result
            else:
                results[name] = self.install_package_safe(packages[0])
        
        return results
    
    def install_librosa_windows(self) -> bool:
        """Windows용 librosa 특별 설치"""
        print(f"   {Color.CYAN}🎯 Windows용 librosa 특별 설치 시도...{Color.END}")
        
        # 1차 시도: 표준 설치
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "librosa>=0.10.0", "--no-cache-dir"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"   {Color.GREEN}✅ librosa 표준 설치 성공{Color.END}")
                return True
        except:
            pass
        
        # 2차 시도: 의존성 개별 설치
        print(f"   {Color.YELLOW}🔄 librosa 의존성 개별 설치 시도...{Color.END}")
        dependencies = [
            "numba>=0.51.0",
            "scikit-learn>=0.20.0", 
            "joblib>=0.14",
            "decorator>=4.0.5",
            "six>=1.3",
            "packaging>=20.0",
            "lazy-loader>=0.1"
        ]
        
        for dep in dependencies:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep, "--no-cache-dir"
                ], capture_output=True, text=True, timeout=180)
            except:
                continue
        
        # 3차 시도: 조건부 빌드
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "librosa", "--no-deps", "--force-reinstall"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"   {Color.GREEN}✅ librosa 조건부 설치 성공{Color.END}")
                return True
        except:
            pass
        
        print(f"   {Color.RED}❌ librosa 설치 실패 - 데모 모드에서 우회 처리됩니다{Color.END}")
        return False
    
    def install_vision_packages(self) -> Dict[str, bool]:
        """이미지/OCR 패키지 설치"""
        print(f"\\n{Color.BLUE}🔥 Phase 4: 이미지/OCR 패키지 설치 중...{Color.END}")
        
        vision_packages = {
            "opencv-python": ["opencv-python>=4.8.0,<5.0.0"],
            "opencv-contrib-python": ["opencv-contrib-python>=4.8.0,<5.0.0"],
            "pytesseract": ["pytesseract>=0.3.10,<1.0.0"],
            "easyocr": ["easyocr>=1.7.0,<2.0.0"],
        }
        
        results = {}
        for name, packages in vision_packages.items():
            print(f"\\n👁️ {name} 설치 중...")
            results[name] = self.install_package_safe(packages[0])
        
        return results
    
    def install_document_packages(self) -> Dict[str, bool]:
        """문서 처리 패키지 설치"""
        print(f"\\n{Color.BLUE}🔥 Phase 5: 문서 처리 패키지 설치 중...{Color.END}")
        
        doc_packages = {
            "python-docx": ["python-docx>=1.1.0,<2.0.0"],
            "PyPDF2": ["PyPDF2>=3.0.0,<4.0.0"],
            "openpyxl": ["openpyxl>=3.1.0,<4.0.0"],
            "xlsxwriter": ["xlsxwriter>=3.1.0,<4.0.0"],
        }
        
        results = {}
        for name, packages in doc_packages.items():
            print(f"\\n📄 {name} 설치 중...")
            results[name] = self.install_package_safe(packages[0])
        
        return results
    
    def install_language_packages(self) -> Dict[str, bool]:
        """다국어 처리 패키지 설치 (Windows 안전)"""
        print(f"\\n{Color.BLUE}🔥 Phase 6: 다국어 처리 패키지 설치 중...{Color.END}")
        
        lang_packages = {
            "googletrans": ["googletrans==4.0.0rc1"],
            "langdetect": ["langdetect>=1.0.9,<2.0.0"],
            "nltk": ["nltk>=3.8.0,<4.0.0"],
        }
        
        results = {}
        for name, packages in lang_packages.items():
            print(f"\\n🌍 {name} 설치 중...")
            results[name] = self.install_package_safe(packages[0])
        
        # polyglot은 Windows에서 문제가 많으므로 스킵
        print(f"\\n⚠️  polyglot은 Windows 호환성 문제로 스킵됩니다.")
        results["polyglot"] = False
        
        return results
    
    def install_utility_packages(self) -> Dict[str, bool]:
        """유틸리티 패키지 설치"""
        print(f"\\n{Color.BLUE}🔥 Phase 7: 유틸리티 패키지 설치 중...{Color.END}")
        
        util_packages = {
            "requests": ["requests>=2.31.0,<3.0.0"],
            "aiohttp": ["aiohttp>=3.9.0,<4.0.0"],
            "plotly": ["plotly>=5.17.0,<6.0.0"],
            "matplotlib": ["matplotlib>=3.7.0,<4.0.0"],
            "seaborn": ["seaborn>=0.12.0,<1.0.0"],
            "tqdm": ["tqdm>=4.66.0,<5.0.0"],
            "click": ["click>=8.1.0,<9.0.0"],
            "python-dotenv": ["python-dotenv>=1.0.0,<2.0.0"],
            "scikit-learn": ["scikit-learn>=1.3.0,<2.0.0"],
            "scipy": ["scipy>=1.11.0,<2.0.0"],
            "psutil": ["psutil>=5.9.0,<6.0.0"],
        }
        
        results = {}
        for name, packages in util_packages.items():
            print(f"\\n🛠️ {name} 설치 중...")
            results[name] = self.install_package_safe(packages[0])
        
        return results
    
    def test_critical_imports(self) -> Dict[str, bool]:
        """핵심 모듈 import 테스트"""
        print(f"\\n{Color.BLUE}🧪 핵심 모듈 import 테스트 중...{Color.END}")
        
        critical_modules = {
            "streamlit": "streamlit",
            "pandas": "pandas", 
            "numpy": "numpy",
            "PIL": "PIL",
            "openai": "openai",
            "cv2": "cv2",
            "requests": "requests",
            "matplotlib": "matplotlib",
            "sklearn": "sklearn",
        }
        
        test_results = {}
        for display_name, module_name in critical_modules.items():
            try:
                importlib.import_module(module_name)
                print(f"   {Color.GREEN}✅ {display_name} import 성공{Color.END}")
                test_results[display_name] = True
            except ImportError as e:
                print(f"   {Color.RED}❌ {display_name} import 실패: {str(e)[:50]}...{Color.END}")
                test_results[display_name] = False
            except Exception as e:
                print(f"   {Color.YELLOW}⚠️  {display_name} import 경고: {str(e)[:50]}...{Color.END}")
                test_results[display_name] = False
        
        return test_results
    
    def create_demo_mode_config(self):
        """데모 모드 설정 파일 생성"""
        print(f"\\n{Color.BLUE}⚙️  데모 모드 설정 파일 생성 중...{Color.END}")
        
        config = {
            "demo_mode": True,
            "skip_problematic_imports": True,
            "failed_packages": self.failed_packages,
            "alternative_methods": {
                "librosa": "soundfile + basic audio processing",
                "polyglot": "langdetect + googletrans",
                "advanced_audio": "pydub fallback"
            },
            "installation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "platform": platform.platform()
        }
        
        try:
            import json
            with open("demo_mode_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"   {Color.GREEN}✅ demo_mode_config.json 생성 완료{Color.END}")
            return True
        except Exception as e:
            print(f"   {Color.YELLOW}⚠️  설정 파일 생성 실패: {e}{Color.END}")
            return False
    
    def generate_installation_report(self, all_results: Dict[str, Dict[str, bool]]):
        """설치 결과 리포트 생성"""
        print(f"\\n{Color.CYAN}{Color.BOLD}📊 설치 결과 리포트{Color.END}")
        print(f"{Color.WHITE}=" * 60 + Color.END)
        
        total_packages = 0
        successful_packages = 0
        
        for phase, results in all_results.items():
            phase_success = sum(results.values())
            phase_total = len(results)
            phase_rate = (phase_success / phase_total * 100) if phase_total > 0 else 0
            
            color = Color.GREEN if phase_rate >= 80 else Color.YELLOW if phase_rate >= 50 else Color.RED
            print(f"\\n{color}📦 {phase}: {phase_success}/{phase_total} ({phase_rate:.1f}%){Color.END}")
            
            for package, success in results.items():
                status = "✅" if success else "❌"
                pkg_color = Color.GREEN if success else Color.RED
                print(f"   {status} {pkg_color}{package}{Color.END}")
            
            total_packages += phase_total
            successful_packages += phase_success
        
        overall_rate = (successful_packages / total_packages * 100) if total_packages > 0 else 0
        
        print(f"\\n{Color.CYAN}{Color.BOLD}🎯 전체 성공률: {overall_rate:.1f}% ({successful_packages}/{total_packages}){Color.END}")
        
        if overall_rate >= 80:
            print(f"\\n{Color.GREEN}{Color.BOLD}🎉 설치 완료! 솔로몬드 AI v2.1.1 사용 준비 완료{Color.END}")
            print(f"{Color.GREEN}다음 명령어로 데모를 실행하세요:{Color.END}")
            print(f"{Color.CYAN}python demo_quality_enhanced_v21.py{Color.END}")
        elif overall_rate >= 50:
            print(f"\\n{Color.YELLOW}{Color.BOLD}⚠️  부분 설치 완료 - 데모 모드로 실행 가능{Color.END}")
            print(f"{Color.YELLOW}일부 기능은 제한될 수 있지만 핵심 기능은 사용 가능합니다.{Color.END}")
            print(f"{Color.CYAN}python demo_quality_enhanced_v21.py{Color.END}")
        else:
            print(f"\\n{Color.RED}{Color.BOLD}❌ 설치 실패 - 수동 설치 필요{Color.END}")
            print(f"{Color.RED}다음 패키지들을 수동으로 설치해주세요:{Color.END}")
            for pkg in self.failed_packages[:5]:  # 상위 5개만 표시
                print(f"   {Color.RED}pip install {pkg}{Color.END}")
        
        # 설치 로그 저장
        try:
            with open("installation_log.txt", "w", encoding="utf-8") as f:
                f.write("솔로몬드 AI v2.1.1 Windows 설치 로그\\n")
                f.write("=" * 50 + "\\n")
                f.write(f"설치 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"전체 성공률: {overall_rate:.1f}%\\n\\n")
                
                for log_entry in self.install_log:
                    f.write(log_entry + "\\n")
            
            print(f"\\n{Color.BLUE}📄 상세 로그: installation_log.txt{Color.END}")
        except:
            pass
    
    def run_full_installation(self):
        """전체 설치 프로세스 실행"""
        print(f"\\n{Color.MAGENTA}{Color.BOLD}🚀 솔로몬드 AI v2.1.1 Windows 설치 시작!{Color.END}")
        
        # 시스템 확인
        if not self.check_python_version():
            return False
        
        if not self.is_windows:
            print(f"{Color.YELLOW}⚠️  이 스크립트는 Windows 전용입니다.{Color.END}")
            print(f"{Color.YELLOW}   Linux/macOS에서는 requirements.txt를 사용하세요.{Color.END}")
            return False
        
        # pip 업그레이드
        self.check_pip_and_upgrade()
        
        # 단계별 설치
        all_results = {}
        
        all_results["핵심 패키지"] = self.install_core_packages()
        all_results["AI/ML 패키지"] = self.install_ai_packages()
        all_results["음성 처리"] = self.install_audio_packages()
        all_results["이미지/OCR"] = self.install_vision_packages()
        all_results["문서 처리"] = self.install_document_packages()
        all_results["다국어 처리"] = self.install_language_packages()
        all_results["유틸리티"] = self.install_utility_packages()
        
        # Import 테스트
        all_results["Import 테스트"] = self.test_critical_imports()
        
        # 데모 모드 설정
        self.create_demo_mode_config()
        
        # 결과 리포트
        self.generate_installation_report(all_results)
        
        return True

def main():
    """메인 함수"""
    try:
        installer = WindowsInstaller()
        installer.run_full_installation()
    except KeyboardInterrupt:
        print(f"\\n{Color.YELLOW}⚠️  사용자에 의해 설치가 중단되었습니다.{Color.END}")
    except Exception as e:
        print(f"\\n{Color.RED}❌ 설치 중 예상치 못한 오류: {e}{Color.END}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
