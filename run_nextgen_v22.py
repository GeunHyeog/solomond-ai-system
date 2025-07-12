"""
🔥 차세대 멀티모달 AI 시스템 v2.2 런처
원클릭 실행으로 GPT-4V + Claude Vision + Gemini + 3D 모델링 완전 통합 시스템 시작

사용법:
python run_nextgen_v22.py

또는 고급 설정:
python run_nextgen_v22.py --mode demo --port 8501 --api-keys-file keys.json
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import webbrowser
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NextGenLauncher:
    """차세대 시스템 런처"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.requirements_checked = False
        self.system_validated = False
        
        # 지원하는 실행 모드
        self.modes = {
            "demo": {
                "script": "nextgen_multimodal_demo_v22.py",
                "description": "🔥 차세대 데모 시스템 (Streamlit UI)",
                "command": "streamlit run"
            },
            "api": {
                "script": "api_server.py", 
                "description": "🚀 API 서버 모드",
                "command": "python"
            },
            "cli": {
                "script": "core/nextgen_multimodal_ai_v22.py",
                "description": "💻 CLI 테스트 모드",
                "command": "python"
            },
            "jupyter": {
                "script": "notebooks/nextgen_demo.ipynb",
                "description": "📓 Jupyter 노트북",
                "command": "jupyter notebook"
            }
        }
        
        # 필수 의존성
        self.required_packages = [
            "streamlit>=1.28.0",
            "openai>=1.0.0", 
            "anthropic>=0.7.0",
            "google-generativeai>=0.3.0",
            "pillow>=9.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "plotly>=5.0.0",
            "opencv-python>=4.5.0",
            "trimesh>=3.15.0",
            "open3d>=0.16.0"
        ]
        
        # 선택적 의존성 (3D 모델링)
        self.optional_packages = [
            "trimesh[easy]>=3.15.0",
            "open3d>=0.16.0", 
            "pyrender>=0.1.43",
            "pyglet<2.0"
        ]
    
    def display_banner(self):
        """시작 배너 표시"""
        banner = """
        🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
        🔥                                                      🔥
        🔥     차세대 주얼리 AI 플랫폼 v2.2 런처                🔥  
        🔥                                                      🔥
        🔥  💎 GPT-4V + Claude Vision + Gemini 통합           🔥
        🔥  🎨 실시간 3D 모델링 + Rhino 연동                  🔥
        🔥  🇰🇷 한국어 경영진 요약 + 비즈니스 인사이트          🔥
        🔥  ⚡ 실시간 품질 향상 + 멀티모달 분석                🔥
        🔥                                                      🔥
        🔥  Powered by Solomond AI Systems                     🔥
        🔥                                                      🔥
        🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
        """
        print(banner)
        print("\n🚀 차세대 멀티모달 AI 시스템을 시작합니다...\n")
    
    def check_python_version(self):
        """Python 버전 확인"""
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("❌ Python 3.8 이상이 필요합니다.")
            logger.error(f"현재 버전: {sys.version}")
            return False
        
        logger.info(f"✅ Python 버전 확인: {sys.version.split()[0]}")
        return True
    
    def check_requirements(self, install_missing: bool = True):
        """의존성 패키지 확인 및 설치"""
        logger.info("🔍 의존성 패키지 확인 중...")
        
        missing_packages = []
        
        for package in self.required_packages:
            package_name = package.split(">=")[0].split("==")[0]
            try:
                __import__(package_name.replace("-", "_"))
                logger.info(f"✅ {package_name} 설치됨")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"⚠️  {package_name} 누락")
        
        if missing_packages:
            if install_missing:
                logger.info("📦 누락된 패키지 설치 중...")
                return self._install_packages(missing_packages)
            else:
                logger.error(f"❌ 누락된 패키지: {missing_packages}")
                return False
        
        logger.info("✅ 모든 필수 패키지 확인 완료")
        self.requirements_checked = True
        return True
    
    def _install_packages(self, packages: List[str]) -> bool:
        """패키지 설치"""
        try:
            # pip 업그레이드
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # 필수 패키지 설치
            for package in packages:
                logger.info(f"📦 설치 중: {package}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            logger.info("✅ 패키지 설치 완료")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 패키지 설치 실패: {e}")
            return False
    
    def install_optional_packages(self):
        """선택적 패키지 설치 (3D 모델링)"""
        logger.info("🎨 3D 모델링 패키지 설치 중...")
        
        try:
            for package in self.optional_packages:
                logger.info(f"📦 설치 중: {package}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            logger.info("✅ 3D 모델링 패키지 설치 완료")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ 3D 모델링 패키지 설치 중 오류 (시뮬레이션 모드로 실행됩니다): {e}")
            return False
    
    def validate_system(self):
        """시스템 검증"""
        logger.info("🔍 시스템 환경 검증 중...")
        
        # 핵심 파일들 존재 확인
        required_files = [
            "nextgen_multimodal_demo_v22.py",
            "core/nextgen_multimodal_ai_v22.py", 
            "core/jewelry_3d_modeling_v22.py",
            "core/quality_analyzer_v21.py",
            "core/multilingual_processor_v21.py",
            "core/korean_summary_engine_v21.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.base_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ 누락된 파일들: {missing_files}")
            return False
        
        logger.info("✅ 시스템 파일 검증 완료")
        
        # 메모리/디스크 공간 확인
        import shutil
        free_space_gb = shutil.disk_usage(self.base_dir).free // (1024**3)
        
        if free_space_gb < 2:
            logger.warning(f"⚠️ 디스크 공간 부족: {free_space_gb}GB (권장: 2GB 이상)")
        else:
            logger.info(f"✅ 디스크 공간 충분: {free_space_gb}GB")
        
        self.system_validated = True
        return True
    
    def load_api_keys(self, api_keys_file: Optional[str] = None) -> Dict[str, str]:
        """API 키 로드"""
        api_keys = {}
        
        # 파일에서 로드
        if api_keys_file and Path(api_keys_file).exists():
            try:
                with open(api_keys_file, 'r') as f:
                    api_keys = json.load(f)
                logger.info(f"✅ API 키 파일 로드: {api_keys_file}")
            except Exception as e:
                logger.warning(f"⚠️ API 키 파일 로드 실패: {e}")
        
        # 환경변수에서 로드
        env_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"), 
            "google": os.getenv("GOOGLE_API_KEY")
        }
        
        for key, value in env_keys.items():
            if value:
                api_keys[key] = value
                logger.info(f"✅ 환경변수에서 {key} API 키 로드")
        
        if not api_keys:
            logger.info("💡 API 키가 없습니다. 데모 모드로 실행됩니다.")
        else:
            logger.info(f"🔑 API 키 로드 완료: {list(api_keys.keys())}")
        
        return api_keys
    
    def run_mode(self, mode: str, port: int = 8501, api_keys: Dict[str, str] = None, **kwargs):
        """지정된 모드로 실행"""
        if mode not in self.modes:
            logger.error(f"❌ 지원하지 않는 모드: {mode}")
            logger.info(f"지원 모드: {list(self.modes.keys())}")
            return False
        
        mode_info = self.modes[mode]
        script_path = self.base_dir / mode_info["script"]
        
        if not script_path.exists():
            logger.error(f"❌ 스크립트 파일 없음: {script_path}")
            return False
        
        logger.info(f"🚀 {mode_info['description']} 시작...")
        
        # 환경변수 설정
        env = os.environ.copy()
        if api_keys:
            if "openai" in api_keys:
                env["OPENAI_API_KEY"] = api_keys["openai"]
            if "anthropic" in api_keys:
                env["ANTHROPIC_API_KEY"] = api_keys["anthropic"]
            if "google" in api_keys:
                env["GOOGLE_API_KEY"] = api_keys["google"]
        
        try:
            if mode == "demo":
                # Streamlit 실행
                cmd = [
                    "streamlit", "run", str(script_path),
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"
                ]
                
                # Streamlit 프로세스 시작
                process = subprocess.Popen(cmd, env=env)
                
                # 잠시 대기 후 브라우저 열기
                time.sleep(3)
                url = f"http://localhost:{port}"
                logger.info(f"🌐 브라우저에서 열기: {url}")
                webbrowser.open(url)
                
                # 프로세스 대기
                try:
                    process.wait()
                except KeyboardInterrupt:
                    logger.info("\n👋 시스템 종료 중...")
                    process.terminate()
                    
            elif mode == "api":
                # API 서버 실행
                cmd = [sys.executable, str(script_path), "--port", str(port)]
                subprocess.run(cmd, env=env)
                
            elif mode == "cli":
                # CLI 테스트 실행
                cmd = [sys.executable, str(script_path)]
                subprocess.run(cmd, env=env)
                
            elif mode == "jupyter":
                # Jupyter 노트북 실행
                notebook_dir = self.base_dir / "notebooks"
                notebook_dir.mkdir(exist_ok=True)
                
                # 샘플 노트북 생성
                self._create_sample_notebook(notebook_dir / "nextgen_demo.ipynb")
                
                cmd = ["jupyter", "notebook", str(notebook_dir)]
                subprocess.run(cmd, env=env)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 실행 중 오류: {e}")
            return False
    
    def _create_sample_notebook(self, notebook_path: Path):
        """샘플 Jupyter 노트북 생성"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🔥 차세대 주얼리 AI 플랫폼 v2.2\n",
                        "\n",
                        "이 노트북에서 차세대 멀티모달 AI 시스템을 체험해보세요.\n",
                        "\n",
                        "## 주요 기능\n",
                        "- 🤖 GPT-4V + Claude Vision + Gemini 동시 분석\n",
                        "- 🎨 실시간 3D 주얼리 모델링\n",
                        "- 💎 Rhino 호환 파일 생성\n",
                        "- 🇰🇷 한국어 경영진 요약"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 차세대 시스템 import\n",
                        "import sys\n",
                        "sys.path.append('../')\n",
                        "\n",
                        "from core.nextgen_multimodal_ai_v22 import get_nextgen_multimodal_ai, get_nextgen_capabilities\n",
                        "from core.jewelry_3d_modeling_v22 import get_jewelry_3d_modeler, get_3d_modeling_capabilities\n",
                        "\n",
                        "print('🔥 차세대 주얼리 AI 플랫폼 v2.2 로드 완료!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# 시스템 기능 확인\n",
                        "nextgen_capabilities = get_nextgen_capabilities()\n",
                        "modeling_capabilities = get_3d_modeling_capabilities()\n",
                        "\n",
                        "print('🚀 차세대 AI 기능:')\n",
                        "for key, value in nextgen_capabilities.items():\n",
                        "    print(f'  {key}: {value}')\n",
                        "\n",
                        "print('\\n🎨 3D 모델링 기능:')\n",
                        "for key, value in modeling_capabilities.items():\n",
                        "    print(f'  {key}: {value}')"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python", 
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    
    def interactive_setup(self):
        """대화형 설정"""
        print("\n🔧 차세대 시스템 설정을 시작합니다...")
        
        # 모드 선택
        print("\n📋 실행 모드를 선택하세요:")
        for i, (mode, info) in enumerate(self.modes.items(), 1):
            print(f"  {i}. {info['description']}")
        
        while True:
            try:
                choice = input("\n선택 (1-4): ").strip()
                mode_index = int(choice) - 1
                if 0 <= mode_index < len(self.modes):
                    selected_mode = list(self.modes.keys())[mode_index]
                    break
                else:
                    print("❌ 잘못된 선택입니다. 다시 입력해주세요.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
        
        # 포트 설정 (데모 모드인 경우)
        port = 8501
        if selected_mode == "demo":
            port_input = input(f"\n포트 번호 (기본값: {port}): ").strip()
            if port_input:
                try:
                    port = int(port_input)
                except ValueError:
                    print(f"❌ 잘못된 포트 번호. 기본값 {port}을 사용합니다.")
        
        # API 키 설정
        print("\n🔑 API 키 설정 (선택사항):")
        print("  API 키가 없어도 데모 모드로 실행 가능합니다.")
        
        api_keys = {}
        for api_name in ["openai", "anthropic", "google"]:
            key = input(f"{api_name.upper()} API Key (선택사항): ").strip()
            if key:
                api_keys[api_name] = key
        
        # 3D 모델링 패키지 설치 여부
        install_3d = input("\n🎨 3D 모델링 패키지를 설치하시겠습니까? (y/N): ").strip().lower()
        if install_3d in ['y', 'yes']:
            self.install_optional_packages()
        
        return selected_mode, port, api_keys
    
    def run_full_setup(self, mode: str = "demo", port: int = 8501, 
                      api_keys_file: Optional[str] = None,
                      interactive: bool = False, install_missing: bool = True):
        """전체 설정 및 실행"""
        
        self.display_banner()
        
        # 1. Python 버전 확인
        if not self.check_python_version():
            return False
        
        # 2. 의존성 확인
        if not self.check_requirements(install_missing):
            return False
        
        # 3. 시스템 검증
        if not self.validate_system():
            return False
        
        # 4. 대화형 설정 (필요한 경우)
        if interactive:
            mode, port, api_keys = self.interactive_setup()
        else:
            api_keys = self.load_api_keys(api_keys_file)
        
        # 5. 시스템 실행
        logger.info("🚀 차세대 멀티모달 AI 시스템 시작...")
        success = self.run_mode(mode, port, api_keys)
        
        if success:
            logger.info("✅ 시스템이 성공적으로 실행되었습니다!")
        else:
            logger.error("❌ 시스템 실행 실패")
        
        return success

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="🔥 차세대 주얼리 AI 플랫폼 v2.2 런처")
    
    parser.add_argument("--mode", default="demo", choices=["demo", "api", "cli", "jupyter"],
                       help="실행 모드 (기본값: demo)")
    parser.add_argument("--port", type=int, default=8501,
                       help="포트 번호 (기본값: 8501)")
    parser.add_argument("--api-keys-file", type=str,
                       help="API 키 JSON 파일 경로")
    parser.add_argument("--interactive", action="store_true",
                       help="대화형 설정 모드")
    parser.add_argument("--no-install", action="store_true",
                       help="누락된 패키지 자동 설치 비활성화")
    parser.add_argument("--install-3d", action="store_true",
                       help="3D 모델링 패키지 설치")
    
    args = parser.parse_args()
    
    # 런처 인스턴스 생성
    launcher = NextGenLauncher()
    
    # 3D 모델링 패키지 설치 (요청된 경우)
    if args.install_3d:
        launcher.install_optional_packages()
    
    # 전체 설정 및 실행
    success = launcher.run_full_setup(
        mode=args.mode,
        port=args.port,
        api_keys_file=args.api_keys_file,
        interactive=args.interactive,
        install_missing=not args.no_install
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
