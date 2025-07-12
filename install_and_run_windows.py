#!/usr/bin/env python3
"""
Windows 자동 설치 및 실행 스크립트 v2.1.2
모든 필요한 파일을 다운로드하고 설치합니다.

사용법:
  python install_and_run_windows.py
"""

import os
import sys
import subprocess
import urllib.request
import tempfile
import shutil
from pathlib import Path
import json

class WindowsInstaller:
    def __init__(self):
        self.current_dir = Path.cwd()
        self.github_base = "https://raw.githubusercontent.com/GeunHyeog/solomond-ai-system/main"
        self.required_files = [
            "requirements_windows_v212.txt",
            "simple_large_file_test.py", 
            "run_large_file_test.py",
            "run_full_test_simulation.py"
        ]
        
    def print_header(self):
        print("🎯 주얼리 AI 플랫폼 v2.1 - Windows 자동 설치기")
        print("=" * 60)
        print("🔧 시스템 요구사항 확인 및 자동 설치를 시작합니다.")
        print("=" * 60)
        
    def check_python_version(self):
        """Python 버전 확인"""
        version = sys.version_info
        print(f"🐍 Python 버전: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 이상이 필요합니다.")
            return False
        
        print("✅ Python 버전 확인 완료")
        return True
        
    def download_file(self, filename: str) -> bool:
        """GitHub에서 파일 다운로드"""
        url = f"{self.github_base}/{filename}"
        local_path = self.current_dir / filename
        
        try:
            print(f"📥 다운로드 중: {filename}")
            urllib.request.urlretrieve(url, local_path)
            print(f"✅ 다운로드 완료: {filename}")
            return True
        except Exception as e:
            print(f"❌ 다운로드 실패 {filename}: {e}")
            return False
    
    def download_required_files(self):
        """필수 파일들 다운로드"""
        print("\n1️⃣ 필수 파일 다운로드")
        success_count = 0
        
        for filename in self.required_files:
            if self.download_file(filename):
                success_count += 1
        
        print(f"\n📊 다운로드 결과: {success_count}/{len(self.required_files)} 성공")
        return success_count == len(self.required_files)
    
    def install_basic_requirements(self):
        """기본 필수 패키지만 설치"""
        print("\n2️⃣ 기본 패키지 설치")
        
        basic_packages = [
            "numpy",
            "opencv-python", 
            "psutil",
            "Pillow",
            "matplotlib"
        ]
        
        for package in basic_packages:
            try:
                print(f"📦 설치 중: {package}")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--user"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ 설치 성공: {package}")
                else:
                    print(f"⚠️ 설치 실패: {package} - {result.stderr}")
            except Exception as e:
                print(f"❌ 설치 오류 {package}: {e}")
    
    def install_optional_packages(self):
        """선택적 패키지 설치 (실패해도 계속 진행)"""
        print("\n3️⃣ 고급 패키지 설치 (선택적)")
        
        optional_packages = [
            "openai-whisper",
            "librosa", 
            "soundfile",
            "moviepy",
            "pytesseract"
        ]
        
        for package in optional_packages:
            try:
                print(f"📦 설치 시도: {package}")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--user"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ 설치 성공: {package}")
                else:
                    print(f"⚠️ 설치 건너뜀: {package} (선택 사항)")
            except subprocess.TimeoutExpired:
                print(f"⏰ 설치 시간 초과: {package} (건너뜀)")
            except Exception as e:
                print(f"⚠️ 설치 건너뜀: {package} - {str(e)}")
    
    def create_simple_test_script(self):
        """간단한 테스트 스크립트 생성"""
        script_content = '''#!/usr/bin/env python3
"""
Windows 간단 테스트 실행기
"""

import os
import sys
import time
import json
from datetime import datetime

def run_simple_test():
    """간단한 시스템 테스트"""
    print("🎯 간단 시스템 테스트 시작")
    print("=" * 40)
    
    # Python 모듈 테스트
    modules_to_test = [
        ("numpy", "import numpy as np"),
        ("opencv", "import cv2"),
        ("psutil", "import psutil"),
        ("PIL", "from PIL import Image"),
        ("matplotlib", "import matplotlib.pyplot as plt")
    ]
    
    results = {"test_time": datetime.now().isoformat(), "modules": {}}
    
    for name, import_code in modules_to_test:
        try:
            exec(import_code)
            print(f"✅ {name}: 사용 가능")
            results["modules"][name] = "OK"
        except ImportError as e:
            print(f"❌ {name}: 사용 불가 - {e}")
            results["modules"][name] = f"ERROR: {e}"
    
    # 시스템 정보
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\\n💻 시스템 정보:")
        print(f"   메모리: {memory.total // (1024**3)}GB (사용률: {memory.percent}%)")
        print(f"   CPU 코어: {psutil.cpu_count()}")
        
        results["system"] = {
            "memory_gb": memory.total // (1024**3),
            "memory_percent": memory.percent,
            "cpu_cores": psutil.cpu_count()
        }
    except:
        print("⚠️ 시스템 정보를 가져올 수 없습니다.")
    
    # 간단한 이미지 처리 테스트
    try:
        import numpy as np
        import cv2
        
        # 간단한 이미지 생성
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [100, 150, 200]
        
        cv2.putText(test_image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("test_output.png", test_image)
        
        if os.path.exists("test_output.png"):
            print("✅ 이미지 처리 테스트: 성공")
            results["image_processing"] = "OK"
            os.remove("test_output.png")
        else:
            print("❌ 이미지 처리 테스트: 실패")
            results["image_processing"] = "FAIL"
            
    except Exception as e:
        print(f"❌ 이미지 처리 테스트: 실패 - {e}")
        results["image_processing"] = f"ERROR: {e}"
    
    # 결과 저장
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n📋 테스트 결과가 test_results.json에 저장되었습니다.")
    print("🎉 간단 테스트 완료!")
    
    return results

if __name__ == "__main__":
    try:
        run_simple_test()
    except KeyboardInterrupt:
        print("\\n⏹️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\\n❌ 오류 발생: {e}")
'''
        
        script_path = self.current_dir / "windows_simple_test.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ 간단 테스트 스크립트 생성: {script_path}")
    
    def run_installation(self):
        """전체 설치 과정 실행"""
        try:
            self.print_header()
            
            # Python 버전 확인
            if not self.check_python_version():
                return False
            
            # 파일 다운로드
            if not self.download_required_files():
                print("⚠️ 일부 파일 다운로드에 실패했지만 계속 진행합니다.")
            
            # 패키지 설치
            self.install_basic_requirements()
            self.install_optional_packages()
            
            # 간단 테스트 스크립트 생성
            self.create_simple_test_script()
            
            print("\n" + "=" * 60)
            print("🎉 설치 완료!")
            print("=" * 60)
            print("📝 사용 가능한 명령어:")
            print("   python windows_simple_test.py          # 시스템 테스트")
            print("   python simple_large_file_test.py       # 간단 파일 테스트")
            print("   python run_full_test_simulation.py     # 전체 테스트 시뮬레이션")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ 설치 중 오류 발생: {e}")
            return False

def main():
    """메인 함수"""
    installer = WindowsInstaller()
    
    try:
        success = installer.run_installation()
        
        if success:
            print("\n🚀 설치가 완료되었습니다!")
            
            # 바로 간단 테스트 실행할지 물어보기
            choice = input("\n간단한 시스템 테스트를 지금 실행하시겠습니까? (y/N): ").strip().lower()
            
            if choice == 'y':
                print("\n🧪 간단 테스트 실행...")
                try:
                    import subprocess
                    subprocess.run([sys.executable, "windows_simple_test.py"])
                except Exception as e:
                    print(f"테스트 실행 오류: {e}")
        else:
            print("\n❌ 설치에 문제가 있었습니다.")
            
    except KeyboardInterrupt:
        print("\n⏹️ 설치가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()
