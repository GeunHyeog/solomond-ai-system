"""
솔로몬드 AI v2.3 의존성 충돌 긴급 해결 스크립트
🚨 googletrans + httpx 버전 충돌 해결

실행 방법:
python fix_dependencies.py
"""

import subprocess
import sys
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command):
    """명령어 실행 및 결과 반환"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        logger.error(f"명령어 실행 오류: {e}")
        return False, "", str(e)

def fix_dependency_conflicts():
    """의존성 충돌 해결"""
    
    print("🔥 솔로몬드 AI v2.3 의존성 충돌 긴급 해결")
    print("=" * 60)
    
    # 1. 문제 모듈 제거
    print("\n🚨 STEP 1: 충돌 모듈 제거")
    
    modules_to_remove = [
        "googletrans",
        "httpx", 
        "httpcore",
        "h11"
    ]
    
    for module in modules_to_remove:
        print(f"제거 중: {module}")
        success, stdout, stderr = run_command(f"pip uninstall {module} -y")
        if success:
            print(f"✅ {module} 제거 완료")
        else:
            print(f"⚠️ {module} 제거 실패: {stderr}")
    
    # 2. 호환 가능한 버전 설치
    print("\n🔧 STEP 2: 호환 버전 설치")
    
    compatible_packages = [
        "httpcore==0.18.0",
        "httpx==0.25.2", 
        "h11==0.14.0",
        "googletrans==3.1.0a0"
    ]
    
    for package in compatible_packages:
        print(f"설치 중: {package}")
        success, stdout, stderr = run_command(f"pip install {package}")
        if success:
            print(f"✅ {package} 설치 완료")
        else:
            print(f"❌ {package} 설치 실패: {stderr}")
    
    # 3. 백업 번역 모듈 생성
    print("\n🔄 STEP 3: 백업 번역 모듈 생성")
    
    backup_translator_code = '''"""
백업 번역 모듈 - googletrans 충돌 시 사용
"""

class BackupTranslator:
    """백업 번역기 - 간단한 언어 감지 및 번역"""
    
    def __init__(self):
        self.languages = {
            'ko': 'Korean',
            'en': 'English', 
            'zh': 'Chinese',
            'ja': 'Japanese'
        }
    
    def detect(self, text):
        """간단한 언어 감지"""
        # 한글 문자 확인
        korean_chars = sum(1 for char in text if '가' <= char <= '힣')
        if korean_chars > len(text) * 0.3:
            return type('obj', (object,), {'lang': 'ko'})
        
        # 중국어 문자 확인
        chinese_chars = sum(1 for char in text if '\\u4e00' <= char <= '\\u9fff')
        if chinese_chars > len(text) * 0.1:
            return type('obj', (object,), {'lang': 'zh'})
        
        # 일본어 문자 확인  
        japanese_chars = sum(1 for char in text if '\\u3040' <= char <= '\\u309f' or '\\u30a0' <= char <= '\\u30ff')
        if japanese_chars > len(text) * 0.1:
            return type('obj', (object,), {'lang': 'ja'})
        
        # 기본값: 영어
        return type('obj', (object,), {'lang': 'en'})
    
    def translate(self, text, dest='ko'):
        """백업 번역 (실제 번역 없이 원문 반환)"""
        return type('obj', (object,), {
            'text': f"[백업번역] {text}",
            'src': 'auto',
            'dest': dest
        })

# 글로벌 인스턴스
backup_translator = BackupTranslator()

def get_backup_translator():
    """백업 번역기 반환"""
    return backup_translator
'''
    
    try:
        with open('core/backup_translator.py', 'w', encoding='utf-8') as f:
            f.write(backup_translator_code)
        print("✅ 백업 번역 모듈 생성 완료")
    except Exception as e:
        print(f"❌ 백업 번역 모듈 생성 실패: {e}")
    
    # 4. 다국어 번역기 수정
    print("\n🔧 STEP 4: 다국어 번역기 수정")
    
    try:
        # 기존 파일 읽기
        with open('core/multilingual_translator.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 안전한 import로 수정
        modified_content = content.replace(
            "from googletrans import Translator",
            """try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    from .backup_translator import get_backup_translator
    Translator = get_backup_translator
    GOOGLETRANS_AVAILABLE = False"""
        )
        
        # 파일 저장
        with open('core/multilingual_translator.py', 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("✅ 다국어 번역기 수정 완료")
        
    except Exception as e:
        print(f"❌ 다국어 번역기 수정 실패: {e}")
    
    # 5. 테스트
    print("\n🧪 STEP 5: 수정 사항 테스트")
    
    test_commands = [
        "python -c \"import httpx; print('httpx:', httpx.__version__)\"",
        "python -c \"import httpcore; print('httpcore:', httpcore.__version__)\"", 
        "python -c \"from core.backup_translator import get_backup_translator; print('백업 번역기 OK')\"",
        "python -c \"from core.multilingual_translator import JewelryMultilingualTranslator; print('다국어 번역기 OK')\""
    ]
    
    for cmd in test_commands:
        print(f"테스트: {cmd}")
        success, stdout, stderr = run_command(cmd)
        if success:
            print(f"✅ {stdout.strip()}")
        else:
            print(f"❌ {stderr.strip()}")
    
    # 6. 하이브리드 LLM 매니저 테스트
    print("\n🎯 STEP 6: 하이브리드 LLM 매니저 테스트")
    
    success, stdout, stderr = run_command(
        "python -c \"from core.hybrid_llm_manager_v23 import HybridLLMManagerV23; print('하이브리드 LLM 매니저 OK')\""
    )
    
    if success:
        print("🎉 하이브리드 LLM 매니저 정상 로드!")
        print(stdout.strip())
        return True
    else:
        print("❌ 하이브리드 LLM 매니저 로드 실패:")
        print(stderr.strip())
        return False

def create_requirements_fix():
    """수정된 requirements 파일 생성"""
    
    print("\n📋 STEP 7: 수정된 requirements 생성")
    
    fixed_requirements = """# 솔로몬드 AI v2.3 수정된 의존성
# 의존성 충돌 해결 버전

# 핵심 AI 라이브러리
openai>=1.95.0
anthropic>=0.57.1
google-generativeai>=0.8.5

# HTTP 라이브러리 (호환 버전)
httpcore==0.18.0
httpx==0.25.2
h11==0.14.0

# 번역 라이브러리 (호환 버전)
googletrans==3.1.0a0

# UI 라이브러리
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0

# 오디오/비디오 처리
moviepy>=1.0.3
whisper>=1.0.0

# 시스템 유틸리티
psutil>=5.9.0
asyncio-mqtt>=0.11.0

# 기타 필수 라이브러리
requests>=2.31.0
aiohttp>=3.8.0
python-dotenv>=1.0.0
"""
    
    try:
        with open('requirements_fixed.txt', 'w', encoding='utf-8') as f:
            f.write(fixed_requirements)
        print("✅ requirements_fixed.txt 생성 완료")
        return True
    except Exception as e:
        print(f"❌ requirements 생성 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    
    print("🔥 솔로몬드 AI v2.3 의존성 충돌 긴급 해결 시작")
    print("🚨 발견된 문제: googletrans + httpx 버전 충돌")
    print("=" * 70)
    
    success_count = 0
    total_steps = 3
    
    # 1. 의존성 충돌 해결
    print("\n🔧 의존성 충돌 해결 실행...")
    if fix_dependency_conflicts():
        success_count += 1
        print("✅ 의존성 충돌 해결 완료")
    else:
        print("❌ 의존성 충돌 해결 실패")
    
    # 2. Requirements 파일 생성
    print("\n📋 Requirements 파일 생성...")
    if create_requirements_fix():
        success_count += 1
        print("✅ Requirements 파일 생성 완료")
    else:
        print("❌ Requirements 파일 생성 실패")
    
    # 3. 최종 테스트
    print("\n🧪 최종 시스템 테스트...")
    final_test_success, stdout, stderr = run_command(
        "python -c \"from core.hybrid_llm_manager_v23 import HybridLLMManagerV23; manager = HybridLLMManagerV23(); print('전체 시스템 정상!')\""
    )
    
    if final_test_success:
        success_count += 1
        print("✅ 전체 시스템 정상 작동 확인")
        print(stdout.strip())
    else:
        print("❌ 전체 시스템 테스트 실패")
        print(stderr.strip())
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("🔥 의존성 충돌 해결 결과")
    print("=" * 70)
    
    success_rate = (success_count / total_steps) * 100
    
    print(f"성공률: {success_count}/{total_steps} ({success_rate:.1f}%)")
    
    if success_count == total_steps:
        print("🎉 모든 의존성 충돌 해결 완료!")
        print("✅ 하이브리드 LLM 매니저 정상 작동")
        print("🚀 핫픽스 UI 실행 준비 완료")
        
        print("\n🎯 다음 단계:")
        print("1. streamlit run jewelry_stt_ui_v23_hotfix.py")
        print("2. 멀티파일 업로드 테스트")
        print("3. 실제 AI 분석 확인")
        
        return True
        
    elif success_count >= 2:
        print("⚠️ 부분 성공 - 일부 기능 제한적 사용 가능")
        print("🔧 백업 모드로 작동 가능")
        
        print("\n🔄 백업 실행 방법:")
        print("1. streamlit run jewelry_stt_ui_v23_hotfix.py")
        print("2. 백업 번역 모드로 작동")
        
        return False
        
    else:
        print("❌ 의존성 충돌 해결 실패")
        print("🚨 수동 해결 필요")
        
        print("\n📞 긴급 지원:")
        print("- 전화: 010-2983-0338")
        print("- 이메일: solomond.jgh@gmail.com")
        
        return False

if __name__ == "__main__":
    main()
