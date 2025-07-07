#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 v3.0 - Import 테스트 스크립트
모든 모듈의 import 가능성을 검증하는 테스트
"""

import sys
import traceback
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name, description):
    """모듈 import 테스트"""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: OK")
        return True
    except Exception as e:
        print(f"❌ {description}: FAIL")
        print(f"   오류: {e}")
        return False

def test_from_import(import_statement, description):
    """from import 테스트"""
    try:
        exec(import_statement)
        print(f"✅ {description}: OK")
        return True
    except Exception as e:
        print(f"❌ {description}: FAIL")
        print(f"   오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 솔로몬드 AI 시스템 v3.0 - Import 테스트")
    print("=" * 60)
    
    # 테스트 결과 추적
    tests_passed = 0
    tests_total = 0
    
    # 1. 기본 Python 패키지 테스트
    print("\n📦 1. 기본 의존성 테스트")
    basic_tests = [
        ("fastapi", "FastAPI 프레임워크"),
        ("uvicorn", "Uvicorn ASGI 서버"),
        ("multipart", "Multipart 파일 처리")
    ]
    
    for module, desc in basic_tests:
        tests_total += 1
        if test_import(module, desc):
            tests_passed += 1
    
    # 2. AI 패키지 테스트
    print("\n🎤 2. AI 분석 패키지 테스트")
    ai_tests = [
        ("whisper", "OpenAI Whisper STT"),
        ("psutil", "시스템 모니터링")
    ]
    
    for module, desc in ai_tests:
        tests_total += 1
        if test_import(module, desc):
            tests_passed += 1
    
    # 3. 프로젝트 모듈 테스트
    print("\n🏗️ 3. 프로젝트 모듈 테스트")
    project_tests = [
        ("config", "설정 모듈"),
        ("config.settings", "설정 관리"),
        ("core", "핵심 모듈"),
        ("core.analyzer", "STT 분석 엔진"),
        ("core.file_processor", "파일 처리기"),
        ("core.workflow", "워크플로우 관리"),
        ("api", "API 모듈"),
        ("api.app", "FastAPI 앱"),
        ("api.routes", "API 라우트"),
        ("ui", "UI 모듈"),
        ("ui.templates", "템플릿 관리"),
        ("utils", "유틸리티 모듈"),
        ("utils.logger", "로깅 시스템"),
        ("utils.memory", "메모리 관리")
    ]
    
    for module, desc in project_tests:
        tests_total += 1
        if test_import(module, desc):
            tests_passed += 1
    
    # 4. 기능별 from import 테스트
    print("\n⚙️ 4. 기능별 import 테스트")
    function_tests = [
        ("from core.analyzer import get_analyzer", "STT 분석기 함수"),
        ("from api.app import create_app", "FastAPI 앱 팩토리"),
        ("from ui.templates import get_main_template", "메인 템플릿 함수"),
        ("from utils.logger import get_logger", "로거 함수"),
        ("from utils.memory import get_memory_manager", "메모리 관리자 함수")
    ]
    
    for import_stmt, desc in function_tests:
        tests_total += 1
        if test_from_import(import_stmt, desc):
            tests_passed += 1
    
    # 5. 실제 인스턴스 생성 테스트
    print("\n🔬 5. 인스턴스 생성 테스트")
    try:
        from core.analyzer import get_analyzer
        analyzer = get_analyzer()
        print("✅ STT 분석기 인스턴스 생성: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ STT 분석기 인스턴스 생성: FAIL - {e}")
    tests_total += 1
    
    try:
        from api.app import create_app
        app = create_app()
        print("✅ FastAPI 앱 인스턴스 생성: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ FastAPI 앱 인스턴스 생성: FAIL - {e}")
    tests_total += 1
    
    # 최종 결과
    print("\n" + "=" * 60)
    print(f"📊 테스트 결과: {tests_passed}/{tests_total} 통과")
    success_rate = (tests_passed / tests_total) * 100
    print(f"🎯 성공률: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 테스트 성공! 시스템이 정상적으로 구성되었습니다.")
        return True
    elif success_rate >= 70:
        print("⚠️ 부분 성공. 일부 기능에 문제가 있을 수 있습니다.")
        return False
    else:
        print("❌ 테스트 실패. 시스템 구성에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
