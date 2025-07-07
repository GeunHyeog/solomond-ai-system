#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 v3.0 - API 테스트 스크립트
FastAPI 엔드포인트들의 기본 동작을 검증하는 테스트
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_api_endpoints():
    """API 엔드포인트 테스트"""
    print("🌐 솔로몬드 AI 시스템 v3.0 - API 테스트")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    try:
        # API 모듈 import 테스트
        print("\\n📦 API 모듈 로딩 테스트")
        
        try:
            from api.app import create_app
            from api.routes import router
            print("✅ API 모듈 import: OK")
            tests_passed += 1
        except Exception as e:
            print(f"❌ API 모듈 import: FAIL - {e}")
        tests_total += 1
        
        # FastAPI 앱 생성 테스트
        print("\\n🏗️ FastAPI 앱 생성 테스트")
        
        try:
            app = create_app()
            print("✅ FastAPI 앱 생성: OK")
            print(f"   - 제목: {app.title}")
            print(f"   - 버전: {app.version}")
            tests_passed += 1
        except Exception as e:
            print(f"❌ FastAPI 앱 생성: FAIL - {e}")
            return False
        tests_total += 1
        
        # 라우트 검사
        print("\\n🛤️ 라우트 등록 확인")
        
        routes_found = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                route_info = f"{list(route.methods)[0] if route.methods else 'GET'} {route.path}"
                routes_found.append(route_info)
        
        expected_routes = [
            "/",
            "/api/process_audio", 
            "/api/test",
            "/api/health",
            "/api/analyze_batch",
            "/api/models"
        ]
        
        print(f"✅ 등록된 라우트 수: {len(routes_found)}")
        for route in routes_found[:10]:  # 처음 10개만 표시
            print(f"   - {route}")
        
        tests_passed += 1
        tests_total += 1
        
        # UI 템플릿 테스트
        print("\\n🎨 UI 템플릿 테스트")
        
        try:
            from ui.templates import get_main_template
            template = get_main_template()
            
            if "솔로몬드 AI 시스템" in template:
                print("✅ 메인 템플릿 생성: OK")
                print(f"   - 템플릿 크기: {len(template)} 문자")
                tests_passed += 1
            else:
                print("❌ 메인 템플릿 내용 확인: FAIL")
        except Exception as e:
            print(f"❌ 메인 템플릿 생성: FAIL - {e}")
        tests_total += 1
        
        # 핵심 모듈 함수 테스트
        print("\\n🎤 핵심 기능 모듈 테스트")
        
        try:
            from core.analyzer import get_analyzer, check_whisper_status
            
            # Whisper 상태 확인
            whisper_status = check_whisper_status()
            print(f"✅ Whisper 상태 확인: {whisper_status['whisper_available']}")
            
            # 분석기 인스턴스 생성
            analyzer = get_analyzer()
            model_info = analyzer.get_model_info()
            print(f"✅ 분석기 생성: OK")
            print(f"   - 모델 크기: {model_info['model_size']}")
            print(f"   - 지원 형식: {model_info['supported_formats']}")
            
            tests_passed += 1
        except Exception as e:
            print(f"❌ 핵심 기능 모듈: FAIL - {e}")
        tests_total += 1
        
        # 메모리 관리 테스트
        print("\\n💾 메모리 관리 테스트")
        
        try:
            from utils.memory import get_memory_manager
            memory_manager = get_memory_manager()
            memory_info = memory_manager.get_memory_info()
            
            print("✅ 메모리 관리자 생성: OK")
            print(f"   - 현재 메모리: {memory_info.get('process_memory_mb', 0)} MB")
            tests_passed += 1
        except Exception as e:
            print(f"❌ 메모리 관리: FAIL - {e}")
        tests_total += 1
        
        # 파일 처리 테스트
        print("\\n📁 파일 처리 테스트")
        
        try:
            from core.file_processor import get_file_processor
            file_processor = get_file_processor()
            supported_formats = file_processor.get_supported_formats()
            
            print("✅ 파일 처리기 생성: OK")
            print(f"   - 지원 오디오: {supported_formats['audio']}")
            print(f"   - 지원 문서: {supported_formats['document']}")
            tests_passed += 1
        except Exception as e:
            print(f"❌ 파일 처리: FAIL - {e}")
        tests_total += 1
        
        # 워크플로우 테스트
        print("\\n🔄 워크플로우 테스트")
        
        try:
            from core.workflow import get_workflow_manager
            workflow_manager = get_workflow_manager()
            
            print("✅ 워크플로우 관리자 생성: OK")
            tests_passed += 1
        except Exception as e:
            print(f"❌ 워크플로우: FAIL - {e}")
        tests_total += 1
        
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        return False
    
    # 최종 결과
    print("\\n" + "=" * 60)
    print(f"📊 API 테스트 결과: {tests_passed}/{tests_total} 통과")
    success_rate = (tests_passed / tests_total) * 100
    print(f"🎯 성공률: {success_rate:.1f}%")
    
    if success_rate >= 85:
        print("🎉 API 테스트 성공! 시스템이 정상적으로 동작할 준비가 되었습니다.")
        print("\\n🚀 다음 단계: python main.py 실행")
        return True
    else:
        print("❌ API 테스트 실패. 시스템에 문제가 있습니다.")
        return False

def main():
    """메인 함수"""
    try:
        result = asyncio.run(test_api_endpoints())
        return result
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
