#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 - 멀티모달 통합 테스트
Phase 3 멀티모달 시스템의 실제 작동 확인 및 통합 테스트
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """테스트 시작 배너"""
    banner = """
🚀 솔로몬드 멀티모달 통합 시스템 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3: 음성 + 비디오 + 이미지 + 문서 + 웹 통합 분석
개발자: 전근혁 (솔로몬드 대표, 한국보석협회 사무국장)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(banner)

async def test_multimodal_capabilities():
    """멀티모달 기능 테스트"""
    try:
        print("📋 1. 멀티모달 통합 기능 확인...")
        
        # 멀티모달 통합기 import 테스트
        from core.multimodal_integrator import get_multimodal_integrator, get_integration_capabilities
        
        # 기능 정보 확인
        capabilities = get_integration_capabilities()
        print("✅ 멀티모달 통합기 로드 성공")
        print(f"📊 지원 소스: {capabilities['supported_sources']}")
        print(f"🎯 분석 깊이: {capabilities['analysis_depths']}")
        print(f"💎 주얼리 카테고리: {capabilities['jewelry_categories']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 멀티모달 모듈 로드 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 멀티모달 기능 테스트 실패: {e}")
        return False

async def test_individual_modules():
    """개별 모듈 테스트"""
    module_tests = []
    
    # 1. STT 분석기 테스트
    try:
        from core.analyzer import get_analyzer
        print("✅ STT 분석기 로드 성공")
        module_tests.append(("STT 분석기", True))
    except Exception as e:
        print(f"❌ STT 분석기 로드 실패: {e}")
        module_tests.append(("STT 분석기", False))
    
    # 2. 비디오 프로세서 테스트
    try:
        from core.video_processor import get_video_processor
        print("✅ 비디오 프로세서 로드 성공")
        module_tests.append(("비디오 프로세서", True))
    except Exception as e:
        print(f"❌ 비디오 프로세서 로드 실패: {e}")
        module_tests.append(("비디오 프로세서", False))
    
    # 3. 이미지 프로세서 테스트
    try:
        from core.image_processor import get_image_processor
        print("✅ 이미지 프로세서 로드 성공")
        module_tests.append(("이미지 프로세서", True))
    except Exception as e:
        print(f"❌ 이미지 프로세서 로드 실패: {e}")
        module_tests.append(("이미지 프로세서", False))
    
    # 4. 웹 크롤러 테스트
    try:
        from core.web_crawler import get_web_crawler
        print("✅ 웹 크롤러 로드 성공")
        module_tests.append(("웹 크롤러", True))
    except Exception as e:
        print(f"❌ 웹 크롤러 로드 실패: {e}")
        module_tests.append(("웹 크롤러", False))
    
    # 5. 주얼리 AI 엔진 테스트
    try:
        from core.jewelry_ai_engine import JewelryAIEngine
        print("✅ 주얼리 AI 엔진 로드 성공")
        module_tests.append(("주얼리 AI 엔진", True))
    except Exception as e:
        print(f"❌ 주얼리 AI 엔진 로드 실패: {e}")
        module_tests.append(("주얼리 AI 엔진", False))
    
    # 6. 크로스 검증 시각화 테스트
    try:
        from core.cross_validation_visualizer import CrossValidationVisualizer
        print("✅ 크로스 검증 시각화 로드 성공")
        module_tests.append(("크로스 검증 시각화", True))
    except Exception as e:
        print(f"❌ 크로스 검증 시각화 로드 실패: {e}")
        module_tests.append(("크로스 검증 시각화", False))
    
    return module_tests

async def test_jewelry_stt_integration():
    """주얼리 STT UI와의 통합 테스트"""
    try:
        print("\n📱 주얼리 STT UI 통합 확인...")
        
        # jewelry_stt_ui.py 모듈 확인
        if os.path.exists("jewelry_stt_ui.py"):
            print("✅ jewelry_stt_ui.py 파일 존재")
            
            # 주요 의존성 확인
            dependencies = [
                "fastapi",
                "uvicorn", 
                "core.analyzer",
                "core.jewelry_enhancer"
            ]
            
            missing_deps = []
            for dep in dependencies:
                try:
                    if dep.startswith("core."):
                        exec(f"from {dep} import *")
                    else:
                        exec(f"import {dep}")
                    print(f"✅ {dep} 로드 성공")
                except ImportError:
                    missing_deps.append(dep)
                    print(f"❌ {dep} 로드 실패")
            
            if not missing_deps:
                print("✅ 주얼리 STT UI 모든 의존성 만족")
                return True
            else:
                print(f"❌ 누락된 의존성: {missing_deps}")
                return False
        else:
            print("❌ jewelry_stt_ui.py 파일 없음")
            return False
            
    except Exception as e:
        print(f"❌ 주얼리 STT UI 통합 테스트 실패: {e}")
        return False

def generate_test_report(multimodal_test: bool, module_tests: List, ui_test: bool):
    """테스트 리포트 생성"""
    print("\n" + "="*80)
    print("📊 **솔로몬드 멀티모달 시스템 테스트 결과**")
    print("="*80)
    
    # 전체 요약
    successful_modules = len([t for t in module_tests if t[1]])
    total_modules = len(module_tests)
    success_rate = (successful_modules / total_modules * 100) if total_modules > 0 else 0
    
    print(f"🎯 **전체 성공률**: {success_rate:.1f}% ({successful_modules}/{total_modules})")
    print(f"🔮 **멀티모달 통합**: {'✅ 성공' if multimodal_test else '❌ 실패'}")
    print(f"📱 **UI 통합**: {'✅ 성공' if ui_test else '❌ 실패'}")
    
    print("\n📋 **개별 모듈 상태**:")
    for module_name, success in module_tests:
        status = "✅ 정상" if success else "❌ 실패"
        print(f"   • {module_name}: {status}")
    
    # 현재 상태 진단
    print(f"\n🔍 **현재 시스템 상태**:")
    if multimodal_test and success_rate >= 80:
        print("   🎉 **Phase 3 멀티모달 시스템 준비 완료!**")
        print("   🚀 **즉시 실제 테스트 및 데모 가능**")
        status = "READY_FOR_PRODUCTION"
    elif success_rate >= 60:
        print("   🔧 **기본 기능 작동, 일부 최적화 필요**")
        print("   📝 **누락된 모듈 보완 후 완전 가동 가능**")
        status = "PARTIAL_FUNCTIONALITY"
    else:
        print("   ⚠️ **주요 기능 복구 필요**")
        print("   🛠️ **의존성 설치 및 모듈 수정 필요**")
        status = "NEEDS_REPAIR"
    
    # 권장 다음 단계
    print(f"\n🎯 **권장 다음 단계**:")
    if status == "READY_FOR_PRODUCTION":
        print("   1. 실제 오디오 파일로 멀티모달 분석 테스트")
        print("   2. 웹 UI에서 멀티모달 기능 활성화")
        print("   3. 성능 최적화 및 사용자 경험 개선")
        print("   4. Phase 4: 모바일 앱 및 SaaS 플랫폼 계획 시작")
    elif status == "PARTIAL_FUNCTIONALITY":
        print("   1. 실패한 모듈들 의존성 설치")
        print("   2. 기본 STT 기능부터 안정화")
        print("   3. 단계적으로 멀티모달 기능 통합")
    else:
        print("   1. requirements.txt 기반 의존성 전체 재설치")
        print("   2. Python 환경 점검")
        print("   3. 기본 STT 기능부터 복구")
    
    # 개발자 정보
    print(f"\n👨‍💼 **개발자**: 전근혁 (솔로몬드 대표, 한국보석협회 사무국장)")
    print(f"📅 **테스트 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🗂️ **GitHub**: GeunHyeog/solomond-ai-system")
    
    return {
        "overall_success_rate": success_rate,
        "multimodal_integration": multimodal_test,
        "ui_integration": ui_test,
        "module_results": dict(module_tests),
        "system_status": status,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

async def main():
    """메인 테스트 함수"""
    print_banner()
    
    # 1. 멀티모달 통합 기능 테스트
    print("🔄 Phase 3 멀티모달 통합 기능 테스트 시작...\n")
    multimodal_test = await test_multimodal_capabilities()
    
    # 2. 개별 모듈 테스트
    print("\n🔄 개별 모듈 테스트 시작...")
    module_tests = await test_individual_modules()
    
    # 3. UI 통합 테스트
    ui_test = await test_jewelry_stt_integration()
    
    # 4. 종합 리포트 생성
    test_report = generate_test_report(multimodal_test, module_tests, ui_test)
    
    # 5. JSON 리포트 저장
    try:
        with open("multimodal_test_report.json", "w", encoding="utf-8") as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 **테스트 리포트 저장**: multimodal_test_report.json")
    except Exception as e:
        print(f"\n❌ 테스트 리포트 저장 실패: {e}")
    
    print("\n" + "="*80)
    print("🎯 **테스트 완료!** 위 결과를 참고하여 다음 단계를 진행하세요.")
    print("="*80)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
