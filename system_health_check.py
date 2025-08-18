#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 상태 점검 및 복원 스크립트
모듈화 후 사용 불가능해진 기능들을 자동으로 감지하고 복원
"""

import os
import sys
import importlib
from pathlib import Path
import subprocess
import json
from typing import Dict, List, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_core_imports():
    """핵심 모듈 import 상태 확인"""
    print("[CHECK] 핵심 모듈 Import 상태 점검...")
    
    core_modules = {
        'real_analysis_engine': 'core.real_analysis_engine',
        'comprehensive_message_extractor': 'core.comprehensive_message_extractor', 
        'audio_converter': 'core.audio_converter',
        'performance_monitor': 'core.performance_monitor',
        'solomond_ai': 'solomond_ai',
        'config_manager': 'solomond_ai.utils.config_manager'
    }
    
    results = {}
    
    for name, module_path in core_modules.items():
        try:
            importlib.import_module(module_path)
            results[name] = {"status": "[OK] OK", "error": None}
            print(f"  {name}: [OK] 정상")
        except ImportError as e:
            results[name] = {"status": "[FAILED] FAILED", "error": str(e)}
            print(f"  {name}: [FAILED] 실패 - {e}")
        except Exception as e:
            results[name] = {"status": "[WARNING] WARNING", "error": str(e)}
            print(f"  {name}: [WARNING] 경고 - {e}")
    
    return results

def check_streamlit_ui_functions():
    """Streamlit UI 함수들의 정의 상태 확인"""
    print("\n[UI] Streamlit UI 함수 상태 점검...")
    
    ui_file_path = project_root / "jewelry_stt_ui_v23_real.py"
    
    if not ui_file_path.exists():
        print("  ❌ UI 파일을 찾을 수 없습니다")
        return {"ui_file_exists": False}
    
    with open(ui_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 필수 함수들 체크
    required_functions = [
        'render_step1_basic_info',
        'render_step2_upload', 
        'render_step3_review',
        'render_step4_report',
        'render_youtube_tab',
        'render_browser_search_tab',
        'render_navigation_bar',
        '_can_proceed_to_next_step'
    ]
    
    function_status = {}
    
    for func_name in required_functions:
        if f"def {func_name}" in content:
            function_status[func_name] = "[OK] 정의됨"
            print(f"  {func_name}: [OK] 정상")
        else:
            function_status[func_name] = "[MISSING] 누락"
            print(f"  {func_name}: [MISSING] 누락")
    
    return {
        "ui_file_exists": True,
        "functions": function_status
    }

def check_session_state_structure():
    """세션 상태 구조 일관성 확인"""
    print("\n🔄 세션 상태 구조 점검...")
    
    ui_file_path = project_root / "jewelry_stt_ui_v23_real.py"
    
    with open(ui_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 세션 상태 초기화 패턴들 확인
    session_patterns = [
        "'workflow_step'",
        "'project_info'",
        "'uploaded_files_data'",
        "'analysis_results'",
        "'final_report'"
    ]
    
    init_patterns = {
        "'uploaded_files_data': []": "❌ 잘못된 초기화 (list)",
        "'uploaded_files_data': {}": "✅ 올바른 초기화 (dict)"
    }
    
    print("  세션 상태 변수:")
    for pattern in session_patterns:
        if pattern in content:
            print(f"    {pattern}: ✅ 존재")
        else:
            print(f"    {pattern}: ❌ 누락")
    
    print("  초기화 패턴:")
    for pattern, status in init_patterns.items():
        if pattern in content:
            print(f"    {pattern}: {status}")
    
    return True

def check_workflow_navigation():
    """워크플로우 네비게이션 로직 확인"""
    print("\n🧭 워크플로우 네비게이션 점검...")
    
    ui_file_path = project_root / "jewelry_stt_ui_v23_real.py"
    
    with open(ui_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 워크플로우 단계 전환 로직 확인
    workflow_checks = {
        "workflow_step == 1": "1단계 조건문",
        "workflow_step == 2": "2단계 조건문", 
        "workflow_step == 3": "3단계 조건문",
        "workflow_step == 4": "4단계 조건문",
        "st.session_state.workflow_step = 2": "2단계 전환",
        "st.session_state.workflow_step = 3": "3단계 전환",
        "st.session_state.workflow_step = 4": "4단계 전환"
    }
    
    for pattern, description in workflow_checks.items():
        if pattern in content:
            print(f"  {description}: ✅ 존재")
        else:
            print(f"  {description}: ❌ 누락")
    
    return True

def check_modular_system_integration():
    """모듈러 시스템 통합 상태 확인"""
    print("\n🧩 모듈러 시스템 통합 점검...")
    
    try:
        from solomond_ai import SolomondAI
        from solomond_ai.utils import ConfigManager
        
        # 간단한 초기화 테스트
        config = ConfigManager()
        app = SolomondAI(domain="jewelry", engines=["image"], theme="jewelry")
        
        print("  모듈러 시스템 초기화: ✅ 성공")
        print(f"  도메인: {app.domain}")
        print(f"  엔진: {app.engines}")
        print(f"  테마: {app.theme}")
        
        return {"status": "✅ 정상", "domain": app.domain, "engines": app.engines}
        
    except Exception as e:
        print(f"  모듈러 시스템 초기화: ❌ 실패 - {e}")
        return {"status": "❌ 실패", "error": str(e)}

def check_file_dependencies():
    """파일 의존성 확인"""
    print("\n📁 파일 의존성 점검...")
    
    required_files = [
        "jewelry_stt_ui_v23_real.py",
        "solomond_ai/__init__.py",
        "solomond_ai/utils/config_manager.py",
        "core/real_analysis_engine.py",
        "setup.py",
        "requirements_v23_windows.txt"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  {file_path}: ✅ 존재")
        else:
            print(f"  {file_path}: ❌ 누락")
            missing_files.append(file_path)
    
    return {"missing_files": missing_files}

def generate_health_report():
    """전체 시스템 상태 보고서 생성"""
    print("\n📊 전체 시스템 상태 보고서 생성 중...")
    
    report = {
        "timestamp": "2025-07-29T01:50:00Z",
        "system_name": "솔로몬드 AI v2.4",
        "checks": {}
    }
    
    # 각 점검 항목 실행
    report["checks"]["core_imports"] = check_core_imports()
    report["checks"]["ui_functions"] = check_streamlit_ui_functions()
    report["checks"]["session_state"] = check_session_state_structure()
    report["checks"]["workflow_navigation"] = check_workflow_navigation()
    report["checks"]["modular_integration"] = check_modular_system_integration()
    report["checks"]["file_dependencies"] = check_file_dependencies()
    
    # 보고서 저장
    report_file = project_root / "SYSTEM_HEALTH_REPORT.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📋 보고서가 저장되었습니다: {report_file}")
    
    # 요약 출력
    print("\n" + "="*50)
    print("🎯 시스템 상태 요약")
    print("="*50)
    
    # 핵심 모듈 상태
    core_ok = sum(1 for v in report["checks"]["core_imports"].values() if v["status"] == "✅ OK")
    core_total = len(report["checks"]["core_imports"])
    print(f"핵심 모듈: {core_ok}/{core_total} 정상")
    
    # UI 함수 상태  
    if report["checks"]["ui_functions"]["ui_file_exists"]:
        ui_ok = sum(1 for v in report["checks"]["ui_functions"]["functions"].values() if v == "✅ 정의됨")
        ui_total = len(report["checks"]["ui_functions"]["functions"])
        print(f"UI 함수: {ui_ok}/{ui_total} 정상")
    else:
        print("UI 함수: ❌ UI 파일 누락")
    
    # 모듈러 시스템
    modular_status = report["checks"]["modular_integration"]["status"]
    print(f"모듈러 시스템: {modular_status}")
    
    # 파일 의존성
    missing_count = len(report["checks"]["file_dependencies"]["missing_files"])
    if missing_count == 0:
        print("파일 의존성: ✅ 모든 파일 존재")
    else:
        print(f"파일 의존성: ❌ {missing_count}개 파일 누락")
    
    # 전체 점수 계산
    total_issues = 0
    total_issues += core_total - core_ok
    if report["checks"]["ui_functions"]["ui_file_exists"]:
        total_issues += len(report["checks"]["ui_functions"]["functions"]) - ui_ok
    total_issues += missing_count
    
    if total_issues == 0:
        print("\n🎉 전체 시스템 상태: 모든 기능 정상!")
    else:
        print(f"\n⚠️ 전체 시스템 상태: {total_issues}개 문제 발견")
    
    return report

if __name__ == "__main__":
    print("[DEBUG] 솔로몬드 AI 시스템 상태 점검 시작...")
    print("="*60)
    
    try:
        report = generate_health_report()
        
        print("\n✅ 시스템 점검이 완료되었습니다!")
        print("상세한 내용은 SYSTEM_HEALTH_REPORT.json 파일을 확인하세요.")
        
    except Exception as e:
        print(f"\n❌ 시스템 점검 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()