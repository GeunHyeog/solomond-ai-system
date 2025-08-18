#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 SOLOMOND AI Agent Interface
Claude Code에서 "/agent solomond-project-manager" 명령어로 호출되는 
공식 에이전트 인터페이스

사용법 (Claude Code에서):
/agent solomond-project-manager analyze
/agent solomond-project-manager health  
/agent solomond-project-manager fix
/agent solomond-project-manager optimize

Author: SOLOMOND AI Team
Version: 2.0.0 (Final Integration)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 통합 프로젝트 매니저 임포트
try:
    from solomond_unified_project_manager import SOLOMONDProjectManager
except ImportError:
    print("ERROR: SOLOMOND Project Manager module not found")
    print("Please ensure solomond_unified_project_manager.py is in the same directory")
    sys.exit(1)

def run_solomond_agent(command: str = "info"):
    """SOLOMOND AI 에이전트 실행"""
    
    # 에이전트 초기화
    try:
        manager = SOLOMONDProjectManager()
        print(f"[SOLOMOND AI] Initializing Project Manager v{manager.version}...")
    except Exception as e:
        print(f"[ERROR] Failed to initialize manager: {e}")
        return False
    
    # 명령어 실행
    try:
        if command == "analyze":
            print("[SOLOMOND AI] Executing comprehensive codebase analysis...")
            result = manager.analyze_codebase()
            print_analysis_summary(result)
            
        elif command == "health":
            print("[SOLOMOND AI] Performing system health diagnosis...")
            result = manager.analyze_codebase()
            health_report = manager.generate_health_report(result)
            
            # 건강도 보고서 저장
            report_file = Path("solomond_health_report.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(health_report)
            
            print_health_summary(result, str(report_file))
            
        elif command == "fix":
            print("[SOLOMOND AI] Analyzing auto-fixable issues...")
            result = manager.analyze_codebase()
            print_fix_summary(result)
            
        elif command == "optimize":
            print("[SOLOMOND AI] Generating optimization recommendations...")
            result = manager.analyze_codebase()
            print_optimization_summary(result)
            
        elif command == "info":
            capabilities = manager.get_agent_capabilities()
            print_agent_info(capabilities)
            
        else:
            print(f"[ERROR] Unknown command: {command}")
            print("Available commands: analyze, health, fix, optimize, info")
            return False
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Command execution failed: {e}")
        return False

def print_analysis_summary(result):
    """분석 결과 요약 출력"""
    summary = result.get("project_summary", {})
    health = result.get("health_assessment", {})
    
    print("\n=== SOLOMOND AI CODEBASE ANALYSIS COMPLETE ===")
    print(f"Files analyzed: {summary.get('files_analyzed', 0)}")
    print(f"Total code lines: {summary.get('total_lines', 0):,}")
    print(f"Issues found: {summary.get('total_issues', 0)}")
    print(f"  - Critical: {summary.get('critical_issues', 0)}")
    print(f"  - High: {summary.get('high_issues', 0)}")
    print(f"  - Medium: {summary.get('medium_issues', 0)}")
    
    if health:
        score = health.get("overall_score", 0)
        status = "EXCELLENT" if score >= 90 else "GOOD" if score >= 80 else "FAIR" if score >= 70 else "NEEDS IMPROVEMENT"
        print(f"System Health: {score}/100 ({status})")
    
    insights = result.get("solomond_specific_insights", [])
    if insights:
        print(f"\nSOLOMOND AI Insights: {len(insights)} recommendations")
    
    if result.get("auto_fix_available"):
        print("\n[AUTO-FIX] Some issues can be automatically fixed")

def print_health_summary(result, report_file):
    """건강도 진단 요약 출력"""
    health = result.get("health_assessment", {})
    
    print("\n=== SOLOMOND AI SYSTEM HEALTH DIAGNOSIS ===")
    score = health.get("overall_score", 0)
    status = "EXCELLENT" if score >= 90 else "GOOD" if score >= 80 else "FAIR" if score >= 70 else "NEEDS IMPROVEMENT"
    
    print(f"Overall Health: {score}/100 ({status})")
    print(f"Critical Issues: {health.get('critical_issues', 0)}")
    print(f"High Priority Issues: {health.get('high_issues', 0)}")
    print(f"Files Analyzed: {health.get('files_analyzed', 0)}")
    
    recommendations = health.get("recommendations", [])
    if recommendations:
        print(f"\nHealth Recommendations:")
        for rec in recommendations[:3]:
            print(f"  - {rec}")
    
    print(f"\nDetailed Report: {report_file}")

def print_fix_summary(result):
    """자동 수정 분석 요약 출력"""
    auto_fixable = [
        issue for issue in result.get("issues_found", [])
        if issue.get("pattern_name", "") in [
            "threadpool_without_context", "memory_leak_cuda", 
            "streamlit_heavy_no_cache", "file_open_no_context"
        ]
    ]
    
    print("\n=== SOLOMOND AI AUTO-FIX ANALYSIS ===")
    print(f"Auto-fixable Issues: {len(auto_fixable)}")
    
    if auto_fixable:
        print("\nFixable Patterns:")
        print("  - ThreadPool context manager missing")
        print("  - GPU memory cleanup missing")
        print("  - Streamlit cache not applied")
        print("  - File context manager missing")
        print("\nTo apply fixes automatically:")
        print("  python agent_interface_safe.py solomond-project-manager fix --auto")
    else:
        print("No auto-fixable issues found. System is stable.")

def print_optimization_summary(result):
    """최적화 제안 요약 출력"""
    recommendations = result.get("optimization_recommendations", [])
    insights = result.get("solomond_specific_insights", [])
    
    print("\n=== SOLOMOND AI OPTIMIZATION ANALYSIS ===")
    
    if recommendations:
        print(f"Optimization Recommendations: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:3], 1):
            priority = rec.get("priority", "").upper()
            title = rec.get("title", "")
            print(f"  {i}. [{priority}] {title}")
    
    if insights:
        print(f"\nSystem Insights: {len(insights)}")
        for insight in insights[:2]:
            title = insight.get("title", "")
            level = insight.get("level", "").upper()
            print(f"  - [{level}] {title}")
    
    print("\nCore Optimization Areas:")
    print("  1. ThreadPool resource safety")
    print("  2. GPU memory management")
    print("  3. Streamlit performance caching")
    print("  4. SOLOMOND AI specific optimizations")

def print_agent_info(capabilities):
    """에이전트 정보 출력"""
    print(f"\n=== {capabilities.get('name', '')} v{capabilities.get('version', '')} ===")
    print(f"Role: {capabilities.get('role', '')}")
    print(f"Serena Integration: {'COMPLETE' if capabilities.get('serena_integrated') else 'INCOMPLETE'}")
    
    print(f"\nAnalysis Capabilities:")
    print(f"  - Analysis Patterns: {capabilities.get('analysis_patterns', 0)}")
    print(f"  - Auto-fix Patterns: {capabilities.get('auto_fix_patterns', 0)}")
    
    print(f"\nCore Features:")
    for feature in capabilities.get("core_capabilities", [])[:5]:
        print(f"  - {feature}")
    
    print(f"\nSOLOMOND AI Specializations:")
    for spec in capabilities.get("solomond_specializations", []):
        print(f"  - {spec}")

def main():
    """메인 함수"""
    # 명령어 인자 처리
    command = sys.argv[1] if len(sys.argv) > 1 else "info"
    
    # SOLOMOND AI 에이전트 실행
    success = run_solomond_agent(command)
    
    if success:
        print(f"\n[SUCCESS] SOLOMOND AI command '{command}' completed")
        return 0
    else:
        print(f"\n[FAILED] SOLOMOND AI command '{command}' failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)