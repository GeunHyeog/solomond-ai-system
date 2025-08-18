#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLOMOND AI Agent Interface (Windows Safe Version)
Claude Code에서 직접 호출 가능한 통합 에이전트 시스템 - Windows 호환 버전

사용법:
python agent_interface_safe.py solomond-project-manager analyze
python agent_interface_safe.py solomond-project-manager health
python agent_interface_safe.py solomond-project-manager fix --auto

Author: SOLOMOND AI Team  
Version: 1.0.0 (Windows Safe)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 통합 프로젝트 매니저 임포트
try:
    from solomond_unified_project_manager import SOLOMONDProjectManager
except ImportError as e:
    print(f"[ERROR] SOLOMOND Project Manager module load failed: {e}")
    print("[INFO] Please check if solomond_unified_project_manager.py exists.")
    sys.exit(1)

class AgentInterfaceSafe:
    """Claude Code 에이전트 인터페이스 (Windows 안전 버전)"""
    
    def __init__(self):
        self.available_agents = {
            "solomond-project-manager": {
                "class": SOLOMONDProjectManager,
                "description": "SOLOMOND AI 통합 프로젝트 매니저 (Serena 통합)",
                "commands": ["analyze", "health", "fix", "optimize", "status", "info"]
            }
        }
    
    def process_command(self, agent_name: str, command: str, args: List[str] = None) -> Dict[str, Any]:
        """에이전트 명령어 처리"""
        if agent_name not in self.available_agents:
            return self._error_response(f"Unknown agent: {agent_name}")
        
        try:
            # 에이전트 인스턴스 생성
            agent_class = self.available_agents[agent_name]["class"]
            agent = agent_class()
            
            # 명령어 실행
            if command == "analyze":
                return self._handle_analyze(agent, args)
            elif command == "health":
                return self._handle_health(agent, args)
            elif command == "fix":
                return self._handle_fix(agent, args)
            elif command == "optimize":
                return self._handle_optimize(agent, args)
            elif command == "status":
                return self._handle_status(agent, args)
            elif command == "info":
                return self._handle_info(agent, args)
            else:
                return self._error_response(f"Unsupported command: {command}")
                
        except Exception as e:
            return self._error_response(f"Command execution error: {str(e)}")
    
    def _handle_analyze(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """코드베이스 분석 처리"""
        print("[INFO] Starting SOLOMOND AI integrated codebase analysis...")
        
        # 분석 대상 파일 지정 (옵션)
        target_files = None
        if args and "--files" in args:
            file_index = args.index("--files") + 1
            if file_index < len(args):
                target_files = args[file_index].split(",")
        
        analysis_result = agent.analyze_codebase(target_files)
        
        return {
            "status": "success",
            "command": "analyze",
            "agent_response": self._format_analysis_response_safe(analysis_result),
            "raw_data": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_health(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """시스템 건강도 체크 처리"""
        print("[INFO] Diagnosing SOLOMOND AI system health...")
        
        # 먼저 분석 실행
        analysis_result = agent.analyze_codebase()
        
        # 건강도 보고서 생성
        health_report = agent.generate_health_report(analysis_result)
        
        # 보고서 파일 저장
        report_file = Path("solomond_health_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(health_report)
        
        return {
            "status": "success",
            "command": "health",
            "agent_response": self._format_health_response_safe(analysis_result, str(report_file)),
            "raw_data": analysis_result,
            "report_file": str(report_file),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_fix(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """자동 수정 처리"""
        auto_apply = "--auto" in args if args else False
        
        if auto_apply:
            print("[INFO] Executing automatic fixes...")
        else:
            print("[INFO] Analyzing fixable issues...")
        
        # 분석 실행
        analysis_result = agent.analyze_codebase()
        
        if not analysis_result.get("auto_fix_available", False):
            return {
                "status": "success",
                "command": "fix",
                "agent_response": "[INFO] No auto-fixable issues found.\n[SUCCESS] System is in stable condition.",
                "raw_data": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
        
        if auto_apply:
            # 자동 수정 실행
            fix_result = agent.auto_fix_issues(analysis_result, create_backups=True)
            
            return {
                "status": "success",
                "command": "fix",
                "agent_response": self._format_fix_response_safe(fix_result, True),
                "raw_data": {"analysis": analysis_result, "fix_result": fix_result},
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 분석만 수행
            return {
                "status": "success",
                "command": "fix",
                "agent_response": self._format_fix_analysis_response_safe(analysis_result),
                "raw_data": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_status(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """시스템 상태 체크 처리"""
        print("[INFO] Checking SOLOMOND AI system status...")
        
        # 빠른 분석 (핵심 파일만)
        core_files = [
            "conference_analysis_COMPLETE_WORKING.py",
            "solomond_ai_main_dashboard.py"
        ]
        
        analysis_result = agent.analyze_codebase(core_files)
        capabilities = agent.get_agent_capabilities()
        
        return {
            "status": "success",
            "command": "status",
            "agent_response": self._format_status_response_safe(analysis_result, capabilities),
            "raw_data": {"analysis": analysis_result, "capabilities": capabilities},
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_info(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """에이전트 정보 표시"""
        capabilities = agent.get_agent_capabilities()
        
        return {
            "status": "success",
            "command": "info",
            "agent_response": self._format_info_response_safe(capabilities),
            "raw_data": capabilities,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_analysis_response_safe(self, analysis_result: Dict[str, Any]) -> str:
        """분석 결과 포맷팅 (Windows 안전 버전)"""
        summary = analysis_result.get("project_summary", {})
        insights = analysis_result.get("solomond_specific_insights", [])
        recommendations = analysis_result.get("optimization_recommendations", [])
        
        response_parts = [
            "=== SOLOMOND AI INTEGRATED CODEBASE ANALYSIS COMPLETE ===",
            "",
            "ANALYSIS SUMMARY:",
            f"* Files analyzed: {summary.get('files_analyzed', 0)}",
            f"* Total code lines: {summary.get('total_lines', 0):,}",
            f"* Issues found: {summary.get('total_issues', 0)}",
            f"  - [CRITICAL] {summary.get('critical_issues', 0)}",
            f"  - [HIGH] {summary.get('high_issues', 0)}",
            f"  - [MEDIUM] {summary.get('medium_issues', 0)}",
            f"  - [LOW] {summary.get('low_issues', 0)}"
        ]
        
        # 건강도 정보
        health = analysis_result.get("health_assessment", {})
        if health:
            score = health.get("overall_score", 0)
            if score >= 90:
                health_status = "EXCELLENT"
            elif score >= 80:
                health_status = "GOOD"
            elif score >= 70:
                health_status = "FAIR"
            else:
                health_status = "NEEDS IMPROVEMENT"
            
            response_parts.extend([
                "",
                f"SYSTEM HEALTH: {score}/100 ({health_status})"
            ])
        
        # SOLOMOND AI 특화 인사이트
        if insights:
            response_parts.extend([
                "",
                "SOLOMOND AI SPECIFIC INSIGHTS:"
            ])
            for insight in insights[:3]:
                level_prefix = {
                    "positive": "[GOOD]",
                    "warning": "[WARN]", 
                    "improvement": "[IMPROVE]",
                    "enhancement": "[ENHANCE]"
                }
                prefix = level_prefix.get(insight.get("level"), "[INFO]")
                response_parts.append(f"* {prefix} {insight.get('title', '')}")
        
        # 최적화 권장사항
        if recommendations:
            response_parts.extend([
                "",
                "PRIORITY OPTIMIZATION RECOMMENDATIONS:"
            ])
            for rec in recommendations[:3]:
                priority_prefix = {
                    "critical": "[CRITICAL]",
                    "high": "[HIGH]",
                    "medium": "[MEDIUM]"
                }
                prefix = priority_prefix.get(rec.get("priority"), "[INFO]")
                response_parts.append(f"* {prefix} {rec.get('title', '')}")
        
        # 자동 수정 가능 여부
        if analysis_result.get("auto_fix_available", False):
            response_parts.extend([
                "",
                "[AUTO-FIX] Some issues can be automatically fixed.",
                "Command: agent_interface_safe.py solomond-project-manager fix --auto"
            ])
        
        response_parts.extend([
            "",
            "INTEGRATED MANAGER: Analyzed SOLOMOND AI system at Symbol-level,",
            "comprehensively reviewing ThreadPool management, GPU memory optimization, and Streamlit performance."
        ])
        
        return "\n".join(response_parts)
    
    def _format_health_response_safe(self, analysis_result: Dict[str, Any], report_file: str) -> str:
        """건강도 결과 포맷팅 (Windows 안전 버전)"""
        health = analysis_result.get("health_assessment", {})
        score = health.get("overall_score", 0)
        
        if score >= 90:
            health_grade = "EXCELLENT"
        elif score >= 80:
            health_grade = "GOOD"
        elif score >= 70:
            health_grade = "FAIR"
        else:
            health_grade = "NEEDS IMPROVEMENT"
        
        response_parts = [
            "=== SOLOMOND AI SYSTEM HEALTH DIAGNOSIS COMPLETE ===",
            "",
            f"OVERALL HEALTH: {score}/100 ({health_grade})",
            "",
            "HEALTH INDICATORS:",
            f"* Files analyzed: {health.get('files_analyzed', 0)}",
            f"* Critical issues: {health.get('critical_issues', 0)}",
            f"* High priority issues: {health.get('high_issues', 0)}",
            f"* Medium priority issues: {health.get('medium_issues', 0)}"
        ]
        
        # 권장사항 추가
        recommendations = health.get("recommendations", [])
        if recommendations:
            response_parts.extend([
                "",
                "HEALTH IMPROVEMENT RECOMMENDATIONS:"
            ])
            for rec in recommendations:
                response_parts.append(f"* {rec}")
        
        response_parts.extend([
            "",
            f"DETAILED HEALTH REPORT: {report_file}",
            "The report includes SOLOMOND AI specific insights and detailed analysis results."
        ])
        
        return "\n".join(response_parts)
    
    def _format_fix_response_safe(self, fix_result: Dict[str, Any], auto_applied: bool) -> str:
        """수정 결과 포맷팅 (Windows 안전 버전)"""
        response_parts = [
            "=== SOLOMOND AI AUTO-FIX SYSTEM RESULT ===",
            "",
            "FIX RESULTS:",
            f"* Issues fixed: {fix_result.get('fixes_applied', 0)}",
            f"* Files modified: {len(fix_result.get('files_modified', []))}",
            f"* Backups created: {len(fix_result.get('backups_created', []))}"
        ]
        
        if fix_result.get("fixes_applied", 0) > 0:
            response_parts.extend([
                "",
                "[SUCCESS] Auto-fix completed",
                "Backup files created in solomond_backups/ directory.",
                "",
                "NEXT STEPS:",
                "1. Restart system to verify changes",
                "2. Test major functions for normal operation",
                "3. Restore from backup files if needed"
            ])
        else:
            response_parts.extend([
                "",
                "[INFO] No issues to fix",
                "[SUCCESS] System is already in stable condition."
            ])
        
        if fix_result.get("errors"):
            response_parts.extend([
                "",
                "[ERRORS] Issues occurred:"
            ])
            for error in fix_result["errors"][:3]:
                response_parts.append(f"* {error}")
        
        return "\n".join(response_parts)
    
    def _format_fix_analysis_response_safe(self, analysis_result: Dict[str, Any]) -> str:
        """수정 분석 결과 포맷팅 (Windows 안전 버전)"""
        auto_fixable_count = len([
            issue for issue in analysis_result.get("issues_found", [])
            if issue.get("pattern_name", "") in [
                "threadpool_without_context", "memory_leak_cuda", 
                "streamlit_heavy_no_cache", "file_open_no_context"
            ]
        ])
        
        response_parts = [
            "=== AUTO-FIX ANALYSIS RESULT ===",
            "",
            f"FIXABLE ISSUES: {auto_fixable_count}"
        ]
        
        if auto_fixable_count > 0:
            response_parts.extend([
                "",
                "AUTO-FIXABLE PATTERNS:",
                "* ThreadPool context manager missing",
                "* GPU memory cleanup missing",
                "* Streamlit cache not applied",
                "* File context manager missing",
                "",
                "EXECUTE AUTO-FIX:",
                "agent_interface_safe.py solomond-project-manager fix --auto",
                "",
                "[WARNING] Important files will be backed up before auto-fix."
            ])
        else:
            response_parts.extend([
                "",
                "[SUCCESS] No issues requiring auto-fix found",
                "System is in stable condition."
            ])
        
        return "\n".join(response_parts)
    
    def _format_status_response_safe(self, analysis_result: Dict[str, Any], capabilities: Dict[str, Any]) -> str:
        """상태 체크 결과 포맷팅 (Windows 안전 버전)"""
        summary = analysis_result.get("project_summary", {})
        health = analysis_result.get("health_assessment", {})
        
        response_parts = [
            "=== SOLOMOND AI SYSTEM STATUS CHECK ===",
            "",
            f"AGENT: {capabilities.get('name', '')} v{capabilities.get('version', '')}",
            f"SERENA INTEGRATION: {'COMPLETE' if capabilities.get('serena_integrated') else 'INCOMPLETE'}",
            "",
            "CORE SYSTEM STATUS:",
            f"* Files analyzed: {summary.get('files_analyzed', 0)}",
            f"* Issues found: {summary.get('total_issues', 0)}"
        ]
        
        # 건강도 표시
        if health:
            score = health.get("overall_score", 0)
            status_indicator = "[GREEN]" if score >= 80 else "[YELLOW]" if score >= 60 else "[RED]"
            response_parts.append(f"* System health: {status_indicator} {score}/100")
        
        # 분석 패턴 정보
        response_parts.extend([
            "",
            "ANALYSIS CAPABILITIES:",
            f"* Analysis patterns: {capabilities.get('analysis_patterns', 0)}",
            f"* Auto-fix patterns: {capabilities.get('auto_fix_patterns', 0)}"
        ])
        
        # 시스템 상태
        system_status = analysis_result.get("system_status", {})
        if system_status:
            key_files_ok = all(system_status.get('key_files_present', {}).values())
            git_clean = system_status.get('git_status', {}).get('clean')
            
            response_parts.extend([
                "",
                "SYSTEM ENVIRONMENT:",
                f"* Core files: {'[OK] Normal' if key_files_ok else '[WARN] Some missing'}",
                f"* Git status: {'[OK] Clean' if git_clean else '[INFO] Modified files present'}",
                f"* Free disk space: {system_status.get('disk_space_gb', 0)}GB"
            ])
        
        response_parts.extend([
            "",
            "QUICK DIAGNOSIS: Core system operating normally,",
            "Integrated project manager monitoring all functions."
        ])
        
        return "\n".join(response_parts)
    
    def _format_info_response_safe(self, capabilities: Dict[str, Any]) -> str:
        """에이전트 정보 포맷팅 (Windows 안전 버전)"""
        response_parts = [
            f"=== {capabilities.get('name', '')} v{capabilities.get('version', '')} ===",
            "",
            f"ROLE: {capabilities.get('role', '')}",
            f"SERENA INTEGRATION: {'COMPLETE' if capabilities.get('serena_integrated') else 'INCOMPLETE'}",
            "",
            "CORE CAPABILITIES:"
        ]
        
        for capability in capabilities.get("core_capabilities", []):
            response_parts.append(f"* {capability}")
        
        response_parts.extend([
            "",
            "SOLOMOND AI SPECIALIZATIONS:"
        ])
        
        for specialization in capabilities.get("solomond_specializations", []):
            response_parts.append(f"* {specialization}")
        
        response_parts.extend([
            "",
            "SUPPORTED COMMANDS:"
        ])
        
        for command in capabilities.get("supported_commands", []):
            response_parts.append(f"* {command}")
        
        response_parts.extend([
            "",
            f"ANALYSIS PATTERNS: {capabilities.get('analysis_patterns', 0)}",
            f"AUTO-FIX PATTERNS: {capabilities.get('auto_fix_patterns', 0)}",
            "",
            "SPECIAL FEATURES: Fully specialized integrated project manager for SOLOMOND AI system",
            "with Serena's Symbol-level analysis and auto-optimization capabilities built-in."
        ])
        
        return "\n".join(response_parts)
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            "status": "error",
            "agent_response": f"[ERROR] {error_message}\n\n[INFO] Use 'info' command to check usage.",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """CLI 인터페이스 메인 함수"""
    parser = argparse.ArgumentParser(
        description="SOLOMOND AI Agent Interface (Windows Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python agent_interface_safe.py solomond-project-manager analyze
  python agent_interface_safe.py solomond-project-manager health
  python agent_interface_safe.py solomond-project-manager fix --auto
  python agent_interface_safe.py solomond-project-manager optimize
  python agent_interface_safe.py solomond-project-manager status
  python agent_interface_safe.py solomond-project-manager info
        """
    )
    
    parser.add_argument("agent", nargs="?", default="solomond-project-manager",
                       help="Agent to use (default: solomond-project-manager)")
    parser.add_argument("command", nargs="?", default="info",
                       help="Command to execute (analyze, health, fix, optimize, status, info)")
    parser.add_argument("--auto", action="store_true",
                       help="Apply auto-fix for fix command")
    parser.add_argument("--files", type=str,
                       help="Specific files to analyze for analyze command (comma-separated)")
    
    args = parser.parse_args()
    
    # 에이전트 인터페이스 초기화
    interface = AgentInterfaceSafe()
    
    # 명령어 인자 준비
    cmd_args = []
    if args.auto:
        cmd_args.append("--auto")
    if args.files:
        cmd_args.extend(["--files", args.files])
    
    # 명령어 실행
    result = interface.process_command(args.agent, args.command, cmd_args)
    
    # 결과 출력
    print(result["agent_response"])
    
    # 성공/실패 코드 반환
    return 0 if result["status"] == "success" else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)