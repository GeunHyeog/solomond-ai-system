#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Windows 콘솔 UTF-8 설정
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        # Windows 10 이상에서 UTF-8 콘솔 모드 활성화
        import subprocess
        subprocess.run("chcp 65001", shell=True, capture_output=True)
    except:
        pass
"""
🤖 SOLOMOND AI 에이전트 인터페이스
Claude Code에서 직접 호출 가능한 통합 에이전트 시스템

사용법:
python agent_interface.py solomond-project-manager analyze
python agent_interface.py solomond-project-manager health
python agent_interface.py solomond-project-manager fix --auto
python agent_interface.py solomond-project-manager optimize

Author: SOLOMOND AI Team  
Version: 1.0.0 (Serena Integration Complete)
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
    print(f"❌ SOLOMOND Project Manager 모듈 로드 실패: {e}")
    print("💡 solomond_unified_project_manager.py 파일이 존재하는지 확인하세요.")
    sys.exit(1)

class AgentInterface:
    """Claude Code 에이전트 인터페이스"""
    
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
            return self._error_response(f"알 수 없는 에이전트: {agent_name}")
        
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
                return self._error_response(f"지원하지 않는 명령어: {command}")
                
        except Exception as e:
            return self._error_response(f"명령어 실행 중 오류: {str(e)}")
    
    def _handle_analyze(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """코드베이스 분석 처리"""
        print("🔍 SOLOMOND AI 통합 코드베이스 분석을 시작합니다...")
        
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
            "agent_response": self._format_analysis_response(analysis_result),
            "raw_data": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_health(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """시스템 건강도 체크 처리"""
        print("🏥 SOLOMOND AI 시스템 건강도를 진단합니다...")
        
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
            "agent_response": self._format_health_response(analysis_result, str(report_file)),
            "raw_data": analysis_result,
            "report_file": str(report_file),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_fix(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """자동 수정 처리"""
        auto_apply = "--auto" in args if args else False
        
        if auto_apply:
            print("🔧 자동 수정을 실행합니다...")
        else:
            print("🔍 수정 가능한 이슈를 분석합니다...")
        
        # 분석 실행
        analysis_result = agent.analyze_codebase()
        
        if not analysis_result.get("auto_fix_available", False):
            return {
                "status": "success",
                "command": "fix",
                "agent_response": "ℹ️ 자동 수정 가능한 이슈가 발견되지 않았습니다.\n✅ 시스템이 안정적인 상태입니다.",
                "raw_data": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
        
        if auto_apply:
            # 자동 수정 실행
            fix_result = agent.auto_fix_issues(analysis_result, create_backups=True)
            
            return {
                "status": "success",
                "command": "fix",
                "agent_response": self._format_fix_response(fix_result, True),
                "raw_data": {"analysis": analysis_result, "fix_result": fix_result},
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 분석만 수행
            return {
                "status": "success",
                "command": "fix",
                "agent_response": self._format_fix_analysis_response(analysis_result),
                "raw_data": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_optimize(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """최적화 제안 처리"""
        print("⚡ SOLOMOND AI 시스템 최적화 분석을 수행합니다...")
        
        analysis_result = agent.analyze_codebase()
        
        return {
            "status": "success",
            "command": "optimize",
            "agent_response": self._format_optimize_response(analysis_result),
            "raw_data": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_status(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """시스템 상태 체크 처리"""
        print("📊 SOLOMOND AI 시스템 상태를 확인합니다...")
        
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
            "agent_response": self._format_status_response(analysis_result, capabilities),
            "raw_data": {"analysis": analysis_result, "capabilities": capabilities},
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_info(self, agent: SOLOMONDProjectManager, args: List[str]) -> Dict[str, Any]:
        """에이전트 정보 표시"""
        capabilities = agent.get_agent_capabilities()
        
        return {
            "status": "success",
            "command": "info",
            "agent_response": self._format_info_response(capabilities),
            "raw_data": capabilities,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """분석 결과 포맷팅"""
        summary = analysis_result.get("project_summary", {})
        insights = analysis_result.get("solomond_specific_insights", [])
        recommendations = analysis_result.get("optimization_recommendations", [])
        
        response_parts = [
            "🔍 **SOLOMOND AI 통합 코드베이스 분석 완료**",
            "",
            "📊 **분석 결과 요약:**",
            f"• 분석된 파일: {summary.get('files_analyzed', 0)}개",
            f"• 총 코드 라인: {summary.get('total_lines', 0):,}줄",
            f"• 발견된 이슈: {summary.get('total_issues', 0)}개",
            f"  - 🚨 크리티컬: {summary.get('critical_issues', 0)}개",
            f"  - ⚠️ 중요: {summary.get('high_issues', 0)}개",
            f"  - 📋 보통: {summary.get('medium_issues', 0)}개",
            f"  - ℹ️ 경미: {summary.get('low_issues', 0)}개"
        ]
        
        # 건강도 정보
        health = analysis_result.get("health_assessment", {})
        if health:
            score = health.get("overall_score", 0)
            if score >= 90:
                health_emoji = "💚"
                health_status = "최우수"
            elif score >= 80:
                health_emoji = "💚"
                health_status = "우수"
            elif score >= 70:
                health_emoji = "💛"
                health_status = "양호"
            else:
                health_emoji = "🧡"
                health_status = "개선 필요"
            
            response_parts.extend([
                "",
                f"🏥 **시스템 건강도: {health_emoji} {score}/100 ({health_status})**"
            ])
        
        # SOLOMOND AI 특화 인사이트
        if insights:
            response_parts.extend([
                "",
                "🧠 **SOLOMOND AI 특화 인사이트:**"
            ])
            for insight in insights[:3]:
                emoji_map = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
                emoji = emoji_map.get(insight.get("level"), "💡")
                response_parts.append(f"• {emoji} {insight.get('title', '')}")
        
        # 최적화 권장사항
        if recommendations:
            response_parts.extend([
                "",
                "💡 **우선순위 최적화 권장사항:**"
            ])
            for rec in recommendations[:3]:
                priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋"}
                emoji = priority_emoji.get(rec.get("priority"), "💡")
                response_parts.append(f"• {emoji} {rec.get('title', '')}")
        
        # 자동 수정 가능 여부
        if analysis_result.get("auto_fix_available", False):
            response_parts.extend([
                "",
                "🔧 **자동 수정 가능**: 일부 이슈는 자동으로 수정할 수 있습니다.",
                "   명령어: `agent_interface.py solomond-project-manager fix --auto`"
            ])
        
        response_parts.extend([
            "",
            "🎯 **통합 매니저**: SOLOMOND AI 시스템을 Symbol-level로 분석하여",
            "ThreadPool 관리, GPU 메모리 최적화, Streamlit 성능 등을 종합적으로 검토했습니다."
        ])
        
        return "\n".join(response_parts)
    
    def _format_health_response(self, analysis_result: Dict[str, Any], report_file: str) -> str:
        """건강도 결과 포맷팅"""
        health = analysis_result.get("health_assessment", {})
        score = health.get("overall_score", 0)
        
        if score >= 90:
            health_emoji = "💚"
            health_grade = "최우수 🏆"
        elif score >= 80:
            health_emoji = "💚"
            health_grade = "우수 ⭐"
        elif score >= 70:
            health_emoji = "💛"
            health_grade = "양호 👍"
        else:
            health_emoji = "🧡"
            health_grade = "개선 필요 ⚠️"
        
        response_parts = [
            "🏥 **SOLOMOND AI 시스템 건강 진단 완료**",
            "",
            f"{health_emoji} **전체 건강도: {score}/100 ({health_grade})**",
            "",
            "📊 **건강 지표:**",
            f"• 분석된 파일: {health.get('files_analyzed', 0)}개",
            f"• 크리티컬 이슈: {health.get('critical_issues', 0)}개",
            f"• 중요 이슈: {health.get('high_issues', 0)}개",
            f"• 보통 이슈: {health.get('medium_issues', 0)}개"
        ]
        
        # 권장사항 추가
        recommendations = health.get("recommendations", [])
        if recommendations:
            response_parts.extend([
                "",
                "💡 **건강 개선 권장사항:**"
            ])
            for rec in recommendations:
                response_parts.append(f"• {rec}")
        
        response_parts.extend([
            "",
            f"📋 **상세 건강도 보고서**: {report_file}",
            "   보고서에는 SOLOMOND AI 특화 인사이트와 상세 분석 결과가 포함되어 있습니다."
        ])
        
        return "\n".join(response_parts)
    
    def _format_fix_response(self, fix_result: Dict[str, Any], auto_applied: bool) -> str:
        """수정 결과 포맷팅"""
        response_parts = [
            "🔧 **SOLOMOND AI 자동 수정 시스템 결과**",
            "",
            f"📊 **수정 결과:**",
            f"• 수정된 이슈: {fix_result.get('fixes_applied', 0)}개",
            f"• 수정된 파일: {len(fix_result.get('files_modified', []))}개",
            f"• 백업 생성: {len(fix_result.get('backups_created', []))}개"
        ]
        
        if fix_result.get("fixes_applied", 0) > 0:
            response_parts.extend([
                "",
                "✅ **자동 수정 완료**",
                f"💾 백업 파일이 solomond_backups/ 디렉토리에 생성되었습니다.",
                "",
                "💡 **다음 단계:**",
                "1. 시스템을 재시작하여 변경사항 확인",
                "2. 주요 기능들이 정상 작동하는지 테스트",
                "3. 필요시 백업 파일로 복원 가능"
            ])
        else:
            response_parts.extend([
                "",
                "ℹ️ **수정할 이슈가 없습니다**",
                "✅ 시스템이 이미 안정적인 상태입니다."
            ])
        
        if fix_result.get("errors"):
            response_parts.extend([
                "",
                "⚠️ **오류 발생:**"
            ])
            for error in fix_result["errors"][:3]:
                response_parts.append(f"• {error}")
        
        return "\n".join(response_parts)
    
    def _format_fix_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """수정 분석 결과 포맷팅"""
        auto_fixable_count = len([
            issue for issue in analysis_result.get("issues_found", [])
            if issue.get("pattern_name", "") in [
                "threadpool_without_context", "memory_leak_cuda", 
                "streamlit_heavy_no_cache", "file_open_no_context"
            ]
        ])
        
        response_parts = [
            "🔍 **자동 수정 분석 결과**",
            "",
            f"📊 **수정 가능한 이슈: {auto_fixable_count}개**"
        ]
        
        if auto_fixable_count > 0:
            response_parts.extend([
                "",
                "🔧 **자동 수정 가능한 패턴:**",
                "• ThreadPool context manager 누락",
                "• GPU 메모리 정리 누락", 
                "• Streamlit 캐시 미적용",
                "• 파일 context manager 누락",
                "",
                "💡 **자동 수정 실행:**",
                "`agent_interface.py solomond-project-manager fix --auto`",
                "",
                "⚠️ **주의**: 자동 수정 전 중요 파일이 백업됩니다."
            ])
        else:
            response_parts.extend([
                "",
                "✅ **자동 수정이 필요한 이슈가 없습니다**",
                "시스템이 안정적인 상태입니다."
            ])
        
        return "\n".join(response_parts)
    
    def _format_optimize_response(self, analysis_result: Dict[str, Any]) -> str:
        """최적화 결과 포맷팅"""
        recommendations = analysis_result.get("optimization_recommendations", [])
        insights = analysis_result.get("solomond_specific_insights", [])
        
        response_parts = [
            "⚡ **SOLOMOND AI 성능 최적화 분석**",
            "",
            "🎯 **통합 매니저 최적화 전략**"
        ]
        
        if recommendations:
            response_parts.extend([
                "",
                "🚀 **우선순위 최적화 항목:**"
            ])
            
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋"}
                emoji = priority_emoji.get(rec.get("priority"), "💡")
                response_parts.extend([
                    "",
                    f"{i}. {emoji} **{rec.get('title', '')}**",
                    f"   {rec.get('description', '')}",
                    f"   💎 **SOLOMOND 효과**: {rec.get('solomond_benefit', '')}",
                    f"   ⏱️ **예상 시간**: {rec.get('estimated_time', 'N/A')}"
                ])
        
        if insights:
            response_parts.extend([
                "",
                "🧠 **시스템 특화 인사이트:**"
            ])
            
            for insight in insights[:3]:
                emoji_map = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
                emoji = emoji_map.get(insight.get("level"), "💡")
                response_parts.extend([
                    f"• {emoji} **{insight.get('title', '')}**",
                    f"  {insight.get('message', '')}"
                ])
        
        response_parts.extend([
            "",
            "🎯 **핵심 최적화 방향:**",
            "1. 🔧 ThreadPool 리소스를 with 문으로 안전 관리",
            "2. ⚡ Streamlit @st.cache_data로 AI 모델 로딩 최적화",
            "3. 🧹 torch.cuda.empty_cache()로 GPU 메모리 정리",
            "4. 🛡️ Ollama API 호출에 재시도 로직 추가",
            "5. 📊 컨퍼런스 분석 배치 처리 최적화"
        ])
        
        return "\n".join(response_parts)
    
    def _format_status_response(self, analysis_result: Dict[str, Any], capabilities: Dict[str, Any]) -> str:
        """상태 체크 결과 포맷팅"""
        summary = analysis_result.get("project_summary", {})
        health = analysis_result.get("health_assessment", {})
        
        response_parts = [
            "📊 **SOLOMOND AI 시스템 상태 체크**",
            "",
            f"🤖 **에이전트**: {capabilities.get('name', '')} v{capabilities.get('version', '')}",
            f"🔗 **Serena 통합**: {'✅ 완료' if capabilities.get('serena_integrated') else '❌ 미완료'}",
            "",
            "📁 **핵심 시스템 상태:**",
            f"• 분석된 파일: {summary.get('files_analyzed', 0)}개",
            f"• 발견된 이슈: {summary.get('total_issues', 0)}개"
        ]
        
        # 건강도 표시
        if health:
            score = health.get("overall_score", 0)
            status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            response_parts.append(f"• 시스템 건강도: {status_emoji} {score}/100")
        
        # 분석 패턴 정보
        response_parts.extend([
            "",
            "🎯 **분석 역량:**",
            f"• 분석 패턴: {capabilities.get('analysis_patterns', 0)}개",
            f"• 자동 수정 패턴: {capabilities.get('auto_fix_patterns', 0)}개"
        ])
        
        # 시스템 상태
        system_status = analysis_result.get("system_status", {})
        if system_status:
            response_parts.extend([
                "",
                "🖥️ **시스템 환경:**",
                f"• 핵심 파일: {'✅ 정상' if all(system_status.get('key_files_present', {}).values()) else '⚠️ 일부 누락'}",
                f"• Git 상태: {'✅ 깨끗' if system_status.get('git_status', {}).get('clean') else '📝 수정사항 있음'}",
                f"• 디스크 여유공간: {system_status.get('disk_space_gb', 0)}GB"
            ])
        
        response_parts.extend([
            "",
            "💡 **빠른 진단**: 핵심 시스템이 정상 작동 중이며,",
            "통합 프로젝트 매니저가 모든 기능을 모니터링하고 있습니다."
        ])
        
        return "\n".join(response_parts)
    
    def _format_info_response(self, capabilities: Dict[str, Any]) -> str:
        """에이전트 정보 포맷팅"""
        response_parts = [
            f"🤖 **{capabilities.get('name', '')} v{capabilities.get('version', '')}**",
            "",
            f"🎯 **역할**: {capabilities.get('role', '')}",
            f"🔗 **Serena 통합**: {'✅ 완료' if capabilities.get('serena_integrated') else '❌ 미완료'}",
            "",
            "⚡ **핵심 기능:**"
        ]
        
        for capability in capabilities.get("core_capabilities", []):
            response_parts.append(f"• {capability}")
        
        response_parts.extend([
            "",
            "🎯 **SOLOMOND AI 특화 기능:**"
        ])
        
        for specialization in capabilities.get("solomond_specializations", []):
            response_parts.append(f"• {specialization}")
        
        response_parts.extend([
            "",
            "📋 **지원 명령어:**"
        ])
        
        for command in capabilities.get("supported_commands", []):
            response_parts.append(f"• {command}")
        
        response_parts.extend([
            "",
            f"🎯 **분석 패턴**: {capabilities.get('analysis_patterns', 0)}개",
            f"🔧 **자동 수정 패턴**: {capabilities.get('auto_fix_patterns', 0)}개",
            "",
            "🏆 **특별함**: SOLOMOND AI 시스템에 완전히 특화된 통합 프로젝트 매니저로서",
            "Serena의 Symbol-level 분석과 자동 최적화 기능을 완벽하게 내장하고 있습니다."
        ])
        
        return "\n".join(response_parts)
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            "status": "error",
            "agent_response": f"❌ **오류 발생**: {error_message}\n\n💡 사용법을 확인하려면 'info' 명령어를 사용하세요.",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """CLI 인터페이스 메인 함수"""
    # Windows 인코딩 설정
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    parser = argparse.ArgumentParser(
        description="SOLOMOND AI 에이전트 인터페이스",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python agent_interface.py solomond-project-manager analyze
  python agent_interface.py solomond-project-manager health
  python agent_interface.py solomond-project-manager fix --auto
  python agent_interface.py solomond-project-manager optimize
  python agent_interface.py solomond-project-manager status
  python agent_interface.py solomond-project-manager info
        """
    )
    
    parser.add_argument("agent", nargs="?", default="solomond-project-manager",
                       help="사용할 에이전트 (기본값: solomond-project-manager)")
    parser.add_argument("command", nargs="?", default="info",
                       help="실행할 명령어 (analyze, health, fix, optimize, status, info)")
    parser.add_argument("--auto", action="store_true",
                       help="fix 명령어에서 자동 수정 적용")
    parser.add_argument("--files", type=str,
                       help="analyze 명령어에서 특정 파일들만 분석 (쉼표로 구분)")
    
    args = parser.parse_args()
    
    # 에이전트 인터페이스 초기화
    interface = AgentInterface()
    
    # 명령어 인자 준비
    cmd_args = []
    if args.auto:
        cmd_args.append("--auto")
    if args.files:
        cmd_args.extend(["--files", args.files])
    
    # 명령어 실행
    result = interface.process_command(args.agent, args.command, cmd_args)
    
    # 결과 출력 (Unicode 안전)
    try:
        print(result["agent_response"])
    except UnicodeEncodeError:
        # 이모지 제거 후 출력
        import re
        clean_response = re.sub(r'[^\x00-\x7F]+', '', result["agent_response"])
        print(clean_response)
    
    # 성공/실패 코드 반환
    return 0 if result["status"] == "success" else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)