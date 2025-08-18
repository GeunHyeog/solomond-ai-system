#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Serena Claude Code Interface
Claude Code의 /agent serena 명령어 인터페이스

이 모듈은 Claude Code에서 /agent serena 명령어로 호출되는
실제 서브에이전트 인터페이스를 구현합니다.

사용법:
/agent serena analyze          # 코드 분석 수행
/agent serena fix             # 자동 수정 적용
/agent serena health          # 프로젝트 건강도 체크  
/agent serena optimize        # 성능 최적화 제안
/agent serena info            # 에이전트 정보
/agent serena help            # 도움말

Author: Serena & SOLOMOND AI Team
Version: 1.0.0
For: Claude Code Sub-Agent System
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Serena 모듈들 임포트
try:
    from serena_claude_agent import SerenaClaudeAgent
    from serena_mcp_integration import SerenaMCPIntegration, SerenaAnalysisConfig
    from serena_auto_optimizer import SerenaAutoOptimizer
    from agent_serena_integration import SerenaCodeAnalyzer
except ImportError as e:
    print(f"❌ Serena 모듈 로드 실패: {e}")
    print("💡 모든 serena_*.py 파일이 존재하는지 확인하세요.")
    sys.exit(1)

class SerenaClaudeInterface:
    """Serena Claude Code 서브에이전트 인터페이스"""
    
    def __init__(self):
        self.agent_name = "serena"
        self.version = "1.0.0"
        self.persona = {
            "name": "Serena",
            "role": "SOLOMOND AI 전문 코딩 에이전트",
            "greeting": "안녕하세요! 저는 SOLOMOND AI 시스템 전문 코딩 에이전트 Serena입니다. 🤖",
            "expertise": [
                "Python 코드 Symbol-level 분석",
                "ThreadPool 및 리소스 관리 최적화", 
                "GPU 메모리 누수 탐지 및 해결",
                "Streamlit 성능 최적화",
                "Ollama AI 통합 안정성 개선",
                "SOLOMOND AI 시스템 아키텍처 분석"
            ]
        }
        
        # 명령어 매핑
        self.commands = {
            "analyze": self.cmd_analyze,
            "fix": self.cmd_fix,
            "health": self.cmd_health,
            "optimize": self.cmd_optimize,
            "info": self.cmd_info,
            "help": self.cmd_help,
            "status": self.cmd_status,
            "report": self.cmd_report
        }
    
    def process_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        """명령어 처리"""
        if command not in self.commands:
            return self._error_response(f"알 수 없는 명령어: {command}. 'help'를 입력하여 사용법을 확인하세요.")
        
        try:
            return self.commands[command](args or [])
        except Exception as e:
            return self._error_response(f"명령어 실행 중 오류: {str(e)}")
    
    def cmd_analyze(self, args: List[str]) -> Dict[str, Any]:
        """코드 분석 명령어"""
        print("🔍 Serena: SOLOMOND AI 시스템 코드 분석을 시작합니다...")
        
        try:
            # MCP 통합 분석 실행
            config = SerenaAnalysisConfig(
                analysis_depth="comprehensive",
                focus_areas=[
                    "threadpool_management",
                    "memory_optimization",
                    "streamlit_performance",
                    "ollama_integration"
                ]
            )
            
            mcp_integration = SerenaMCPIntegration(config)
            result = mcp_integration.analyze_with_mcp_tools()
            
            # 사용자 친화적 응답 생성
            response = {
                "status": "success",
                "command": "analyze",
                "serena_response": self._format_analysis_response(result),
                "raw_data": result,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            return self._error_response(f"분석 실행 실패: {str(e)}")
    
    def cmd_fix(self, args: List[str]) -> Dict[str, Any]:
        """자동 수정 명령어"""
        print("🔧 Serena: 자동 수정 시스템을 실행합니다...")
        
        # 확인 모드 (기본값)
        auto_apply = "--auto" in args
        
        if not auto_apply:
            print("💡 안전을 위해 분석만 수행합니다. 실제 수정을 원하면 --auto 플래그를 사용하세요.")
        
        try:
            optimizer = SerenaAutoOptimizer()
            result = optimizer.analyze_and_fix_project(auto_apply=auto_apply)
            
            # 자동 수정 스크립트 생성 (크리티컬 이슈가 있는 경우)
            script_generated = False
            if result["analysis_summary"]["critical_fixes"] > 0:
                fix_script = optimizer.generate_auto_fix_script(result)
                script_path = Path("serena_auto_fix.py")
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(fix_script)
                
                script_generated = True
                print(f"📝 자동 수정 스크립트 생성: {script_path}")
            
            response = {
                "status": "success",
                "command": "fix",
                "serena_response": self._format_fix_response(result, auto_apply, script_generated),
                "raw_data": result,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            return self._error_response(f"자동 수정 실행 실패: {str(e)}")
    
    def cmd_health(self, args: List[str]) -> Dict[str, Any]:
        """프로젝트 건강도 체크 명령어"""
        print("🏥 Serena: SOLOMOND AI 시스템 건강도를 검진합니다...")
        
        try:
            claude_agent = SerenaClaudeAgent()
            result = claude_agent.analyze_project()
            
            # 건강도 점수 계산
            health_score = result.get("health_score", 0)
            
            # 건강도 등급 결정
            if health_score >= 90:
                health_grade = "최우수 🏆"
                health_emoji = "💚"
            elif health_score >= 80:
                health_grade = "우수 ⭐"
                health_emoji = "💚"
            elif health_score >= 70:
                health_grade = "양호 👍"
                health_emoji = "💛"
            elif health_score >= 60:
                health_grade = "보통 👌"
                health_emoji = "🧡"
            else:
                health_grade = "주의 필요 ⚠️"
                health_emoji = "❤️"
            
            response = {
                "status": "success",
                "command": "health",
                "serena_response": self._format_health_response(result, health_score, health_grade, health_emoji),
                "raw_data": result,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            return self._error_response(f"건강도 체크 실패: {str(e)}")
    
    def cmd_optimize(self, args: List[str]) -> Dict[str, Any]:
        """성능 최적화 제안 명령어"""
        print("⚡ Serena: 성능 최적화 분석을 수행합니다...")
        
        try:
            # 종합 분석 실행
            mcp_integration = SerenaMCPIntegration()
            analysis_result = mcp_integration.analyze_with_mcp_tools()
            
            # 최적화 제안 추출
            recommendations = analysis_result.get("optimization_recommendations", [])
            insights = analysis_result.get("solomond_specific_insights", [])
            
            response = {
                "status": "success", 
                "command": "optimize",
                "serena_response": self._format_optimize_response(recommendations, insights),
                "raw_data": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            return self._error_response(f"최적화 분석 실패: {str(e)}")
    
    def cmd_info(self, args: List[str]) -> Dict[str, Any]:
        """에이전트 정보 명령어"""
        claude_agent = SerenaClaudeAgent()
        agent_info = claude_agent.get_agent_info()
        
        response = {
            "status": "success",
            "command": "info",
            "serena_response": self._format_info_response(agent_info),
            "raw_data": agent_info,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def cmd_help(self, args: List[str]) -> Dict[str, Any]:
        """도움말 명령어"""
        help_text = """
🤖 Serena - SOLOMOND AI 전문 코딩 에이전트

📋 사용 가능한 명령어:

🔍 analyze          - SOLOMOND AI 시스템 종합 코드 분석
                      Symbol-level 분석으로 ThreadPool, 메모리 누수, 
                      성능 병목점을 정밀하게 탐지합니다.

🔧 fix [--auto]     - 자동 수정 시스템
                      --auto: 즉시 수정 적용 (위험)
                      기본값: 분석 후 수정 스크립트 생성

🏥 health           - 프로젝트 건강도 진단
                      시스템 안정성과 코드 품질을 점수로 평가합니다.

⚡ optimize         - 성능 최적화 제안
                      Streamlit 캐싱, GPU 메모리 관리, 
                      Ollama 통합 최적화 방안을 제시합니다.

ℹ️  info             - Serena 에이전트 정보
                      전문 분야, 기능, 버전 정보를 확인합니다.

📊 status           - 현재 시스템 상태 간단 체크

📈 report           - 종합 분석 보고서 생성

❓ help             - 이 도움말

💡 사용 예시:
   /agent serena analyze
   /agent serena fix --auto
   /agent serena health
   /agent serena optimize

🎯 Serena의 특화 영역:
   • SOLOMOND AI 컨퍼런스 분석 시스템 최적화
   • ThreadPoolExecutor 리소스 관리 자동화
   • GPU 메모리 누수 탐지 및 해결
   • Streamlit 성능 튜닝
   • Ollama AI 모델 통합 안정성 개선
"""
        
        response = {
            "status": "success",
            "command": "help",
            "serena_response": help_text,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def cmd_status(self, args: List[str]) -> Dict[str, Any]:
        """시스템 상태 간단 체크"""
        print("📊 Serena: 시스템 상태를 빠르게 확인합니다...")
        
        try:
            # 핵심 파일들 존재 여부 확인
            core_files = [
                "conference_analysis_COMPLETE_WORKING.py",
                "solomond_ai_main_dashboard.py",
                "dual_brain_integration.py"
            ]
            
            file_status = {}
            for file_name in core_files:
                file_path = Path(file_name)
                file_status[file_name] = {
                    "exists": file_path.exists(),
                    "size_kb": file_path.stat().st_size // 1024 if file_path.exists() else 0
                }
            
            # 간단한 분석 실행
            claude_agent = SerenaClaudeAgent()
            quick_result = claude_agent.analyze_project(core_files[:2])  # 2개 파일만 빠르게
            
            response = {
                "status": "success",
                "command": "status",
                "serena_response": self._format_status_response(file_status, quick_result),
                "raw_data": {"file_status": file_status, "analysis": quick_result},
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            return self._error_response(f"상태 체크 실패: {str(e)}")
    
    def cmd_report(self, args: List[str]) -> Dict[str, Any]:
        """종합 분석 보고서 생성"""
        print("📈 Serena: 종합 분석 보고서를 생성합니다...")
        
        try:
            # 전체 분석 실행
            mcp_integration = SerenaMCPIntegration()
            mcp_result = mcp_integration.analyze_with_mcp_tools()
            
            optimizer = SerenaAutoOptimizer()
            opt_result = optimizer.analyze_and_fix_project(auto_apply=False)
            
            # 보고서 생성
            report = self._generate_comprehensive_report(mcp_result, opt_result)
            
            # 파일로 저장
            report_file = Path("serena_comprehensive_report.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            response = {
                "status": "success",
                "command": "report",
                "serena_response": f"📄 종합 분석 보고서가 생성되었습니다: {report_file}\\n\\n{report[:500]}...",
                "raw_data": {"mcp_result": mcp_result, "opt_result": opt_result},
                "report_file": str(report_file),
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            return self._error_response(f"보고서 생성 실패: {str(e)}")
    
    def _format_analysis_response(self, result: Dict[str, Any]) -> str:
        """분석 결과 포맷팅"""
        summary = result.get("project_summary", {})
        insights = result.get("solomond_specific_insights", [])
        recommendations = result.get("optimization_recommendations", [])
        
        response_parts = [
            f"🔍 **SOLOMOND AI 시스템 분석 완료**",
            f"",
            f"📊 **분석 결과 요약:**",
            f"• 분석된 파일: {summary.get('files_analyzed', 0)}개",
            f"• 코드 라인 수: {summary.get('total_lines', 0):,}줄",  
            f"• 발견된 이슈: {summary.get('total_issues', 0)}개",
            f"• 크리티컬 이슈: {summary.get('critical_issues', 0)}개 🚨",
            f"• 높은 우선순위: {summary.get('high_issues', 0)}개 ⚠️",
            f"• 보통 우선순위: {summary.get('medium_issues', 0)}개 📋",
        ]
        
        if insights:
            response_parts.extend([
                f"",
                f"🧠 **SOLOMOND AI 특화 인사이트:**"
            ])
            for insight in insights[:3]:  # 상위 3개만
                emoji = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
                response_parts.append(
                    f"• {emoji.get(insight.get('level'), '💡')} {insight.get('title', '')}"
                )
        
        if recommendations:
            response_parts.extend([
                f"",
                f"💡 **우선순위 권장사항:**"
            ])
            for rec in recommendations[:2]:  # 상위 2개만
                priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋"}
                response_parts.append(
                    f"• {priority_emoji.get(rec.get('priority'), '💡')} {rec.get('title', '')}"
                )
                response_parts.append(f"  {rec.get('description', '')}")
        
        response_parts.extend([
            f"",
            f"🎯 **Serena의 분석**: SOLOMOND AI 시스템의 핵심 구조를 Symbol-level로 분석했습니다.",
            f"ThreadPool 관리, GPU 메모리 최적화, Streamlit 성능 등 전문 영역을 중점적으로 검토했습니다."
        ])
        
        return "\\n".join(response_parts)
    
    def _format_fix_response(self, result: Dict[str, Any], auto_applied: bool, script_generated: bool) -> str:
        """수정 결과 포맷팅"""
        summary = result.get("analysis_summary", {})
        
        response_parts = [
            f"🔧 **자동 수정 시스템 결과**",
            f"",
            f"📊 **수정 가능한 이슈 분석:**",
            f"• 분석된 파일: {summary.get('files_analyzed', 0)}개",
            f"• 수정 가능한 이슈: {summary.get('fixable_issues', 0)}개",
            f"• 크리티컬 수정사항: {summary.get('critical_fixes', 0)}개"
        ]
        
        if auto_applied:
            response_parts.extend([
                f"",
                f"✅ **자동 수정 적용됨:** {summary.get('auto_fixes_applied', 0)}개 파일",
                f"💾 백업 파일이 serena_backups/ 디렉토리에 생성되었습니다."
            ])
        else:
            response_parts.extend([
                f"",
                f"📋 **안전 모드로 실행됨** (실제 수정 안함)"
            ])
        
        if script_generated:
            response_parts.extend([
                f"",
                f"📝 **자동 수정 스크립트 생성:**",
                f"• 파일: serena_auto_fix.py",
                f"• 실행 방법: python serena_auto_fix.py",
                f"⚠️  **주의**: 실행 전 중요 파일을 백업하세요!"
            ])
        
        response_parts.extend([
            f"",
            f"🎯 **Serena의 제안**: ThreadPoolExecutor와 파일 I/O를 with 문으로 감싸고,",
            f"GPU 메모리 정리를 추가하면 시스템 안정성이 크게 향상됩니다."
        ])
        
        return "\\n".join(response_parts)
    
    def _format_health_response(self, result: Dict[str, Any], health_score: float, health_grade: str, health_emoji: str) -> str:
        """건강도 결과 포맷팅"""
        response_parts = [
            f"🏥 **SOLOMOND AI 시스템 건강 진단**",
            f"",
            f"{health_emoji} **전체 건강도: {health_score:.1f}/100 ({health_grade})**",
            f"",
            f"📊 **건강 지표:**",
            f"• 분석된 파일: {result.get('files_analyzed', 0)}개",
            f"• 발견된 이슈: {len(result.get('issues_found', []))}개"
        ]
        
        # 건강도별 맞춤 조언
        if health_score >= 90:
            response_parts.extend([
                f"",
                f"🎉 **훌륭합니다!** SOLOMOND AI 시스템이 매우 건강한 상태입니다.",
                f"현재 코드 품질을 유지하면서 새로운 기능 개발에 집중하세요."
            ])
        elif health_score >= 70:
            response_parts.extend([
                f"",
                f"👍 **양호한 상태입니다.** 몇 가지 개선사항이 있지만 전반적으로 안정적입니다.",
                f"크리티컬 이슈부터 우선적으로 해결하시길 권장합니다."
            ])
        else:
            response_parts.extend([
                f"",
                f"⚠️  **주의가 필요합니다.** 시스템 안정성을 위해 즉시 개선이 필요합니다.",
                f"ThreadPool 관리와 메모리 누수 문제를 우선 해결하세요."
            ])
        
        # 추천사항 추가
        recommendations = result.get("recommendations", [])
        if recommendations:
            response_parts.extend([
                f"",
                f"💡 **건강 개선 방안:**"
            ])
            for rec in recommendations[:3]:
                response_parts.append(f"• {rec}")
        
        return "\\n".join(response_parts)
    
    def _format_optimize_response(self, recommendations: List[Dict], insights: List[Dict]) -> str:
        """최적화 결과 포맷팅"""
        response_parts = [
            f"⚡ **SOLOMOND AI 성능 최적화 분석**",
            f"",
            f"🎯 **Serena의 최적화 전략**"
        ]
        
        if recommendations:
            response_parts.append(f"")
            response_parts.append(f"🚀 **우선순위 최적화 항목:**")
            
            for i, rec in enumerate(recommendations[:3], 1):
                priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋"}
                response_parts.extend([
                    f"",
                    f"{i}. {priority_emoji.get(rec.get('priority'), '💡')} **{rec.get('title', '')}**",
                    f"   {rec.get('description', '')}",
                    f"   💎 **SOLOMOND 효과**: {rec.get('solomond_benefit', '성능 향상')}"
                ])
        
        if insights:
            response_parts.extend([
                f"",
                f"🧠 **시스템 인사이트:**"
            ])
            
            for insight in insights[:2]:
                emoji = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
                response_parts.extend([
                    f"• {emoji.get(insight.get('level'), '💡')} {insight.get('title', '')}",
                    f"  {insight.get('message', '')}"
                ])
        
        response_parts.extend([
            f"",
            f"🎯 **Serena의 핵심 제안:**",
            f"1. ThreadPoolExecutor를 with 문으로 감싸 리소스 안전 관리",
            f"2. Streamlit @st.cache_data로 AI 모델 로딩 최적화", 
            f"3. torch.cuda.empty_cache()로 GPU 메모리 정리",
            f"4. Ollama API 호출에 재시도 로직 추가"
        ])
        
        return "\\n".join(response_parts)
    
    def _format_info_response(self, agent_info: Dict[str, Any]) -> str:
        """에이전트 정보 포맷팅"""
        response_parts = [
            f"🤖 **{agent_info.get('name', 'Serena')} v{agent_info.get('version', '1.0.0')}**",
            f"",
            f"🎯 **역할**: {agent_info.get('role', '')}",
            f"",
            f"⚡ **전문 분야:**"
        ]
        
        for expertise in agent_info.get("expertise", [])[:6]:
            response_parts.append(f"• {expertise}")
        
        response_parts.extend([
            f"",
            f"🛠️  **핵심 기능:**"
        ])
        
        for capability in agent_info.get("capabilities", [])[:5]:
            response_parts.append(f"• {capability}")
        
        response_parts.extend([
            f"",
            f"📋 **지원 명령어:**"
        ])
        
        for command in agent_info.get("supported_commands", []):
            response_parts.append(f"• {command}")
        
        response_parts.extend([
            f"",
            f"💬 **응답 스타일**: {agent_info.get('response_style', '정밀하고 실용적인 기술 조언')}",
            f"",
            f"🏆 **Serena의 특별함**: SOLOMOND AI 시스템에 특화된 코딩 에이전트로서",
            f"Symbol-level 분석과 자동 최적화 기능을 제공합니다."
        ])
        
        return "\\n".join(response_parts)
    
    def _format_status_response(self, file_status: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """상태 체크 결과 포맷팅"""
        response_parts = [
            f"📊 **SOLOMOND AI 시스템 상태 체크**",
            f"",
            f"📁 **핵심 파일 상태:**"
        ]
        
        for file_name, status in file_status.items():
            emoji = "✅" if status["exists"] else "❌"
            size_info = f"({status['size_kb']}KB)" if status["exists"] else ""
            response_parts.append(f"• {emoji} {file_name} {size_info}")
        
        # 빠른 분석 결과
        health_score = analysis_result.get("health_score", 0)
        status_emoji = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "🔴"
        
        response_parts.extend([
            f"",
            f"🏥 **시스템 건강도**: {status_emoji} {health_score:.1f}/100",
            f"🔍 **발견된 이슈**: {len(analysis_result.get('issues_found', []))}개",
            f"",
            f"💡 **빠른 진단**: 핵심 파일들이 {'정상' if all(s['exists'] for s in file_status.values()) else '일부 누락'}이며,",
            f"시스템이 {'양호한' if health_score >= 70 else '개선이 필요한'} 상태입니다."
        ])
        
        return "\\n".join(response_parts)
    
    def _generate_comprehensive_report(self, mcp_result: Dict[str, Any], opt_result: Dict[str, Any]) -> str:
        """종합 분석 보고서 생성"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            f"# 🤖 Serena - SOLOMOND AI 시스템 종합 분석 보고서",
            f"",
            f"**생성 시간**: {timestamp}  ",
            f"**분석 엔진**: Serena v1.0.0 (Claude Code Sub-Agent)  ",
            f"**대상 시스템**: SOLOMOND AI Conference Analysis Platform",
            f"",
            f"## 📊 분석 요약",
            f"",
            f"### 🔍 MCP 통합 분석",
            f"- **분석된 파일**: {mcp_result.get('project_summary', {}).get('files_analyzed', 0)}개",
            f"- **총 코드 라인**: {mcp_result.get('project_summary', {}).get('total_lines', 0):,}줄", 
            f"- **발견된 이슈**: {mcp_result.get('project_summary', {}).get('total_issues', 0)}개",
            f"- **분석 시간**: {mcp_result.get('analysis_metadata', {}).get('duration_ms', 0):.1f}ms",
            f"",
            f"### 🔧 자동 최적화 분석",
            f"- **수정 가능한 이슈**: {opt_result.get('analysis_summary', {}).get('fixable_issues', 0)}개",
            f"- **크리티컬 수정사항**: {opt_result.get('analysis_summary', {}).get('critical_fixes', 0)}개",
            f"",
            f"## 🎯 핵심 발견사항",
            f""
        ]
        
        # SOLOMOND AI 특화 인사이트
        insights = mcp_result.get("solomond_specific_insights", [])
        if insights:
            report_lines.extend([
                f"### 🧠 SOLOMOND AI 특화 인사이트",
                f""
            ])
            
            for insight in insights:
                emoji = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
                report_lines.extend([
                    f"#### {emoji.get(insight.get('level'), '💡')} {insight.get('title', '')}",
                    f"{insight.get('message', '')}",
                    f"**영향**: {insight.get('impact', '')}",
                    f""
                ])
        
        # 최적화 권장사항
        recommendations = mcp_result.get("optimization_recommendations", [])
        if recommendations:
            report_lines.extend([
                f"### 💡 최적화 권장사항",
                f""
            ])
            
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋"}
                report_lines.extend([
                    f"#### {i}. {priority_emoji.get(rec.get('priority'), '💡')} {rec.get('title', '')}",
                    f"**우선순위**: {rec.get('priority', '').upper()}  ",
                    f"**설명**: {rec.get('description', '')}  ",
                    f"**예상 효과**: {rec.get('solomond_benefit', '')}  ",
                    f"**소요 시간**: {rec.get('estimated_time', 'N/A')}",
                    f""
                ])
        
        # 결론 및 다음 단계
        total_issues = mcp_result.get("project_summary", {}).get("total_issues", 0)
        critical_issues = mcp_result.get("project_summary", {}).get("critical_issues", 0)
        
        report_lines.extend([
            f"## 🏁 결론 및 다음 단계",
            f"",
            f"### 📈 전체 평가",
            f"SOLOMOND AI 시스템은 ",
        ])
        
        if critical_issues == 0:
            report_lines.append(f"**양호한 상태**입니다. 크리티컬 이슈가 발견되지 않아 안정적으로 운영되고 있습니다.")
        elif critical_issues <= 2:
            report_lines.append(f"**개선이 필요한 상태**입니다. {critical_issues}개의 크리티컬 이슈를 우선 해결하세요.")
        else:
            report_lines.append(f"**즉시 조치가 필요한 상태**입니다. {critical_issues}개의 크리티컬 이슈가 시스템 안정성을 위협합니다.")
        
        report_lines.extend([
            f"",
            f"### 🎯 우선순위 액션 아이템",
            f"1. **ThreadPool 리소스 관리**: with 문을 사용한 안전한 실행자 관리",
            f"2. **GPU 메모리 최적화**: torch.cuda.empty_cache() 정기 호출",
            f"3. **Streamlit 성능 향상**: @st.cache_data 데코레이터 적용",
            f"4. **에러 처리 강화**: Ollama API 호출에 예외 처리 추가",
            f"",
            f"### 🤖 Serena의 최종 제안",
            f"SOLOMOND AI 시스템의 안정성과 성능을 위해 ThreadPool 관리를 가장 우선으로 개선하시길 권장합니다. ",
            f"자동 수정 스크립트를 활용하면 대부분의 크리티컬 이슈를 신속하게 해결할 수 있습니다.",
            f"",
            f"---",
            f"*이 보고서는 Serena (SOLOMOND AI 전문 코딩 에이전트)에 의해 자동 생성되었습니다.*"
        ])
        
        return "\\n".join(report_lines)
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            "status": "error",
            "serena_response": f"❌ **오류 발생**: {error_message}\\n\\n💡 'help' 명령어로 사용법을 확인하거나, 문제가 지속되면 SOLOMOND AI 팀에 문의하세요.",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """CLI 인터페이스 메인 함수"""
    parser = argparse.ArgumentParser(
        description="Serena - SOLOMOND AI 전문 코딩 에이전트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python serena_claude_interface.py analyze
  python serena_claude_interface.py fix --auto
  python serena_claude_interface.py health
  python serena_claude_interface.py help
        """
    )
    
    parser.add_argument("command", nargs="?", default="help",
                       help="실행할 명령어 (analyze, fix, health, optimize, info, help, status, report)")
    parser.add_argument("--auto", action="store_true",
                       help="fix 명령어에서 자동 수정 적용")
    
    args = parser.parse_args()
    
    # Serena 인터페이스 초기화
    interface = SerenaClaudeInterface()
    
    # 명령어 처리
    cmd_args = []
    if args.auto:
        cmd_args.append("--auto")
    
    result = interface.process_command(args.command, cmd_args)
    
    # 결과 출력
    if result["status"] == "success":
        print(result["serena_response"])
        return 0
    else:
        print(result["serena_response"])
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)