#!/usr/bin/env python3
"""
클로드와 실시간 소통을 위한 웹 테스트 커뮤니케이터
- Playwright MCP 실행 결과를 클로드가 이해할 수 있는 형태로 변환
- 자동 문제 해결 제안 및 실행
- 실시간 테스트 피드백 시스템
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from core.web_test_analyzer import WebTestAnalyzer, TestResult, TestIssue

logger = logging.getLogger(__name__)

class ClaudeWebCommunicator:
    """클로드와 웹 테스트 소통을 위한 클래스"""
    
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.analyzer = WebTestAnalyzer()
        self.communication_log = []
        
        # 클로드 소통 로그 디렉토리
        self.claude_dir = self.session_dir / "claude_communication"
        self.claude_dir.mkdir(exist_ok=True)
        
    def generate_realtime_report(self, playwright_result: Dict) -> Dict[str, Any]:
        """
        Playwright MCP 실행 결과를 실시간으로 분석하여 클로드용 리포트 생성
        
        Args:
            playwright_result: Playwright MCP 함수 실행 결과
            
        Returns:
            클로드가 이해할 수 있는 형태의 리포트
        """
        timestamp = datetime.now().isoformat()
        
        # 기본 리포트 구조
        claude_report = {
            "timestamp": timestamp,
            "test_session": self.session_dir.name,
            "playwright_action": playwright_result.get("action", "unknown"),
            "status": "analyzing",
            "immediate_findings": [],
            "claude_actions_needed": [],
            "auto_fix_suggestions": [],
            "conversation_context": ""
        }
        
        # Playwright 액션 타입별 분석
        action = playwright_result.get("action")
        
        if action == "screenshot":
            claude_report.update(self._analyze_screenshot_result(playwright_result))
        elif action == "page_content":
            claude_report.update(self._analyze_content_result(playwright_result))
        elif action == "console_logs":
            claude_report.update(self._analyze_console_result(playwright_result))
        elif action == "click" or action == "fill":
            claude_report.update(self._analyze_interaction_result(playwright_result))
        elif action == "network_response":
            claude_report.update(self._analyze_network_result(playwright_result))
        
        # 클로드 대화 컨텍스트 생성
        claude_report["conversation_context"] = self._generate_conversation_context(claude_report)
        
        # 리포트 저장
        self._save_claude_report(claude_report)
        
        return claude_report
    
    def _analyze_screenshot_result(self, result: Dict) -> Dict:
        """스크린샷 결과 분석"""
        analysis = {
            "visual_analysis": {
                "screenshot_path": result.get("screenshot_path"),
                "page_status": "captured",
                "visible_errors": []
            }
        }
        
        # 스크린샷 경로가 있으면 OCR로 에러 메시지 감지 시도
        if result.get("screenshot_path"):
            analysis["claude_actions_needed"].append({
                "action": "visual_inspection",
                "description": "스크린샷을 확인하여 UI 문제점 분석",
                "priority": "medium",
                "mcp_function": "mcp__filesystem__read_file",
                "parameters": {"path": result["screenshot_path"]}
            })
        
        return analysis
    
    def _analyze_content_result(self, result: Dict) -> Dict:
        """페이지 콘텐츠 결과 분석"""
        content = result.get("content", "")
        
        analysis = {
            "content_analysis": {
                "page_length": len(content),
                "has_content": len(content) > 0,
                "detected_issues": []
            }
        }
        
        # 일반적인 에러 패턴 검색
        error_patterns = [
            ("404", "페이지를 찾을 수 없음"),
            ("500", "서버 내부 에러"),
            ("Error", "JavaScript 에러 발생"),
            ("Exception", "예외 상황 발생"),
            ("Loading", "로딩 상태 지속")
        ]
        
        for pattern, description in error_patterns:
            if pattern.lower() in content.lower():
                issue = {
                    "type": "content_error",
                    "pattern": pattern,
                    "description": description,
                    "severity": "high" if pattern in ["404", "500", "Error"] else "medium"
                }
                analysis["content_analysis"]["detected_issues"].append(issue)
                
                # 클로드 액션 제안
                analysis.setdefault("claude_actions_needed", []).append({
                    "action": "investigate_error",
                    "description": f"{description} - 상세 분석 필요",
                    "priority": "high",
                    "suggested_mcp": "mcp__playwright__evaluate",
                    "parameters": {"expression": "console.error"}
                })
        
        return analysis
    
    def _analyze_console_result(self, result: Dict) -> Dict:
        """콘솔 로그 결과 분석"""
        logs = result.get("console_logs", [])
        
        analysis = {
            "console_analysis": {
                "total_logs": len(logs),
                "error_count": 0,
                "warning_count": 0,
                "critical_errors": []
            }
        }
        
        for log in logs:
            log_str = str(log).lower()
            
            if "error" in log_str or "exception" in log_str:
                analysis["console_analysis"]["error_count"] += 1
                
                if any(critical in log_str for critical in ["uncaught", "fatal", "crash"]):
                    analysis["console_analysis"]["critical_errors"].append(log)
                    
                    # 심각한 에러에 대한 즉시 액션 제안
                    analysis.setdefault("claude_actions_needed", []).append({
                        "action": "fix_critical_error",
                        "description": f"심각한 JavaScript 에러: {log}",
                        "priority": "critical",
                        "auto_fixable": False,
                        "investigation_needed": True
                    })
            
            elif "warning" in log_str:
                analysis["console_analysis"]["warning_count"] += 1
        
        return analysis
    
    def _analyze_interaction_result(self, result: Dict) -> Dict:
        """상호작용 결과 분석"""
        success = result.get("success", False)
        error = result.get("error")
        
        analysis = {
            "interaction_analysis": {
                "action_type": result.get("action"),
                "success": success,
                "element_found": result.get("element_found", True)
            }
        }
        
        if not success:
            analysis["claude_actions_needed"] = [{
                "action": "debug_interaction",
                "description": f"상호작용 실패: {error or '원인 불명'}",
                "priority": "high",
                "suggested_solutions": [
                    "요소 선택자 확인",
                    "페이지 로딩 완료 대기",
                    "요소 visibility 확인"
                ]
            }]
            
            # 자동 수정 제안
            analysis["auto_fix_suggestions"] = [
                {
                    "fix_type": "wait_and_retry",
                    "description": "요소 로딩 대기 후 재시도",
                    "mcp_sequence": [
                        "mcp__playwright__wait_for_selector",
                        "mcp__playwright__click"
                    ]
                }
            ]
        
        return analysis
    
    def _analyze_network_result(self, result: Dict) -> Dict:
        """네트워크 응답 결과 분석"""
        status_code = result.get("status_code", 0)
        response_time = result.get("response_time", 0)
        
        analysis = {
            "network_analysis": {
                "status_code": status_code,
                "response_time": response_time,
                "is_success": 200 <= status_code < 300,
                "performance_issue": response_time > 3000
            }
        }
        
        # 상태 코드별 분석
        if status_code >= 400:
            severity = "critical" if status_code >= 500 else "high"
            analysis["claude_actions_needed"] = [{
                "action": "fix_network_error",
                "description": f"HTTP {status_code} 에러 해결 필요",
                "priority": severity,
                "investigation_areas": [
                    "API 엔드포인트 확인",
                    "서버 상태 점검",
                    "네트워크 연결 확인"
                ]
            }]
        
        # 성능 문제
        if response_time > 3000:
            analysis["claude_actions_needed"] = analysis.get("claude_actions_needed", []) + [{
                "action": "optimize_performance",
                "description": f"응답 시간 느림: {response_time}ms",
                "priority": "medium",
                "optimization_suggestions": [
                    "캐싱 구현",
                    "API 최적화",
                    "로딩 지표 개선"
                ]
            }]
        
        return analysis
    
    def _generate_conversation_context(self, report: Dict) -> str:
        """클로드와의 대화를 위한 컨텍스트 생성"""
        context = f"""
🤖 **웹 테스트 실시간 분석 결과**

**시간**: {report['timestamp']}
**액션**: {report['playwright_action']}
**상태**: {report['status']}

"""
        
        # 즉시 발견된 문제들
        if report.get("immediate_findings"):
            context += "**🚨 발견된 문제들:**\n"
            for finding in report["immediate_findings"]:
                context += f"- {finding}\n"
            context += "\n"
        
        # 클로드가 해야 할 액션들
        if report.get("claude_actions_needed"):
            context += "**🎯 클로드에게 요청할 액션들:**\n"
            for i, action in enumerate(report["claude_actions_needed"], 1):
                context += f"{i}. **{action['action']}** ({action['priority']})\n"
                context += f"   - {action['description']}\n"
                if action.get("suggested_mcp"):
                    context += f"   - 사용할 MCP: `{action['suggested_mcp']}`\n"
                context += "\n"
        
        # 자동 수정 제안
        if report.get("auto_fix_suggestions"):
            context += "**🔧 자동 수정 가능한 항목들:**\n"
            for suggestion in report["auto_fix_suggestions"]:
                context += f"- {suggestion['description']}\n"
            context += "\n"
        
        context += "**💬 클로드야, 위 결과를 바탕으로 다음 액션을 실행해줘!**"
        
        return context
    
    def _save_claude_report(self, report: Dict):
        """클로드 리포트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.claude_dir / f"claude_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 최신 리포트 링크 생성
        latest_file = self.claude_dir / "latest_report.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"클로드 리포트 저장됨: {report_file}")
    
    def create_claude_action_script(self, actions: List[Dict]) -> str:
        """클로드가 실행할 수 있는 MCP 스크립트 생성"""
        script = "# 🤖 클로드 자동 실행 스크립트\n\n"
        
        for i, action in enumerate(actions, 1):
            script += f"## {i}. {action['action']}\n"
            script += f"**설명**: {action['description']}\n"
            script += f"**우선순위**: {action['priority']}\n\n"
            
            if action.get("suggested_mcp"):
                script += f"```python\n"
                script += f"# MCP 함수 실행\n"
                script += f"result = await {action['suggested_mcp']}(\n"
                
                if action.get("parameters"):
                    for key, value in action["parameters"].items():
                        script += f"    {key}='{value}',\n"
                
                script += f")\n"
                script += f"print(f'실행 결과: {{result}}')\n"
                script += f"```\n\n"
        
        return script
    
    def get_latest_conversation_context(self) -> str:
        """최신 대화 컨텍스트 가져오기"""
        latest_file = self.claude_dir / "latest_report.json"
        
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            return report.get("conversation_context", "")
        
        return "아직 분석 결과가 없습니다."

def main():
    """테스트용 메인 함수"""
    from pathlib import Path
    
    # 테스트 세션 디렉토리 생성
    session_dir = Path("test_results/test_session_20250723_120000")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    communicator = ClaudeWebCommunicator(session_dir)
    
    # 예시 Playwright 결과
    playwright_result = {
        "action": "screenshot",
        "screenshot_path": "test_screenshot.png",
        "success": True,
        "timestamp": datetime.now().isoformat()
    }
    
    # 실시간 리포트 생성
    report = communicator.generate_realtime_report(playwright_result)
    
    print("🤖 클로드 커뮤니케이션 테스트:")
    print(f"리포트 생성됨: {len(report)} 항목")
    print(f"클로드 액션 필요: {len(report.get('claude_actions_needed', []))} 개")
    
    # 대화 컨텍스트 출력
    print("\n" + "="*50)
    print("클로드와의 대화 컨텍스트:")
    print("="*50)
    print(report["conversation_context"])

if __name__ == "__main__":
    main()