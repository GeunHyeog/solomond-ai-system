#!/usr/bin/env python3
"""
웹 테스트 결과 자동 분석 엔진
- Playwright MCP 캡처 결과 분석
- 에러 패턴 감지 및 분류
- 클로드와의 소통을 위한 리포트 생성
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TestIssue:
    """테스트 이슈 데이터 클래스"""
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    location: str  # CSS selector or URL
    screenshot_path: Optional[str] = None
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False

@dataclass
class TestResult:
    """테스트 결과 데이터 클래스"""
    test_name: str
    status: str  # passed, failed, error
    duration: float
    issues: List[TestIssue]
    screenshot_path: Optional[str] = None
    console_logs: List[str] = None
    network_errors: List[str] = None

class WebTestAnalyzer:
    """웹 테스트 결과 분석기"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.analysis_patterns = self._load_analysis_patterns()
        
    def _load_analysis_patterns(self) -> Dict:
        """분석 패턴 로드"""
        return {
            "error_patterns": {
                "javascript_error": r"(TypeError|ReferenceError|SyntaxError|Error):\s*(.+)",
                "network_error": r"(404|500|502|503|timeout|CORS|ERR_)",
                "security_error": r"(CSP|CSRF|XSS|blocked|refused)",
                "performance_error": r"(slow|timeout|memory|leak)"
            },
            "ui_patterns": {
                "missing_element": r"element not found|selector.*not.*found",
                "layout_issue": r"overflow|z-index|position|display",
                "responsive_issue": r"viewport|media query|breakpoint",
                "accessibility_issue": r"aria|alt|contrast|focus"
            },
            "functional_patterns": {
                "form_error": r"validation|required|invalid|submit",
                "auth_error": r"login|authentication|authorization|token",
                "data_error": r"loading|fetch|api|database|null"
            }
        }
    
    def analyze_test_session(self, session_dir: Path) -> Dict[str, Any]:
        """테스트 세션 전체 분석"""
        logger.info(f"테스트 세션 분석 시작: {session_dir}")
        
        analysis_result = {
            "session_id": session_dir.name,
            "analysis_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "critical_issues": 0,
                "auto_fixable_issues": 0
            },
            "test_results": [],
            "critical_issues": [],
            "recommendations": [],
            "claude_report": ""
        }
        
        # 테스트 결과 파일들 분석
        for result_file in session_dir.glob("test_result_*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                
                test_result = self._analyze_single_test(test_data)
                analysis_result["test_results"].append(test_result)
                analysis_result["summary"]["total_tests"] += 1
                
                if test_result.status == "passed":
                    analysis_result["summary"]["passed"] += 1
                else:
                    analysis_result["summary"]["failed"] += 1
                
                # 심각한 이슈 수집
                for issue in test_result.issues:
                    if issue.severity == "critical":
                        analysis_result["critical_issues"].append(issue)
                        analysis_result["summary"]["critical_issues"] += 1
                    
                    if issue.auto_fixable:
                        analysis_result["summary"]["auto_fixable_issues"] += 1
                        
            except Exception as e:
                logger.error(f"테스트 결과 분석 실패: {result_file}, {e}")
        
        # 추천사항 생성
        analysis_result["recommendations"] = self._generate_recommendations(analysis_result)
        
        # 클로드 리포트 생성
        analysis_result["claude_report"] = self._generate_claude_report(analysis_result)
        
        # 분석 결과 저장
        analysis_file = session_dir / "analysis_result.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)
        
        return analysis_result
    
    def _analyze_single_test(self, test_data: Dict) -> TestResult:
        """단일 테스트 결과 분석"""
        issues = []
        
        # 콘솔 로그 분석
        if test_data.get("console_logs"):
            for log in test_data["console_logs"]:
                issues.extend(self._analyze_console_log(log))
        
        # 네트워크 에러 분석
        if test_data.get("network_errors"):
            for error in test_data["network_errors"]:
                issues.extend(self._analyze_network_error(error))
        
        # 스크린샷 분석 (기본적인 패턴 매칭)
        if test_data.get("screenshot_analysis"):
            issues.extend(self._analyze_screenshot_data(test_data["screenshot_analysis"]))
        
        return TestResult(
            test_name=test_data.get("test_name", "Unknown"),
            status=test_data.get("status", "unknown"),
            duration=test_data.get("duration", 0),
            issues=issues,
            screenshot_path=test_data.get("screenshot_path"),
            console_logs=test_data.get("console_logs", []),
            network_errors=test_data.get("network_errors", [])
        )
    
    def _analyze_console_log(self, log_entry: str) -> List[TestIssue]:
        """콘솔 로그 분석"""
        issues = []
        
        for pattern_name, pattern in self.analysis_patterns["error_patterns"].items():
            if re.search(pattern, log_entry, re.IGNORECASE):
                severity = self._determine_severity(pattern_name, log_entry)
                suggested_fix = self._suggest_fix(pattern_name, log_entry)
                
                issue = TestIssue(
                    issue_type=f"console_{pattern_name}",
                    severity=severity,
                    description=f"콘솔 에러 감지: {log_entry[:100]}...",
                    location="browser_console",
                    suggested_fix=suggested_fix,
                    auto_fixable=self._is_auto_fixable(pattern_name)
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_network_error(self, error: str) -> List[TestIssue]:
        """네트워크 에러 분석"""
        issues = []
        
        if re.search(r"404|not found", error, re.IGNORECASE):
            issue = TestIssue(
                issue_type="network_404",
                severity="high",
                description=f"리소스를 찾을 수 없음: {error}",
                location="network",
                suggested_fix="URL 경로 확인 또는 파일 존재 여부 확인",
                auto_fixable=False
            )
            issues.append(issue)
        
        elif re.search(r"500|server error", error, re.IGNORECASE):
            issue = TestIssue(
                issue_type="network_500",
                severity="critical",
                description=f"서버 에러: {error}",
                location="server",
                suggested_fix="서버 로그 확인 및 백엔드 디버깅 필요",
                auto_fixable=False
            )
            issues.append(issue)
        
        return issues
    
    def _analyze_screenshot_data(self, screenshot_data: Dict) -> List[TestIssue]:
        """스크린샷 데이터 분석 (OCR 텍스트 기반)"""
        issues = []
        
        if screenshot_data.get("error_messages"):
            for error_msg in screenshot_data["error_messages"]:
                issue = TestIssue(
                    issue_type="ui_error_message",
                    severity="high",
                    description=f"화면에 에러 메시지 표시: {error_msg}",
                    location="ui_display",
                    suggested_fix="에러 메시지 원인 분석 및 수정",
                    auto_fixable=False
                )
                issues.append(issue)
        
        return issues
    
    def _determine_severity(self, pattern_name: str, content: str) -> str:
        """에러 심각도 결정"""
        critical_keywords = ["critical", "fatal", "crash", "memory", "security"]
        high_keywords = ["error", "exception", "failed", "timeout"]
        
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in critical_keywords):
            return "critical"
        elif any(keyword in content_lower for keyword in high_keywords):
            return "high"
        elif pattern_name in ["javascript_error", "network_error"]:
            return "high"
        else:
            return "medium"
    
    def _suggest_fix(self, pattern_name: str, content: str) -> str:
        """수정 제안 생성"""
        fix_suggestions = {
            "javascript_error": "JavaScript 코드 검토 및 디버깅",
            "network_error": "네트워크 연결 및 API 엔드포인트 확인",
            "security_error": "보안 정책 및 CORS 설정 검토",
            "performance_error": "성능 최적화 및 메모리 사용량 검토"
        }
        
        return fix_suggestions.get(pattern_name, "상세 분석 및 로그 확인 필요")
    
    def _is_auto_fixable(self, pattern_name: str) -> bool:
        """자동 수정 가능 여부 판단"""
        auto_fixable_patterns = ["ui_refresh", "cache_clear", "form_reset"]
        return pattern_name in auto_fixable_patterns
    
    def _generate_recommendations(self, analysis_result: Dict) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        critical_count = analysis_result["summary"]["critical_issues"]
        if critical_count > 0:
            recommendations.append(f"🚨 {critical_count}개의 심각한 문제가 발견되었습니다. 즉시 수정이 필요합니다.")
        
        auto_fix_count = analysis_result["summary"]["auto_fixable_issues"]
        if auto_fix_count > 0:
            recommendations.append(f"🔧 {auto_fix_count}개의 자동 수정 가능한 문제가 있습니다.")
        
        failed_tests = analysis_result["summary"]["failed"]
        total_tests = analysis_result["summary"]["total_tests"]
        if failed_tests > 0:
            failure_rate = (failed_tests / total_tests) * 100
            recommendations.append(f"📊 테스트 실패율: {failure_rate:.1f}% ({failed_tests}/{total_tests})")
        
        return recommendations
    
    def _generate_claude_report(self, analysis_result: Dict) -> str:
        """클로드와의 소통을 위한 리포트 생성"""
        report = f"""
# 🔍 웹 테스트 자동 분석 리포트

## 📊 테스트 요약
- **세션 ID**: {analysis_result['session_id']}
- **분석 시간**: {analysis_result['analysis_time']}
- **총 테스트**: {analysis_result['summary']['total_tests']}개
- **성공**: {analysis_result['summary']['passed']}개
- **실패**: {analysis_result['summary']['failed']}개
- **심각한 문제**: {analysis_result['summary']['critical_issues']}개

## 🚨 발견된 주요 문제들
"""
        
        for i, issue in enumerate(analysis_result['critical_issues'][:5], 1):
            report += f"""
### {i}. {issue.issue_type}
- **심각도**: {issue.severity}
- **설명**: {issue.description}
- **위치**: {issue.location}
- **제안 수정**: {issue.suggested_fix}
"""
        
        report += f"""
## 💡 추천사항
"""
        for rec in analysis_result['recommendations']:
            report += f"- {rec}\n"
        
        report += f"""
## 🤖 클로드에게 요청할 수 있는 작업들
1. **자동 수정**: `playwright_auto_fix_issues()` 실행
2. **상세 분석**: `analyze_specific_error()` 실행  
3. **테스트 재실행**: `rerun_failed_tests()` 실행
4. **성능 최적화**: `optimize_performance()` 실행

**준비된 MCP 함수들을 사용해서 위 작업들을 자동으로 실행할 수 있습니다.**
"""
        
        return report

def main():
    """테스트용 메인 함수"""
    analyzer = WebTestAnalyzer()
    
    # 예시 테스트 데이터 생성
    test_data = {
        "test_name": "로그인 기능 테스트",
        "status": "failed",
        "duration": 5.2,
        "console_logs": [
            "TypeError: Cannot read property 'value' of null",
            "Network error: 404 - /api/login not found"
        ],
        "network_errors": [
            "404: GET /api/login - Not Found"
        ],
        "screenshot_analysis": {
            "error_messages": ["사용자 이름 또는 비밀번호가 잘못되었습니다."]
        }
    }
    
    # 단일 테스트 분석
    result = analyzer._analyze_single_test(test_data)
    
    print("🔍 테스트 분석 결과:")
    print(f"테스트: {result.test_name}")
    print(f"상태: {result.status}")
    print(f"발견된 문제: {len(result.issues)}개")
    
    for issue in result.issues:
        print(f"- {issue.issue_type}: {issue.description}")

if __name__ == "__main__":
    main()