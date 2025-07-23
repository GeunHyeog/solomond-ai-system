#!/usr/bin/env python3
"""
웹 브라우저 테스트 자동화 시스템 with Playwright MCP
- 클로드와 실시간 소통을 위한 브라우저 테스트 자동화
- 테스트 결과 자동 캡처 및 분석
- 에러 감지 및 자동 해결 시도
"""

import json
import time
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from claude_web_communicator import ClaudeWebCommunicator
from core.web_test_analyzer import WebTestAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebTestAutomator:
    """웹 테스트 자동화 클래스"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.test_results_dir = self.project_root / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        # 테스트 세션 설정
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.test_results_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # 통합 컴포넌트 초기화
        self.communicator = ClaudeWebCommunicator(self.session_dir)
        self.analyzer = WebTestAnalyzer(str(self.project_root))
        self.demo_captures = []  # demo_capture_system.py 호환성
        
        logger.info(f"테스트 세션 시작: {self.session_id}")
        logger.info(f"클로드 커뮤니케이터 연결: {self.communicator.claude_dir}")
        
    def start_test_monitoring(self, url: str, test_scenarios: List[Dict] = None):
        """
        브라우저 테스트 모니터링 시작
        
        Args:
            url: 테스트할 웹페이지 URL
            test_scenarios: 테스트 시나리오 리스트
        """
        logger.info(f"테스트 모니터링 시작: {url}")
        
        # 기본 테스트 시나리오
        if not test_scenarios:
            test_scenarios = [
                {"name": "페이지 로드 테스트", "action": "page_load"},
                {"name": "UI 요소 검증", "action": "ui_validation"},
                {"name": "기능 테스트", "action": "functional_test"},
                {"name": "에러 감지", "action": "error_detection"}
            ]
        
        # 테스트 설정 저장
        test_config = {
            "session_id": self.session_id,
            "url": url,
            "scenarios": test_scenarios,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        config_file = self.session_dir / "test_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        
        return {
            "session_id": self.session_id,
            "config_file": str(config_file),
            "test_scenarios": test_scenarios,
            "communicator": self.communicator,
            "demo_capture_compatible": True,
            "message": "테스트 모니터링이 시작되었습니다. Claude Code 재시작 후 Playwright MCP로 실행하세요."
        }
        
    def create_test_report_template(self):
        """테스트 리포트 템플릿 생성"""
        template = {
            "session_info": {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "url": "",
                "browser": "chromium"
            },
            "test_results": [],
            "issues_found": [],
            "recommendations": [],
            "claude_communication": {
                "auto_report": True,
                "issue_notifications": True,
                "solution_suggestions": True
            }
        }
        
        template_file = self.session_dir / "test_report_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
            
        return template_file
        
    def integrate_with_demo_capture(self, demo_capture_data: List[Dict] = None):
        """기존 demo_capture_system.py 데이터와 통합"""
        if demo_capture_data:
            self.demo_captures = demo_capture_data
            logger.info(f"Demo capture 데이터 {len(demo_capture_data)}개 통합됨")
            
            # 클로드 커뮤니케이터에 데이터 전달
            for capture in demo_capture_data:
                playwright_result = {
                    "action": "demo_capture",
                    "screenshot_path": capture.get("screenshot_path"),
                    "streamlit_data": capture.get("streamlit_data", {}),
                    "text_content": capture.get("text_content", []),
                    "timestamp": capture.get("timestamp"),
                    "page_info": capture.get("page_info", {})
                }
                
                # 실시간 분석 리포트 생성
                claude_report = self.communicator.generate_realtime_report(playwright_result)
                logger.info(f"클로드 리포트 생성: {claude_report['timestamp']}")
        
        return {
            "integrated_captures": len(self.demo_captures),
            "claude_reports_generated": len(demo_capture_data) if demo_capture_data else 0,
            "analysis_ready": True
        }
    
    def create_playwright_mcp_script(self):
        """Playwright MCP 실행을 위한 스크립트 템플릿 생성"""
        script_content = '''
# 🤖 통합 웹 테스트 자동화 시스템
# Claude Code 재시작 후 다음 MCP 함수들을 사용하세요:
# Demo Capture System과 완전 호환 가능!

## 1. 브라우저 시작 및 페이지 이동
```python
# 브라우저 시작
await mcp__playwright__launch_browser(browser_type="chromium", headless=True)

# 페이지 이동
await mcp__playwright__goto(url="YOUR_TEST_URL")

# 스크린샷 캡처
screenshot_path = await mcp__playwright__screenshot(path="test_screenshot.png")
```

## 2. 페이지 상태 분석
```python
# 페이지 제목 확인
title = await mcp__playwright__get_title()

# 현재 URL 확인
current_url = await mcp__playwright__get_url()

# 페이지 콘텐츠 추출
content = await mcp__playwright__get_content()
```

## 3. 에러 감지 및 분석
```python
# 콘솔 에러 확인
console_logs = await mcp__playwright__get_console_logs()

# 네트워크 에러 확인
network_errors = await mcp__playwright__get_network_errors()

# JavaScript 에러 감지
js_errors = await mcp__playwright__evaluate("window.onerror")
```

## 4. 자동 테스트 실행
```python
# 폼 입력
await mcp__playwright__fill(selector="#input-field", value="test data")

# 버튼 클릭
await mcp__playwright__click(selector="#submit-button")

# 요소 대기
await mcp__playwright__wait_for_selector(selector="#result-container")
```

## 5. 클로드와 실시간 소통
```python
# 웹 테스트 자동화 시스템 사용
from web_test_automation import WebTestAutomator
from claude_web_communicator import ClaudeWebCommunicator

# 자동화 시스템 초기화
automator = WebTestAutomator()
result = automator.start_test_monitoring("http://localhost:8503")

# Playwright MCP 실행 결과를 클로드 리포트로 변환
playwright_result = {
    "action": "screenshot",
    "screenshot_path": "test.png",
    "success": True
}

# 실시간 클로드 소통
claude_report = automator.communicator.generate_realtime_report(playwright_result)
print(claude_report["conversation_context"])  # 클로드와 대화할 내용
```

## 6. Demo Capture System 통합
```python
# 기존 demo_capture_system.py 데이터 활용
demo_data = [...] # demo_capture_system에서 생성된 데이터
automator.integrate_with_demo_capture(demo_data)

# 통합 분석 실행
analysis = automator.analyzer.analyze_test_session(automator.session_dir)
print(analysis["claude_report"])  # 클로드용 종합 리포트
```
'''
        
        script_file = self.session_dir / "playwright_mcp_guide.md"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        return script_file

def main():
    """메인 실행 함수"""
    automator = WebTestAutomator()
    
    # 예시: 테스트 시나리오 설정
    test_scenarios = [
        {
            "name": "로그인 기능 테스트",
            "action": "login_test",
            "selector": "#login-form",
            "expected": "로그인 성공"
        },
        {
            "name": "데이터 로딩 테스트", 
            "action": "data_loading",
            "selector": ".data-container",
            "expected": "데이터 표시"
        },
        {
            "name": "반응형 UI 테스트",
            "action": "responsive_test",
            "viewport": {"width": 1920, "height": 1080},
            "expected": "올바른 레이아웃"
        }
    ]
    
    # 테스트 모니터링 시작
    result = automator.start_test_monitoring(
        url="http://localhost:8503",  # Streamlit 앱 기본 URL
        test_scenarios=test_scenarios
    )
    
    # 템플릿 파일들 생성
    template_file = automator.create_test_report_template()
    script_file = automator.create_playwright_mcp_script()
    
    print("\n🚀 웹 테스트 자동화 시스템 준비 완료!")
    print(f"세션 ID: {result['session_id']}")
    print(f"설정 파일: {result['config_file']}")
    print(f"리포트 템플릿: {template_file}")
    print(f"Playwright MCP 가이드: {script_file}")
    print("\n📌 다음 단계:")
    print("1. Claude Code 재시작")
    print("2. Playwright MCP 함수들 사용 가능 확인")
    print("3. 생성된 가이드 파일 참조하여 테스트 실행")
    
    return result

if __name__ == "__main__":
    main()