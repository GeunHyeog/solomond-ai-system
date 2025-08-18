#!/usr/bin/env python3
"""
통합 개발 툴킷 - MCP 서버 대안
Playwright, GitHub, Supabase, 웹검색 기능을 직접 구현
"""

import asyncio
import json
import os
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from config import SETTINGS

# Playwright 임포트 (선택적)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Supabase 임포트 (선택적)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

class IntegratedDevelopmentToolkit:
    """통합 개발 툴킷"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[INIT] 통합 개발 툴킷 초기화 완료 - Session: {self.session_id}")
        
        # 기본 설정
        self.output_dir = Path("toolkit_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Supabase 설정 (직접 설정 + 환경변수 백업)
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://qviccikgyspkyqpemert.supabase.co')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')  # ANON KEY는 별도 필요
        self.supabase_access_token = os.environ.get('SUPABASE_ACCESS_TOKEN', 'SUPABASE_TOKEN_NOT_SET')
        self.supabase_client = None
        
        if self.supabase_url and self.supabase_key and SUPABASE_AVAILABLE:
            try:
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                print("[OK] Supabase 연결 완료")
            except Exception as e:
                print(f"[ERROR] Supabase 연결 실패: {e}")
    
    # 1. Playwright 브라우저 자동화
    async def launch_browser_session(self, url: str = "http://f"localhost:{SETTINGS['PORT']}"", headless: bool = False):
        """브라우저 세션 시작"""
        
        if not PLAYWRIGHT_AVAILABLE:
            print("[ERROR] Playwright 설치 필요: pip install playwright")
            return None
        
        print(f"[BROWSER] 브라우저 세션 시작: {url}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            
            # 스크린샷 캡처
            screenshot_path = self.output_dir / f"browser_session_{self.session_id}.png"
            await page.screenshot(path=str(screenshot_path))
            
            print(f"[SCREENSHOT] 스크린샷 저장: {screenshot_path}")
            
            # 페이지 정보 수집
            page_info = {
                "url": page.url,
                "title": await page.title(),
                "timestamp": datetime.now().isoformat(),
                "screenshot": str(screenshot_path)
            }
            
            return {"browser": browser, "page": page, "info": page_info}
    
    async def capture_page_content(self, page, selector: str = None):
        """페이지 콘텐츠 캡처"""
        
        if selector:
            element = await page.query_selector(selector)
            if element:
                content = await element.inner_text()
            else:
                content = "Element not found"
        else:
            content = await page.content()
        
        return content
    
    # 2. GitHub 연동 (requests 사용)
    def github_api_request(self, endpoint: str, method: str = "GET", data: Dict = None):
        """GitHub API 요청"""
        
        github_token = os.environ.get('GITHUB_ACCESS_TOKEN', 'GITHUB_TOKEN_NOT_SET')
        
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }
        
        url = f"https://api.github.com/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] GitHub API 오류: {e}")
            return None
    
    def get_repo_info(self, owner: str, repo: str):
        """저장소 정보 조회"""
        return self.github_api_request(f"repos/{owner}/{repo}")
    
    def list_issues(self, owner: str, repo: str, state: str = "open"):
        """이슈 목록 조회"""
        return self.github_api_request(f"repos/{owner}/{repo}/issues?state={state}")
    
    def create_issue(self, owner: str, repo: str, title: str, body: str, labels: List[str] = None):
        """이슈 생성"""
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        
        return self.github_api_request(f"repos/{owner}/{repo}/issues", "POST", data)
    
    # 3. Supabase 데이터베이스 연동
    def supabase_query(self, table: str, operation: str = "select", data: Dict = None, filters: Dict = None):
        """Supabase 쿼리 실행"""
        
        if not self.supabase_client:
            print("[ERROR] Supabase 클라이언트가 초기화되지 않았습니다.")
            return None
        
        try:
            query = self.supabase_client.table(table)
            
            if operation == "select":
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                result = query.execute()
                
            elif operation == "insert":
                result = query.insert(data).execute()
                
            elif operation == "update":
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                result = query.update(data).execute()
                
            elif operation == "delete":
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                result = query.delete().execute()
            
            return result.data
            
        except Exception as e:
            print(f"[ERROR] Supabase 쿼리 오류: {e}")
            return None
    
    def save_development_log(self, action: str, details: Dict):
        """개발 로그 저장"""
        log_entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        
        return self.supabase_query("development_logs", "insert", log_entry)
    
    # 4. 웹 검색 기능
    def web_search(self, query: str, search_engine: str = "duckduckgo"):
        """웹 검색 실행"""
        
        if search_engine == "duckduckgo":
            return self._duckduckgo_search(query)
        else:
            print(f"[ERROR] 지원하지 않는 검색 엔진: {search_engine}")
            return None
    
    def _duckduckgo_search(self, query: str, max_results: int = 10):
        """DuckDuckGo 검색"""
        
        try:
            import duckduckgo_search
            with duckduckgo_search.DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append(result)
                
                return results
                
        except ImportError:
            print("[ERROR] duckduckgo-search 설치 필요: pip install duckduckgo-search")
            return None
        except Exception as e:
            print(f"[ERROR] 웹 검색 오류: {e}")
            return None
    
    def fetch_webpage_content(self, url: str):
        """웹페이지 콘텐츠 가져오기"""
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 간단한 HTML 파싱
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 텍스트만 추출
            text_content = soup.get_text()
            
            return {
                "url": url,
                "title": soup.title.string if soup.title else "No title",
                "content": text_content[:5000],  # 첫 5000자만
                "status_code": response.status_code
            }
            
        except ImportError:
            print("[ERROR] beautifulsoup4 설치 필요: pip install beautifulsoup4")
            return None
        except Exception as e:
            print(f"[ERROR] 웹페이지 가져오기 오류: {e}")
            return None
    
    # 5. 통합 워크플로우
    async def integrated_development_workflow(self, task_description: str):
        """통합 개발 워크플로우"""
        
        print(f"[START] 통합 개발 워크플로우 시작: {task_description}")
        
        workflow_log = {
            "task": task_description,
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. 브라우저 세션 시작
        browser_session = await self.launch_browser_session()
        if browser_session:
            workflow_log["steps"].append("browser_launched")
        
        # 2. GitHub 저장소 정보 확인
        repo_info = self.get_repo_info("GeunHyeog", "solomond-ai-system")
        if repo_info:
            workflow_log["steps"].append("github_checked")
        
        # 3. 개발 로그 저장 (Supabase)
        if self.supabase_client:
            self.save_development_log("workflow_start", {"task": task_description})
            workflow_log["steps"].append("log_saved")
        
        # 4. 관련 정보 웹 검색
        search_results = self.web_search(f"Python {task_description}")
        if search_results:
            workflow_log["steps"].append("web_searched")
        
        # 결과 저장
        output_file = self.output_dir / f"workflow_{self.session_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(workflow_log, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 워크플로우 완료 - 결과: {output_file}")
        
        return workflow_log

# 사용 예시
async def main():
    toolkit = IntegratedDevelopmentToolkit()
    
    # 통합 워크플로우 실행
    result = await toolkit.integrated_development_workflow("Streamlit 앱 개발")
    
    print(f"워크플로우 결과: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())