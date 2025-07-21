#!/usr/bin/env python3
"""
Server Status Checker using Playwright
Checks multiple localhost servers for status, errors, and captures screenshots
"""

import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("ERROR: Playwright installation required: pip install playwright")

class ServerStatusChecker:
    """서버 상태 검사 시스템"""
    
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.check_dir = Path("server_checks")
        self.check_dir.mkdir(exist_ok=True)
        
        # 검사 결과
        self.results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Server Status Checker initialized")
        print(f"Results directory: {self.check_dir}")
        print(f"Target servers: {', '.join(servers)}")
    
    async def check_all_servers(self):
        """모든 서버 상태 검사"""
        
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: Playwright is not installed.")
            return None
        
        print(f"Starting server status check...")
        
        async with async_playwright() as p:
            # 브라우저 시작 (헤드리스 모드)
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080}
            )
            
            try:
                for server_url in self.servers:
                    print(f"\nChecking: {server_url}")
                    result = await self.check_single_server(context, server_url)
                    self.results.append(result)
                    
                    # 간단한 결과 출력
                    if result['success']:
                        print(f"SUCCESS {server_url}: {result['page_title']}")
                    else:
                        print(f"FAILED {server_url}: {result['error']}")
                
                # 결과 리포트 생성
                report = self.generate_report()
                
                print(f"\nCheck completed! Results:")
                print(f"- Success: {sum(1 for r in self.results if r['success'])}/{len(self.results)}")
                print(f"- Failed: {sum(1 for r in self.results if not r['success'])}/{len(self.results)}")
                
                return report
                
            except Exception as e:
                print(f"ERROR during check: {e}")
                return None
            
            finally:
                await browser.close()
    
    async def check_single_server(self, context, server_url: str) -> Dict[str, Any]:
        """단일 서버 상태 검사"""
        
        page = await context.new_page()
        
        # 콘솔 로그 수집
        console_logs = []
        page.on("console", lambda msg: console_logs.append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location
        }))
        
        # 페이지 에러 수집
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        
        try:
            # 페이지 로드 시도
            start_time = time.time()
            response = await page.goto(server_url, wait_until='networkidle', timeout=10000)
            load_time = time.time() - start_time
            
            # 기본 정보 수집
            page_title = await page.title()
            page_url = page.url
            viewport = await page.evaluate('() => ({ width: window.innerWidth, height: window.innerHeight })')
            
            # 스크린샷 캐쳐
            screenshot_path = self.check_dir / f"screenshot_{server_url.replace(':', '_').replace('/', '_')}_{self.session_id}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Streamlit 특화 정보 수집
            streamlit_info = await self.extract_streamlit_info(page)
            
            # DOM 요소 확인
            dom_info = await self.extract_dom_info(page)
            
            # 네트워크 상태 확인
            network_info = {
                "response_status": response.status if response else None,
                "response_ok": response.ok if response else False,
                "load_time_seconds": round(load_time, 2)
            }
            
            result = {
                "server_url": server_url,
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "page_title": page_title,
                "page_url": page_url,
                "viewport": viewport,
                "network_info": network_info,
                "streamlit_info": streamlit_info,
                "dom_info": dom_info,
                "console_logs": console_logs,
                "page_errors": page_errors,
                "screenshot_path": str(screenshot_path),
                "error": None
            }
            
        except Exception as e:
            result = {
                "server_url": server_url,
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "console_logs": console_logs,
                "page_errors": page_errors,
                "screenshot_path": None
            }
        
        finally:
            await page.close()
        
        return result
    
    async def extract_streamlit_info(self, page) -> Dict[str, Any]:
        """Streamlit 특화 정보 추출"""
        
        try:
            streamlit_info = await page.evaluate('''() => {
                const data = {};
                
                // Streamlit 앱 감지
                data.is_streamlit = !!document.querySelector('[data-testid="stApp"]');
                
                // 현재 활성 탭
                const activeTab = document.querySelector('[data-baseweb="tab"][aria-selected="true"]');
                if (activeTab) {
                    data.current_tab = activeTab.textContent;
                }
                
                // 파일 업로드 영역
                const fileUploaders = document.querySelectorAll('[data-testid="stFileUploader"]');
                data.file_uploaders_count = fileUploaders.length;
                
                // 버튼들
                const buttons = document.querySelectorAll('button[kind="primary"], button[kind="secondary"]');
                data.buttons = Array.from(buttons).map(btn => btn.textContent);
                
                // 메트릭 컨테이너
                const metrics = {};
                document.querySelectorAll('[data-testid="metric-container"]').forEach((metric, index) => {
                    const divs = metric.querySelectorAll('div');
                    if (divs.length >= 2) {
                        const label = divs[0].textContent;
                        const value = divs[divs.length-1].textContent;
                        metrics[label || `metric_${index}`] = value;
                    }
                });
                data.metrics = metrics;
                
                // 에러/성공 메시지
                const messages = {
                    success: Array.from(document.querySelectorAll('.stSuccess')).map(el => el.textContent),
                    error: Array.from(document.querySelectorAll('.stError')).map(el => el.textContent),
                    warning: Array.from(document.querySelectorAll('.stWarning')).map(el => el.textContent),
                    info: Array.from(document.querySelectorAll('.stInfo')).map(el => el.textContent)
                };
                data.messages = messages;
                
                // 프로그레스 바
                const progressBars = document.querySelectorAll('[data-testid="stProgress"]');
                data.progress_bars = Array.from(progressBars).map(pb => {
                    const progressElement = pb.querySelector('[role="progressbar"]');
                    return progressElement ? progressElement.getAttribute('aria-valuenow') : '0';
                });
                
                // 사이드바 상태
                data.sidebar_visible = !!document.querySelector('[data-testid="stSidebar"]');
                
                return data;
            }''')
            
            return streamlit_info
            
        except Exception as e:
            print(f"WARNING: Streamlit info extraction failed: {e}")
            return {"error": str(e)}
    
    async def extract_dom_info(self, page) -> Dict[str, Any]:
        """DOM 기본 정보 추출"""
        
        try:
            dom_info = await page.evaluate('''() => {
                const data = {};
                
                // 기본 DOM 정보
                data.body_exists = !!document.body;
                data.head_exists = !!document.head;
                
                // 스크립트 태그 수
                data.script_tags_count = document.querySelectorAll('script').length;
                
                // CSS 링크 수
                data.css_links_count = document.querySelectorAll('link[rel="stylesheet"]').length;
                
                // 이미지 수
                data.images_count = document.querySelectorAll('img').length;
                
                // 폼 수
                data.forms_count = document.querySelectorAll('form').length;
                
                // 입력 필드 수
                data.input_fields_count = document.querySelectorAll('input, textarea, select').length;
                
                // React/Streamlit 관련 요소
                data.react_root = !!document.querySelector('[id*="root"], [class*="stApp"]');
                
                // 로딩 상태 감지
                data.loading_elements = document.querySelectorAll('[class*="loading"], [class*="spinner"]').length;
                
                return data;
            }''')
            
            return dom_info
            
        except Exception as e:
            print(f"WARNING: DOM info extraction failed: {e}")
            return {"error": str(e)}
    
    def generate_report(self) -> Dict[str, Any]:
        """검사 결과 리포트 생성"""
        
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "servers_checked": len(self.servers),
            "successful_checks": sum(1 for r in self.results if r["success"]),
            "failed_checks": sum(1 for r in self.results if not r["success"]),
            "results": self.results,
            "summary": self.generate_summary()
        }
        
        # JSON 파일로 저장
        report_path = self.check_dir / f"server_check_report_{self.session_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved: {report_path}")
        
        return report
    
    def generate_summary(self) -> Dict[str, Any]:
        """요약 정보 생성"""
        
        successful_results = [r for r in self.results if r["success"]]
        failed_results = [r for r in self.results if not r["success"]]
        
        summary = {
            "total_servers": len(self.servers),
            "successful_servers": len(successful_results),
            "failed_servers": len(failed_results),
            "success_rate": f"{len(successful_results)/len(self.servers)*100:.1f}%" if self.servers else "0%"
        }
        
        # 성공한 서버들의 정보
        if successful_results:
            summary["successful_servers_info"] = []
            for result in successful_results:
                info = {
                    "url": result["server_url"],
                    "title": result.get("page_title", "Unknown"),
                    "load_time": result.get("network_info", {}).get("load_time_seconds", 0),
                    "is_streamlit": result.get("streamlit_info", {}).get("is_streamlit", False)
                }
                summary["successful_servers_info"].append(info)
        
        # 실패한 서버들의 정보
        if failed_results:
            summary["failed_servers_info"] = []
            for result in failed_results:
                info = {
                    "url": result["server_url"],
                    "error": result.get("error", "Unknown error")
                }
                summary["failed_servers_info"].append(info)
        
        return summary

async def main():
    """서버 상태 검사 실행"""
    
    print("Server Status Checker with Playwright")
    print("=" * 50)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright installation required:")
        print("pip install playwright")
        print("playwright install chromium")
        return
    
    # 검사할 서버 목록
    servers_to_check = [
        "http://localhost:8503",
        "http://localhost:8504"
    ]
    
    # 검사 시스템 시작
    checker = ServerStatusChecker(servers_to_check)
    
    print("\nStarting server status check...")
    
    # 검사 실행
    report = await checker.check_all_servers()
    
    if report:
        print("\nDetailed Summary:")
        print(f"- Servers checked: {report['summary']['total_servers']}")
        print(f"- Success rate: {report['summary']['success_rate']}")
        
        if report["summary"]["successful_servers"]:
            print(f"\nSuccessful servers:")
            for server in report["summary"]["successful_servers_info"]:
                print(f"  - {server['url']}: {server['title']} (Load time: {server['load_time']}s)")
        
        if report["summary"]["failed_servers"]:
            print(f"\nFailed servers:")
            for server in report["summary"]["failed_servers_info"]:
                print(f"  - {server['url']}: {server['error']}")
        
        print(f"\nDetailed report: {checker.check_dir}/server_check_report_{checker.session_id}.json")

if __name__ == "__main__":
    asyncio.run(main())