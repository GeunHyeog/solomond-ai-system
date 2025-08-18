#!/usr/bin/env python3
"""
Playwright 시연 캐쳐 시스템
브라우저에서 시연한 내용을 자동으로 캡처하고 분석
"""

import asyncio
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from config import SETTINGS

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("❌ Playwright 설치 필요: pip install playwright")

class DemoCaptureSystem:
    """시연 캐쳐 시스템"""
    
    def __init__(self, streamlit_url: str = "http://f"localhost:{SETTINGS['PORT']}""):
        self.streamlit_url = streamlit_url
        self.capture_dir = Path("demo_captures")
        self.capture_dir.mkdir(exist_ok=True)
        
        # 캐쳐 설정
        self.capture_interval = 3.0  # 3초마다 캐쳐
        self.max_captures = 50       # 최대 50개 캐쳐
        
        # 캐쳐 데이터
        self.captures = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎭 시연 캐쳐 시스템 초기화 완료")
        print(f"📁 캐쳐 저장 위치: {self.capture_dir}")
        print(f"🌐 대상 URL: {self.streamlit_url}")
    
    async def start_capture_session(self, duration_minutes: int = 10):
        """시연 캐쳐 세션 시작"""
        
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright가 설치되지 않았습니다.")
            return None
        
        print(f"🚀 시연 캐쳐 시작 ({duration_minutes}분 동안)")
        
        async with async_playwright() as p:
            # 브라우저 시작
            browser = await p.chromium.launch(headless=False)  # 브라우저 보이게
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            try:
                # Streamlit 앱 열기
                await page.goto(self.streamlit_url)
                await page.wait_for_load_state('networkidle')
                
                print("✅ Streamlit 앱 로드 완료")
                print("💡 이제 브라우저에서 자유롭게 시연하세요!")
                print("📸 3초마다 자동으로 캐쳐됩니다...")
                
                # 캐쳐 루프
                end_time = time.time() + (duration_minutes * 60)
                capture_count = 0
                
                while time.time() < end_time and capture_count < self.max_captures:
                    
                    # 현재 상태 캐쳐
                    capture_data = await self.capture_current_state(page)
                    
                    if capture_data:
                        self.captures.append(capture_data)
                        capture_count += 1
                        
                        print(f"📸 캐쳐 {capture_count}: {capture_data['timestamp']} - {capture_data['page_info']['title']}")
                    
                    # 대기
                    await asyncio.sleep(self.capture_interval)
                
                # 세션 완료
                session_report = await self.generate_session_report()
                
                print(f"✅ 시연 캐쳐 세션 완료!")
                print(f"📊 총 {len(self.captures)}개 캐쳐 수집")
                
                return session_report
                
            except Exception as e:
                print(f"❌ 캐쳐 중 오류: {e}")
                return None
            
            finally:
                await browser.close()
    
    async def capture_current_state(self, page) -> Optional[Dict[str, Any]]:
        """현재 브라우저 상태 캐쳐"""
        
        try:
            timestamp = datetime.now().isoformat()
            
            # 스크린샷 캐쳐
            screenshot_path = self.capture_dir / f"screenshot_{self.session_id}_{len(self.captures):03d}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            
            # 페이지 정보 추출
            page_info = {
                'title': await page.title(),
                'url': page.url,
                'viewport': await page.evaluate('() => ({ width: window.innerWidth, height: window.innerHeight })')
            }
            
            # Streamlit 특화 데이터 추출
            streamlit_data = await self.extract_streamlit_data(page)
            
            # DOM 텍스트 내용 추출
            text_content = await page.evaluate('''() => {
                // 주요 텍스트 내용 추출
                const textElements = [];
                
                // 메트릭 추출
                document.querySelectorAll('[data-testid="metric-container"]').forEach(metric => {
                    const label = metric.querySelector('[data-testid="metric-container"] div')?.textContent;
                    const value = metric.querySelector('[data-testid="metric-container"] div:last-child')?.textContent;
                    if (label && value) {
                        textElements.push(`${label}: ${value}`);
                    }
                });
                
                // 성공/에러 메시지 추출
                document.querySelectorAll('.stSuccess, .stError, .stWarning, .stInfo').forEach(msg => {
                    textElements.push(msg.textContent);
                });
                
                // 업로드된 파일 정보
                document.querySelectorAll('[data-testid="stFileUploader"] span').forEach(file => {
                    if (file.textContent.includes('.')) {
                        textElements.push(`파일: ${file.textContent}`);
                    }
                });
                
                // 프로그레스 바 정보
                document.querySelectorAll('[data-testid="stProgress"] div[role="progressbar"]').forEach(progress => {
                    const value = progress.getAttribute('aria-valuenow');
                    if (value) {
                        textElements.push(`진행률: ${value}%`);
                    }
                });
                
                return textElements;
            }''')
            
            capture_data = {
                'timestamp': timestamp,
                'capture_index': len(self.captures),
                'screenshot_path': str(screenshot_path),
                'page_info': page_info,
                'streamlit_data': streamlit_data,
                'text_content': text_content,
                'session_id': self.session_id
            }
            
            return capture_data
            
        except Exception as e:
            print(f"⚠️ 캐쳐 실패: {e}")
            return None
    
    async def extract_streamlit_data(self, page) -> Dict[str, Any]:
        """Streamlit 특화 데이터 추출"""
        
        try:
            streamlit_info = await page.evaluate('''() => {
                const data = {};
                
                // 현재 활성 탭 감지
                const activeTab = document.querySelector('[data-baseweb="tab"][aria-selected="true"]');
                if (activeTab) {
                    data.current_tab = activeTab.textContent;
                }
                
                // 파일 업로드 상태
                const fileUploaders = document.querySelectorAll('[data-testid="stFileUploader"]');
                data.uploaded_files = Array.from(fileUploaders).map(uploader => {
                    const files = uploader.querySelectorAll('span');
                    return Array.from(files).map(f => f.textContent).filter(t => t.includes('.'));
                }).flat();
                
                // 버튼 상태
                const buttons = document.querySelectorAll('button[kind="primary"], button[kind="secondary"]');
                data.visible_buttons = Array.from(buttons).map(btn => btn.textContent);
                
                // 메트릭 값들
                const metrics = {};
                document.querySelectorAll('[data-testid="metric-container"]').forEach((metric, index) => {
                    const spans = metric.querySelectorAll('div span, div div');
                    if (spans.length >= 2) {
                        const label = spans[0].textContent;
                        const value = spans[spans.length-1].textContent;
                        metrics[label] = value;
                    }
                });
                data.metrics = metrics;
                
                // 진행 상태
                const progressBars = document.querySelectorAll('[data-testid="stProgress"]');
                data.progress_bars = Array.from(progressBars).map(pb => {
                    const progressElement = pb.querySelector('[role="progressbar"]');
                    return progressElement ? progressElement.getAttribute('aria-valuenow') : '0';
                });
                
                return data;
            }''')
            
            return streamlit_info
            
        except Exception as e:
            print(f"⚠️ Streamlit 데이터 추출 실패: {e}")
            return {}
    
    async def generate_session_report(self) -> Dict[str, Any]:
        """세션 리포트 생성"""
        
        if not self.captures:
            return {"error": "캐쳐된 데이터가 없습니다."}
        
        # 세션 분석
        session_analysis = {
            'session_id': self.session_id,
            'total_captures': len(self.captures),
            'duration': 'N/A',
            'start_time': self.captures[0]['timestamp'] if self.captures else None,
            'end_time': self.captures[-1]['timestamp'] if self.captures else None,
        }
        
        # 시간 계산
        if len(self.captures) >= 2:
            start = datetime.fromisoformat(self.captures[0]['timestamp'])
            end = datetime.fromisoformat(self.captures[-1]['timestamp'])
            duration = (end - start).total_seconds()
            session_analysis['duration'] = f"{duration:.1f}초"
        
        # 활동 분석
        activity_analysis = self.analyze_user_activity()
        
        # 업로드된 파일 분석
        file_analysis = self.analyze_uploaded_files()
        
        # 결과 분석
        results_analysis = self.analyze_analysis_results()
        
        # 전체 리포트
        session_report = {
            'session_info': session_analysis,
            'activity_summary': activity_analysis,
            'file_uploads': file_analysis,
            'analysis_results': results_analysis,
            'captures': self.captures,
            'recommendations': self.generate_recommendations()
        }
        
        # JSON 파일로 저장
        report_path = self.capture_dir / f"session_report_{self.session_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(session_report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 세션 리포트 저장: {report_path}")
        
        return session_report
    
    def analyze_user_activity(self) -> Dict[str, Any]:
        """사용자 활동 분석"""
        
        tabs_used = set()
        buttons_clicked = set()
        
        for capture in self.captures:
            streamlit_data = capture.get('streamlit_data', {})
            
            # 사용된 탭
            current_tab = streamlit_data.get('current_tab')
            if current_tab:
                tabs_used.add(current_tab)
            
            # 보인 버튼들
            visible_buttons = streamlit_data.get('visible_buttons', [])
            buttons_clicked.update(visible_buttons)
        
        return {
            'tabs_used': list(tabs_used),
            'buttons_available': list(buttons_clicked),
            'total_interactions': len(self.captures)
        }
    
    def analyze_uploaded_files(self) -> Dict[str, Any]:
        """업로드된 파일 분석"""
        
        all_files = set()
        
        for capture in self.captures:
            streamlit_data = capture.get('streamlit_data', {})
            uploaded_files = streamlit_data.get('uploaded_files', [])
            all_files.update(uploaded_files)
        
        # 파일 타입 분류
        audio_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.flac', '.mp4'])]
        image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])]
        
        return {
            'total_files': len(all_files),
            'audio_files': audio_files,
            'image_files': image_files,
            'all_files': list(all_files)
        }
    
    def analyze_analysis_results(self) -> Dict[str, Any]:
        """분석 결과 분석"""
        
        success_messages = []
        error_messages = []
        metrics_evolution = []
        
        for capture in self.captures:
            text_content = capture.get('text_content', [])
            streamlit_data = capture.get('streamlit_data', {})
            
            # 성공/에러 메시지 수집
            for text in text_content:
                if '✅' in text or '성공' in text:
                    success_messages.append(text)
                elif '❌' in text or '실패' in text or '오류' in text:
                    error_messages.append(text)
            
            # 메트릭 변화 추적
            metrics = streamlit_data.get('metrics', {})
            if metrics:
                metrics_evolution.append({
                    'timestamp': capture['timestamp'],
                    'metrics': metrics
                })
        
        return {
            'success_count': len(success_messages),
            'error_count': len(error_messages),
            'success_messages': list(set(success_messages)),
            'error_messages': list(set(error_messages)),
            'metrics_evolution': metrics_evolution
        }
    
    def generate_recommendations(self) -> List[str]:
        """개선 추천사항 생성"""
        
        recommendations = []
        
        # 활동 분석 기반 추천
        activity = self.analyze_user_activity()
        if len(activity['tabs_used']) >= 3:
            recommendations.append("✅ 다양한 기능을 시연하셨습니다!")
        
        # 파일 업로드 분석 기반 추천
        files = self.analyze_uploaded_files()
        if files['total_files'] > 0:
            recommendations.append(f"✅ {files['total_files']}개 파일로 실제 분석을 시연하셨습니다!")
        
        # 결과 분석 기반 추천
        results = self.analyze_analysis_results()
        if results['success_count'] > results['error_count']:
            recommendations.append("✅ 대부분의 분석이 성공적으로 완료되었습니다!")
        
        if not recommendations:
            recommendations.append("💡 더 많은 기능을 시연해보세요!")
        
        return recommendations

async def main():
    """시연 캐쳐 실행"""
    
    print("🎭 Playwright 시연 캐쳐 시스템")
    print("=" * 50)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright 설치 필요:")
        print("pip install playwright")
        print("playwright install chromium")
        return
    
    # 캐쳐 시스템 시작
    capture_system = DemoCaptureSystem()
    
    print("\n🚀 시연 캐쳐를 시작합니다...")
    print("💡 브라우저가 자동으로 열립니다.")
    print("📸 시연하시면서 자동으로 캐쳐됩니다.")
    print("⏰ 10분 동안 또는 최대 50회 캐쳐됩니다.")
    
    # 캐쳐 세션 시작
    session_report = await capture_system.start_capture_session(duration_minutes=10)
    
    if session_report:
        print("\n📊 시연 캐쳐 완료 요약:")
        print(f"- 총 캐쳐: {session_report['session_info']['total_captures']}개")
        print(f"- 소요 시간: {session_report['session_info']['duration']}")
        print(f"- 사용한 탭: {', '.join(session_report['activity_summary']['tabs_used'])}")
        print(f"- 업로드한 파일: {session_report['file_uploads']['total_files']}개")
        print(f"- 성공한 분석: {session_report['analysis_results']['success_count']}회")
        
        print("\n💡 추천사항:")
        for rec in session_report['recommendations']:
            print(f"  {rec}")
        
        print(f"\n📁 상세 리포트: demo_captures/session_report_{capture_system.session_id}.json")

if __name__ == "__main__":
    asyncio.run(main())