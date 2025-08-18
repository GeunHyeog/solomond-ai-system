#!/usr/bin/env python3
"""
통합 개발 툴킷 사용 예시
"""

import asyncio
import os
from integrated_development_toolkit import IntegratedDevelopmentToolkit
from config import SETTINGS

async def example_browser_automation():
    """브라우저 자동화 예시"""
    
    print("🌐 브라우저 자동화 예시")
    
    toolkit = IntegratedDevelopmentToolkit()
    
    # 브라우저 세션 시작 (화면에 표시)
    session = await toolkit.launch_browser_session("http://f"localhost:{SETTINGS['PORT']}"", headless=False)
    
    if session:
        page = session["page"]
        
        # 페이지 내용 캡처
        content = await toolkit.capture_page_content(page)
        print(f"페이지 내용 길이: {len(content)} 문자")
        
        # 브라우저 종료
        await session["browser"].close()

def example_github_integration():
    """GitHub 연동 예시"""
    
    print("🐙 GitHub 연동 예시")
    
    toolkit = IntegratedDevelopmentToolkit()
    
    # 저장소 정보 조회
    repo_info = toolkit.get_repo_info("GeunHyeog", "solomond-ai-system")
    if repo_info:
        print(f"저장소: {repo_info['full_name']}")
        print(f"설명: {repo_info.get('description', 'No description')}")
        print(f"스타 수: {repo_info['stargazers_count']}")
    
    # 이슈 목록 조회
    issues = toolkit.list_issues("GeunHyeog", "solomond-ai-system", "open")
    if issues:
        print(f"열린 이슈 수: {len(issues)}")
        for issue in issues[:3]:  # 최근 3개만 표시
            print(f"  - #{issue['number']}: {issue['title']}")

def example_supabase_integration():
    """Supabase 연동 예시"""
    
    print("🗄️ Supabase 연동 예시")
    
    # 환경 변수 설정 확인
    if not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_ANON_KEY'):
        print("❌ Supabase 환경 변수가 설정되지 않았습니다.")
        print("   SUPABASE_URL과 SUPABASE_ANON_KEY를 설정하세요.")
        return
    
    toolkit = IntegratedDevelopmentToolkit()
    
    if toolkit.supabase_client:
        # 개발 로그 저장 예시
        log_result = toolkit.save_development_log(
            "test_action", 
            {"message": "Supabase 연동 테스트"}
        )
        
        if log_result:
            print("✅ 개발 로그 저장 완료")
        else:
            print("❌ 개발 로그 저장 실패")

def example_web_search():
    """웹 검색 예시"""
    
    print("🔍 웹 검색 예시")
    
    toolkit = IntegratedDevelopmentToolkit()
    
    # 웹 검색 실행
    search_results = toolkit.web_search("Python Streamlit 개발 가이드")
    
    if search_results:
        print(f"검색 결과 {len(search_results)}개:")
        for i, result in enumerate(search_results[:5], 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['href']}")
            print(f"   요약: {result['body'][:100]}...")
            print()
    
    # 웹페이지 내용 가져오기
    webpage = toolkit.fetch_webpage_content("https://docs.streamlit.io/")
    if webpage:
        print(f"웹페이지 제목: {webpage['title']}")
        print(f"내용 길이: {len(webpage['content'])} 문자")

async def example_integrated_workflow():
    """통합 워크플로우 예시"""
    
    print("🚀 통합 워크플로우 예시")
    
    toolkit = IntegratedDevelopmentToolkit()
    
    # 전체 워크플로우 실행
    result = await toolkit.integrated_development_workflow(
        "Streamlit 앱에 새로운 기능 추가"
    )
    
    print("워크플로우 완료 단계:")
    for step in result["steps"]:
        print(f"  ✅ {step}")

async def main():
    """메인 실행 함수"""
    
    print("🛠️ 통합 개발 툴킷 사용 예시")
    print("=" * 50)
    
    # 각 기능별 예시 실행
    try:
        # 1. GitHub 연동 (동기)
        example_github_integration()
        print()
        
        # 2. 웹 검색 (동기)
        example_web_search()
        print()
        
        # 3. Supabase 연동 (동기)
        example_supabase_integration()
        print()
        
        # 4. 브라우저 자동화 (비동기)
        await example_browser_automation()
        print()
        
        # 5. 통합 워크플로우 (비동기)
        await example_integrated_workflow()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    
    print("\n✅ 모든 예시 실행 완료!")

if __name__ == "__main__":
    asyncio.run(main())