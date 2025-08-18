#!/usr/bin/env python3
"""
Claude Code 헬퍼 - 한 줄로 모든 요청 처리
사용자가 자연어로 요청하면 자동으로 적절한 도구를 선택해서 실행
"""

from quick_request import q, github, search, browser, remember, files

# 메인 헬퍼 함수들
def help_me(request: str):
    """메인 헬퍼 - 자연어 요청을 자동으로 처리"""
    return q(request)

def h(request: str):
    """더 짧은 버전"""
    return q(request)

# 사용 예시와 가이드
def show_examples():
    """사용 예시 보기"""
    examples = [
        "h('GitHub 저장소 정보')",
        "h('Python tutorial 검색')",
        "h('https://docs.python.org 브라우저로 열어줘')", 
        "h('현재 디렉토리 파일 목록')",
        "h('이 정보를 기억해줘: Claude Code는 편리하다')",
        "github('이슈 목록')",
        "search('FastAPI 튜토리얼')",
        "browser('https://github.com')",
    ]
    
    print("=== Claude Code 자동 헬퍼 사용법 ===")
    print()
    for example in examples:
        print(f"  {example}")
    print()
    print("💡 팁: h() 함수로 자연어 요청을 하면 자동으로 적절한 도구를 선택합니다!")

if __name__ == "__main__":
    show_examples()