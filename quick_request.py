#!/usr/bin/env python3
"""
빠른 요청 처리기
Claude Code에서 한 줄로 사용할 수 있는 간편한 인터페이스
"""

import asyncio
from smart_mcp_router import execute

def q(request: str):
    """빠른 요청 처리 - 한 글자 함수로 최대한 간편하게"""
    result = execute(request)
    
    print(f"[REQUEST] 요청: {request}")
    print(f"[TOOL] 도구: {result['tool_used']}")
    print(f"[RESULT] 결과:")
    
    # 결과 타입에 따라 다르게 출력
    execution_result = result['execution_result']
    
    if isinstance(execution_result, dict):
        for key, value in execution_result.items():
            print(f"  {key}: {value}")
    elif isinstance(execution_result, list):
        for i, item in enumerate(execution_result[:5], 1):  # 최대 5개만
            if isinstance(item, dict) and 'title' in item:
                print(f"  {i}. {item['title']}")
            else:
                print(f"  {i}. {item}")
    else:
        print(f"  {execution_result}")
    
    return result

# 더 구체적인 함수들
def github(action: str = "repo info"):
    """GitHub 관련 요청"""
    return q(f"GitHub {action}")

def search(query: str):
    """웹 검색"""
    return q(f"검색 {query}")

def browser(url: str = "https://www.google.com"):
    """브라우저 열기"""
    return q(f"브라우저로 {url} 열어줘")

def remember(content: str):
    """정보 기억"""
    return q(f"기억해줘: {content}")

def files(action: str = "list"):
    """파일 작업"""
    return q(f"파일 {action}")

# 사용 예시
if __name__ == "__main__":
    print("=== 빠른 요청 처리기 테스트 ===")
    
    # 간단한 테스트
    q("GitHub 저장소 정보")
    print()
    
    search("Python FastAPI tutorial")
    print()
    
    github("이슈 목록")
    print()
    
    print("[SUCCESS] 테스트 완료!")