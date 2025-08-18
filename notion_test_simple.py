#!/usr/bin/env python3
"""
Notion API 간단 테스트
"""

import requests
import json

# Notion API 설정
import os
NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "NOTION_TOKEN_NOT_SET")
NOTION_VERSION = "2022-06-28"

headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json"
}

def test_notion_simple():
    """Notion API 간단 테스트"""
    print("Notion API 테스트 시작...")
    
    try:
        # 사용자 정보 조회
        response = requests.get(
            "https://api.notion.com/v1/users/me",
            headers=headers,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            user_data = response.json()
            print("SUCCESS: Notion API 연결 성공!")
            print(f"User ID: {user_data.get('id', 'Unknown')}")
            print(f"User Name: {user_data.get('name', 'Unknown')}")
            print(f"User Type: {user_data.get('type', 'Unknown')}")
            
            # 결과 저장
            with open("notion_success_result.json", "w", encoding="utf-8") as f:
                json.dump(user_data, f, indent=2, ensure_ascii=False)
            
            return True
        else:
            error_data = response.json()
            print("ERROR: Notion API 연결 실패!")
            print(f"Error: {error_data}")
            
            with open("notion_error_result.json", "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            return False
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    test_notion_simple()