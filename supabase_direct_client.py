#!/usr/bin/env python3
"""
Supabase Direct Client - MCP 서버 설정 정보를 활용한 직접 연결
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class SupabaseDirectClient:
    """Supabase 직접 클라이언트"""
    
    def __init__(self):
        # MCP 서버 설정 정보 활용
        self.project_ref = "qviccikgyspkyqpemert"
        self.access_token = os.environ.get("SUPABASE_ACCESS_TOKEN", "SUPABASE_TOKEN_NOT_SET")
        self.base_url = f"https://{self.project_ref}.supabase.co"
        self.api_url = f"{self.base_url}/rest/v1"
        
        # API 헤더 설정
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'apikey': self.access_token,
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        
        print(f"[SUPABASE] Direct client initialized - Project: {self.project_ref}")
    
    def test_connection(self):
        """연결 테스트"""
        try:
            # 테이블 목록 조회로 연결 테스트
            response = requests.get(
                f"{self.api_url}/",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print("[OK] Supabase 연결 성공")
                return True
            else:
                print(f"[ERROR] Supabase 연결 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Supabase 연결 오류: {e}")
            return False
    
    def list_tables(self):
        """데이터베이스 테이블 목록 조회"""
        try:
            # PostgreSQL 시스템 테이블에서 사용자 테이블 조회
            response = requests.get(
                f"{self.api_url}/",
                headers=self.headers
            )
            
            if response.status_code == 200:
                # OpenAPI 스키마에서 테이블 정보 추출
                schema = response.json()
                if 'paths' in schema:
                    tables = [path.strip('/') for path in schema['paths'].keys() if path.startswith('/')]
                    print(f"[INFO] 발견된 테이블: {len(tables)}개")
                    return tables
                else:
                    print("[INFO] 테이블 정보를 찾을 수 없습니다")
                    return []
            else:
                print(f"[ERROR] 테이블 목록 조회 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[ERROR] 테이블 목록 조회 오류: {e}")
            return []
    
    def create_development_logs_table(self):
        """개발 로그 테이블 생성 (SQL 실행)"""
        
        # Supabase SQL API를 통한 테이블 생성
        sql_query = """
        CREATE TABLE IF NOT EXISTS development_logs (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(50),
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            action VARCHAR(100),
            details JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            # SQL 실행 (Supabase의 RPC 기능 사용)
            response = requests.post(
                f"{self.base_url}/rest/v1/rpc/exec_sql",
                headers=self.headers,
                json={"query": sql_query}
            )
            
            if response.status_code in [200, 201]:
                print("[OK] development_logs 테이블 생성 완료")
                return True
            else:
                print(f"[INFO] 테이블 생성 시도 - 상태: {response.status_code}")
                print(f"[INFO] 응답: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 테이블 생성 오류: {e}")
            return False
    
    def insert_log(self, session_id: str, action: str, details: Dict):
        """개발 로그 삽입"""
        
        log_data = {
            "session_id": session_id,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/development_logs",
                headers=self.headers,
                json=log_data
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"[OK] 로그 저장 완료 - ID: {result[0].get('id', 'N/A')}")
                return result
            else:
                print(f"[ERROR] 로그 저장 실패: {response.status_code}")
                print(f"[ERROR] 응답: {response.text}")
                return None
                
        except Exception as e:
            print(f"[ERROR] 로그 저장 오류: {e}")
            return None
    
    def get_logs(self, session_id: str = None, limit: int = 10):
        """개발 로그 조회"""
        
        url = f"{self.api_url}/development_logs"
        params = {"order": "created_at.desc", "limit": limit}
        
        if session_id:
            params["session_id"] = f"eq.{session_id}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                logs = response.json()
                print(f"[OK] 로그 조회 완료 - {len(logs)}개")
                return logs
            else:
                print(f"[ERROR] 로그 조회 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[ERROR] 로그 조회 오류: {e}")
            return []

# 사용 예시
if __name__ == "__main__":
    client = SupabaseDirectClient()
    
    # 연결 테스트
    if client.test_connection():
        
        # 테이블 목록 조회
        tables = client.list_tables()
        print(f"테이블들: {tables}")
        
        # 개발 로그 테이블 생성 시도
        client.create_development_logs_table()
        
        # 로그 삽입 테스트
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = client.insert_log(
            session_id=session_id,
            action="supabase_test",
            details={"message": "Supabase 직접 연결 테스트"}
        )
        
        if result:
            # 로그 조회 테스트
            logs = client.get_logs(session_id=session_id)
            print(f"저장된 로그: {logs}")