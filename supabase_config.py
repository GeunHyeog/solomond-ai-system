#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Supabase 데이터베이스 설정 및 연동
SOLOMOND AI 홀리스틱 컨퍼런스 분석 시스템
"""

import os
from supabase import create_client, Client
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class SupabaseManager:
    """Supabase 데이터베이스 관리자"""
    
    def __init__(self):
        # Supabase 설정 (환경변수에서 읽어오기)
        self.supabase_url = os.getenv('SUPABASE_URL', '')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY', '')
        
        if not self.supabase_url or not self.supabase_key:
            print("[WARNING] Supabase 환경변수가 설정되지 않았습니다.")
            print("SUPABASE_URL과 SUPABASE_ANON_KEY를 설정해주세요.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                print("[SUCCESS] Supabase 클라이언트 초기화 완료")
            except Exception as e:
                print(f"[ERROR] Supabase 연결 실패: {e}")
                self.client = None
    
    def is_connected(self) -> bool:
        """Supabase 연결 상태 확인"""
        return self.client is not None
    
    def create_fragments_table(self) -> bool:
        """fragments 테이블 생성"""
        if not self.client:
            return False
        
        # Supabase에서는 SQL 함수로 테이블 생성
        try:
            # RPC를 통해 테이블 생성 SQL 실행
            result = self.client.rpc('create_fragments_table_if_not_exists').execute()
            print("[SUCCESS] fragments 테이블 생성/확인 완료")
            return True
        except Exception as e:
            print(f"[ERROR] 테이블 생성 실패: {e}")
            # 직접 insert를 시도해서 테이블 존재 여부 확인
            try:
                # 빈 쿼리로 테이블 존재 확인
                self.client.table('fragments').select('*').limit(1).execute()
                print("[INFO] fragments 테이블이 이미 존재합니다")
                return True
            except:
                print("[WARNING] fragments 테이블이 존재하지 않습니다. Supabase 대시보드에서 수동으로 생성해주세요.")
                return False
    
    def insert_fragment(self, fragment_data: Dict[str, Any]) -> bool:
        """단일 조각 데이터 삽입"""
        if not self.client:
            return False
        
        try:
            # keywords는 JSON 문자열로 변환
            if isinstance(fragment_data.get('keywords'), list):
                fragment_data['keywords'] = json.dumps(fragment_data['keywords'])
            
            result = self.client.table('fragments').insert(fragment_data).execute()
            return True
        except Exception as e:
            print(f"[ERROR] 조각 삽입 실패: {e}")
            return False
    
    def insert_fragments_batch(self, fragments: List[Dict[str, Any]]) -> bool:
        """배치로 조각 데이터 삽입"""
        if not self.client:
            return False
        
        try:
            # keywords 리스트를 JSON 문자열로 변환
            for fragment in fragments:
                if isinstance(fragment.get('keywords'), list):
                    fragment['keywords'] = json.dumps(fragment['keywords'])
            
            result = self.client.table('fragments').insert(fragments).execute()
            print(f"[SUCCESS] {len(fragments)}개 조각 배치 삽입 완료")
            return True
        except Exception as e:
            print(f"[ERROR] 배치 삽입 실패: {e}")
            return False
    
    def get_fragments(self, conference_name: str = None) -> List[Dict[str, Any]]:
        """조각 데이터 조회"""
        if not self.client:
            return []
        
        try:
            query = self.client.table('fragments').select('*')
            
            if conference_name:
                # conference_name으로 필터링 (fragment_id 패턴 기반)
                query = query.ilike('fragment_id', f'%{conference_name}%')
            
            result = query.execute()
            
            fragments = []
            for row in result.data:
                # keywords JSON 문자열을 리스트로 변환
                if isinstance(row.get('keywords'), str):
                    try:
                        row['keywords'] = json.loads(row['keywords'])
                    except:
                        row['keywords'] = []
                fragments.append(row)
            
            return fragments
        except Exception as e:
            print(f"[ERROR] 조각 조회 실패: {e}")
            return []
    
    def get_fragment_count(self, conference_name: str = None) -> int:
        """조각 개수 조회"""
        if not self.client:
            return 0
        
        try:
            query = self.client.table('fragments').select('*', count='exact')
            
            if conference_name:
                query = query.ilike('fragment_id', f'%{conference_name}%')
            
            result = query.execute()
            return result.count or 0
        except Exception as e:
            print(f"[ERROR] 조각 개수 조회 실패: {e}")
            return 0
    
    def delete_fragments(self, conference_name: str) -> bool:
        """특정 컨퍼런스의 조각들 삭제"""
        if not self.client:
            return False
        
        try:
            result = self.client.table('fragments').delete().ilike('fragment_id', f'%{conference_name}%').execute()
            print(f"[SUCCESS] {conference_name} 조각 삭제 완료")
            return True
        except Exception as e:
            print(f"[ERROR] 조각 삭제 실패: {e}")
            return False

def setup_supabase_env():
    """Supabase 환경변수 설정 가이드"""
    print("=" * 60)
    print("🚀 Supabase 설정 가이드")
    print("=" * 60)
    print()
    print("1. Supabase 프로젝트 생성:")
    print("   - https://supabase.com 접속")
    print("   - 새 프로젝트 생성")
    print()
    print("2. 환경변수 설정:")
    print("   - SUPABASE_URL: 프로젝트 URL")
    print("   - SUPABASE_ANON_KEY: anon/public 키")
    print()
    print("3. fragments 테이블 생성 SQL:")
    print("""
    CREATE TABLE fragments (
        fragment_id TEXT PRIMARY KEY,
        file_source TEXT,
        file_type TEXT,
        timestamp TEXT,
        speaker TEXT,
        content TEXT,
        confidence REAL,
        keywords TEXT,
        embedding BYTEA,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    print()
    print("4. RLS (Row Level Security) 설정:")
    print("   - fragments 테이블에서 RLS 비활성화 또는")
    print("   - 적절한 정책 설정")

def test_supabase_connection():
    """Supabase 연결 테스트"""
    manager = SupabaseManager()
    
    if not manager.is_connected():
        setup_supabase_env()
        return False
    
    print("✅ Supabase 연결 성공!")
    
    # 테이블 생성/확인
    manager.create_fragments_table()
    
    # 테스트 데이터 삽입
    test_fragment = {
        'fragment_id': 'test_001',
        'file_source': 'test.txt',
        'file_type': 'text',
        'timestamp': datetime.now().isoformat(),
        'speaker': 'Test Speaker',
        'content': 'This is a test fragment for Supabase connection.',
        'confidence': 0.95,
        'keywords': ['test', 'supabase', 'connection']
    }
    
    if manager.insert_fragment(test_fragment):
        print("✅ 테스트 데이터 삽입 성공!")
        
        # 조회 테스트
        fragments = manager.get_fragments()
        count = manager.get_fragment_count()
        print(f"✅ 조회 테스트 성공! 총 {count}개 조각")
        
        # 테스트 데이터 삭제
        manager.delete_fragments('test')
        print("✅ 테스트 데이터 정리 완료!")
        
        return True
    else:
        print("❌ 테스트 데이터 삽입 실패")
        return False

if __name__ == "__main__":
    test_supabase_connection()