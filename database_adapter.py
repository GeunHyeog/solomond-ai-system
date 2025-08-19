#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 데이터베이스 어댑터
SQLite와 Supabase 간 원활한 전환을 위한 통합 인터페이스
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

# Supabase 클라이언트 (선택적 임포트)
try:
    from supabase_config import SupabaseManager
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[WARNING] Supabase 모듈을 찾을 수 없습니다. SQLite 모드로 동작합니다.")

class DatabaseInterface(ABC):
    """데이터베이스 인터페이스"""
    
    @abstractmethod
    def is_connected(self) -> bool:
        pass
    
    @abstractmethod
    def create_fragments_table(self) -> bool:
        pass
    
    @abstractmethod
    def insert_fragment(self, fragment_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def insert_fragments_batch(self, fragments: List[Dict[str, Any]]) -> bool:
        pass
    
    @abstractmethod
    def get_fragments(self, conference_name: str = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_fragment_count(self, conference_name: str = None) -> int:
        pass
    
    @abstractmethod
    def delete_fragments(self, conference_name: str) -> bool:
        pass

class SQLiteAdapter(DatabaseInterface):
    """SQLite 데이터베이스 어댑터"""
    
    def __init__(self, conference_name: str = "default"):
        self.conference_name = conference_name
        self.db_path = f"conference_analysis_{conference_name}.db"
    
    def is_connected(self) -> bool:
        return os.path.exists(self.db_path)
    
    def create_fragments_table(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fragments (
                    fragment_id TEXT PRIMARY KEY,
                    file_source TEXT,
                    file_type TEXT,
                    timestamp TEXT,
                    speaker TEXT,
                    content TEXT,
                    confidence REAL,
                    keywords TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] SQLite 테이블 생성 실패: {e}")
            return False
    
    def insert_fragment(self, fragment_data: Dict[str, Any]) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # keywords를 JSON 문자열로 변환
            if isinstance(fragment_data.get('keywords'), list):
                fragment_data['keywords'] = json.dumps(fragment_data['keywords'])
            
            cursor.execute('''
                INSERT OR REPLACE INTO fragments 
                (fragment_id, file_source, file_type, timestamp, speaker, content, confidence, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fragment_data.get('fragment_id'),
                fragment_data.get('file_source'),
                fragment_data.get('file_type'),
                fragment_data.get('timestamp'),
                fragment_data.get('speaker'),
                fragment_data.get('content'),
                fragment_data.get('confidence'),
                fragment_data.get('keywords')
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ERROR] SQLite 조각 삽입 실패: {e}")
            return False
    
    def insert_fragments_batch(self, fragments: List[Dict[str, Any]]) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for fragment in fragments:
                if isinstance(fragment.get('keywords'), list):
                    fragment['keywords'] = json.dumps(fragment['keywords'])
                
                cursor.execute('''
                    INSERT OR REPLACE INTO fragments 
                    (fragment_id, file_source, file_type, timestamp, speaker, content, confidence, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fragment.get('fragment_id'),
                    fragment.get('file_source'),
                    fragment.get('file_type'),
                    fragment.get('timestamp'),
                    fragment.get('speaker'),
                    fragment.get('content'),
                    fragment.get('confidence'),
                    fragment.get('keywords')
                ))
            
            conn.commit()
            conn.close()
            print(f"[SUCCESS] SQLite {len(fragments)}개 조각 배치 삽입 완료")
            return True
        except Exception as e:
            print(f"[ERROR] SQLite 배치 삽입 실패: {e}")
            return False
    
    def get_fragments(self, conference_name: str = None) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM fragments ORDER BY created_at')
            rows = cursor.fetchall()
            
            fragments = []
            for row in rows:
                fragment = {
                    'fragment_id': row[0],
                    'file_source': row[1],
                    'file_type': row[2],
                    'timestamp': row[3],
                    'speaker': row[4],
                    'content': row[5],
                    'confidence': row[6],
                    'keywords': json.loads(row[7]) if row[7] else []
                }
                fragments.append(fragment)
            
            conn.close()
            return fragments
        except Exception as e:
            print(f"[ERROR] SQLite 조각 조회 실패: {e}")
            return []
    
    def get_fragment_count(self, conference_name: str = None) -> int:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM fragments')
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
        except Exception as e:
            print(f"[ERROR] SQLite 조각 개수 조회 실패: {e}")
            return 0
    
    def delete_fragments(self, conference_name: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM fragments WHERE fragment_id LIKE ?', (f'%{conference_name}%',))
            
            conn.commit()
            conn.close()
            print(f"[SUCCESS] SQLite {conference_name} 조각 삭제 완료")
            return True
        except Exception as e:
            print(f"[ERROR] SQLite 조각 삭제 실패: {e}")
            return False

class SupabaseAdapter(DatabaseInterface):
    """Supabase 데이터베이스 어댑터"""
    
    def __init__(self, conference_name: str = "default"):
        self.conference_name = conference_name
        if SUPABASE_AVAILABLE:
            self.manager = SupabaseManager()
        else:
            self.manager = None
    
    def is_connected(self) -> bool:
        return self.manager and self.manager.is_connected()
    
    def create_fragments_table(self) -> bool:
        if not self.manager:
            return False
        return self.manager.create_fragments_table()
    
    def insert_fragment(self, fragment_data: Dict[str, Any]) -> bool:
        if not self.manager:
            return False
        return self.manager.insert_fragment(fragment_data)
    
    def insert_fragments_batch(self, fragments: List[Dict[str, Any]]) -> bool:
        if not self.manager:
            return False
        return self.manager.insert_fragments_batch(fragments)
    
    def get_fragments(self, conference_name: str = None) -> List[Dict[str, Any]]:
        if not self.manager:
            return []
        return self.manager.get_fragments(conference_name or self.conference_name)
    
    def get_fragment_count(self, conference_name: str = None) -> int:
        if not self.manager:
            return 0
        return self.manager.get_fragment_count(conference_name or self.conference_name)
    
    def delete_fragments(self, conference_name: str) -> bool:
        if not self.manager:
            return False
        return self.manager.delete_fragments(conference_name)

class DatabaseFactory:
    """데이터베이스 팩토리"""
    
    @staticmethod
    def create_database(db_type: str = "auto", conference_name: str = "default") -> DatabaseInterface:
        """
        데이터베이스 인스턴스 생성
        
        Args:
            db_type: "auto", "sqlite", "supabase"
            conference_name: 컨퍼런스 이름
        """
        
        if db_type == "auto":
            # 자동 선택: Supabase 환경변수가 있으면 Supabase, 아니면 SQLite
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if supabase_url and supabase_key and SUPABASE_AVAILABLE:
                print("[INFO] Supabase 모드로 초기화")
                return SupabaseAdapter(conference_name)
            else:
                print("[INFO] SQLite 모드로 초기화")
                return SQLiteAdapter(conference_name)
        
        elif db_type == "supabase":
            if not SUPABASE_AVAILABLE:
                raise Exception("Supabase 모듈을 사용할 수 없습니다. SQLite를 사용하세요.")
            return SupabaseAdapter(conference_name)
        
        elif db_type == "sqlite":
            return SQLiteAdapter(conference_name)
        
        else:
            raise ValueError(f"지원하지 않는 데이터베이스 타입: {db_type}")

def migrate_sqlite_to_supabase(conference_name: str = "my_conference"):
    """SQLite 데이터를 Supabase로 마이그레이션"""
    print(f"SQLite -> Supabase 마이그레이션 시작: {conference_name}")
    
    # SQLite에서 데이터 읽기
    sqlite_db = SQLiteAdapter(conference_name)
    if not sqlite_db.is_connected():
        print("[ERROR] SQLite 데이터베이스를 찾을 수 없습니다.")
        return False
    
    fragments = sqlite_db.get_fragments()
    if not fragments:
        print("[WARNING] 마이그레이션할 데이터가 없습니다.")
        return True
    
    print(f"[INFO] {len(fragments)}개 조각 발견")
    
    # Supabase로 데이터 이동
    supabase_db = SupabaseAdapter(conference_name)
    if not supabase_db.is_connected():
        print("[ERROR] Supabase에 연결할 수 없습니다.")
        return False
    
    # 테이블 생성
    supabase_db.create_fragments_table()
    
    # 배치 삽입
    if supabase_db.insert_fragments_batch(fragments):
        print(f"[SUCCESS] {len(fragments)}개 조각 마이그레이션 완료!")
        return True
    else:
        print("[ERROR] Supabase 마이그레이션 실패")
        return False

def test_database_factory():
    """데이터베이스 팩토리 테스트"""
    print("=" * 50)
    print("데이터베이스 팩토리 테스트")
    print("=" * 50)
    
    # 자동 선택 테스트
    db = DatabaseFactory.create_database("auto", "test_conference")
    print(f"선택된 데이터베이스: {type(db).__name__}")
    print(f"연결 상태: {db.is_connected()}")
    
    if db.is_connected():
        # 테이블 생성
        db.create_fragments_table()
        
        # 테스트 데이터
        test_fragment = {
            'fragment_id': 'factory_test_001',
            'file_source': 'factory_test.txt',
            'file_type': 'text',
            'timestamp': datetime.now().isoformat(),
            'speaker': 'Factory Tester',
            'content': 'Database factory test fragment.',
            'confidence': 0.99,
            'keywords': ['factory', 'test', 'database']
        }
        
        # 삽입 테스트
        if db.insert_fragment(test_fragment):
            print("[SUCCESS] 테스트 조각 삽입 성공")
            
            # 조회 테스트
            fragments = db.get_fragments()
            count = db.get_fragment_count()
            print(f"[SUCCESS] 조회 성공: {count}개 조각")
            
            # 정리
            db.delete_fragments('factory_test')
            print("[SUCCESS] 테스트 데이터 정리 완료")
        else:
            print("[ERROR] 테스트 조각 삽입 실패")

if __name__ == "__main__":
    test_database_factory()