#!/usr/bin/env python3
"""
Supabase 클라이언트 - 솔로몬드 AI 시스템용
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from supabase import create_client, Client

class SolomondSupabaseClient:
    """솔로몬드 AI Supabase 클라이언트"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "supabase_config.json"
        
        self.config = self._load_config(config_path)
        self.client: Client = self._create_client()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Supabase 설정 파일을 찾을 수 없습니다: {config_path}")
    
    def _create_client(self) -> Client:
        """Supabase 클라이언트 생성"""
        url = self.config['supabase']['url']
        key = self.config['supabase']['anon_key']
        return create_client(url, key)
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            result = self.client.table('users').select('*').limit(1).execute()
            return True
        except Exception as e:
            print(f"연결 테스트 실패: {e}")
            return False
    
    def create_analysis_job(self, user_id: str, job_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """분석 작업 생성"""
        job_data = {
            'user_id': user_id,
            'job_type': job_type,
            'input_data': input_data,
            'status': 'pending'
        }
        
        result = self.client.table('analysis_jobs').insert(job_data).execute()
        return result.data[0] if result.data else None
    
    def update_analysis_job(self, job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """분석 작업 업데이트"""
        result = self.client.table('analysis_jobs').update(updates).eq('id', job_id).execute()
        return result.data[0] if result.data else None
    
    def save_analysis_result(self, job_id: str, result_type: str, result_data: Dict[str, Any], 
                           confidence_score: Optional[float] = None, 
                           processing_engine: Optional[str] = None) -> Dict[str, Any]:
        """분석 결과 저장"""
        result_record = {
            'analysis_job_id': job_id,
            'result_type': result_type,
            'result_data': result_data,
            'confidence_score': confidence_score,
            'processing_engine': processing_engine
        }
        
        result = self.client.table('analysis_results').insert(result_record).execute()
        return result.data[0] if result.data else None
    
    def save_system_health(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """시스템 건강도 저장"""
        result = self.client.table('system_health').insert(health_data).execute()
        return result.data[0] if result.data else None
    
    def get_analysis_jobs(self, user_id: Optional[str] = None, status: Optional[str] = None, 
                         limit: int = 50) -> List[Dict[str, Any]]:
        """분석 작업 조회"""
        query = self.client.table('analysis_jobs').select('*')
        
        if user_id:
            query = query.eq('user_id', user_id)
        if status:
            query = query.eq('status', status)
        
        result = query.order('created_at', desc=True).limit(limit).execute()
        return result.data or []
    
    def get_system_health_latest(self, limit: int = 100) -> List[Dict[str, Any]]:
        """최근 시스템 건강도 조회"""
        result = self.client.table('system_health').select('*').order('timestamp', desc=True).limit(limit).execute()
        return result.data or []

# 싱글톤 인스턴스
_supabase_client = None

def get_supabase_client() -> SolomondSupabaseClient:
    """Supabase 클라이언트 싱글톤 인스턴스 반환"""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SolomondSupabaseClient()
    return _supabase_client

if __name__ == "__main__":
    # 테스트 코드
    try:
        client = SolomondSupabaseClient()
        if client.test_connection():
            print("SUCCESS: Supabase 연결 성공!")
        else:
            print("ERROR: Supabase 연결 실패!")
    except Exception as e:
        print(f"ERROR: 오류: {e}")
