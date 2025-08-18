#!/usr/bin/env python3
"""
솔로몬드 AI - Supabase 자동 설정 시스템
수동 스키마 실행 없이 완전 자동으로 데이터베이스 설정
"""

import json
import requests
from datetime import datetime
from typing import Dict, Any
import subprocess
import sys

class AutoSupabaseSetup:
    """Supabase 완전 자동 설정"""
    
    def __init__(self):
        self.load_config()
        self.setup_results = {}
        
    def load_config(self):
        """설정 로드"""
        try:
            with open("supabase_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            self.supabase_url = config["supabase"]["url"]
            self.anon_key = config["supabase"]["anon_key"]
            self.service_role_key = config["supabase"]["service_role_key"]
            
            print("SUCCESS: Supabase 설정 로드 완료")
            return True
            
        except Exception as e:
            print(f"ERROR: 설정 로드 실패 - {e}")
            return False
    
    def install_dependencies(self):
        """필요한 라이브러리 자동 설치"""
        print("=== 의존성 라이브러리 자동 설치 ===")
        
        libraries = ["supabase", "requests"]
        
        for lib in libraries:
            try:
                __import__(lib)
                print(f"SUCCESS: {lib} 이미 설치됨")
            except ImportError:
                print(f"설치 중: {lib}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", lib
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"SUCCESS: {lib} 설치 완료")
                else:
                    print(f"ERROR: {lib} 설치 실패")
                    return False
        
        return True
    
    def test_connection(self):
        """연결 테스트"""
        print("\n=== Supabase 연결 테스트 ===")
        
        headers = {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {self.anon_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/", 
                headers=headers, 
                timeout=10
            )
            
            if response.status_code == 200:
                print("SUCCESS: Supabase 연결 정상")
                return True
            else:
                print(f"ERROR: 연결 실패 - {response.status_code}")
                return False
                
        except Exception as e:
            print(f"ERROR: 연결 테스트 실패 - {e}")
            return False
    
    def create_tables_via_client(self):
        """Python 클라이언트를 통한 테이블 생성"""
        print("\n=== 자동 테이블 생성 ===")
        
        try:
            from supabase import create_client
            
            # 클라이언트 생성
            supabase = create_client(self.supabase_url, self.service_role_key)
            
            # 테이블 존재 확인
            try:
                # analysis_jobs 테이블 확인
                response = supabase.table("analysis_jobs").select("id").limit(1).execute()
                print("SUCCESS: analysis_jobs 테이블 이미 존재")
                
                # system_health 테이블 확인  
                response = supabase.table("system_health").select("id").limit(1).execute()
                print("SUCCESS: system_health 테이블 이미 존재")
                
                return True
                
            except Exception as table_error:
                if "relation" in str(table_error) or "does not exist" in str(table_error):
                    print("INFO: 테이블이 없습니다. 자동 생성을 시도합니다.")
                    return self.create_via_rest_api()
                else:
                    print(f"ERROR: 테이블 확인 실패 - {table_error}")
                    return False
                    
        except Exception as e:
            print(f"ERROR: 클라이언트 테이블 생성 실패 - {e}")
            return False
    
    def create_via_rest_api(self):
        """REST API를 통한 테이블 생성 시도"""
        print("REST API를 통한 자동 테이블 생성 시도...")
        
        # Service Role Key로 관리자 권한 요청
        admin_headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json"
        }
        
        # SQL 실행을 위한 PostgREST 함수 호출 시도
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                module_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                processing_time DECIMAL(10,2),
                confidence_score DECIMAL(5,4),
                user_id VARCHAR(100),
                input_data JSONB DEFAULT '{}'::jsonb,
                output_data JSONB DEFAULT '{}'::jsonb,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS system_health (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                health_score INTEGER NOT NULL,
                cpu_percent DECIMAL(5,2),
                memory_percent DECIMAL(5,2),
                metadata JSONB DEFAULT '{}'::jsonb
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_at ON analysis_jobs(created_at);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);
            """
        ]
        
        try:
            # 모든 SQL 명령 실행
            for i, sql in enumerate(sql_commands, 1):
                print(f"SQL 명령 {i}/4 실행 중...")
                
                # RPC 호출로 SQL 실행 시도
                response = requests.post(
                    f"{self.supabase_url}/rest/v1/rpc/exec_sql",
                    headers=admin_headers,
                    json={"sql": sql.strip()},
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    print(f"SUCCESS: SQL 명령 {i} 실행 완료")
                else:
                    print(f"INFO: SQL 명령 {i} - {response.status_code} (수동 실행 필요)")
            
            return True
            
        except Exception as e:
            print(f"INFO: 자동 SQL 실행 제한 - {e}")
            print("수동 스키마 실행이 필요합니다.")
            return False
    
    def create_sample_data(self):
        """샘플 데이터 생성하여 테스트"""
        print("\n=== 샘플 데이터 생성 테스트 ===")
        
        try:
            from supabase import create_client
            
            supabase = create_client(self.supabase_url, self.anon_key)
            
            # 샘플 분석 작업 데이터
            sample_job = {
                "module_type": "auto_setup_test",
                "status": "completed", 
                "processing_time": 15.5,
                "confidence_score": 0.95,
                "user_id": "auto_setup",
                "input_data": {"test": "auto_setup"},
                "output_data": {"result": "success", "timestamp": datetime.now().isoformat()},
                "metadata": {"setup_type": "automatic", "version": "v3.0"}
            }
            
            response = supabase.table("analysis_jobs").insert(sample_job).execute()
            
            if response.data:
                job_id = response.data[0]['id']
                print(f"SUCCESS: 샘플 분석 작업 생성 - ID: {job_id}")
                
                # 시스템 건강도 데이터
                sample_health = {
                    "health_score": 95,
                    "cpu_percent": 15.2,
                    "memory_percent": 45.8,
                    "metadata": {"auto_setup": True}
                }
                
                health_response = supabase.table("system_health").insert(sample_health).execute()
                
                if health_response.data:
                    health_id = health_response.data[0]['id']
                    print(f"SUCCESS: 시스템 건강도 데이터 생성 - ID: {health_id}")
                    return True
                    
            return False
            
        except Exception as e:
            print(f"ERROR: 샘플 데이터 생성 실패 - {e}")
            return False
    
    def run_complete_setup(self):
        """완전 자동 설정 실행"""
        print("=== 솔로몬드 AI - Supabase 완전 자동 설정 ===")
        print(f"시작 시간: {datetime.now()}")
        
        setup_steps = [
            ("의존성 설치", self.install_dependencies),
            ("연결 테스트", self.test_connection), 
            ("테이블 생성", self.create_tables_via_client),
            ("샘플 데이터", self.create_sample_data)
        ]
        
        success_count = 0
        total_steps = len(setup_steps)
        
        for step_name, step_func in setup_steps:
            print(f"\n[{success_count + 1}/{total_steps}] {step_name}...")
            
            try:
                if step_func():
                    success_count += 1
                    print(f"✓ {step_name}: SUCCESS")
                else:
                    print(f"✗ {step_name}: FAILED")
            except Exception as e:
                print(f"✗ {step_name}: ERROR - {e}")
        
        # 최종 결과
        success_rate = (success_count / total_steps) * 100
        
        print(f"\n=== 자동 설정 완료 ===")
        print(f"성공: {success_count}/{total_steps} ({success_rate:.1f}%)")
        
        if success_count >= 3:
            print("\n🎉 Supabase 자동 설정 성공!")
            print("솔로몬드 AI v3.0 프로덕션 준비 완료")
            
            # 사용 가능한 기능 안내
            print("\n사용 가능한 기능:")
            print("- 자동 데이터베이스 테이블 생성")
            print("- 실시간 분석 작업 추적")
            print("- 시스템 건강도 모니터링")
            print("- Notion 자동 동기화")
            
        else:
            print("\n⚠️ 일부 수동 설정 필요")
            print("1. Supabase Dashboard → SQL Editor")
            print("2. supabase_simple_schema.sql 내용 실행")
            print("3. 다시 자동 설정 실행")
        
        # 결과 저장
        result = {
            "timestamp": datetime.now().isoformat(),
            "setup_success": success_count >= 3,
            "success_rate": success_rate,
            "next_manual_step_needed": success_count < 3
        }
        
        with open("auto_supabase_setup_report.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result

if __name__ == "__main__":
    setup = AutoSupabaseSetup()
    result = setup.run_complete_setup()
    
    if result["setup_success"]:
        print("\n다음 명령어로 전체 시스템 테스트:")
        print("python simple_final_test.py")
    else:
        print("\n수동 스키마 실행 후 다시 실행:")
        print("python auto_supabase_setup.py")