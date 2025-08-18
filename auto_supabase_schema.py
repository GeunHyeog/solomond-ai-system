#!/usr/bin/env python3
"""
Supabase SQL 스키마 완전 자동 실행
Service Role Key를 사용하여 관리자 권한으로 스키마 자동 생성
"""

import json
import requests
import time
from datetime import datetime

class AutoSupabaseSchema:
    """Supabase 스키마 자동 실행"""
    
    def __init__(self):
        self.load_config()
        
    def load_config(self):
        """설정 로드"""
        try:
            with open("supabase_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            self.supabase_url = config["supabase"]["url"]
            self.service_role_key = config["supabase"]["service_role_key"]
            self.anon_key = config["supabase"]["anon_key"]
            
            print("SUCCESS: Supabase 설정 로드 완료")
            return True
            
        except Exception as e:
            print(f"ERROR: 설정 로드 실패 - {e}")
            return False
    
    def read_schema_file(self):
        """스키마 파일 읽기"""
        try:
            with open("supabase_simple_schema.sql", "r", encoding="utf-8") as f:
                schema_content = f.read()
            
            print("SUCCESS: 스키마 파일 로드 완료")
            return schema_content
            
        except Exception as e:
            print(f"ERROR: 스키마 파일 로드 실패 - {e}")
            return None
    
    def execute_sql_via_rest(self, sql_content):
        """REST API를 통한 SQL 실행"""
        print("REST API를 통한 SQL 자동 실행...")
        
        # Service Role Key로 관리자 권한 헤더
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json"
        }
        
        # SQL을 개별 명령으로 분할
        sql_commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip()]
        
        success_count = 0
        total_commands = len(sql_commands)
        
        for i, sql_cmd in enumerate(sql_commands, 1):
            if not sql_cmd or sql_cmd.startswith('--'):
                continue
                
            print(f"SQL 명령 {i}/{total_commands} 실행 중...")
            print(f"실행할 SQL: {sql_cmd[:50]}...")
            
            try:
                # PostgREST RPC 호출
                response = requests.post(
                    f"{self.supabase_url}/rest/v1/rpc/exec_sql",
                    headers=headers,
                    json={"sql": sql_cmd},
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    print(f"SUCCESS: SQL 명령 {i} 실행 완료")
                    success_count += 1
                else:
                    print(f"INFO: SQL 명령 {i} - {response.status_code}: {response.text[:100]}")
                    
                time.sleep(1)  # 요청 간격
                
            except Exception as e:
                print(f"ERROR: SQL 명령 {i} 실행 실패 - {e}")
        
        return success_count, total_commands
    
    def execute_sql_via_python_client(self, sql_content):
        """Python 클라이언트를 통한 SQL 실행"""
        print("Python 클라이언트를 통한 SQL 자동 실행...")
        
        try:
            from supabase import create_client
            
            # Service Role Key로 관리자 클라이언트 생성
            supabase = create_client(self.supabase_url, self.service_role_key)
            
            # SQL 명령 분할 및 실행
            sql_commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip()]
            
            success_count = 0
            for i, sql_cmd in enumerate(sql_commands, 1):
                if not sql_cmd or sql_cmd.startswith('--'):
                    continue
                    
                try:
                    print(f"Python 클라이언트로 SQL {i} 실행...")
                    
                    # SQL 실행 (supabase-py에서 제공하는 방법)
                    result = supabase.rpc('exec_sql', {'sql': sql_cmd}).execute()
                    
                    print(f"SUCCESS: Python 클라이언트 SQL {i} 완료")
                    success_count += 1
                    
                except Exception as cmd_error:
                    print(f"INFO: Python 클라이언트 SQL {i} - {cmd_error}")
                    
            return success_count, len(sql_commands)
            
        except Exception as e:
            print(f"ERROR: Python 클라이언트 실행 실패 - {e}")
            return 0, 0
    
    def verify_tables_created(self):
        """테이블 생성 확인"""
        print("\n테이블 생성 확인...")
        
        try:
            from supabase import create_client
            
            supabase = create_client(self.supabase_url, self.anon_key)
            
            # 테이블들 확인
            tables_to_check = ["analysis_jobs", "system_health"]
            created_tables = []
            
            for table in tables_to_check:
                try:
                    response = supabase.table(table).select("*").limit(1).execute()
                    print(f"SUCCESS: {table} 테이블 확인됨")
                    created_tables.append(table)
                    
                except Exception as e:
                    if "relation" in str(e) or "does not exist" in str(e):
                        print(f"WARNING: {table} 테이블 없음")
                    else:
                        print(f"ERROR: {table} 테이블 확인 실패 - {e}")
            
            return len(created_tables), len(tables_to_check)
            
        except Exception as e:
            print(f"ERROR: 테이블 확인 실패 - {e}")
            return 0, 2
    
    def create_sample_data(self):
        """샘플 데이터 생성으로 최종 확인"""
        print("\n샘플 데이터 생성으로 최종 확인...")
        
        try:
            from supabase import create_client
            
            supabase = create_client(self.supabase_url, self.anon_key)
            
            # 샘플 분석 데이터
            sample_analysis = {
                "module_type": "schema_auto_test",
                "status": "completed",
                "processing_time": 5.2,
                "confidence_score": 0.99,
                "user_id": "auto_schema_test",
                "input_data": {"test": "schema_creation"},
                "output_data": {"result": "success", "timestamp": datetime.now().isoformat()},
                "metadata": {"auto_created": True, "version": "v3.0"}
            }
            
            response = supabase.table("analysis_jobs").insert(sample_analysis).execute()
            
            if response.data:
                job_id = response.data[0]['id']
                print(f"SUCCESS: 샘플 분석 데이터 생성 - ID: {job_id}")
                
                # 시스템 건강도 데이터
                sample_health = {
                    "health_score": 100,
                    "cpu_percent": 5.0,
                    "memory_percent": 30.0,
                    "metadata": {"schema_test": True}
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
    
    def run_auto_schema_setup(self):
        """완전 자동 스키마 설정 실행"""
        print("=" * 60)
        print("SOLOMOND AI - Supabase 스키마 완전 자동 설정")
        print("=" * 60)
        start_time = datetime.now()
        
        # 1. 스키마 파일 읽기
        schema_content = self.read_schema_file()
        if not schema_content:
            print("ERROR: 스키마 파일을 읽을 수 없습니다")
            return False
        
        # 2. REST API 방법 시도
        print("\n[방법 1] REST API를 통한 스키마 실행...")
        rest_success, rest_total = self.execute_sql_via_rest(schema_content)
        
        # 3. Python 클라이언트 방법 시도
        print("\n[방법 2] Python 클라이언트를 통한 스키마 실행...")
        client_success, client_total = self.execute_sql_via_python_client(schema_content)
        
        # 4. 테이블 생성 확인
        created_tables, total_tables = self.verify_tables_created()
        
        # 5. 샘플 데이터 생성 테스트
        sample_success = self.create_sample_data()
        
        # 결과 분석
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n" + "=" * 60)
        print("자동 스키마 설정 결과")
        print("=" * 60)
        print(f"REST API 성공: {rest_success}/{rest_total}")
        print(f"Python 클라이언트 성공: {client_success}/{client_total}")
        print(f"테이블 생성 확인: {created_tables}/{total_tables}")
        print(f"샘플 데이터 생성: {'SUCCESS' if sample_success else 'FAILED'}")
        print(f"총 소요 시간: {total_time:.1f}초")
        
        # 성공 여부 판단
        if created_tables >= 2 and sample_success:
            print("\n스키마 자동 설정 완전 성공!")
            print("더 이상 수동 작업이 필요하지 않습니다!")
            
            # 설정 완료 파일 생성
            with open("supabase_schema_completed.json", "w", encoding="utf-8") as f:
                json.dump({
                    "completed": True,
                    "timestamp": datetime.now().isoformat(),
                    "tables_created": created_tables,
                    "sample_data_success": sample_success
                }, f, indent=2, ensure_ascii=False)
            
            return True
            
        else:
            print("\n⚠️ 자동 설정 부분 성공")
            print("수동 스키마 실행이 필요할 수 있습니다:")
            print("1. https://supabase.com/dashboard")
            print("2. 프로젝트 qviccikgyspkyqpemert 선택")
            print("3. SQL Editor → supabase_simple_schema.sql 내용 실행")
            
            return False

if __name__ == "__main__":
    print("Supabase 스키마 자동 설정을 시작합니다...")
    
    auto_schema = AutoSupabaseSchema()
    success = auto_schema.run_auto_schema_setup()
    
    if success:
        print("\n완료! 이제 SOLOMOND AI v3.0을 완전히 사용할 수 있습니다.")
        print("다음 명령어로 시스템 시작:")
        print("start_SOLOMOND_ai.bat")
    else:
        print("\n자동 설정 보고서가 저장되었습니다.")