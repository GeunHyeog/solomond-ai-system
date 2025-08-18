#!/usr/bin/env python3
"""
Supabase 연결 설정 및 데이터베이스 스키마 구축
솔로몬드 AI 시스템을 위한 완전한 DB 구조 설계
"""

import os
import json
from datetime import datetime
from pathlib import Path

class SupabaseSetup:
    """Supabase 설정 및 스키마 관리"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "supabase_config.json"
        
    def create_config_template(self):
        """Supabase 설정 템플릿 생성"""
        config_template = {
            "supabase": {
                "url": "YOUR_SUPABASE_URL_HERE",
                "anon_key": "YOUR_SUPABASE_ANON_KEY_HERE", 
                "service_role_key": "YOUR_SUPABASE_SERVICE_ROLE_KEY_HERE",
                "db_password": "YOUR_DATABASE_PASSWORD_HERE"
            },
            "database": {
                "host": "db.YOUR_PROJECT_REF.supabase.co",
                "port": 5432,
                "database": "postgres",
                "user": "postgres"
            },
            "created_at": datetime.now().isoformat(),
            "status": "template"
        }
        
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_template, f, indent=2, ensure_ascii=False)
        
        print(f"SUCCESS: Supabase 설정 템플릿 생성됨: {self.config_file}")
        return config_template
    
    def generate_database_schema(self):
        """솔로몬드 AI를 위한 데이터베이스 스키마 생성"""
        schema_sql = """
-- 솔로몬드 AI 시스템 데이터베이스 스키마
-- 생성일: {timestamp}

-- 1. 사용자 및 세션 관리
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    user_type VARCHAR(20) DEFAULT 'standard',
    metadata JSONB DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{{}}'::jsonb
);

-- 2. 분석 작업 관리
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- 'conference', 'gemstone', 'crawler', '3d_cad'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    input_data JSONB NOT NULL,
    output_data JSONB DEFAULT '{{}}'::jsonb,
    file_paths JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    processing_time_seconds INTEGER,
    metadata JSONB DEFAULT '{{}}'::jsonb
);

-- 3. 파일 관리
CREATE TABLE IF NOT EXISTS uploaded_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    analysis_job_id UUID REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    original_filename VARCHAR(255) NOT NULL,
    stored_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    file_type VARCHAR(100),
    mime_type VARCHAR(100),
    upload_status VARCHAR(20) DEFAULT 'uploaded',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{{}}'::jsonb
);

-- 4. 분석 결과 저장
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    analysis_job_id UUID REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    result_type VARCHAR(50) NOT NULL, -- 'stt', 'ocr', 'summary', 'translation', etc.
    result_data JSONB NOT NULL,
    confidence_score DECIMAL(5,4),
    processing_engine VARCHAR(50), -- 'whisper', 'easyocr', 'ollama', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{{}}'::jsonb
);

-- 5. 시스템 모니터링
CREATE TABLE IF NOT EXISTS system_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    health_score INTEGER NOT NULL,
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    process_memory_mb DECIMAL(10,2),
    active_modules INTEGER,
    total_analyses_count INTEGER,
    recommendations JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{{}}'::jsonb
);

-- 6. 모듈별 통계
CREATE TABLE IF NOT EXISTS module_statistics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    module_name VARCHAR(50) NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_processing_time_seconds DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(module_name, date)
);

-- 7. Notion 연동 로그
CREATE TABLE IF NOT EXISTS notion_sync_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    analysis_job_id UUID REFERENCES analysis_jobs(id),
    notion_page_id VARCHAR(100),
    sync_type VARCHAR(30), -- 'create', 'update', 'delete'
    sync_status VARCHAR(20) DEFAULT 'pending',
    sync_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    synced_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB DEFAULT '{{}}'::jsonb
);

-- 8. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON analysis_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_at ON analysis_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_analysis_job_id ON uploaded_files(analysis_job_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_job_id ON analysis_results(analysis_job_id);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);
CREATE INDEX IF NOT EXISTS idx_module_statistics_date ON module_statistics(date);

-- 9. 함수 생성
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 10. 트리거 생성
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_module_statistics_updated_at 
    BEFORE UPDATE ON module_statistics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 11. RLS (Row Level Security) 설정
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE uploaded_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

-- 12. 기본 정책 생성 (사용자는 자신의 데이터만 접근)
CREATE POLICY IF NOT EXISTS "Users can view own data" ON users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY IF NOT EXISTS "Users can view own analysis_jobs" ON analysis_jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY IF NOT EXISTS "Users can insert own analysis_jobs" ON analysis_jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 13. 초기 데이터 삽입
INSERT INTO users (id, email, name, user_type) VALUES 
    ('00000000-0000-0000-0000-000000000001', 'admin@solomond.ai', 'System Admin', 'admin')
ON CONFLICT (id) DO NOTHING;

-- 14. 뷰 생성
CREATE OR REPLACE VIEW analysis_summary AS
SELECT 
    DATE(created_at) as analysis_date,
    job_type,
    COUNT(*) as total_jobs,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
    AVG(processing_time_seconds) as avg_processing_time
FROM analysis_jobs 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), job_type
ORDER BY analysis_date DESC, job_type;

COMMENT ON TABLE users IS '솔로몬드 AI 사용자 관리';
COMMENT ON TABLE analysis_jobs IS '분석 작업 관리 및 추적';
COMMENT ON TABLE uploaded_files IS '업로드된 파일 메타데이터';
COMMENT ON TABLE analysis_results IS '분석 결과 저장';
COMMENT ON TABLE system_health IS '시스템 건강도 모니터링';
COMMENT ON TABLE module_statistics IS '모듈별 사용 통계';
COMMENT ON TABLE notion_sync_log IS 'Notion 연동 로그';
""".format(timestamp=datetime.now().isoformat())
        
        schema_file = self.project_root / "supabase_schema.sql"
        with open(schema_file, "w", encoding="utf-8") as f:
            f.write(schema_sql)
        
        print(f"SUCCESS: 데이터베이스 스키마 생성됨: {schema_file}")
        return schema_file
    
    def generate_python_client(self):
        """Supabase Python 클라이언트 생성"""
        client_code = '''#!/usr/bin/env python3
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
'''
        
        client_file = self.project_root / "supabase_client.py"
        with open(client_file, "w", encoding="utf-8") as f:
            f.write(client_code)
        
        print(f"SUCCESS: Supabase 클라이언트 생성됨: {client_file}")
        return client_file
    
    def setup_complete_system(self):
        """완전한 Supabase 시스템 설정"""
        print("=== 솔로몬드 AI Supabase 시스템 설정 ===")
        
        # 1. 설정 템플릿 생성
        config = self.create_config_template()
        
        # 2. 데이터베이스 스키마 생성
        schema_file = self.generate_database_schema()
        
        # 3. Python 클라이언트 생성
        client_file = self.generate_python_client()
        
        # 4. 요구사항 파일 업데이트
        requirements = [
            "supabase>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "python-dotenv>=1.0.0"
        ]
        
        req_file = self.project_root / "requirements_supabase.txt"
        with open(req_file, "w") as f:
            f.write("\\n".join(requirements))
        
        print(f"SUCCESS: Supabase 요구사항 파일: {req_file}")
        
        # 5. 설정 가이드 생성
        guide = self.generate_setup_guide()
        
        print("\\n=== 설정 완료 ===")
        print("다음 파일들이 생성되었습니다:")
        print(f"  1. 설정 템플릿: {self.config_file}")
        print(f"  2. DB 스키마: {schema_file}")
        print(f"  3. Python 클라이언트: {client_file}")
        print(f"  4. 요구사항: {req_file}")
        print(f"  5. 설정 가이드: supabase_setup_guide.md")
        
        return {
            "config_file": self.config_file,
            "schema_file": schema_file,
            "client_file": client_file,
            "requirements_file": req_file
        }
    
    def generate_setup_guide(self):
        """설정 가이드 생성"""
        guide_content = """# 🗄️ Supabase 설정 가이드

## 1. Supabase 프로젝트 생성
1. https://supabase.com 방문
2. 새 프로젝트 생성
3. 데이터베이스 비밀번호 설정

## 2. API 키 확인
프로젝트 대시보드에서 다음 정보 수집:
- Project URL
- Anon/Public Key  
- Service Role Key

## 3. 설정 파일 업데이트
`supabase_config.json` 파일을 실제 값으로 업데이트:

```json
{
  "supabase": {
    "url": "https://YOUR_PROJECT_REF.supabase.co",
    "anon_key": "YOUR_ACTUAL_ANON_KEY",
    "service_role_key": "YOUR_ACTUAL_SERVICE_ROLE_KEY",
    "db_password": "YOUR_DATABASE_PASSWORD"
  }
}
```

## 4. 데이터베이스 스키마 실행
Supabase SQL Editor에서 `supabase_schema.sql` 실행

## 5. Python 패키지 설치
```bash
pip install -r requirements_supabase.txt
```

## 6. 연결 테스트
```bash
python supabase_client.py
```

## 7. 솔로몬드 AI 통합
메인 대시보드와 모듈들이 자동으로 Supabase를 사용하도록 업데이트됩니다.

## 보안 주의사항
- API 키를 Git에 커밋하지 마세요
- Service Role Key는 서버사이드에서만 사용
- RLS(Row Level Security) 정책 확인
"""
        
        guide_file = self.project_root / "supabase_setup_guide.md"
        with open(guide_file, "w", encoding="utf-8") as f:
            f.write(guide_content)
        
        print(f"SUCCESS: 설정 가이드 생성됨: {guide_file}")
        return guide_file

if __name__ == "__main__":
    setup = SupabaseSetup()
    setup.setup_complete_system()