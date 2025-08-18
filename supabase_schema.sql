
-- 솔로몬드 AI 시스템 데이터베이스 스키마
-- 생성일: 2025-07-30T16:11:36.670023

-- 1. 사용자 및 세션 관리
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    user_type VARCHAR(20) DEFAULT 'standard',
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 2. 분석 작업 관리
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- 'conference', 'gemstone', 'crawler', '3d_cad'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    input_data JSONB NOT NULL,
    output_data JSONB DEFAULT '{}'::jsonb,
    file_paths JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    processing_time_seconds INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb
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
    metadata JSONB DEFAULT '{}'::jsonb
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
    metadata JSONB DEFAULT '{}'::jsonb
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
    metadata JSONB DEFAULT '{}'::jsonb
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
    metadata JSONB DEFAULT '{}'::jsonb
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
