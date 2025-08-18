
-- 솔로몬드 AI 간단 스키마 (수동 실행용)
-- Supabase SQL Editor에서 실행하세요

-- 1. 분석 작업 테이블
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

-- 2. 시스템 건강도 테이블
CREATE TABLE IF NOT EXISTS system_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    health_score INTEGER NOT NULL,
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- 3. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_at ON analysis_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);

-- 4. RLS 활성화 (선택사항)
-- ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;
