-- 솔로몬드 AI 시스템 PostgreSQL 초기화 스크립트
-- SQLite에서 PostgreSQL로 마이그레이션 지원

-- 데이터베이스 및 사용자 설정 (이미 docker-compose에서 설정됨)
-- CREATE DATABASE solomond_ai;
-- CREATE USER solomond_user WITH PASSWORD 'solomond_pass_dev';
-- GRANT ALL PRIVILEGES ON DATABASE solomond_ai TO solomond_user;

-- 주얼리 용어 테이블 (SQLite 호환)
CREATE TABLE IF NOT EXISTS jewelry_terms (
    id SERIAL PRIMARY KEY,
    term_key VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    korean TEXT,
    english TEXT,
    chinese TEXT,
    japanese TEXT,
    thai TEXT,
    description_ko TEXT,
    description_en TEXT,
    phonetic_ko TEXT,
    phonetic_en TEXT,
    frequency INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_jewelry_category ON jewelry_terms(category);
CREATE INDEX IF NOT EXISTS idx_jewelry_korean ON jewelry_terms(korean);
CREATE INDEX IF NOT EXISTS idx_jewelry_english ON jewelry_terms(english);
CREATE INDEX IF NOT EXISTS idx_jewelry_chinese ON jewelry_terms(chinese);

-- 용어 수정 이력 테이블
CREATE TABLE IF NOT EXISTS term_corrections (
    id SERIAL PRIMARY KEY,
    original_text TEXT NOT NULL,
    corrected_text TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용 통계 테이블
CREATE TABLE IF NOT EXISTS usage_stats (
    id SERIAL PRIMARY KEY,
    term_key VARCHAR(255) NOT NULL,
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (term_key) REFERENCES jewelry_terms (term_key)
);

-- 시스템 로그 테이블
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_level VARCHAR(20) NOT NULL,
    module VARCHAR(100),
    message TEXT NOT NULL,
    extra_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 성능 메트릭 테이블
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    unit VARCHAR(20),
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 기본 데이터 삽입
INSERT INTO jewelry_terms (term_key, category, subcategory, korean, english, chinese, description_ko) VALUES
('diamond', '보석류', 'precious', '다이아몬드', 'diamond', '钻石', '가장 단단한 천연 보석'),
('ruby', '보석류', 'precious', '루비', 'ruby', '红宝石', '빨간색 강옥'),
('sapphire', '보석류', 'precious', '사파이어', 'sapphire', '蓝宝石', '루비 외의 모든 색상 강옥'),
('emerald', '보석류', 'precious', '에메랄드', 'emerald', '祖母绿', '녹색 베릴'),
('pearl', '보석류', 'organic', '진주', 'pearl', '珍珠', '조개에서 생성되는 유기보석'),
('carat', '등급', 'measurement', '캐럿', 'carat', '克拉', '다이아몬드 중량 단위 (0.2g)'),
('cut', '등급', 'quality', '컷', 'cut', '切工', '다이아몬드 연마 품질'),
('color', '등급', 'quality', '컬러', 'color', '颜色', '다이아몬드 색상 등급'),
('clarity', '등급', 'quality', '클래리티', 'clarity', '净度', '다이아몬드 투명도'),
('gia', '등급', 'institute', 'GIA', 'GIA', '美国宝石学院', '미국 보석학회')
ON CONFLICT (term_key) DO NOTHING;

-- 권한 설정
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO solomond_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO solomond_user;

-- 업데이트 트리거 (updated_at 자동 갱신)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_jewelry_terms_updated_at 
    BEFORE UPDATE ON jewelry_terms 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();