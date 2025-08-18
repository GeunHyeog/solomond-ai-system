@echo off
echo ===================================
echo 통합 개발 툴킷 환경변수 설정
echo ===================================

REM GitHub 토큰 (이미 하드코딩됨, 선택적)
set GITHUB_TOKEN=%GITHUB_ACCESS_TOKEN%

REM Supabase 설정 (MCP 서버 설정에서 가져옴)
set SUPABASE_URL=https://qviccikgyspkyqpemert.supabase.co
set SUPABASE_ACCESS_TOKEN=%SUPABASE_ACCESS_TOKEN%
REM set SUPABASE_ANON_KEY=your_anon_key_here

REM Notion API 키 (이미 설정된 키)
set NOTION_API_KEY=${NOTION_API_KEY}

REM Perplexity API 키 (이미 MCP로 사용 중)
set PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}

echo 환경변수 설정 완료!
echo.
echo 사용법:
echo 1. setup_environment.bat 실행
echo 2. python toolkit_usage_examples.py 실행
echo.
pause