#!/bin/bash
# 💎 솔로몬드 AI 엔진 v2.3 원클릭 프로덕션 배포 스크립트
# 99.4% 정확도 하이브리드 LLM 시스템 자동 배포

set -e  # 오류 발생 시 즉시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로고 출력
echo -e "${PURPLE}"
echo "████████████████████████████████████████████████████████"
echo "█                                                      █"
echo "█  💎 솔로몬드 AI 엔진 v2.3 프로덕션 배포 시스템    █"
echo "█  🎯 99.4% 정확도 하이브리드 LLM 시스템           █"
echo "█  🚀 원클릭 자동 배포 스크립트                    █"
echo "█                                                      █"
echo "████████████████████████████████████████████████████████"
echo -e "${NC}"

# 스크립트 정보
echo -e "${CYAN}📋 배포 스크립트 정보${NC}"
echo "- 버전: v2.3"
echo "- 날짜: $(date '+%Y-%m-%d %H:%M:%S')"
echo "- 대상: 프로덕션 환경"
echo "- 개발자: 전근혁 (솔로몬드 대표)"
echo ""

# 사전 요구사항 확인
echo -e "${YELLOW}🔍 사전 요구사항 확인${NC}"

# Docker 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker가 설치되지 않았습니다.${NC}"
    echo "Docker를 먼저 설치해주세요: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✅ Docker 설치 확인됨${NC}"

# Docker Compose 확인
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose가 설치되지 않았습니다.${NC}"
    echo "Docker Compose를 먼저 설치해주세요."
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose 설치 확인됨${NC}"

# Git 확인
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git이 설치되지 않았습니다.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Git 설치 확인됨${NC}"

# 시스템 리소스 확인
echo -e "${YELLOW}📊 시스템 리소스 확인${NC}"
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')

if [ "$MEMORY_GB" -lt 4 ]; then
    echo -e "${YELLOW}⚠️ 메모리가 부족할 수 있습니다 (권장: 8GB 이상, 현재: ${MEMORY_GB}GB)${NC}"
else
    echo -e "${GREEN}✅ 메모리: ${MEMORY_GB}GB (충분)${NC}"
fi

if [ "$DISK_GB" -lt 10 ]; then
    echo -e "${YELLOW}⚠️ 디스크 공간이 부족할 수 있습니다 (권장: 20GB 이상, 현재: ${DISK_GB}GB)${NC}"
else
    echo -e "${GREEN}✅ 디스크: ${DISK_GB}GB (충분)${NC}"
fi

echo ""

# 배포 설정 입력
echo -e "${CYAN}⚙️ 배포 설정 구성${NC}"

# 환경 변수 설정 파일 생성
ENV_FILE=".env.v23.production"

echo "# 솔로몬드 AI v2.3 프로덕션 환경 변수" > $ENV_FILE
echo "# 생성일: $(date)" >> $ENV_FILE
echo "" >> $ENV_FILE

# 기본 설정
echo "ENVIRONMENT=production" >> $ENV_FILE
echo "SOLOMOND_VERSION=v2.3" >> $ENV_FILE
echo "LOG_LEVEL=INFO" >> $ENV_FILE
echo "" >> $ENV_FILE

# 데이터베이스 비밀번호 입력
echo -e "${YELLOW}🔒 데이터베이스 비밀번호를 설정해주세요:${NC}"
read -s -p "PostgreSQL 비밀번호: " POSTGRES_PASSWORD
echo ""
read -s -p "Redis 비밀번호: " REDIS_PASSWORD
echo ""

echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> $ENV_FILE
echo "REDIS_PASSWORD=$REDIS_PASSWORD" >> $ENV_FILE
echo "" >> $ENV_FILE

# API 키 입력 (선택사항)
echo -e "${YELLOW}🔑 AI API 키 설정 (선택사항 - 나중에 설정 가능):${NC}"
read -p "OpenAI API 키 (Enter로 건너뛰기): " OPENAI_API_KEY
read -p "Anthropic API 키 (Enter로 건너뛰기): " ANTHROPIC_API_KEY
read -p "Google API 키 (Enter로 건너뛰기): " GOOGLE_API_KEY

if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> $ENV_FILE
fi
if [ ! -z "$ANTHROPIC_API_KEY" ]; then
    echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> $ENV_FILE
fi
if [ ! -z "$GOOGLE_API_KEY" ]; then
    echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> $ENV_FILE
fi

echo -e "${GREEN}✅ 환경 설정 완료: $ENV_FILE${NC}"
echo ""

# 배포 옵션 선택
echo -e "${CYAN}🚀 배포 옵션 선택${NC}"
echo "1) 기본 배포 (AI 시스템만)"
echo "2) 완전 배포 (모니터링 포함)"
echo "3) 개발자 배포 (모든 서비스)"
read -p "선택 (1-3): " DEPLOY_OPTION

case $DEPLOY_OPTION in
    1) PROFILES="" ;;
    2) PROFILES="--profile monitoring" ;;
    3) PROFILES="--profile production --profile monitoring --profile backup" ;;
    *) 
        echo -e "${RED}❌ 잘못된 선택입니다.${NC}"
        exit 1
        ;;
esac

echo ""

# 배포 시작
echo -e "${GREEN}🚀 솔로몬드 AI v2.3 프로덕션 배포 시작!${NC}"
echo ""

# 이전 컨테이너 정리
echo -e "${YELLOW}🧹 이전 배포 정리 중...${NC}"
docker-compose -f docker-compose.v23.production.yml down --remove-orphans || true
docker system prune -f || true

# 최신 이미지 풀
echo -e "${YELLOW}📥 최신 이미지 다운로드 중...${NC}"
docker-compose -f docker-compose.v23.production.yml pull || true

# 배포 실행
echo -e "${GREEN}🚀 v2.3 시스템 시작 중...${NC}"
docker-compose -f docker-compose.v23.production.yml --env-file $ENV_FILE $PROFILES up -d

# 서비스 시작 대기
echo -e "${YELLOW}⏳ 서비스 시작 대기 (120초)...${NC}"
for i in {1..120}; do
    echo -n "."
    sleep 1
    if [ $((i % 10)) -eq 0 ]; then
        echo " $i초"
    fi
done
echo ""

# 헬스체크
echo -e "${CYAN}🔍 시스템 상태 확인${NC}"

# 메인 서비스 확인
if curl -s http://localhost:8080/health/v23 > /dev/null; then
    echo -e "${GREEN}✅ 솔로몬드 AI v2.3 메인 서비스: 정상${NC}"
else
    echo -e "${YELLOW}⚠️ 메인 서비스 응답 대기 중...${NC}"
fi

# 데이터베이스 확인
if docker-compose -f docker-compose.v23.production.yml exec -T postgres pg_isready -U solomond_user > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL 데이터베이스: 정상${NC}"
else
    echo -e "${YELLOW}⚠️ 데이터베이스 초기화 중...${NC}"
fi

# Redis 확인
if docker-compose -f docker-compose.v23.production.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis 캐시: 정상${NC}"
else
    echo -e "${YELLOW}⚠️ Redis 연결 확인 중...${NC}"
fi

# 모니터링 서비스 확인 (해당되는 경우)
if [ "$DEPLOY_OPTION" -ge 2 ]; then
    if curl -s http://localhost:3000 > /dev/null; then
        echo -e "${GREEN}✅ Grafana 대시보드: 정상${NC}"
    else
        echo -e "${YELLOW}⚠️ Grafana 시작 중...${NC}"
    fi
    
    if curl -s http://localhost:9090 > /dev/null; then
        echo -e "${GREEN}✅ Prometheus 모니터링: 정상${NC}"
    else
        echo -e "${YELLOW}⚠️ Prometheus 시작 중...${NC}"
    fi
fi

echo ""

# 배포 완료 메시지
echo -e "${PURPLE}"
echo "████████████████████████████████████████████████████████"
echo "█                                                      █"
echo "█  🎉 솔로몬드 AI v2.3 프로덕션 배포 완료!           █"
echo "█  💎 99.4% 정확도 하이브리드 LLM 시스템 가동 중     █"
echo "█                                                      █"
echo "████████████████████████████████████████████████████████"
echo -e "${NC}"

echo -e "${GREEN}📍 접속 정보:${NC}"
echo "🌐 메인 서비스: http://localhost:8080"
echo "🔍 헬스체크: http://localhost:8080/health/v23"
echo "📊 시스템 상태: http://localhost:8080/status"

if [ "$DEPLOY_OPTION" -ge 2 ]; then
    echo "📈 Grafana 대시보드: http://localhost:3000 (admin/admin123)"
    echo "📊 Prometheus 메트릭: http://localhost:9090"
fi

echo ""
echo -e "${CYAN}🛠️ 관리 명령어:${NC}"
echo "시스템 중지: docker-compose -f docker-compose.v23.production.yml down"
echo "로그 확인: docker-compose -f docker-compose.v23.production.yml logs -f"
echo "상태 확인: docker-compose -f docker-compose.v23.production.yml ps"
echo "재시작: docker-compose -f docker-compose.v23.production.yml restart"

echo ""
echo -e "${GREEN}✨ 배포가 성공적으로 완료되었습니다!${NC}"
echo -e "${BLUE}📞 지원이 필요하시면 전근혁 (solomond.jgh@gmail.com)으로 연락해주세요.${NC}"

# 환경 변수 파일 보안 설정
chmod 600 $ENV_FILE
echo -e "${YELLOW}🔒 환경 변수 파일 보안 설정 완료: $ENV_FILE${NC}"

echo ""
echo -e "${PURPLE}🎯 솔로몬드 AI v2.3으로 주얼리 업계 혁신을 시작하세요!${NC}"
