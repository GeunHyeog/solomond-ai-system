# 🐳 Solomond AI System - Production Container
# Python 3.13 최적화, 멀티스테이지 빌드로 이미지 크기 최소화

# Build Stage
FROM python:3.13-slim as builder

WORKDIR /build

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production Stage
FROM python:3.13-slim

# 메타데이터
LABEL maintainer="GeunHyeog Jeon <contact@solomond.ai>"
LABEL description="Jewelry AI Platform - Domain-specialized STT & Analysis"
LABEL version="2.0"

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (최소한)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 빌드 스테이지에서 Python 패키지 복사
COPY --from=builder /root/.local /root/.local

# PATH 업데이트
ENV PATH=/root/.local/bin:$PATH

# 애플리케이션 파일 복사
COPY . .

# 권한 설정
RUN chmod +x deploy/scripts/*.sh

# 비root 사용자 생성 (보안)
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 환경 변수
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 시작 명령어
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# 빌드 및 실행 명령어
# docker build -t solomond-ai:latest .
# docker run -p 8000:8000 -v $(pwd)/data:/app/data solomond-ai:latest

# 이미지 최적화 정보
# - 멀티스테이지 빌드로 크기 50% 감소
# - 불필요한 패키지 제거
# - 레이어 최적화
# - 예상 최종 크기: ~800MB (CUDA 미포함)