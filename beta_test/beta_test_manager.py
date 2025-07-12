#!/usr/bin/env python3
"""
솔로몬드 AI 플랫폼 v2.1 베타 테스트 매니저

현장 테스트 및 피드백 수집을 위한 통합 관리 시스템
- 한국보석협회 실무진 베타 테스트
- 홍콩 주얼리쇼 현장 적용 테스트
- 실시간 성능 모니터링 및 피드백 수집
"""

import json
import time
import datetime
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('beta_test_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestSession:
    """베타 테스트 세션 정보"""
    session_id: str
    user_name: str
    organization: str  # "한국보석협회", "솔로몬드", "홍콩쇼" 등
    role: str  # "실무진", "경영진", "바이어", "전문가" 등
    test_scenario: str
    start_time: str
    end_time: Optional[str] = None
    files_processed: int = 0
    success_rate: float = 0.0
    user_satisfaction: Optional[int] = None  # 1-10 점수
    feedback_comments: str = ""
    technical_issues: List[str] = None
    performance_metrics: Dict = None

@dataclass
class FeedbackEntry:
    """피드백 엔트리"""
    feedback_id: str
    session_id: str
    timestamp: str
    category: str  # "기능성", "사용성", "성능", "품질", "개선요청"
    rating: int  # 1-5 별점
    comment: str
    priority: str  # "긴급", "높음", "보통", "낮음"
    status: str = "신규"  # "신규", "검토중", "처리중", "완료"

class BetaTestManager:
    """베타 테스트 매니저"""
    
    def __init__(self, db_path: str = "beta_test_data.db"):
        """초기화"""
        self.db_path = db_path
        self.active_sessions = {}
        self.performance_monitor = PerformanceMonitor()
        self._init_database()
        
        logger.info("🎯 베타 테스트 매니저 시작")
        logger.info("📊 데이터베이스 경로: %s", db_path)
        
    def _init_database(self):
        """데이터베이스 초기화"""
        with self._get_db_connection() as conn:
            # 테스트 세션 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_name TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    role TEXT NOT NULL,
                    test_scenario TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    files_processed INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    user_satisfaction INTEGER,
                    feedback_comments TEXT,
                    technical_issues TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            # 피드백 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT DEFAULT '신규',
                    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
                )
            ''')
            
            # 성능 메트릭 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    processing_time REAL,
                    file_size_mb REAL,
                    accuracy_score REAL,
                    user_wait_time REAL,
                    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
                )
            ''')
            
        logger.info("✅ 데이터베이스 초기화 완료")
    
    @contextmanager
    def _get_db_connection(self):
        """데이터베이스 연결 관리"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def start_test_session(self, user_name: str, organization: str, 
                          role: str, test_scenario: str)