#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 자율 스케줄러 - SOLOMOND AI 자율성 강화 시스템
Autonomous Scheduler for SOLOMOND AI Agent System

핵심 기능:
1. 자동 트리거 조건 모니터링
2. 정기 분석 스케줄링
3. 상황 인식 자동 분석
4. 자율적 의사결정
"""

import os
import json
import time
import schedule
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import logging

# 로깅 설정 (Windows 호환성)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutonomousScheduler:
    """SOLOMOND AI 자율 스케줄러"""
    
    def __init__(self):
        """스케줄러 초기화"""
        self.config = self._load_config()
        self.is_running = False
        self.last_analysis_time = None
        self.analysis_history_dir = Path("analysis_history")
        self.analysis_history_dir.mkdir(exist_ok=True)
        
        # n8n 워크플로우 엔드포인트
        self.n8n_base_url = "http://localhost:5678/webhook"
        self.master_trigger_url = f"{self.n8n_base_url}/solomond-ai-trigger"
        
        logger.info("SOLOMOND AI 자율 스케줄러 초기화 완료")
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        default_config = {
            "autonomous_mode": True,
            "monitoring_intervals": {
                "trend_check": 30,  # 30분마다 트렌드 확인
                "system_health": 15,  # 15분마다 시스템 상태 확인
                "data_analysis": 120  # 2시간마다 데이터 분석
            },
            "auto_triggers": {
                "industry_news": True,
                "market_analysis": True,
                "competitor_tracking": False,
                "technology_trends": True
            },
            "analysis_thresholds": {
                "min_interval_hours": 1,  # 최소 1시간 간격
                "max_daily_analyses": 10,  # 하루 최대 10회 분석
                "priority_boost_keywords": ["주얼리", "보석", "AI", "트렌드", "혁신"]
            }
        }
        
        config_file = Path("autonomous_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패, 기본값 사용: {e}")
        
        return default_config
    
    def start_autonomous_mode(self):
        """자율 모드 시작"""
        if self.is_running:
            logger.info("자율 모드가 이미 실행 중입니다")
            return
        
        self.is_running = True
        logger.info("SOLOMOND AI 자율 모드 시작")
        
        # 스케줄 등록
        self._register_schedules()
        
        # 백그라운드 스레드에서 스케줄러 실행
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("자율 스케줄러 활성화 완료")
    
    def _register_schedules(self):
        """자동 스케줄 등록"""
        intervals = self.config["monitoring_intervals"]
        
        # 트렌드 모니터링
        schedule.every(intervals["trend_check"]).minutes.do(self._monitor_trends)
        
        # 시스템 건강도 확인
        schedule.every(intervals["system_health"]).minutes.do(self._check_system_health)
        
        # 정기 데이터 분석
        schedule.every(intervals["data_analysis"]).minutes.do(self._perform_periodic_analysis)
        
        # 매일 아침 9시 일일 보고서
        schedule.every().day.at("09:00").do(self._generate_daily_report)
        
        # 매주 월요일 주간 트렌드 분석
        schedule.every().monday.at("10:00").do(self._perform_weekly_analysis)
        
        logger.info("자동 스케줄 등록 완료")
    
    def _run_scheduler(self):
        """스케줄러 실행 루프"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 스케줄 확인
            except Exception as e:
                logger.error(f"스케줄러 실행 오류: {e}")
                time.sleep(60)
    
    def _monitor_trends(self):
        """트렌드 모니터링"""
        logger.info("트렌드 모니터링 시작")
        
        try:
            # 업계 트렌드 키워드 검색
            trend_keywords = ["jewelry AI", "gemstone technology", "smart jewelry", "주얼리 트렌드"]
            
            for keyword in trend_keywords:
                if self._should_analyze_keyword(keyword):
                    self._trigger_autonomous_analysis({
                        "type": "trend_analysis",
                        "keyword": keyword,
                        "source": "autonomous_trend_monitoring",
                        "priority": "normal",
                        "description": f"자동 트렌드 분석: {keyword}"
                    })
                    
        except Exception as e:
            logger.error(f"트렌드 모니터링 오류: {e}")
    
    def _check_system_health(self):
        """시스템 건강도 확인"""
        logger.info("시스템 건강도 확인")
        
        try:
            health_status = {
                "n8n": self._check_n8n_health(),
                "ollama": self._check_ollama_health(),
                "disk_space": self._check_disk_space(),
                "memory_usage": self._check_memory_usage()
            }
            
            # 문제 발견 시 자동 대응
            if not health_status["n8n"]:
                logger.warning("n8n 서비스 문제 감지")
                self._handle_service_issue("n8n")
            
            if not health_status["ollama"]:
                logger.warning("Ollama 서비스 문제 감지")
                self._handle_service_issue("ollama")
            
            # 건강도 보고서 저장
            self._save_health_report(health_status)
            
        except Exception as e:
            logger.error(f"시스템 건강도 확인 오류: {e}")
    
    def _perform_periodic_analysis(self):
        """정기 데이터 분석"""
        logger.info("정기 데이터 분석 시작")
        
        try:
            # 최근 분석 간격 확인
            if not self._should_perform_analysis():
                logger.info("분석 간격 조건 불만족, 건너뛰기")
                return
            
            # 분석할 데이터 소스 결정
            analysis_targets = self._identify_analysis_targets()
            
            for target in analysis_targets:
                self._trigger_autonomous_analysis({
                    "type": "periodic_analysis", 
                    "target": target,
                    "source": "autonomous_periodic",
                    "priority": "low",
                    "description": f"정기 자동 분석: {target['name']}"
                })
                
        except Exception as e:
            logger.error(f"정기 분석 오류: {e}")
    
    def _generate_daily_report(self):
        """일일 보고서 생성"""
        logger.info("일일 보고서 생성")
        
        try:
            # 어제 분석 결과 수집
            yesterday = datetime.now() - timedelta(days=1)
            daily_summary = self._collect_daily_summary(yesterday)
            
            # 듀얼 브레인 시스템에 보고서 생성 요청
            report_request = {
                "type": "daily_report",
                "date": yesterday.strftime("%Y-%m-%d"),
                "summary": daily_summary,
                "source": "autonomous_daily_report",
                "priority": "normal",
                "description": "자동 생성 일일 보고서"
            }
            
            self._trigger_autonomous_analysis(report_request)
            
        except Exception as e:
            logger.error(f"일일 보고서 생성 오류: {e}")
    
    def _perform_weekly_analysis(self):
        """주간 트렌드 분석"""
        logger.info("주간 트렌드 분석 시작")
        
        try:
            # 지난 주 데이터 종합 분석
            weekly_data = self._collect_weekly_data()
            
            analysis_request = {
                "type": "weekly_trends",
                "data": weekly_data,
                "source": "autonomous_weekly_analysis",
                "priority": "high",
                "description": "주간 트렌드 종합 분석"
            }
            
            self._trigger_autonomous_analysis(analysis_request)
            
        except Exception as e:
            logger.error(f"주간 분석 오류: {e}")
    
    def _trigger_autonomous_analysis(self, request_data: Dict[str, Any]):
        """자율 분석 트리거"""
        try:
            # n8n 마스터 오케스트레이터에 요청
            response = requests.post(
                self.master_trigger_url,
                json={
                    **request_data,
                    "autonomous": True,
                    "timestamp": datetime.now().isoformat(),
                    "scheduler_id": f"auto_{int(time.time())}"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"자율 분석 트리거 성공: {request_data['type']}")
                self.last_analysis_time = datetime.now()
            else:
                logger.warning(f"자율 분석 트리거 실패: {response.status_code}")
                
        except Exception as e:
            logger.error(f"자율 분석 트리거 오류: {e}")
    
    def _should_analyze_keyword(self, keyword: str) -> bool:
        """키워드 분석 필요성 판단"""
        # 우선순위 키워드인지 확인
        priority_keywords = self.config["analysis_thresholds"]["priority_boost_keywords"]
        return any(pk.lower() in keyword.lower() for pk in priority_keywords)
    
    def _should_perform_analysis(self) -> bool:
        """분석 수행 조건 확인"""
        if not self.last_analysis_time:
            return True
        
        min_interval = self.config["analysis_thresholds"]["min_interval_hours"]
        time_diff = datetime.now() - self.last_analysis_time
        
        return time_diff >= timedelta(hours=min_interval)
    
    def _check_n8n_health(self) -> bool:
        """n8n 서비스 상태 확인"""
        try:
            response = requests.get("http://localhost:5678/api/v1/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_ollama_health(self) -> bool:
        """Ollama 서비스 상태 확인"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """디스크 공간 확인"""
        import shutil
        total, used, free = shutil.disk_usage(".")
        return {
            "total_gb": total // (1024**3),
            "used_gb": used // (1024**3),
            "free_gb": free // (1024**3),
            "usage_percent": (used / total) * 100
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 확인"""
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total // (1024**3),
            "used_gb": memory.used // (1024**3),
            "available_gb": memory.available // (1024**3),
            "usage_percent": memory.percent
        }
    
    def _handle_service_issue(self, service_name: str):
        """서비스 문제 자동 대응"""
        logger.info(f"{service_name} 서비스 문제 자동 대응 시작")
        # 추후 자동 재시작 로직 구현
    
    def _save_health_report(self, health_status: Dict[str, Any]):
        """건강도 보고서 저장"""
        report_file = self.analysis_history_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "health_status": health_status
            }, f, ensure_ascii=False, indent=2)
    
    def _identify_analysis_targets(self) -> List[Dict[str, Any]]:
        """분석 대상 식별"""
        return [
            {"name": "업계 뉴스", "type": "news", "priority": "normal"},
            {"name": "기술 트렌드", "type": "technology", "priority": "high"},
            {"name": "시장 분석", "type": "market", "priority": "normal"}
        ]
    
    def _collect_daily_summary(self, date: datetime) -> Dict[str, Any]:
        """일일 요약 수집"""
        return {
            "date": date.strftime("%Y-%m-%d"),
            "analyses_performed": 0,  # 실제 분석 수 계산
            "key_insights": [],
            "system_performance": "good"
        }
    
    def _collect_weekly_data(self) -> Dict[str, Any]:
        """주간 데이터 수집"""
        return {
            "week_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "total_analyses": 0,
            "trending_topics": [],
            "performance_metrics": {}
        }
    
    def stop_autonomous_mode(self):
        """자율 모드 정지"""
        self.is_running = False
        schedule.clear()
        logger.info("자율 모드 정지 완료")
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "is_running": self.is_running,
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "scheduled_jobs": len(schedule.jobs),
            "config": self.config
        }

# 스트림릿 인터페이스
def render_autonomous_dashboard():
    """자율 스케줄러 대시보드"""
    import streamlit as st
    
    st.title("🤖 SOLOMOND AI 자율 스케줄러")
    st.markdown("**AI 에이전트의 자율성 강화 시스템**")
    
    # 스케줄러 인스턴스
    if 'autonomous_scheduler' not in st.session_state:
        st.session_state.autonomous_scheduler = AutonomousScheduler()
    
    scheduler = st.session_state.autonomous_scheduler
    
    # 현재 상태 표시
    col1, col2, col3, col4 = st.columns(4)
    
    status = scheduler.get_status()
    
    with col1:
        status_icon = "🟢" if status["is_running"] else "🔴"
        st.metric("자율 모드", f"{status_icon} {'활성화' if status['is_running'] else '비활성화'}")
    
    with col2:
        st.metric("스케줄된 작업", status["scheduled_jobs"])
    
    with col3:
        last_analysis = status["last_analysis"]
        if last_analysis:
            last_time = datetime.fromisoformat(last_analysis)
            time_ago = datetime.now() - last_time
            st.metric("마지막 분석", f"{time_ago.seconds//60}분 전")
        else:
            st.metric("마지막 분석", "없음")
    
    with col4:
        st.metric("시스템 상태", "정상")
    
    # 제어 버튼
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 자율 모드 시작", disabled=status["is_running"]):
            scheduler.start_autonomous_mode()
            st.success("자율 모드가 시작되었습니다!")
            st.rerun()
    
    with col2:
        if st.button("🛑 자율 모드 정지", disabled=not status["is_running"]):
            scheduler.stop_autonomous_mode()
            st.success("자율 모드가 정지되었습니다!")
            st.rerun()
    
    # 설정 표시
    st.subheader("⚙️ 현재 설정")
    st.json(status["config"])
    
    # 로그 표시
    st.subheader("📋 최근 로그")
    if Path("autonomous_scheduler.log").exists():
        with open("autonomous_scheduler.log", 'r', encoding='utf-8') as f:
            logs = f.readlines()[-20:]  # 최근 20줄
            for log in logs:
                st.text(log.strip())

if __name__ == "__main__":
    import streamlit as st
    
    st.set_page_config(
        page_title="자율 스케줄러",
        page_icon="🤖",
        layout="wide"
    )
    
    render_autonomous_dashboard()