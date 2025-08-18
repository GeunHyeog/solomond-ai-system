#!/usr/bin/env python3
"""
솔로몬드 AI 분석 결과 자동 저장 워크플로우
모든 모듈의 분석 결과를 자동으로 Notion과 Supabase에 저장

통합 기능:
- 4개 모듈 분석 결과 자동 캡처
- 실시간 Notion 동기화
- Supabase 데이터베이스 저장
- 시스템 모니터링 연동
- 자동 백업 및 복구
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading
from dataclasses import dataclass
import logging

# 프로젝트 내 모듈 import
try:
    from notion_supabase_sync import NotionSupabaseSync
    from shared.system_monitor import get_current_system_status
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """분석 결과 데이터 클래스"""
    id: str
    module_type: str  # 'conference', 'gemstone', 'crawler', '3d_cad'
    status: str  # 'pending', 'processing', 'completed', 'failed'
    input_files: List[str]
    output_data: Dict[str, Any]
    processing_time: float
    confidence_score: Optional[float]
    user_id: str
    created_at: datetime
    metadata: Dict[str, Any]

class AnalysisWorkflowManager:
    """분석 워크플로우 통합 관리자"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.active_analyses = {}  # 진행 중인 분석 추적
        self.completed_analyses = []  # 완료된 분석 기록
        
        # 워크플로우 설정
        self.workflow_config = self._load_workflow_config()
        
        # 통합 시스템 초기화
        if INTEGRATIONS_AVAILABLE:
            self.notion_sync = NotionSupabaseSync()
        else:
            self.notion_sync = None
            logger.warning("통합 시스템을 사용할 수 없습니다")
        
        # 자동 저장 스레드
        self.auto_save_enabled = True
        self.auto_save_thread = None
        
        logger.info("분석 워크플로우 매니저 초기화 완료")
    
    def _load_workflow_config(self) -> Dict[str, Any]:
        """워크플로우 설정 로드"""
        default_config = {
            "auto_save_interval": 30,  # 30초마다 자동 저장
            "max_concurrent_analyses": 5,
            "backup_enabled": True,
            "notion_sync_enabled": True,
            "supabase_sync_enabled": True,
            "system_monitoring_enabled": True,
            "module_settings": {
                "conference": {
                    "priority": "high",
                    "timeout": 300,
                    "auto_backup": True
                },
                "gemstone": {
                    "priority": "medium", 
                    "timeout": 180,
                    "auto_backup": True
                },
                "crawler": {
                    "priority": "low",
                    "timeout": 600,
                    "auto_backup": False
                },
                "3d_cad": {
                    "priority": "medium",
                    "timeout": 240,
                    "auto_backup": True
                }
            }
        }
        
        config_file = self.project_root / "workflow_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 기본값과 병합
                    default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"설정 파일 로드 실패: {e}")
        
        return default_config
    
    def start_analysis(self, module_type: str, input_data: Dict[str, Any], 
                      user_id: str = "system") -> str:
        """분석 작업 시작"""
        
        # 분석 ID 생성
        analysis_id = f"{module_type}_{int(time.time())}_{hash(str(input_data)) % 10000}"
        
        # 분석 객체 생성
        analysis_result = AnalysisResult(
            id=analysis_id,
            module_type=module_type,
            status="pending",
            input_files=input_data.get("files", []),
            output_data={},
            processing_time=0.0,
            confidence_score=None,
            user_id=user_id,
            created_at=datetime.now(),
            metadata=input_data.get("metadata", {})
        )
        
        # 활성 분석에 추가
        self.active_analyses[analysis_id] = analysis_result
        
        # 초기 상태 저장
        self._save_analysis_state(analysis_result)
        
        logger.info(f"분석 시작: {analysis_id} ({module_type})")
        return analysis_id
    
    def update_analysis_progress(self, analysis_id: str, 
                               status: str = None,
                               output_data: Dict[str, Any] = None,
                               confidence_score: float = None,
                               processing_time: float = None):
        """분석 진행 상황 업데이트"""
        
        if analysis_id not in self.active_analyses:
            logger.warning(f"알 수 없는 분석 ID: {analysis_id}")
            return False
        
        analysis = self.active_analyses[analysis_id]
        
        # 업데이트 적용
        if status:
            analysis.status = status
        if output_data:
            analysis.output_data.update(output_data)
        if confidence_score is not None:
            analysis.confidence_score = confidence_score
        if processing_time is not None:
            analysis.processing_time = processing_time
        
        # 상태 저장
        self._save_analysis_state(analysis)
        
        # 완료된 경우 후처리
        if status == "completed":
            self._handle_analysis_completion(analysis_id)
        elif status == "failed":
            self._handle_analysis_failure(analysis_id)
        
        logger.info(f"분석 업데이트: {analysis_id} -> {status}")
        return True
    
    def _handle_analysis_completion(self, analysis_id: str):
        """분석 완료 후처리"""
        analysis = self.active_analyses[analysis_id]
        
        try:
            # 1. Notion 동기화
            if self.workflow_config["notion_sync_enabled"] and self.notion_sync:
                notion_data = self._convert_to_notion_format(analysis)
                sync_result = self.notion_sync.sync_analysis_result(notion_data)
                analysis.metadata["notion_sync"] = sync_result
                logger.info(f"Notion 동기화 완료: {analysis_id}")
            
            # 2. 시스템 상태 업데이트
            if self.workflow_config["system_monitoring_enabled"]:
                self._update_system_statistics(analysis)
                logger.info(f"시스템 통계 업데이트: {analysis_id}")
            
            # 3. 백업 생성
            if self.workflow_config["backup_enabled"]:
                self._create_analysis_backup(analysis)
                logger.info(f"백업 생성 완료: {analysis_id}")
            
            # 4. 완료 목록으로 이동
            self.completed_analyses.append(analysis)
            del self.active_analyses[analysis_id]
            
            # 5. 완료 알림
            self._send_completion_notification(analysis)
            
        except Exception as e:
            logger.error(f"분석 완료 후처리 오류: {e}")
            analysis.status = "failed"
            analysis.metadata["error"] = str(e)
    
    def _handle_analysis_failure(self, analysis_id: str):
        """분석 실패 후처리"""
        analysis = self.active_analyses[analysis_id]
        
        try:
            # 실패 로그 저장
            failure_log = {
                "analysis_id": analysis_id,
                "module_type": analysis.module_type,
                "error_time": datetime.now().isoformat(),
                "input_files": analysis.input_files,
                "metadata": analysis.metadata
            }
            
            self._save_failure_log(failure_log)
            
            # 완료 목록으로 이동 (실패한 것도 기록)
            self.completed_analyses.append(analysis)
            del self.active_analyses[analysis_id]
            
            logger.error(f"분석 실패 처리 완료: {analysis_id}")
            
        except Exception as e:
            logger.error(f"분석 실패 후처리 오류: {e}")
    
    def _convert_to_notion_format(self, analysis: AnalysisResult) -> Dict[str, Any]:
        """분석 결과를 Notion 형식으로 변환"""
        
        # 모듈 타입을 한국어로 변환
        module_names = {
            "conference": "컨퍼런스 분석",
            "gemstone": "보석 산지 분석", 
            "crawler": "웹 크롤러",
            "3d_cad": "3D CAD 변환"
        }
        
        return {
            "id": analysis.id,
            "type": module_names.get(analysis.module_type, analysis.module_type),
            "status": "완료" if analysis.status == "completed" else analysis.status,
            "created_at": analysis.created_at.isoformat(),
            "processing_time": analysis.processing_time,
            "files": analysis.input_files,
            "confidence": analysis.confidence_score or 0.0,
            "user": analysis.user_id,
            "summary": self._generate_analysis_summary(analysis),
            "results": analysis.output_data
        }
    
    def _generate_analysis_summary(self, analysis: AnalysisResult) -> str:
        """분석 결과 요약 생성"""
        
        summary_parts = []
        
        # 기본 정보
        summary_parts.append(f"{analysis.module_type} 모듈 분석 완료")
        
        # 파일 정보
        if analysis.input_files:
            summary_parts.append(f"처리된 파일: {len(analysis.input_files)}개")
        
        # 처리 시간
        if analysis.processing_time > 0:
            summary_parts.append(f"처리 시간: {analysis.processing_time:.1f}초")
        
        # 신뢰도
        if analysis.confidence_score:
            summary_parts.append(f"신뢰도: {analysis.confidence_score:.2f}")
        
        # 주요 결과
        if analysis.output_data:
            if "extracted_text" in analysis.output_data:
                text_preview = analysis.output_data["extracted_text"][:100]
                summary_parts.append(f"추출된 텍스트: {text_preview}...")
            
            if "keywords" in analysis.output_data:
                keywords = analysis.output_data["keywords"][:5]
                summary_parts.append(f"키워드: {', '.join(keywords)}")
        
        return " | ".join(summary_parts)
    
    def _save_analysis_state(self, analysis: AnalysisResult):
        """분석 상태 저장"""
        state_file = self.project_root / "analysis_states" / f"{analysis.id}.json"
        state_file.parent.mkdir(exist_ok=True)
        
        state_data = {
            "id": analysis.id,
            "module_type": analysis.module_type,
            "status": analysis.status,
            "input_files": analysis.input_files,
            "output_data": analysis.output_data,
            "processing_time": analysis.processing_time,
            "confidence_score": analysis.confidence_score,
            "user_id": analysis.user_id,
            "created_at": analysis.created_at.isoformat(),
            "metadata": analysis.metadata
        }
        
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
    
    def _update_system_statistics(self, analysis: AnalysisResult):
        """시스템 통계 업데이트"""
        stats_file = self.project_root / "system_statistics.json"
        
        # 기존 통계 로드
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = {
                "total_analyses": 0,
                "module_stats": {},
                "daily_stats": {},
                "last_updated": None
            }
        
        # 통계 업데이트
        stats["total_analyses"] += 1
        
        # 모듈별 통계
        if analysis.module_type not in stats["module_stats"]:
            stats["module_stats"][analysis.module_type] = {
                "count": 0,
                "avg_processing_time": 0,
                "success_rate": 0
            }
        
        module_stats = stats["module_stats"][analysis.module_type]
        module_stats["count"] += 1
        
        # 평균 처리 시간 업데이트
        if analysis.processing_time > 0:
            current_avg = module_stats["avg_processing_time"]
            count = module_stats["count"]
            new_avg = (current_avg * (count - 1) + analysis.processing_time) / count
            module_stats["avg_processing_time"] = new_avg
        
        # 일별 통계
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in stats["daily_stats"]:
            stats["daily_stats"][today] = {"count": 0, "modules": {}}
        
        stats["daily_stats"][today]["count"] += 1
        if analysis.module_type not in stats["daily_stats"][today]["modules"]:
            stats["daily_stats"][today]["modules"][analysis.module_type] = 0
        stats["daily_stats"][today]["modules"][analysis.module_type] += 1
        
        stats["last_updated"] = datetime.now().isoformat()
        
        # 저장
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def _create_analysis_backup(self, analysis: AnalysisResult):
        """분석 결과 백업 생성"""
        backup_dir = self.project_root / "backups" / analysis.module_type
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_file = backup_dir / f"{analysis.id}_backup.json"
        
        backup_data = {
            "analysis": {
                "id": analysis.id,
                "module_type": analysis.module_type,
                "status": analysis.status,
                "input_files": analysis.input_files,
                "output_data": analysis.output_data,
                "processing_time": analysis.processing_time,
                "confidence_score": analysis.confidence_score,
                "user_id": analysis.user_id,
                "created_at": analysis.created_at.isoformat(),
                "metadata": analysis.metadata
            },
            "backup_info": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    def _save_failure_log(self, failure_log: Dict[str, Any]):
        """실패 로그 저장"""
        log_dir = self.project_root / "logs" / "failures"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"failure_{datetime.now().strftime('%Y%m%d')}.json"
        
        # 기존 로그 로드
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(failure_log)
        
        # 저장
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    
    def _send_completion_notification(self, analysis: AnalysisResult):
        """완료 알림 (향후 이메일, 슬랙 등으로 확장 가능)"""
        notification = {
            "type": "analysis_completed",
            "analysis_id": analysis.id,
            "module_type": analysis.module_type,
            "processing_time": analysis.processing_time,
            "confidence_score": analysis.confidence_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # 알림 로그 저장
        notification_file = self.project_root / "notifications.json"
        
        if notification_file.exists():
            with open(notification_file, "r", encoding="utf-8") as f:
                notifications = json.load(f)
        else:
            notifications = []
        
        notifications.append(notification)
        
        # 최근 100개만 유지
        notifications = notifications[-100:]
        
        with open(notification_file, "w", encoding="utf-8") as f:
            json.dump(notifications, f, indent=2, ensure_ascii=False)
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """분석 상태 조회"""
        if analysis_id in self.active_analyses:
            analysis = self.active_analyses[analysis_id]
            return {
                "id": analysis.id,
                "status": analysis.status,
                "progress": "진행중",
                "processing_time": analysis.processing_time,
                "created_at": analysis.created_at.isoformat()
            }
        
        # 완료된 분석에서 검색
        for analysis in self.completed_analyses:
            if analysis.id == analysis_id:
                return {
                    "id": analysis.id,
                    "status": analysis.status,
                    "progress": "완료",
                    "processing_time": analysis.processing_time,
                    "created_at": analysis.created_at.isoformat(),
                    "confidence_score": analysis.confidence_score
                }
        
        return None
    
    def get_system_summary(self) -> Dict[str, Any]:
        """시스템 전체 요약"""
        return {
            "active_analyses": len(self.active_analyses),
            "completed_analyses": len(self.completed_analyses),
            "workflow_config": self.workflow_config,
            "integrations_available": INTEGRATIONS_AVAILABLE,
            "last_updated": datetime.now().isoformat()
        }

# 싱글톤 인스턴스
_workflow_manager = None

def get_workflow_manager() -> AnalysisWorkflowManager:
    """워크플로우 매니저 싱글톤 인스턴스 반환"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = AnalysisWorkflowManager()
    return _workflow_manager

# 편의 함수들
def start_analysis(module_type: str, input_data: Dict[str, Any], user_id: str = "system") -> str:
    """분석 시작 (편의 함수)"""
    return get_workflow_manager().start_analysis(module_type, input_data, user_id)

def update_analysis(analysis_id: str, **kwargs) -> bool:
    """분석 업데이트 (편의 함수)"""
    return get_workflow_manager().update_analysis_progress(analysis_id, **kwargs)

def get_analysis_status(analysis_id: str) -> Optional[Dict[str, Any]]:
    """분석 상태 조회 (편의 함수)"""
    return get_workflow_manager().get_analysis_status(analysis_id)

if __name__ == "__main__":
    # 테스트 코드
    print("=== 분석 워크플로우 통합 시스템 테스트 ===")
    
    manager = AnalysisWorkflowManager()
    
    # 샘플 분석 시작
    analysis_id = manager.start_analysis(
        module_type="conference",
        input_data={
            "files": ["test_audio.m4a", "test_image.png"],
            "metadata": {"conference": "JGA25", "user": "test_user"}
        },
        user_id="test_user"
    )
    
    print(f"분석 시작: {analysis_id}")
    
    # 진행 상황 업데이트
    manager.update_analysis_progress(
        analysis_id,
        status="processing",
        output_data={"progress": 50},
        processing_time=25.5
    )
    
    # 완료 처리
    manager.update_analysis_progress(
        analysis_id,
        status="completed",
        output_data={
            "extracted_text": "JGA25 주얼리 컨퍼런스에서 발표된 내용...",
            "keywords": ["다이아몬드", "주얼리", "트렌드"],
            "stt_confidence": 0.89
        },
        confidence_score=0.89,
        processing_time=45.2
    )
    
    # 상태 확인
    status = manager.get_analysis_status(analysis_id)
    print(f"최종 상태: {status}")
    
    # 시스템 요약
    summary = manager.get_system_summary()
    print(f"시스템 요약: {summary}")
    
    print("SUCCESS: 워크플로우 시스템 테스트 완료!")