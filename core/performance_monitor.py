#!/usr/bin/env python3
"""
성능 모니터링 시스템
실시간 성공률, 처리 시간, 오류 추적
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict, deque

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.logger = self._setup_logging()
        
        # 성능 데이터 저장소
        self.analysis_history = deque(maxlen=max_history)
        self.error_history = deque(maxlen=max_history)
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "partial_successes": 0,
            "session_start": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # 파일 타입별 통계
        self.file_type_stats = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "failed": 0,
            "avg_time": 0.0,
            "total_time": 0.0
        })
        
        # 실시간 통계 (최근 10분)
        self.recent_window = timedelta(minutes=10)
        self.recent_analyses = deque(maxlen=100)
        
        # 스레드 안전성
        self._lock = threading.Lock()
        
        self.logger.info("🔍 성능 모니터링 시스템 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.PerformanceMonitor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def record_analysis(self, file_name: str, file_type: str, 
                       processing_time: float, status: str, 
                       error_msg: Optional[str] = None, 
                       additional_info: Dict[str, Any] = None):
        """분석 결과 기록"""
        with self._lock:
            timestamp = datetime.now()
            
            # 기본 기록
            analysis_record = {
                "timestamp": timestamp.isoformat(),
                "file_name": file_name,
                "file_type": file_type,
                "processing_time": processing_time,
                "status": status,  # success, failed, partial
                "error_msg": error_msg,
                "additional_info": additional_info or {}
            }
            
            self.analysis_history.append(analysis_record)
            self.recent_analyses.append(analysis_record)
            
            # 전체 통계 업데이트
            self.performance_stats["total_analyses"] += 1
            if status == "success":
                self.performance_stats["successful_analyses"] += 1
            elif status == "failed":
                self.performance_stats["failed_analyses"] += 1
            elif status == "partial":
                self.performance_stats["partial_successes"] += 1
            
            # 파일 타입별 통계 업데이트
            file_stats = self.file_type_stats[file_type]
            file_stats["total"] += 1
            file_stats["total_time"] += processing_time
            file_stats["avg_time"] = file_stats["total_time"] / file_stats["total"]
            
            if status == "success":
                file_stats["success"] += 1
            elif status == "failed":
                file_stats["failed"] += 1
            
            # 오류 기록
            if error_msg:
                self.error_history.append({
                    "timestamp": timestamp.isoformat(),
                    "file_name": file_name,
                    "file_type": file_type,
                    "error_msg": error_msg,
                    "status": status
                })
            
            self.performance_stats["last_updated"] = timestamp.isoformat()
    
    def get_success_rate(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """성공률 계산"""
        with self._lock:
            if time_window:
                # 특정 시간 윈도우 내 데이터만 분석
                cutoff_time = datetime.now() - time_window
                relevant_analyses = [
                    record for record in self.recent_analyses
                    if datetime.fromisoformat(record["timestamp"]) >= cutoff_time
                ]
            else:
                # 전체 데이터 분석
                relevant_analyses = list(self.analysis_history)
            
            if not relevant_analyses:
                return {
                    "success_rate": 0.0,
                    "total_analyses": 0,
                    "successful": 0,
                    "failed": 0,
                    "partial": 0,
                    "time_window": str(time_window) if time_window else "all_time"
                }
            
            total = len(relevant_analyses)
            successful = sum(1 for r in relevant_analyses if r["status"] == "success")
            failed = sum(1 for r in relevant_analyses if r["status"] == "failed")
            partial = sum(1 for r in relevant_analyses if r["status"] == "partial")
            
            # 부분 성공도 절반 점수로 계산
            effective_success = successful + (partial * 0.5)
            success_rate = (effective_success / total) * 100 if total > 0 else 0
            
            return {
                "success_rate": round(success_rate, 2),
                "total_analyses": total,
                "successful": successful,
                "failed": failed,
                "partial": partial,
                "effective_success": round(effective_success, 1),
                "time_window": str(time_window) if time_window else "all_time"
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """전체 성능 요약"""
        with self._lock:
            overall_stats = self.get_success_rate()
            recent_stats = self.get_success_rate(self.recent_window)
            
            # 파일 타입별 성능
            file_type_performance = {}
            for file_type, stats in self.file_type_stats.items():
                if stats["total"] > 0:
                    success_rate = (stats["success"] / stats["total"]) * 100
                    file_type_performance[file_type] = {
                        "success_rate": round(success_rate, 2),
                        "avg_processing_time": round(stats["avg_time"], 2),
                        "total_processed": stats["total"],
                        "successful": stats["success"],
                        "failed": stats["failed"]
                    }
            
            # 최근 오류 분석
            recent_errors = list(self.error_history)[-10:]  # 최근 10개 오류
            error_types = defaultdict(int)
            for error in recent_errors:
                error_msg = error.get("error_msg", "")
                if "M4A" in error_msg or "음성" in error_msg:
                    error_types["audio_processing"] += 1
                elif "이미지" in error_msg or "OCR" in error_msg:
                    error_types["image_processing"] += 1
                elif "메모리" in error_msg or "memory" in error_msg.lower():
                    error_types["memory_issues"] += 1
                else:
                    error_types["other"] += 1
            
            return {
                "overall_performance": overall_stats,
                "recent_performance": recent_stats,
                "file_type_performance": file_type_performance,
                "error_analysis": {
                    "total_errors": len(self.error_history),
                    "recent_errors": len(recent_errors),
                    "error_types": dict(error_types)
                },
                "system_stats": {
                    "session_duration": str(datetime.now() - datetime.fromisoformat(self.performance_stats["session_start"])),
                    "total_files_processed": self.performance_stats["total_analyses"],
                    "last_updated": self.performance_stats["last_updated"]
                }
            }
    
    def get_recommendations(self) -> List[Dict[str, str]]:
        """성능 개선 추천사항"""
        recommendations = []
        summary = self.get_performance_summary()
        
        overall_success = summary["overall_performance"]["success_rate"]
        
        # 전체 성공률 기반 추천
        if overall_success < 70:
            recommendations.append({
                "priority": "high",
                "category": "전체 성능",
                "issue": f"전체 성공률이 {overall_success}%로 낮음",
                "recommendation": "시스템 의존성 확인, 파일 형식 검증 강화 필요"
            })
        elif overall_success < 85:
            recommendations.append({
                "priority": "medium",
                "category": "전체 성능",
                "issue": f"전체 성공률이 {overall_success}%로 개선 여지 있음",
                "recommendation": "에러 로그 분석 및 특정 파일 타입 최적화 권장"
            })
        
        # 파일 타입별 추천
        for file_type, perf in summary["file_type_performance"].items():
            if perf["success_rate"] < 60:
                recommendations.append({
                    "priority": "high",
                    "category": f"{file_type} 처리",
                    "issue": f"{file_type} 파일 성공률 {perf['success_rate']}%",
                    "recommendation": f"{file_type} 전용 최적화 및 에러 핸들링 강화 필요"
                })
            
            if file_type == "image" and perf["avg_processing_time"] > 15:
                recommendations.append({
                    "priority": "medium",
                    "category": "이미지 처리 속도",
                    "issue": f"이미지 처리 시간이 {perf['avg_processing_time']}초로 김",
                    "recommendation": "이미지 해상도 최적화, OCR 파라미터 조정 권장"
                })
        
        # 오류 패턴 기반 추천
        error_types = summary["error_analysis"]["error_types"]
        if error_types.get("audio_processing", 0) > 2:
            recommendations.append({
                "priority": "medium",
                "category": "오디오 처리",
                "issue": "오디오 처리 오류 빈발",
                "recommendation": "M4A 변환기 점검, FFmpeg 설치 확인 필요"
            })
        
        if error_types.get("memory_issues", 0) > 1:
            recommendations.append({
                "priority": "high",
                "category": "메모리 관리",
                "issue": "메모리 관련 오류 발생",
                "recommendation": "가비지 컬렉션 강화, 배치 크기 축소 권장"
            })
        
        return recommendations
    
    def export_report(self, file_path: Optional[str] = None) -> str:
        """성능 리포트 내보내기"""
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"performance_report_{timestamp}.json"
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "recommendations": self.get_recommendations(),
            "detailed_stats": dict(self.file_type_stats),
            "recent_analyses": list(self.recent_analyses)[-20:],  # 최근 20개
            "recent_errors": list(self.error_history)[-10:]  # 최근 10개 오류
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📊 성능 리포트 내보내기 완료: {file_path}")
            return file_path
        
        except Exception as e:
            self.logger.error(f"❌ 성능 리포트 내보내기 실패: {e}")
            return ""
    
    def reset_stats(self):
        """통계 초기화"""
        with self._lock:
            self.analysis_history.clear()
            self.error_history.clear()
            self.recent_analyses.clear()
            self.file_type_stats.clear()
            self.performance_stats = {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "partial_successes": 0,
                "session_start": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self.logger.info("📊 성능 통계 초기화 완료")

# 전역 모니터 인스턴스
global_performance_monitor = PerformanceMonitor()

def record_analysis_result(file_name: str, file_type: str, processing_time: float, 
                         status: str, error_msg: Optional[str] = None, 
                         additional_info: Dict[str, Any] = None):
    """간편 분석 결과 기록 함수"""
    global_performance_monitor.record_analysis(
        file_name, file_type, processing_time, status, error_msg, additional_info
    )

def get_current_success_rate() -> Dict[str, Any]:
    """현재 성공률 조회"""
    return global_performance_monitor.get_success_rate()

def get_system_performance() -> Dict[str, Any]:
    """시스템 성능 요약"""
    return global_performance_monitor.get_performance_summary()

if __name__ == "__main__":
    # 테스트 코드
    monitor = PerformanceMonitor()
    
    # 테스트 데이터 추가
    monitor.record_analysis("test1.jpg", "image", 12.5, "success")
    monitor.record_analysis("test2.mp3", "audio", 8.2, "success")
    monitor.record_analysis("test3.m4a", "audio", 15.1, "failed", "M4A 변환 실패")
    
    # 성능 요약 출력
    summary = monitor.get_performance_summary()
    print("성능 요약:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 추천사항 출력
    recommendations = monitor.get_recommendations()
    print("\n추천사항:")
    for rec in recommendations:
        print(f"- [{rec['priority']}] {rec['category']}: {rec['recommendation']}")