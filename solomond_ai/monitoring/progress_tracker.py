"""
진행률 추적기
파일 처리 진행률 및 예상 완료 시간 계산
"""

import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging

class ProgressTracker:
    """진행률 추적 및 예상 시간 계산 클래스"""
    
    def __init__(self, total_items: int = 0):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = None
        self.current_item_start = None
        
        # 항목별 처리 기록
        self.item_records = []
        self.processing_times = []
        
        # 예측 모델
        self.size_based_estimates = {
            'audio': 8.0,    # 초/MB
            'image': 2.0,    # 초/MB
            'video': 1.0,    # 초/MB
            'text': 0.5      # 초/MB
        }
        
        # 콜백 함수들
        self.progress_callbacks = []
        
    def start_tracking(self, total_items: int = None):
        """추적 시작"""
        if total_items is not None:
            self.total_items = total_items
        
        self.start_time = time.time()
        self.processed_items = 0
        self.item_records = []
        self.processing_times = []
        
        logging.info(f"Progress tracking started for {self.total_items} items")
    
    def start_item(self, item_info: Dict[str, Any] = None):
        """개별 항목 처리 시작"""
        self.current_item_start = time.time()
        
        if item_info:
            item_info['start_time'] = self.current_item_start
            self.item_records.append(item_info)
    
    def complete_item(self, success: bool = True, details: Dict[str, Any] = None):
        """개별 항목 처리 완료"""
        if self.current_item_start is None:
            logging.warning("complete_item called without start_item")
            return
        
        end_time = time.time()
        processing_time = end_time - self.current_item_start
        
        self.processed_items += 1
        self.processing_times.append(processing_time)
        
        # 최근 항목 기록 업데이트
        if self.item_records:
            last_record = self.item_records[-1]
            last_record.update({
                'end_time': end_time,
                'processing_time': processing_time,
                'success': success,
                'details': details or {}
            })
        
        # 진행률 콜백 호출
        progress_info = self.get_progress_info()
        for callback in self.progress_callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                logging.error(f"Progress callback error: {e}")
        
        self.current_item_start = None
        
        logging.debug(f"Item {self.processed_items}/{self.total_items} completed in {processing_time:.2f}s")
    
    def add_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """진행률 콜백 추가"""
        self.progress_callbacks.append(callback)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """현재 진행률 정보"""
        if self.start_time is None:
            return {'status': 'not_started'}
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 진행률 계산
        progress_percentage = (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0
        
        # 예상 완료 시간 계산
        estimated_completion = self._calculate_estimated_completion(current_time)
        
        # 현재 처리 속도
        items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'remaining_items': self.total_items - self.processed_items,
            'progress_percentage': round(progress_percentage, 1),
            'elapsed_time': round(elapsed_time, 1),
            'estimated_time_remaining': estimated_completion['remaining_seconds'],
            'estimated_completion_time': estimated_completion['completion_time'],
            'items_per_second': round(items_per_second, 2),
            'average_item_time': round(sum(self.processing_times) / len(self.processing_times), 2) if self.processing_times else 0,
            'status': 'completed' if self.processed_items >= self.total_items else 'processing'
        }
    
    def _calculate_estimated_completion(self, current_time: float) -> Dict[str, Any]:
        """예상 완료 시간 계산"""
        if self.processed_items == 0:
            return {
                'remaining_seconds': 0,
                'completion_time': None
            }
        
        # 최근 처리 시간 기반 예측 (가중 평균)
        if len(self.processing_times) >= 3:
            # 최근 3개 항목의 가중 평균 (최신 항목에 더 높은 가중치)
            recent_times = self.processing_times[-3:]
            weights = [1, 2, 3]  # 최신 항목에 더 높은 가중치
            weighted_avg = sum(t * w for t, w in zip(recent_times, weights)) / sum(weights)
        else:
            # 전체 평균
            weighted_avg = sum(self.processing_times) / len(self.processing_times)
        
        # 남은 항목 수와 예상 처리 시간
        remaining_items = self.total_items - self.processed_items
        estimated_remaining_seconds = remaining_items * weighted_avg
        
        # 완료 예상 시각
        completion_timestamp = current_time + estimated_remaining_seconds
        completion_time = datetime.fromtimestamp(completion_timestamp)
        
        return {
            'remaining_seconds': round(estimated_remaining_seconds, 1),
            'completion_time': completion_time.strftime('%H:%M:%S')
        }
    
    def estimate_by_file_size(self, files_info: List[Dict[str, Any]]) -> Dict[str, float]:
        """파일 크기 기반 처리 시간 예측"""
        estimates_by_type = {}
        
        for file_info in files_info:
            file_type = file_info.get('type', 'unknown')
            file_size_mb = file_info.get('size_mb', 0)
            
            if file_type in self.size_based_estimates:
                estimated_time = file_size_mb * self.size_based_estimates[file_type]
                
                if file_type not in estimates_by_type:
                    estimates_by_type[file_type] = 0
                estimates_by_type[file_type] += estimated_time
        
        return estimates_by_type
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭"""
        if not self.processing_times:
            return {}
        
        times = self.processing_times
        total_time = sum(times)
        
        return {
            'total_processing_time': round(total_time, 2),
            'average_time_per_item': round(total_time / len(times), 2),
            'fastest_item_time': round(min(times), 2),
            'slowest_item_time': round(max(times), 2),
            'throughput_items_per_minute': round(len(times) / (total_time / 60), 2) if total_time > 0 else 0,
            'processing_efficiency': self._calculate_efficiency()
        }
    
    def _calculate_efficiency(self) -> float:
        """처리 효율성 계산 (0-100%)"""
        if len(self.processing_times) < 2:
            return 100.0
        
        # 시간 변동성 기반 효율성
        times = self.processing_times
        avg_time = sum(times) / len(times)
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5
        
        # 변동계수 (CV) 계산
        coefficient_of_variation = (std_dev / avg_time) if avg_time > 0 else 1
        
        # 효율성 = 100 - (변동계수 * 100), 최소 0%
        efficiency = max(0, 100 - (coefficient_of_variation * 100))
        
        return round(efficiency, 1)
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """상세 처리 보고서"""
        progress = self.get_progress_info()
        metrics = self.get_performance_metrics()
        
        # 성공/실패 통계
        successful_items = sum(1 for record in self.item_records if record.get('success', True))
        failed_items = len(self.item_records) - successful_items
        
        return {
            'progress': progress,
            'performance_metrics': metrics,
            'success_statistics': {
                'successful_items': successful_items,
                'failed_items': failed_items,
                'success_rate': round(successful_items / len(self.item_records) * 100, 1) if self.item_records else 0
            },
            'item_records': self.item_records,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """성능 기반 권장사항 생성"""
        recommendations = []
        
        if not self.processing_times:
            return recommendations
        
        # 평균 처리 시간 분석
        avg_time = sum(self.processing_times) / len(self.processing_times)
        
        if avg_time > 30:  # 30초 이상
            recommendations.append("💡 파일 크기가 큰 경우 배치 크기를 줄여보세요")
        
        # 변동성 분석
        efficiency = self._calculate_efficiency()
        if efficiency < 70:
            recommendations.append("⚠️ 처리 시간 변동이 큽니다. 시스템 리소스를 확인해보세요")
        
        # 실패율 분석
        if self.item_records:
            failure_rate = sum(1 for r in self.item_records if not r.get('success', True)) / len(self.item_records)
            if failure_rate > 0.1:  # 10% 이상 실패
                recommendations.append("🔧 높은 실패율이 감지되었습니다. 입력 데이터를 확인해보세요")
        
        if not recommendations:
            recommendations.append("✅ 처리 성능이 양호합니다")
        
        return recommendations
    
    def reset(self):
        """추적 데이터 초기화"""
        self.processed_items = 0
        self.start_time = None
        self.current_item_start = None
        self.item_records = []
        self.processing_times = []
        
        logging.info("Progress tracking reset")