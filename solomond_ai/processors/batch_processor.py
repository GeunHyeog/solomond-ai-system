"""
배치 처리기
다중 파일의 병렬 처리 및 진행률 관리
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class BatchProcessor:
    """배치 처리 및 병렬 실행 관리 클래스"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.current_batch = None
        self.progress_callback = None
        self.cancel_event = threading.Event()
        
    def process_batch(self, items: List[Any], 
                     processor_func: Callable[[Any], Dict[str, Any]],
                     progress_callback: Callable[[int, int, Dict[str, Any]], None] = None) -> List[Dict[str, Any]]:
        """배치 처리 실행"""
        
        self.current_batch = {
            'total_items': len(items),
            'processed_items': 0,
            'start_time': time.time(),
            'results': [],
            'errors': []
        }
        
        self.progress_callback = progress_callback
        self.cancel_event.clear()
        
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 작업 제출
                future_to_item = {
                    executor.submit(self._safe_process_item, item, processor_func): item 
                    for item in items
                }
                
                # 결과 수집
                for future in as_completed(future_to_item):
                    if self.cancel_event.is_set():
                        logging.info("Batch processing cancelled")
                        break
                    
                    item = future_to_item[future]
                    result = future.result()
                    
                    results.append(result)
                    self.current_batch['processed_items'] += 1
                    
                    # 진행률 콜백 호출
                    if self.progress_callback:
                        self.progress_callback(
                            self.current_batch['processed_items'],
                            self.current_batch['total_items'],
                            result
                        )
                
                self.current_batch['results'] = results
                self.current_batch['end_time'] = time.time()
                
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            self.current_batch['error'] = str(e)
        
        return results
    
    def _safe_process_item(self, item: Any, processor_func: Callable[[Any], Dict[str, Any]]) -> Dict[str, Any]:
        """안전한 개별 아이템 처리"""
        try:
            start_time = time.time()
            result = processor_func(item)
            end_time = time.time()
            
            # 처리 시간 추가
            if isinstance(result, dict):
                result['processing_time'] = end_time - start_time
                result['processed_at'] = end_time
            
            return result
            
        except Exception as e:
            logging.error(f"Item processing error: {e}")
            return {
                'item': str(item),
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'processed_at': time.time()
            }
    
    def cancel_batch(self):
        """배치 처리 취소"""
        self.cancel_event.set()
        logging.info("Batch processing cancellation requested")
    
    def get_progress(self) -> Dict[str, Any]:
        """현재 진행률 정보"""
        if not self.current_batch:
            return {'status': 'no_batch_running'}
        
        elapsed_time = time.time() - self.current_batch['start_time']
        processed = self.current_batch['processed_items']
        total = self.current_batch['total_items']
        
        progress_percentage = (processed / total * 100) if total > 0 else 0
        
        # 예상 완료 시간 계산
        if processed > 0:
            avg_time_per_item = elapsed_time / processed
            remaining_items = total - processed
            estimated_time_remaining = avg_time_per_item * remaining_items
        else:
            estimated_time_remaining = 0
        
        return {
            'total_items': total,
            'processed_items': processed,
            'remaining_items': total - processed,
            'progress_percentage': round(progress_percentage, 1),
            'elapsed_time': round(elapsed_time, 1),
            'estimated_time_remaining': round(estimated_time_remaining, 1),
            'avg_time_per_item': round(elapsed_time / processed, 2) if processed > 0 else 0,
            'status': 'completed' if processed >= total else 'processing'
        }
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """배치 처리 요약 정보"""
        if not self.current_batch:
            return {'status': 'no_batch_data'}
        
        results = self.current_batch.get('results', [])
        
        # 성공/실패 통계
        successful = sum(1 for r in results if r.get('success', True))
        failed = len(results) - successful
        
        # 처리 시간 통계
        processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        
        total_time = self.current_batch.get('end_time', time.time()) - self.current_batch['start_time']
        
        return {
            'total_items': self.current_batch['total_items'],
            'processed_items': len(results),
            'successful_items': successful,
            'failed_items': failed,
            'success_rate': (successful / len(results) * 100) if results else 0,
            'total_processing_time': round(total_time, 2),
            'average_item_time': round(sum(processing_times) / len(processing_times), 2) if processing_times else 0,
            'min_item_time': round(min(processing_times), 2) if processing_times else 0,
            'max_item_time': round(max(processing_times), 2) if processing_times else 0,
            'items_per_second': round(len(results) / total_time, 2) if total_time > 0 else 0
        }
    
    def process_files_by_engine(self, files_by_type: Dict[str, List[str]], 
                               engines: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """파일 타입별 엔진으로 배치 처리"""
        all_results = {}
        
        for file_type, file_list in files_by_type.items():
            if file_type in engines and file_list:
                engine = engines[file_type]
                
                logging.info(f"Processing {len(file_list)} {file_type} files")
                
                # 해당 엔진으로 배치 처리
                results = self.process_batch(
                    file_list,
                    engine.analyze_file,
                    self._create_progress_callback(file_type)
                )
                
                all_results[file_type] = results
        
        return all_results
    
    def _create_progress_callback(self, file_type: str) -> Callable[[int, int, Dict[str, Any]], None]:
        """파일 타입별 진행률 콜백 생성"""
        def progress_callback(processed: int, total: int, result: Dict[str, Any]):
            logging.info(f"{file_type.upper()} Progress: {processed}/{total} "
                        f"({processed/total*100:.1f}%) - "
                        f"Latest: {result.get('file_path', 'unknown')}")
        
        return progress_callback
    
    def estimate_processing_time(self, files_by_type: Dict[str, List[str]]) -> Dict[str, float]:
        """파일 타입별 예상 처리 시간 (초)"""
        
        # 경험적 처리 시간 (파일 크기 기반)
        time_estimates = {
            'audio': 8.0,    # 초/MB
            'image': 2.0,    # 초/MB  
            'video': 1.0,    # 초/MB
            'text': 0.5      # 초/MB
        }
        
        estimates = {}
        
        for file_type, file_list in files_by_type.items():
            if file_type not in time_estimates:
                continue
            
            total_size_mb = 0
            for file_path in file_list:
                try:
                    from pathlib import Path
                    size_bytes = Path(file_path).stat().st_size
                    total_size_mb += size_bytes / (1024 * 1024)
                except:
                    total_size_mb += 10  # 기본값 10MB
            
            estimated_time = total_size_mb * time_estimates[file_type]
            estimates[file_type] = round(estimated_time, 1)
        
        return estimates