"""
솔로몬드 AI 시스템 - 메모리 관리
시스템 메모리 및 리소스 관리 모듈
"""

import gc
import psutil
import os
from typing import Dict, Optional

class MemoryManager:
    """메모리 관리 클래스"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_info(self) -> Dict:
        """현재 메모리 사용량 정보 반환"""
        try:
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            return {
                "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "process_memory_percent": round(self.process.memory_percent(), 2),
                "system_memory_total_gb": round(virtual_memory.total / 1024 / 1024 / 1024, 2),
                "system_memory_available_gb": round(virtual_memory.available / 1024 / 1024 / 1024, 2),
                "system_memory_percent": virtual_memory.percent
            }
        except Exception as e:
            return {
                "error": f"메모리 정보 수집 실패: {e}",
                "process_memory_mb": 0,
                "process_memory_percent": 0
            }
    
    def cleanup(self) -> Dict:
        """메모리 정리 수행"""
        try:
            before_memory = self.get_memory_info()["process_memory_mb"]
            
            # 가비지 콜렉션 수행
            collected = gc.collect()
            
            after_memory = self.get_memory_info()["process_memory_mb"]
            freed_mb = round(before_memory - after_memory, 2)
            
            return {
                "success": True,
                "objects_collected": collected,
                "memory_freed_mb": freed_mb,
                "before_memory_mb": before_memory,
                "after_memory_mb": after_memory
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_memory_threshold(self, threshold_percent: float = 80.0) -> bool:
        """메모리 사용량이 임계치를 초과했는지 확인"""
        memory_info = self.get_memory_info()
        return memory_info.get("system_memory_percent", 0) > threshold_percent
    
    def auto_cleanup_if_needed(self, threshold_percent: float = 80.0) -> Optional[Dict]:
        """메모리 사용량이 높으면 자동 정리"""
        if self.check_memory_threshold(threshold_percent):
            return self.cleanup()
        return None

# 전역 메모리 관리자 인스턴스
_memory_manager_instance = None

def get_memory_manager() -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance

# 편의 함수들
def get_current_memory() -> Dict:
    """현재 메모리 사용량 편의 함수"""
    return get_memory_manager().get_memory_info()

def cleanup_memory() -> Dict:
    """메모리 정리 편의 함수"""
    return get_memory_manager().cleanup()

def auto_cleanup() -> Optional[Dict]:
    """자동 메모리 정리 편의 함수"""
    return get_memory_manager().auto_cleanup_if_needed()
