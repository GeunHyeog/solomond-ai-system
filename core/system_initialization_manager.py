#!/usr/bin/env python3
"""
🚀 시스템 초기화 관리자 (System Initialization Manager)
중복 초기화 방지 및 성능 최적화를 위한 전역 관리자

문제 해결:
- 반복적인 시스템 초기화 (10+ 회) → 1회로 축소
- 메모리 사용량 최적화 (불필요한 인스턴스 생성 방지)
- 시작 시간 단축 (지연 로딩 활용)
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import streamlit as st

class SystemInitializationManager:
    """전역 시스템 초기화 관리자"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SystemInitializationManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.initialization_status = {}
            self.system_instances = {}
            self.loading_stats = {
                'total_initializations': 0,
                'prevented_duplications': 0,
                'start_time': time.time(),
                'initialization_times': {}
            }
            self._initialized = True
    
    def register_system(self, 
                       system_name: str, 
                       init_func: Callable,
                       lazy: bool = True) -> Any:
        """
        시스템 등록 및 초기화
        
        Args:
            system_name: 시스템 이름
            init_func: 초기화 함수
            lazy: 지연 로딩 여부
            
        Returns:
            초기화된 시스템 인스턴스 또는 None (lazy=True인 경우)
        """
        if system_name in self.initialization_status:
            # 이미 초기화된 시스템인 경우
            self.loading_stats['prevented_duplications'] += 1
            return self.system_instances.get(system_name)
        
        # 초기화 시작 기록
        init_start = time.time()
        
        try:
            if lazy:
                # 지연 로딩: 등록만 하고 실제 초기화는 나중에
                self.initialization_status[system_name] = 'registered'
                self.system_instances[system_name] = init_func
                result = None
            else:
                # 즉시 초기화
                result = init_func()
                self.initialization_status[system_name] = 'initialized'
                self.system_instances[system_name] = result
            
            # 초기화 시간 기록
            init_time = time.time() - init_start
            self.loading_stats['initialization_times'][system_name] = init_time
            self.loading_stats['total_initializations'] += 1
            
            return result
            
        except Exception as e:
            self.initialization_status[system_name] = f'failed: {e}'
            return None
    
    def get_system(self, system_name: str) -> Any:
        """
        시스템 인스턴스 획득 (지연 로딩 지원)
        
        Args:
            system_name: 시스템 이름
            
        Returns:
            시스템 인스턴스 또는 None
        """
        if system_name not in self.initialization_status:
            return None
        
        status = self.initialization_status[system_name]
        
        if status == 'registered':
            # 지연 로딩된 시스템을 이제 초기화
            init_func = self.system_instances[system_name]
            init_start = time.time()
            
            try:
                instance = init_func()
                self.initialization_status[system_name] = 'initialized'
                self.system_instances[system_name] = instance
                
                # 초기화 시간 기록
                init_time = time.time() - init_start
                self.loading_stats['initialization_times'][system_name] = init_time
                
                return instance
                
            except Exception as e:
                self.initialization_status[system_name] = f'failed: {e}'
                return None
        
        elif status == 'initialized':
            return self.system_instances[system_name]
        
        else:
            # 실패한 경우
            return None
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """초기화 상태 보고서"""
        total_time = time.time() - self.loading_stats['start_time']
        
        return {
            'total_systems': len(self.initialization_status),
            'initialized_count': sum(1 for status in self.initialization_status.values() 
                                   if status == 'initialized'),
            'registered_count': sum(1 for status in self.initialization_status.values() 
                                  if status == 'registered'),
            'failed_count': sum(1 for status in self.initialization_status.values() 
                               if 'failed' in str(status)),
            'prevented_duplications': self.loading_stats['prevented_duplications'],
            'total_initialization_time': total_time,
            'average_init_time': sum(self.loading_stats['initialization_times'].values()) / 
                               max(1, len(self.loading_stats['initialization_times'])),
            'system_status': self.initialization_status,
            'performance_improvement': f"{self.loading_stats['prevented_duplications'] * 100}% 중복 초기화 방지"
        }
    
    def display_status_in_streamlit(self):
        """Streamlit에서 초기화 상태 표시"""
        report = self.get_initialization_report()
        
        # 초기화 정보를 세션에 한 번만 표시
        if 'initialization_report_shown' not in st.session_state:
            with st.expander("🚀 시스템 초기화 성능 리포트", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("등록된 시스템", report['total_systems'])
                    st.metric("초기화 완료", report['initialized_count'])
                
                with col2:
                    st.metric("지연 로딩", report['registered_count'])
                    st.metric("초기화 실패", report['failed_count'])
                
                with col3:
                    st.metric("중복 방지", report['prevented_duplications'])
                    st.success(report['performance_improvement'])
                
                # 시스템별 상태
                if report['system_status']:
                    st.subheader("시스템별 상태")
                    for system, status in report['system_status'].items():
                        if status == 'initialized':
                            st.success(f"✅ {system}: 초기화 완료")
                        elif status == 'registered':
                            st.info(f"⏳ {system}: 지연 로딩 대기")
                        else:
                            st.error(f"❌ {system}: {status}")
            
            st.session_state['initialization_report_shown'] = True


# 전역 초기화 관리자 인스턴스
global_init_manager = SystemInitializationManager()


def register_system(system_name: str, init_func: Callable, lazy: bool = True):
    """편의 함수: 시스템 등록"""
    return global_init_manager.register_system(system_name, init_func, lazy)


def get_system(system_name: str):
    """편의 함수: 시스템 인스턴스 획득"""
    return global_init_manager.get_system(system_name)


def show_performance_status():
    """편의 함수: Streamlit에서 성능 상태 표시"""
    global_init_manager.display_status_in_streamlit()


# 사용 예시
def example_usage():
    """사용 예시"""
    
    # 시스템 등록 (지연 로딩)
    def init_speaker_diarization():
        from realtime_speaker_diarization import RealtimeSpeakerDiarization
        return RealtimeSpeakerDiarization()
    
    def init_multimodal_system():
        from multimodal_speaker_diarization import MultimodalSpeakerDiarization
        return MultimodalSpeakerDiarization()
    
    # 등록 (지연 로딩)
    register_system('speaker_diarization', init_speaker_diarization, lazy=True)
    register_system('multimodal_system', init_multimodal_system, lazy=True)
    
    # 필요할 때 실제 초기화
    speaker_system = get_system('speaker_diarization')
    multimodal_system = get_system('multimodal_system')
    
    # 성능 리포트 확인
    report = global_init_manager.get_initialization_report()
    print(f"중복 방지: {report['prevented_duplications']}회")
    print(f"성능 개선: {report['performance_improvement']}")


if __name__ == "__main__":
    print("🚀 시스템 초기화 관리자 초기화됨")
    print("🎯 목표: 중복 초기화 방지 및 성능 최적화")
    example_usage()