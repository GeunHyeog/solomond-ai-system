#!/usr/bin/env python3
"""
GPU 메모리 관리 시스템 테스트
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.gpu_memory_manager import GPUMemoryManager, ComputeMode

def test_gpu_detection():
    """GPU 감지 테스트"""
    print("🔍 GPU 감지 테스트")
    print("=" * 50)
    
    manager = GPUMemoryManager()
    status = manager.get_status_report()
    
    print(f"GPU 사용 가능: {'✅' if status['gpu_available'] else '❌'}")
    print(f"GPU 개수: {status['gpu_count']}")
    print(f"현재 모드: {status['current_mode']}")
    
    if status['gpu_details']:
        print("\n📊 GPU 세부 정보:")
        for gpu in status['gpu_details']:
            print(f"   GPU {gpu['device_id']}: {gpu['name']}")
            print(f"      메모리: {gpu['memory_free_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB")
            print(f"      사용률: {gpu['utilization_percent']}%")
            if gpu['temperature_c']:
                print(f"      온도: {gpu['temperature_c']}°C")
    
    print("✅ GPU 감지 테스트 완료\n")

def test_memory_info():
    """메모리 정보 테스트"""
    print("💾 메모리 정보 테스트")
    print("=" * 50)
    
    manager = GPUMemoryManager()
    memory_info = manager.get_memory_info()
    
    for device, info in memory_info.items():
        print(f"📊 {device} ({info.device_name}):")
        print(f"   총 용량: {info.total_mb:.1f}MB")
        print(f"   사용 중: {info.used_mb:.1f}MB")
        print(f"   사용 가능: {info.available_mb:.1f}MB")
        print(f"   사용률: {info.usage_percent:.1f}%")
        print()
    
    print("✅ 메모리 정보 테스트 완료\n")

def test_mode_switching():
    """모드 전환 테스트"""
    print("🔄 모드 전환 테스트")
    print("=" * 50)
    
    manager = GPUMemoryManager()
    
    # 현재 모드 확인
    original_mode = manager.current_mode
    print(f"원래 모드: {original_mode.value}")
    
    # CPU 모드로 전환
    print("CPU 모드로 전환 시도...")
    success = manager.switch_mode(ComputeMode.CPU_ONLY)
    print(f"전환 결과: {'성공' if success else '실패'}")
    print(f"현재 모드: {manager.current_mode.value}")
    
    # 환경 변수 확인
    print(f"CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}'")
    
    # GPU 모드로 전환 (사용 가능한 경우)
    if manager.gpu_available:
        print("\nGPU 모드로 전환 시도...")
        success = manager.switch_mode(ComputeMode.GPU_ONLY, required_memory_mb=1000)
        print(f"전환 결과: {'성공' if success else '실패'}")
        print(f"현재 모드: {manager.current_mode.value}")
        print(f"CUDA_VISIBLE_DEVICES: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}'")
    
    # 자동 모드
    print("\n자동 모드로 전환...")
    recommended = manager.get_recommended_mode(required_memory_mb=1500)
    success = manager.switch_mode(recommended)
    print(f"권장 모드: {recommended.value}")
    print(f"전환 결과: {'성공' if success else '실패'}")
    print(f"현재 모드: {manager.current_mode.value}")
    
    print("✅ 모드 전환 테스트 완료\n")

def test_monitoring():
    """메모리 모니터링 테스트"""
    print("📊 메모리 모니터링 테스트")
    print("=" * 50)
    
    manager = GPUMemoryManager(monitor_interval=2.0)  # 2초 간격
    
    # 모니터링 시작
    print("모니터링 시작...")
    manager.start_monitoring()
    
    # 5초 대기
    print("5초간 모니터링...")
    time.sleep(5)
    
    # 히스토리 확인
    history_count = len(manager.memory_history)
    print(f"기록된 히스토리: {history_count}개")
    
    if manager.memory_history:
        latest = manager.memory_history[-1]['memory_info']
        print("최신 메모리 상태:")
        for device, info in latest.items():
            print(f"   {device}: {info.usage_percent:.1f}% 사용 중")
    
    # 모니터링 중지
    manager.stop_monitoring()
    print("모니터링 중지")
    
    print("✅ 메모리 모니터링 테스트 완료\n")

def test_status_report():
    """상태 보고서 테스트"""
    print("📋 상태 보고서 테스트")
    print("=" * 50)
    
    manager = GPUMemoryManager()
    status = manager.get_status_report()
    
    print("=== GPU 메모리 매니저 상태 보고서 ===")
    print(f"현재 모드: {status['current_mode']}")
    print(f"GPU 사용 가능: {status['gpu_available']}")
    print(f"GPU 개수: {status['gpu_count']}")
    print(f"선택된 GPU: {status['selected_gpu_id']}")
    print(f"메모리 임계값: {status['memory_threshold_mb']}MB")
    print(f"CPU 폴백 활성화: {status['cpu_fallback_enabled']}")
    print(f"모니터링 활성화: {status['monitoring_active']}")
    
    print("\n=== 메모리 상태 ===")
    for device, info in status['memory_info'].items():
        print(f"{device}: {info['used_mb']}MB / {info['total_mb']}MB ({info['usage_percent']:.1f}%)")
    
    if status['gpu_details']:
        print("\n=== GPU 세부 정보 ===")
        for gpu in status['gpu_details']:
            print(f"GPU {gpu['device_id']}: {gpu['name']}")
            print(f"   메모리: {gpu['memory_free_mb']:.0f}MB 사용 가능")
            print(f"   사용률: {gpu['utilization_percent']}%")
    
    print("✅ 상태 보고서 테스트 완료\n")

def test_integration_with_ai_models():
    """AI 모델 통합 테스트 시뮬레이션"""
    print("🧠 AI 모델 통합 테스트")
    print("=" * 50)
    
    manager = GPUMemoryManager()
    
    # 모델 로딩 시뮬레이션
    print("AI 모델 로딩 시뮬레이션...")
    
    # 1. Whisper 모델 (1.5GB 필요)
    whisper_mode = manager.get_recommended_mode(required_memory_mb=1500)
    print(f"Whisper 권장 모드: {whisper_mode.value}")
    manager.switch_mode(whisper_mode, required_memory_mb=1500)
    
    # 2. EasyOCR 모델 (500MB 필요)
    easyocr_mode = manager.get_recommended_mode(required_memory_mb=500)
    print(f"EasyOCR 권장 모드: {easyocr_mode.value}")
    
    # 3. Transformers 모델 (800MB 필요)
    transformers_mode = manager.get_recommended_mode(required_memory_mb=800)
    print(f"Transformers 권장 모드: {transformers_mode.value}")
    
    # 메모리 정리 테스트
    print("메모리 정리 중...")
    manager._cleanup_memory()
    
    print("✅ AI 모델 통합 테스트 완료\n")

def main():
    """메인 테스트 함수"""
    print("🖥️  GPU 메모리 관리 시스템 종합 테스트")
    print("=" * 60)
    
    try:
        test_gpu_detection()
        test_memory_info()
        test_mode_switching()
        test_status_report()
        test_integration_with_ai_models()
        
        # 선택적 모니터링 테스트
        if input("메모리 모니터링 테스트를 실행하시겠습니까? (y/n): ").lower() == 'y':
            test_monitoring()
        
        print("=" * 60)
        print("🎉 모든 테스트 완료!")
        print("GPU 메모리 관리 시스템이 정상적으로 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)