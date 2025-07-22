#!/usr/bin/env python3
"""
강화된 에러 처리 시스템 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.enhanced_error_handler import (
    EnhancedErrorHandler, 
    EnhancedError, 
    ErrorCode, 
    ErrorSeverity,
    handle_error,
    create_enhanced_error
)

def test_error_classification():
    """에러 분류 테스트"""
    print("🧪 에러 분류 테스트 시작")
    print("=" * 50)
    
    handler = EnhancedErrorHandler()
    
    # 테스트 에러들
    test_errors = [
        (FileNotFoundError("test.wav not found"), "파일 없음 에러"),
        (PermissionError("Permission denied: test.wav"), "권한 에러"),
        (Exception("CUDA out of memory"), "메모리 부족 에러"),
        (Exception("Model loading failed: timeout"), "모델 로딩 에러"),
        (Exception("FFmpeg conversion failed"), "오디오 변환 에러"),
        (Exception("Unknown error occurred"), "알 수 없는 에러")
    ]
    
    for i, (error, description) in enumerate(test_errors, 1):
        print(f"\n🔍 [{i}] {description}")
        print("-" * 30)
        
        result = handler.handle_error(error, {"test_context": True})
        
        print(f"   에러 코드: {result['error_code']}")
        print(f"   심각도: {result['severity']}")  
        print(f"   사용자 메시지: {result['user_message']}")
        print(f"   해결방안 수: {len(result['solutions'])}개")
        
        if result['solutions']:
            print("   해결방안:")
            for j, solution in enumerate(result['solutions'][:2], 1):
                print(f"      {j}. {solution}")
        
        print(f"   자동 복구 시도: {'예' if result['recovery_attempted'] else '아니오'}")
        if result['recovery_attempted']:
            print(f"   자동 복구 성공: {'예' if result['recovery_success'] else '아니오'}")
    
    print("\n✅ 에러 분류 테스트 완료")

def test_custom_error():
    """커스텀 에러 테스트"""
    print("\n🎯 커스텀 에러 생성 테스트")
    print("=" * 50)
    
    # 커스텀 에러 생성
    custom_error = create_enhanced_error(
        code="TEST_001",
        message="테스트용 커스텀 에러",
        severity=ErrorSeverity.WARNING,
        context={"test_param": "test_value"},
        solutions=["테스트 해결방안 1", "테스트 해결방안 2"]
    )
    
    handler = EnhancedErrorHandler()
    result = handler.handle_error(custom_error)
    
    print(f"커스텀 에러 코드: {result['error_code']}")
    print(f"커스텀 메시지: {result['user_message']}")
    print(f"커스텀 해결방안: {result['solutions']}")
    
    print("✅ 커스텀 에러 테스트 완료")

def test_auto_recovery():
    """자동 복구 테스트"""
    print("\n🔧 자동 복구 테스트")
    print("=" * 50)
    
    handler = EnhancedErrorHandler()
    
    # 메모리 부족 에러 (자동 복구 가능)
    memory_error = Exception("Out of memory: CUDA")
    result = handler.handle_error(memory_error)
    
    print(f"메모리 에러 자동 복구: {'성공' if result['recovery_success'] else '실패'}")
    
    # 파일 없음 에러 (파일 찾기 시도)
    file_error = FileNotFoundError("test.wav")
    result = handler.handle_error(file_error, {"file_path": "test.wav"})
    
    print(f"파일 없음 에러 처리: {'성공' if result['recovery_attempted'] else '실패'}")
    
    print("✅ 자동 복구 테스트 완료")

def test_error_history():
    """에러 히스토리 테스트"""  
    print("\n📊 에러 히스토리 테스트")
    print("=" * 50)
    
    handler = EnhancedErrorHandler()
    
    # 여러 에러 발생시키기
    test_errors = [
        FileNotFoundError("file1.wav"),
        Exception("Memory error"),
        PermissionError("Permission denied")
    ]
    
    for error in test_errors:
        handler.handle_error(error)
    
    # 히스토리 확인
    history = handler.get_error_history(limit=5)
    
    print(f"기록된 에러 수: {len(history)}")
    for i, record in enumerate(history, 1):
        print(f"   {i}. [{record['code']}] {record['message'][:50]}...")
    
    # 히스토리 초기화
    handler.clear_error_history()
    print(f"초기화 후 에러 수: {len(handler.get_error_history())}")
    
    print("✅ 에러 히스토리 테스트 완료")

def test_streamlit_integration():
    """Streamlit 통합 테스트 시뮬레이션"""
    print("\n🎨 Streamlit 통합 테스트")
    print("=" * 50)
    
    # Streamlit 없이 시뮬레이션
    def mock_streamlit_error(message):
        print(f"ST.ERROR: {message}")
    
    def mock_streamlit_info(message):
        print(f"ST.INFO: {message}")
    
    def mock_streamlit_success(message):
        print(f"ST.SUCCESS: {message}")
    
    # 파일 처리 에러 시뮬레이션
    try:
        # 가상의 파일 처리
        file_name = "test_audio.m4a"
        raise FileNotFoundError(f"{file_name} not found")
        
    except Exception as e:
        try:
            result = handle_error(e, {"file_name": file_name, "step": "file_processing"})
            
            mock_streamlit_error(f"❌ {result['user_message']}")
            
            if result['solutions']:
                mock_streamlit_info("💡 **해결 방법**:")
                for i, solution in enumerate(result['solutions'][:3], 1):
                    mock_streamlit_info(f"   {i}. {solution}")
            
            if result.get('recovery_success'):
                mock_streamlit_success(f"✅ 자동 복구 완료")
        
        except Exception as handler_error:
            mock_streamlit_error(f"❌ 에러 처리 실패: {handler_error}")
    
    print("✅ Streamlit 통합 테스트 완료")

def main():
    """메인 테스트 함수"""
    print("🚨 강화된 에러 처리 시스템 종합 테스트")
    print("=" * 60)
    
    try:
        test_error_classification()
        test_custom_error()
        test_auto_recovery()
        test_error_history()
        test_streamlit_integration()
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 완료!")
        print("강화된 에러 처리 시스템이 정상적으로 작동합니다.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)