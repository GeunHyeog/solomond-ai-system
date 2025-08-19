#!/usr/bin/env python3
"""
🎯 SOLOMOND AI v7.4 최종 통합 테스트
Complete Final Integration Test
"""

import requests
import time
import json
from typing import Dict, Any

def test_system_status() -> bool:
    """시스템 상태 확인"""
    try:
        response = requests.get("http://localhost:8610", timeout=10)
        print(f"시스템 접근 가능: HTTP {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"시스템 접근 실패: {e}")
        return False

def test_ollama_connection() -> bool:
    """Ollama AI 연결 테스트"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            model_count = len(models.get("models", []))
            print(f"Ollama 연결 성공: {model_count}개 모델")
            return True
        else:
            print(f"Ollama 연결 실패: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Ollama 연결 오류: {e}")
        return False

def test_database_connection() -> bool:
    """데이터베이스 연결 테스트"""
    try:
        from database_adapter import DatabaseFactory
        db = DatabaseFactory.create_database("auto", "test_conference")
        success = db.create_fragments_table()
        print(f"데이터베이스 연결: {'성공' if success else '실패'}")
        return success
    except Exception as e:
        print(f"데이터베이스 오류: {e}")
        return False

def test_youtube_support() -> bool:
    """유튜브/웹 동영상 지원 테스트"""
    try:
        import yt_dlp
        print("yt-dlp 사용 가능: 1000+ 플랫폼 지원")
        return True
    except ImportError:
        print("yt-dlp 없음: 웹 동영상 분석 불가")
        return False

def run_complete_test():
    """완전한 시스템 테스트 실행"""
    print("SOLOMOND AI v7.4 최종 통합 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("시스템 상태", test_system_status),
        ("Ollama AI", test_ollama_connection),
        ("데이터베이스", test_database_connection),
        ("웹 동영상", test_youtube_support)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name} 테스트 중...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"{test_name} 테스트 오류: {e}")
            results[test_name] = False
        
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("최종 테스트 결과")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "통과" if result else "실패"
        print(f"{test_name:15} | {status}")
    
    print("-" * 60)
    print(f"전체 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("모든 테스트 통과! 시스템 완전 준비됨")
    elif passed >= total * 0.75:
        print("대부분 테스트 통과 - 일부 기능 제한적")
    else:
        print("다수 테스트 실패 - 시스템 점검 필요")
    
    return passed, total

if __name__ == "__main__":
    run_complete_test()