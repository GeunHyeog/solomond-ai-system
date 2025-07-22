#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자동 진짜 분석 테스트
실제 분석 엔진을 사용하여 서로 다른 내용에 대해 다른 결과가 나오는지 확인
"""

import sys
import os
import tempfile
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_files():
    """테스트용 파일들 생성"""
    
    test_cases = [
        {
            "name": "다이아몬드 반지 구매 상담",
            "content": """안녕하세요. 다이아몬드 반지를 찾고 있어요.
약혼반지로 쓸 건데 1캐럿 정도로 생각하고 있습니다.
GIA 인증서 있는 걸로요. 가격이 얼마나 할까요?
할인도 가능한지 궁금해요. 언제 매장에서 볼 수 있나요?""",
            "expected_keywords": ["다이아몬드", "반지", "가격", "구매", "약혼"]
        },
        {
            "name": "날씨와 일상 대화", 
            "content": """오늘 날씨가 정말 좋네요.
햇살이 따뜻하고 바람도 시원해요.
커피 한 잔 마시면서 공원에서 산책하고 싶어요.
이런 날씨에는 기분이 좋아져요.""",
            "expected_keywords": ["날씨", "커피", "산책", "공원"]
        },
        {
            "name": "컴퓨터 프로그래밍 문의",
            "content": """파이썬 프로그래밍을 배우고 있어요.
데이터 분석에 관심이 많습니다.
머신러닝 라이브러리 추천해주세요.
어떤 책이 좋을까요?""",
            "expected_keywords": ["파이썬", "프로그래밍", "데이터", "머신러닝"]
        }
    ]
    
    files = []
    for case in test_cases:
        # 임시 텍스트 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(case["content"])
            files.append({
                "name": case["name"],
                "path": f.name,
                "content": case["content"],
                "expected_keywords": case["expected_keywords"]
            })
    
    return files

def test_real_analysis_engine():
    """실제 분석 엔진 자동 테스트"""
    
    print("=" * 70)
    print("자동 진짜 분석 테스트 시작")
    print("=" * 70)
    
    try:
        from core.real_analysis_engine import analyze_file_real
        print("SUCCESS: 실제 분석 엔진 로드 성공")
    except ImportError as e:
        print(f"FAILED: 실제 분석 엔진 로드 실패: {e}")
        return False
    
    # 테스트 파일들 생성
    test_files = create_test_files()
    print(f"INFO: {len(test_files)}개 테스트 파일 생성 완료")
    
    results = []
    
    for i, file_info in enumerate(test_files, 1):
        print(f"\n[테스트 {i}] {file_info['name']}")
        print("-" * 50)
        print(f"입력 내용: {file_info['content'][:100]}...")
        
        try:
            # 실제 분석 엔진으로 파일 분석
            context = {"project_info": {"topic": "테스트"}}
            result = analyze_file_real(file_info['path'], 'document', 'ko', context)
            
            if result.get('status') == 'success':
                print("SUCCESS: 분석 완료")
                
                # 주요 결과 추출
                full_text = result.get('full_text', '')
                summary = result.get('summary', '')
                comprehensive_messages = result.get('comprehensive_messages', {})
                
                analysis_result = {
                    "name": file_info['name'],
                    "full_text": full_text,
                    "summary": summary,
                    "comprehensive_messages": comprehensive_messages,
                    "success": True
                }
                
                # 종합 메시지 추출 결과 확인
                if comprehensive_messages and comprehensive_messages.get('status') == 'success':
                    main_summary = comprehensive_messages.get('main_summary', {})
                    one_line_summary = main_summary.get('one_line_summary', '')
                    customer_status = main_summary.get('customer_status', '')
                    urgency = main_summary.get('urgency_indicator', '')
                    
                    print(f"종합 메시지: {one_line_summary}")
                    print(f"고객 상태: {customer_status}")
                    print(f"긴급도: {urgency}")
                    
                    analysis_result.update({
                        "one_line_summary": one_line_summary,
                        "customer_status": customer_status,
                        "urgency": urgency
                    })
                else:
                    print("WARNING: 종합 메시지 추출 실패")
                    analysis_result.update({
                        "one_line_summary": "N/A",
                        "customer_status": "N/A", 
                        "urgency": "N/A"
                    })
                
                # 텍스트 추출 확인
                print(f"추출된 텍스트: {full_text[:100]}...")
                print(f"기본 요약: {summary[:100]}...")
                
            else:
                print(f"FAILED: 분석 실패 - {result.get('error', 'Unknown error')}")
                analysis_result = {
                    "name": file_info['name'],
                    "success": False,
                    "error": result.get('error', 'Unknown error')
                }
            
            results.append(analysis_result)
            
        except Exception as e:
            print(f"ERROR: 분석 중 오류 - {e}")
            results.append({
                "name": file_info['name'],
                "success": False,
                "error": str(e)
            })
    
    # 임시 파일들 정리
    for file_info in test_files:
        try:
            os.unlink(file_info['path'])
        except:
            pass
    
    # 결과 분석
    analyze_results(results)
    
    return results

def analyze_results(results):
    """결과 분석 및 진짜/가짜 분석 판단"""
    
    print("\n" + "=" * 70)
    print("결과 분석")
    print("=" * 70)
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) < 2:
        print("WARNING: 성공한 분석이 2개 미만이어서 비교 불가")
        return
    
    # 1. 텍스트 추출 확인
    full_texts = [r.get('full_text', '') for r in successful_results]
    unique_texts = set(full_texts)
    
    print(f"1. 텍스트 추출 다양성: {len(unique_texts)}개의 서로 다른 텍스트")
    if len(unique_texts) > 1:
        print("   ✅ SUCCESS: 입력에 따라 다른 텍스트 추출")
    else:
        print("   ❌ FAILED: 모든 입력에 같은 텍스트 (가짜 분석 의심)")
    
    # 2. 기본 요약 확인
    summaries = [r.get('summary', '') for r in successful_results]
    unique_summaries = set(summaries)
    
    print(f"2. 기본 요약 다양성: {len(unique_summaries)}개의 서로 다른 요약")
    if len(unique_summaries) > 1:
        print("   ✅ SUCCESS: 입력에 따라 다른 요약 생성")
    else:
        print("   ❌ FAILED: 모든 입력에 같은 요약 (가짜 분석 의심)")
    
    # 3. 종합 메시지 추출 확인
    one_line_summaries = [r.get('one_line_summary', '') for r in successful_results if r.get('one_line_summary', '') != 'N/A']
    unique_one_lines = set(one_line_summaries)
    
    print(f"3. 종합 메시지 다양성: {len(unique_one_lines)}개의 서로 다른 핵심 메시지")
    if len(unique_one_lines) > 1:
        print("   ✅ SUCCESS: 입력에 따라 다른 핵심 메시지 생성")
    else:
        print("   ❌ FAILED: 모든 입력에 같은 핵심 메시지 (가짜 분석 의심)")
    
    # 4. 고객 상태 분석 확인
    customer_statuses = [r.get('customer_status', '') for r in successful_results if r.get('customer_status', '') != 'N/A']
    unique_statuses = set(customer_statuses)
    
    print(f"4. 고객 상태 다양성: {len(unique_statuses)}개의 서로 다른 고객 상태")
    if len(unique_statuses) > 1:
        print("   ✅ SUCCESS: 입력에 따라 다른 고객 상태 분석")
    else:
        print("   ❌ FAILED: 모든 입력에 같은 고객 상태 (가짜 분석 의심)")
    
    # 종합 판정
    success_count = sum([
        len(unique_texts) > 1,
        len(unique_summaries) > 1,
        len(unique_one_lines) > 1,
        len(unique_statuses) > 1
    ])
    
    print(f"\n종합 판정: {success_count}/4 항목 통과")
    
    if success_count >= 3:
        print("🎉 결론: 진짜 분석이 정상적으로 작동하고 있습니다!")
        print("   입력에 따라 다른 분석 결과를 생성하고 있습니다.")
    elif success_count >= 2:
        print("⚠️  결론: 부분적으로 진짜 분석이지만 개선이 필요합니다.")
    else:
        print("🚨 결론: 가짜 분석일 가능성이 높습니다!")
        print("   모든 입력에 대해 비슷한 결과를 반환하고 있습니다.")
    
    # 상세 결과 출력
    print("\n상세 결과:")
    for i, result in enumerate(successful_results, 1):
        print(f"{i}. {result['name']}")
        print(f"   - 핵심 메시지: {result.get('one_line_summary', 'N/A')[:50]}...")
        print(f"   - 고객 상태: {result.get('customer_status', 'N/A')}")

if __name__ == "__main__":
    test_real_analysis_engine()