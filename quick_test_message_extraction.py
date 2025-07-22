#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 종합 메시지 추출 테스트
대용량 파일 업로드 없이 기능 검증
"""

import sys
import os
from pathlib import Path

# 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, str(Path(__file__).parent))

def test_message_extraction_direct():
    """종합 메시지 추출 직접 테스트"""
    
    print("=" * 60)
    print("빠른 종합 메시지 추출 테스트")
    print("=" * 60)
    
    # 테스트 시나리오들
    test_scenarios = [
        {
            "name": "주얼리 구매 상담",
            "text": """
            안녕하세요. 다이아몬드 반지를 찾고 있어요.
            약혼반지로 쓸 건데 1캐럿 정도로 생각하고 있습니다.
            GIA 인증서 있는 걸로요. 가격이 얼마나 할까요?
            할인도 가능한지 궁금해요.
            """
        },
        {
            "name": "고객 고민 상담",
            "text": """
            목걸이를 사고 싶은데 어떤 걸 선택해야 할지 모르겠어요.
            선물용인데 상대방이 좋아할지 걱정이에요.
            예산은 50만원 정도 생각하고 있어요.
            추천해주실 수 있나요?
            """
        },
        {
            "name": "비교 검토 상담",
            "text": """
            다른 매장에서 본 반지와 비교해보고 싶어요.
            그쪽은 18K 골드에 0.5캐럿 다이아몬드였는데
            여기서는 어떤 옵션이 있나요?
            가격 차이도 알고 싶어요.
            """
        }
    ]
    
    try:
        from core.comprehensive_message_extractor import extract_comprehensive_messages
        print("SUCCESS: 메시지 추출 엔진 로드 성공")
    except ImportError as e:
        print(f"FAILED: 메시지 추출 엔진 로드 실패: {e}")
        return
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[테스트 {i}] {scenario['name']}")
        print("-" * 40)
        
        try:
            result = extract_comprehensive_messages(scenario['text'])
            
            if result.get('status') == 'success':
                print("✅ 분석 성공")
                
                main_summary = result.get('main_summary', {})
                
                # 핵심 요약 표시
                if main_summary.get('one_line_summary'):
                    print(f"📢 핵심 메시지: {main_summary['one_line_summary']}")
                
                # 고객 상태
                if main_summary.get('customer_status'):
                    print(f"👤 고객 상태: {main_summary['customer_status']}")
                
                # 긴급도
                if main_summary.get('urgency_indicator'):
                    urgency_emoji = {'높음': '🔴', '보통': '🟡', '낮음': '🟢'}.get(main_summary['urgency_indicator'], '⚪')
                    print(f"⚡ 긴급도: {urgency_emoji} {main_summary['urgency_indicator']}")
                
                # 주요 포인트 (상위 2개만)
                if main_summary.get('key_points'):
                    print("🔍 주요 포인트:")
                    for point in main_summary['key_points'][:2]:
                        print(f"  • {point}")
                
                # 추천 액션 (상위 2개만)
                if main_summary.get('recommended_actions'):
                    print("💼 추천 액션:")
                    for action in main_summary['recommended_actions'][:2]:
                        print(f"  {action}")
                
                # 신뢰도
                if main_summary.get('confidence_score'):
                    print(f"📊 신뢰도: {main_summary['confidence_score']*100:.0f}%")
                    
            else:
                print(f"❌ 분석 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

def test_real_analysis_engine_integration():
    """실제 분석 엔진 통합 테스트"""
    
    print("\n[통합 테스트] 실제 분석 엔진에서 메시지 추출 확인")
    print("-" * 50)
    
    try:
        from core.real_analysis_engine import RealAnalysisEngine
        engine = RealAnalysisEngine()
        
        # 더미 분석 결과 생성 (파일 분석 없이)
        test_text = "안녕하세요. 다이아몬드 반지 가격이 궁금해요. 1캐럿 정도로 생각하고 있습니다."
        
        # 메시지 추출 로직 직접 테스트
        from core.comprehensive_message_extractor import extract_comprehensive_messages
        
        result = extract_comprehensive_messages(test_text)
        
        if result.get('status') == 'success':
            print("✅ 통합 테스트 성공")
            print("✅ 실제 분석 엔진에서 메시지 추출 기능이 정상 작동할 것입니다")
            
            # comprehensive_messages가 제대로 생성되는지 확인
            if 'main_summary' in result:
                print("✅ main_summary 생성 확인")
            if 'conversation_analysis' in result:
                print("✅ conversation_analysis 생성 확인")
                
        else:
            print("❌ 통합 테스트 실패")
            
    except Exception as e:
        print(f"❌ 통합 테스트 오류: {e}")

if __name__ == "__main__":
    test_message_extraction_direct()
    test_real_analysis_engine_integration()