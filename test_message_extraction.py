#!/usr/bin/env python3
"""
종합 메시지 추출 엔진 테스트
사용자 핵심 요구사항: "이 사람들이 무엇을 말하는지" 명확히 파악
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_comprehensive_message_extraction():
    """종합 메시지 추출 테스트"""
    
    print("🧠 종합 메시지 추출 엔진 테스트 시작")
    
    try:
        from core.comprehensive_message_extractor import extract_comprehensive_messages
        print("✅ 메시지 추출 엔진 임포트 성공")
    except ImportError as e:
        print(f"❌ 메시지 추출 엔진 임포트 실패: {e}")
        return
    
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
    
    for scenario in test_scenarios:
        print(f"\n🎯 테스트: {scenario['name']}")
        print("=" * 50)
        
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
                    print(f"⚡ 긴급도: {main_summary['urgency_indicator']}")
                
                # 주요 포인트
                if main_summary.get('key_points'):
                    print("🔍 주요 포인트:")
                    for point in main_summary['key_points'][:3]:
                        print(f"  • {point}")
                
                # 추천 액션
                if main_summary.get('recommended_actions'):
                    print("💼 추천 액션:")
                    for action in main_summary['recommended_actions']:
                        print(f"  {action}")
                
                # 대화 분석
                conv_analysis = result.get('conversation_analysis', {})
                if conv_analysis.get('intent'):
                    intent_info = conv_analysis['intent']
                    print(f"🎯 대화 의도: {intent_info.get('description', '')}")
                    print(f"📊 신뢰도: {intent_info.get('confidence', 0)*100:.0f}%")
                    
            else:
                print(f"❌ 분석 실패: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    test_comprehensive_message_extraction()