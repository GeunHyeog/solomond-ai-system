#!/usr/bin/env python3
"""
주얼리 특화 STT 시스템 테스트 스크립트
Jewelry-Enhanced STT System Test Script

사용법:
python test_jewelry_stt.py
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_jewelry_enhancer():
    """주얼리 특화 모듈 단독 테스트"""
    print("="*60)
    print("🧪 주얼리 특화 모듈 단독 테스트")
    print("="*60)
    
    try:
        from core.jewelry_enhancer import get_jewelry_enhancer, enhance_jewelry_transcription
        
        # 테스트 텍스트들
        test_texts = [
            "오늘 다이몬드 4씨에 대해 말씀드리겠습니다",
            "새파이어 1캐럿 가격이 궁금합니다",
            "에머랄드 지아이에이 감정서가 있나요",
            "이 루비는 비둘기피 색상입니다",
            "플래티넘 PT950 반지를 찾고 있어요",
            "도매가로 할인 가능한지 문의드립니다"
        ]
        
        enhancer = get_jewelry_enhancer()
        print(f"✅ 주얼리 특화 모듈 로드 성공")
        
        # 개선 엔진 통계 출력
        stats = enhancer.get_enhancement_stats()
        print(f"📊 용어 데이터베이스: {stats['total_terms']}개 용어")
        print(f"🔧 수정 규칙: {stats['correction_rules']}개")
        print(f"📚 지원 카테고리: {len(stats['categories'])}개")
        print()
        
        # 각 테스트 텍스트 처리
        for i, text in enumerate(test_texts, 1):
            print(f"테스트 {i}: {text}")
            
            # 주얼리 특화 처리
            result = enhance_jewelry_transcription(text, "ko", include_analysis=True)
            
            if result["enhanced_text"] != text:
                print(f"  ✨ 개선됨: {result['enhanced_text']}")
            else:
                print(f"  ✅ 수정불필요")
            
            if result["corrections"]:
                print(f"  🔧 수정사항: {len(result['corrections'])}개")
                for correction in result["corrections"][:2]:  # 최대 2개만 표시
                    print(f"     '{correction['original']}' → '{correction['corrected']}'")
            
            if result["detected_terms"]:
                terms_by_category = {}
                for term_info in result["detected_terms"]:
                    category = term_info["category"]
                    if category not in terms_by_category:
                        terms_by_category[category] = []
                    terms_by_category[category].append(term_info["term"])
                
                print(f"  📚 발견된 용어:")
                for category, terms in terms_by_category.items():
                    print(f"     {category}: {', '.join(set(terms))}")
            
            if "summary" in result and result["summary"]:
                print(f"  💡 요약: {result['summary'][:50]}...")
            
            print()
        
        print("✅ 주얼리 특화 모듈 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 주얼리 특화 모듈 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_stt():
    """통합된 STT 시스템 테스트"""
    print("="*60)
    print("🧪 통합 STT 시스템 테스트 (시뮬레이션)")
    print("="*60)
    
    try:
        from core.analyzer import get_analyzer, check_whisper_status, get_jewelry_features_info
        
        # 시스템 상태 확인
        status = check_whisper_status()
        print(f"🎤 Whisper 사용 가능: {status['whisper_available']}")
        print(f"💎 주얼리 특화 기능: {status['jewelry_enhancement_available']}")
        
        if status.get('import_error'):
            print(f"⚠️ {status['import_error']}")
        
        if status.get('jewelry_enhancement_error'):
            print(f"⚠️ {status['jewelry_enhancement_error']}")
        
        # 주얼리 특화 기능 정보
        jewelry_info = get_jewelry_features_info()
        print(f"💎 주얼리 특화 기능 상태: {jewelry_info['available']}")
        
        if jewelry_info['available']:
            features = jewelry_info['features']
            print(f"📚 용어 DB 버전: {features.get('terms_database_version', 'unknown')}")
            print(f"🔧 총 용어 수: {features.get('total_terms', 0)}개")
        
        # 분석기 인스턴스 생성
        analyzer = get_analyzer(enable_jewelry_enhancement=True)
        model_info = analyzer.get_model_info()
        
        print(f"🤖 모델 크기: {model_info['model_size']}")
        print(f"💎 주얼리 특화 모드: {model_info['jewelry_enhancement']}")
        print(f"🌍 지원 언어: {len(model_info['supported_languages'])}개")
        print(f"📁 지원 형식: {', '.join(model_info['supported_formats'])}")
        
        print("✅ 통합 STT 시스템 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 통합 STT 시스템 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_file_processing_simulation():
    """파일 처리 시뮬레이션 테스트"""
    print("="*60)
    print("🧪 파일 처리 시뮬레이션 테스트")
    print("="*60)
    
    try:
        from core.analyzer import get_analyzer
        
        # 시뮬레이션된 STT 결과들 (실제 주얼리 세미나에서 나올 수 있는 내용)
        simulated_results = [
            {
                "filename": "jewelry_seminar_diamond_grading.mp3",
                "transcribed_text": "안녕하세요 오늘은 다이몬드 4씨 등급에 대해 설명드리겠습니다. 지아이에이 감정서를 보시면 컷 컬러 클래리티 캐럿이 표시되어 있습니다.",
                "language": "ko"
            },
            {
                "filename": "ruby_pricing_discussion.wav", 
                "transcribed_text": "미얀마 루비 2캐럿 가격을 문의주셨는데요, 현재 도매가 기준으로 할인 가능합니다. 비둘기피 색상이면 더 높은 가격대입니다.",
                "language": "ko"
            },
            {
                "filename": "international_trade_meeting.m4a",
                "transcribed_text": "FOB 가격으로 제안드리고 통관 절차는 저희가 도와드리겠습니다. 원산지 증명서와 감정서를 준비해 주세요.",
                "language": "ko"
            }
        ]
        
        analyzer = get_analyzer(enable_jewelry_enhancement=True)
        
        for i, sim_result in enumerate(simulated_results, 1):
            print(f"📁 파일 {i}: {sim_result['filename']}")
            print(f"📝 원본 텍스트: {sim_result['transcribed_text']}")
            
            # 주얼리 특화 처리 시뮬레이션
            try:
                from core.jewelry_enhancer import enhance_jewelry_transcription
                
                jewelry_result = enhance_jewelry_transcription(
                    sim_result['transcribed_text'],
                    sim_result['language'],
                    include_analysis=True
                )
                
                print(f"✨ 개선된 텍스트: {jewelry_result['enhanced_text']}")
                
                if jewelry_result['corrections']:
                    print(f"🔧 수정사항 {len(jewelry_result['corrections'])}개:")
                    for correction in jewelry_result['corrections']:
                        print(f"   '{correction['original']}' → '{correction['corrected']}'")
                
                if jewelry_result['detected_terms']:
                    print(f"📚 발견된 주얼리 용어 {len(jewelry_result['detected_terms'])}개:")
                    categories = {}
                    for term in jewelry_result['detected_terms']:
                        cat = term['category']
                        if cat not in categories:
                            categories[cat] = []
                        categories[cat].append(term['term'])
                    
                    for cat, terms in categories.items():
                        print(f"   {cat}: {', '.join(set(terms))}")
                
                if 'analysis' in jewelry_result:
                    analysis = jewelry_result['analysis']
                    if analysis.get('identified_topics'):
                        print(f"🎯 식별된 주제: {', '.join(analysis['identified_topics'])}")
                    
                    if analysis.get('business_insights'):
                        print(f"💡 비즈니스 인사이트: {analysis['business_insights'][0]}")
                
                if jewelry_result.get('summary'):
                    print(f"📄 요약: {jewelry_result['summary']}")
                
            except Exception as e:
                print(f"⚠️ 주얼리 특화 처리 오류: {e}")
            
            print("-" * 40)
        
        print("✅ 파일 처리 시뮬레이션 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 파일 처리 시뮬레이션 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 솔로몬드 주얼리 특화 STT 시스템 테스트")
    print(f"📅 테스트 시작: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 테스트 결과 추적
    test_results = []
    
    # 1. 주얼리 특화 모듈 단독 테스트
    result1 = test_jewelry_enhancer()
    test_results.append(("주얼리 특화 모듈", result1))
    
    # 2. 통합 STT 시스템 테스트
    result2 = test_integrated_stt()
    test_results.append(("통합 STT 시스템", result2))
    
    # 3. 파일 처리 시뮬레이션 테스트
    result3 = asyncio.run(test_file_processing_simulation())
    test_results.append(("파일 처리 시뮬레이션", result3))
    
    # 최종 결과 요약
    print("="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print()
    print(f"전체 결과: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("💎 주얼리 특화 STT 시스템이 정상적으로 작동합니다.")
        
        print()
        print("🚀 다음 단계 권장사항:")
        print("1. 실제 주얼리 세미나 음성 파일로 테스트")
        print("2. UI에서 주얼리 특화 기능 활성화")
        print("3. 전근혁 대표님께 시연 및 피드백 수집")
        print("4. 추가 주얼리 용어 데이터베이스 확장")
        
    else:
        print("⚠️ 일부 테스트에서 문제가 발견되었습니다.")
        print("문제를 해결한 후 다시 테스트해주세요.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
