#!/usr/bin/env python3
"""
다국어 처리 모듈 테스트
"""

import os
import sys
sys.path.append('.')

from core.multilingual_processor import multilingual_processor

def test_multilingual_processor():
    """다국어 처리 모듈 테스트"""
    
    print("[INFO] 다국어 처리 모듈 테스트 시작")
    print("=" * 50)
    
    # 설치 가이드 확인
    guide = multilingual_processor.get_installation_guide()
    print("[INFO] 기능 사용 가능성:")
    for feature, available in guide['available_features'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}")
    
    if guide['missing_packages']:
        print("\n[WARNING] 누락된 패키지:")
        for pkg in guide['missing_packages']:
            print(f"  - {pkg['package']}: {pkg['command']} ({pkg['purpose']})")
        print(f"\n[INFO] 전체 설치: {guide['install_all']}")
    
    print()
    
    # 지원 언어 목록
    lang_info = multilingual_processor.get_supported_languages()
    print(f"[INFO] 지원 언어: {lang_info['total_languages']}개")
    for code, info in lang_info['supported_languages'].items():
        print(f"  - {code}: {info['name']} ({info['name_en']}) - {len(info['jewelry_terms'])}개 전문용어")
    
    print()
    
    # 테스트 텍스트들 (다국어)
    test_texts = [
        {
            "text": "이 다이아몬드 반지는 정말 아름답습니다. 1캐럭 다이아몬드가 사용되었어요.",
            "expected_lang": "ko",
            "description": "한국어 주얼리 텍스트"
        },
        {
            "text": "This diamond ring is absolutely beautiful. It features a 1-carat diamond with excellent cut and clarity.",
            "expected_lang": "en", 
            "description": "영어 주얼리 텍스트"
        },
        {
            "text": "这个钻石戒指非常漂亮。它镶嵌了一颗1克拉的钻石，切工和净度都很优秀。",
            "expected_lang": "zh",
            "description": "중국어 주얼리 텍스트"
        },
        {
            "text": "このダイヤモンドリングはとても美しいです。1カラットのダイヤモンドが使用されています。",
            "expected_lang": "ja",
            "description": "일본어 주얼리 텍스트"
        }
    ]
    
    # 언어 감지 테스트
    print("[TEST] 언어 감지 테스트:")
    for i, test_case in enumerate(test_texts, 1):
        print(f"\n[TEST {i}] {test_case['description']}")
        print(f"텍스트: {test_case['text']}")
        
        detection_result = multilingual_processor.detect_language(test_case['text'])
        
        if detection_result['status'] == 'success':
            detected = detection_result['detected_language']
            mapped = detection_result['mapped_language']
            confidence = detection_result['confidence']
            
            print(f"✅ 감지 성공: {detected} -> {mapped} (신뢰도: {confidence:.2f})")
            print(f"예상 언어 일치: {'✅' if mapped == test_case['expected_lang'] else '❌'}")
            
            if detection_result['is_supported']:
                lang_info = detection_result['language_info']
                print(f"지원 언어: {lang_info.get('name', 'N/A')}")
        else:
            print(f"❌ 감지 실패: {detection_result.get('error', 'Unknown')}")
    
    print("\n" + "="*30)
    
    # 키워드 추출 테스트
    print("\n[TEST] 다국어 키워드 추출 테스트:")
    for i, test_case in enumerate(test_texts, 1):
        print(f"\n[TEST {i}] {test_case['description']}")
        
        keyword_result = multilingual_processor.extract_multilingual_jewelry_keywords(
            test_case['text'], 'auto'
        )
        
        if keyword_result['status'] == 'success':
            primary_kw = keyword_result['primary_keywords']
            cross_kw = keyword_result['cross_language_keywords']
            total = keyword_result['total_keywords']
            
            print(f"✅ 키워드 추출 성공: {total}개")
            print(f"주 언어 키워드: {len(primary_kw)}개")
            for kw in primary_kw:
                print(f"  - {kw['keyword']} ({kw['language']}, {kw['category']})")
            
            if cross_kw:
                print(f"타 언어 키워드: {len(cross_kw)}개")
                for kw in cross_kw[:3]:  # 처음 3개만
                    print(f"  - {kw['keyword']} ({kw['language']}, {kw['category']})")
            
            print(f"키워드 밀도: {keyword_result['keyword_density']}%")
            print(f"다국어 텍스트: {'✅' if keyword_result['language_analysis']['is_multilingual'] else '❌'}")
        else:
            print(f"❌ 키워드 추출 실패: {keyword_result.get('error', 'Unknown')}")
    
    print("\n" + "="*30)
    
    # 번역 테스트 (가능한 경우)
    if guide['available_features']['translation']:
        print("\n[TEST] 번역 테스트:")
        
        korean_text = "이 다이아몬드 목걸이는 최고급 품질입니다."
        target_languages = ['en', 'zh', 'ja']
        
        print(f"원본 텍스트 (한국어): {korean_text}")
        
        for target_lang in target_languages:
            lang_name = multilingual_processor.supported_languages[target_lang]['name']
            print(f"\n[번역 -> {target_lang} ({lang_name})]")
            
            translation_result = multilingual_processor.translate_text(
                korean_text, target_lang=target_lang, source_lang='ko'
            )
            
            if translation_result['status'] == 'success':
                translated = translation_result['translated_text']
                print(f"✅ 번역 성공: {translated}")
            else:
                print(f"❌ 번역 실패: {translation_result.get('error', 'Unknown')}")
    
    else:
        print("\n[INFO] 번역 기능을 사용할 수 없습니다.")
        print("설치 명령: pip install googletrans==4.0.0rc1")
    
    print("\n" + "="*50)
    print("[INFO] 다국어 처리 모듈 테스트 완료")
    print()
    print("다음 단계:")
    print("1. Streamlit UI에 언어 선택 기능 추가")
    print("2. 다국어 분석 결과 표시")
    print("3. 실시간 언어 감지 및 번역")

if __name__ == "__main__":
    test_multilingual_processor()