import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

# 진짜 분석 테스트
test_cases = [
    "안녕하세요. 다이아몬드 반지 가격이 궁금해요. 1캐럿으로 생각하고 있습니다.",
    "오늘 날씨가 정말 좋네요. 커피 한 잔 마시면서 산책하고 싶어요.",
    "파이썬 프로그래밍을 배우고 있어요. 데이터 분석에 관심이 많습니다."
]

try:
    from core.comprehensive_message_extractor import extract_comprehensive_messages
    print("SUCCESS: 메시지 추출 엔진 로드 성공")
    
    results = []
    for i, text in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {text[:30]}...")
        result = extract_comprehensive_messages(text)
        if result.get("status") == "success":
            summary = result.get("main_summary", {}).get("one_line_summary", "No summary")
            print(f"결과: {summary}")
            results.append(summary)
        else:
            results.append("FAILED")
    
    print(f"\n결과 분석:")
    print(f"총 {len(set(results))}개의 서로 다른 결과")
    if len(set(results)) > 1:
        print("SUCCESS: 진짜 분석 중\!")
    else:
        print("SUSPICIOUS: 가짜 분석 가능성\!")
        
except Exception as e:
    print(f"ERROR: {e}")

