#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "core"))

def analyze_conversations():
    try:
        from comprehensive_message_extractor import ComprehensiveMessageExtractor
        
        # 테스트 데이터 읽기
        jewelry_file = PROJECT_ROOT / "test_data" / "jewelry_consultation.txt"
        tech_file = PROJECT_ROOT / "test_data" / "tech_conference.txt"
        
        test_cases = []
        
        if jewelry_file.exists():
            with open(jewelry_file, 'r', encoding='utf-8') as f:
                test_cases.append({
                    'title': '주얼리 매장 고객 상담',
                    'content': f.read(),
                    'context': {
                        'participants': '상담사 김미영, 고객 박지영',
                        'conference_name': '브라이덜 주얼리 상담',
                        'situation': '매장 상담',
                        'keywords': '결혼반지, 다이아몬드, 1캐럿, 브라이덜'
                    }
                })
        
        if tech_file.exists():
            with open(tech_file, 'r', encoding='utf-8') as f:
                test_cases.append({
                    'title': 'AI 기술 컨퍼런스',
                    'content': f.read(),
                    'context': {
                        'participants': '발표자 이영수, 청중1 박민수, 청중2 김하늘',
                        'conference_name': 'AI 기술의 미래와 현재',
                        'situation': '기술 컨퍼런스',
                        'keywords': 'AI, GPT, RAG, 기술동향'
                    }
                })
        
        if not test_cases:
            return {"error": "테스트 데이터 파일을 찾을 수 없습니다."}
            
        # 종합 메시지 추출기 초기화
        extractor = ComprehensiveMessageExtractor()
        
        results = []
        
        for case in test_cases:
            # 종합 분석 실행
            result = extractor.extract_key_messages(case['content'], case['context'])
            
            # 결과 정리
            analysis_result = {
                'title': case['title'],
                'context': case['context'],
                'analysis': result
            }
            
            results.append(analysis_result)
        
        # 결과를 JSON 파일로 저장
        output_file = PROJECT_ROOT / "analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return {"success": True, "results_count": len(results), "output_file": str(output_file)}
        
    except Exception as e:
        return {"error": f"분석 실패: {str(e)}"}

if __name__ == "__main__":
    result = analyze_conversations()
    print(json.dumps(result, ensure_ascii=False, indent=2))