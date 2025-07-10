#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1.1 - 실제 업무 시뮬레이션용 샘플 데이터 세트 생성기
한국보석협회 회원사 베타 테스트용 실제 시나리오 기반 테스트 데이터 구축

작성자: 전근혁 (솔로몬드 대표, 한국보석협회 사무국장)
생성일: 2025.07.11
목적: 현장 테스트 환경 완성 및 검증
"""

import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JewelryTestDataGenerator:
    """주얼리 업계 특화 테스트 데이터 생성기"""
    
    def __init__(self, output_dir="test_data_v211"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 주얼리 업계 실제 시나리오 데이터
        self.company_profiles = self._load_company_profiles()
        self.meeting_scenarios = self._load_meeting_scenarios()
        self.jewelry_terms = self._load_jewelry_terms()
        self.test_requirements = self._load_test_requirements()
        
        logger.info(f"🎯 테스트 데이터 생성기 초기화 완료 - 출력 디렉토리: {self.output_dir}")
    
    def _load_company_profiles(self):
        """한국보석협회 회원사 프로필 (실제 기반 익명화)"""
        return {
            "large_enterprise": [
                {
                    "name": "대원주얼리그룹",
                    "type": "대기업",
                    "specialty": "다이아몬드 도매, 국제무역",
                    "employees": 150,
                    "main_languages": ["한국어", "영어", "중국어"],
                    "typical_meetings": ["국제무역회의", "대규모전시회", "임원회의"],
                    "data_volume": "large",
                    "test_focus": ["성능", "확장성", "다국어지원"]
                },
                {
                    "name": "동양보석",
                    "type": "대기업", 
                    "specialty": "금은 제조, 브랜드 운영",
                    "employees": 200,
                    "main_languages": ["한국어", "영어", "일본어"],
                    "typical_meetings": ["제품기획회의", "브랜드전략회의", "해외진출회의"],
                    "data_volume": "large",
                    "test_focus": ["통합성", "정확도", "실시간분석"]
                }
            ],
            "medium_enterprise": [
                {
                    "name": "한국보석공예",
                    "type": "중견기업",
                    "specialty": "수제 주얼리 제작",
                    "employees": 45,
                    "main_languages": ["한국어", "영어"],
                    "typical_meetings": ["제품개발회의", "고객상담", "직원교육"],
                    "data_volume": "medium",
                    "test_focus": ["사용성", "정확도", "ROI"]
                },
                {
                    "name": "프리미엄젬스",
                    "type": "중견기업",
                    "specialty": "보석 감정, 거래",
                    "employees": 30,
                    "main_languages": ["한국어", "영어"],
                    "typical_meetings": ["감정회의", "시장분석회의", "고객교육"],
                    "data_volume": "medium", 
                    "test_focus": ["전문용어정확도", "문서분석", "효율성"]
                }
            ],
            "small_specialist": [
                {
                    "name": "마스터젬스튜디오",
                    "type": "소규모전문업체",
                    "specialty": "맞춤 주얼리 디자인",
                    "employees": 8,
                    "main_languages": ["한국어"],
                    "typical_meetings": ["고객상담", "디자인검토", "제작논의"],
                    "data_volume": "small",
                    "test_focus": ["모바일활용", "신속성", "전문성"]
                },
                {
                    "name": "다이아몬드전문가",
                    "type": "소규모전문업체", 
                    "specialty": "다이아몬드 전문 감정",
                    "employees": 5,
                    "main_languages": ["한국어", "영어"],
                    "typical_meetings": ["감정서작성", "고객설명", "교육세미나"],
                    "data_volume": "small",
                    "test_focus": ["정확도", "전문용어", "현장활용"]
                }
            ]
        }
    
    def run_complete_test_environment_setup(self):
        """전체 테스트 환경 설정 실행"""
        logger.info("🚀 솔로몬드 AI v2.1.1 테스트 환경 구축 시작")
        
        # 1. 샘플 데이터 생성
        transcript_files = self.generate_sample_meeting_transcripts()
        document_files = self.generate_sample_documents()
        
        # 2. 테스트 설정 파일 생성
        config_file = self.generate_test_scenarios_config()
        
        # 3. 성능 벤치마크 스위트 생성
        benchmark_file = self.generate_performance_benchmark_suite()
        
        # 4. 요약 리포트 생성
        summary = {
            "setup_complete": True,
            "setup_date": datetime.now().isoformat(),
            "generated_files": {
                "transcripts": len(transcript_files),
                "documents": len(document_files),
                "config_files": 1,
                "benchmark_files": 1
            },
            "ready_for_testing": True,
            "next_steps": [
                "베타 테스터 회원사 선정 및 연락",
                "테스트 환경 원격 설치 지원",
                "실제 데이터로 시스템 검증",
                "피드백 수집 시스템 활성화"
            ]
        }
        
        summary_file = self.output_dir / "test_environment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("🎉 테스트 환경 구축 완료!")
        logger.info(f"📁 출력 디렉토리: {self.output_dir.absolute()}")
        logger.info(f"📊 생성된 파일: {summary['generated_files']}")
        
        return summary

if __name__ == "__main__":
    # 테스트 데이터 생성기 실행
    generator = JewelryTestDataGenerator()
    result = generator.run_complete_test_environment_setup()
    
    print("\n" + "="*60)
    print("🎯 솔로몬드 AI v2.1.1 테스트 환경 구축 완료")
    print("="*60)
    print(f"📊 생성된 테스트 데이터: {result['generated_files']}")
    print(f"📁 출력 위치: test_data_v211/")
    print("\n다음 단계:")
    for i, step in enumerate(result['next_steps'], 1):
        print(f"{i}. {step}")
    print("="*60)
