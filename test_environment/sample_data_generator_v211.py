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
    
    def _load_meeting_scenarios(self):
        """회의 시나리오 데이터 로드"""
        return {
            "international_trade": {
                "title": "홍콩 주얼리쇼 현장 회의",
                "participants": ["한국 바이어", "홍콩 공급업체", "통역사"],
                "languages": ["한국어", "중국어", "영어"],
                "duration": "45분",
                "topics": ["가격 협상", "품질 기준", "납기 일정", "결제 조건"],
                "challenges": ["다국어 환경", "전문 용어", "배경 소음"]
            },
            "product_development": {
                "title": "신제품 개발 회의",
                "participants": ["디자이너", "제작팀", "마케팅팀", "임원"],
                "languages": ["한국어"],
                "duration": "90분",
                "topics": ["트렌드 분석", "원가 계산", "출시 전략", "타겟 고객"],
                "challenges": ["전문 용어", "수치 정확성", "긴 회의 시간"]
            },
            "customer_consultation": {
                "title": "고객 맞춤 주얼리 상담",
                "participants": ["고객", "디자이너", "판매자"],
                "languages": ["한국어", "영어"],
                "duration": "30분",
                "topics": ["디자인 요구사항", "예산", "제작 기간", "A/S 정책"],
                "challenges": ["감정적 표현", "세부 요구사항", "실시간 번역"]
            }
        }
    
    def _load_jewelry_terms(self):
        """주얼리 전문 용어 사전"""
        return {
            "gemstones": {
                "diamond": ["다이아몬드", "diamond", "钻石", "ダイヤモンド"],
                "ruby": ["루비", "ruby", "红宝石", "ルビー"],
                "sapphire": ["사파이어", "sapphire", "蓝宝石", "サファイア"],
                "emerald": ["에메랄드", "emerald", "祖母绿", "エメラルド"]
            },
            "quality_grades": {
                "cut": ["컷", "cut", "切工", "カット"],
                "color": ["색상", "color", "颜色", "カラー"],
                "clarity": ["투명도", "clarity", "净度", "クラリティ"],
                "carat": ["캐럿", "carat", "克拉", "カラット"]
            },
            "metals": {
                "gold": ["금", "gold", "黄金", "ゴールド"],
                "silver": ["은", "silver", "银", "シルバー"],
                "platinum": ["플래티넘", "platinum", "铂金", "プラチナ"]
            }
        }
    
    def _load_test_requirements(self):
        """테스트 요구사항 정의"""
        return {
            "accuracy": {
                "stt_korean": 95.0,
                "stt_english": 90.0,
                "stt_chinese": 85.0,
                "ocr_accuracy": 95.0,
                "term_recognition": 98.0
            },
            "performance": {
                "max_processing_time": 30,  # seconds
                "concurrent_users": 10,
                "file_size_limit": 100,  # MB
                "response_time": 5  # seconds
            },
            "quality": {
                "noise_tolerance": 20,  # dB SNR
                "image_resolution": 1920,
                "audio_quality": 44100,  # Hz
                "compression_ratio": 0.8
            }
        }
    
    def generate_sample_meeting_transcripts(self):
        """샘플 회의 녹취록 생성"""
        transcripts = []
        
        for scenario_key, scenario in self.meeting_scenarios.items():
            transcript = {
                "scenario": scenario_key,
                "title": scenario["title"],
                "timestamp": datetime.now().isoformat(),
                "participants": scenario["participants"],
                "duration": scenario["duration"],
                "languages": scenario["languages"],
                "content": self._generate_realistic_dialogue(scenario),
                "metadata": {
                    "audio_quality": random.uniform(0.8, 0.95),
                    "noise_level": random.uniform(0.1, 0.3),
                    "clarity_score": random.uniform(0.85, 0.98)
                }
            }
            
            filename = f"transcript_{scenario_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
            
            transcripts.append(filepath)
            logger.info(f"📝 생성된 녹취록: {filename}")
        
        return transcripts
    
    def _generate_realistic_dialogue(self, scenario):
        """실제 같은 대화 내용 생성"""
        if scenario["title"] == "홍콩 주얼리쇼 현장 회의":
            return [
                {"speaker": "한국 바이어", "text": "안녕하세요. 다이아몬드 원석 가격을 문의드립니다.", "timestamp": "00:00:10", "language": "korean"},
                {"speaker": "홍콩 공급업체", "text": "Hello, what carat range are you looking for?", "timestamp": "00:00:20", "language": "english"},
                {"speaker": "통역사", "text": "몇 캐럿 범위를 원하시는지 묻고 있습니다.", "timestamp": "00:00:25", "language": "korean"},
                {"speaker": "한국 바이어", "text": "1캐럿에서 3캐럿 사이의 VVS1 등급을 찾고 있습니다.", "timestamp": "00:00:35", "language": "korean"}
            ]
        elif scenario["title"] == "신제품 개발 회의":
            return [
                {"speaker": "디자이너", "text": "올 시즌 트렌드는 미니멀한 디자인이 주목받고 있습니다.", "timestamp": "00:01:00", "language": "korean"},
                {"speaker": "제작팀", "text": "18K 골드로 제작할 경우 원가는 대략 15만원 정도 예상됩니다.", "timestamp": "00:02:30", "language": "korean"},
                {"speaker": "마케팅팀", "text": "타겟 가격대는 30만원 선에서 형성하는 것이 좋겠습니다.", "timestamp": "00:03:45", "language": "korean"}
            ]
        else:  # customer_consultation
            return [
                {"speaker": "고객", "text": "결혼반지를 맞춤 제작하고 싶은데요.", "timestamp": "00:00:05", "language": "korean"},
                {"speaker": "디자이너", "text": "축하드립니다! 어떤 스타일을 원하시는지요?", "timestamp": "00:00:15", "language": "korean"},
                {"speaker": "고객", "text": "클래식하면서도 모던한 느낌이요. 예산은 200만원 정도입니다.", "timestamp": "00:00:30", "language": "korean"}
            ]
    
    def generate_sample_documents(self):
        """샘플 문서 생성"""
        documents = []
        
        doc_types = ["price_list", "certification", "design_specification"]
        
        for doc_type in doc_types:
            document = {
                "type": doc_type,
                "title": self._get_document_title(doc_type),
                "content": self._generate_document_content(doc_type),
                "timestamp": datetime.now().isoformat(),
                "language": "korean",
                "metadata": {
                    "pages": random.randint(1, 5),
                    "ocr_confidence": random.uniform(0.9, 0.99),
                    "image_quality": random.uniform(0.85, 0.95)
                }
            }
            
            filename = f"document_{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            
            documents.append(filepath)
            logger.info(f"📄 생성된 문서: {filename}")
        
        return documents
    
    def _get_document_title(self, doc_type):
        """문서 제목 생성"""
        titles = {
            "price_list": "2025년 상반기 다이아몬드 가격표",
            "certification": "GIA 다이아몬드 감정서",
            "design_specification": "커스텀 웨딩링 디자인 명세서"
        }
        return titles.get(doc_type, "일반 문서")
    
    def _generate_document_content(self, doc_type):
        """문서 내용 생성"""
        if doc_type == "price_list":
            return {
                "header": "다이아몬드 도매 가격표",
                "date": "2025년 7월",
                "items": [
                    {"carat": "1.0", "color": "D", "clarity": "VVS1", "price": "8,500,000원"},
                    {"carat": "1.5", "color": "E", "clarity": "VVS2", "price": "15,200,000원"},
                    {"carat": "2.0", "color": "F", "clarity": "VS1", "price": "22,800,000원"}
                ],
                "notes": "가격은 시장 상황에 따라 변동될 수 있습니다."
            }
        elif doc_type == "certification":
            return {
                "certificate_number": "GIA-2157849630",
                "stone_type": "Natural Diamond",
                "carat_weight": "1.01",
                "color_grade": "E",
                "clarity_grade": "VVS2",
                "cut_grade": "Excellent",
                "measurements": "6.44 x 6.47 x 3.98 mm",
                "issue_date": "2025-07-11"
            }
        else:  # design_specification
            return {
                "design_name": "클래식 솔리테어 웨딩링",
                "metal": "18K 화이트골드",
                "center_stone": "1.0ct 다이아몬드 (E, VVS2)",
                "setting": "6프롱 세팅",
                "ring_size": "13호",
                "estimated_price": "3,200,000원",
                "production_time": "3-4주"
            }
    
    def generate_test_scenarios_config(self):
        """테스트 시나리오 설정 파일 생성"""
        config = {
            "test_version": "v2.1.1",
            "created_date": datetime.now().isoformat(),
            "test_categories": {
                "functional": {
                    "stt_accuracy": {
                        "test_files": ["transcript_international_trade", "transcript_product_development"],
                        "expected_accuracy": 95.0,
                        "languages": ["korean", "english", "chinese"]
                    },
                    "ocr_processing": {
                        "test_files": ["document_price_list", "document_certification"],
                        "expected_accuracy": 95.0,
                        "formats": ["pdf", "image", "scan"]
                    },
                    "multilingual_support": {
                        "test_files": ["transcript_international_trade"],
                        "expected_translation_accuracy": 90.0,
                        "target_language": "korean"
                    }
                },
                "performance": {
                    "processing_speed": {
                        "max_file_size": "100MB",
                        "max_processing_time": "30s",
                        "concurrent_users": 10
                    },
                    "quality_metrics": {
                        "audio_snr_threshold": 20,
                        "image_resolution_min": 1920,
                        "ocr_confidence_min": 0.90
                    }
                },
                "usability": {
                    "user_interface": {
                        "mobile_compatibility": True,
                        "real_time_feedback": True,
                        "quality_indicators": True
                    },
                    "workflow": {
                        "one_click_analysis": True,
                        "batch_processing": True,
                        "export_formats": ["json", "pdf", "docx"]
                    }
                }
            },
            "beta_testers": {
                "target_companies": list(self.company_profiles.keys()),
                "test_duration": "2주",
                "feedback_collection": "daily",
                "success_criteria": {
                    "overall_satisfaction": 4.5,
                    "feature_completeness": 90.0,
                    "performance_rating": 4.0
                }
            }
        }
        
        config_file = self.output_dir / "test_scenarios_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"⚙️ 테스트 설정 파일 생성: {config_file}")
        return config_file
    
    def generate_performance_benchmark_suite(self):
        """성능 벤치마크 스위트 생성"""
        benchmark = {
            "benchmark_version": "v2.1.1",
            "created_date": datetime.now().isoformat(),
            "system_requirements": {
                "minimum": {
                    "cpu": "Intel i5 또는 동급",
                    "memory": "8GB RAM",
                    "storage": "10GB 여유공간",
                    "python": "3.8+"
                },
                "recommended": {
                    "cpu": "Intel i7 또는 동급",
                    "memory": "16GB RAM",
                    "storage": "50GB 여유공간",
                    "python": "3.11+"
                }
            },
            "test_suites": {
                "audio_processing": {
                    "tests": [
                        {
                            "name": "korean_stt_accuracy",
                            "input": "sample_korean_audio.wav",
                            "expected_wer": 0.05,  # Word Error Rate
                            "timeout": 30
                        },
                        {
                            "name": "multilingual_detection",
                            "input": "sample_mixed_language.wav",
                            "expected_accuracy": 0.90,
                            "timeout": 45
                        }
                    ]
                },
                "document_processing": {
                    "tests": [
                        {
                            "name": "jewelry_terms_extraction",
                            "input": "sample_certificate.pdf",
                            "expected_terms": 15,
                            "timeout": 20
                        },
                        {
                            "name": "price_table_ocr",
                            "input": "sample_price_list.jpg",
                            "expected_accuracy": 0.95,
                            "timeout": 25
                        }
                    ]
                },
                "integration": {
                    "tests": [
                        {
                            "name": "end_to_end_analysis",
                            "inputs": ["audio_file", "document_file"],
                            "expected_coherence": 0.85,
                            "timeout": 60
                        }
                    ]
                }
            },
            "performance_targets": self.test_requirements["performance"],
            "quality_targets": self.test_requirements["quality"]
        }
        
        benchmark_file = self.output_dir / "performance_benchmark_suite.json"
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 벤치마크 스위트 생성: {benchmark_file}")
        return benchmark_file
    
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
