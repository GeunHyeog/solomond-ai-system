"""
의료 분야 분석 예시
새로운 도메인에서 솔로몬드 AI 모듈 활용 방법 시연
"""

import sys
from pathlib import Path

# 모듈 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solomond_ai import SolomondAI
from solomond_ai.utils import ConfigManager, setup_logger

def main():
    """의료 분석 예시 실행"""
    
    # 의료 도메인 전용 로거 설정
    logger = setup_logger("medical_analysis", level="INFO")
    logger.info("Starting Medical Conference Analysis")
    
    # 의료 도메인 설정 파일 생성 (YAML)
    config_path = "medical_config.yaml"
    create_medical_config(config_path)
    
    # 설정 파일로부터 솔로몬드 AI 초기화
    app = SolomondAI.from_config(config_path)
    
    logger.info(f"Initialized Medical Analysis System")
    logger.info(f"Domain: {app.domain}")
    logger.info(f"Theme: {app.theme}")
    
    # 의료 도메인 특화 키워드 설정
    medical_keywords = [
        "환자", "진단", "치료", "수술", "의약품", "임상시험",
        "증상", "질병", "병원", "의사", "간호사", "처방",
        "patient", "diagnosis", "treatment", "surgery", "medication",
        "clinical", "symptom", "disease", "hospital", "doctor"
    ]
    
    print("\n🏥 Medical Analysis System Initialized")
    print(f"📋 Target Keywords: {', '.join(medical_keywords[:10])}...")
    
    # 의료 컨퍼런스 샘플 파일들 (가상)
    sample_medical_files = [
        "medical_conference_2025/keynote_speech.wav",           # 기조연설
        "medical_conference_2025/surgical_procedure.mp4",      # 수술 영상
        "medical_conference_2025/research_slides_01.jpg",      # 연구 슬라이드
        "medical_conference_2025/research_slides_02.jpg", 
        "medical_conference_2025/clinical_trial_results.pdf",  # 임상시험 결과
        "medical_conference_2025/patient_case_study.txt"       # 환자 사례
    ]
    
    print(f"\n📁 Target Files for Analysis:")
    for i, file in enumerate(sample_medical_files, 1):
        print(f"  {i}. {file}")
    
    # 의료 도메인 특화 분석 시뮬레이션
    print(f"\n🔬 Simulating Medical Analysis...")
    
    # 각 엔진별 의료 특화 분석 시뮬레이션
    simulate_medical_analysis(app, sample_medical_files, medical_keywords)
    
    print(f"\n📊 Medical Analysis Features:")
    print(f"  ✅ Medical terminology recognition")
    print(f"  ✅ Clinical data extraction")
    print(f"  ✅ Patient privacy protection")
    print(f"  ✅ HIPAA compliance ready")
    print(f"  ✅ Medical image analysis")
    print(f"  ✅ Surgical video processing")
    
    logger.info("Medical analysis example completed successfully")

def create_medical_config(config_path: str):
    """의료 도메인 설정 파일 생성"""
    
    medical_config = {
        "project": {
            "name": "의료 컨퍼런스 분석 시스템",
            "domain": "medical",
            "version": "1.0.0"
        },
        "engines": {
            "audio": {
                "model": "whisper-small",  # 의료 용어 인식을 위해 더 큰 모델
                "language": "ko",
                "enabled": True,
                "medical_terms_boost": True  # 의료 용어 강화
            },
            "image": {
                "ocr_engine": "easyocr",
                "languages": ["ko", "en"],
                "enabled": True,
                "medical_image_mode": True  # 의료 이미지 특화 모드
            },
            "video": {
                "sample_frames": 10,  # 수술 영상은 더 많은 프레임
                "enabled": True,
                "surgical_video_detection": True
            },
            "text": {
                "language": "ko",
                "use_transformers": True,
                "enabled": True,
                "medical_ner": True  # 의료 개체명 인식
            }
        },
        "ui": {
            "layout": "four_step",
            "theme": "medical",
            "title": "의료 컨퍼런스 AI 분석 시스템"
        },
        "analysis": {
            "cross_validation": True,
            "confidence_threshold": 0.8,  # 의료 분야는 높은 신뢰도 필요
            "report_format": "medical_standard",
            "privacy_protection": True,  # 환자 정보 보호
            "hipaa_compliance": True
        },
        "processing": {
            "max_workers": 2,  # 의료 데이터는 신중한 처리
            "timeout_seconds": 600,  # 긴 수술 영상 고려
            "memory_limit_mb": 4096
        },
        "medical_specific": {
            "terminology_database": "medical_terms_ko.json",
            "anatomical_structure_detection": True,
            "medication_recognition": True,
            "diagnosis_code_extraction": True,
            "patient_data_anonymization": True
        }
    }
    
    import yaml
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(medical_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Medical config created: {config_path}")

def simulate_medical_analysis(app: SolomondAI, files: list, keywords: list):
    """의료 분석 시뮬레이션"""
    
    print(f"\n🎯 Medical Domain Analysis Simulation")
    print(f"─" * 50)
    
    # 각 파일별 시뮬레이션 결과
    simulation_results = {
        "keynote_speech.wav": {
            "engine": "audio",
            "duration": "45분 23초",
            "detected_terms": ["임상시험", "신약개발", "환자안전", "부작용"],
            "speaker_count": 1,
            "medical_accuracy": "94%"
        },
        "surgical_procedure.mp4": {
            "engine": "video", 
            "duration": "2시간 15분",
            "detected_procedures": ["복강경수술", "미세침습", "봉합"],
            "key_frames": 127,
            "surgical_phase_detection": "성공"
        },
        "research_slides_01.jpg": {
            "engine": "image",
            "extracted_text": "Phase III 임상시험 결과 분석",
            "medical_charts": 3,
            "data_tables": 2,
            "ocr_confidence": "91%"
        },
        "clinical_trial_results.pdf": {
            "engine": "text",
            "pages": 47,
            "patient_count": "1,247명",
            "statistical_data": "p<0.001 유의성 확인",
            "medical_terminology": 156
        }
    }
    
    # 시뮬레이션 결과 출력
    for filename, result in simulation_results.items():
        if any(f in filename for f in files):
            engine = result["engine"].upper()
            print(f"\n📄 {filename}")
            print(f"   Engine: {engine}")
            
            for key, value in result.items():
                if key != "engine":
                    print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # 통합 분석 결과
    print(f"\n🔬 Medical Integration Analysis")
    print(f"   Consistency Score: 87.3/100")
    print(f"   Medical Terms Detected: 234개")
    print(f"   Patient Safety Compliance: ✅ 통과")
    print(f"   Privacy Protection: ✅ 적용됨")
    
    # 의료진을 위한 권장사항
    recommendations = [
        "🩺 진단 정확도 향상을 위한 추가 검사 권장",
        "💊 약물 상호작용 검토 필요",
        "📊 임상데이터 통계 분석 보완 요구",
        "🏥 병원 내 프로토콜 업데이트 제안"
    ]
    
    print(f"\n💡 Medical Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")

if __name__ == "__main__":
    print("🏥 솔로몬드 AI - 의료 분석 예시")
    print("=" * 50)
    
    main()