"""
주얼리 업계 분석 예시
기존 JGA25 컨퍼런스 분석 시스템을 모듈화된 구조로 재구현
"""

import sys
from pathlib import Path

# 모듈 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solomond_ai import SolomondAI
from solomond_ai.processors import FileProcessor, BatchProcessor
from solomond_ai.utils import setup_logger

def main():
    """주얼리 분석 예시 실행"""
    
    # 로깅 설정
    logger = setup_logger("jewelry_analysis", level="INFO")
    logger.info("Starting Jewelry Analysis Example")
    
    # 솔로몬드 AI 초기화 (주얼리 도메인)
    app = SolomondAI(
        domain="jewelry",
        engines=["audio", "image", "video", "text"],
        ui_layout="four_step",
        theme="jewelry"
    )
    
    logger.info(f"Initialized SolomondAI for {app.domain} domain")
    logger.info(f"Available engines: {list(app.engine_instances.keys())}")
    
    # 샘플 파일들 (실제 경로로 변경 필요)
    sample_files = [
        "user_files/conference_audio.wav",
        "user_files/presentation_slides_001.jpg", 
        "user_files/presentation_slides_002.jpg",
        "user_files/demo_video.mov",
        "user_files/conference_notes.txt"
    ]
    
    # 파일 존재 확인 및 필터링
    file_processor = FileProcessor()
    valid_files, invalid_files = file_processor.validate_files(sample_files)
    
    if invalid_files:
        logger.warning(f"Invalid files found: {len(invalid_files)}")
        for invalid in invalid_files:
            logger.warning(f"  - {invalid['file_path']}: {invalid['error']}")
    
    if not valid_files:
        logger.error("No valid files found. Please check file paths.")
        print("📁 Sample files needed:")
        for file in sample_files:
            print(f"  - {file}")
        return
    
    logger.info(f"Valid files: {len(valid_files)}")
    
    # 파일 타입별 분류
    organized_files = file_processor.organize_files_by_type(valid_files)
    
    print("\n📊 File Organization:")
    for file_type, files in organized_files.items():
        if files:
            print(f"  {file_type.upper()}: {len(files)} files")
    
    # 배치 처리 설정
    batch_processor = BatchProcessor(max_workers=2)
    
    # 처리 시간 예측
    time_estimates = batch_processor.estimate_processing_time(organized_files)
    total_estimated_time = sum(time_estimates.values())
    
    print(f"\n⏱️ Estimated Processing Time: {total_estimated_time:.1f} seconds")
    for file_type, estimate in time_estimates.items():
        if estimate > 0:
            print(f"  {file_type.upper()}: {estimate:.1f}s")
    
    # 분석 실행
    print("\n🚀 Starting Analysis...")
    
    try:
        # 멀티모달 분석 실행
        results = app.analyze(valid_files)
        
        print("\n✅ Analysis Completed!")
        
        # 결과 요약
        print("\n📋 Results Summary:")
        for engine_name, engine_results in results.items():
            if engine_name == "integration":
                consistency_score = engine_results.get("consistency_score", 0)
                print(f"  Integration: {consistency_score:.1f}/100 consistency score")
                
                # 공통 키워드
                keywords = engine_results.get("common_keywords", [])
                if keywords:
                    print(f"  Common Keywords: {', '.join(keywords[:5])}")
                
                # 인사이트
                insights = engine_results.get("insights", {})
                total_files = insights.get("total_files_processed", 0)
                print(f"  Total Files Processed: {total_files}")
                
            else:
                successful = sum(1 for r in engine_results if r.get("success", False))
                total = len(engine_results)
                print(f"  {engine_name.upper()}: {successful}/{total} files processed successfully")
        
        # 상세 결과 저장
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"jewelry_analysis_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Detailed results saved: {output_file}")
        
        # 권장사항 출력
        recommendations = results.get("integration", {}).get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
    
    finally:
        # 정리
        file_processor.cleanup_workspace()
        logger.info("Jewelry analysis example completed")

def create_sample_config():
    """주얼리 도메인용 샘플 설정 파일 생성"""
    from solomond_ai.utils import ConfigManager
    
    config_manager = ConfigManager()
    
    # 주얼리 특화 설정
    config_manager.set("project.name", "주얼리 컨퍼런스 분석 시스템")
    config_manager.set("project.domain", "jewelry")
    config_manager.set("ui.theme", "jewelry")
    config_manager.set("ui.title", "JGA25 컨퍼런스 분석")
    config_manager.set("analysis.report_format", "jewelry_standard")
    
    # 설정 파일 저장
    config_manager.save_config("jewelry_config.yaml")
    print("Sample config created: jewelry_config.yaml")

if __name__ == "__main__":
    print("SOLOMOND AI - Jewelry Analysis Example")
    print("=" * 50)
    
    # 샘플 설정 파일 생성 (선택사항)
    create_sample_config()
    
    # 메인 분석 실행
    main()