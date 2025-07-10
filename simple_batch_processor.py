"""
간단한 배치 처리 스크립트
실제 작동하는 하이브리드 LLM 시스템으로 여러 파일을 순차 처리

사용법:
1. files/ 폴더에 처리할 파일들을 넣기
2. python simple_batch_processor.py 실행
"""

import os
import asyncio
import glob
from pathlib import Path
import time
import json

def create_files_directory():
    """files 디렉토리 생성"""
    files_dir = Path("files")
    files_dir.mkdir(exist_ok=True)
    print(f"📁 파일 디렉토리 준비: {files_dir.absolute()}")
    return files_dir

def find_files_to_process():
    """처리할 파일들 찾기"""
    files_dir = create_files_directory()
    
    # 지원하는 파일 형식
    supported_extensions = [
        "*.mp3", "*.wav", "*.m4a", "*.mp4", "*.mov", "*.avi",
        "*.jpg", "*.jpeg", "*.png", "*.gif",
        "*.pdf", "*.docx", "*.txt"
    ]
    
    all_files = []
    for ext in supported_extensions:
        pattern = files_dir / ext
        files = glob.glob(str(pattern))
        all_files.extend(files)
    
    if not all_files:
        print("⚠️ files/ 폴더에 처리할 파일이 없습니다.")
        print("지원 형식: MP3, WAV, M4A, MP4, MOV, AVI, JPG, PNG, PDF, DOCX, TXT")
        return []
    
    print(f"📋 처리할 파일 {len(all_files)}개 발견:")
    for i, file_path in enumerate(all_files, 1):
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"   {i}. {Path(file_path).name} ({file_size:.2f}MB)")
    
    return all_files

async def process_single_file_with_hybrid_llm(file_path):
    """하이브리드 LLM으로 단일 파일 처리"""
    
    file_name = Path(file_path).name
    file_size = os.path.getsize(file_path) / (1024*1024)
    
    print(f"\n🔄 처리 시작: {file_name} ({file_size:.2f}MB)")
    start_time = time.time()
    
    try:
        # 하이브리드 LLM 매니저 import
        from core.hybrid_llm_manager import HybridLLMManager
        
        manager = HybridLLMManager()
        
        # 파일 타입별 데이터 준비
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.mp3', '.wav', '.m4a']:
            input_data = {
                'audio': file_path,
                'text': f'{file_name} 음성 파일 분석',
                'context': '주얼리 업계 배치 분석'
            }
            analysis_type = 'audio_jewelry_analysis'
            
        elif file_ext in ['.mp4', '.mov', '.avi']:
            input_data = {
                'video': file_path,
                'text': f'{file_name} 동영상 파일 분석',
                'context': '주얼리 업계 배치 분석'
            }
            analysis_type = 'video_jewelry_analysis'
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
            input_data = {
                'image': file_path,
                'text': f'{file_name} 이미지 파일 분석',
                'context': '주얼리 업계 배치 분석'
            }
            analysis_type = 'image_jewelry_analysis'
            
        else:
            input_data = {
                'document': file_path,
                'text': f'{file_name} 문서 파일 분석',
                'context': '주얼리 업계 배치 분석'
            }
            analysis_type = 'document_jewelry_analysis'
        
        # 하이브리드 LLM 분석 실행
        result = await manager.analyze_with_best_model(input_data, analysis_type)
        
        processing_time = time.time() - start_time
        
        # 결과 정리
        analysis_result = {
            'file_name': file_name,
            'file_size_mb': file_size,
            'processing_time': processing_time,
            'selected_model': result.model_type.value,
            'confidence': result.confidence,
            'jewelry_relevance': result.jewelry_relevance,
            'content': result.content,
            'token_usage': result.token_usage,
            'cost': result.cost
        }
        
        print(f"✅ 완료: {file_name}")
        print(f"   🤖 선택된 모델: {result.model_type.value}")
        print(f"   📊 신뢰도: {result.confidence:.2f}")
        print(f"   💎 주얼리 관련성: {result.jewelry_relevance:.2f}")
        print(f"   ⏱️ 처리시간: {processing_time:.2f}초")
        print(f"   📝 분석 내용: {result.content[:100]}...")
        
        return analysis_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_result = {
            'file_name': file_name,
            'file_size_mb': file_size,
            'processing_time': processing_time,
            'error': str(e),
            'status': 'failed'
        }
        
        print(f"❌ 실패: {file_name}")
        print(f"   오류: {str(e)}")
        
        return error_result

def save_results_to_file(results):
    """결과를 파일로 저장"""
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # JSON 결과 저장
    json_file = results_dir / f"batch_analysis_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 텍스트 리포트 저장
    report_file = results_dir / f"batch_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("💎 솔로몬드 하이브리드 LLM 배치 분석 리포트\n")
        f.write("=" * 60 + "\n\n")
        
        successful_files = [r for r in results['individual_results'] if 'error' not in r]
        failed_files = [r for r in results['individual_results'] if 'error' in r]
        
        f.write(f"📊 전체 통계:\n")
        f.write(f"   - 총 파일 수: {len(results['individual_results'])}\n")
        f.write(f"   - 성공: {len(successful_files)}개\n")
        f.write(f"   - 실패: {len(failed_files)}개\n")
        f.write(f"   - 전체 처리시간: {results['total_processing_time']:.2f}초\n\n")
        
        if successful_files:
            f.write("✅ 성공한 파일들:\n")
            for result in successful_files:
                f.write(f"\n📁 {result['file_name']}\n")
                f.write(f"   🤖 모델: {result.get('selected_model', 'N/A')}\n")
                f.write(f"   📊 신뢰도: {result.get('confidence', 0):.2f}\n")
                f.write(f"   💎 주얼리 관련성: {result.get('jewelry_relevance', 0):.2f}\n")
                f.write(f"   ⏱️ 처리시간: {result['processing_time']:.2f}초\n")
                f.write(f"   📝 분석: {result.get('content', 'N/A')[:200]}...\n")
        
        if failed_files:
            f.write("\n❌ 실패한 파일들:\n")
            for result in failed_files:
                f.write(f"\n📁 {result['file_name']}\n")
                f.write(f"   오류: {result['error']}\n")
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   📄 JSON: {json_file}")
    print(f"   📋 리포트: {report_file}")

async def run_batch_processing():
    """배치 처리 메인 함수"""
    
    print("🚀 솔로몬드 하이브리드 LLM 배치 처리 시스템")
    print("=" * 60)
    
    # 처리할 파일 찾기
    files_to_process = find_files_to_process()
    
    if not files_to_process:
        print("\n📁 files/ 폴더에 처리할 파일을 넣고 다시 실행해주세요.")
        return
    
    print(f"\n🔄 {len(files_to_process)}개 파일 순차 처리 시작...")
    
    start_time = time.time()
    results = []
    
    # 각 파일을 순차적으로 처리
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n📋 진행률: {i}/{len(files_to_process)}")
        
        result = await process_single_file_with_hybrid_llm(file_path)
        results.append(result)
        
        # 처리 간 잠시 대기 (시스템 안정성)
        if i < len(files_to_process):
            await asyncio.sleep(1)
    
    total_time = time.time() - start_time
    
    # 전체 결과 정리
    batch_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_files': len(files_to_process),
        'total_processing_time': total_time,
        'individual_results': results
    }
    
    # 결과 저장
    save_results_to_file(batch_results)
    
    # 최종 요약
    successful_files = [r for r in results if 'error' not in r]
    failed_files = [r for r in results if 'error' in r]
    
    print(f"\n" + "=" * 60)
    print(f"🎉 배치 처리 완료!")
    print(f"   📊 총 {len(files_to_process)}개 파일 처리")
    print(f"   ✅ 성공: {len(successful_files)}개")
    print(f"   ❌ 실패: {len(failed_files)}개")
    print(f"   ⏱️ 전체 시간: {total_time:.2f}초")
    print(f"   📈 평균 시간: {total_time/len(files_to_process):.2f}초/파일")
    
    if successful_files:
        avg_confidence = sum(r.get('confidence', 0) for r in successful_files) / len(successful_files)
        avg_jewelry_relevance = sum(r.get('jewelry_relevance', 0) for r in successful_files) / len(successful_files)
        print(f"   📊 평균 신뢰도: {avg_confidence:.2f}")
        print(f"   💎 평균 주얼리 관련성: {avg_jewelry_relevance:.2f}")

def main():
    """메인 함수"""
    
    # files 디렉토리 생성
    files_dir = create_files_directory()
    
    print("💡 사용법:")
    print(f"1. {files_dir} 폴더에 분석할 파일들을 넣어주세요")
    print("2. 지원 형식: MP3, WAV, M4A, MP4, MOV, AVI, JPG, PNG, PDF, DOCX, TXT")
    print("3. 이 스크립트를 다시 실행하세요")
    print()
    
    # 현재 파일 확인
    files_to_process = find_files_to_process()
    
    if files_to_process:
        response = input("🔄 지금 처리를 시작하시겠습니까? (y/n): ")
        if response.lower() in ['y', 'yes', '예', 'ㅇ']:
            asyncio.run(run_batch_processing())
        else:
            print("⏸️ 처리를 취소했습니다.")
    else:
        print(f"⏳ 파일을 {files_dir} 폴더에 넣고 다시 실행해주세요.")

if __name__ == "__main__":
    main()
