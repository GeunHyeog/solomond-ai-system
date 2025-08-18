#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JGA 2025 D1 파일들 종합 분석 스크립트
실제 컨퍼런스 파일들을 모듈 1로 분석
"""

import sys
import os
import json
from pathlib import Path
import tempfile
import shutil

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "modules" / "module1_conference"))
sys.path.append(str(PROJECT_ROOT / "core"))

def analyze_jga_conference():
    """JGA 2025 컨퍼런스 파일들 종합 분석"""
    
    try:
        from conference_analysis_performance_optimized import PerformanceOptimizedConferenceAnalyzer
        
        print("=== JGA 2025 컨퍼런스 실제 파일 분석 ===")
        
        # 분석기 초기화
        analyzer = PerformanceOptimizedConferenceAnalyzer()
        
        # JGA 파일 경로
        jga_folder = PROJECT_ROOT / "user_files" / "JGA2025_D1"
        
        if not jga_folder.exists():
            return {"error": "JGA2025_D1 폴더를 찾을 수 없습니다."}
        
        # 파일들 스캔
        audio_files = []
        image_files = []
        video_files = []
        
        for file_path in jga_folder.iterdir():
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in ['.wav', '.m4a', '.mp3']:
                    audio_files.append(file_path)
                elif suffix in ['.jpg', '.jpeg', '.png']:
                    image_files.append(file_path)
                elif suffix in ['.mov', '.mp4', '.avi']:
                    video_files.append(file_path)
        
        print(f"발견된 파일들:")
        print(f"  오디오: {len(audio_files)}개")
        print(f"  이미지: {len(image_files)}개") 
        print(f"  비디오: {len(video_files)}개")
        
        # 컨퍼런스 컨텍스트 설정
        context = {
            'conference_name': 'JGA 2025 - The Rise of the Eco-Friendly Luxury Consumer',
            'participants': 'Lianne Ng (Chow Tai Fook), Henry Tse (Ancardi/Nyrelle/JRNE), Pui In Catherine Siu (PICS Fine Jewellery)',
            'date': '2025-06-19',
            'venue': 'Hong Kong Convention and Exhibition Centre',
            'session_time': '2:30pm - 3:30pm',
            'keywords': 'sustainability, eco-friendly luxury, jewellery, ESG, green consumption, luxury brands'
        }
        
        results = {
            'conference_info': context,
            'audio_analysis': [],
            'image_analysis': [],
            'video_analysis': []
        }
        
        # 오디오 파일 분석 (우선순위)
        if audio_files:
            print(f"\n{'='*50}")
            print("🎵 오디오 파일 분석 시작")
            print(f"{'='*50}")
            
            for i, audio_file in enumerate(audio_files[:3], 1):  # 처음 3개만
                print(f"\n[{i}] 분석 중: {audio_file.name}")
                
                try:
                    # 오디오 파일을 임시 위치에 복사 (한글 경로 문제 해결)
                    with tempfile.NamedTemporaryFile(suffix=audio_file.suffix, delete=False) as tmp_file:
                        shutil.copy2(audio_file, tmp_file.name)
                        temp_audio_path = tmp_file.name
                    
                    # 성능 최적화 분석 실행
                    analysis_options = {
                        'speaker_detection': True,
                        'topic_analysis': True, 
                        'sentiment_analysis': True,
                        'summary_generation': True
                    }
                    
                    # 실제 음성 분석 실행
                    audio_result = analyzer._analyze_audio_content(
                        temp_audio_path, 
                        analysis_options,
                        filename=audio_file.name
                    )
                    
                    results['audio_analysis'].append({
                        'filename': audio_file.name,
                        'analysis': audio_result
                    })
                    
                    print(f"  ✅ {audio_file.name} 분석 완료")
                    
                    # 임시 파일 정리
                    os.unlink(temp_audio_path)
                    
                except Exception as e:
                    print(f"  ❌ {audio_file.name} 분석 실패: {str(e)}")
        
        # 결과 저장
        output_file = PROJECT_ROOT / "jga_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*50}")
        print("✅ JGA 2025 컨퍼런스 분석 완료!")
        print(f"📁 결과 저장: {output_file}")
        print(f"{'='*50}")
        
        return {
            "success": True, 
            "results_file": str(output_file),
            "conference_info": context,
            "files_analyzed": {
                "audio": len(results['audio_analysis']),
                "images": len(image_files),
                "videos": len(video_files)
            }
        }
        
    except Exception as e:
        return {"error": f"분석 실패: {str(e)}"}

if __name__ == "__main__":
    result = analyze_jga_conference()
    print(json.dumps(result, ensure_ascii=False, indent=2))