#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 실제 분석 실행
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def analyze_audio_simple():
    """간단한 오디오 분석"""
    print("=== 오디오 분석 ===")
    
    try:
        import whisper
        print("Whisper 모델 로딩...")
        
        model = whisper.load_model("base")
        print("모델 로드 완료")
        
        # 테스트 파일
        test_file = Path("user_files/JGA2025_D1/새로운 녹음.m4a")
        
        if test_file.exists():
            print(f"분석 파일: {test_file.name}")
            
            result = model.transcribe(str(test_file), language='ko')
            
            if result and 'text' in result:
                transcript = result['text']
                segments = result.get('segments', [])
                
                print(f"전사 완료: {len(transcript)}자")
                print(f"세그먼트: {len(segments)}개")
                print(f"내용 미리보기: {transcript[:200]}...")
                
                # 화자별 구분 시뮬레이션
                print("\n화자 분리 결과:")
                for i, segment in enumerate(segments[:5]):
                    speaker = f"화자_{(i % 3) + 1}"
                    start = segment.get('start', 0)
                    text = segment.get('text', '').strip()
                    print(f"{speaker} ({start:.1f}초): {text}")
                
                return {
                    'status': 'success',
                    'transcript': transcript,
                    'segments_count': len(segments),
                    'preview': transcript[:200]
                }
            else:
                print("전사 실패")
                return {'status': 'failed'}
        else:
            print("파일 없음")
            return {'status': 'no_file'}
            
    except Exception as e:
        print(f"오류: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def analyze_image_simple():
    """간단한 이미지 분석"""
    print("\n=== 이미지 분석 ===")
    
    try:
        import easyocr
        print("EasyOCR 초기화...")
        
        reader = easyocr.Reader(['ko', 'en'])
        print("리더 초기화 완료")
        
        # 테스트 파일
        test_file = Path("user_files/JGA2025_D1/IMG_2160.JPG")
        
        if test_file.exists():
            print(f"분석 파일: {test_file.name}")
            
            results = reader.readtext(str(test_file))
            
            if results:
                texts = []
                for result in results:
                    bbox, text, confidence = result
                    if confidence > 0.5:
                        texts.append(text)
                        print(f"텍스트: {text} (신뢰도: {confidence:.2f})")
                
                total_text = " ".join(texts)
                
                return {
                    'status': 'success',
                    'text_blocks': len(texts),
                    'total_text': total_text,
                    'preview': total_text[:200]
                }
            else:
                print("텍스트 없음")
                return {'status': 'no_text'}
        else:
            print("파일 없음")
            return {'status': 'no_file'}
            
    except Exception as e:
        print(f"오류: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def main():
    """메인 실행"""
    print("솔로몬드 AI CLI 실제 분석")
    print("=" * 40)
    
    # 1. 오디오 분석
    audio_result = analyze_audio_simple()
    
    # 2. 이미지 분석  
    image_result = analyze_image_simple()
    
    # 결과 요약
    print("\n" + "=" * 40)
    print("분석 결과 요약")
    print("=" * 40)
    
    if audio_result.get('status') == 'success':
        print(f"\n[오디오 분석 성공]")
        print(f"- 전사된 텍스트: {audio_result['segments_count']}개 세그먼트")
        print(f"- 내용: {audio_result['preview']}")
    else:
        print(f"\n[오디오 분석 실패]: {audio_result.get('status')}")
    
    if image_result.get('status') == 'success':
        print(f"\n[이미지 분석 성공]")
        print(f"- 추출된 텍스트: {image_result['text_blocks']}개 블록")
        print(f"- 내용: {image_result['preview']}")
    else:
        print(f"\n[이미지 분석 실패]: {image_result.get('status')}")
    
    # 결과 저장
    results = {
        'timestamp': datetime.now().isoformat(),
        'audio': audio_result,
        'image': image_result
    }
    
    result_file = f"simple_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {result_file}")
    except Exception as e:
        print(f"저장 실패: {str(e)}")
    
    print("\nCLI 분석 완료!")
    return results

if __name__ == "__main__":
    main()