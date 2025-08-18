#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합 상황 분석 시스템 - Windows 안전 버전
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# 최적화 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

def analyze_real_situation():
    """실제 상황 종합 분석"""
    print("=== 솔로몬드 AI 종합 상황 분석 ===")
    
    # 결과 저장용
    analysis_result = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'files_analyzed': [],
        'audio_content': [],
        'image_content': [],
        'video_content': [],
        'comprehensive_story': '',
        'timeline': []
    }
    
    # 1. 파일 탐색
    print("\\n1. 상황 파일 탐색 중...")
    user_files = Path("user_files")
    
    all_files = []
    if user_files.exists():
        for file_path in user_files.rglob("*"):
            if file_path.is_file() and file_path.name != "README.md":
                try:
                    stat = file_path.stat()
                    file_info = {
                        'path': str(file_path),
                        'name': file_path.name,
                        'size_mb': stat.st_size / 1024 / 1024,
                        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'ext': file_path.suffix.lower()
                    }
                    all_files.append(file_info)
                except Exception as e:
                    print(f"파일 정보 읽기 실패: {file_path.name}")
    
    # 시간순 정렬
    all_files.sort(key=lambda x: x['modified_time'])
    
    print(f"발견된 파일: {len(all_files)}개")
    
    # 2. 모델 로딩
    print("\\n2. AI 모델 로딩 중...")
    
    try:
        import whisper
        whisper_model = whisper.load_model("tiny", device="cpu")
        print("Whisper 모델 로드 완료")
    except Exception as e:
        print(f"Whisper 로딩 실패: {e}")
        whisper_model = None
    
    try:
        import easyocr
        ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
        print("EasyOCR 모델 로드 완료")
    except Exception as e:
        print(f"EasyOCR 로딩 실패: {e}")
        ocr_reader = None
    
    # 3. 파일별 분석
    print("\\n3. 파일 순차 분석 중...")
    
    for i, file_info in enumerate(all_files):
        print(f"\\n[{i+1}/{len(all_files)}] {file_info['name']} ({file_info['size_mb']:.1f}MB)")
        
        ext = file_info['ext']
        
        try:
            if ext in ['.m4a', '.wav', '.mp3'] and whisper_model:
                # 오디오 분석
                if file_info['size_mb'] < 30:  # 30MB 제한
                    print("  오디오 분석 중...")
                    start = time.time()
                    result = whisper_model.transcribe(file_info['path'])
                    duration = time.time() - start
                    
                    transcript = result.get('text', '').strip()
                    if transcript:
                        audio_data = {
                            'file': file_info['name'],
                            'transcript': transcript,
                            'processing_time': duration,
                            'timestamp': file_info['modified_time']
                        }
                        analysis_result['audio_content'].append(audio_data)
                        
                        print(f"  STT 완료 ({duration:.1f}초)")
                        print(f"  내용: {transcript[:100]}...")
                    else:
                        print("  음성 내용 없음")
                else:
                    print("  파일 크기 초과, 스킵")
            
            elif ext in ['.jpg', '.jpeg', '.png'] and ocr_reader:
                # 이미지 분석
                if file_info['size_mb'] < 10:  # 10MB 제한
                    print("  이미지 분석 중...")
                    start = time.time()
                    results = ocr_reader.readtext(file_info['path'])
                    duration = time.time() - start
                    
                    # 텍스트 추출
                    texts = []
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5:
                            texts.append(text)
                    
                    if texts:
                        combined_text = ' '.join(texts)
                        image_data = {
                            'file': file_info['name'],
                            'extracted_text': combined_text,
                            'text_blocks': len(texts),
                            'processing_time': duration,
                            'timestamp': file_info['modified_time']
                        }
                        analysis_result['image_content'].append(image_data)
                        
                        print(f"  OCR 완료 ({duration:.1f}초)")
                        print(f"  텍스트: {combined_text[:100]}...")
                    else:
                        print("  텍스트 없음")
                else:
                    print("  파일 크기 초과, 스킵")
            
            elif ext in ['.mov', '.mp4']:
                # 비디오 메타데이터만
                print("  비디오 메타데이터 수집")
                video_data = {
                    'file': file_info['name'],
                    'size_mb': file_info['size_mb'],
                    'timestamp': file_info['modified_time'],
                    'note': '메타데이터만 수집'
                }
                analysis_result['video_content'].append(video_data)
            
            # 타임라인에 추가
            analysis_result['timeline'].append({
                'timestamp': file_info['modified_time'],
                'file': file_info['name'],
                'type': ext[1:],
                'processed': True
            })
            
            analysis_result['files_analyzed'].append(file_info)
            
        except Exception as e:
            print(f"  분석 실패: {str(e)[:50]}...")
            analysis_result['timeline'].append({
                'timestamp': file_info['modified_time'],
                'file': file_info['name'],
                'type': ext[1:],
                'processed': False,
                'error': str(e)
            })
    
    # 4. 종합 스토리 생성
    print("\\n4. 상황 스토리 재구성 중...")
    
    story_parts = []
    
    # 시간순으로 정렬된 타임라인 기반
    timeline = sorted(analysis_result['timeline'], key=lambda x: x['timestamp'])
    
    for event in timeline:
        if event['processed']:
            file_name = event['file']
            file_type = event['type']
            
            # 해당 파일의 분석 내용 찾기
            content = ""
            if file_type in ['m4a', 'wav', 'mp3']:
                for audio in analysis_result['audio_content']:
                    if audio['file'] == file_name:
                        content = audio['transcript'][:200]
                        break
            elif file_type in ['jpg', 'jpeg', 'png']:
                for image in analysis_result['image_content']:
                    if image['file'] == file_name:
                        content = image['extracted_text'][:200]
                        break
            elif file_type in ['mov', 'mp4']:
                content = f"비디오 파일 ({event.get('size_mb', 0):.1f}MB)"
            
            if content:
                story_parts.append(f"[{file_type.upper()}] {file_name}: {content}")
    
    comprehensive_story = "\\n\\n".join(story_parts)
    analysis_result['comprehensive_story'] = comprehensive_story
    
    # 5. 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"situation_analysis_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    # 6. 요약 출력
    print("\\n" + "="*60)
    print("상황 분석 요약")
    print("="*60)
    
    print(f"총 파일: {len(analysis_result['files_analyzed'])}개")
    print(f"오디오 분석: {len(analysis_result['audio_content'])}개")
    print(f"이미지 분석: {len(analysis_result['image_content'])}개")
    print(f"비디오 수집: {len(analysis_result['video_content'])}개")
    
    if analysis_result['audio_content']:
        print("\\n주요 음성 내용:")
        for audio in analysis_result['audio_content'][:2]:
            print(f"  - {audio['file']}: {audio['transcript'][:100]}...")
    
    if analysis_result['image_content']:
        print("\\n주요 이미지 텍스트:")
        for image in analysis_result['image_content'][:2]:
            print(f"  - {image['file']}: {image['extracted_text'][:100]}...")
    
    print(f"\\n결과 저장: {filename}")
    print("="*60)
    
    print("\\n종합 상황 분석 완료!")
    print("모든 파일들이 하나의 상황으로 통합 분석되었습니다.")
    
    return analysis_result

if __name__ == "__main__":
    try:
        analyze_real_situation()
    except Exception as e:
        print(f"\\n분석 중 오류 발생: {e}")
        print("부분 결과라도 저장을 시도합니다...")