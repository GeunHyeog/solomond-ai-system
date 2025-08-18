#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캐시 내용 직접 검사
"""

import pickle
import gzip
from pathlib import Path

def inspect_cache_file(cache_file):
    """캐시 파일 내용 검사"""
    print(f"\n=== {cache_file.name} ===")
    
    try:
        with gzip.open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"타입: {type(data)}")
        
        if isinstance(data, dict):
            print(f"키들: {list(data.keys())}")
            
            # 각 키의 내용 확인
            for key, value in data.items():
                print(f"\n[{key}]: {type(value)}")
                
                if key == 'transcription' and value:
                    if isinstance(value, str):
                        print(f"  텍스트 길이: {len(value)}자")
                        print(f"  내용 미리보기: {value[:200]}...")
                    elif isinstance(value, dict) and 'text' in value:
                        text = value['text']
                        print(f"  텍스트 길이: {len(text)}자")
                        print(f"  내용 미리보기: {text[:200]}...")
                
                elif key == 'speaker_analysis' and value:
                    if isinstance(value, dict):
                        print(f"  화자 분석 키들: {list(value.keys())}")
                        if 'speakers' in value:
                            speakers = value['speakers']
                            print(f"  감지된 화자: {len(speakers)}명")
                            for speaker_id, speaker_data in list(speakers.items())[:3]:
                                if isinstance(speaker_data, dict) and 'segments' in speaker_data:
                                    segments = speaker_data['segments']
                                    print(f"    {speaker_id}: {len(segments)}개 세그먼트")
                
                elif key == 'error':
                    print(f"  오류: {value}")
                
                elif isinstance(value, str) and len(value) > 100:
                    print(f"  길이: {len(value)}자")
                    print(f"  미리보기: {value[:100]}...")
                
                elif isinstance(value, (int, float)):
                    print(f"  값: {value}")
                
                else:
                    print(f"  값: {str(value)[:100]}...")
        
        return data
        
    except Exception as e:
        print(f"오류: {str(e)}")
        return None

def main():
    """메인 실행"""
    print("캐시 파일 내용 검사")
    print("=" * 50)
    
    cache_dir = Path("cache/conference_analysis")
    cache_files = list(cache_dir.glob("audio_*.pkl.gz"))
    
    print(f"캐시 파일 {len(cache_files)}개 발견")
    
    results = {}
    for cache_file in cache_files:
        data = inspect_cache_file(cache_file)
        if data:
            results[cache_file.name] = data
    
    # 전사 텍스트가 있는 파일 찾기
    print(f"\n" + "=" * 50)
    print("전사 텍스트 확인")
    print("=" * 50)
    
    for filename, data in results.items():
        if isinstance(data, dict):
            # transcription 키 확인
            if 'transcription' in data and data['transcription']:
                transcription = data['transcription']
                
                if isinstance(transcription, str) and len(transcription) > 10:
                    print(f"\n[{filename}] - 전사 텍스트 발견!")
                    print(f"길이: {len(transcription)}자")
                    print(f"내용: {transcription[:500]}...")
                
                elif isinstance(transcription, dict) and 'text' in transcription:
                    text = transcription['text']
                    print(f"\n[{filename}] - 전사 텍스트 발견!")
                    print(f"길이: {len(text)}자")
                    print(f"내용: {text[:500]}...")
            
            # speaker_analysis 확인
            if 'speaker_analysis' in data and data['speaker_analysis']:
                speaker_analysis = data['speaker_analysis']
                if isinstance(speaker_analysis, dict) and 'speakers' in speaker_analysis:
                    speakers = speaker_analysis['speakers']
                    print(f"\n[{filename}] - 화자 분석 발견!")
                    print(f"화자 수: {len(speakers)}명")
                    
                    # 각 화자의 대화 내용 확인
                    for speaker_id, speaker_data in list(speakers.items())[:2]:  # 처음 2명만
                        if isinstance(speaker_data, dict) and 'segments' in speaker_data:
                            segments = speaker_data['segments']
                            print(f"\n{speaker_id}: {len(segments)}개 세그먼트")
                            
                            for i, segment in enumerate(segments[:3]):  # 처음 3개만
                                if isinstance(segment, dict) and 'text' in segment:
                                    text = segment['text']
                                    start = segment.get('start_time', segment.get('start', 0))
                                    print(f"  ({start:.1f}초): {text}")

if __name__ == "__main__":
    main()