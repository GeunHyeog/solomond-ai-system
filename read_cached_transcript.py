#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
캐시된 전사 결과 읽기
"""

import os
import pickle
import gzip
from pathlib import Path
import json
from datetime import datetime

def read_cached_analysis():
    """캐시된 분석 결과 읽기"""
    print("캐시된 분석 결과 확인 중...")
    
    cache_dir = Path("cache/conference_analysis")
    cache_files = list(cache_dir.glob("audio_*.pkl.gz"))
    
    print(f"캐시 파일 {len(cache_files)}개 발견")
    
    all_results = {}
    
    for cache_file in cache_files:
        print(f"\n캐시 읽는 중: {cache_file.name}")
        
        try:
            with gzip.open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            print(f"캐시 데이터 로드 성공")
            
            # 캐시 데이터 구조 확인
            if isinstance(cached_data, dict):
                # 전사 결과 찾기
                transcript = ""
                segments = []
                
                if 'transcript' in cached_data:
                    transcript = cached_data['transcript']
                elif 'text' in cached_data:
                    transcript = cached_data['text']
                elif 'result' in cached_data and 'text' in cached_data['result']:
                    transcript = cached_data['result']['text']
                
                if 'segments' in cached_data:
                    segments = cached_data['segments']
                elif 'result' in cached_data and 'segments' in cached_data['result']:
                    segments = cached_data['result']['segments']
                
                if transcript:
                    print(f"전사 텍스트: {len(transcript)}자")
                    print(f"세그먼트: {len(segments)}개")
                    
                    # 화자별 분석
                    speakers_analysis = analyze_speakers_from_segments(segments, transcript)
                    
                    all_results[cache_file.name] = {
                        'transcript': transcript,
                        'segments': segments,
                        'speakers_analysis': speakers_analysis,
                        'cache_file': str(cache_file)
                    }
                    
                    print(f"분석 완료: {len(speakers_analysis['speakers'])}명 화자")
                else:
                    print("전사 텍스트 없음")
                    # 캐시 구조 출력
                    print(f"캐시 키들: {list(cached_data.keys())}")
            else:
                print(f"예상치 못한 캐시 형식: {type(cached_data)}")
                
        except Exception as e:
            print(f"캐시 읽기 오류: {str(e)}")
    
    return all_results

def analyze_speakers_from_segments(segments, full_transcript):
    """세그먼트에서 화자 분리 분석"""
    if not segments:
        return {
            'speakers': {},
            'timeline': [],
            'stats': {}
        }
    
    # 화자 분리 (시간 기반 간단한 방식)
    speakers = {}
    timeline = []
    current_speaker = 1
    
    for i, segment in enumerate(segments):
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        # 화자 변경 감지 (침묵 또는 톤 변화 시뮬레이션)
        if i > 0:
            prev_end = segments[i-1].get('end', 0)
            silence = start_time - prev_end
            
            # 2초 이상 침묵시 화자 변경 가능성
            if silence > 2.0:
                current_speaker = (current_speaker % 4) + 1
        
        speaker_id = f"화자_{current_speaker}"
        
        # 화자별 통계
        if speaker_id not in speakers:
            speakers[speaker_id] = {
                'total_time': 0,
                'word_count': 0,
                'segments': [],
                'avg_confidence': 0
            }
        
        duration = end_time - start_time
        word_count = len(text.split())
        
        speakers[speaker_id]['total_time'] += duration
        speakers[speaker_id]['word_count'] += word_count
        speakers[speaker_id]['segments'].append({
            'start': start_time,
            'end': end_time,
            'text': text,
            'duration': duration
        })
        
        # 타임라인 추가
        timeline.append({
            'speaker': speaker_id,
            'start': round(start_time, 1),
            'end': round(end_time, 1),
            'text': text,
            'index': i
        })
    
    # 통계 계산
    total_time = sum(s['total_time'] for s in speakers.values())
    stats = {
        'total_speakers': len(speakers),
        'total_duration': total_time,
        'total_words': sum(s['word_count'] for s in speakers.values()),
        'total_segments': len(segments)
    }
    
    return {
        'speakers': speakers,
        'timeline': timeline,
        'stats': stats
    }

def print_transcript_results(results):
    """전사 결과 출력"""
    print("\n" + "=" * 80)
    print("전체 스크립트 및 화자별 분석 결과")
    print("=" * 80)
    
    for cache_name, data in results.items():
        print(f"\n[캐시: {cache_name}]")
        
        transcript = data['transcript']
        speakers_analysis = data['speakers_analysis']
        
        print(f"전체 스크립트 길이: {len(transcript)}자")
        print(f"감지된 화자: {speakers_analysis['stats']['total_speakers']}명")
        print(f"총 세그먼트: {speakers_analysis['stats']['total_segments']}개")
        
        # 전체 스크립트 출력 (처음 1000자)
        print(f"\n【전체 스크립트】")
        print("-" * 60)
        if len(transcript) > 1000:
            print(f"{transcript[:1000]}...")
            print(f"[{len(transcript)}자 중 처음 1000자만 표시]")
        else:
            print(transcript)
        
        # 화자별 통계
        print(f"\n【화자별 통계】")
        print("-" * 60)
        for speaker_id, speaker_data in speakers_analysis['speakers'].items():
            print(f"{speaker_id}:")
            print(f"  - 발화 시간: {speaker_data['total_time']:.1f}초")
            print(f"  - 단어 수: {speaker_data['word_count']}개")
            print(f"  - 발화 횟수: {len(speaker_data['segments'])}회")
        
        # 화자별 대화 내용 (타임라인)
        print(f"\n【화자별 대화 타임라인】")
        print("-" * 60)
        
        timeline = speakers_analysis['timeline']
        for i, entry in enumerate(timeline[:20]):  # 처음 20개
            speaker = entry['speaker']
            start = entry['start']
            text = entry['text']
            
            print(f"{speaker} ({start}초): {text}")
            
            if i == 19 and len(timeline) > 20:
                print(f"... [총 {len(timeline)}개 대화 중 처음 20개만 표시]")

def save_transcript_report(results):
    """전사 리포트 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 상세 JSON 저장
    json_file = f"cached_transcript_analysis_{timestamp}.json"
    try:
        # JSON 직렬화를 위해 데이터 정리
        clean_results = {}
        for cache_name, data in results.items():
            clean_results[cache_name] = {
                'transcript': data['transcript'],
                'speakers_analysis': data['speakers_analysis'],
                'cache_file': data['cache_file']
            }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        print(f"\n상세 결과 저장: {json_file}")
    except Exception as e:
        print(f"JSON 저장 실패: {str(e)}")
    
    # 텍스트 리포트 저장
    txt_file = f"cached_transcript_report_{timestamp}.txt"
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("솔로몬드 AI - 캐시된 전체 스크립트 분석 리포트\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.now().isoformat()}\n\n")
            
            for cache_name, data in results.items():
                f.write(f"\n[캐시: {cache_name}]\n")
                f.write(f"전체 스크립트 길이: {len(data['transcript'])}자\n")
                
                speakers_analysis = data['speakers_analysis']
                f.write(f"감지된 화자: {speakers_analysis['stats']['total_speakers']}명\n")
                f.write(f"총 세그먼트: {speakers_analysis['stats']['total_segments']}개\n\n")
                
                f.write("【전체 스크립트】\n")
                f.write("-" * 60 + "\n")
                f.write(data['transcript'] + "\n\n")
                
                f.write("【화자별 대화 타임라인】\n")
                f.write("-" * 60 + "\n")
                
                for entry in speakers_analysis['timeline']:
                    speaker = entry['speaker']
                    start = entry['start']
                    text = entry['text']
                    f.write(f"{speaker} ({start}초): {text}\n")
                
                f.write("\n" + "=" * 60 + "\n")
        
        print(f"텍스트 리포트 저장: {txt_file}")
    except Exception as e:
        print(f"텍스트 저장 실패: {str(e)}")

def main():
    """메인 실행"""
    print("솔로몬드 AI - 캐시된 전사 결과 분석")
    print("=" * 60)
    
    # 캐시된 결과 읽기
    results = read_cached_analysis()
    
    if results:
        # 결과 출력
        print_transcript_results(results)
        
        # 리포트 저장
        save_transcript_report(results)
        
        print(f"\n" + "=" * 60)
        print("캐시된 전사 분석 완료!")
        print("=" * 60)
    else:
        print("캐시된 결과 없음")
    
    return results

if __name__ == "__main__":
    main()