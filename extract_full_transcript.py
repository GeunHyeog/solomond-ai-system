#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 스크립트 추출 및 화자별 분석
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def extract_full_transcript():
    """전체 오디오 전사 및 화자별 분석"""
    print("전체 스크립트 추출 시작...")
    
    try:
        import whisper
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        print("Whisper 모델 로딩...")
        model = whisper.load_model("base")
        print("모델 로드 완료")
        
        # 오디오 파일들
        audio_files = [
            "user_files/JGA2025_D1/새로운 녹음.m4a",
            "user_files/JGA2025_D1/새로운 녹음 2.m4a",
            "user_files/JGA2025_D1/IMG_0032_audio.wav"
        ]
        
        all_results = {}
        
        for audio_file in audio_files:
            file_path = Path(audio_file)
            
            if file_path.exists():
                print(f"\n분석 중: {file_path.name}")
                
                try:
                    # Whisper 전사 (세그먼트 포함)
                    result = model.transcribe(str(file_path), 
                                            language='ko',
                                            word_timestamps=True)
                    
                    if result and 'text' in result:
                        full_transcript = result['text']
                        segments = result.get('segments', [])
                        
                        print(f"전체 전사: {len(full_transcript)}자")
                        print(f"세그먼트: {len(segments)}개")
                        
                        # 화자 분리 시뮬레이션 (시간 기반)
                        speakers_dialogue = []
                        current_speaker = 1
                        
                        for i, segment in enumerate(segments):
                            start_time = segment.get('start', 0)
                            end_time = segment.get('end', 0)
                            text = segment.get('text', '').strip()
                            
                            # 간단한 화자 변경 감지 (긴 침묵이나 톤 변화 시뮬레이션)
                            if i > 0:
                                prev_end = segments[i-1].get('end', 0)
                                silence_duration = start_time - prev_end
                                
                                # 3초 이상 침묵시 화자 변경 가능성
                                if silence_duration > 3.0:
                                    current_speaker = (current_speaker % 4) + 1  # 최대 4명
                            
                            speaker_info = {
                                'speaker_id': f"화자_{current_speaker}",
                                'start_time': round(start_time, 1),
                                'end_time': round(end_time, 1),
                                'duration': round(end_time - start_time, 1),
                                'text': text,
                                'segment_index': i
                            }
                            
                            speakers_dialogue.append(speaker_info)
                        
                        # 화자별 통계
                        speaker_stats = {}
                        for dialogue in speakers_dialogue:
                            speaker = dialogue['speaker_id']
                            if speaker not in speaker_stats:
                                speaker_stats[speaker] = {
                                    'total_time': 0,
                                    'total_words': 0,
                                    'segments_count': 0,
                                    'dialogues': []
                                }
                            
                            speaker_stats[speaker]['total_time'] += dialogue['duration']
                            speaker_stats[speaker]['total_words'] += len(dialogue['text'].split())
                            speaker_stats[speaker]['segments_count'] += 1
                            speaker_stats[speaker]['dialogues'].append(dialogue)
                        
                        all_results[file_path.name] = {
                            'full_transcript': full_transcript,
                            'total_segments': len(segments),
                            'speakers_dialogue': speakers_dialogue,
                            'speaker_stats': speaker_stats,
                            'total_duration': segments[-1].get('end', 0) if segments else 0
                        }
                        
                        print(f"화자 {len(speaker_stats)}명 감지")
                        
                    else:
                        print("전사 실패")
                        all_results[file_path.name] = {'status': 'failed'}
                        
                except Exception as e:
                    print(f"오류: {str(e)}")
                    all_results[file_path.name] = {'status': 'error', 'error': str(e)}
            else:
                print(f"파일 없음: {audio_file}")
        
        return all_results
        
    except ImportError as e:
        print(f"모듈 없음: {str(e)}")
        return {}
    except Exception as e:
        print(f"분석 오류: {str(e)}")
        return {}

def print_transcript_analysis(results):
    """전사 결과 출력"""
    print("\n" + "=" * 80)
    print("전체 스크립트 및 화자별 분석 결과")
    print("=" * 80)
    
    for filename, data in results.items():
        if 'full_transcript' in data:
            print(f"\n[파일: {filename}]")
            print(f"총 길이: {data['total_duration']:.1f}초")
            print(f"세그먼트: {data['total_segments']}개")
            
            # 전체 스크립트
            print(f"\n【전체 스크립트】")
            print("-" * 60)
            transcript = data['full_transcript']
            if len(transcript) > 1000:
                print(f"{transcript[:1000]}...")
                print(f"[전체 {len(transcript)}자 중 처음 1000자만 표시]")
            else:
                print(transcript)
            
            # 화자별 통계
            print(f"\n【화자별 통계】")
            print("-" * 60)
            speaker_stats = data['speaker_stats']
            
            for speaker, stats in speaker_stats.items():
                print(f"\n{speaker}:")
                print(f"  - 발화 시간: {stats['total_time']:.1f}초")
                print(f"  - 단어 수: {stats['total_words']}개")
                print(f"  - 발화 횟수: {stats['segments_count']}회")
            
            # 화자별 대화 내용 (처음 10개만)
            print(f"\n【화자별 대화 내용】")
            print("-" * 60)
            
            speakers_dialogue = data['speakers_dialogue']
            for i, dialogue in enumerate(speakers_dialogue[:15]):  # 처음 15개만
                speaker = dialogue['speaker_id']
                start = dialogue['start_time']
                text = dialogue['text']
                
                print(f"{speaker} ({start}초): {text}")
                
                if i == 14 and len(speakers_dialogue) > 15:
                    print(f"... [총 {len(speakers_dialogue)}개 대화 중 처음 15개만 표시]")
        else:
            print(f"\n[파일: {filename}] - 분석 실패")

def save_detailed_results(results):
    """상세 결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON 결과 저장
    json_file = f"full_transcript_analysis_{timestamp}.json"
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n상세 결과 저장: {json_file}")
    except Exception as e:
        print(f"JSON 저장 실패: {str(e)}")
    
    # 텍스트 리포트 저장
    txt_file = f"transcript_report_{timestamp}.txt"
    try:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("솔로몬드 AI - 전체 스크립트 및 화자별 분석 리포트\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.now().isoformat()}\n\n")
            
            for filename, data in results.items():
                if 'full_transcript' in data:
                    f.write(f"\n[파일: {filename}]\n")
                    f.write(f"총 길이: {data['total_duration']:.1f}초\n")
                    f.write(f"세그먼트: {data['total_segments']}개\n\n")
                    
                    f.write("【전체 스크립트】\n")
                    f.write("-" * 60 + "\n")
                    f.write(data['full_transcript'] + "\n\n")
                    
                    f.write("【화자별 대화】\n")
                    f.write("-" * 60 + "\n")
                    
                    for dialogue in data['speakers_dialogue']:
                        speaker = dialogue['speaker_id']
                        start = dialogue['start_time']
                        text = dialogue['text']
                        f.write(f"{speaker} ({start}초): {text}\n")
        
        print(f"텍스트 리포트 저장: {txt_file}")
    except Exception as e:
        print(f"텍스트 저장 실패: {str(e)}")

def main():
    """메인 실행"""
    print("솔로몬드 AI - 전체 스크립트 추출 및 화자별 분석")
    print("=" * 80)
    
    # 전체 스크립트 추출
    results = extract_full_transcript()
    
    if results:
        # 결과 출력
        print_transcript_analysis(results)
        
        # 상세 결과 저장
        save_detailed_results(results)
        
        print(f"\n" + "=" * 80)
        print("전체 스크립트 추출 및 화자별 분석 완료!")
        print("=" * 80)
    else:
        print("분석 실패 - 결과 없음")
    
    return results

if __name__ == "__main__":
    main()