# -*- coding: utf-8 -*-
"""
간단한 화자 구분 분석기 - Windows 안전 버전
사회자 1명 + 발표자 3명 구분
"""

import os
import sys
import time
from pathlib import Path
import json

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

def analyze_speakers_with_whisper(audio_path):
    """Whisper로 화자 구분 분석"""
    
    print(f"화자 분석: {os.path.basename(audio_path)}")
    
    try:
        import whisper
        
        # Whisper 모델 로드
        model = whisper.load_model("base")
        
        # 타임스탬프와 함께 transcription
        result = model.transcribe(
            audio_path, 
            language="ko",
            word_timestamps=True,
            verbose=False
        )
        
        segments = result.get('segments', [])
        total_duration = result.get('duration', 0)
        
        print(f"  총 길이: {total_duration:.1f}초")
        print(f"  세그먼트: {len(segments)}개")
        
        # 화자 구분 로직
        speaker_segments = assign_speakers_by_pattern(segments)
        
        # 화자별 통계
        speaker_stats = calculate_speaker_stats(speaker_segments)
        
        return {
            'filename': os.path.basename(audio_path),
            'total_duration': total_duration,
            'segments': speaker_segments,
            'speaker_stats': speaker_stats,
            'success': True
        }
        
    except Exception as e:
        print(f"  오류: {str(e)}")
        return {
            'filename': os.path.basename(audio_path),
            'error': str(e), 
            'success': False
        }

def assign_speakers_by_pattern(segments):
    """패턴 기반 화자 할당"""
    
    speaker_segments = []
    current_speaker = 'moderator'  # 시작은 사회자
    
    for i, segment in enumerate(segments):
        text = segment['text'].strip()
        
        # 화자 변경 조건들
        speaker_change = False
        
        # 1. 긴 침묵 후 = 화자 변경 가능성
        if i > 0:
            prev_end = segments[i-1]['end']
            current_start = segment['start']
            silence = current_start - prev_end
            
            if silence > 2.0:  # 2초 이상 침묵
                speaker_change = True
        
        # 2. 텍스트 길이 기반 판단
        text_length = len(text)
        
        # 3. 키워드 기반 판단
        moderator_keywords = ['감사', '네', '다음', '질문', '발표', '소개']
        has_moderator_keyword = any(keyword in text for keyword in moderator_keywords)
        
        # 화자 결정 로직
        if speaker_change:
            if has_moderator_keyword or text_length < 30:
                current_speaker = 'moderator'
            else:
                # 발표자 순환 (1, 2, 3)
                speaker_num = (i // 3) % 3 + 1
                current_speaker = f'speaker_{speaker_num}'
        
        # 짧고 사회자 키워드가 있으면 무조건 사회자
        if text_length < 20 and has_moderator_keyword:
            current_speaker = 'moderator'
        
        speaker_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': text,
            'speaker': current_speaker,
            'duration': segment['end'] - segment['start']
        })
    
    return speaker_segments

def calculate_speaker_stats(segments):
    """화자별 통계 계산"""
    
    stats = {}
    
    for segment in segments:
        speaker = segment['speaker']
        
        if speaker not in stats:
            stats[speaker] = {
                'total_duration': 0,
                'segment_count': 0,
                'total_text_length': 0,
                'texts': []
            }
        
        stats[speaker]['total_duration'] += segment['duration']
        stats[speaker]['segment_count'] += 1
        stats[speaker]['total_text_length'] += len(segment['text'])
        stats[speaker]['texts'].append(segment['text'])
    
    # 역할 할당
    role_mapping = {
        'moderator': '사회자',
        'speaker_1': '발표자 1',
        'speaker_2': '발표자 2', 
        'speaker_3': '발표자 3'
    }
    
    for speaker in stats:
        stats[speaker]['role'] = role_mapping.get(speaker, '미분류')
        
        # 대표 텍스트 (가장 긴 발언)
        if stats[speaker]['texts']:
            longest_text = max(stats[speaker]['texts'], key=len)
            stats[speaker]['sample_text'] = longest_text[:100] + '...' if len(longest_text) > 100 else longest_text
    
    return stats

def analyze_all_audio_files():
    """모든 오디오 파일 분석"""
    
    audio_folder = "user_files/audio"
    
    if not os.path.exists(audio_folder):
        print("user_files/audio 폴더가 없습니다.")
        return {}
    
    audio_files = []
    for file in os.listdir(audio_folder):
        if file.lower().endswith(('.wav', '.m4a', '.mp3')):
            audio_files.append(os.path.join(audio_folder, file))
    
    if not audio_files:
        print("오디오 파일이 없습니다.")
        return {}
    
    print(f"오디오 파일 {len(audio_files)}개 발견")
    
    results = {}
    
    for audio_path in audio_files:
        result = analyze_speakers_with_whisper(audio_path)
        results[os.path.basename(audio_path)] = result
    
    return results

def print_results(results):
    """결과 출력"""
    
    print("\n" + "="*50)
    print("화자 구분 분석 결과")
    print("="*50)
    
    for filename, result in results.items():
        print(f"\n파일: {filename}")
        
        if not result['success']:
            print(f"  오류: {result['error']}")
            continue
        
        duration = result['total_duration']
        print(f"  총 길이: {duration:.1f}초 ({duration/60:.1f}분)")
        
        # 화자별 통계
        speaker_stats = result['speaker_stats']
        print(f"  감지된 화자: {len(speaker_stats)}명")
        
        for speaker, stats in speaker_stats.items():
            role = stats['role']
            speak_time = stats['total_duration']
            segments = stats['segment_count']
            
            print(f"    {role}: {speak_time:.1f}초 ({segments}회 발언)")
            
            # 샘플 텍스트
            if 'sample_text' in stats:
                print(f"      샘플: {stats['sample_text']}")

def save_results(results):
    """결과 저장"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # JSON 저장
    json_filename = f"speaker_analysis_{timestamp}.json"
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n분석 결과 저장: {json_filename}")
    except Exception as e:
        print(f"JSON 저장 실패: {str(e)}")
    
    # 텍스트 리포트 저장
    report_filename = f"speaker_report_{timestamp}.txt"
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("화자 구분 분석 리포트\n")
            f.write(f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            for filename, result in results.items():
                f.write(f"파일: {filename}\n")
                
                if not result['success']:
                    f.write(f"  오류: {result['error']}\n\n")
                    continue
                
                duration = result['total_duration']
                f.write(f"총 길이: {duration:.1f}초 ({duration/60:.1f}분)\n")
                
                speaker_stats = result['speaker_stats']
                f.write(f"감지된 화자: {len(speaker_stats)}명\n\n")
                
                for speaker, stats in speaker_stats.items():
                    role = stats['role']
                    speak_time = stats['total_duration']
                    segments = stats['segment_count']
                    
                    f.write(f"  {role}:\n")
                    f.write(f"    발언 시간: {speak_time:.1f}초\n")
                    f.write(f"    발언 횟수: {segments}회\n")
                    
                    if 'sample_text' in stats:
                        f.write(f"    샘플 텍스트: {stats['sample_text']}\n")
                    f.write("\n")
                
                f.write("-" * 30 + "\n\n")
        
        print(f"텍스트 리포트 저장: {report_filename}")
    except Exception as e:
        print(f"리포트 저장 실패: {str(e)}")

def main():
    """메인 실행"""
    
    print("=== 간단한 화자 구분 분석기 ===")
    print("사회자 1명 + 발표자 3명을 구분합니다.")
    
    try:
        # 모든 오디오 파일 분석
        results = analyze_all_audio_files()
        
        if results:
            # 결과 출력
            print_results(results)
            
            # 결과 저장
            save_results(results)
            
            print("\n분석 완료!")
            
        else:
            print("\n분석할 오디오 파일이 없습니다.")
            print("user_files/audio/ 폴더에 오디오 파일을 넣어주세요.")
    
    except KeyboardInterrupt:
        print("\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()