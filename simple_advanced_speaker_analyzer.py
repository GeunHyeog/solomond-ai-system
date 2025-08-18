# -*- coding: utf-8 -*-
"""
고급 화자 구분 분석기 - Windows 안전 버전
사회자 1명 + 발표자 3명을 정교하게 구분
"""

import os
import sys
import time
from pathlib import Path
import json

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

def analyze_speakers_advanced(audio_path):
    """고급 로직으로 화자 구분 분석"""
    
    print(f"고급 화자 분석: {os.path.basename(audio_path)}")
    
    try:
        import whisper
        
        # Whisper 모델로 상세 분석
        model = whisper.load_model("base")
        
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
        
        # 정교한 화자 구분
        speaker_analysis = advanced_speaker_detection(segments)
        
        return {
            'filename': os.path.basename(audio_path),
            'total_duration': total_duration,
            'segments_count': len(segments),
            'speaker_analysis': speaker_analysis,
            'full_text': result['text'],
            'success': True
        }
        
    except Exception as e:
        print(f"  오류: {str(e)}")
        return {'error': str(e), 'success': False}

def advanced_speaker_detection(segments):
    """정교한 화자 감지 로직"""
    
    # 키워드 정의
    moderator_keywords = ['감사', '네', '다음', '질문', '발표', '소개', '시간', '마이크', '박수']
    transition_words = ['그런데', '그래서', '따라서', '또한', '그리고', '하지만', '우선']
    
    speaker_segments = []
    current_speaker = 'moderator'
    speaker_stats = {
        'moderator': {'duration': 0, 'count': 0, 'texts': []},
        'speaker1': {'duration': 0, 'count': 0, 'texts': []},
        'speaker2': {'duration': 0, 'count': 0, 'texts': []},
        'speaker3': {'duration': 0, 'count': 0, 'texts': []}
    }
    
    for i, segment in enumerate(segments):
        text = segment['text'].strip()
        duration = segment['end'] - segment['start']
        
        # 화자 변경 조건들
        should_change_speaker = False
        new_speaker = current_speaker
        
        # 1. 침묵 기반 화자 변경
        if i > 0:
            prev_end = segments[i-1]['end']
            current_start = segment['start']
            silence = current_start - prev_end
            
            if silence > 2.5:  # 2.5초 이상 침묵
                should_change_speaker = True
        
        # 2. 텍스트 길이 기반 판단
        text_length = len(text)
        
        # 3. 키워드 기반 판단
        has_moderator_keywords = any(keyword in text for keyword in moderator_keywords)
        has_transition_words = any(word in text for word in transition_words)
        
        # 화자 결정 로직
        if should_change_speaker:
            if has_moderator_keywords or (text_length < 40 and not has_transition_words):
                new_speaker = 'moderator'
            else:
                # 발표자 순환
                speakers = ['speaker1', 'speaker2', 'speaker3']
                current_idx = speakers.index(current_speaker) if current_speaker in speakers else 0
                new_speaker = speakers[(current_idx + 1) % 3]
        
        # 강제 사회자 조건
        if has_moderator_keywords and text_length < 50:
            new_speaker = 'moderator'
        
        # 긴 발언은 발표자일 가능성 높음
        if text_length > 150 and duration > 8.0:
            if current_speaker == 'moderator':
                new_speaker = 'speaker1'  # 첫 번째 발표자로
        
        current_speaker = new_speaker
        
        # 세그먼트 저장
        speaker_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'duration': duration,
            'text': text,
            'speaker': current_speaker,
            'text_length': text_length,
            'silence_before': silence if i > 0 else 0
        })
        
        # 통계 업데이트
        speaker_stats[current_speaker]['duration'] += duration
        speaker_stats[current_speaker]['count'] += 1
        speaker_stats[current_speaker]['texts'].append(text)
    
    # 통계 기반 역할 재할당
    final_assignments = reassign_roles_by_stats(speaker_stats)
    
    return {
        'segments': speaker_segments,
        'statistics': speaker_stats,
        'role_assignments': final_assignments
    }

def reassign_roles_by_stats(speaker_stats):
    """통계 기반 역할 재할당"""
    
    # 각 화자의 특성 점수 계산
    role_scores = {}
    
    for speaker, stats in speaker_stats.items():
        if stats['count'] == 0:
            continue
            
        # 사회자 특성 점수
        avg_duration = stats['duration'] / stats['count']
        combined_text = ' '.join(stats['texts']).lower()
        
        moderator_score = 0
        
        # 짧은 평균 발언시간
        if avg_duration < 4.0:
            moderator_score += 2
        
        # 많은 발언 횟수
        if stats['count'] > 3:
            moderator_score += 1
        
        # 사회자 키워드 포함
        moderator_keywords = ['감사', '네', '다음', '질문', '발표', '소개']
        keyword_count = sum(1 for kw in moderator_keywords if kw in combined_text)
        moderator_score += keyword_count
        
        # 발표자 특성 점수 (긴 발언, 적은 횟수)
        speaker_score = avg_duration * 0.5 + (stats['duration'] / 60) * 2
        
        role_scores[speaker] = {
            'moderator_score': moderator_score,
            'speaker_score': speaker_score,
            'avg_duration': avg_duration,
            'total_duration': stats['duration']
        }
    
    # 역할 할당
    assignments = {}
    
    # 가장 높은 moderator_score를 가진 화자를 사회자로
    if role_scores:
        moderator = max(role_scores.keys(), key=lambda x: role_scores[x]['moderator_score'])
        assignments[moderator] = '사회자'
        
        # 나머지를 발표자로 (speaker_score 순)
        remaining = [s for s in role_scores.keys() if s != moderator]
        remaining.sort(key=lambda x: role_scores[x]['speaker_score'], reverse=True)
        
        for i, speaker in enumerate(remaining[:3]):
            assignments[speaker] = f'발표자 {i+1}'
    
    return assignments

def analyze_first_audio():
    """첫 번째 오디오 파일 고급 분석"""
    
    print("=== 고급 화자 구분 분석기 ===")
    print("사회자 1명 + 발표자 3명을 정교하게 구분합니다.")
    
    audio_folder = "user_files/audio"
    
    if not os.path.exists(audio_folder):
        print("user_files/audio 폴더가 없습니다.")
        return
    
    audio_files = []
    for file in os.listdir(audio_folder):
        if file.lower().endswith(('.wav', '.m4a', '.mp3')):
            audio_files.append(os.path.join(audio_folder, file))
    
    if not audio_files:
        print("오디오 파일이 없습니다.")
        return
    
    # 크기 순으로 정렬 (작은 파일부터)
    audio_files.sort(key=lambda x: os.path.getsize(x))
    first_file = audio_files[0]
    
    file_size = os.path.getsize(first_file) // 1024
    print(f"분석 대상: {os.path.basename(first_file)} ({file_size}KB)")
    
    if file_size > 10000:  # 10MB 이상
        print("파일이 큽니다. 시간이 오래 걸릴 수 있습니다.")
    
    start_time = time.time()
    result = analyze_speakers_advanced(first_file)
    analysis_time = time.time() - start_time
    
    if result['success']:
        print(f"분석 완료 ({analysis_time:.1f}초)")
        print_detailed_results(result)
        save_detailed_results(result)
    else:
        print(f"분석 실패: {result['error']}")

def print_detailed_results(result):
    """상세 결과 출력"""
    
    print("\n" + "="*60)
    print("고급 화자 구분 결과")
    print("="*60)
    
    analysis = result['speaker_analysis']
    assignments = analysis['role_assignments']
    statistics = analysis['statistics']
    
    print(f"파일: {result['filename']}")
    print(f"길이: {result['total_duration']:.1f}초 ({result['total_duration']/60:.1f}분)")
    print(f"세그먼트: {result['segments_count']}개")
    print(f"감지된 화자: {len([s for s in statistics.values() if s['count'] > 0])}명")
    
    print("\n화자별 분석:")
    for speaker_id, role in assignments.items():
        if speaker_id in statistics and statistics[speaker_id]['count'] > 0:
            stats = statistics[speaker_id]
            avg_duration = stats['duration'] / stats['count']
            
            print(f"\n{role} ({speaker_id}):")
            print(f"  발언 시간: {stats['duration']:.1f}초")
            print(f"  발언 횟수: {stats['count']}회")
            print(f"  평균 길이: {avg_duration:.1f}초")
            print(f"  비율: {(stats['duration']/result['total_duration'])*100:.1f}%")
            
            # 대표 발언 (가장 긴 텍스트)
            if stats['texts']:
                longest = max(stats['texts'], key=len)
                sample = longest[:100] + "..." if len(longest) > 100 else longest
                print(f"  대표 발언: {sample}")

def save_detailed_results(result):
    """상세 결과 저장"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # JSON 저장
    filename = f"advanced_speaker_result_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n상세 결과 저장: {filename}")
    except Exception as e:
        print(f"저장 실패: {str(e)}")
    
    # 텍스트 리포트
    report_filename = f"speaker_analysis_report_{timestamp}.txt"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("고급 화자 구분 분석 리포트\n")
            f.write(f"파일: {result['filename']}\n")
            f.write(f"분석 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            analysis = result['speaker_analysis']
            assignments = analysis['role_assignments']
            statistics = analysis['statistics']
            
            f.write(f"총 길이: {result['total_duration']:.1f}초\n")
            f.write(f"세그먼트: {result['segments_count']}개\n\n")
            
            for speaker_id, role in assignments.items():
                if speaker_id in statistics and statistics[speaker_id]['count'] > 0:
                    stats = statistics[speaker_id]
                    f.write(f"{role} ({speaker_id}):\n")
                    f.write(f"  발언 시간: {stats['duration']:.1f}초\n")
                    f.write(f"  발언 횟수: {stats['count']}회\n")
                    f.write(f"  비율: {(stats['duration']/result['total_duration'])*100:.1f}%\n")
                    
                    if stats['texts']:
                        longest = max(stats['texts'], key=len)
                        f.write(f"  대표 발언: {longest[:200]}...\n")
                    f.write("\n")
            
            f.write("\n전체 텍스트:\n")
            f.write("-" * 30 + "\n")
            f.write(result['full_text'])
        
        print(f"리포트 저장: {report_filename}")
        
    except Exception as e:
        print(f"리포트 저장 실패: {str(e)}")

def main():
    """메인 실행"""
    
    try:
        analyze_first_audio()
    except KeyboardInterrupt:
        print("\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()