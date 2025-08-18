# -*- coding: utf-8 -*-
"""
고급 화자 구분 시스템 - 정교한 화자 분리
사회자 1명 + 발표자 3명을 더 정확하게 구분
"""

import os
import sys
import time
from pathlib import Path
import json
import numpy as np

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

class AdvancedSpeakerDiarization:
    """정교한 화자 구분 시스템"""
    
    def __init__(self):
        self.moderator_keywords = [
            '감사', '네', '다음', '질문', '발표', '소개', '시간', '마이크',
            '안녕', '시작', '끝', '마무리', '박수', '정리', '요약'
        ]
        
        self.speaker_transition_words = [
            '그런데', '그래서', '따라서', '또한', '그리고', '하지만',
            '우선', '먼저', '다음으로', '마지막으로', '결론적으로'
        ]
        
    def analyze_with_advanced_logic(self, audio_path):
        """고급 로직으로 화자 구분 분석"""
        
        print(f"🎯 고급 화자 분석: {os.path.basename(audio_path)}")
        
        try:
            import whisper
            
            # Whisper 모델로 상세 분석
            model = whisper.load_model("base")
            
            # 단어별 타임스탬프 포함 분석
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
            speaker_segments = self._advanced_speaker_assignment(segments)
            
            # 화자별 통계 및 프로필
            speaker_profiles = self._create_speaker_profiles(speaker_segments)
            
            # 역할 재할당 (통계 기반)
            final_assignments = self._reassign_roles_by_statistics(speaker_profiles)
            
            return {
                'filename': os.path.basename(audio_path),
                'total_duration': total_duration,
                'segments_count': len(segments),
                'speaker_segments': speaker_segments,
                'speaker_profiles': speaker_profiles,
                'final_assignments': final_assignments,
                'success': True
            }
            
        except Exception as e:
            print(f"  오류: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def _advanced_speaker_assignment(self, segments):
        """정교한 화자 할당 로직"""
        
        print("  🔍 정교한 화자 구분 중...")
        
        speaker_segments = []
        current_speaker = 'moderator'
        speaker_history = ['moderator']
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            duration = segment['end'] - segment['start']
            
            # 다양한 특성 분석
            features = self._analyze_segment_features(segment, i, segments)
            
            # 화자 결정
            predicted_speaker = self._predict_speaker(features, current_speaker, speaker_history)
            
            # 연속성 검사 및 보정
            if i > 0:
                predicted_speaker = self._apply_continuity_rules(
                    predicted_speaker, speaker_segments[-3:], features
                )
            
            speaker_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'duration': duration,
                'text': text,
                'speaker': predicted_speaker,
                'confidence': features['confidence'],
                'features': features
            })
            
            current_speaker = predicted_speaker
            speaker_history.append(predicted_speaker)
            
            # 최근 10개만 유지
            if len(speaker_history) > 10:
                speaker_history.pop(0)
        
        print(f"    화자 구분 완료: {len(set(seg['speaker'] for seg in speaker_segments))}명")
        
        return speaker_segments
    
    def _analyze_segment_features(self, segment, index, all_segments):
        """세그먼트 특성 분석"""
        
        text = segment['text'].strip()
        duration = segment['end'] - segment['start']
        
        features = {
            'text_length': len(text),
            'duration': duration,
            'words_per_second': len(text.split()) / duration if duration > 0 else 0,
            'has_moderator_keywords': any(kw in text for kw in self.moderator_keywords),
            'has_transition_words': any(tw in text for tw in self.speaker_transition_words),
            'is_question': '?' in text or '질문' in text,
            'is_short_response': len(text) < 30,
            'silence_before': 0,
            'silence_after': 0,
            'confidence': 0.5
        }
        
        # 이전/이후 침묵 계산
        if index > 0:
            prev_end = all_segments[index-1]['end']
            current_start = segment['start']
            features['silence_before'] = current_start - prev_end
        
        if index < len(all_segments) - 1:
            current_end = segment['end']
            next_start = all_segments[index+1]['start']
            features['silence_after'] = next_start - current_end
        
        return features
    
    def _predict_speaker(self, features, current_speaker, history):
        """특성 기반 화자 예측"""
        
        # 규칙 기반 점수 계산
        scores = {
            'moderator': 0,
            'speaker1': 0,
            'speaker2': 0,
            'speaker3': 0
        }
        
        # 사회자 특성
        if features['has_moderator_keywords']:
            scores['moderator'] += 3
        
        if features['is_question']:
            scores['moderator'] += 2
        
        if features['is_short_response']:
            scores['moderator'] += 1
        
        if features['duration'] < 3.0:  # 3초 미만
            scores['moderator'] += 1
        
        # 발표자 특성
        if features['duration'] > 10.0:  # 10초 이상
            for speaker in ['speaker1', 'speaker2', 'speaker3']:
                scores[speaker] += 2
        
        if features['text_length'] > 100:  # 긴 텍스트
            for speaker in ['speaker1', 'speaker2', 'speaker3']:
                scores[speaker] += 2
        
        if features['has_transition_words']:
            for speaker in ['speaker1', 'speaker2', 'speaker3']:
                scores[speaker] += 1
        
        # 침묵 기반 화자 변경
        if features['silence_before'] > 2.0:  # 2초 이상 침묵
            # 현재 화자가 아닌 다른 화자에게 점수 추가
            for speaker in scores:
                if speaker != current_speaker:
                    scores[speaker] += 1
        
        # 연속성 페널티 (같은 화자가 너무 많이 연속되면)
        recent_speakers = history[-5:]  # 최근 5개
        current_count = recent_speakers.count(current_speaker)
        
        if current_count >= 3:  # 3회 이상 연속
            scores[current_speaker] -= 1
        
        # 최고 점수 화자 선택
        predicted = max(scores, key=scores.get)
        confidence = scores[predicted] / max(sum(scores.values()), 1)
        
        return predicted
    
    def _apply_continuity_rules(self, predicted, recent_segments, features):
        """연속성 규칙 적용"""
        
        if not recent_segments:
            return predicted
        
        # 매우 짧은 세그먼트들은 이전 화자와 합칠 수 있음
        if features['duration'] < 1.0 and features['text_length'] < 10:
            return recent_segments[-1]['speaker']
        
        # 같은 화자가 너무 많이 연속되면 강제 변경
        if len(recent_segments) >= 3:
            recent_speakers = [seg['speaker'] for seg in recent_segments]
            if all(s == predicted for s in recent_speakers[-3:]):
                # 다른 화자로 변경
                all_speakers = ['moderator', 'speaker1', 'speaker2', 'speaker3']
                available = [s for s in all_speakers if s != predicted]
                if available:
                    return available[0]
        
        return predicted
    
    def _create_speaker_profiles(self, speaker_segments):
        """화자별 프로필 생성"""
        
        profiles = {}
        
        for segment in speaker_segments:
            speaker = segment['speaker']
            
            if speaker not in profiles:
                profiles[speaker] = {
                    'total_duration': 0,
                    'segment_count': 0,
                    'total_words': 0,
                    'avg_segment_length': 0,
                    'longest_segment': 0,
                    'shortest_segment': float('inf'),
                    'texts': [],
                    'speaking_ratio': 0
                }
            
            profile = profiles[speaker]
            profile['total_duration'] += segment['duration']
            profile['segment_count'] += 1
            profile['total_words'] += len(segment['text'].split())
            profile['longest_segment'] = max(profile['longest_segment'], segment['duration'])
            profile['shortest_segment'] = min(profile['shortest_segment'], segment['duration'])
            profile['texts'].append(segment['text'])
        
        # 통계 계산
        total_duration = sum(p['total_duration'] for p in profiles.values())
        
        for speaker, profile in profiles.items():
            if profile['segment_count'] > 0:
                profile['avg_segment_length'] = profile['total_duration'] / profile['segment_count']
                profile['speaking_ratio'] = profile['total_duration'] / total_duration if total_duration > 0 else 0
                profile['words_per_minute'] = (profile['total_words'] / profile['total_duration']) * 60 if profile['total_duration'] > 0 else 0
                
                # 대표 텍스트 (가장 긴 발언)
                profile['representative_text'] = max(profile['texts'], key=len) if profile['texts'] else ""
        
        return profiles
    
    def _reassign_roles_by_statistics(self, profiles):
        """통계 기반 역할 재할당"""
        
        print("  📊 통계 기반 역할 할당 중...")
        
        # 사회자 특성: 짧은 발언, 많은 횟수
        # 발표자 특성: 긴 발언, 적은 횟수
        
        role_scores = {}
        
        for speaker, profile in profiles.items():
            # 사회자 점수 계산
            moderator_score = 0
            
            if profile['avg_segment_length'] < 5.0:  # 평균 5초 미만
                moderator_score += 2
            
            if profile['segment_count'] > len(profiles) * 2:  # 다른 화자들보다 많은 발언
                moderator_score += 2
            
            if profile['speaking_ratio'] < 0.3:  # 전체의 30% 미만
                moderator_score += 1
            
            # 키워드 기반 점수
            combined_text = ' '.join(profile['texts']).lower()
            keyword_count = sum(1 for kw in self.moderator_keywords if kw in combined_text)
            moderator_score += keyword_count
            
            role_scores[speaker] = {
                'moderator_score': moderator_score,
                'speaker_score': profile['avg_segment_length'] + profile['speaking_ratio'] * 10
            }
        
        # 역할 할당
        assignments = {}
        
        # 가장 높은 moderator_score를 가진 사람을 사회자로
        moderator = max(role_scores.keys(), key=lambda x: role_scores[x]['moderator_score'])
        assignments[moderator] = '사회자'
        
        # 나머지를 발표자로 (speaker_score 순으로)
        remaining = [s for s in role_scores.keys() if s != moderator]
        remaining.sort(key=lambda x: role_scores[x]['speaker_score'], reverse=True)
        
        for i, speaker in enumerate(remaining[:3]):
            assignments[speaker] = f'발표자 {i+1}'
        
        # 할당되지 않은 화자는 미분류
        for speaker in profiles.keys():
            if speaker not in assignments:
                assignments[speaker] = '미분류'
        
        return assignments

def analyze_user_audio_advanced():
    """사용자 오디오 파일 고급 분석"""
    
    print("=== 고급 화자 구분 시스템 ===")
    print("사회자 1명 + 발표자 3명을 정교하게 구분합니다.")
    
    audio_folder = "user_files/audio"
    
    if not os.path.exists(audio_folder):
        print("❌ user_files/audio 폴더가 없습니다.")
        return
    
    audio_files = []
    for file in os.listdir(audio_folder):
        if file.lower().endswith(('.wav', '.m4a', '.mp3')):
            audio_files.append(os.path.join(audio_folder, file))
    
    if not audio_files:
        print("❌ 오디오 파일이 없습니다.")
        return
    
    print(f"📁 발견된 파일: {len(audio_files)}개")
    
    # 크기 순으로 정렬 (작은 파일부터)
    audio_files.sort(key=lambda x: os.path.getsize(x))
    
    diarizer = AdvancedSpeakerDiarization()
    
    # 첫 번째 파일만 분석 (시간 절약)
    first_file = audio_files[0]
    file_size = os.path.getsize(first_file) // 1024  # KB
    
    print(f"\n🎯 분석 대상: {os.path.basename(first_file)} ({file_size}KB)")
    
    if file_size > 5000:  # 5MB 이상
        print("⚠️ 파일이 큽니다. 시간이 오래 걸릴 수 있습니다.")
    
    start_time = time.time()
    result = diarizer.analyze_with_advanced_logic(first_file)
    analysis_time = time.time() - start_time
    
    if result['success']:
        print(f"\n✅ 분석 완료 ({analysis_time:.1f}초)")
        print_advanced_results(result)
        save_advanced_results(result)
    else:
        print(f"❌ 분석 실패: {result['error']}")

def print_advanced_results(result):
    """고급 분석 결과 출력"""
    
    print("\n" + "="*60)
    print("🎭 고급 화자 구분 결과")
    print("="*60)
    
    print(f"📁 파일: {result['filename']}")
    print(f"⏱️ 길이: {result['total_duration']:.1f}초 ({result['total_duration']/60:.1f}분)")
    print(f"📊 세그먼트: {result['segments_count']}개")
    
    # 최종 역할 할당
    assignments = result['final_assignments']
    profiles = result['speaker_profiles']
    
    print(f"\n👥 감지된 화자: {len(assignments)}명")
    
    for speaker_id, role in assignments.items():
        if speaker_id in profiles:
            profile = profiles[speaker_id]
            duration = profile['total_duration']
            segments = profile['segment_count']
            ratio = profile['speaking_ratio'] * 100
            
            print(f"\n🎯 {role} ({speaker_id}):")
            print(f"   발언 시간: {duration:.1f}초 ({ratio:.1f}%)")
            print(f"   발언 횟수: {segments}회")
            print(f"   평균 길이: {profile['avg_segment_length']:.1f}초")
            
            # 대표 발언
            if profile.get('representative_text'):
                sample = profile['representative_text'][:100]
                if len(profile['representative_text']) > 100:
                    sample += "..."
                print(f"   대표 발언: {sample}")

def save_advanced_results(result):
    """고급 분석 결과 저장"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # JSON 저장
    json_filename = f"advanced_speaker_analysis_{timestamp}.json"
    
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 상세 결과 저장: {json_filename}")
    except Exception as e:
        print(f"❌ 저장 실패: {str(e)}")
    
    # 요약 리포트 저장
    report_filename = f"speaker_summary_{timestamp}.txt"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("고급 화자 구분 분석 리포트\n")
            f.write(f"파일: {result['filename']}\n")
            f.write(f"분석 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            assignments = result['final_assignments']
            profiles = result['speaker_profiles']
            
            f.write(f"총 분석 시간: {result['total_duration']:.1f}초\n")
            f.write(f"감지된 화자: {len(assignments)}명\n\n")
            
            for speaker_id, role in assignments.items():
                if speaker_id in profiles:
                    profile = profiles[speaker_id]
                    f.write(f"{role} ({speaker_id}):\n")
                    f.write(f"  발언 시간: {profile['total_duration']:.1f}초\n")
                    f.write(f"  발언 횟수: {profile['segment_count']}회\n")
                    f.write(f"  비율: {profile['speaking_ratio']*100:.1f}%\n")
                    
                    if profile.get('representative_text'):
                        f.write(f"  대표 발언: {profile['representative_text'][:200]}...\n")
                    f.write("\n")
        
        print(f"📋 요약 리포트 저장: {report_filename}")
        
    except Exception as e:
        print(f"❌ 리포트 저장 실패: {str(e)}")

def main():
    """메인 실행"""
    
    try:
        analyze_user_audio_advanced()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n💥 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()