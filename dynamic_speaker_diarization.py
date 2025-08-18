# -*- coding: utf-8 -*-
"""
동적 화자 구분 시스템 - 인원 수 자동 감지
상황에 따라 화자 수가 달라지는 회의/발표/상담에 대응
사회자 유무, 참가자 수를 자동으로 감지하고 역할 할당
"""

import os
import sys
import time
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

class DynamicSpeakerDiarization:
    """동적 화자 구분 시스템"""
    
    def __init__(self):
        # 다양한 역할 키워드 정의
        self.role_keywords = {
            'moderator': ['감사', '네', '다음', '질문', '발표', '소개', '시간', '마이크', '박수', 
                         '시작', '끝', '마무리', '정리', '요약', '안녕하세요'],
            'presenter': ['발표', '설명', '보여드리', '말씀드리', '결론', '데이터', '그래프', 
                         '결과', '분석', '연구', '조사'],
            'participant': ['질문', '궁금', '의견', '생각', '제안', '동의', '반대'],
            'expert': ['전문', '경험', '기술적', '구체적', '세부사항', '방법론']
        }
        
        # 대화 패턴 키워드
        self.conversation_patterns = {
            'question': ['?', '질문', '궁금', '어떻게', '왜', '언제', '어디서'],
            'answer': ['답변', '대답', '설명', '그것은', '왜냐하면'],
            'agreement': ['맞습니다', '동의', '그렇습니다', '좋습니다'],
            'transition': ['그런데', '그래서', '다음으로', '또한', '하지만']
        }
    
    def analyze_dynamic_speakers(self, audio_path):
        """동적 화자 구분 분석"""
        
        print(f"동적 화자 분석: {os.path.basename(audio_path)}")
        
        try:
            import whisper
            
            # Whisper로 상세 분석
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
            
            # 1단계: 화자 변화점 감지
            speaker_changes = self._detect_speaker_changes(segments)
            
            # 2단계: 동적 화자 수 추정
            estimated_speakers = self._estimate_speaker_count(speaker_changes)
            
            # 3단계: 화자별 프로필 생성
            speaker_profiles = self._create_dynamic_profiles(speaker_changes)
            
            # 4단계: 상황별 역할 할당
            role_assignments = self._assign_dynamic_roles(speaker_profiles, estimated_speakers)
            
            # 5단계: 대화 패턴 분석
            conversation_analysis = self._analyze_conversation_patterns(speaker_changes, role_assignments)
            
            return {
                'filename': os.path.basename(audio_path),
                'total_duration': total_duration,
                'estimated_speaker_count': estimated_speakers,
                'speaker_changes': speaker_changes,
                'speaker_profiles': speaker_profiles,
                'role_assignments': role_assignments,
                'conversation_analysis': conversation_analysis,
                'full_text': result['text'],
                'success': True
            }
            
        except Exception as e:
            print(f"  오류: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def _detect_speaker_changes(self, segments):
        """화자 변화점 감지 - 더 정교한 로직"""
        
        print("  화자 변화점 감지 중...")
        
        speaker_segments = []
        current_speaker_id = 0
        speaker_history = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            duration = segment['end'] - segment['start']
            
            # 화자 변경 신호들
            change_indicators = self._calculate_change_indicators(segment, i, segments)
            
            # 화자 변경 결정
            should_change = self._should_change_speaker(change_indicators, speaker_history)
            
            if should_change:
                current_speaker_id += 1
            
            speaker_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'duration': duration,
                'text': text,
                'speaker_id': f'speaker_{current_speaker_id}',
                'change_indicators': change_indicators,
                'segment_index': i
            }
            
            speaker_segments.append(speaker_segment)
            speaker_history.append(speaker_segment)
            
            # 최근 5개만 유지
            if len(speaker_history) > 5:
                speaker_history.pop(0)
        
        return speaker_segments
    
    def _calculate_change_indicators(self, segment, index, all_segments):
        """화자 변경 지표 계산"""
        
        text = segment['text'].strip()
        duration = segment['end'] - segment['start']
        
        indicators = {
            'silence_before': 0,
            'silence_after': 0,
            'text_length_change': 0,
            'speaking_speed_change': 0,
            'topic_shift': False,
            'question_answer_pattern': False,
            'strong_change_signal': False
        }
        
        # 침묵 시간 계산
        if index > 0:
            prev_end = all_segments[index-1]['end']
            current_start = segment['start']
            indicators['silence_before'] = current_start - prev_end
        
        if index < len(all_segments) - 1:
            current_end = segment['end']
            next_start = all_segments[index+1]['start']
            indicators['silence_after'] = next_start - current_end
        
        # 텍스트 길이 변화
        if index > 0:
            prev_length = len(all_segments[index-1]['text'])
            current_length = len(text)
            indicators['text_length_change'] = abs(current_length - prev_length) / max(prev_length, 1)
        
        # 말하기 속도 변화
        if duration > 0:
            words_per_second = len(text.split()) / duration
            if index > 0:
                prev_duration = all_segments[index-1]['end'] - all_segments[index-1]['start']
                if prev_duration > 0:
                    prev_speed = len(all_segments[index-1]['text'].split()) / prev_duration
                    indicators['speaking_speed_change'] = abs(words_per_second - prev_speed) / max(prev_speed, 0.1)
        
        # 주제 전환 감지
        transition_words = ['그런데', '그래서', '다음으로', '또한', '하지만', '그리고', '마지막으로']
        indicators['topic_shift'] = any(word in text for word in transition_words)
        
        # 질문-답변 패턴
        if '?' in text or any(q in text for q in ['질문', '궁금', '어떻게']):
            indicators['question_answer_pattern'] = True
        
        # 강한 변경 신호 (호명, 인사 등)
        strong_signals = ['안녕하세요', '감사합니다', '말씀드리겠습니다', '질문 있습니다']
        indicators['strong_change_signal'] = any(signal in text for signal in strong_signals)
        
        return indicators
    
    def _should_change_speaker(self, indicators, history):
        """화자 변경 결정"""
        
        # 가중치 기반 점수 계산
        change_score = 0
        
        # 침묵 시간 (가장 강한 지표)
        if indicators['silence_before'] > 3.0:  # 3초 이상
            change_score += 5
        elif indicators['silence_before'] > 1.5:  # 1.5초 이상
            change_score += 3
        elif indicators['silence_before'] > 0.8:  # 0.8초 이상
            change_score += 1
        
        # 강한 변경 신호
        if indicators['strong_change_signal']:
            change_score += 4
        
        # 질문-답변 패턴
        if indicators['question_answer_pattern']:
            change_score += 2
        
        # 주제 전환
        if indicators['topic_shift']:
            change_score += 1
        
        # 텍스트 길이 급변
        if indicators['text_length_change'] > 2.0:  # 200% 이상 변화
            change_score += 2
        
        # 말하기 속도 급변
        if indicators['speaking_speed_change'] > 1.0:  # 100% 이상 변화
            change_score += 1
        
        # 연속성 페널티 (같은 화자가 너무 오래 지속되면)
        if len(history) >= 3:
            recent_speakers = [h['speaker_id'] for h in history[-3:]]
            if len(set(recent_speakers)) == 1:  # 모두 같은 화자
                change_score += 1
        
        # 임계값: 3점 이상이면 화자 변경
        return change_score >= 3
    
    def _estimate_speaker_count(self, speaker_segments):
        """동적 화자 수 추정"""
        
        print("  화자 수 추정 중...")
        
        # 기본적으로 감지된 unique speaker_id 개수
        unique_speakers = set(seg['speaker_id'] for seg in speaker_segments)
        basic_count = len(unique_speakers)
        
        # 화자별 통계 분석
        speaker_stats = defaultdict(lambda: {'duration': 0, 'segments': 0, 'texts': []})
        
        for seg in speaker_segments:
            speaker_id = seg['speaker_id']
            speaker_stats[speaker_id]['duration'] += seg['duration']
            speaker_stats[speaker_id]['segments'] += 1
            speaker_stats[speaker_id]['texts'].append(seg['text'])
        
        # 너무 짧은 발언은 동일 화자일 수 있음 (병합 고려)
        filtered_count = 0
        for speaker_id, stats in speaker_stats.items():
            # 3초 이상 발언하거나 2회 이상 발언한 경우만 실제 화자로 인정
            if stats['duration'] >= 3.0 or stats['segments'] >= 2:
                filtered_count += 1
        
        estimated_count = max(filtered_count, 1)  # 최소 1명
        
        print(f"    기본 감지: {basic_count}명")
        print(f"    필터링 후: {estimated_count}명")
        
        return estimated_count
    
    def _create_dynamic_profiles(self, speaker_segments):
        """동적 화자 프로필 생성"""
        
        print("  화자 프로필 생성 중...")
        
        profiles = defaultdict(lambda: {
            'total_duration': 0,
            'segment_count': 0,
            'texts': [],
            'avg_segment_length': 0,
            'speaking_ratio': 0,
            'role_scores': defaultdict(int),
            'conversation_patterns': defaultdict(int)
        })
        
        total_duration = sum(seg['duration'] for seg in speaker_segments)
        
        for seg in speaker_segments:
            speaker_id = seg['speaker_id']
            profile = profiles[speaker_id]
            
            profile['total_duration'] += seg['duration']
            profile['segment_count'] += 1
            profile['texts'].append(seg['text'])
            
            # 역할 키워드 점수 계산
            text_lower = seg['text'].lower()
            for role, keywords in self.role_keywords.items():
                keyword_count = sum(1 for kw in keywords if kw in text_lower)
                profile['role_scores'][role] += keyword_count
            
            # 대화 패턴 점수 계산
            for pattern, keywords in self.conversation_patterns.items():
                pattern_count = sum(1 for kw in keywords if kw in text_lower)
                profile['conversation_patterns'][pattern] += pattern_count
        
        # 통계 계산
        for speaker_id, profile in profiles.items():
            if profile['segment_count'] > 0:
                profile['avg_segment_length'] = profile['total_duration'] / profile['segment_count']
                profile['speaking_ratio'] = profile['total_duration'] / total_duration if total_duration > 0 else 0
                
                # 대표 텍스트
                profile['representative_text'] = max(profile['texts'], key=len) if profile['texts'] else ""
        
        return dict(profiles)
    
    def _assign_dynamic_roles(self, profiles, speaker_count):
        """상황별 동적 역할 할당"""
        
        print("  동적 역할 할당 중...")
        
        assignments = {}
        
        # 역할 점수 기반 정렬
        speakers_by_role_score = []
        
        for speaker_id, profile in profiles.items():
            # 각 역할별 적합도 계산
            role_fitness = {}
            
            for role in ['moderator', 'presenter', 'participant', 'expert']:
                fitness = profile['role_scores'][role]
                
                # 추가 특성 고려
                if role == 'moderator':
                    # 사회자는 짧은 발언, 많은 횟수
                    if profile['avg_segment_length'] < 5.0:
                        fitness += 2
                    if profile['segment_count'] > speaker_count:
                        fitness += 2
                    if profile['speaking_ratio'] < 0.4:
                        fitness += 1
                
                elif role == 'presenter':
                    # 발표자는 긴 발언, 높은 비율
                    if profile['avg_segment_length'] > 8.0:
                        fitness += 2
                    if profile['speaking_ratio'] > 0.3:
                        fitness += 2
                
                elif role == 'participant':
                    # 참가자는 중간 정도 발언
                    if 3.0 < profile['avg_segment_length'] < 10.0:
                        fitness += 1
                    if profile['conversation_patterns']['question'] > 0:
                        fitness += 2
                
                role_fitness[role] = fitness
            
            # 최적 역할 선택
            best_role = max(role_fitness, key=role_fitness.get)
            best_score = role_fitness[best_role]
            
            speakers_by_role_score.append((speaker_id, best_role, best_score, profile))
        
        # 점수 순으로 정렬
        speakers_by_role_score.sort(key=lambda x: x[2], reverse=True)
        
        # 상황별 역할 할당 전략
        if speaker_count == 1:
            # 1명: 발표자 또는 참가자
            assignments[speakers_by_role_score[0][0]] = '단독 발화자'
            
        elif speaker_count == 2:
            # 2명: 진행자+발표자 또는 대화 참가자들
            if speakers_by_role_score[0][1] == 'moderator':
                assignments[speakers_by_role_score[0][0]] = '진행자'
                assignments[speakers_by_role_score[1][0]] = '발표자'
            else:
                assignments[speakers_by_role_score[0][0]] = '참가자 A'
                assignments[speakers_by_role_score[1][0]] = '참가자 B'
                
        else:
            # 3명 이상: 다양한 역할 할당
            role_counter = {'moderator': 0, 'presenter': 0, 'participant': 0}
            
            for speaker_id, suggested_role, score, profile in speakers_by_role_score:
                if suggested_role == 'moderator' and role_counter['moderator'] == 0:
                    assignments[speaker_id] = '사회자'
                    role_counter['moderator'] += 1
                elif suggested_role == 'presenter' and role_counter['presenter'] < 2:
                    role_counter['presenter'] += 1
                    assignments[speaker_id] = f'발표자 {role_counter["presenter"]}'
                else:
                    role_counter['participant'] += 1
                    assignments[speaker_id] = f'참가자 {role_counter["participant"]}'
        
        return assignments
    
    def _analyze_conversation_patterns(self, speaker_segments, assignments):
        """대화 패턴 분석"""
        
        # 발언 순서, 상호작용 패턴 등 분석
        patterns = {
            'turn_taking_pattern': [],  # 발언 순서
            'interaction_pairs': [],    # 상호작용 쌍
            'dominant_speaker': None,   # 주도적 화자
            'conversation_flow': 'unknown'  # 대화 흐름 유형
        }
        
        # 발언 순서 패턴
        for seg in speaker_segments:
            patterns['turn_taking_pattern'].append(seg['speaker_id'])
        
        # 주도적 화자 (가장 많이 발언한 화자)
        speaker_counts = defaultdict(int)
        for seg in speaker_segments:
            speaker_counts[seg['speaker_id']] += 1
        
        if speaker_counts:
            dominant_speaker_id = max(speaker_counts, key=speaker_counts.get)
            patterns['dominant_speaker'] = assignments.get(dominant_speaker_id, dominant_speaker_id)
        
        # 대화 흐름 유형 판단
        unique_speakers = len(set(seg['speaker_id'] for seg in speaker_segments))
        total_segments = len(speaker_segments)
        
        if unique_speakers == 1:
            patterns['conversation_flow'] = '단독 발표'
        elif unique_speakers == 2 and total_segments > 10:
            patterns['conversation_flow'] = '대화/인터뷰'
        elif unique_speakers >= 3:
            patterns['conversation_flow'] = '회의/토론'
        else:
            patterns['conversation_flow'] = '짧은 대화'
        
        return patterns

def analyze_user_audio_dynamic():
    """사용자 오디오 파일 동적 분석"""
    
    print("=== 동적 화자 구분 시스템 ===")
    print("인원 수와 역할을 자동으로 감지합니다.")
    
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
    
    print(f"발견된 파일: {len(audio_files)}개")
    
    # 크기 순으로 정렬
    audio_files.sort(key=lambda x: os.path.getsize(x))
    
    diarizer = DynamicSpeakerDiarization()
    
    # 첫 번째 파일 분석
    first_file = audio_files[0]
    file_size = os.path.getsize(first_file) // 1024
    
    print(f"\n분석 대상: {os.path.basename(first_file)} ({file_size}KB)")
    
    start_time = time.time()
    result = diarizer.analyze_dynamic_speakers(first_file)
    analysis_time = time.time() - start_time
    
    if result['success']:
        print(f"\n분석 완료 ({analysis_time:.1f}초)")
        print_dynamic_results(result)
        save_dynamic_results(result)
    else:
        print(f"분석 실패: {result['error']}")

def print_dynamic_results(result):
    """동적 분석 결과 출력"""
    
    print("\n" + "="*60)
    print("동적 화자 구분 결과")
    print("="*60)
    
    print(f"파일: {result['filename']}")
    print(f"길이: {result['total_duration']:.1f}초")
    print(f"추정 화자 수: {result['estimated_speaker_count']}명")
    
    # 대화 패턴 분석
    conv_analysis = result['conversation_analysis']
    print(f"대화 유형: {conv_analysis['conversation_flow']}")
    if conv_analysis['dominant_speaker']:
        print(f"주도적 화자: {conv_analysis['dominant_speaker']}")
    
    # 화자별 정보
    profiles = result['speaker_profiles']
    assignments = result['role_assignments']
    
    print(f"\n화자별 분석:")
    for speaker_id, role in assignments.items():
        if speaker_id in profiles:
            profile = profiles[speaker_id]
            duration = profile['total_duration']
            segments = profile['segment_count']
            ratio = profile['speaking_ratio'] * 100
            
            print(f"\n{role} ({speaker_id}):")
            print(f"  발언 시간: {duration:.1f}초 ({ratio:.1f}%)")
            print(f"  발언 횟수: {segments}회")
            print(f"  평균 길이: {profile['avg_segment_length']:.1f}초")
            
            # 주요 특성
            top_role = max(profile['role_scores'], key=profile['role_scores'].get)
            if profile['role_scores'][top_role] > 0:
                print(f"  주요 특성: {top_role} ({profile['role_scores'][top_role]}점)")
            
            # 대표 발언
            if profile.get('representative_text'):
                sample = profile['representative_text'][:80]
                if len(profile['representative_text']) > 80:
                    sample += "..."
                print(f"  대표 발언: {sample}")

def save_dynamic_results(result):
    """동적 분석 결과 저장"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # JSON 저장
    json_filename = f"dynamic_speaker_analysis_{timestamp}.json"
    
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n상세 결과 저장: {json_filename}")
    except Exception as e:
        print(f"저장 실패: {str(e)}")
    
    # 요약 리포트
    report_filename = f"dynamic_speaker_report_{timestamp}.txt"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("동적 화자 구분 분석 리포트\n")
            f.write(f"파일: {result['filename']}\n")
            f.write(f"분석 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"총 길이: {result['total_duration']:.1f}초\n")
            f.write(f"추정 화자 수: {result['estimated_speaker_count']}명\n")
            f.write(f"대화 유형: {result['conversation_analysis']['conversation_flow']}\n\n")
            
            profiles = result['speaker_profiles']
            assignments = result['role_assignments']
            
            for speaker_id, role in assignments.items():
                if speaker_id in profiles:
                    profile = profiles[speaker_id]
                    f.write(f"{role} ({speaker_id}):\n")
                    f.write(f"  발언 시간: {profile['total_duration']:.1f}초\n")
                    f.write(f"  발언 횟수: {profile['segment_count']}회\n")
                    f.write(f"  비율: {profile['speaking_ratio']*100:.1f}%\n")
                    
                    if profile.get('representative_text'):
                        f.write(f"  대표 발언: {profile['representative_text'][:150]}...\n")
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
        analyze_user_audio_dynamic()
    except KeyboardInterrupt:
        print("\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()