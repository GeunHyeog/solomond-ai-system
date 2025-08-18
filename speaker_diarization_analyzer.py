# -*- coding: utf-8 -*-
"""
화자 구분 분석기 - 사회자 1명 + 발표자 3명 구분
오디오에서 각 화자를 구분하여 누가 언제 무엇을 말했는지 분석
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

class SpeakerDiarizationAnalyzer:
    """화자 구분 분석 시스템"""
    
    def __init__(self):
        self.expected_speakers = {
            'moderator': '사회자',
            'speaker1': '발표자 1',
            'speaker2': '발표자 2', 
            'speaker3': '발표자 3'
        }
        
        # 사용 가능한 화자 분리 도구 확인
        self.tools_available = self._check_diarization_tools()
        
    def _check_diarization_tools(self):
        """화자 분리 도구 사용 가능성 확인"""
        
        print("화자 분리 도구 확인 중...")
        
        tools = {
            'pyannote': False,
            'whisper_enhanced': False,
            'resemblyzer': False,
            'basic_vad': False
        }
        
        # pyannote.audio 확인 (가장 정확함)
        try:
            import torch
            # pyannote는 별도 설치 필요하므로 일단 False
            print("  pyannote.audio: 미설치 (별도 설치 필요)")
        except:
            pass
        
        # Whisper 기반 화자 구분 확인
        try:
            import whisper
            tools['whisper_enhanced'] = True
            print("  Whisper (타임스탬프): 사용 가능")
        except:
            print("  Whisper: 불가능")
        
        # 기본 음성 활동 감지
        try:
            import librosa
            import numpy as np
            tools['basic_vad'] = True
            print("  기본 VAD (음성 구간 감지): 사용 가능")
        except:
            print("  기본 VAD: 불가능")
        
        return tools
    
    def analyze_speakers_in_audio(self, audio_path: str) -> Dict[str, Any]:
        """오디오 파일에서 화자 구분 분석"""
        
        print(f"\n🎤 화자 구분 분석: {os.path.basename(audio_path)}")
        
        if not os.path.exists(audio_path):
            return {'error': '파일이 존재하지 않습니다'}
        
        result = {
            'filename': os.path.basename(audio_path),
            'file_size': os.path.getsize(audio_path),
            'total_duration': 0,
            'speakers_detected': 0,
            'speaker_segments': [],
            'speaker_profiles': {},
            'transcript_by_speaker': {},
            'analysis_method': 'unknown'
        }
        
        # 사용 가능한 방법으로 분석
        if self.tools_available['whisper_enhanced']:
            result = self._analyze_with_whisper_timestamps(audio_path, result)
        elif self.tools_available['basic_vad']:
            result = self._analyze_with_basic_vad(audio_path, result)
        else:
            result['error'] = '화자 분리 도구를 사용할 수 없습니다'
        
        # 화자 역할 추정
        if result['speakers_detected'] > 0:
            result = self._assign_speaker_roles(result)
        
        return result
    
    def _analyze_with_whisper_timestamps(self, audio_path: str, result: Dict) -> Dict:
        """Whisper 타임스탬프 기반 화자 구분"""
        
        print("  Whisper 타임스탬프 분석 사용")
        
        try:
            import whisper
            
            # 상세 타임스탬프와 함께 transcription
            model = whisper.load_model("base")
            
            # word_timestamps=True로 단어별 타임스탬프 활성화
            transcript_result = model.transcribe(
                audio_path, 
                language="ko",
                word_timestamps=True,
                verbose=False
            )
            
            result['total_duration'] = transcript_result.get('duration', 0)
            result['analysis_method'] = 'whisper_timestamps'
            
            # segments 분석
            segments = transcript_result.get('segments', [])
            
            if segments:
                # 음성 변화점 기반 화자 추정
                speaker_changes = self._detect_speaker_changes_from_segments(segments)
                
                result['speaker_segments'] = speaker_changes
                result['speakers_detected'] = len(set(seg['speaker_id'] for seg in speaker_changes))
                
                # 화자별 텍스트 정리
                for segment in speaker_changes:
                    speaker_id = segment['speaker_id']
                    if speaker_id not in result['transcript_by_speaker']:
                        result['transcript_by_speaker'][speaker_id] = []
                    
                    result['transcript_by_speaker'][speaker_id].append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text']
                    })
                
                print(f"    화자 {result['speakers_detected']}명 감지")
                print(f"    총 구간: {len(speaker_changes)}개")
            
            return result
            
        except Exception as e:
            print(f"    Whisper 분석 오류: {str(e)}")
            result['error'] = str(e)
            return result
    
    def _detect_speaker_changes_from_segments(self, segments: List[Dict]) -> List[Dict]:
        """Whisper segments에서 화자 변화점 감지"""
        
        speaker_segments = []
        current_speaker = 0
        
        for i, segment in enumerate(segments):
            # 간단한 휴리스틱: 긴 침묵이나 음성 패턴 변화로 화자 추정
            
            # 이전 세그먼트와의 시간 간격 확인
            if i > 0:
                prev_end = segments[i-1]['end']
                current_start = segment['start']
                silence_duration = current_start - prev_end
                
                # 3초 이상 침묵이면 화자 변경 가능성
                if silence_duration > 3.0:
                    current_speaker = (current_speaker + 1) % 4  # 4명 순환
            
            # 세그먼트 길이나 텍스트 패턴으로도 추가 판단 가능
            text_length = len(segment['text'])
            
            # 짧은 발언은 사회자일 가능성 (질문, 소개 등)
            if text_length < 50 and any(word in segment['text'] for word in ['네', '감사', '다음', '질문']):
                speaker_id = 'moderator'
            else:
                speaker_id = f'speaker{(current_speaker % 3) + 1}' if current_speaker > 0 else 'moderator'
            
            speaker_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'speaker_id': speaker_id,
                'confidence': 0.7  # 기본 신뢰도
            })
        
        return speaker_segments
    
    def _analyze_with_basic_vad(self, audio_path: str, result: Dict) -> Dict:
        """기본 음성 활동 감지 기반 분석"""
        
        print("  기본 VAD 분석 사용")
        
        try:
            import librosa
            import numpy as np
            
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=16000)
            result['total_duration'] = len(y) / sr
            result['analysis_method'] = 'basic_vad'
            
            # 간단한 음성 구간 감지
            # RMS 에너지 기반 음성 활동 감지
            frame_length = int(0.025 * sr)  # 25ms 프레임
            hop_length = int(0.01 * sr)     # 10ms 홉
            
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 임계값 기반 음성 구간 찾기
            threshold = np.percentile(rms, 60)  # 상위 40% 에너지
            voice_frames = rms > threshold
            
            # 연속된 음성 구간 찾기
            voice_segments = []
            in_voice = False
            start_frame = 0
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_voice:
                    start_frame = i
                    in_voice = True
                elif not is_voice and in_voice:
                    # 음성 구간 종료
                    start_time = start_frame * hop_length / sr
                    end_time = i * hop_length / sr
                    
                    if end_time - start_time > 1.0:  # 1초 이상인 구간만
                        voice_segments.append({
                            'start': start_time,
                            'end': end_time,
                            'speaker_id': f'speaker{len(voice_segments) % 4}',
                            'text': f'[{len(voice_segments)+1}번째 발언 구간]',
                            'confidence': 0.5
                        })
                    in_voice = False
            
            result['speaker_segments'] = voice_segments
            result['speakers_detected'] = min(len(voice_segments), 4)
            
            print(f"    음성 구간 {len(voice_segments)}개 감지")
            
            return result
            
        except Exception as e:
            print(f"    VAD 분석 오류: {str(e)}")
            result['error'] = str(e)
            return result
    
    def _assign_speaker_roles(self, result: Dict) -> Dict:
        """감지된 화자들에게 역할 할당"""
        
        print("  화자 역할 할당 중...")
        
        speaker_profiles = {}
        transcript_by_speaker = result.get('transcript_by_speaker', {})
        
        # 각 화자의 특성 분석
        for speaker_id, segments in transcript_by_speaker.items():
            total_text = ' '.join([seg['text'] for seg in segments])
            total_duration = sum([seg['end'] - seg['start'] for seg in segments])
            segment_count = len(segments)
            
            # 화자 특성 분석
            avg_segment_length = len(total_text) / segment_count if segment_count > 0 else 0
            
            # 사회자 특성: 짧은 발언, 많은 세그먼트, 특정 키워드
            moderator_keywords = ['감사', '네', '다음', '질문', '발표', '소개', '시간']
            moderator_score = sum(1 for keyword in moderator_keywords if keyword in total_text)
            
            # 발표자 특성: 긴 발언, 적은 세그먼트
            speaker_score = avg_segment_length / 10  # 평균 길이 기반 점수
            
            speaker_profiles[speaker_id] = {
                'total_duration': total_duration,
                'segment_count': segment_count,
                'total_text_length': len(total_text),
                'avg_segment_length': avg_segment_length,
                'moderator_score': moderator_score,
                'speaker_score': speaker_score,
                'sample_text': total_text[:100] + '...' if len(total_text) > 100 else total_text
            }
        
        # 역할 할당
        sorted_speakers = sorted(
            speaker_profiles.keys(), 
            key=lambda x: speaker_profiles[x]['moderator_score'], 
            reverse=True
        )
        
        role_assignments = {}
        if sorted_speakers:
            # 가장 높은 moderator_score를 가진 화자를 사회자로
            role_assignments[sorted_speakers[0]] = '사회자'
            
            # 나머지를 발표자로
            for i, speaker in enumerate(sorted_speakers[1:4], 1):
                role_assignments[speaker] = f'발표자 {i}'
        
        # 프로필에 역할 정보 추가
        for speaker_id, profile in speaker_profiles.items():
            profile['assigned_role'] = role_assignments.get(speaker_id, '미분류')
        
        result['speaker_profiles'] = speaker_profiles
        result['role_assignments'] = role_assignments
        
        print(f"    역할 할당 완료: {len(role_assignments)}명")
        
        return result
    
    def analyze_all_audio_files(self) -> Dict[str, Any]:
        """user_files/audio의 모든 오디오 파일 화자 구분 분석"""
        
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
        
        all_results = {}
        
        for audio_path in audio_files:
            result = self.analyze_speakers_in_audio(audio_path)
            all_results[os.path.basename(audio_path)] = result
        
        return all_results
    
    def save_speaker_analysis(self, results: Dict[str, Any]):
        """화자 분석 결과 저장"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"speaker_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n분석 결과 저장: {filename}")
        except Exception as e:
            print(f"저장 실패: {str(e)}")
        
        # 요약 리포트도 생성
        self._create_speaker_report(results, timestamp)
    
    def _create_speaker_report(self, results: Dict[str, Any], timestamp: str):
        """화자 분석 요약 리포트 생성"""
        
        report_filename = f"speaker_report_{timestamp}.md"
        
        report_lines = []
        report_lines.append("# 화자 구분 분석 리포트")
        report_lines.append(f"생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for filename, result in results.items():
            if 'error' in result:
                continue
                
            report_lines.append(f"## 📁 {filename}")
            report_lines.append("")
            
            # 기본 정보
            duration = result.get('total_duration', 0)
            speakers_count = result.get('speakers_detected', 0)
            
            report_lines.append(f"- **총 길이**: {duration:.1f}초 ({duration/60:.1f}분)")
            report_lines.append(f"- **감지된 화자**: {speakers_count}명")
            report_lines.append(f"- **분석 방법**: {result.get('analysis_method', 'unknown')}")
            report_lines.append("")
            
            # 화자별 정보
            if 'speaker_profiles' in result:
                report_lines.append("### 👥 화자별 정보")
                report_lines.append("")
                
                for speaker_id, profile in result['speaker_profiles'].items():
                    role = profile.get('assigned_role', '미분류')
                    duration = profile.get('total_duration', 0)
                    segments = profile.get('segment_count', 0)
                    
                    report_lines.append(f"#### {role} ({speaker_id})")
                    report_lines.append(f"- 발언 시간: {duration:.1f}초")
                    report_lines.append(f"- 발언 횟수: {segments}회")
                    report_lines.append(f"- 샘플 텍스트: {profile.get('sample_text', '')}")
                    report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("*솔로몬드 AI 화자 구분 분석기로 생성됨*")
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"요약 리포트 저장: {report_filename}")
        except Exception as e:
            print(f"리포트 저장 실패: {str(e)}")

def main():
    """메인 실행"""
    
    print("=== 화자 구분 분석기 ===")
    print("사회자 1명 + 발표자 3명을 구분하여 분석합니다.")
    
    try:
        analyzer = SpeakerDiarizationAnalyzer()
        
        # 모든 오디오 파일 분석
        results = analyzer.analyze_all_audio_files()
        
        if results:
            print(f"\n" + "="*60)
            print("화자 구분 분석 결과")
            print("="*60)
            
            for filename, result in results.items():
                print(f"\n파일: {filename}")
                
                if 'error' in result:
                    print(f"  오류: {result['error']}")
                    continue
                
                duration = result.get('total_duration', 0)
                speakers = result.get('speakers_detected', 0)
                
                print(f"  길이: {duration:.1f}초 ({duration/60:.1f}분)")
                print(f"  화자: {speakers}명 감지")
                
                # 역할 할당 결과
                if 'role_assignments' in result:
                    print("  역할 할당:")
                    for speaker_id, role in result['role_assignments'].items():
                        profile = result['speaker_profiles'].get(speaker_id, {})
                        segments = profile.get('segment_count', 0)
                        print(f"    {role}: {segments}회 발언")
            
            # 결과 저장
            analyzer.save_speaker_analysis(results)
            
        else:
            print("\n분석할 오디오 파일이 없습니다.")
            print("user_files/audio/ 폴더에 오디오 파일을 넣어주세요.")
    
    except KeyboardInterrupt:
        print("\n분석이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")

if __name__ == "__main__":
    main()