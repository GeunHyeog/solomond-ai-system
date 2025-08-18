#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
화자별 대화 분석 및 마인드맵 생성 시스템
Enhanced Speaker Identification 통합 버전
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 최적화 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

# 향상된 화자 구분 시스템 import
try:
    from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
    ENHANCED_SPEAKER_ID_AVAILABLE = True
    print("[SUCCESS] Enhanced Speaker Identifier 로드 완료")
except ImportError as e:
    ENHANCED_SPEAKER_ID_AVAILABLE = False
    print(f"[WARNING] Enhanced Speaker Identifier 로드 실패: {e}")

class SpeakerMindmapAnalyzer:
    """화자별 분석 및 마인드맵 생성기"""
    
    def __init__(self, expected_speakers=3):
        self.speakers_data = {}
        self.conversation_flow = []
        self.topics_mapping = {}
        self.mindmap_data = {}
        self.expected_speakers = expected_speakers
        
        # 향상된 화자 구분 시스템 초기화
        if ENHANCED_SPEAKER_ID_AVAILABLE:
            self.enhanced_speaker_id = EnhancedSpeakerIdentifier(expected_speakers)
            print(f"[INFO] 향상된 화자 구분 시스템 활성화 (예상 화자: {expected_speakers}명)")
        else:
            self.enhanced_speaker_id = None
            print("[INFO] 기본 화자 구분 시스템 사용")
        
    def analyze_audio_files(self, audio_files_path="user_files/audio"):
        """오디오 파일들 분석"""
        print("=== 화자별 대화 분석 시작 ===")
        
        audio_path = Path(audio_files_path)
        if not audio_path.exists():
            print(f"오디오 폴더가 없습니다: {audio_files_path}")
            return
        
        # 모델 로딩
        try:
            import whisper
            model = whisper.load_model("tiny", device="cpu")
            print("Whisper 모델 로드 완료")
        except Exception as e:
            print(f"Whisper 로딩 실패: {e}")
            return
        
        # 오디오 파일들 처리
        audio_files = list(audio_path.glob("*.m4a")) + list(audio_path.glob("*.wav"))
        
        for i, audio_file in enumerate(audio_files):
            print(f"\\n[{i+1}/{len(audio_files)}] {audio_file.name} 분석 중...")
            
            try:
                # STT 처리
                result = model.transcribe(str(audio_file))
                
                # 세그먼트별 분석
                if 'segments' in result:
                    self._analyze_segments(audio_file.name, result['segments'])
                else:
                    # 전체 텍스트 분석
                    self._analyze_full_text(audio_file.name, result.get('text', ''))
                
                print(f"  완료: {len(result.get('segments', []))}개 세그먼트")
                
            except Exception as e:
                print(f"  오류: {e}")
        
        # 화자별 주제 분석
        self._extract_topics_by_speaker()
        
        # 마인드맵 데이터 생성
        self._generate_mindmap_data()
        
        del model
    
    def _analyze_single_file(self, file_path):
        """단일 파일 분석 (Streamlit에서 사용)"""
        try:
            import whisper
            model = whisper.load_model("tiny", device="cpu")
            
            result = model.transcribe(file_path)
            
            filename = Path(file_path).name
            
            # 세그먼트별 분석
            if 'segments' in result:
                self._analyze_segments(filename, result['segments'])
            else:
                # 전체 텍스트 분석
                self._analyze_full_text(filename, result.get('text', ''))
            
            del model
            return True
            
        except Exception as e:
            print(f"파일 분석 오류 {file_path}: {e}")
            return False
        
    def _analyze_segments(self, filename, segments):
        """세그먼트별 화자 및 내용 분석 (향상된 시스템 적용)"""
        
        # 의미있는 세그먼트만 필터링
        meaningful_segments = []
        for segment in segments:
            text = segment.get('text', '').strip()
            if len(text) > 10:  # 의미있는 길이의 텍스트만
                meaningful_segments.append(segment)
        
        if not meaningful_segments:
            return
        
        # 향상된 화자 구분 시스템 사용
        if self.enhanced_speaker_id:
            print(f"  [INFO] 향상된 화자 구분 적용 (세그먼트 {len(meaningful_segments)}개)")
            speaker_segments = self.enhanced_speaker_id.identify_speakers_from_segments(meaningful_segments)
        else:
            # 기본 시스템 사용
            print(f"  [INFO] 기본 화자 구분 적용 (세그먼트 {len(meaningful_segments)}개)")
            speaker_segments = []
            for segment in meaningful_segments:
                segment_copy = segment.copy()
                segment_copy['speaker'] = self._identify_speaker_fallback(segment.get('text', ''), segment.get('start', 0))
                speaker_segments.append(segment_copy)
        
        # 결과 처리
        for segment in speaker_segments:
            speaker = segment.get('speaker', '화자_1')
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', start_time + 1)
            
            # 화자별 데이터 저장
            if speaker not in self.speakers_data:
                self.speakers_data[speaker] = {
                    'texts': [],
                    'topics': [],
                    'speaking_time': 0,
                    'files': set()
                }
            
            self.speakers_data[speaker]['texts'].append({
                'text': text,
                'timestamp': start_time,
                'duration': end_time - start_time,
                'file': filename
            })
            self.speakers_data[speaker]['files'].add(filename)
            self.speakers_data[speaker]['speaking_time'] += (end_time - start_time)
            
            # 대화 흐름에 추가
            self.conversation_flow.append({
                'speaker': speaker,
                'text': text,
                'timestamp': start_time,
                'duration': end_time - start_time,
                'file': filename
            })
        
        # 화자 구분 품질 평가
        self._evaluate_speaker_quality(speaker_segments)
    
    def _analyze_full_text(self, filename, full_text):
        """전체 텍스트 분석 (세그먼트 정보가 없는 경우)"""
        if len(full_text.strip()) > 20:
            # 문장 단위로 분리
            sentences = [s.strip() for s in full_text.split('.') if len(s.strip()) > 10]
            
            for i, sentence in enumerate(sentences):
                speaker = f"화자_{(i % 3) + 1}"  # 3명의 화자로 가정
                
                if speaker not in self.speakers_data:
                    self.speakers_data[speaker] = {
                        'texts': [],
                        'topics': [],
                        'speaking_time': 0,
                        'files': set()
                    }
                
                self.speakers_data[speaker]['texts'].append({
                    'text': sentence,
                    'timestamp': i * 10,  # 가상 타임스탬프
                    'file': filename
                })
                self.speakers_data[speaker]['files'].add(filename)
    
    def _identify_speaker_fallback(self, text, timestamp):
        """화자 식별 (기본 시스템 - 시간대별 규칙 기반)"""
        # 시간대별 화자 추정 (기본 방식)
        if timestamp < 30:
            return "화자_1"
        elif timestamp < 60:
            return "화자_2"
        else:
            return "화자_3"
    
    def _evaluate_speaker_quality(self, speaker_segments):
        """화자 구분 품질 평가"""
        if not speaker_segments:
            return
        
        # 화자별 발언 수 계산
        speaker_counts = {}
        total_duration = 0
        
        for segment in speaker_segments:
            speaker = segment.get('speaker', '화자_1')
            duration = segment.get('end', 0) - segment.get('start', 0)
            
            if speaker not in speaker_counts:
                speaker_counts[speaker] = {'count': 0, 'duration': 0}
            
            speaker_counts[speaker]['count'] += 1
            speaker_counts[speaker]['duration'] += duration
            total_duration += duration
        
        print(f"  [QUALITY] 감지된 화자 수: {len(speaker_counts)}명")
        
        for speaker, data in speaker_counts.items():
            percentage = (data['duration'] / total_duration * 100) if total_duration > 0 else 0
            print(f"    {speaker}: {data['count']}회 발언, {data['duration']:.1f}초 ({percentage:.1f}%)")
        
        # 화자 분포 균형 확인
        durations = [data['duration'] for data in speaker_counts.values()]
        if len(durations) > 1:
            import numpy as np
            balance_std = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
            if balance_std < 0.5:
                print(f"  [QUALITY] 화자 분포: 균형적 (편차계수: {balance_std:.2f})")
            else:
                print(f"  [QUALITY] 화자 분포: 불균형 (편차계수: {balance_std:.2f})")
    
    def get_speaker_statistics(self):
        """화자별 상세 통계 반환"""
        stats = {}
        
        for speaker, data in self.speakers_data.items():
            if self.enhanced_speaker_id:
                # 향상된 시스템에서 화자 특성 분석
                combined_text = " ".join([item['text'] for item in data['texts']])
                features = self.enhanced_speaker_id.extract_text_features(combined_text)
                
                stats[speaker] = {
                    'utterance_count': len(data['texts']),
                    'total_speaking_time': data['speaking_time'],
                    'avg_utterance_duration': data['speaking_time'] / len(data['texts']) if data['texts'] else 0,
                    'text_features': features,
                    'speech_style': self.enhanced_speaker_id._classify_speech_style(features),
                    'dominant_patterns': self.enhanced_speaker_id._identify_dominant_patterns(combined_text)
                }
            else:
                # 기본 통계
                stats[speaker] = {
                    'utterance_count': len(data['texts']),
                    'total_speaking_time': data['speaking_time'],
                    'avg_utterance_duration': data['speaking_time'] / len(data['texts']) if data['texts'] else 0
                }
        
        return stats
    
    def _extract_topics_by_speaker(self):
        """화자별 주제 추출"""
        print("\\n화자별 주제 분석 중...")
        
        # 주제 키워드 사전
        topic_keywords = {
            '비즈니스': ['사업', '비즈니스', '매출', '수익', '고객', '마케팅', '브랜드'],
            '기술': ['기술', '시스템', '플랫폼', '개발', '구현', '솔루션'],
            '전략': ['전략', '계획', '목표', '방향', '비전', '미션'],
            '운영': ['운영', '관리', '프로세스', '업무', '효율', '최적화'],
            '시장': ['시장', '경쟁', '트렌드', '동향', '분석', '예측']
        }
        
        for speaker, data in self.speakers_data.items():
            speaker_topics = {}
            all_text = ' '.join([item['text'] for item in data['texts']])
            
            # 주제별 언급 빈도 계산
            for topic, keywords in topic_keywords.items():
                count = sum(all_text.lower().count(keyword) for keyword in keywords)
                if count > 0:
                    speaker_topics[topic] = count
            
            # 상위 주제들 저장
            sorted_topics = sorted(speaker_topics.items(), key=lambda x: x[1], reverse=True)
            data['topics'] = sorted_topics[:3]  # 상위 3개 주제
            
            print(f"  {speaker}: {[t[0] for t in sorted_topics[:3]]}")
    
    def _generate_mindmap_data(self):
        """마인드맵 데이터 생성"""
        print("\\n마인드맵 데이터 생성 중...")
        
        # 중심 주제
        self.mindmap_data = {
            'name': '대화 종합 분석',
            'children': []
        }
        
        # 화자별 노드 생성
        for speaker, data in self.speakers_data.items():
            speaker_node = {
                'name': speaker,
                'size': len(data['texts']) * 10,  # 발언량에 비례한 크기
                'children': []
            }
            
            # 주제별 하위 노드
            for topic, count in data['topics']:
                topic_node = {
                    'name': f"{topic} ({count})",
                    'size': count * 5,
                    'children': []
                }
                
                # 주요 발언들
                topic_texts = [
                    item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
                    for item in data['texts'] 
                    if any(keyword in item['text'].lower() for keyword in ['사업', '기술', '전략'] if topic in ['비즈니스', '기술', '전략'])
                ][:3]  # 최대 3개
                
                for text in topic_texts:
                    topic_node['children'].append({
                        'name': text,
                        'size': 5
                    })
                
                speaker_node['children'].append(topic_node)
            
            self.mindmap_data['children'].append(speaker_node)
    
    def generate_conversation_summary(self):
        """대화 요약 생성"""
        print("\\n대화 요약 생성 중...")
        
        summary = {
            'overview': {},
            'speaker_analysis': {},
            'topic_flow': [],
            'key_insights': []
        }
        
        # 전체 개요
        summary['overview'] = {
            'total_speakers': len(self.speakers_data),
            'total_utterances': sum(len(data['texts']) for data in self.speakers_data.values()),
            'main_topics': list(set(topic for data in self.speakers_data.values() for topic, _ in data['topics']))
        }
        
        # 화자별 분석
        for speaker, data in self.speakers_data.items():
            summary['speaker_analysis'][speaker] = {
                'utterance_count': len(data['texts']),
                'main_topics': [topic for topic, _ in data['topics']],
                'key_points': [
                    item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
                    for item in data['texts'][:3]  # 주요 발언 3개
                ]
            }
        
        # 주제 흐름 분석
        timeline_topics = []
        for item in sorted(self.conversation_flow, key=lambda x: x['timestamp']):
            # 주제 식별
            text_lower = item['text'].lower()
            identified_topic = "기타"
            for topic in ['비즈니스', '기술', '전략', '운영', '시장']:
                if any(keyword in text_lower for keyword in ['사업', '기술', '전략', '운영', '시장']):
                    identified_topic = topic
                    break
            
            timeline_topics.append({
                'timestamp': item['timestamp'],
                'speaker': item['speaker'],
                'topic': identified_topic,
                'content': item['text'][:100]
            })
        
        summary['topic_flow'] = timeline_topics[:10]  # 상위 10개
        
        # 주요 인사이트
        insights = []
        
        # 화자별 특징
        for speaker, data in self.speakers_data.items():
            if data['topics']:
                main_topic = data['topics'][0][0]
                insights.append(f"{speaker}는 주로 {main_topic}에 대해 발언함")
        
        # 대화 패턴
        if len(self.speakers_data) > 1:
            insights.append(f"{len(self.speakers_data)}명의 화자가 참여한 다자간 대화")
        
        summary['key_insights'] = insights
        
        return summary
    
    def save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 마인드맵 데이터 저장
        mindmap_filename = f"mindmap_data_{timestamp}.json"
        with open(mindmap_filename, 'w', encoding='utf-8') as f:
            json.dump(self.mindmap_data, f, ensure_ascii=False, indent=2)
        
        # 화자 분석 데이터 저장
        speakers_filename = f"speakers_analysis_{timestamp}.json"
        
        # 화자 데이터에서 set을 list로 변환
        speakers_data_serializable = {}
        for speaker, data in self.speakers_data.items():
            speakers_data_serializable[speaker] = {
                'texts': data['texts'],
                'topics': data['topics'],
                'speaking_time': data['speaking_time'],
                'files': list(data['files'])  # set을 list로 변환
            }
        
        with open(speakers_filename, 'w', encoding='utf-8') as f:
            json.dump(speakers_data_serializable, f, ensure_ascii=False, indent=2)
        
        # 대화 요약 저장
        summary = self.generate_conversation_summary()
        summary_filename = f"conversation_summary_{timestamp}.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\\n결과 저장 완료:")
        print(f"  마인드맵: {mindmap_filename}")
        print(f"  화자 분석: {speakers_filename}")
        print(f"  대화 요약: {summary_filename}")
        
        return {
            'mindmap_file': mindmap_filename,
            'speakers_file': speakers_filename,
            'summary_file': summary_filename
        }
    
    def print_analysis_results(self):
        """분석 결과 출력"""
        print("\\n" + "="*60)
        print("화자별 대화 분석 결과")
        print("="*60)
        
        for speaker, data in self.speakers_data.items():
            print(f"\\n👤 {speaker}")
            print(f"   발언 수: {len(data['texts'])}회")
            print(f"   참여 파일: {', '.join(data['files'])}")
            print(f"   주요 주제: {', '.join([t[0] for t in data['topics']])}")
            
            # 주요 발언 예시
            if data['texts']:
                print(f"   주요 발언:")
                for i, text_item in enumerate(data['texts'][:2]):
                    print(f"     {i+1}. {text_item['text'][:80]}...")
        
        print("\\n" + "="*60)
        
        # 대화 요약 출력
        summary = self.generate_conversation_summary()
        
        print("\\n📊 대화 개요:")
        print(f"   총 화자 수: {summary['overview']['total_speakers']}명")
        print(f"   총 발언 수: {summary['overview']['total_utterances']}회")
        print(f"   주요 주제: {', '.join(summary['overview']['main_topics'])}")
        
        print("\\n💡 주요 인사이트:")
        for insight in summary['key_insights']:
            print(f"   • {insight}")

def main():
    """메인 실행"""
    analyzer = SpeakerMindmapAnalyzer()
    
    # 오디오 파일 분석
    analyzer.analyze_audio_files()
    
    # 결과 출력
    analyzer.print_analysis_results()
    
    # 결과 저장
    files = analyzer.save_results()
    
    print(f"\\n✅ 화자별 분석 및 마인드맵 생성 완료!")
    print(f"   생성된 파일: {len(files)}개")

if __name__ == "__main__":
    main()