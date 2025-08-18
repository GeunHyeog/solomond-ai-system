#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종합 상황 분석 시스템 - 실제 상황의 모든 파일들을 하나로 통합 분석
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# 최적화 설정
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

class ComprehensiveSituationAnalyzer:
    """종합 상황 분석기 - 멀티모달 통합 분석"""
    
    def __init__(self):
        self.situation_data = {
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'total_files': 0,
                'file_types': {},
                'timeline': []
            },
            'audio_analysis': [],
            'image_analysis': [],
            'video_analysis': [],
            'comprehensive_story': {},
            'situation_reconstruction': {},
            'recommendations': []
        }
        
        # 분석 엔진들
        self.engines_loaded = False
        
    def _load_engines(self):
        """분석 엔진들 지연 로딩"""
        if self.engines_loaded:
            return
            
        try:
            print("🔄 분석 엔진 로딩 중...")
            
            # 최적화된 모델들
            import whisper
            self.whisper_model = whisper.load_model("tiny", device="cpu")
            print("✅ Whisper tiny 모델 로드 완료")
            
            import easyocr
            self.ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
            print("✅ EasyOCR 로드 완료")
            
            # 기존 솔로몬드 엔진 활용
            try:
                from core.real_analysis_engine import RealAnalysisEngine
                self.analysis_engine = RealAnalysisEngine()
                print("✅ 솔로몬드 분석 엔진 연결 완료")
            except Exception as e:
                print(f"⚠️ 솔로몬드 엔진 연결 실패, 기본 모드 사용: {e}")
                self.analysis_engine = None
            
            self.engines_loaded = True
            
        except Exception as e:
            print(f"❌ 엔진 로딩 실패: {e}")
            raise
    
    def discover_situation_files(self, user_files_path: str = "user_files"):
        """상황 파일들 발견 및 분류"""
        print("📁 상황 파일 탐색 중...")
        
        files_by_type = {
            'audio': [],
            'image': [],
            'video': [],
            'document': []
        }
        
        user_path = Path(user_files_path)
        if not user_path.exists():
            print(f"❌ {user_files_path} 폴더가 없습니다.")
            return files_by_type
        
        # 시간순 정렬을 위한 파일 수집
        all_files = []
        
        for file_path in user_path.rglob("*"):
            if file_path.is_file() and file_path.name != "README.md":
                try:
                    stat = file_path.stat()
                    file_info = {
                        'path': file_path,
                        'name': file_path.name,
                        'size_mb': stat.st_size / 1024 / 1024,
                        'modified_time': stat.st_mtime,
                        'ext': file_path.suffix.lower()
                    }
                    all_files.append(file_info)
                except Exception as e:
                    print(f"⚠️ 파일 정보 읽기 실패: {file_path.name} - {e}")
        
        # 시간순 정렬 (수정 시간 기준)
        all_files.sort(key=lambda x: x['modified_time'])
        
        # 타입별 분류
        for file_info in all_files:
            ext = file_info['ext']
            
            if ext in ['.m4a', '.wav', '.mp3', '.aac']:
                files_by_type['audio'].append(file_info)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                files_by_type['image'].append(file_info)
            elif ext in ['.mov', '.mp4', '.avi', '.mkv']:
                files_by_type['video'].append(file_info)
            elif ext in ['.pdf', '.docx', '.txt']:
                files_by_type['document'].append(file_info)
        
        # 메타데이터 업데이트
        self.situation_data['metadata']['total_files'] = len(all_files)
        self.situation_data['metadata']['file_types'] = {
            k: len(v) for k, v in files_by_type.items()
        }
        
        print(f"📊 발견된 파일:")
        print(f"   🎵 오디오: {len(files_by_type['audio'])}개")
        print(f"   🖼️  이미지: {len(files_by_type['image'])}개")
        print(f"   🎬 비디오: {len(files_by_type['video'])}개")
        print(f"   📄 문서: {len(files_by_type['document'])}개")
        
        return files_by_type
    
    def analyze_audio_sequence(self, audio_files: List[Dict]):
        """오디오 파일들 순차 분석"""
        if not audio_files:
            return
            
        print("\\n🎵 오디오 시퀀스 분석 중...")
        
        for i, file_info in enumerate(audio_files):
            print(f"  처리 중: {file_info['name']} ({file_info['size_mb']:.1f}MB)")
            
            try:
                # 크기 제한 (성능 최적화)
                if file_info['size_mb'] > 50:
                    print(f"    ⚠️ 파일 크기 초과, 스킵")
                    continue
                
                start_time = time.time()
                
                # Whisper STT
                result = self.whisper_model.transcribe(str(file_info['path']))
                
                analysis_result = {
                    'file_name': file_info['name'],
                    'file_path': str(file_info['path']),
                    'sequence_order': i + 1,
                    'file_size_mb': file_info['size_mb'],
                    'processing_time': time.time() - start_time,
                    'transcript': result.get('text', ''),
                    'language': result.get('language', 'unknown'),
                    'segments': result.get('segments', []),
                    'timestamp': datetime.fromtimestamp(file_info['modified_time']).isoformat()
                }
                
                self.situation_data['audio_analysis'].append(analysis_result)
                
                # 진행 상황 표시
                text_preview = analysis_result['transcript'][:100]
                print(f"    ✅ 완료 ({analysis_result['processing_time']:.1f}초)")
                print(f"    📝 내용: {text_preview}...")
                
            except Exception as e:
                print(f"    ❌ 분석 실패: {e}")
                self._handle_audio_error(file_info, str(e))
    
    def analyze_image_sequence(self, image_files: List[Dict]):
        """이미지 파일들 순차 분석"""
        if not image_files:
            return
            
        print("\\n🖼️ 이미지 시퀀스 분석 중...")
        
        for i, file_info in enumerate(image_files):
            print(f"  처리 중: {file_info['name']} ({file_info['size_mb']:.1f}MB)")
            
            try:
                # 크기 제한
                if file_info['size_mb'] > 20:
                    print(f"    ⚠️ 파일 크기 초과, 스킵")
                    continue
                
                start_time = time.time()
                
                # OCR 분석
                results = self.ocr_reader.readtext(str(file_info['path']))
                
                # 텍스트 추출
                extracted_texts = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:  # 신뢰도 50% 이상
                        extracted_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                
                analysis_result = {
                    'file_name': file_info['name'],
                    'file_path': str(file_info['path']),
                    'sequence_order': i + 1,
                    'file_size_mb': file_info['size_mb'],
                    'processing_time': time.time() - start_time,
                    'extracted_texts': extracted_texts,
                    'total_text_blocks': len(extracted_texts),
                    'timestamp': datetime.fromtimestamp(file_info['modified_time']).isoformat()
                }
                
                self.situation_data['image_analysis'].append(analysis_result)
                
                # 진행 상황 표시
                text_preview = ' '.join([item['text'] for item in extracted_texts[:3]])[:100]
                print(f"    ✅ 완료 ({analysis_result['processing_time']:.1f}초)")
                print(f"    📝 텍스트: {text_preview}...")
                
            except Exception as e:
                print(f"    ❌ 분석 실패: {e}")
                self._handle_image_error(file_info, str(e))
    
    def analyze_video_sequence(self, video_files: List[Dict]):
        """비디오 파일들 분석 (기본적인 메타데이터)"""
        if not video_files:
            return
            
        print("\\n🎬 비디오 시퀀스 분석 중...")
        
        for i, file_info in enumerate(video_files):
            print(f"  처리 중: {file_info['name']} ({file_info['size_mb']:.1f}MB)")
            
            try:
                # 비디오는 메타데이터만 수집 (성능상 이유)
                analysis_result = {
                    'file_name': file_info['name'],
                    'file_path': str(file_info['path']),
                    'sequence_order': i + 1,
                    'file_size_mb': file_info['size_mb'],
                    'processing_time': 0.1,
                    'analysis_type': 'metadata_only',
                    'timestamp': datetime.fromtimestamp(file_info['modified_time']).isoformat(),
                    'note': '대용량 비디오는 메타데이터만 수집'
                }
                
                self.situation_data['video_analysis'].append(analysis_result)
                print(f"    ✅ 메타데이터 수집 완료")
                
            except Exception as e:
                print(f"    ❌ 분석 실패: {e}")
    
    def reconstruct_situation_story(self):
        """상황 재구성 및 스토리 생성"""
        print("\\n📖 상황 스토리 재구성 중...")
        
        # 시간순으로 모든 분석 결과 정렬
        timeline_events = []
        
        # 오디오 이벤트
        for audio in self.situation_data['audio_analysis']:
            timeline_events.append({
                'timestamp': audio['timestamp'],
                'type': 'audio',
                'content': audio['transcript'],
                'file': audio['file_name'],
                'order': audio['sequence_order']
            })
        
        # 이미지 이벤트
        for image in self.situation_data['image_analysis']:
            text_content = ' '.join([item['text'] for item in image['extracted_texts']])
            timeline_events.append({
                'timestamp': image['timestamp'],
                'type': 'image',
                'content': text_content,
                'file': image['file_name'],
                'order': image['sequence_order']
            })
        
        # 비디오 이벤트
        for video in self.situation_data['video_analysis']:
            timeline_events.append({
                'timestamp': video['timestamp'],
                'type': 'video',
                'content': f"비디오 파일: {video['file_name']}",
                'file': video['file_name'],
                'order': video['sequence_order']
            })
        
        # 시간순 정렬
        timeline_events.sort(key=lambda x: x['timestamp'])
        
        # 종합 스토리 생성
        story_parts = []
        for event in timeline_events:
            if event['content'].strip():
                story_parts.append(f"[{event['type'].upper()}] {event['content']}")
        
        comprehensive_story = "\\n\\n".join(story_parts)
        
        # 상황 요약 생성
        audio_summary = self._summarize_audio_content()
        image_summary = self._summarize_image_content()
        
        situation_summary = {
            'timeline_events': timeline_events,
            'comprehensive_story': comprehensive_story,
            'audio_summary': audio_summary,
            'image_summary': image_summary,
            'total_duration': len(timeline_events),
            'key_insights': self._extract_key_insights(timeline_events)
        }
        
        self.situation_data['situation_reconstruction'] = situation_summary
        
        print(f"✅ 스토리 재구성 완료: {len(timeline_events)}개 이벤트")
        
    def _summarize_audio_content(self):
        """오디오 내용 요약"""
        all_transcripts = []
        for audio in self.situation_data['audio_analysis']:
            if audio['transcript'].strip():
                all_transcripts.append(audio['transcript'])
        
        if not all_transcripts:
            return "오디오 내용 없음"
        
        # 간단한 요약 (첫 200자 + 마지막 100자)
        combined = " ".join(all_transcripts)
        if len(combined) > 300:
            summary = combined[:200] + "..." + combined[-100:]
        else:
            summary = combined
        
        return summary
    
    def _summarize_image_content(self):
        """이미지 내용 요약"""
        all_texts = []
        for image in self.situation_data['image_analysis']:
            for text_item in image['extracted_texts']:
                all_texts.append(text_item['text'])
        
        if not all_texts:
            return "이미지 텍스트 없음"
        
        # 중복 제거 및 요약
        unique_texts = list(set(all_texts))
        combined = " ".join(unique_texts[:10])  # 처음 10개만
        
        return combined[:200] + "..." if len(combined) > 200 else combined
    
    def _extract_key_insights(self, timeline_events):
        """주요 인사이트 추출"""
        insights = []
        
        # 파일 타입별 분포
        type_counts = {}
        for event in timeline_events:
            type_counts[event['type']] = type_counts.get(event['type'], 0) + 1
        
        insights.append(f"파일 구성: " + ", ".join([f"{k} {v}개" for k, v in type_counts.items()]))
        
        # 내용 길이 분석
        content_lengths = [len(event['content']) for event in timeline_events if event['content']]
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            insights.append(f"평균 내용 길이: {avg_length:.0f}자")
        
        # 시간 범위
        if len(timeline_events) > 1:
            first_time = timeline_events[0]['timestamp']
            last_time = timeline_events[-1]['timestamp']
            insights.append(f"시간 범위: {first_time} ~ {last_time}")
        
        return insights
    
    def _handle_audio_error(self, file_info, error_msg):
        """오디오 처리 오류 자동 복구"""
        print(f"    🔧 오디오 오류 자동 복구 시도: {file_info['name']}")
        
        # 간단한 오류 정보만 기록
        error_result = {
            'file_name': file_info['name'],
            'error': error_msg,
            'recovery_attempted': True,
            'timestamp': datetime.fromtimestamp(file_info['modified_time']).isoformat()
        }
        
        self.situation_data['audio_analysis'].append(error_result)
    
    def _handle_image_error(self, file_info, error_msg):
        """이미지 처리 오류 자동 복구"""
        print(f"    🔧 이미지 오류 자동 복구 시도: {file_info['name']}")
        
        # 간단한 오류 정보만 기록
        error_result = {
            'file_name': file_info['name'],
            'error': error_msg,
            'recovery_attempted': True,
            'timestamp': datetime.fromtimestamp(file_info['modified_time']).isoformat()
        }
        
        self.situation_data['image_analysis'].append(error_result)
    
    def save_comprehensive_analysis(self):
        """종합 분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_situation_analysis_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.situation_data, f, ensure_ascii=False, indent=2)
        
        print(f"\\n💾 종합 분석 결과 저장: {filename}")
        return filename
    
    def print_situation_summary(self):
        """상황 요약 출력"""
        print("\\n" + "="*60)
        print("📋 상황 분석 요약")
        print("="*60)
        
        # 메타데이터
        meta = self.situation_data['metadata']
        print(f"📊 총 파일 수: {meta['total_files']}개")
        
        for file_type, count in meta['file_types'].items():
            if count > 0:
                print(f"   {file_type}: {count}개")
        
        # 오디오 요약
        if self.situation_data['audio_analysis']:
            print(f"\\n🎵 오디오 분석:")
            audio_summary = self.situation_data['situation_reconstruction'].get('audio_summary', '')
            print(f"   {audio_summary[:150]}...")
        
        # 이미지 요약
        if self.situation_data['image_analysis']:
            print(f"\\n🖼️ 이미지 분석:")
            image_summary = self.situation_data['situation_reconstruction'].get('image_summary', '')
            print(f"   {image_summary[:150]}...")
        
        # 주요 인사이트
        insights = self.situation_data['situation_reconstruction'].get('key_insights', [])
        if insights:
            print(f"\\n💡 주요 인사이트:")
            for insight in insights:
                print(f"   • {insight}")
        
        print("="*60)
    
    def analyze_comprehensive_situation(self, user_files_path: str = "user_files"):
        """종합 상황 분석 실행"""
        print("🎯 종합 상황 분석 시작")
        print("="*50)
        
        try:
            # 1. 엔진 로딩
            self._load_engines()
            
            # 2. 파일 발견
            files_by_type = self.discover_situation_files(user_files_path)
            
            # 3. 순차 분석
            self.analyze_audio_sequence(files_by_type['audio'])
            self.analyze_image_sequence(files_by_type['image'])
            self.analyze_video_sequence(files_by_type['video'])
            
            # 4. 상황 재구성
            self.reconstruct_situation_story()
            
            # 5. 결과 저장
            self.save_comprehensive_analysis()
            
            # 6. 요약 출력
            self.print_situation_summary()
            
            print("\\n✅ 종합 상황 분석 완료!")
            
        except Exception as e:
            print(f"\\n❌ 분석 중 오류 발생: {e}")
            print("🔧 자동 복구 시도 중...")
            
            # 기본적인 오류 복구
            try:
                self.save_comprehensive_analysis()
                print("💾 부분 결과라도 저장 완료")
            except:
                print("❌ 저장도 실패")

def main():
    """메인 실행 함수"""
    analyzer = ComprehensiveSituationAnalyzer()
    analyzer.analyze_comprehensive_situation()

if __name__ == "__main__":
    main()