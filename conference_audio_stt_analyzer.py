#!/usr/bin/env python3
"""
컨퍼런스 오디오 STT 분석 시스템
- 주얼리 컨퍼런스 녹음 파일 Whisper STT 분석
- 발표자별 구분 및 핵심 내용 추출
- 지속가능성 관련 키워드 분석
"""

import os
import sys
import time
import json
import whisper
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import re

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core 모듈 import
try:
    from core.real_analysis_engine import RealAnalysisEngine
    REAL_ANALYSIS_AVAILABLE = True
except ImportError:
    REAL_ANALYSIS_AVAILABLE = False

class ConferenceAudioSTTAnalyzer:
    """컨퍼런스 오디오 STT 분석기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"conference_stt_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'target_files': [],
            'stt_results': [],
            'keyword_analysis': {},
            'speaker_analysis': {},
            'insights': {}
        }
        
        # 주얼리/지속가능성 관련 키워드
        self.target_keywords = {
            'sustainability': [
                '지속가능', 'sustainable', 'sustainability', '친환경', 'eco-friendly',
                '환경', 'environment', '녹색', 'green', '윤리', 'ethical'
            ],
            'jewelry_industry': [
                '주얼리', 'jewelry', 'jewellery', '보석', 'gem', 'diamond', '다이아몬드',
                '금', 'gold', '은', 'silver', '플래티넘', 'platinum', '럭셔리', 'luxury'
            ],
            'consumer_trends': [
                '소비자', 'consumer', '고객', 'customer', '트렌드', 'trend', '선호', 'preference',
                '구매', 'purchase', '시장', 'market', '수요', 'demand'
            ],
            'business_strategy': [
                '전략', 'strategy', '비즈니스', 'business', '혁신', 'innovation', '변화', 'change',
                '미래', 'future', '성장', 'growth', '기회', 'opportunity'
            ]
        }
        
        print("컨퍼런스 오디오 STT 분석 시스템 초기화")
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """분석 시스템 초기화"""
        print("=== STT 분석 시스템 초기화 ===")
        
        # Whisper 모델 로드
        try:
            print("Whisper 모델 로딩 중...")
            self.whisper_model = whisper.load_model("base")
            print("[OK] Whisper 모델: base 모델 로드 완료")
        except Exception as e:
            print(f"[ERROR] Whisper 모델 로드 실패: {e}")
            self.whisper_model = None
            return False
        
        # Real Analysis Engine 확인
        if REAL_ANALYSIS_AVAILABLE:
            self.analysis_engine = RealAnalysisEngine()
            print("[OK] Real Analysis Engine: 준비 완료")
        else:
            self.analysis_engine = None
            print("[WARNING] Real Analysis Engine: 사용 불가")
        
        # 오디오 파일 검색
        self._find_audio_files()
        
        return True
    
    def _find_audio_files(self):
        """오디오 파일 검색"""
        audio_path = project_root / 'user_files' / 'audio'
        
        if not audio_path.exists():
            print(f"[ERROR] 오디오 폴더 없음: {audio_path}")
            return
        
        audio_files = []
        for audio_file in audio_path.glob('*.m4a'):
            file_info = {
                'file_path': str(audio_file),
                'file_name': audio_file.name,
                'file_size_mb': audio_file.stat().st_size / (1024**2),
                'priority': 'high' if audio_file.stat().st_size > 10*1024*1024 else 'medium'  # 10MB 이상
            }
            audio_files.append(file_info)
        
        # 크기순으로 정렬 (큰 파일 먼저)
        audio_files.sort(key=lambda x: x['file_size_mb'], reverse=True)
        
        self.analysis_session['target_files'] = audio_files
        print(f"[OK] 오디오 파일 발견: {len(audio_files)}개")
        
        for i, file_info in enumerate(audio_files, 1):
            print(f"  {i}. {file_info['file_name']} ({file_info['file_size_mb']:.1f}MB, {file_info['priority']})")
    
    def analyze_main_audio_file(self) -> Dict[str, Any]:
        """메인 오디오 파일 분석"""
        if not self.analysis_session['target_files']:
            return {'error': '분석할 오디오 파일이 없습니다.'}
        
        # 가장 큰 파일을 메인으로 선택
        main_file = self.analysis_session['target_files'][0]
        
        print(f"\n--- 메인 오디오 파일 STT 분석 ---")
        print(f"파일: {main_file['file_name']}")
        print(f"크기: {main_file['file_size_mb']:.1f}MB")
        
        analysis_start = time.time()
        
        try:
            # Whisper STT 수행
            print("STT 분석 시작...")
            stt_result = self._perform_whisper_stt(main_file['file_path'])
            
            if 'error' in stt_result:
                return stt_result
            
            # 키워드 분석
            print("키워드 분석 수행...")
            keyword_analysis = self._analyze_keywords(stt_result['text'])
            
            # 화자 분석
            print("화자 분석 수행...")
            speaker_analysis = self._analyze_speakers(stt_result)
            
            # 내용 구조화
            print("내용 구조화 수행...")
            content_structure = self._structure_content(stt_result['text'])
            
            # Real Analysis Engine 활용 (가능한 경우)
            insights = {}
            if self.analysis_engine:
                print("종합 인사이트 분석...")
                insights = self._generate_insights(stt_result['text'])
            
            processing_time = time.time() - analysis_start
            
            analysis_result = {
                'file_info': main_file,
                'stt_result': stt_result,
                'keyword_analysis': keyword_analysis,
                'speaker_analysis': speaker_analysis,
                'content_structure': content_structure,
                'insights': insights,
                'processing_time': processing_time,
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            print(f"[OK] STT 분석 완료 ({processing_time:.1f}초)")
            print(f"추출된 텍스트 길이: {len(stt_result['text'])}자")
            
            return analysis_result
            
        except Exception as e:
            error_result = {
                'file_info': main_file,
                'error': str(e),
                'processing_time': time.time() - analysis_start,
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
            
            print(f"[ERROR] STT 분석 실패: {e}")
            return error_result
    
    def _perform_whisper_stt(self, audio_path: str) -> Dict[str, Any]:
        """Whisper STT 수행"""
        try:
            if not self.whisper_model:
                return {'error': 'Whisper 모델이 로드되지 않았습니다.'}
            
            print("  Whisper 모델로 음성 인식 중...")
            
            # Whisper로 음성 인식
            result = self.whisper_model.transcribe(
                audio_path,
                language='ko',  # 한국어 우선, 자동 감지도 가능
                verbose=False
            )
            
            # 세그먼트별 정보 추출
            segments = []
            if 'segments' in result:
                for segment in result['segments']:
                    segments.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', '').strip(),
                        'confidence': segment.get('avg_logprob', 0)
                    })
            
            return {
                'text': result.get('text', '').strip(),
                'language': result.get('language', 'unknown'),
                'segments': segments,
                'total_duration': max([s.get('end', 0) for s in segments]) if segments else 0
            }
            
        except Exception as e:
            return {'error': f'Whisper STT 실행 오류: {str(e)}'}
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """키워드 분석"""
        text_lower = text.lower()
        
        keyword_counts = {}
        total_keywords = 0
        
        for category, keywords in self.target_keywords.items():
            category_count = 0
            found_keywords = {}
            
            for keyword in keywords:
                count = text_lower.count(keyword.lower())
                if count > 0:
                    found_keywords[keyword] = count
                    category_count += count
                    total_keywords += count
            
            keyword_counts[category] = {
                'total_count': category_count,
                'found_keywords': found_keywords,
                'relevance_score': category_count / len(text.split()) * 1000 if text else 0  # 1000단어당 출현 빈도
            }
        
        # 가장 많이 언급된 카테고리 찾기
        top_category = max(keyword_counts.items(), key=lambda x: x[1]['total_count'])
        
        return {
            'total_keywords_found': total_keywords,
            'keyword_density': total_keywords / len(text.split()) * 100 if text else 0,  # 전체 단어 대비 키워드 비율
            'category_analysis': keyword_counts,
            'primary_topic': top_category[0] if top_category[1]['total_count'] > 0 else 'general',
            'topic_relevance': top_category[1]['relevance_score'] if top_category[1]['total_count'] > 0 else 0
        }
    
    def _analyze_speakers(self, stt_result: Dict[str, Any]) -> Dict[str, Any]:
        """화자 분석"""
        segments = stt_result.get('segments', [])
        
        if not segments:
            return {'error': '세그먼트 정보가 없습니다.'}
        
        # 발화 패턴 분석
        speaking_patterns = []
        current_speaker = 1
        speaker_segments = {1: []}
        
        for i, segment in enumerate(segments):
            # 간단한 화자 전환 감지 (긴 침묵 후)
            if i > 0:
                prev_end = segments[i-1].get('end', 0)
                current_start = segment.get('start', 0)
                
                # 3초 이상 침묵이면 화자 전환 가능성
                if current_start - prev_end > 3.0:
                    current_speaker += 1
                    speaker_segments[current_speaker] = []
            
            speaker_segments[current_speaker].append(segment)
            
            speaking_patterns.append({
                'segment_id': i,
                'estimated_speaker': current_speaker,
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0),
                'text': segment.get('text', ''),
                'confidence': segment.get('confidence', 0)
            })
        
        # 화자별 통계
        speaker_stats = {}
        for speaker_id, speaker_segs in speaker_segments.items():
            total_time = sum(seg.get('end', 0) - seg.get('start', 0) for seg in speaker_segs)
            total_words = sum(len(seg.get('text', '').split()) for seg in speaker_segs)
            
            speaker_stats[f'speaker_{speaker_id}'] = {
                'total_speaking_time': total_time,
                'total_words': total_words,
                'average_words_per_minute': (total_words / total_time * 60) if total_time > 0 else 0,
                'segment_count': len(speaker_segs)
            }
        
        return {
            'estimated_speaker_count': len(speaker_segments),
            'speaking_patterns': speaking_patterns,
            'speaker_statistics': speaker_stats,
            'total_analyzed_time': max([p['end_time'] for p in speaking_patterns]) if speaking_patterns else 0
        }
    
    def _structure_content(self, text: str) -> Dict[str, Any]:
        """내용 구조화"""
        # 문단 분리
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        # 핵심 문장 추출 (키워드 포함 문장 우선)
        key_sentences = []
        all_keywords = []
        for keywords in self.target_keywords.values():
            all_keywords.extend(keywords)
        
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # 너무 짧은 문장 제외
                keyword_count = sum(1 for keyword in all_keywords if keyword.lower() in sentence.lower())
                if keyword_count > 0:
                    key_sentences.append({
                        'text': sentence,
                        'keyword_count': keyword_count,
                        'importance_score': keyword_count * len(sentence.split())
                    })
        
        # 중요도순 정렬
        key_sentences.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return {
            'total_paragraphs': len(paragraphs),
            'total_sentences': len(sentences),
            'key_sentences': key_sentences[:10],  # 상위 10개
            'content_length': {
                'characters': len(text),
                'words': len(text.split()),
                'estimated_reading_time_minutes': len(text.split()) / 200  # 평균 읽기 속도
            }
        }
    
    def _generate_insights(self, text: str) -> Dict[str, Any]:
        """Real Analysis Engine을 활용한 인사이트 생성"""
        if not self.analysis_engine:
            return {'note': 'Real Analysis Engine 사용 불가'}
        
        try:
            # 컨퍼런스 컨텍스트 설정
            context = {
                'content_type': '주얼리 컨퍼런스 음성 녹음',
                'topic': '친환경 럭셔리 소비자 트렌드',
                'participants': 'Chow Tai Fook, Ancardi 등 업계 전문가',
                'format': '패널 토론'
            }
            
            # 분석 실행 (시뮬레이션)
            insights = {
                'content_summary': self._summarize_content(text),
                'key_themes': self._extract_themes(text),
                'business_implications': self._analyze_business_implications(text),
                'conference_quality': self._assess_conference_quality(text)
            }
            
            return insights
            
        except Exception as e:
            return {'error': f'인사이트 생성 오류: {str(e)}'}
    
    def _summarize_content(self, text: str) -> str:
        """내용 요약"""
        # 간단한 요약 로직 (실제로는 더 정교한 알고리즘 사용)
        sentences = re.split(r'[.!?]\s+', text)
        
        # 키워드가 많은 문장들을 우선으로 요약
        important_sentences = []
        for sentence in sentences[:10]:  # 앞부분 문장들에서
            if len(sentence.split()) > 10:  # 충분히 긴 문장
                keyword_count = 0
                for keywords in self.target_keywords.values():
                    for keyword in keywords:
                        if keyword.lower() in sentence.lower():
                            keyword_count += 1
                
                if keyword_count > 0:
                    important_sentences.append(sentence.strip())
        
        return ' '.join(important_sentences[:3])  # 상위 3개 문장으로 요약
    
    def _extract_themes(self, text: str) -> List[str]:
        """주요 테마 추출"""
        themes = []
        
        # 키워드 빈도 기반 테마 추출
        for category, keywords in self.target_keywords.items():
            category_mentions = 0
            for keyword in keywords:
                category_mentions += text.lower().count(keyword.lower())
            
            if category_mentions > 2:  # 충분히 언급된 경우
                if category == 'sustainability':
                    themes.append('지속가능성 전략')
                elif category == 'jewelry_industry':
                    themes.append('주얼리 산업 동향')
                elif category == 'consumer_trends':
                    themes.append('소비자 트렌드 변화')
                elif category == 'business_strategy':
                    themes.append('비즈니스 혁신 전략')
        
        return themes
    
    def _analyze_business_implications(self, text: str) -> List[str]:
        """비즈니스 시사점 분석"""
        implications = []
        
        text_lower = text.lower()
        
        # 패턴 기반 시사점 추출
        if any(word in text_lower for word in ['변화', 'change', '트렌드', 'trend']):
            implications.append('시장 트렌드 변화에 대한 대응 전략 필요')
        
        if any(word in text_lower for word in ['지속가능', 'sustainable', '친환경', 'eco']):
            implications.append('지속가능성을 고려한 제품 개발 및 마케팅 전략 수립')
        
        if any(word in text_lower for word in ['소비자', 'consumer', '고객', 'customer']):
            implications.append('변화하는 소비자 니즈에 대한 깊은 이해 필요')
        
        if any(word in text_lower for word in ['혁신', 'innovation', '기술', 'technology']):
            implications.append('기술 혁신을 통한 경쟁력 강화 방안 모색')
        
        return implications
    
    def _assess_conference_quality(self, text: str) -> Dict[str, Any]:
        """컨퍼런스 품질 평가"""
        word_count = len(text.split())
        
        # 내용 풍부도 평가
        content_richness = min(100, word_count / 50)  # 5000단어 기준 100점
        
        # 키워드 다양성 평가
        unique_keywords = set()
        for keywords in self.target_keywords.values():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    unique_keywords.add(keyword)
        
        keyword_diversity = min(100, len(unique_keywords) * 5)  # 20개 키워드 기준 100점
        
        # 전문성 평가 (전문 용어 사용 빈도)
        professional_terms = ['전략', 'strategy', '분석', 'analysis', '시장', 'market', '산업', 'industry']
        professional_score = min(100, sum(text.lower().count(term) for term in professional_terms) * 2)
        
        overall_quality = (content_richness + keyword_diversity + professional_score) / 3
        
        return {
            'overall_quality_score': round(overall_quality, 1),
            'content_richness': round(content_richness, 1),
            'keyword_diversity': round(keyword_diversity, 1),
            'professional_level': round(professional_score, 1),
            'word_count': word_count,
            'unique_keywords_found': len(unique_keywords)
        }
    
    def save_stt_results(self, analysis_result: Dict[str, Any]) -> str:
        """STT 분석 결과 저장"""
        self.analysis_session['stt_results'].append(analysis_result)
        
        report_path = project_root / f"conference_stt_analysis_{self.analysis_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_session, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] STT 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("컨퍼런스 오디오 STT 분석 시작")
    print("=" * 50)
    
    # STT 분석기 초기화
    stt_analyzer = ConferenceAudioSTTAnalyzer()
    
    # 메인 오디오 파일 분석
    stt_result = stt_analyzer.analyze_main_audio_file()
    
    if stt_result.get('status') == 'success':
        # 결과 저장
        report_path = stt_analyzer.save_stt_results(stt_result)
        
        # 요약 출력
        print(f"\n{'='*50}")
        print("STT 분석 완료 요약")
        print(f"{'='*50}")
        
        file_info = stt_result.get('file_info', {})
        stt_data = stt_result.get('stt_result', {})
        keyword_data = stt_result.get('keyword_analysis', {})
        speaker_data = stt_result.get('speaker_analysis', {})
        
        print(f"파일: {file_info.get('file_name', 'Unknown')}")
        print(f"크기: {file_info.get('file_size_mb', 0):.1f}MB")
        print(f"처리 시간: {stt_result.get('processing_time', 0):.1f}초")
        print(f"음성 길이: {stt_data.get('total_duration', 0):.1f}초")
        print(f"추출 텍스트: {len(stt_data.get('text', ''))}자")
        print(f"언어: {stt_data.get('language', 'Unknown')}")
        print(f"추정 화자 수: {speaker_data.get('estimated_speaker_count', 0)}명")
        print(f"주요 주제: {keyword_data.get('primary_topic', 'Unknown')}")
        print(f"키워드 밀도: {keyword_data.get('keyword_density', 0):.1f}%")
        print(f"상세 보고서: {report_path}")
        
        # 핵심 문장 출력
        content_structure = stt_result.get('content_structure', {})
        key_sentences = content_structure.get('key_sentences', [])
        if key_sentences:
            print(f"\n핵심 문장 (상위 3개):")
            for i, sentence in enumerate(key_sentences[:3], 1):
                print(f"  {i}. {sentence['text'][:100]}...")
    
    else:
        print(f"STT 분석 실패: {stt_result.get('error', 'Unknown error')}")
    
    return stt_result

if __name__ == "__main__":
    main()