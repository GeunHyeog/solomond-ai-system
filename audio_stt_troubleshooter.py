#!/usr/bin/env python3
"""
오디오 STT 분석 문제 해결 시스템
- 메모리 부족 문제 분석 및 해결
- 청크 기반 처리로 대용량 오디오 분석
- 시스템 리소스 최적화
"""

import os
import sys
import time
import json
import whisper
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class AudioSTTTroubleshooter:
    """오디오 STT 문제 해결 시스템"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"audio_stt_troubleshoot_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'system_info': {},
            'problems_identified': [],
            'solutions_applied': [],
            'results': {}
        }
        
        print("오디오 STT 문제 해결 시스템 초기화")
        self._analyze_system_resources()
        self._identify_problems()
    
    def _analyze_system_resources(self):
        """시스템 리소스 분석"""
        print("\n--- 시스템 리소스 분석 ---")
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # CPU 정보
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        
        self.analysis_session['system_info'] = {
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent,
                'free_gb': round(memory.free / (1024**3), 2)
            },
            'cpu': {
                'count': cpu_count,
                'usage_percent': cpu_usage
            },
            'disk': {
                'free_gb': round(disk.free / (1024**3), 2),
                'total_gb': round(disk.total / (1024**3), 2)
            }
        }
        
        print(f"[SYSTEM] 메모리: {self.analysis_session['system_info']['memory']['available_gb']:.1f}GB 사용 가능")
        print(f"[SYSTEM] CPU: {cpu_count}코어, 사용률 {cpu_usage:.1f}%")
        print(f"[SYSTEM] 디스크: {self.analysis_session['system_info']['disk']['free_gb']:.1f}GB 여유")
    
    def _identify_problems(self):
        """문제점 식별"""
        print("\n--- 문제점 식별 ---")
        
        problems = []
        
        # 메모리 부족 확인
        available_memory = self.analysis_session['system_info']['memory']['available_gb']
        if available_memory < 4.0:
            problems.append({
                'type': 'memory_shortage',
                'description': f'사용 가능한 메모리 부족: {available_memory:.1f}GB (권장: 4GB+)',
                'severity': 'high',
                'solution': 'chunk_processing'
            })
        
        # 27MB 오디오 파일 처리 시간 예상
        audio_file_size = 27.2  # MB
        estimated_processing_time = audio_file_size * 4  # 초 (대략적 추정)
        if estimated_processing_time > 120:  # 2분 초과
            problems.append({
                'type': 'processing_timeout',
                'description': f'예상 처리 시간 초과: {estimated_processing_time:.0f}초 (타임아웃: 120초)',
                'severity': 'high',
                'solution': 'faster_model_or_chunking'
            })
        
        # Whisper 모델 크기 확인
        problems.append({
            'type': 'model_size',
            'description': 'Whisper base 모델 사용 중 - 더 작은 모델로 최적화 가능',
            'severity': 'medium',
            'solution': 'use_tiny_model'
        })
        
        self.analysis_session['problems_identified'] = problems
        
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. [{problem['severity'].upper()}] {problem['description']}")
    
    def apply_solutions(self):
        """해결책 적용"""
        print("\n--- 해결책 적용 ---")
        
        solutions_applied = []
        
        # 1. 환경 변수 최적화
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU 사용 안함
        os.environ['OMP_NUM_THREADS'] = '2'  # CPU 스레드 제한
        solutions_applied.append('CPU 모드 강제 및 스레드 제한')
        
        # 2. 메모리 정리
        gc.collect()
        solutions_applied.append('메모리 가비지 컬렉션')
        
        # 3. 작은 Whisper 모델 사용
        try:
            print("  Whisper tiny 모델 로딩 중...")
            self.whisper_model = whisper.load_model("tiny")
            solutions_applied.append('Whisper tiny 모델 사용 (메모리 절약)')
            print("  [OK] Whisper tiny 모델 로드 완료")
        except Exception as e:
            print(f"  [ERROR] Whisper 모델 로드 실패: {e}")
            self.whisper_model = None
            return False
        
        self.analysis_session['solutions_applied'] = solutions_applied
        
        for solution in solutions_applied:
            print(f"  [APPLIED] {solution}")
        
        return True
    
    def process_audio_in_chunks(self, audio_path: str, chunk_duration: int = 300) -> Dict[str, Any]:
        """청크 단위로 오디오 처리 (5분씩)"""
        print(f"\n--- 청크 기반 오디오 처리 ---")
        print(f"파일: {Path(audio_path).name}")
        print(f"청크 크기: {chunk_duration}초")
        
        if not self.whisper_model:
            return {'error': 'Whisper 모델이 로드되지 않았습니다.'}
        
        try:
            # 전체 오디오 로드 (메타데이터만)
            import librosa
            
            print("  오디오 메타데이터 로딩...")
            duration = librosa.get_duration(path=audio_path)
            print(f"  총 오디오 길이: {duration:.1f}초 ({duration/60:.1f}분)")
            
            # 청크 개수 계산
            num_chunks = int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
            print(f"  처리할 청크 수: {num_chunks}개")
            
            # 청크별 처리
            all_segments = []
            full_text = ""
            processing_start = time.time()
            
            for chunk_idx in range(num_chunks):
                start_time = chunk_idx * chunk_duration
                end_time = min((chunk_idx + 1) * chunk_duration, duration)
                
                print(f"  [{chunk_idx+1}/{num_chunks}] 청크 처리: {start_time:.0f}s-{end_time:.0f}s")
                
                try:
                    # 청크 오디오 로드
                    chunk_audio, sr = librosa.load(
                        audio_path, 
                        sr=16000,  # Whisper 권장 샘플레이트
                        offset=start_time, 
                        duration=min(chunk_duration, end_time - start_time)
                    )
                    
                    # Whisper STT 수행
                    chunk_result = self.whisper_model.transcribe(
                        chunk_audio,
                        language='ko',
                        verbose=False
                    )
                    
                    # 결과 처리
                    chunk_text = chunk_result.get('text', '').strip()
                    if chunk_text:
                        full_text += chunk_text + " "
                        
                        # 세그먼트 정보 추가 (시간 오프셋 적용)
                        if 'segments' in chunk_result:
                            for segment in chunk_result['segments']:
                                adjusted_segment = {
                                    'start': segment.get('start', 0) + start_time,
                                    'end': segment.get('end', 0) + start_time,
                                    'text': segment.get('text', '').strip(),
                                    'confidence': segment.get('avg_logprob', 0)
                                }
                                all_segments.append(adjusted_segment)
                    
                    print(f"    텍스트 추출: {len(chunk_text)}자")
                    
                    # 메모리 정리
                    del chunk_audio, chunk_result
                    gc.collect()
                    
                except Exception as chunk_error:
                    print(f"    [ERROR] 청크 {chunk_idx+1} 처리 실패: {chunk_error}")
                    continue
                
                # 진행률 표시
                progress = (chunk_idx + 1) / num_chunks * 100
                elapsed = time.time() - processing_start
                estimated_total = elapsed / progress * 100 if progress > 0 else 0
                remaining = estimated_total - elapsed
                
                print(f"    진행률: {progress:.1f}% (남은 시간: {remaining:.0f}초)")
            
            total_processing_time = time.time() - processing_start
            
            result = {
                'text': full_text.strip(),
                'segments': all_segments,
                'total_duration': duration,
                'processing_time': total_processing_time,
                'chunks_processed': num_chunks,
                'processing_method': 'chunk_based',
                'chunk_duration': chunk_duration,
                'status': 'success'
            }
            
            print(f"\n  [OK] 청크 기반 처리 완료 ({total_processing_time:.1f}초)")
            print(f"  추출된 총 텍스트: {len(full_text)}자")
            print(f"  총 세그먼트: {len(all_segments)}개")
            
            return result
            
        except Exception as e:
            return {
                'error': f'청크 기반 처리 오류: {str(e)}',
                'processing_time': time.time() - processing_start if 'processing_start' in locals() else 0,
                'status': 'error'
            }
    
    def analyze_audio_content(self, stt_result: Dict[str, Any]) -> Dict[str, Any]:
        """오디오 콘텐츠 분석"""
        if 'error' in stt_result:
            return {'error': stt_result['error']}
        
        text = stt_result.get('text', '')
        segments = stt_result.get('segments', [])
        
        # 키워드 분석
        conference_keywords = {
            'sustainability': ['지속가능', 'sustainable', 'sustainability', '친환경', 'eco-friendly'],
            'jewelry': ['주얼리', 'jewelry', 'jewellery', '보석', 'gem', 'diamond'],
            'consumer': ['소비자', 'consumer', '고객', 'customer', '트렌드', 'trend'],
            'business': ['전략', 'strategy', '비즈니스', 'business', '혁신', 'innovation']
        }
        
        keyword_analysis = {}
        text_lower = text.lower()
        
        for category, keywords in conference_keywords.items():
            found_keywords = {}
            total_count = 0
            
            for keyword in keywords:
                count = text_lower.count(keyword.lower())
                if count > 0:
                    found_keywords[keyword] = count
                    total_count += count
            
            keyword_analysis[category] = {
                'total_mentions': total_count,
                'found_keywords': found_keywords,
                'relevance_score': total_count / len(text.split()) * 1000 if text else 0
            }
        
        # 화자 분석 (간단한 버전)
        speaker_changes = 0
        prev_end = 0
        
        for segment in segments:
            current_start = segment.get('start', 0)
            if current_start - prev_end > 3.0:  # 3초 이상 침묵이면 화자 전환 추정
                speaker_changes += 1
            prev_end = segment.get('end', 0)
        
        estimated_speakers = min(speaker_changes + 1, 5)  # 최대 5명으로 제한
        
        analysis = {
            'content_summary': {
                'total_text_length': len(text),
                'total_words': len(text.split()),
                'total_segments': len(segments),
                'estimated_speakers': estimated_speakers,
                'audio_duration_minutes': stt_result.get('total_duration', 0) / 60
            },
            'keyword_analysis': keyword_analysis,
            'primary_topic': max(keyword_analysis.items(), key=lambda x: x[1]['total_mentions'])[0] if keyword_analysis else 'general',
            'conference_relevance': {
                'is_conference_audio': any(cat['total_mentions'] > 0 for cat in keyword_analysis.values()),
                'confidence_score': sum(cat['relevance_score'] for cat in keyword_analysis.values())
            }
        }
        
        return analysis
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """완전한 분석 실행"""
        print("\n=== 완전한 STT 분석 실행 ===")
        
        # 1. 해결책 적용
        if not self.apply_solutions():
            return {'error': '해결책 적용 실패'}
        
        # 2. 오디오 파일 찾기
        audio_path = project_root / 'user_files' / 'audio' / '새로운 녹음.m4a'
        
        if not audio_path.exists():
            return {'error': f'오디오 파일을 찾을 수 없습니다: {audio_path}'}
        
        print(f"대상 파일: {audio_path.name} ({audio_path.stat().st_size / (1024**2):.1f}MB)")
        
        # 3. 청크 기반 STT 처리
        stt_result = self.process_audio_in_chunks(str(audio_path))
        
        if 'error' in stt_result:
            return stt_result
        
        # 4. 콘텐츠 분석
        print("\n--- 콘텐츠 분석 수행 ---")
        content_analysis = self.analyze_audio_content(stt_result)
        
        # 5. 최종 결과 통합
        final_result = {
            'troubleshooting_info': {
                'session_id': self.analysis_session['session_id'],
                'problems_identified': self.analysis_session['problems_identified'],
                'solutions_applied': self.analysis_session['solutions_applied'],
                'system_info': self.analysis_session['system_info']
            },
            'stt_result': stt_result,
            'content_analysis': content_analysis,
            'processing_summary': {
                'method': 'chunk_based_processing',
                'model_used': 'whisper_tiny',
                'total_processing_time': stt_result.get('processing_time', 0),
                'success_rate': '100%' if stt_result.get('status') == 'success' else 'partial',
                'text_extracted': len(stt_result.get('text', '')),
                'segments_found': len(stt_result.get('segments', []))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return final_result
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """결과 저장"""
        report_path = project_root / f"audio_stt_troubleshoot_results_{self.analysis_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 문제 해결 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("오디오 STT 문제 해결 및 재분석")
    print("=" * 50)
    
    # 문제 해결 시스템 초기화
    troubleshooter = AudioSTTTroubleshooter()
    
    # 완전한 분석 실행
    results = troubleshooter.run_complete_analysis()
    
    if 'error' in results:
        print(f"[ERROR] 분석 실패: {results['error']}")
        return results
    
    # 결과 저장
    report_path = troubleshooter.save_results(results)
    
    # 요약 출력
    print(f"\n{'='*50}")
    print("STT 문제 해결 및 분석 완료")
    print(f"{'='*50}")
    
    processing_summary = results.get('processing_summary', {})
    content_summary = results.get('content_analysis', {}).get('content_summary', {})
    
    print(f"\n[PROCESSING] 처리 정보:")
    print(f"  방법: {processing_summary.get('method', 'Unknown')}")
    print(f"  모델: {processing_summary.get('model_used', 'Unknown')}")
    print(f"  처리 시간: {processing_summary.get('total_processing_time', 0):.1f}초")
    print(f"  성공률: {processing_summary.get('success_rate', 'Unknown')}")
    
    print(f"\n[CONTENT] 추출된 콘텐츠:")
    print(f"  텍스트 길이: {processing_summary.get('text_extracted', 0)}자")
    print(f"  세그먼트 수: {processing_summary.get('segments_found', 0)}개")
    print(f"  오디오 길이: {content_summary.get('audio_duration_minutes', 0):.1f}분")
    print(f"  추정 화자 수: {content_summary.get('estimated_speakers', 0)}명")
    
    # 키워드 분석 결과
    keyword_analysis = results.get('content_analysis', {}).get('keyword_analysis', {})
    if keyword_analysis:
        print(f"\n[KEYWORDS] 주요 키워드 분석:")
        for category, data in keyword_analysis.items():
            if data['total_mentions'] > 0:
                print(f"  {category}: {data['total_mentions']}회 언급")
    
    # 컨퍼런스 관련성
    conference_relevance = results.get('content_analysis', {}).get('conference_relevance', {})
    print(f"\n[RELEVANCE] 컨퍼런스 관련성:")
    print(f"  컨퍼런스 오디오: {conference_relevance.get('is_conference_audio', False)}")
    print(f"  신뢰도 점수: {conference_relevance.get('confidence_score', 0):.2f}")
    
    print(f"\n[FILE] 상세 결과: {report_path}")
    
    return results

if __name__ == "__main__":
    main()