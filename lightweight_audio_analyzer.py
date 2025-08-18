#!/usr/bin/env python3
"""
경량화 오디오 분석기
- 최소한의 리소스로 오디오 분석
- 빠른 메타데이터 추출 및 샘플 분석
- 타임아웃 방지를 위한 최적화
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import librosa
import numpy as np

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class LightweightAudioAnalyzer:
    """경량화 오디오 분석기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"lightweight_audio_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'analysis_approach': 'lightweight_metadata_only',
            'results': {}
        }
        
        print("경량화 오디오 분석기 초기화")
    
    def analyze_audio_metadata(self, audio_path: str) -> Dict[str, Any]:
        """오디오 메타데이터 분석 (빠른 처리)"""
        print(f"\n--- 오디오 메타데이터 분석 ---")
        print(f"파일: {Path(audio_path).name}")
        
        analysis_start = time.time()
        
        try:
            # 기본 파일 정보
            file_path = Path(audio_path)
            file_size_mb = file_path.stat().st_size / (1024**2)
            
            # librosa로 빠른 메타데이터 추출
            print("  메타데이터 로딩 중...")
            duration = librosa.get_duration(path=audio_path)
            
            # 샘플 오디오 로드 (첫 30초만)
            print("  샘플 오디오 분석 중...")
            sample_audio, sr = librosa.load(audio_path, sr=16000, duration=30.0)
            
            # 오디오 특성 분석
            audio_features = self._analyze_audio_features(sample_audio, sr)
            
            # 음성 활동 감지 (간단한 버전)
            voice_activity = self._detect_voice_activity(sample_audio, sr)
            
            processing_time = time.time() - analysis_start
            
            metadata_result = {
                'file_info': {
                    'file_name': file_path.name,
                    'file_size_mb': file_size_mb,
                    'total_duration_seconds': duration,
                    'total_duration_minutes': duration / 60,
                    'sample_rate': sr
                },
                'audio_features': audio_features,
                'voice_activity': voice_activity,
                'conference_assessment': self._assess_conference_relevance(audio_features, voice_activity, duration),
                'processing_info': {
                    'analysis_method': 'metadata_and_sample',
                    'sample_duration': min(30.0, duration),
                    'processing_time': processing_time
                },
                'status': 'success'
            }
            
            print(f"  [OK] 메타데이터 분석 완료 ({processing_time:.2f}초)")
            print(f"  총 길이: {duration/60:.1f}분")
            print(f"  음성 활동: {voice_activity['voice_ratio']*100:.1f}%")
            
            return metadata_result
            
        except Exception as e:
            error_result = {
                'file_info': {'file_name': Path(audio_path).name},
                'error': str(e),
                'processing_time': time.time() - analysis_start,
                'status': 'error'
            }
            
            print(f"  [ERROR] 메타데이터 분석 실패: {e}")
            return error_result
    
    def _analyze_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """오디오 특성 분석"""
        # RMS 에너지 계산
        rms_energy = librosa.feature.rms(y=audio)[0]
        avg_energy = np.mean(rms_energy)
        
        # 제로 크로싱 비율
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        avg_zcr = np.mean(zcr)
        
        # 스펙트럼 센트로이드 (음성 특성)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)
        
        # MFCC 특성 (음성 인식용)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return {
            'energy_level': float(avg_energy),
            'zero_crossing_rate': float(avg_zcr),
            'spectral_centroid': float(avg_centroid),
            'mfcc_features': mfcc_mean.tolist(),
            'audio_quality': 'good' if avg_energy > 0.01 else 'low'
        }
    
    def _detect_voice_activity(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """음성 활동 감지"""
        # 간단한 음성 활동 감지 (에너지 기반)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        # 프레임 단위로 에너지 계산
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)
        
        # 음성 활동 임계값 (동적으로 계산)
        energy_threshold = np.percentile(frame_energies, 30)  # 하위 30% 기준
        
        # 음성 활동 프레임 식별
        voice_frames = frame_energies > energy_threshold
        voice_ratio = np.sum(voice_frames) / len(voice_frames)
        
        # 연속 음성 구간 감지
        voice_segments = []
        in_voice = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_frames):
            time_pos = i * hop_length / sr
            
            if is_voice and not in_voice:
                start_time = time_pos
                in_voice = True
            elif not is_voice and in_voice:
                if time_pos - start_time > 1.0:  # 1초 이상 음성 구간만 포함
                    voice_segments.append({
                        'start': start_time,
                        'end': time_pos,
                        'duration': time_pos - start_time
                    })
                in_voice = False
        
        return {
            'voice_ratio': float(voice_ratio),
            'voice_segments_count': len(voice_segments),
            'longest_voice_segment': float(max([seg['duration'] for seg in voice_segments]) if voice_segments else 0),
            'total_voice_time': float(sum([seg['duration'] for seg in voice_segments])),
            'is_voice_content': bool(voice_ratio > 0.3)  # 30% 이상이면 음성 콘텐츠로 판단
        }
    
    def _assess_conference_relevance(self, audio_features: Dict, voice_activity: Dict, duration: float) -> Dict[str, Any]:
        """컨퍼런스 관련성 평가"""
        relevance_score = 0
        assessment_factors = []
        
        # 1. 길이 평가 (20분 이상이면 컨퍼런스 가능성 높음)
        if duration > 1200:  # 20분
            relevance_score += 30
            assessment_factors.append('장시간 녹음 (컨퍼런스 적합)')
        
        # 2. 음성 활동 비율 평가
        if voice_activity['voice_ratio'] > 0.5:
            relevance_score += 25
            assessment_factors.append('높은 음성 활동 비율')
        
        # 3. 음성 구간 수 평가 (여러 화자 추정)
        if voice_activity['voice_segments_count'] > 10:
            relevance_score += 20
            assessment_factors.append('다수 음성 구간 (다중 화자 가능)')
        
        # 4. 오디오 품질 평가
        if audio_features['audio_quality'] == 'good':
            relevance_score += 15
            assessment_factors.append('양호한 오디오 품질')
        
        # 5. 스펙트럼 특성 평가 (음성 특성)
        if 1000 < audio_features['spectral_centroid'] < 4000:
            relevance_score += 10
            assessment_factors.append('음성 특성 스펙트럼')
        
        # 관련성 레벨 결정
        if relevance_score >= 70:
            relevance_level = 'high'
        elif relevance_score >= 40:
            relevance_level = 'medium'
        else:
            relevance_level = 'low'
        
        return {
            'relevance_score': int(relevance_score),
            'relevance_level': relevance_level,
            'assessment_factors': assessment_factors,
            'is_likely_conference': bool(relevance_score >= 40),
            'recommended_analysis': 'full_stt' if relevance_score >= 60 else 'selective_sampling'
        }
    
    def generate_content_preview(self, metadata_result: Dict[str, Any]) -> Dict[str, Any]:
        """콘텐츠 미리보기 생성"""
        print("\n--- 콘텐츠 미리보기 생성 ---")
        
        if 'error' in metadata_result:
            return {'error': metadata_result['error']}
        
        file_info = metadata_result.get('file_info', {})
        voice_activity = metadata_result.get('voice_activity', {})
        conference_assessment = metadata_result.get('conference_assessment', {})
        
        # 예상 콘텐츠 유형 결정
        duration_minutes = file_info.get('total_duration_minutes', 0)
        
        if duration_minutes > 45:
            expected_content_type = '전체 컨퍼런스 세션 녹음'
        elif duration_minutes > 20:
            expected_content_type = '패널 토론 또는 주요 발표'
        else:
            expected_content_type = '짧은 발표 또는 질의응답'
        
        # 예상 화자 수 추정
        voice_segments = voice_activity.get('voice_segments_count', 0)
        if voice_segments > 30:
            estimated_speakers = '3-5명 (패널 토론)'
        elif voice_segments > 15:
            estimated_speakers = '2-3명 (대화형 세션)'
        else:
            estimated_speakers = '1-2명 (단독 발표)'
        
        # STT 분석 전략 제안
        if conference_assessment.get('relevance_score', 0) >= 70:
            stt_strategy = {
                'recommended_approach': 'full_transcription',
                'estimated_processing_time': f"{duration_minutes * 2:.0f}분",
                'expected_text_length': f"{duration_minutes * 150:.0f}자 (예상)",
                'priority': 'high'
            }
        else:
            stt_strategy = {
                'recommended_approach': 'sample_transcription',
                'estimated_processing_time': f"{min(duration_minutes * 0.5, 10):.0f}분",
                'expected_text_length': f"{min(duration_minutes * 50, 500):.0f}자 (예상)",
                'priority': 'medium'
            }
        
        preview = {
            'content_overview': {
                'file_name': file_info.get('file_name', ''),
                'duration': f"{duration_minutes:.1f}분",
                'file_size': f"{file_info.get('file_size_mb', 0):.1f}MB",
                'expected_content_type': expected_content_type,
                'estimated_speakers': estimated_speakers
            },
            'technical_assessment': {
                'audio_quality': metadata_result.get('audio_features', {}).get('audio_quality', 'unknown'),
                'voice_activity_ratio': f"{voice_activity.get('voice_ratio', 0)*100:.1f}%",
                'conference_relevance': conference_assessment.get('relevance_level', 'unknown'),
                'analysis_confidence': 'high' if voice_activity.get('is_voice_content', False) else 'medium'
            },
            'stt_analysis_strategy': stt_strategy,
            'key_insights_preview': [
                f"총 {duration_minutes:.0f}분 분량의 오디오 콘텐츠",
                f"음성 활동이 {voice_activity.get('voice_ratio', 0)*100:.1f}% 포함",
                f"컨퍼런스 관련성: {conference_assessment.get('relevance_level', 'unknown')}",
                f"{len(conference_assessment.get('assessment_factors', []))}개 긍정적 평가 요소 확인"
            ]
        }
        
        print("[OK] 콘텐츠 미리보기 생성 완료")
        return preview
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """완전한 경량 분석 실행"""
        print("\n=== 경량 오디오 분석 실행 ===")
        
        # 오디오 파일 경로
        audio_path = project_root / 'user_files' / 'audio' / '새로운 녹음.m4a'
        
        if not audio_path.exists():
            return {'error': f'오디오 파일을 찾을 수 없습니다: {audio_path}'}
        
        print(f"대상 파일: {audio_path.name}")
        
        # 1. 메타데이터 분석
        metadata_result = self.analyze_audio_metadata(str(audio_path))
        
        if 'error' in metadata_result:
            return metadata_result
        
        # 2. 콘텐츠 미리보기 생성
        content_preview = self.generate_content_preview(metadata_result)
        
        # 3. 최종 결과 통합
        final_result = {
            'analysis_info': {
                'session_id': self.analysis_session['session_id'],
                'analysis_approach': self.analysis_session['analysis_approach'],
                'timestamp': datetime.now().isoformat()
            },
            'metadata_analysis': metadata_result,
            'content_preview': content_preview,
            'recommendations': {
                'immediate_insights': self._generate_immediate_insights(metadata_result, content_preview),
                'next_steps': self._recommend_next_steps(content_preview)
            }
        }
        
        return final_result
    
    def _generate_immediate_insights(self, metadata: Dict, preview: Dict) -> List[str]:
        """즉시 확인 가능한 인사이트 생성"""
        insights = []
        
        file_info = metadata.get('file_info', {})
        voice_activity = metadata.get('voice_activity', {})
        conference_assessment = metadata.get('conference_assessment', {})
        
        # 파일 크기 및 길이 인사이트
        duration_min = file_info.get('total_duration_minutes', 0)
        if duration_min > 50:
            insights.append(f"장시간 녹음 ({duration_min:.0f}분) - 전체 컨퍼런스 세션으로 추정")
        
        # 음성 활동 인사이트
        if voice_activity.get('voice_ratio', 0) > 0.6:
            insights.append("높은 음성 활동 비율 - 활발한 토론 또는 발표 진행")
        
        # 컨퍼런스 관련성 인사이트
        if conference_assessment.get('is_likely_conference', False):
            insights.append("컨퍼런스 오디오로 확인 - 전문적 분석 가치 높음")
        
        # 다중 화자 가능성
        if voice_activity.get('voice_segments_count', 0) > 20:
            insights.append("다수의 음성 구간 감지 - 패널 토론 형태로 추정")
        
        if not insights:
            insights.append("기본적인 음성 콘텐츠 확인 - 추가 분석으로 상세 내용 파악 가능")
        
        return insights
    
    def _recommend_next_steps(self, preview: Dict) -> List[Dict[str, str]]:
        """다음 단계 권장사항"""
        stt_strategy = preview.get('stt_analysis_strategy', {})
        
        steps = []
        
        if stt_strategy.get('priority') == 'high':
            steps.append({
                'step': 'STT 전체 분석 수행',
                'method': '청크 기반 처리로 전체 텍스트 추출',
                'expected_time': stt_strategy.get('estimated_processing_time', '60분'),
                'value': '컨퍼런스 발언 내용 완전 파악'
            })
        else:
            steps.append({
                'step': 'STT 샘플 분석 수행',
                'method': '핵심 구간 선별 텍스트 추출',
                'expected_time': stt_strategy.get('estimated_processing_time', '10분'),
                'value': '주요 메시지 파악'
            })
        
        steps.append({
            'step': '키워드 분석 수행',
            'method': 'STT 결과에서 컨퍼런스 관련 키워드 추출',
            'expected_time': '5분',
            'value': '핵심 주제 및 화자 의견 요약'
        })
        
        steps.append({
            'step': '이미지 OCR과 통합 분석',
            'method': '음성 내용과 발표 슬라이드 연계 분석',
            'expected_time': '10분',
            'value': '완전한 컨퍼런스 인사이트 도출'
        })
        
        return steps
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """결과 저장"""
        report_path = project_root / f"lightweight_audio_analysis_{self.analysis_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] 경량 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("경량화 오디오 분석")
    print("=" * 40)
    
    # 경량 분석기 초기화
    analyzer = LightweightAudioAnalyzer()
    
    # 완전한 분석 실행
    results = analyzer.run_complete_analysis()
    
    if 'error' in results:
        print(f"[ERROR] 분석 실패: {results['error']}")
        return results
    
    # 결과 저장
    report_path = analyzer.save_results(results)
    
    # 요약 출력
    print(f"\n{'='*40}")
    print("경량 오디오 분석 완료")
    print(f"{'='*40}")
    
    # 콘텐츠 개요
    content_overview = results.get('content_preview', {}).get('content_overview', {})
    print(f"\n[CONTENT] 콘텐츠 개요:")
    print(f"  파일: {content_overview.get('file_name', 'Unknown')}")
    print(f"  길이: {content_overview.get('duration', 'Unknown')}")
    print(f"  크기: {content_overview.get('file_size', 'Unknown')}")
    print(f"  유형: {content_overview.get('expected_content_type', 'Unknown')}")
    print(f"  화자: {content_overview.get('estimated_speakers', 'Unknown')}")
    
    # 기술적 평가
    tech_assessment = results.get('content_preview', {}).get('technical_assessment', {})
    print(f"\n[TECHNICAL] 기술적 평가:")
    print(f"  오디오 품질: {tech_assessment.get('audio_quality', 'Unknown')}")
    print(f"  음성 활동: {tech_assessment.get('voice_activity_ratio', 'Unknown')}")
    print(f"  컨퍼런스 관련성: {tech_assessment.get('conference_relevance', 'Unknown')}")
    print(f"  분석 신뢰도: {tech_assessment.get('analysis_confidence', 'Unknown')}")
    
    # 즉시 인사이트
    immediate_insights = results.get('recommendations', {}).get('immediate_insights', [])
    print(f"\n[INSIGHTS] 즉시 확인 인사이트:")
    for i, insight in enumerate(immediate_insights, 1):
        print(f"  {i}. {insight}")
    
    # 다음 단계
    next_steps = results.get('recommendations', {}).get('next_steps', [])
    print(f"\n[NEXT] 권장 다음 단계:")
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step['step']} ({step['expected_time']})")
    
    print(f"\n[FILE] 상세 결과: {report_path}")
    
    return results

if __name__ == "__main__":
    main()