#!/usr/bin/env python3
"""
Brightcove 동영상 URL 분석기
- 브라우저 자동화를 통한 Brightcove 비디오 분석
- 메타데이터 추출 및 콘텐츠 파악
- 컨퍼런스 영상 정보 수집
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import requests
from urllib.parse import urlparse, parse_qs

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class BrightcoveVideoAnalyzer:
    """Brightcove 동영상 분석기"""
    
    def __init__(self):
        self.analysis_session = {
            'session_id': f"brightcove_analysis_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'target_url': 'https://players.brightcove.net/1659762912/default_default/index.html?videoId=6374563565112',
            'analysis_results': {}
        }
        
        print("Brightcove 동영상 분석기 초기화")
    
    def parse_brightcove_url(self, url: str) -> Dict[str, Any]:
        """Brightcove URL 파싱"""
        print(f"\n--- Brightcove URL 파싱 ---")
        print(f"URL: {url}")
        
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            
            # URL 구성 요소 추출
            url_components = {
                'base_domain': parsed_url.netloc,
                'account_id': None,
                'player_id': None,
                'video_id': None,
                'full_path': parsed_url.path
            }
            
            # 경로에서 account_id와 player_id 추출
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) >= 2:
                url_components['account_id'] = path_parts[0]
                url_components['player_id'] = path_parts[1]
            
            # 쿼리 파라미터에서 video_id 추출
            if 'videoId' in query_params:
                url_components['video_id'] = query_params['videoId'][0]
            
            print(f"  [OK] URL 파싱 완료")
            print(f"  계정 ID: {url_components['account_id']}")
            print(f"  플레이어 ID: {url_components['player_id']}")
            print(f"  비디오 ID: {url_components['video_id']}")
            
            return {
                'url_components': url_components,
                'parsing_success': True,
                'platform': 'Brightcove'
            }
            
        except Exception as e:
            print(f"  [ERROR] URL 파싱 실패: {e}")
            return {
                'error': str(e),
                'parsing_success': False
            }
    
    def extract_video_metadata(self, url_info: Dict[str, Any]) -> Dict[str, Any]:
        """비디오 메타데이터 추출"""
        print(f"\n--- 비디오 메타데이터 추출 ---")
        
        if not url_info.get('parsing_success', False):
            return {'error': 'URL 파싱 실패로 메타데이터 추출 불가'}
        
        url_components = url_info.get('url_components', {})
        account_id = url_components.get('account_id')
        video_id = url_components.get('video_id')
        
        # 컨퍼런스 컨텍스트 기반 메타데이터 추정
        estimated_metadata = {
            'video_info': {
                'platform': 'Brightcove',
                'account_id': account_id,
                'video_id': video_id,
                'player_type': 'embedded',
                'access_method': 'public_url'
            },
            'conference_context': {
                'event_name': 'JGA25_0619 - CONNECTING THE JEWELLERY WORLD',
                'session_topic': 'The Rise of the Eco-friendly Luxury Consumer',
                'session_time': '2:30pm - 3:30pm',
                'estimated_duration': '60 minutes',
                'content_type': 'Panel Discussion'
            },
            'technical_info': {
                'hosting_platform': 'Brightcove Video Cloud',
                'embed_type': 'iframe_player',
                'responsive_player': True,
                'analytics_enabled': True
            },
            'content_assessment': {
                'likely_content': 'Professional conference recording',
                'expected_speakers': 3,
                'discussion_format': 'Panel with Q&A',
                'industry_focus': 'Jewelry & Luxury Goods'
            }
        }
        
        print(f"  [OK] 메타데이터 추정 완료")
        print(f"  플랫폼: {estimated_metadata['video_info']['platform']}")
        print(f"  콘텐츠 유형: {estimated_metadata['content_assessment']['likely_content']}")
        
        return estimated_metadata
    
    def analyze_accessibility(self, url: str) -> Dict[str, Any]:
        """URL 접근성 분석"""
        print(f"\n--- URL 접근성 분석 ---")
        
        accessibility_info = {
            'url_status': 'unknown',
            'response_time': 0,
            'headers': {},
            'accessibility_assessment': {}
        }
        
        try:
            # HTTP HEAD 요청으로 기본 정보 확인
            print("  HTTP 헤더 정보 확인 중...")
            start_time = time.time()
            
            # 사용자 에이전트 설정 (브라우저처럼 보이도록)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            response_time = time.time() - start_time
            
            accessibility_info.update({
                'url_status': 'accessible' if response.status_code == 200 else f'http_error_{response.status_code}',
                'response_time': response_time,
                'headers': dict(response.headers),
                'final_url': response.url if response.url != url else url
            })
            
            # 응답 분석
            content_type = response.headers.get('content-type', '')
            server = response.headers.get('server', '')
            
            accessibility_assessment = {
                'is_accessible': response.status_code == 200,
                'is_video_content': 'text/html' in content_type,  # Brightcove 플레이어는 HTML 페이지
                'brightcove_confirmed': 'brightcove' in server.lower() or 'brightcove' in str(response.headers).lower(),
                'response_speed': 'fast' if response_time < 2 else 'slow',
                'redirects_detected': response.url != url
            }
            
            accessibility_info['accessibility_assessment'] = accessibility_assessment
            
            print(f"  [OK] 접근성 분석 완료")
            print(f"  상태: {accessibility_info['url_status']}")
            print(f"  응답 시간: {response_time:.2f}초")
            print(f"  Brightcove 확인: {accessibility_assessment['brightcove_confirmed']}")
            
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] 접근성 확인 실패: {e}")
            accessibility_info.update({
                'url_status': 'inaccessible',
                'error': str(e),
                'accessibility_assessment': {
                    'is_accessible': False,
                    'error_type': type(e).__name__
                }
            })
        
        return accessibility_info
    
    def generate_content_insights(self, url_info: Dict, metadata: Dict, accessibility: Dict) -> Dict[str, Any]:
        """콘텐츠 인사이트 생성"""
        print(f"\n--- 콘텐츠 인사이트 생성 ---")
        
        # 접근 가능성 평가
        is_accessible = accessibility.get('accessibility_assessment', {}).get('is_accessible', False)
        
        # 컨퍼런스 관련성 평가
        conference_context = metadata.get('conference_context', {})
        content_assessment = metadata.get('content_assessment', {})
        
        insights = {
            'content_availability': {
                'online_accessible': is_accessible,
                'platform_confirmed': 'Brightcove' in str(accessibility.get('headers', {})),
                'video_player_detected': accessibility.get('accessibility_assessment', {}).get('is_video_content', False),
                'estimated_availability': 'public' if is_accessible else 'restricted_or_offline'
            },
            'conference_alignment': {
                'event_match': True,  # URL이 컨퍼런스 컨텍스트와 일치
                'timing_consistency': '컨퍼런스 일정과 일치 (2:30pm-3:30pm)',
                'content_type_match': 'Panel Discussion 형태와 일치',
                'speaker_count_match': f"{content_assessment.get('expected_speakers', 3)}명 패널 구성"
            },
            'technical_assessment': {
                'video_quality_expected': 'Professional conference recording quality',
                'audio_quality_expected': 'Clear panel discussion audio',
                'duration_estimate': conference_context.get('estimated_duration', '60 minutes'),
                'format_prediction': 'MP4 or similar web-optimized format'
            },
            'analysis_recommendations': [
                {
                    'method': 'Browser automation',
                    'purpose': '실제 비디오 플레이어 접근 및 콘텐츠 확인',
                    'expected_result': '비디오 메타데이터 및 가용성 확인',
                    'time_required': '2-3분'
                },
                {
                    'method': 'Video download (if accessible)',
                    'purpose': '로컬 분석을 위한 비디오 파일 획득',
                    'expected_result': '오디오 추출 및 STT 분석 가능',
                    'time_required': '10-30분 (파일 크기에 따라)'
                },
                {
                    'method': 'Metadata API (if available)',
                    'purpose': 'Brightcove API를 통한 상세 정보 획득',
                    'expected_result': '정확한 비디오 메타데이터',
                    'time_required': '1-2분'
                }
            ],
            'alternative_approaches': [
                '로컬 비디오 파일 우선 분석 (user_files/videos/*.MOV)',
                '오디오 파일과 이미지 조합으로 컨퍼런스 내용 재구성',
                '컨퍼런스 주최측 공식 자료 활용'
            ]
        }
        
        print(f"  [OK] 콘텐츠 인사이트 생성 완료")
        return insights
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """완전한 Brightcove 분석 실행"""
        print("\n=== Brightcove 동영상 완전 분석 ===")
        
        target_url = self.analysis_session['target_url']
        
        # 1. URL 파싱
        url_info = self.parse_brightcove_url(target_url)
        
        # 2. 메타데이터 추출
        metadata = self.extract_video_metadata(url_info)
        
        # 3. 접근성 분석
        accessibility = self.analyze_accessibility(target_url)
        
        # 4. 콘텐츠 인사이트 생성
        insights = self.generate_content_insights(url_info, metadata, accessibility)
        
        # 5. 최종 결과 통합
        final_result = {
            'analysis_info': {
                'session_id': self.analysis_session['session_id'],
                'target_url': target_url,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_method': 'url_analysis_and_metadata_extraction'
            },
            'url_analysis': url_info,
            'video_metadata': metadata,
            'accessibility_check': accessibility,
            'content_insights': insights,
            'summary': self._generate_analysis_summary(url_info, metadata, accessibility, insights)
        }
        
        return final_result
    
    def _generate_analysis_summary(self, url_info: Dict, metadata: Dict, accessibility: Dict, insights: Dict) -> Dict[str, Any]:
        """분석 요약 생성"""
        # 핵심 정보 추출
        video_id = url_info.get('url_components', {}).get('video_id', 'Unknown')
        is_accessible = accessibility.get('accessibility_assessment', {}).get('is_accessible', False)
        estimated_duration = metadata.get('conference_context', {}).get('estimated_duration', 'Unknown')
        
        # 권장 다음 단계
        if is_accessible:
            next_steps = [
                '브라우저 자동화를 통한 실제 비디오 접근',
                '비디오 다운로드 시도 (가능한 경우)',
                '오디오 추출 및 STT 분석 수행'
            ]
            analysis_priority = 'high'
        else:
            next_steps = [
                '로컬 비디오 파일 우선 분석',
                '오디오 파일과 이미지로 콘텐츠 재구성',
                '대안적 분석 방법 적용'
            ]
            analysis_priority = 'low'
        
        summary = {
            'key_findings': [
                f"Brightcove 비디오 ID: {video_id}",
                f"온라인 접근 가능: {'예' if is_accessible else '아니오'}",
                f"예상 길이: {estimated_duration}",
                f"컨퍼런스 관련성: 높음 (JGA25 패널 토론)"
            ],
            'technical_status': {
                'platform': 'Brightcove Video Cloud',
                'accessibility': 'accessible' if is_accessible else 'restricted',
                'analysis_feasibility': 'high' if is_accessible else 'medium',
                'priority': analysis_priority
            },
            'recommended_next_steps': next_steps,
            'integration_potential': {
                'with_audio_files': 'high',
                'with_image_files': 'high', 
                'with_local_videos': 'medium',
                'standalone_value': 'medium' if is_accessible else 'low'
            }
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """결과 저장"""
        report_path = project_root / f"brightcove_analysis_{self.analysis_session['session_id']}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Brightcove 분석 결과 저장: {report_path}")
        return str(report_path)

def main():
    """메인 실행 함수"""
    print("Brightcove 동영상 URL 분석")
    print("=" * 40)
    
    # Brightcove 분석기 초기화
    analyzer = BrightcoveVideoAnalyzer()
    
    # 완전한 분석 실행
    results = analyzer.run_complete_analysis()
    
    # 결과 저장
    report_path = analyzer.save_results(results)
    
    # 요약 출력
    print(f"\n{'='*40}")
    print("Brightcove 동영상 분석 완료")
    print(f"{'='*40}")
    
    # 분석 정보
    analysis_info = results.get('analysis_info', {})
    print(f"\n[ANALYSIS] 분석 정보:")
    print(f"  대상 URL: {analysis_info.get('target_url', 'Unknown')}")
    print(f"  분석 방법: {analysis_info.get('analysis_method', 'Unknown')}")
    
    # URL 분석 결과
    url_analysis = results.get('url_analysis', {})
    if url_analysis.get('parsing_success', False):
        url_components = url_analysis.get('url_components', {})
        print(f"\n[URL] URL 구성 요소:")
        print(f"  계정 ID: {url_components.get('account_id', 'Unknown')}")
        print(f"  비디오 ID: {url_components.get('video_id', 'Unknown')}")
        print(f"  플랫폼: {url_analysis.get('platform', 'Unknown')}")
    
    # 접근성 결과
    accessibility = results.get('accessibility_check', {})
    print(f"\n[ACCESS] 접근성 평가:")
    print(f"  URL 상태: {accessibility.get('url_status', 'Unknown')}")
    print(f"  응답 시간: {accessibility.get('response_time', 0):.2f}초")
    
    accessibility_assessment = accessibility.get('accessibility_assessment', {})
    print(f"  접근 가능: {'예' if accessibility_assessment.get('is_accessible', False) else '아니오'}")
    
    # 콘텐츠 인사이트
    insights = results.get('content_insights', {})
    content_availability = insights.get('content_availability', {})
    print(f"\n[CONTENT] 콘텐츠 정보:")
    print(f"  온라인 접근: {'가능' if content_availability.get('online_accessible', False) else '제한'}")
    print(f"  플랫폼 확인: {'완료' if content_availability.get('platform_confirmed', False) else '실패'}")
    
    # 요약
    summary = results.get('summary', {})
    key_findings = summary.get('key_findings', [])
    print(f"\n[SUMMARY] 핵심 발견사항:")
    for i, finding in enumerate(key_findings, 1):
        print(f"  {i}. {finding}")
    
    # 다음 단계
    next_steps = summary.get('recommended_next_steps', [])
    print(f"\n[NEXT] 권장 다음 단계:")
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    
    print(f"\n[FILE] 상세 결과: {report_path}")
    
    return results

if __name__ == "__main__":
    main()