#!/usr/bin/env python3
"""
컨퍼런스 콘텐츠 빠른 요약 시스템
- 기존 분석 결과와 이미지 미리보기 기반
- 시간 소모적인 STT/OCR 분석 없이 빠른 인사이트 제공
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class QuickConferenceSummary:
    """컨퍼런스 콘텐츠 빠른 요약기"""
    
    def __init__(self):
        self.conference_context = {
            'event_name': 'JGA25_0619 - CONNECTING THE JEWELLERY WORLD',
            'topic': 'The Rise of the Eco-friendly Luxury Consumer',
            'date': '19/6/2025 (Thursday)',
            'time': '2:30pm - 3:30pm',
            'venue': 'The Stage, Hall 1B, HKCEC',
            'speakers': [
                'Lianne Ng - Director of Sustainability, Chow Tai Fook Jewellery Group',
                'Henry Tse - CEO, Ancardi, Nyreille & JRNE', 
                'Pui In Catherine Siu - Vice-President Strategy'
            ]
        }
        
        print("컨퍼런스 콘텐츠 빠른 요약 시스템")
    
    def load_existing_analysis(self) -> Dict[str, Any]:
        """기존 분석 결과 로드"""
        # 기존 분석 파일 찾기
        analysis_files = list(project_root.glob("jewelry_conference_analysis_*.json"))
        
        if analysis_files:
            # 가장 최근 파일 로드
            latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            print(f"[OK] 기존 분석 데이터 로드: {latest_file.name}")
            return existing_data
        else:
            print("[INFO] 기존 분석 데이터 없음 - 새로운 분석 필요")
            return {}
    
    def analyze_content_inventory(self, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """콘텐츠 인벤토리 분석"""
        if not existing_data:
            return self._analyze_user_files_directly()
        
        # 기존 데이터에서 파일 정보 추출
        user_files_analysis = existing_data.get('analysis_results', {}).get('user_files_analysis', {})
        
        audio_files = user_files_analysis.get('audio_files', [])
        image_files = user_files_analysis.get('image_files', [])
        video_files = user_files_analysis.get('video_files', [])
        
        inventory = {
            'total_files': len(audio_files) + len(image_files) + len(video_files),
            'file_breakdown': {
                'audio_files': len(audio_files),
                'image_files': len(image_files), 
                'video_files': len(video_files)
            },
            'key_content': {
                'main_audio_recording': self._identify_main_audio(audio_files),
                'conference_images': self._count_conference_images(image_files),
                'video_recordings': self._identify_video_content(video_files)
            },
            'processing_estimates': {
                'total_audio_size_mb': sum(f.get('file_size_mb', 0) for f in audio_files),
                'total_image_count': len(image_files),
                'largest_video_gb': max([f.get('file_size_gb', 0) for f in video_files]) if video_files else 0
            }
        }
        
        return inventory
    
    def _analyze_user_files_directly(self) -> Dict[str, Any]:
        """user_files 폴더 직접 분석"""
        user_files_path = project_root / 'user_files'
        
        if not user_files_path.exists():
            return {'error': 'user_files 폴더를 찾을 수 없습니다.'}
        
        inventory = {
            'total_files': 0,
            'file_breakdown': {'audio_files': 0, 'image_files': 0, 'video_files': 0},
            'key_content': {},
            'processing_estimates': {}
        }
        
        # 오디오 파일
        audio_path = user_files_path / 'audio'
        if audio_path.exists():
            audio_files = list(audio_path.glob('*.m4a'))
            inventory['file_breakdown']['audio_files'] = len(audio_files)
            inventory['total_files'] += len(audio_files)
        
        # 이미지 파일
        image_path = user_files_path / 'images' 
        if image_path.exists():
            image_files = list(image_path.glob('*.JPG'))
            inventory['file_breakdown']['image_files'] = len(image_files)
            inventory['total_files'] += len(image_files)
        
        # 비디오 파일
        video_path = user_files_path / 'videos'
        if video_path.exists():
            video_files = list(video_path.glob('*.MOV'))
            inventory['file_breakdown']['video_files'] = len(video_files)
            inventory['total_files'] += len(video_files)
        
        return inventory
    
    def _identify_main_audio(self, audio_files: List[Dict]) -> Dict[str, Any]:
        """메인 오디오 파일 식별"""
        if not audio_files:
            return {'found': False}
        
        # 가장 큰 파일을 메인으로 간주
        main_audio = max(audio_files, key=lambda x: x.get('file_size_mb', 0))
        
        return {
            'found': True,
            'file_name': main_audio.get('file_name', ''),
            'size_mb': main_audio.get('file_size_mb', 0),
            'estimated_duration_minutes': main_audio.get('file_size_mb', 0) * 0.5,  # 대략적 추정
            'analysis_priority': 'high'
        }
    
    def _count_conference_images(self, image_files: List[Dict]) -> Dict[str, Any]:
        """컨퍼런스 이미지 개수 및 분석"""
        conference_images = [f for f in image_files if 'IMG_2' in f.get('file_name', '')]
        
        return {
            'total_conference_images': len(conference_images),
            'total_images': len(image_files),
            'conference_ratio': len(conference_images) / len(image_files) * 100 if image_files else 0,
            'expected_content': [
                '패널 토론 현장 사진',
                '발표자 프로필 및 소개 화면',
                '프레젠테이션 슬라이드',
                '컨퍼런스 장소 및 참석자'
            ]
        }
    
    def _identify_video_content(self, video_files: List[Dict]) -> Dict[str, Any]:
        """비디오 콘텐츠 식별"""
        if not video_files:
            return {'found': False}
        
        # 파일 크기 기준으로 분류
        large_videos = [v for v in video_files if v.get('file_size_gb', 0) > 1.0]
        small_videos = [v for v in video_files if v.get('file_size_gb', 0) <= 1.0]
        
        return {
            'found': True,
            'total_videos': len(video_files),
            'large_recordings': len(large_videos),
            'short_clips': len(small_videos),
            'largest_file': max(video_files, key=lambda x: x.get('file_size_gb', 0)) if video_files else None,
            'expected_content': '컨퍼런스 세션 전체 녹화 또는 하이라이트'
        }
    
    def generate_conference_insights(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        """컨퍼런스 인사이트 생성"""
        insights = {
            'conference_overview': {
                'event': self.conference_context['event_name'],
                'topic': self.conference_context['topic'],
                'speakers': len(self.conference_context['speakers']),
                'format': 'Panel Discussion with Q&A'
            },
            'content_analysis': {
                'multi_modal_content': inventory.get('total_files', 0) > 20,
                'comprehensive_coverage': self._assess_coverage(inventory),
                'analysis_complexity': self._assess_complexity(inventory)
            },
            'key_themes': [
                '지속가능한 주얼리 산업 혁신',
                '친환경 럭셔리 소비자 트렌드 분석',
                '업계 리더들의 지속가능성 전략',
                '미래 주얼리 시장 전망과 기회'
            ],
            'expected_insights': [
                'Chow Tai Fook의 지속가능성 전략 및 실행 방안',
                'Ancardi CEO Henry Tse의 친환경 럭셔리 시장 분석',
                '소비자 트렌드 변화에 대한 업계 전문가 의견',
                '주얼리 업계의 미래 비즈니스 모델 제안'
            ],
            'business_implications': [
                '주얼리 브랜드의 ESG 경영 전략 수립 필요성',
                '친환경 소재 및 제조 공정 혁신 요구',
                '밀레니얼/Z세대 타겟 마케팅 전략 개발',
                '지속가능성을 통한 브랜드 차별화 기회'
            ]
        }
        
        return insights
    
    def _assess_coverage(self, inventory: Dict[str, Any]) -> str:
        """콘텐츠 커버리지 평가"""
        has_audio = inventory.get('file_breakdown', {}).get('audio_files', 0) > 0
        has_images = inventory.get('file_breakdown', {}).get('image_files', 0) > 10
        has_video = inventory.get('file_breakdown', {}).get('video_files', 0) > 0
        
        if has_audio and has_images and has_video:
            return 'Complete - Audio, Visual, Video coverage available'
        elif has_audio and has_images:
            return 'Good - Audio and visual documentation available'
        elif has_images:
            return 'Moderate - Visual documentation only'
        else:
            return 'Limited - Insufficient content for comprehensive analysis'
    
    def _assess_complexity(self, inventory: Dict[str, Any]) -> str:
        """분석 복잡도 평가"""
        total_files = inventory.get('total_files', 0)
        audio_size = inventory.get('processing_estimates', {}).get('total_audio_size_mb', 0)
        
        if total_files > 25 and audio_size > 20:
            return 'High - Large scale multi-modal analysis required'
        elif total_files > 15:
            return 'Medium - Moderate scale analysis with multiple formats'
        else:
            return 'Low - Simple analysis with limited content'
    
    def create_action_plan(self, inventory: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """실행 계획 생성"""
        plan = {
            'immediate_actions': [],
            'priority_analysis': [],
            'expected_outcomes': [],
            'resource_requirements': {}
        }
        
        # 즉시 실행 가능한 액션
        if inventory.get('file_breakdown', {}).get('image_files', 0) > 0:
            plan['immediate_actions'].append('컨퍼런스 이미지 OCR 텍스트 추출')
        
        # 우선순위 분석
        main_audio = inventory.get('key_content', {}).get('main_audio_recording', {})
        if main_audio.get('found', False):
            plan['priority_analysis'].append(f"메인 오디오 파일 STT 분석 ({main_audio.get('size_mb', 0):.1f}MB)")
        
        video_content = inventory.get('key_content', {}).get('video_recordings', {})
        if video_content.get('found', False):
            plan['priority_analysis'].append('대용량 비디오 파일 오디오 추출 및 분석')
        
        # 예상 결과
        plan['expected_outcomes'] = [
            '패널 토론 핵심 메시지 및 주요 발언 정리',
            '발표자별 주요 의견 및 인사이트 추출',
            '지속가능성 관련 구체적 전략 및 방안 도출',
            '주얼리 업계 트렌드 및 미래 전망 요약'
        ]
        
        # 리소스 요구사항
        total_processing_time = 0
        if main_audio.get('size_mb', 0) > 0:
            total_processing_time += main_audio.get('size_mb', 0) * 2  # 분 단위 추정
        
        plan['resource_requirements'] = {
            'estimated_processing_time_minutes': total_processing_time,
            'required_tools': ['Whisper STT', 'EasyOCR', 'Video Processor'],
            'system_requirements': 'CPU 모드, 메모리 4GB+ 권장'
        }
        
        return plan

def main():
    """메인 실행 함수"""
    print("컨퍼런스 콘텐츠 빠른 요약 분석")
    print("=" * 50)
    
    # 빠른 요약 시스템 초기화
    summary_system = QuickConferenceSummary()
    
    # 1. 기존 분석 결과 로드
    print("\n--- 기존 분석 데이터 확인 ---")
    existing_data = summary_system.load_existing_analysis()
    
    # 2. 콘텐츠 인벤토리 분석
    print("\n--- 콘텐츠 인벤토리 분석 ---")
    inventory = summary_system.analyze_content_inventory(existing_data)
    
    # 3. 컨퍼런스 인사이트 생성
    print("\n--- 컨퍼런스 인사이트 생성 ---")
    insights = summary_system.generate_conference_insights(inventory)
    
    # 4. 실행 계획 생성
    print("\n--- 실행 계획 생성 ---")
    action_plan = summary_system.create_action_plan(inventory, insights)
    
    # 종합 결과 출력
    print(f"\n{'='*60}")
    print("컨퍼런스 콘텐츠 빠른 요약 완료")
    print(f"{'='*60}")
    
    # 컨퍼런스 개요
    print(f"\n[CONFERENCE] 컨퍼런스 정보:")
    overview = insights['conference_overview']
    print(f"  제목: {overview['event']}")
    print(f"  주제: {overview['topic']}")
    print(f"  발표자: {overview['speakers']}명")
    print(f"  형식: {overview['format']}")
    
    # 콘텐츠 현황
    print(f"\n[CONTENT] 콘텐츠 현황:")
    breakdown = inventory.get('file_breakdown', {})
    print(f"  총 파일: {inventory.get('total_files', 0)}개")
    print(f"  오디오: {breakdown.get('audio_files', 0)}개")
    print(f"  이미지: {breakdown.get('image_files', 0)}개")
    print(f"  비디오: {breakdown.get('video_files', 0)}개")
    
    # 콘텐츠 분석
    print(f"\n[ANALYSIS] 콘텐츠 분석:")
    content_analysis = insights['content_analysis']
    print(f"  커버리지: {content_analysis['comprehensive_coverage']}")
    print(f"  복잡도: {content_analysis['analysis_complexity']}")
    
    # 핵심 주제
    print(f"\n[THEMES] 예상 핵심 주제:")
    for i, theme in enumerate(insights['key_themes'], 1):
        print(f"  {i}. {theme}")
    
    # 실행 계획
    print(f"\n[PLAN] 권장 실행 순서:")
    for i, action in enumerate(action_plan['priority_analysis'], 1):
        print(f"  {i}. {action}")
    
    # 예상 결과
    print(f"\n[RESULTS] 예상 분석 결과:")
    for i, outcome in enumerate(action_plan['expected_outcomes'], 1):
        print(f"  {i}. {outcome}")
    
    # 비즈니스 시사점
    print(f"\n[BUSINESS] 비즈니스 시사점:")
    for i, implication in enumerate(insights['business_implications'], 1):
        print(f"  {i}. {implication}")
    
    # 리소스 요구사항
    resources = action_plan['resource_requirements']
    print(f"\n[RESOURCES] 분석 리소스 요구사항:")
    print(f"  예상 처리 시간: {resources.get('estimated_processing_time_minutes', 0):.0f}분")
    print(f"  필요 도구: {', '.join(resources.get('required_tools', []))}")
    print(f"  시스템 요구사항: {resources.get('system_requirements', 'N/A')}")
    
    return {
        'inventory': inventory,
        'insights': insights,
        'action_plan': action_plan
    }

if __name__ == "__main__":
    main()