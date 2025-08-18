#!/usr/bin/env python3
"""
동영상 관련 기능 자동 점검 시스템
1. 고용량 동영상 파일 지원 상태
2. 동영상 URL 분석 기능
3. 사전정보 맥락 반영 시스템
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import importlib.util

class VideoFeaturesChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.check_results = {
            'timestamp': datetime.now().isoformat(),
            'video_file_support': {},
            'video_url_analysis': {},
            'context_integration': {},
            'recommendations': []
        }
    
    def check_video_file_support(self):
        """고용량 동영상 파일 지원 상태 점검"""
        print("=== 1. 고용량 동영상 파일 지원 점검 ===")
        
        # 관련 파일들 확인
        video_related_files = [
            'core/large_video_processor.py',
            'core/real_analysis_engine.py',
            'jewelry_stt_ui_v23_real.py'
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in video_related_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                print(f"✅ {file_path}: 존재")
            else:
                missing_files.append(file_path)
                print(f"❌ {file_path}: 누락")
        
        # large_video_processor.py 상세 분석
        if 'core/large_video_processor.py' in existing_files:
            self.analyze_large_video_processor()
        else:
            print("❌ 대용량 동영상 처리기가 없습니다.")
            self.check_results['video_file_support']['has_processor'] = False
        
        # 지원 포맷 확인
        self.check_supported_video_formats()
        
        # 파일 크기 제한 확인
        self.check_video_file_limits()
    
    def analyze_large_video_processor(self):
        """대용량 동영상 처리기 분석"""
        print("\n--- 대용량 동영상 처리기 분석 ---")
        
        try:
            processor_path = self.project_root / 'core/large_video_processor.py'
            with open(processor_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 주요 기능 확인
            features = {
                'chunk_processing': 'chunk' in content.lower(),
                'memory_optimization': 'memory' in content.lower() and 'optim' in content.lower(),
                'progress_tracking': 'progress' in content.lower(),
                'ffmpeg_support': 'ffmpeg' in content.lower(),
                'opencv_support': 'cv2' in content or 'opencv' in content.lower(),
                'streaming_support': 'stream' in content.lower(),
                'large_file_handling': 'large' in content.lower() and 'file' in content.lower()
            }
            
            for feature, supported in features.items():
                status = "✅" if supported else "❌"
                print(f"{status} {feature}: {'지원' if supported else '미지원'}")
            
            self.check_results['video_file_support']['processor_features'] = features
            
            # 파일 크기 확인
            file_size = processor_path.stat().st_size / 1024  # KB
            print(f"📄 파일 크기: {file_size:.1f}KB")
            
            if file_size > 10:  # 10KB 이상이면 실제 구현된 것으로 판단
                self.check_results['video_file_support']['has_implementation'] = True
            else:
                self.check_results['video_file_support']['has_implementation'] = False
                
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            self.check_results['video_file_support']['analysis_error'] = str(e)
    
    def check_supported_video_formats(self):
        """지원되는 동영상 포맷 확인"""
        print("\n--- 지원 동영상 포맷 확인 ---")
        
        try:
            # Streamlit UI에서 허용되는 파일 형식 확인
            ui_path = self.project_root / 'jewelry_stt_ui_v23_real.py'
            if ui_path.exists():
                with open(ui_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 파일 업로더에서 지원하는 형식 찾기
                video_formats = []
                if 'mp4' in content.lower():
                    video_formats.append('mp4')
                if 'avi' in content.lower():
                    video_formats.append('avi')
                if 'mov' in content.lower():
                    video_formats.append('mov')
                if 'mkv' in content.lower():
                    video_formats.append('mkv')
                if 'webm' in content.lower():
                    video_formats.append('webm')
                
                print(f"📹 지원 포맷: {video_formats}")
                self.check_results['video_file_support']['supported_formats'] = video_formats
                
                if not video_formats:
                    print("⚠️ 명시적인 동영상 포맷 지원이 확인되지 않습니다.")
            else:
                print("❌ UI 파일을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 포맷 확인 실패: {e}")
    
    def check_video_file_limits(self):
        """동영상 파일 크기 제한 확인"""
        print("\n--- 파일 크기 제한 확인 ---")
        
        try:
            # Streamlit 설정에서 최대 업로드 크기 확인
            config_path = self.project_root / '.streamlit/config.toml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_content = f.read()
                
                if 'maxUploadSize' in config_content:
                    # 설정에서 값 추출
                    for line in config_content.split('\n'):
                        if 'maxUploadSize' in line and '=' in line:
                            size_mb = line.split('=')[1].strip()
                            print(f"📏 최대 업로드 크기: {size_mb}MB")
                            self.check_results['video_file_support']['max_upload_size'] = size_mb
                            break
                else:
                    print("⚠️ 최대 업로드 크기 설정이 없습니다.")
            else:
                print("❌ Streamlit 설정 파일이 없습니다.")
                
        except Exception as e:
            print(f"❌ 크기 제한 확인 실패: {e}")
    
    def check_video_url_analysis(self):
        """동영상 URL 분석 기능 점검"""
        print("\n=== 2. 동영상 URL 분석 기능 점검 ===")
        
        # YouTube 처리 관련 파일 확인
        youtube_files = [
            'core/youtube_processor.py',
            'core/youtube_realtime_processor.py'
        ]
        
        for file_path in youtube_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}: 존재")
                self.analyze_youtube_processor(full_path)
            else:
                print(f"❌ {file_path}: 누락")
        
        # URL 처리 기능 확인
        self.check_url_processing_capabilities()
    
    def analyze_youtube_processor(self, file_path):
        """YouTube 처리기 분석"""
        print(f"\n--- {file_path.name} 분석 ---")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 주요 기능 확인
            features = {
                'url_validation': 'url' in content.lower() and 'valid' in content.lower(),
                'video_download': 'download' in content.lower(),
                'audio_extraction': 'audio' in content.lower() and 'extract' in content.lower(),
                'metadata_extraction': 'metadata' in content.lower(),
                'yt_dlp_support': 'yt-dlp' or 'ytdlp' in content.lower(),
                'real_time_processing': 'realtime' in content.lower() or 'real_time' in content.lower(),
                'error_handling': 'try:' in content and 'except' in content
            }
            
            for feature, supported in features.items():
                status = "✅" if supported else "❌"
                print(f"{status} {feature}: {'지원' if supported else '미지원'}")
            
            self.check_results['video_url_analysis'][file_path.name] = features
            
            # 파일 크기로 구현 수준 판단
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"📄 파일 크기: {file_size:.1f}KB")
            
            if file_size > 5:  # 5KB 이상
                implementation_level = "상세 구현"
            elif file_size > 1:
                implementation_level = "기본 구현"
            else:
                implementation_level = "스켈레톤만"
            
            print(f"🔧 구현 수준: {implementation_level}")
            self.check_results['video_url_analysis'][f"{file_path.name}_implementation"] = implementation_level
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
    
    def check_url_processing_capabilities(self):
        """URL 처리 능력 확인"""
        print("\n--- URL 처리 능력 확인 ---")
        
        # 지원 가능한 플랫폼 확인
        supported_platforms = []
        
        try:
            # 각 처리기에서 지원하는 플랫폼 확인
            processors = ['youtube_processor.py', 'youtube_realtime_processor.py']
            
            for processor in processors:
                file_path = self.project_root / 'core' / processor
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    platforms = []
                    if 'youtube' in content:
                        platforms.append('YouTube')
                    if 'vimeo' in content:
                        platforms.append('Vimeo')
                    if 'dailymotion' in content:
                        platforms.append('Dailymotion')
                    if 'twitch' in content:
                        platforms.append('Twitch')
                    
                    if platforms:
                        supported_platforms.extend(platforms)
                        print(f"📹 {processor}: {', '.join(platforms)}")
            
            unique_platforms = list(set(supported_platforms))
            self.check_results['video_url_analysis']['supported_platforms'] = unique_platforms
            print(f"🌐 전체 지원 플랫폼: {unique_platforms}")
            
            if not unique_platforms:
                print("⚠️ 명시적으로 지원되는 플랫폼이 확인되지 않습니다.")
                
        except Exception as e:
            print(f"❌ 플랫폼 확인 실패: {e}")
    
    def check_context_integration(self):
        """사전정보 맥락 반영 시스템 점검"""
        print("\n=== 3. 사전정보 맥락 반영 시스템 점검 ===")
        
        # 관련 파일들 확인
        context_files = [
            'core/comprehensive_message_extractor.py',
            'core/real_analysis_engine.py',
            'jewelry_stt_ui_v23_real.py'
        ]
        
        for file_path in context_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}: 존재")
                self.analyze_context_integration(full_path)
            else:
                print(f"❌ {file_path}: 누락")
        
        # UI에서 사전정보 입력 기능 확인
        self.check_ui_context_input()
    
    def analyze_context_integration(self, file_path):
        """맥락 통합 기능 분석"""
        print(f"\n--- {file_path.name} 맥락 처리 분석 ---")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 맥락 관련 기능 확인
            features = {
                'context_input': 'context' in content.lower() or '맥락' in content,
                'prior_info_processing': 'prior' in content.lower() or '사전' in content,
                'user_input_integration': 'user_input' in content.lower() or 'basic_info' in content.lower(),
                'session_state_context': 'session_state' in content.lower(),
                'context_preservation': 'preserve' in content.lower() or '보존' in content,
                'contextual_analysis': 'contextual' in content.lower() or '맥락적' in content,
                'metadata_integration': 'metadata' in content.lower()
            }
            
            for feature, supported in features.items():
                status = "✅" if supported else "❌"
                print(f"{status} {feature}: {'지원' if supported else '미지원'}")
            
            self.check_results['context_integration'][file_path.name] = features
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
    
    def check_ui_context_input(self):
        """UI 사전정보 입력 기능 확인"""
        print("\n--- UI 사전정보 입력 기능 확인 ---")
        
        try:
            ui_path = self.project_root / 'jewelry_stt_ui_v23_real.py'
            if ui_path.exists():
                with open(ui_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 사전정보 입력 관련 기능 확인
                ui_features = {
                    'basic_info_step': 'basic_info' in content.lower() or '기본정보' in content,
                    'project_context': 'project' in content.lower() and 'info' in content.lower(),
                    'user_context_input': 'text_input' in content and ('context' in content.lower() or '맥락' in content),
                    'step_workflow': 'step1' in content.lower() and 'step2' in content.lower(),
                    'context_persistence': 'session_state' in content and 'context' in content.lower()
                }
                
                for feature, supported in ui_features.items():
                    status = "✅" if supported else "❌"
                    print(f"{status} {feature}: {'지원' if supported else '미지원'}")
                
                self.check_results['context_integration']['ui_features'] = ui_features
                
        except Exception as e:
            print(f"❌ UI 분석 실패: {e}")
    
    def generate_recommendations(self):
        """개선 권장사항 생성"""
        print("\n=== 개선 권장사항 생성 ===")
        
        recommendations = []
        
        # 1. 동영상 파일 지원 개선
        video_support = self.check_results.get('video_file_support', {})
        
        if not video_support.get('has_implementation', False):
            recommendations.append({
                'category': 'VIDEO_FILE_SUPPORT',
                'priority': 'HIGH',
                'issue': '고용량 동영상 파일 처리 기능 부족',
                'solution': '대용량 동영상 처리기 강화',
                'actions': [
                    '청크 단위 동영상 처리 구현',
                    '메모리 효율적 프레임 추출',
                    '진행률 추적 시스템 추가',
                    'FFmpeg 통합 강화'
                ]
            })
        
        if not video_support.get('supported_formats'):
            recommendations.append({
                'category': 'VIDEO_FORMATS',
                'priority': 'MEDIUM',
                'issue': '동영상 포맷 지원 명시 부족',
                'solution': '다양한 동영상 포맷 지원 추가',
                'actions': [
                    'MP4, AVI, MOV, MKV, WebM 형식 지원',
                    'UI에서 지원 포맷 명시',
                    '포맷별 최적화 처리'
                ]
            })
        
        # 2. 동영상 URL 분석 개선
        url_analysis = self.check_results.get('video_url_analysis', {})
        
        if not any('youtube_processor.py' in key for key in url_analysis.keys()):
            recommendations.append({
                'category': 'VIDEO_URL_ANALYSIS',
                'priority': 'HIGH',
                'issue': 'YouTube URL 분석 기능 미비',
                'solution': 'YouTube URL 처리 시스템 구현',
                'actions': [
                    'yt-dlp 라이브러리 통합',
                    'URL 유효성 검증',
                    '자동 오디오 추출',
                    '메타데이터 수집'
                ]
            })
        
        # 3. 맥락 통합 개선
        context_integration = self.check_results.get('context_integration', {})
        
        context_score = 0
        for file_data in context_integration.values():
            if isinstance(file_data, dict):
                context_score += sum(file_data.values())
        
        if context_score < 10:  # 충분한 맥락 기능이 없는 경우
            recommendations.append({
                'category': 'CONTEXT_INTEGRATION',
                'priority': 'MEDIUM',
                'issue': '사전정보 맥락 반영 시스템 미흡',
                'solution': '맥락 인식 분석 시스템 강화',
                'actions': [
                    '사전정보 구조화 저장',
                    '분석 시 맥락 자동 적용',
                    '맥락 기반 결과 해석',
                    '사용자별 맥락 프로파일 관리'
                ]
            })
        
        self.check_results['recommendations'] = recommendations
        
        print(f"생성된 권장사항: {len(recommendations)}개")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['issue']} ({rec['priority']})")
    
    def save_analysis_report(self):
        """분석 보고서 저장"""
        report_path = self.project_root / 'video_features_analysis_report.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.check_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 분석 보고서 저장: {report_path}")
        return report_path
    
    def run_full_check(self):
        """전체 점검 실행"""
        print("동영상 관련 기능 자동 점검 시작")
        print("=" * 60)
        
        try:
            self.check_video_file_support()
            self.check_video_url_analysis()
            self.check_context_integration()
            self.generate_recommendations()
            
            report_path = self.save_analysis_report()
            
            print("\n" + "=" * 60)
            print("점검 완료 요약")
            print("=" * 60)
            
            # 요약 출력
            print(f"📹 동영상 파일 지원: {'✅' if self.check_results.get('video_file_support', {}).get('has_implementation') else '❌'}")
            print(f"🔗 URL 분석 기능: {'✅' if self.check_results.get('video_url_analysis') else '❌'}")
            print(f"📝 맥락 반영 시스템: {'✅' if sum(len(v) for v in self.check_results.get('context_integration', {}).values() if isinstance(v, dict)) > 0 else '❌'}")
            print(f"💡 개선 권장사항: {len(self.check_results.get('recommendations', []))}개")
            
            return self.check_results
            
        except Exception as e:
            print(f"❌ 점검 중 오류 발생: {e}")
            return None

def main():
    checker = VideoFeaturesChecker()
    results = checker.run_full_check()
    
    if results:
        print(f"\n✅ 점검 완료! 상세 보고서: video_features_analysis_report.json")
        return results
    else:
        print(f"\n❌ 점검 실패")
        return None

if __name__ == "__main__":
    main()