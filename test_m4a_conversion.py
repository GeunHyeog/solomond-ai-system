#!/usr/bin/env python3
"""
🎵 M4A 오디오 파일 변환 테스트 시스템
솔로몬드 AI 시스템 - M4A 파일 처리 문제 해결 검증

목적: 개선된 audio_converter의 M4A 파일 처리 성능 테스트
기능: 다양한 M4A 파일 형식에 대한 종합 테스트 및 성능 측정
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import get_logger
from core.audio_converter import AudioConverter, convert_m4a_to_wav, get_audio_info, batch_convert_audio_files

class M4AConversionTester:
    """M4A 파일 변환 테스트 클래스"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.converter = AudioConverter()
        self.test_results = []
        
        # 테스트 디렉토리 설정
        self.test_dir = Path(__file__).parent / "test_audio_files"
        self.test_dir.mkdir(exist_ok=True)
        
        # 결과 저장 디렉토리
        self.results_dir = Path(__file__).parent / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def create_test_m4a_files(self):
        """테스트용 M4A 파일 생성 (FFmpeg 사용)"""
        
        self.logger.info("🎵 테스트용 M4A 파일 생성 중...")
        
        if not self.converter.ffmpeg_available:
            self.logger.warning("⚠️ FFmpeg가 없어서 테스트 파일 생성 불가")
            return []
        
        test_files = []
        
        # 1. 짧은 테스트 톤 생성 (1초)
        short_m4a = self.test_dir / "test_short.m4a"
        if not short_m4a.exists():
            try:
                cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
                    '-acodec', 'aac', '-b:a', '128k', '-y', str(short_m4a)
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    test_files.append(short_m4a)
                    self.logger.info(f"✅ 짧은 테스트 파일 생성: {short_m4a.name}")
            except Exception as e:
                self.logger.warning(f"⚠️ 짧은 테스트 파일 생성 실패: {e}")
        else:
            test_files.append(short_m4a)
        
        # 2. 중간 테스트 톤 생성 (5초)
        medium_m4a = self.test_dir / "test_medium.m4a"
        if not medium_m4a.exists():
            try:
                cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=880:duration=5',
                    '-acodec', 'aac', '-b:a', '128k', '-y', str(medium_m4a)
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    test_files.append(medium_m4a)
                    self.logger.info(f"✅ 중간 테스트 파일 생성: {medium_m4a.name}")
            except Exception as e:
                self.logger.warning(f"⚠️ 중간 테스트 파일 생성 실패: {e}")
        else:
            test_files.append(medium_m4a)
        
        # 3. 스테레오 테스트 파일 생성 (3초)
        stereo_m4a = self.test_dir / "test_stereo.m4a"
        if not stereo_m4a.exists():
            try:
                cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
                    '-af', 'pan=stereo|c0=0.5*c0|c1=0.5*c0',
                    '-acodec', 'aac', '-b:a', '128k', '-y', str(stereo_m4a)
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    test_files.append(stereo_m4a)
                    self.logger.info(f"✅ 스테레오 테스트 파일 생성: {stereo_m4a.name}")
            except Exception as e:
                self.logger.warning(f"⚠️ 스테레오 테스트 파일 생성 실패: {e}")
        else:
            test_files.append(stereo_m4a)
        
        return test_files
    
    def find_existing_m4a_files(self) -> List[Path]:
        """기존 M4A 파일 찾기"""
        
        search_paths = [
            Path.home() / "Downloads",
            Path.home() / "Music",
            Path.home() / "Documents",
            Path(".")  # 현재 디렉토리
        ]
        
        found_files = []
        for search_path in search_paths:
            if search_path.exists():
                m4a_files = list(search_path.glob("*.m4a"))[:3]  # 최대 3개
                found_files.extend(m4a_files)
                if m4a_files:
                    self.logger.info(f"📁 {search_path}에서 {len(m4a_files)}개 M4A 파일 발견")
        
        return found_files[:5]  # 최대 5개만 테스트
    
    def test_single_conversion(self, m4a_path: Path) -> Dict[str, Any]:
        """단일 M4A 파일 변환 테스트"""
        
        test_result = {
            'file_path': str(m4a_path),
            'file_name': m4a_path.name,
            'file_size_mb': 0,
            'conversion_time': 0,
            'success': False,
            'wav_path': None,
            'wav_size_mb': 0,
            'error_message': None,
            'audio_info': {},
            'validation_passed': False
        }
        
        start_time = time.time()
        
        try:
            # 파일 정보 수집
            if m4a_path.exists():
                test_result['file_size_mb'] = round(m4a_path.stat().st_size / (1024*1024), 2)
                test_result['audio_info'] = get_audio_info(str(m4a_path))
            
            self.logger.info(f"🔄 변환 테스트 시작: {m4a_path.name}")
            
            # 진행률 콜백 함수
            progress_messages = []
            def progress_callback(percent, message):
                progress_messages.append(f"{percent}% - {message}")
                self.logger.info(f"   {percent}% - {message}")
            
            # M4A 변환 실행
            wav_path = self.converter.convert_to_wav(
                str(m4a_path), 
                target_sample_rate=16000,
                target_channels=1,
                progress_callback=progress_callback
            )
            
            conversion_time = time.time() - start_time
            test_result['conversion_time'] = round(conversion_time, 2)
            test_result['progress_messages'] = progress_messages
            
            if wav_path and os.path.exists(wav_path):
                test_result['success'] = True
                test_result['wav_path'] = wav_path
                test_result['wav_size_mb'] = round(os.path.getsize(wav_path) / (1024*1024), 2)
                
                # WAV 파일 검증
                test_result['validation_passed'] = self.converter._validate_wav_file(wav_path)
                
                if test_result['validation_passed']:
                    self.logger.info(f"✅ 변환 성공: {m4a_path.name} ({conversion_time:.2f}초)")
                else:
                    self.logger.warning(f"⚠️ 변환은 되었지만 검증 실패: {m4a_path.name}")
                
                # 결과 파일을 결과 디렉토리로 복사
                result_wav = self.results_dir / f"{m4a_path.stem}_converted.wav"
                shutil.copy2(wav_path, result_wav)
                test_result['result_wav_path'] = str(result_wav)
                
                # 임시 파일 정리
                self.converter.cleanup_temp_file(wav_path)
                
            else:
                test_result['error_message'] = "변환 실패 - 출력 파일 없음"
                self.logger.error(f"❌ 변환 실패: {m4a_path.name}")
                
        except Exception as e:
            conversion_time = time.time() - start_time
            test_result['conversion_time'] = round(conversion_time, 2)
            test_result['error_message'] = str(e)
            self.logger.error(f"❌ 변환 오류 {m4a_path.name}: {e}")
        
        return test_result
    
    def run_comprehensive_test(self):
        """종합 M4A 변환 테스트 실행"""
        
        self.logger.info("🎯 M4A 파일 변환 종합 테스트 시작")
        self.logger.info("=" * 60)
        
        # 시스템 환경 확인
        self.logger.info("🔧 시스템 환경:")
        self.logger.info(f"   FFmpeg: {'✅' if self.converter.ffmpeg_available else '❌'}")
        self.logger.info(f"   pydub: {'✅' if hasattr(self.converter, 'PYDUB_AVAILABLE') else '❌'}")
        self.logger.info(f"   librosa: {'✅' if hasattr(self.converter, 'LIBROSA_AVAILABLE') else '❌'}")
        
        # 테스트 파일 준비
        test_files = []
        
        # 1. 기존 M4A 파일 찾기
        existing_files = self.find_existing_m4a_files()
        test_files.extend(existing_files)
        
        # 2. 테스트용 M4A 파일 생성
        generated_files = self.create_test_m4a_files()
        test_files.extend(generated_files)
        
        if not test_files:
            self.logger.warning("⚠️ 테스트할 M4A 파일이 없습니다")
            return
        
        self.logger.info(f"📂 총 {len(test_files)}개 M4A 파일 테스트 예정")
        
        # 개별 파일 테스트
        for i, m4a_file in enumerate(test_files, 1):
            self.logger.info(f"\n🔍 테스트 {i}/{len(test_files)}: {m4a_file.name}")
            test_result = self.test_single_conversion(m4a_file)
            self.test_results.append(test_result)
        
        # 결과 분석 및 보고
        self.generate_test_report()
    
    def generate_test_report(self):
        """테스트 결과 보고서 생성"""
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 M4A 변환 테스트 결과 분석")
        self.logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        validated_tests = sum(1 for r in self.test_results if r['validation_passed'])
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        validation_rate = (validated_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 통계 요약
        self.logger.info(f"📈 테스트 통계:")
        self.logger.info(f"   총 테스트: {total_tests}개")
        self.logger.info(f"   변환 성공: {successful_tests}개 ({success_rate:.1f}%)")
        self.logger.info(f"   검증 통과: {validated_tests}개 ({validation_rate:.1f}%)")
        
        # 성능 분석
        if successful_tests > 0:
            avg_time = sum(r['conversion_time'] for r in self.test_results if r['success']) / successful_tests
            self.logger.info(f"   평균 변환 시간: {avg_time:.2f}초")
        
        # 실패 사례 분석
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            self.logger.info(f"\n❌ 실패 사례 ({len(failed_tests)}개):")
            for fail in failed_tests:
                self.logger.info(f"   - {fail['file_name']}: {fail['error_message']}")
        
        # 상세 결과를 JSON으로 저장
        report_file = self.results_dir / f"m4a_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump({
                'test_summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'validated_tests': validated_tests,
                    'success_rate': success_rate,
                    'validation_rate': validation_rate,
                    'avg_conversion_time': avg_time if successful_tests > 0 else 0
                },
                'detailed_results': self.test_results,
                'test_timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n💾 상세 보고서 저장: {report_file}")
        
        # 결론 및 권장사항
        if success_rate >= 90:
            self.logger.info("\n🎉 결론: M4A 파일 처리 성능 우수!")
        elif success_rate >= 70:
            self.logger.info("\n✅ 결론: M4A 파일 처리 양호 (일부 개선 필요)")
        else:
            self.logger.info("\n⚠️ 결론: M4A 파일 처리 개선 필요")
        
        # 권장사항
        recommendations = []
        if success_rate < 100:
            recommendations.append("실패 사례에 대한 추가 디버깅 필요")
        if validation_rate < success_rate:
            recommendations.append("WAV 파일 검증 로직 개선 필요")
        if not self.converter.ffmpeg_available:
            recommendations.append("FFmpeg 설치 권장 (최고 호환성)")
        
        if recommendations:
            self.logger.info("\n💡 권장사항:")
            for rec in recommendations:
                self.logger.info(f"   - {rec}")

def main():
    """메인 테스트 실행"""
    
    print("🎵 솔로몬드 AI - M4A 파일 변환 테스트 시스템")
    print("=" * 60)
    print("목적: 개선된 audio_converter의 M4A 처리 성능 검증")
    print("범위: 다양한 M4A 파일 형식에 대한 종합 테스트")
    print()
    
    # 테스트 실행
    tester = M4AConversionTester()
    
    try:
        tester.run_comprehensive_test()
        
        print("\n🎉 M4A 변환 테스트 완료!")
        print(f"📁 결과 파일: {tester.results_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 테스트 중단됨")
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # subprocess import 추가
    import subprocess
    from datetime import datetime
    
    main()