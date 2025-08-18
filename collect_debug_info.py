#!/usr/bin/env python3
"""
에러 상황 자동 수집 스크립트
Claude Code에 전달할 디버깅 정보를 자동으로 수집
"""

import sys
import subprocess
import json
import traceback
from datetime import datetime
from pathlib import Path

def collect_system_info():
    """시스템 정보 수집"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    try:
        # 메모리 정보
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        info['memory'] = result.stdout
    except:
        info['memory'] = 'N/A'
    
    try:
        # GPU 정보
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            info['gpu_memory'] = result.stdout.strip()
        else:
            info['gpu_memory'] = 'GPU 없음'
    except:
        info['gpu_memory'] = 'nvidia-smi 없음'
    
    return info

def collect_dependency_info():
    """의존성 정보 수집"""
    deps = {}
    
    packages = [
        'streamlit', 'whisper', 'openai-whisper', 'easyocr', 
        'transformers', 'torch', 'torchaudio', 'torchvision',
        'PIL', 'numpy', 'requests', 'aiohttp'
    ]
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {package}; print({package}.__version__)'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                deps[package] = result.stdout.strip()
            else:
                deps[package] = '설치되지 않음'
        except:
            deps[package] = '확인 실패'
    
    return deps

def collect_streamlit_logs():
    """Streamlit 로그 수집"""
    logs = {}
    
    # Streamlit 로그 파일들
    log_paths = [
        Path.home() / '.streamlit' / 'logs' / 'streamlit.log',
        Path('/tmp/streamlit.log'),
        Path('./streamlit.log')
    ]
    
    for log_path in log_paths:
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 마지막 50줄만
                    logs[str(log_path)] = ''.join(lines[-50:])
            except Exception as e:
                logs[str(log_path)] = f'읽기 실패: {e}'
    
    return logs

def check_audio_video_support():
    """오디오/비디오 지원 확인"""
    support = {}
    
    # FFmpeg 확인
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        support['ffmpeg'] = 'available' if result.returncode == 0 else 'missing'
    except:
        support['ffmpeg'] = 'missing'
    
    # Whisper 모델 확인
    try:
        import whisper
        models = whisper.available_models()
        support['whisper_models'] = models
    except Exception as e:
        support['whisper_models'] = f'error: {e}'
    
    # EasyOCR 언어 확인
    try:
        import easyocr
        reader = easyocr.Reader(['en', 'ko'])
        support['easyocr'] = 'initialized'
    except Exception as e:
        support['easyocr'] = f'error: {e}'
    
    return support

def test_analysis_components():
    """분석 컴포넌트 개별 테스트"""
    results = {}
    
    # Real Analysis Engine 임포트 테스트
    try:
        sys.path.append('/home/SOLOMONDd/claude/SOLOMONDd-ai-system')
        from core.real_analysis_engine import RealAnalysisEngine
        
        engine = RealAnalysisEngine()
        results['real_analysis_engine'] = 'import_success'
        
        # GPU 강제 설정 확인
        import os
        results['gpu_disabled'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')
        
    except Exception as e:
        results['real_analysis_engine'] = f'import_error: {e}'
        results['error_traceback'] = traceback.format_exc()
    
    return results

def generate_debug_report():
    """종합 디버깅 리포트 생성"""
    
    print("🔍 솔로몬드 AI 디버깅 정보 수집 중...")
    print("=" * 60)
    
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'error_context': {
            'files_uploaded': {
                'audio_files': 2,
                'image_files': 23,
                'total_files': 25
            },
            'error_description': '전체적으로 에러 발생',
            'platform': 'Windows Edge Browser'
        },
        'system_info': collect_system_info(),
        'dependencies': collect_dependency_info(),
        'streamlit_logs': collect_streamlit_logs(),
        'av_support': check_audio_video_support(),
        'component_tests': test_analysis_components()
    }
    
    # 리포트 저장
    report_path = Path('/home/SOLOMONDd/claude/SOLOMONDd-ai-system/debug_report.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 디버깅 리포트 저장: {report_path}")
    
    # 요약 출력
    print("\n📊 수집된 정보 요약:")
    print(f"   🐍 Python: {sys.version.split()[0]}")
    print(f"   🖥️ 플랫폼: {sys.platform}")
    print(f"   📦 주요 패키지:")
    
    key_packages = ['streamlit', 'whisper', 'easyocr', 'transformers', 'torch']
    for pkg in key_packages:
        version = report['dependencies'].get(pkg, 'N/A')
        status = "✅" if version not in ['설치되지 않음', '확인 실패'] else "❌"
        print(f"      {status} {pkg}: {version}")
    
    print(f"\n🔧 분석 엔진 상태:")
    engine_status = report['component_tests'].get('real_analysis_engine', 'unknown')
    engine_icon = "✅" if 'success' in engine_status else "❌"
    print(f"   {engine_icon} Real Analysis Engine: {engine_status}")
    
    gpu_status = report['component_tests'].get('gpu_disabled', 'unknown')
    print(f"   🎮 GPU 설정: {gpu_status}")
    
    print(f"\n💡 Claude Code 전달 방법:")
    print(f"   1. 파일 내용 확인: cat {report_path}")
    print(f"   2. 특정 에러 메시지가 있다면 추가로 복사해서 전달")
    print(f"   3. DEBUG_REPORT.md 파일도 함께 작성하여 전달")
    
    return report

if __name__ == "__main__":
    try:
        generate_debug_report()
    except Exception as e:
        print(f"❌ 디버깅 정보 수집 실패: {e}")
        traceback.print_exc()