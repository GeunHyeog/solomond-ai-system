#!/usr/bin/env python3
"""
M4A 파일 자동 변환 유틸리티
아이폰 녹음 파일을 Whisper STT가 처리할 수 있는 WAV 파일로 변환
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

def convert_m4a_to_wav(m4a_file_path: str, output_dir: str = None) -> str:
    """
    M4A 파일을 WAV 파일로 변환
    
    Args:
        m4a_file_path: M4A 파일 경로
        output_dir: 출력 디렉토리 (None이면 원본과 같은 디렉토리)
    
    Returns:
        변환된 WAV 파일 경로
    """
    
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub 라이브러리가 설치되지 않았습니다. pip install pydub로 설치하세요.")
    
    if not os.path.exists(m4a_file_path):
        raise FileNotFoundError(f"M4A 파일을 찾을 수 없습니다: {m4a_file_path}")
    
    # 출력 파일 경로 생성
    input_path = Path(m4a_file_path)
    if output_dir:
        output_path = Path(output_dir) / (input_path.stem + ".wav")
    else:
        output_path = input_path.parent / (input_path.stem + ".wav")
    
    try:
        print(f"🔄 변환 시작: {input_path.name} → {output_path.name}")
        
        # M4A 파일 로드
        audio = AudioSegment.from_file(str(input_path), format="m4a")
        
        # WAV 파일로 변환 (16kHz, Mono - Whisper 최적화)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(output_path), format="wav")
        
        print(f"✅ 변환 완료: {output_path}")
        print(f"   - 샘플레이트: 16kHz")
        print(f"   - 채널: Mono")
        print(f"   - 파일 크기: {output_path.stat().st_size / 1024:.1f} KB")
        
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"M4A 변환 실패: {e}")

def batch_convert_m4a_files(directory: str, output_dir: str = None) -> list:
    """
    디렉토리 내의 모든 M4A 파일을 일괄 변환
    
    Args:
        directory: M4A 파일들이 있는 디렉토리
        output_dir: 출력 디렉토리
    
    Returns:
        변환된 WAV 파일 경로들의 리스트
    """
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")
    
    # M4A 파일들 찾기
    m4a_files = list(directory.glob("*.m4a")) + list(directory.glob("*.M4A"))
    
    if not m4a_files:
        print(f"⚠️ {directory}에서 M4A 파일을 찾을 수 없습니다.")
        return []
    
    print(f"📁 {len(m4a_files)}개의 M4A 파일을 발견했습니다.")
    
    converted_files = []
    
    for m4a_file in m4a_files:
        try:
            wav_file = convert_m4a_to_wav(str(m4a_file), output_dir)
            converted_files.append(wav_file)
        except Exception as e:
            print(f"❌ 변환 실패 {m4a_file.name}: {e}")
    
    print(f"\n📊 변환 완료: {len(converted_files)}/{len(m4a_files)}개 파일")
    return converted_files

def main():
    """명령줄 인터페이스"""
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python quick_m4a_converter.py <M4A_파일_또는_디렉토리> [출력_디렉토리]")
        print("")
        print("예시:")
        print("  python quick_m4a_converter.py recording.m4a")
        print("  python quick_m4a_converter.py ./recordings/ ./converted/")
        return
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not PYDUB_AVAILABLE:
        print("❌ pydub 라이브러리가 필요합니다.")
        print("설치: pip install pydub")
        return
    
    try:
        if os.path.isfile(input_path):
            # 단일 파일 변환
            if input_path.lower().endswith(('.m4a',)):
                wav_file = convert_m4a_to_wav(input_path, output_dir)
                print(f"\n🎉 변환 성공! 이제 이 파일을 Streamlit UI에 업로드하세요:")
                print(f"   {wav_file}")
            else:
                print("❌ M4A 파일이 아닙니다.")
                
        elif os.path.isdir(input_path):
            # 디렉토리 내 일괄 변환
            converted_files = batch_convert_m4a_files(input_path, output_dir)
            if converted_files:
                print(f"\n🎉 일괄 변환 완료! 변환된 파일들:")
                for wav_file in converted_files:
                    print(f"   - {wav_file}")
        else:
            print(f"❌ 파일 또는 디렉토리를 찾을 수 없습니다: {input_path}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()