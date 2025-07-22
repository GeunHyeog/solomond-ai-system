#!/usr/bin/env python3
"""
M4A 파일 처리 문제 빠른 해결
"""

import os
from pydub import AudioSegment

def convert_m4a_to_wav(m4a_file_path):
    """M4A 파일을 WAV로 변환"""
    try:
        print(f"Converting {m4a_file_path}...")
        
        # M4A 파일 로드
        audio = AudioSegment.from_file(m4a_file_path, format="m4a")
        
        # WAV 파일로 변환
        wav_path = m4a_file_path.replace('.m4a', '.wav')
        audio.export(wav_path, format="wav", parameters=["-ar", "16000"])
        
        print(f"✅ 변환 완료: {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return None

# 사용 예시
if __name__ == "__main__":
    # 테스트할 M4A 파일 경로를 입력하세요
    m4a_file = input("M4A 파일 경로를 입력하세요: ")
    
    if os.path.exists(m4a_file):
        wav_file = convert_m4a_to_wav(m4a_file)
        if wav_file:
            print(f"✅ 변환된 파일: {wav_file}")
            print("이제 이 WAV 파일을 Streamlit UI에 업로드하세요.")
    else:
        print("❌ 파일을 찾을 수 없습니다.")