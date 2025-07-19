#!/usr/bin/env python3
"""
실제 Whisper로 홍콩 세미나 음성 분석
"""

import whisper
import os
import time
from pathlib import Path

def transcribe_audio(file_path, model_size="base"):
    """Whisper로 실제 음성-텍스트 변환"""
    
    print(f"🎯 Whisper 실제 음성 분석 시작")
    print(f"📁 파일: {os.path.basename(file_path)}")
    print(f"🧠 모델: {model_size}")
    print("=" * 50)
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    # 파일 크기 확인
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"📏 파일 크기: {file_size:.2f} MB")
    
    try:
        # Whisper 모델 로드
        print(f"🔄 Whisper {model_size} 모델 로딩...")
        start_time = time.time()
        model = whisper.load_model(model_size)
        load_time = time.time() - start_time
        print(f"✅ 모델 로드 완료 ({load_time:.1f}초)")
        
        # 음성 변환 시작
        print(f"🎤 음성-텍스트 변환 중...")
        transcribe_start = time.time()
        
        result = model.transcribe(file_path, language="ko")
        
        transcribe_time = time.time() - transcribe_start
        print(f"✅ 변환 완료 ({transcribe_time:.1f}초)")
        
        # 결과 분석
        text = result["text"]
        segments = result["segments"]
        detected_language = result["language"]
        
        print(f"\n📊 분석 결과:")
        print(f"   🌐 감지된 언어: {detected_language}")
        print(f"   ⏱️ 처리 시간: {transcribe_time:.1f}초")
        print(f"   📝 세그먼트 수: {len(segments)}")
        print(f"   📄 텍스트 길이: {len(text)} 글자")
        
        # 텍스트 출력 (처음 500자)
        print(f"\n📄 변환된 텍스트 (처음 500자):")
        print("-" * 40)
        print(text[:500] + ("..." if len(text) > 500 else ""))
        print("-" * 40)
        
        return {
            "file_name": os.path.basename(file_path),
            "file_size_mb": round(file_size, 2),
            "model_used": model_size,
            "detected_language": detected_language,
            "processing_time": round(transcribe_time, 1),
            "text_length": len(text),
            "segments_count": len(segments),
            "full_text": text,
            "segments": segments
        }
        
    except Exception as e:
        print(f"❌ 변환 중 오류: {str(e)}")
        return None

def main():
    """메인 실행 함수"""
    
    print("🚀 Whisper 실제 음성 분석 테스트")
    print("💎 홍콩 세미나 음성 → 텍스트 변환")
    print("=" * 60)
    
    # 테스트할 음성 파일들
    base_path = "/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1"
    
    audio_files = [
        {
            "path": f"{base_path}/새로운 녹음 2.m4a",
            "description": "추가 음성 (1MB) - 빠른 테스트용"
        },
        {
            "path": f"{base_path}/새로운 녹음.m4a", 
            "description": "메인 음성 (27MB) - 실제 세미나"
        }
    ]
    
    results = []
    
    for i, audio_info in enumerate(audio_files, 1):
        print(f"\n🎵 테스트 {i}/{len(audio_files)}: {audio_info['description']}")
        print("=" * 50)
        
        result = transcribe_audio(audio_info["path"])
        
        if result:
            results.append(result)
            print("✅ 성공적으로 변환됨")
        else:
            print("❌ 변환 실패")
        
        print()
        
        # 사용자 확인 (큰 파일 처리 전) - 자동으로 건너뛰기
        if i == 1 and len(audio_files) > 1:
            print("⏸️ FFmpeg 설치 후 큰 파일 처리 예정")
            break
    
    # 최종 결과 요약
    print("📊 Whisper 분석 최종 요약")
    print("=" * 60)
    print(f"✅ 성공적으로 변환된 파일: {len(results)}/{len(audio_files)}")
    
    if results:
        print("\n📝 변환 결과:")
        for result in results:
            print(f"   📁 {result['file_name']}")
            print(f"      크기: {result['file_size_mb']}MB")
            print(f"      언어: {result['detected_language']}")
            print(f"      처리시간: {result['processing_time']}초")
            print(f"      텍스트: {result['text_length']}글자")
            print(f"      세그먼트: {result['segments_count']}개")
            print()

if __name__ == "__main__":
    main()