#!/usr/bin/env python3
"""
Librosa를 사용한 실제 음성 분석
FFmpeg 없이도 가능한 기본 음성 분석
"""

import librosa
import numpy as np
import os
import time
from pathlib import Path

def analyze_audio_with_librosa(file_path):
    """Librosa로 음성 파일 분석"""
    
    print(f"🎯 Librosa 음성 분석 시작")
    print(f"📁 파일: {os.path.basename(file_path)}")
    print("=" * 50)
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    # 파일 크기 확인
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"📏 파일 크기: {file_size:.2f} MB")
    
    try:
        print("🔄 음성 데이터 로딩...")
        start_time = time.time()
        
        # 음성 파일 로드 (Librosa는 다양한 포맷 지원)
        y, sr = librosa.load(file_path, sr=None)
        
        load_time = time.time() - start_time
        print(f"✅ 로딩 완료 ({load_time:.1f}초)")
        
        # 기본 음성 특성 분석
        print("🔍 음성 특성 분석 중...")
        
        # 1. 기본 정보
        duration = len(y) / sr
        
        # 2. 음성 특성
        rms_energy = librosa.feature.rms(y=y)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # 3. MFCC 특성 (음성 인식용)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 4. 음성 활성 구간 감지
        intervals = librosa.effects.split(y, top_db=20)
        speech_ratio = sum([interval[1] - interval[0] for interval in intervals]) / len(y)
        
        # 5. 품질 평가
        snr_estimate = np.mean(rms_energy) / (np.std(rms_energy) + 1e-8)
        
        print(f"✅ 분석 완료")
        
        # 결과 출력
        print(f"\n📊 음성 분석 결과:")
        print(f"   ⏱️ 실제 길이: {duration:.1f}초 ({duration/60:.1f}분)")
        print(f"   🎵 샘플링 레이트: {sr} Hz")
        print(f"   📈 평균 에너지: {np.mean(rms_energy):.4f}")
        print(f"   🎼 평균 스펙트럴 중심: {np.mean(spectral_centroids):.1f} Hz")
        print(f"   📊 영교차율: {np.mean(zero_crossing_rate):.4f}")
        print(f"   🗣️ 음성 활성 비율: {speech_ratio:.2%}")
        print(f"   ⭐ 품질 점수: {snr_estimate:.2f}")
        
        # 품질 평가
        if snr_estimate > 10:
            quality = "우수"
        elif snr_estimate > 5:
            quality = "양호"
        else:
            quality = "개선 필요"
        
        print(f"   🏆 전체 품질: {quality}")
        
        return {
            "file_name": os.path.basename(file_path),
            "file_size_mb": round(file_size, 2),
            "duration_seconds": round(duration, 1),
            "duration_minutes": round(duration/60, 1),
            "sample_rate": sr,
            "avg_energy": round(np.mean(rms_energy), 4),
            "avg_spectral_centroid": round(np.mean(spectral_centroids), 1),
            "zero_crossing_rate": round(np.mean(zero_crossing_rate), 4),
            "speech_ratio": round(speech_ratio, 3),
            "quality_score": round(snr_estimate, 2),
            "quality_rating": quality,
            "processing_time": round(load_time, 1),
            "speech_intervals_count": len(intervals)
        }
        
    except Exception as e:
        print(f"❌ 분석 중 오류: {str(e)}")
        return None

def main():
    """메인 실행 함수"""
    
    print("🚀 Librosa 실제 음성 분석")
    print("💎 홍콩 세미나 음성 → 특성 분석")
    print("=" * 60)
    
    # 테스트할 음성 파일들
    base_path = "/mnt/c/Users/PC_58410/Desktop/근혁/세미나/202506홍콩쇼/D1"
    
    audio_files = [
        {
            "path": f"{base_path}/새로운 녹음 2.m4a",
            "description": "추가 음성 (1MB)"
        },
        {
            "path": f"{base_path}/새로운 녹음.m4a", 
            "description": "메인 음성 (27MB)"
        }
    ]
    
    results = []
    
    for i, audio_info in enumerate(audio_files, 1):
        print(f"\n🎵 분석 {i}/{len(audio_files)}: {audio_info['description']}")
        print("=" * 50)
        
        result = analyze_audio_with_librosa(audio_info["path"])
        
        if result:
            results.append(result)
            print("✅ 분석 성공")
        else:
            print("❌ 분석 실패")
        
        print()
    
    # 최종 결과 요약
    print("📊 Librosa 분석 최종 요약")
    print("=" * 60)
    print(f"✅ 성공적으로 분석된 파일: {len(results)}/{len(audio_files)}")
    
    if results:
        print("\n📈 분석 결과 비교:")
        for result in results:
            print(f"   📁 {result['file_name']}")
            print(f"      크기: {result['file_size_mb']}MB")
            print(f"      길이: {result['duration_minutes']}분")
            print(f"      품질: {result['quality_rating']} (점수: {result['quality_score']})")
            print(f"      음성비율: {result['speech_ratio']*100:.1f}%")
            print(f"      처리시간: {result['processing_time']}초")
            print()

if __name__ == "__main__":
    main()