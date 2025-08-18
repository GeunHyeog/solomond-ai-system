# -*- coding: utf-8 -*-
"""
빠른 화자 구분 분석기 - 시간 단축 버전
첫 번째 오디오 파일만 빠르게 분석
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 경로 추가  
sys.path.append(str(Path(__file__).parent))

def quick_speaker_analysis():
    """첫 번째 오디오 파일만 빠른 화자 분석"""
    
    print("=== 빠른 화자 구분 분석 ===")
    print("첫 번째 오디오 파일만 분석합니다.")
    
    # 첫 번째 오디오 파일 찾기
    audio_folder = "user_files/audio"
    
    if not os.path.exists(audio_folder):
        print("user_files/audio 폴더가 없습니다.")
        return
    
    audio_files = []
    for file in os.listdir(audio_folder):
        if file.lower().endswith(('.wav', '.m4a', '.mp3')):
            audio_files.append(os.path.join(audio_folder, file))
    
    if not audio_files:
        print("오디오 파일이 없습니다.")
        return
    
    # 가장 작은 파일부터 분석 (시간 단축)
    audio_files.sort(key=lambda x: os.path.getsize(x))
    first_file = audio_files[0]
    file_size = os.path.getsize(first_file)
    
    print(f"분석할 파일: {os.path.basename(first_file)}")
    print(f"파일 크기: {file_size//1024}KB")
    
    if file_size > 5 * 1024 * 1024:  # 5MB 이상
        print("파일이 너무 큽니다. 더 작은 파일로 테스트해주세요.")
        return
    
    try:
        import whisper
        
        print("\nWhisper 분석 시작...")
        start_time = time.time()
        
        # 작은 모델 사용으로 속도 향상
        model = whisper.load_model("tiny")  # base 대신 tiny 사용
        
        # 간단한 transcription (word_timestamps 제외로 속도 향상)
        result = model.transcribe(first_file, language="ko")
        
        analysis_time = time.time() - start_time
        print(f"분석 완료: {analysis_time:.1f}초")
        
        # 텍스트 추출
        full_text = result['text']
        segments = result.get('segments', [])
        
        print(f"\n전체 텍스트 길이: {len(full_text)}글자")
        print(f"세그먼트 수: {len(segments)}개")
        
        # 간단한 화자 구분 (패턴 기반)
        speaker_analysis = simple_speaker_detection(segments, full_text)
        
        print("\n화자 구분 결과:")
        print("="*40)
        
        for speaker, info in speaker_analysis.items():
            print(f"{info['role']}: {info['segments']}회 발언")
            if info['sample_text']:
                preview = info['sample_text'][:50] + "..." if len(info['sample_text']) > 50 else info['sample_text']
                print(f"  샘플: {preview}")
        
        # 간단한 결과 저장
        save_quick_results(first_file, speaker_analysis, full_text)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def simple_speaker_detection(segments, full_text):
    """간단한 화자 감지 로직"""
    
    # 기본 화자 정보 초기화
    speakers = {
        'moderator': {'role': '사회자', 'segments': 0, 'texts': [], 'sample_text': ''},
        'speaker1': {'role': '발표자 1', 'segments': 0, 'texts': [], 'sample_text': ''},
        'speaker2': {'role': '발표자 2', 'segments': 0, 'texts': [], 'sample_text': ''},
        'speaker3': {'role': '발표자 3', 'segments': 0, 'texts': [], 'sample_text': ''}
    }
    
    # 사회자 키워드
    moderator_keywords = ['감사', '네', '다음', '질문', '발표', '소개', '시간', '마이크']
    
    current_speaker = 'moderator'  # 시작은 사회자
    speaker_rotation = ['moderator', 'speaker1', 'speaker2', 'speaker3']
    rotation_index = 0
    
    for i, segment in enumerate(segments):
        text = segment['text'].strip()
        text_length = len(text)
        
        # 화자 변경 조건
        change_speaker = False
        
        # 1. 긴 발언은 발표자일 가능성
        if text_length > 100:
            change_speaker = True
            # 발표자로 변경
            rotation_index = (rotation_index + 1) % 4
            if rotation_index == 0:  # 사회자 턴이면 다음으로
                rotation_index = 1
            current_speaker = speaker_rotation[rotation_index]
        
        # 2. 사회자 키워드가 있으면 사회자
        elif any(keyword in text for keyword in moderator_keywords):
            current_speaker = 'moderator'
            rotation_index = 0
        
        # 3. 매우 짧은 발언도 사회자일 가능성
        elif text_length < 20:
            current_speaker = 'moderator'
        
        # 화자 정보 업데이트
        speakers[current_speaker]['segments'] += 1
        speakers[current_speaker]['texts'].append(text)
    
    # 각 화자의 대표 텍스트 설정
    for speaker_id, info in speakers.items():
        if info['texts']:
            # 가장 긴 텍스트를 샘플로
            info['sample_text'] = max(info['texts'], key=len)
    
    # 발언이 없는 화자 제거
    active_speakers = {k: v for k, v in speakers.items() if v['segments'] > 0}
    
    return active_speakers

def save_quick_results(file_path, speaker_analysis, full_text):
    """빠른 결과 저장"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"quick_speaker_analysis_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("빠른 화자 구분 분석 결과\n")
            f.write(f"파일: {os.path.basename(file_path)}\n")
            f.write(f"분석 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            f.write("화자별 분석:\n")
            for speaker, info in speaker_analysis.items():
                f.write(f"{info['role']}: {info['segments']}회 발언\n")
                if info['sample_text']:
                    f.write(f"  샘플: {info['sample_text'][:100]}...\n")
                f.write("\n")
            
            f.write("\n전체 텍스트:\n")
            f.write("-" * 30 + "\n")
            f.write(full_text)
        
        print(f"\n결과 저장: {filename}")
        
    except Exception as e:
        print(f"저장 실패: {str(e)}")

def main():
    quick_speaker_analysis()

if __name__ == "__main__":
    main()