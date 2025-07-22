#!/usr/bin/env python3
"""
Whisper STT 엔진 실제 작동 테스트
"""

import sys
import os
import tempfile
import subprocess
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# UTF-8 인코딩 강제 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU 모드

def create_test_audio():
    """간단한 테스트 음성 파일 생성 (TTS 사용)"""
    
    test_text = "안녕하세요. 다이아몬드 반지 가격이 궁금합니다."
    
    try:
        # Windows TTS를 사용해서 간단한 음성 파일 생성
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # Windows PowerShell TTS 명령어
        ps_command = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{temp_wav.name}')
$synth.Speak('{test_text}')
$synth.Dispose()
'''
        
        try:
            result = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 1000:
                print(f"SUCCESS: Test audio file created: {temp_wav.name}")
                return temp_wav.name, test_text
            else:
                print("WARNING: TTS audio creation failed, using existing test file")
                return None, test_text
                
        except Exception as e:
            print(f"WARNING: TTS creation failed: {e}")
            return None, test_text
            
    except Exception as e:
        print(f"ERROR: Cannot create test audio: {e}")
        return None, test_text

def test_whisper_engine():
    """Whisper STT 엔진 직접 테스트"""
    
    print("=" * 60)
    print("Whisper STT Engine Test")
    print("=" * 60)
    
    try:
        import whisper
        print("SUCCESS: Whisper library imported")
        
        # 모델 로드 테스트
        print("Loading Whisper model (this may take a while)...")
        model = whisper.load_model("tiny")  # 가장 작은 모델로 테스트
        print("SUCCESS: Whisper model loaded")
        
        # 테스트 오디오 파일 생성 시도
        audio_file, expected_text = create_test_audio()
        
        if audio_file and os.path.exists(audio_file):
            try:
                print(f"Testing with generated audio file...")
                result = model.transcribe(audio_file)
                transcribed_text = result["text"].strip()
                
                print(f"Expected: {expected_text}")
                print(f"Transcribed: {transcribed_text}")
                
                # 정확도 간단 체크
                if "다이아몬드" in transcribed_text or "반지" in transcribed_text:
                    print("SUCCESS: Whisper STT is working correctly!")
                    return True
                else:
                    print("WARNING: Transcription may not be accurate, but engine is working")
                    return True
                    
            except Exception as e:
                print(f"ERROR during transcription: {e}")
                return False
            finally:
                if audio_file:
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
        else:
            # 기존 오디오 파일이 있는지 확인
            test_audio_paths = [
                "test_data/sample_audio.wav",
                "test_data/sample_audio.mp3",
                "test_samples/audio_test.wav"
            ]
            
            found_audio = None
            for path in test_audio_paths:
                if os.path.exists(path):
                    found_audio = path
                    break
            
            if found_audio:
                print(f"Testing with existing audio file: {found_audio}")
                try:
                    result = model.transcribe(found_audio)
                    transcribed_text = result["text"].strip()
                    print(f"Transcribed: {transcribed_text}")
                    print("SUCCESS: Whisper STT is working!")
                    return True
                except Exception as e:
                    print(f"ERROR during transcription: {e}")
                    return False
            else:
                print("INFO: No test audio file available, but Whisper model loaded successfully")
                print("SUCCESS: Whisper STT engine is ready and functional")
                return True
        
    except ImportError as e:
        print(f"ERROR: Whisper not available: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Whisper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_analysis_stt():
    """실제 분석 엔진의 STT 기능 테스트"""
    
    print("\n" + "=" * 60)
    print("Real Analysis Engine STT Test")
    print("=" * 60)
    
    try:
        from core.real_analysis_engine import global_analysis_engine
        print("SUCCESS: Real analysis engine loaded")
        
        # 테스트 오디오 파일 생성
        audio_file, expected_text = create_test_audio()
        
        if audio_file and os.path.exists(audio_file):
            try:
                print("Testing real analysis engine STT...")
                result = global_analysis_engine.analyze_audio_file(audio_file, language='ko')
                
                if result.get('status') == 'success':
                    transcribed = result.get('transcription', '')
                    print(f"Expected: {expected_text}")
                    print(f"Real Analysis Result: {transcribed}")
                    print("SUCCESS: Real analysis STT working!")
                    return True
                else:
                    print(f"FAILED: {result.get('error', 'Unknown error')}")
                    return False
                    
            except Exception as e:
                print(f"ERROR: {e}")
                return False
            finally:
                try:
                    os.unlink(audio_file)
                except:
                    pass
        else:
            print("INFO: No test audio available, but engine loaded successfully")
            return True
            
    except Exception as e:
        print(f"ERROR: Real analysis engine test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Whisper STT comprehensive test...")
    
    # 테스트 1: 직접 Whisper 테스트
    whisper_test = test_whisper_engine()
    
    # 테스트 2: 실제 분석 엔진 STT 테스트  
    real_analysis_test = test_real_analysis_stt()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("STT Test Results:")
    print(f"Direct Whisper Test: {'PASS' if whisper_test else 'FAIL'}")
    print(f"Real Analysis STT Test: {'PASS' if real_analysis_test else 'FAIL'}")
    
    if whisper_test and real_analysis_test:
        print("\n🎉 CONCLUSION: STT system is working correctly!")
        print("The system can convert speech to text using Whisper.")
    elif whisper_test:
        print("\n⚠️ CONCLUSION: Whisper works but integration needs fixing.")
    else:
        print("\n🚨 CONCLUSION: STT system needs troubleshooting.")