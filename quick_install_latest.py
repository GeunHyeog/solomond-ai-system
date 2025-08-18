#!/usr/bin/env python3
"""
최신 모델 빠른 설치
"""

import subprocess
import os
import sys

def find_ollama():
    """Ollama 실행파일 찾기"""
    
    paths = [
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.getenv("USERNAME")),
        r"C:\Program Files\Ollama\ollama.exe"
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Ollama 찾음: {path}")
            return path
    
    print("기본 ollama 명령어 사용")
    return "ollama"

def install_latest_models():
    """최신 모델 설치"""
    
    print("=== 2025년 최신 모델 설치 ===")
    
    ollama_path = find_ollama()
    
    # 한국어에 최적화된 최신 모델들
    models = [
        ("qwen2.5:7b", "Qwen 2.5 7B - 한국어 최강"),
        ("llama3.2:8b", "Llama 3.2 8B - Meta 최신"),
        ("gemma2:9b", "Gemma 2 9B - Google 최신")
    ]
    
    installed = []
    
    for model_id, description in models:
        print(f"\n{description} 설치 중...")
        
        try:
            # ollama pull 실행
            result = subprocess.run([ollama_path, "pull", model_id],
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"설치 완료: {model_id}")
                installed.append(model_id)
            else:
                print(f"설치 실패: {model_id}")
                print(f"오류: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"시간 초과: {model_id}")
        except Exception as e:
            print(f"오류: {str(e)}")
    
    return installed

def main():
    """메인 실행"""
    
    print("최신 AI 모델 빠른 설치")
    print("=" * 30)
    
    installed = install_latest_models()
    
    if installed:
        print(f"\n설치 완료: {len(installed)}개")
        for model in installed:
            print(f"  - {model}")
        
        print(f"\n다음 명령어로 테스트:")
        print(f"python check_gemma3_status.py")
    else:
        print(f"\n설치 실패")
        print(f"수동으로 시도:")
        print(f"ollama pull qwen2.5:7b")

if __name__ == "__main__":
    main()