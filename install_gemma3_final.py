#!/usr/bin/env python3
"""
GEMMA3 최종 설치 시도
"""

import subprocess
import os
import time

def find_ollama():
    """Ollama 실행파일 찾기"""
    
    username = os.getenv("USERNAME", "PC_58410")
    paths = [
        rf"C:\Users\{username}\AppData\Local\Programs\Ollama\ollama.exe",
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe"
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Found Ollama: {path}")
            return path
    
    print("Using default ollama command")
    return "ollama"

def install_latest_models():
    """최신 모델들 설치 시도 (GEMMA3 + Llama 3.2)"""
    
    print("=== 2025년 최신 모델 설치 시도 ===")
    
    ollama_path = find_ollama()
    
    # 2025년 최신 모델들
    latest_models = [
        # GEMMA3 시리즈 (Google 3세대)
        ("gemma3:9b", "GEMMA3 9B - Google 3세대 최신"),
        ("gemma3:7b", "GEMMA3 7B - Google 3세대"),
        ("gemma3:2b", "GEMMA3 2B - Google 3세대 경량"),
        
        # Llama 3.2 시리즈 (Meta 최신)
        ("llama3.2:8b", "Llama 3.2 8B - Meta 최신 모델"),
        ("llama3.2:3b", "Llama 3.2 3B - Meta 경량 모델"),
        ("llama3.2:1b", "Llama 3.2 1B - Meta 초경량 모델")
    ]
    
    installed = []
    
    for model_id, description in latest_models:
        print(f"\n{description} 설치 중...")
        print(f"모델: {model_id}")
        
        try:
            print("다운로드 시작...")
            result = subprocess.run([ollama_path, "pull", model_id],
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"성공: {model_id} 설치 완료!")
                installed.append(model_id)
                break  # 하나만 성공해도 충분
            else:
                print(f"실패: {model_id}")
                print(f"오류: {result.stderr}")
                
                # 모델이 아직 존재하지 않을 가능성 확인
                if "not found" in result.stderr.lower() or "pull access denied" in result.stderr.lower():
                    if "gemma3" in model_id:
                        print(f"  -> GEMMA3가 아직 공식 출시되지 않았을 수 있습니다")
                    elif "llama3.2" in model_id:
                        print(f"  -> Llama 3.2가 Ollama에 아직 등록되지 않았을 수 있습니다")
                
        except subprocess.TimeoutExpired:
            print(f"시간 초과: {model_id}")
        except Exception as e:
            print(f"설치 오류: {str(e)}")
    
    return installed

def check_current_status():
    """현재 설치된 모델 상태 확인"""
    
    print("\n=== 현재 설치된 모델 확인 ===")
    
    try:
        import requests
        r = requests.get('http://localhost:11434/api/tags')
        if r.status_code == 200:
            models = r.json()['models']
            
            print("설치된 최신 모델들:")
            has_gen3 = False
            
            for model in models:
                name = model['name']
                size_gb = model['size'] / (1024**3)
                
                if 'qwen3' in name or 'gemma3' in name:
                    has_gen3 = True
                    print(f"  🥇 {name} ({size_gb:.1f}GB) - 3세대 최신!")
                elif 'gemma2' in name or 'qwen2.5' in name:
                    print(f"  🥈 {name} ({size_gb:.1f}GB) - 2세대 최신")
                else:
                    print(f"  📦 {name} ({size_gb:.1f}GB)")
            
            if has_gen3:
                print(f"\n✅ 3세대 최신 모델 보유 중!")
            else:
                print(f"\n⚠️ 아직 3세대 모델 없음 (2세대가 현재 최신)")
            
            return True
    except Exception as e:
        print(f"상태 확인 실패: {str(e)}")
        return False

def main():
    """메인 실행"""
    
    print("🚀 GEMMA3 최종 설치 시도")
    print("=" * 40)
    
    # 1. 현재 상태 확인
    check_current_status()
    
    # 2. 최신 모델들 설치 시도 (GEMMA3 + Llama 3.2)
    installed = install_latest_models()
    
    if installed:
        print(f"\n🎉 최신 모델 설치 성공!")
        for model in installed:
            if "gemma3" in model:
                print(f"  ✓ {model} (Google 3세대)")
            elif "llama3.2" in model:
                print(f"  ✓ {model} (Meta 최신)")
            else:
                print(f"  ✓ {model}")
        
        print(f"\n다음 단계:")
        print(f"1. 새 모델들을 솔로몬드 AI에 통합")
        print(f"2. Qwen3 vs GEMMA3 vs Llama3.2 성능 비교")
        print(f"3. 최고 성능 모델 선정")
        
    else:
        print(f"\n❌ 추가 모델 설치 실패")
        print(f"가능한 원인:")
        print(f"1. GEMMA3가 아직 정식 출시되지 않음")
        print(f"2. Llama 3.2가 Ollama에 아직 등록되지 않음")
        print(f"3. 액세스 권한 문제")
        
        print(f"\n✅ 현재 사용 가능한 최신 모델:")
        print(f"  - Qwen3:8b (3세대, 한국어 최강)")
        print(f"  - Gemma2:9b (2세대, Google 최신)")
        print(f"  - Gemma2:2b (2세대, Google 경량)")
        
        print(f"\n🎯 권장사항:")
        print(f"Qwen3:8b가 현재 가장 최신이고 강력한 모델입니다!")
    
    # 3. 최종 상태 재확인
    print(f"\n=== 최종 설치 상태 ===")
    check_current_status()

if __name__ == "__main__":
    main()