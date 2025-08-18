#!/usr/bin/env python3
"""
진짜 GEMMA3 모델 설치 (공식 파라미터 버전)
Google 공식: 1B, 4B, 12B, 27B
"""

import subprocess
import os
import time

def find_ollama():
    username = os.getenv("USERNAME", "PC_58410")
    paths = [
        f"C:\\Users\\{username}\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
        "C:\\Program Files\\Ollama\\ollama.exe"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return "ollama"

def install_real_gemma3():
    """실제 GEMMA3 모델들 설치"""
    print("Installing REAL GEMMA3 Models (Official Parameters)")
    print("=" * 60)
    
    ollama_path = find_ollama()
    
    # Google 공식 GEMMA3 파라미터들
    official_gemma3 = [
        ("gemma3:27b", "GEMMA3 27B - Highest Performance"),
        ("gemma3:12b", "GEMMA3 12B - Balanced Performance"),
        ("gemma3:4b", "GEMMA3 4B - Efficient Performance"),
        ("gemma3:1b", "GEMMA3 1B - Lightweight")
    ]
    
    installed = []
    
    for model_id, description in official_gemma3:
        print(f"\\n{description}")
        print(f"Downloading: {model_id}")
        print("Please wait... (This may take 5-15 minutes)")
        
        try:
            # 진행 상황 표시하면서 설치
            process = subprocess.Popen(
                [ollama_path, "pull", model_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 간단한 진행 표시
            dots = 0
            while process.poll() is None:
                print(f"Downloading{'.' * (dots % 4)}", end="\\r")
                dots += 1
                time.sleep(2)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"\\nSUCCESS: {model_id} installed!")
                installed.append(model_id)
                
                # 4B 이상 설치되면 중단 (용량 고려)
                if "4b" in model_id or "12b" in model_id:
                    print("\\nStopping here to save space...")
                    break
                    
            else:
                print(f"\\nFAILED: {model_id}")
                print(f"Error: {stderr}")
                
        except Exception as e:
            print(f"\\nERROR: {str(e)}")
    
    return installed

def test_gemma3_performance(models):
    """GEMMA3 성능 테스트"""
    print("\\n=== GEMMA3 Performance Test ===")
    
    if not models:
        print("No GEMMA3 models to test")
        return
    
    # 가장 작은 모델로 테스트
    test_model = models[0]
    print(f"Testing: {test_model}")
    
    try:
        import requests
        
        payload = {
            "model": test_model,
            "prompt": "한국어로 답변해주세요: 결혼반지 200만원 예산으로 추천해주세요.",
            "stream": False
        }
        
        print("Running performance test...")
        start_time = time.time()
        
        response = requests.post("http://localhost:11434/api/generate",
                               json=payload, timeout=60)
        
        if response.status_code == 200:
            end_time = time.time()
            result = response.json()
            answer = result.get('response', '')
            
            processing_time = end_time - start_time
            
            print(f"\\nGEMMA3 Test Results:")
            print(f"  Model: {test_model}")
            print(f"  Processing Time: {processing_time:.2f} seconds")
            print(f"  Response Length: {len(answer)} characters")
            print(f"  Korean Support: {'YES' if any(ord(c) > 127 for c in answer) else 'NO'}")
            print(f"  Sample Response: {answer[:100]}...")
            
            return True
        else:
            print(f"API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

def main():
    """메인 실행"""
    print("Real GEMMA3 Installation (Google Official)")
    print("=" * 50)
    
    print("Official GEMMA3 Parameters:")
    print("  - gemma3:27b (27 billion parameters)")
    print("  - gemma3:12b (12 billion parameters)")  
    print("  - gemma3:4b  (4 billion parameters)")
    print("  - gemma3:1b  (1 billion parameters)")
    print()
    
    # GEMMA3 설치
    installed = install_real_gemma3()
    
    if installed:
        print(f"\\nSUCCESS: Installed {len(installed)} GEMMA3 models!")
        for model in installed:
            print(f"  - {model}")
        
        # 성능 테스트
        if test_gemma3_performance(installed):
            print("\\nGEMMA3 is working perfectly!")
        
        print("\\nNext Steps:")
        print("1. Update Solomond AI to use GEMMA3")
        print("2. Compare: Qwen3:8b vs GEMMA3:4b vs Gemma2:9b")
        print("3. Select best model for Korean jewelry consultation")
        
    else:
        print("\\nFAILED: Could not install GEMMA3")
        print("Possible reasons:")
        print("1. GEMMA3 not yet available in Ollama")
        print("2. Network or access issues")
        print("3. Model names may be different")
        
        print("\\nTry manual installation:")
        print("  ollama pull gemma3:4b")
        print("  ollama pull gemma3:1b")

if __name__ == "__main__":
    main()