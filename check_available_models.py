#!/usr/bin/env python3
"""
Ollama에서 사용 가능한 모든 모델 확인
"""

import subprocess
import os
import requests
import json

def find_ollama():
    """Ollama 실행파일 찾기"""
    username = os.getenv("USERNAME", "PC_58410")
    paths = [
        f"C:\\Users\\{username}\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
        "C:\\Program Files\\Ollama\\ollama.exe"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return "ollama"

def get_installed_models():
    """현재 설치된 모델들"""
    print("=== Currently Installed Models ===")
    
    try:
        r = requests.get('http://localhost:11434/api/tags')
        if r.status_code == 200:
            models = r.json()['models']
            
            if models:
                print(f"Found {len(models)} installed models:")
                for model in models:
                    name = model['name']
                    size_gb = model['size'] / (1024**3)
                    modified = model['modified_at'][:10]  # Date only
                    
                    # Model type indicator
                    if 'qwen3' in name.lower():
                        status = "3rd Gen (Latest!)"
                    elif 'gemma3' in name.lower():
                        status = "3rd Gen (Latest!)"
                    elif 'qwen2.5' in name.lower() or 'gemma2' in name.lower():
                        status = "2nd Gen (Current)"
                    elif 'llama3' in name.lower():
                        status = "Meta Latest"
                    else:
                        status = "Standard"
                    
                    print(f"  {name:<20} {size_gb:>6.1f}GB  {modified}  [{status}]")
            else:
                print("No models installed")
        else:
            print("Error: Cannot connect to Ollama API")
    except Exception as e:
        print(f"Error: {str(e)}")

def search_popular_models():
    """인기 있는 사용 가능한 모델들 확인"""
    print("\n=== Popular Available Models (Can be downloaded) ===")
    
    ollama_path = find_ollama()
    
    # 카테고리별 인기 모델들
    model_categories = {
        "Latest Generation (2025)": [
            "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
            "llama3.2:8b", "llama3.2:3b", "llama3.2:1b",
            "gemma2:9b", "gemma2:27b", "gemma2:2b"
        ],
        "Specialized Models": [
            "codellama:7b", "codellama:13b", "codellama:34b",
            "mistral:7b", "mistral-nemo:12b",
            "phi3:3.8b", "phi3:14b",
            "deepseek-coder:6.7b", "deepseek-coder:33b"
        ],
        "Multilingual (Korean Support)": [
            "qwen2.5:7b",     # Best for Korean
            "llama3.1:8b",    # Good Korean support
            "gemma2:9b",      # Decent Korean
            "solar:10.7b"     # Korean specialized
        ],
        "Lightweight Models": [
            "gemma2:2b", "llama3.2:1b", "phi3:3.8b",
            "tinyllama:1.1b", "qwen2.5:0.5b"
        ],
        "Large Models (High Performance)": [
            "qwen2.5:32b", "llama3.1:70b", "gemma2:27b",
            "mixtral:8x7b", "wizardlm2:8x22b"
        ]
    }
    
    for category, models in model_categories.items():
        print(f"\n{category}:")
        for model in models:
            # Quick availability check
            availability = check_model_availability(model, ollama_path)
            status = "✓ Available" if availability else "? Unknown"
            print(f"  {model:<20} [{status}]")

def check_model_availability(model_name, ollama_path):
    """모델 사용 가능 여부 확인 (빠른 체크)"""
    try:
        # Simple existence check without downloading
        result = subprocess.run([ollama_path, "show", model_name], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def suggest_best_models():
    """용도별 최고 모델 추천"""
    print("\n=== Recommended Best Models by Use Case ===")
    
    recommendations = {
        "Korean Jewelry Consultation": [
            "qwen2.5:7b    - Best Korean understanding",
            "qwen2.5:14b   - More powerful Korean model",
            "solar:10.7b   - Korean specialized (if available)"
        ],
        "Code Generation": [
            "codellama:7b  - Meta's code specialist",
            "deepseek-coder:6.7b - Strong coding model",
            "qwen2.5:7b    - Good general + code"
        ],
        "Fast Response (Speed Priority)": [
            "gemma2:2b     - Very fast, decent quality",
            "llama3.2:1b   - Ultra-fast lightweight",
            "phi3:3.8b     - Balanced speed/quality"
        ],
        "High Quality (Accuracy Priority)": [
            "qwen2.5:14b   - Top tier reasoning",
            "llama3.1:8b   - Excellent general model", 
            "gemma2:9b     - Strong Google model"
        ],
        "Multilingual Support": [
            "qwen2.5:7b    - Excellent Asian languages",
            "llama3.1:8b   - Good global languages",
            "gemma2:9b     - Decent multilingual"
        ]
    }
    
    for use_case, models in recommendations.items():
        print(f"\n{use_case}:")
        for model in models:
            print(f"  {model}")

def installation_guide():
    """설치 가이드"""
    print("\n=== How to Install Models ===")
    print("Basic installation:")
    print("  ollama pull model:tag")
    print()
    print("Examples:")
    print("  ollama pull qwen2.5:7b       # Download Qwen 2.5 7B")
    print("  ollama pull gemma2:9b        # Download Gemma 2 9B")
    print("  ollama pull llama3.2:8b      # Download Llama 3.2 8B")
    print()
    print("Check installed models:")
    print("  ollama list")
    print()
    print("Remove a model:")
    print("  ollama rm model:tag")
    print()
    print("Update to latest version:")
    print("  ollama pull model:latest")

def main():
    """메인 실행"""
    print("Ollama Available Models Checker")
    print("=" * 50)
    
    # 1. 현재 설치된 모델들
    get_installed_models()
    
    # 2. 사용 가능한 인기 모델들
    search_popular_models()
    
    # 3. 용도별 추천
    suggest_best_models()
    
    # 4. 설치 가이드
    installation_guide()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- Qwen2.5 series: Best for Korean/Asian languages")
    print("- Llama3.x series: Excellent general purpose")
    print("- Gemma2 series: Fast and efficient from Google")
    print("- CodeLlama: Best for programming tasks")
    print("- Use 'ollama pull model:tag' to install any model")

if __name__ == "__main__":
    main()