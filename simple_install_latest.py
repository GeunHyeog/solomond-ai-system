#!/usr/bin/env python3
import subprocess
import os

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

def install_models():
    print("Installing Latest Models: GEMMA3 + Llama 3.2")
    print("=" * 50)
    
    ollama_path = find_ollama()
    
    models = [
        "gemma3:9b",
        "gemma3:7b", 
        "llama3.2:8b",
        "llama3.2:3b"
    ]
    
    installed = []
    
    for model in models:
        print(f"\\nTrying: {model}")
        try:
            result = subprocess.run([ollama_path, "pull", model],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"SUCCESS: {model}")
                installed.append(model)
            else:
                print(f"FAILED: {model}")
                if "not found" in result.stderr.lower():
                    print("  -> Model not available yet")
        except Exception as e:
            print(f"ERROR: {str(e)}")
    
    return installed

def check_status():
    print("\\n=== Current Models ===")
    try:
        import requests
        r = requests.get('http://localhost:11434/api/tags')
        if r.status_code == 200:
            models = r.json()['models']
            for model in models:
                name = model['name']
                size_gb = model['size'] / (1024**3)
                print(f"  {name} ({size_gb:.1f}GB)")
    except:
        print("  Error checking models")

def main():
    print("Latest Models Installation")
    print("=" * 30)
    
    check_status()
    installed = install_models()
    
    if installed:
        print(f"\\nSUCCESS: Installed {len(installed)} new models")
        for model in installed:
            print(f"  - {model}")
    else:
        print("\\nNo new models installed")
        print("Current best: Qwen3:8b (already installed)")
    
    check_status()

if __name__ == "__main__":
    main()