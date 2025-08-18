#!/usr/bin/env python3
"""
Ollama 모델 자동 업데이트 시스템
- 새 버전 모델 자동 감지
- 기존 모델 업데이트
- 성능 비교 후 자동 교체
"""

import subprocess
import os
import time
import requests
from datetime import datetime

class OllamaModelUpdater:
    def __init__(self):
        self.ollama_path = self.find_ollama()
        
    def find_ollama(self):
        username = os.getenv("USERNAME", "PC_58410")
        paths = [
            f"C:\\Users\\{username}\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
            "C:\\Program Files\\Ollama\\ollama.exe"
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        return "ollama"
    
    def get_current_models(self):
        """현재 설치된 모델 목록"""
        try:
            r = requests.get('http://localhost:11434/api/tags')
            if r.status_code == 200:
                models = r.json()['models']
                return [(m['name'], m['modified_at'], m['size']) for m in models]
        except:
            pass
        return []
    
    def check_for_updates(self):
        """업데이트 가능한 모델 확인"""
        print("=== Checking for Model Updates ===")
        
        current_models = self.get_current_models()
        update_candidates = []
        
        # 현재 모델들의 최신 버전 확인
        for model_name, modified_at, size in current_models:
            base_model = model_name.split(':')[0]
            
            # 최신 버전이 있는지 확인할 모델들
            if base_model in ['qwen3', 'gemma3', 'llama3.2', 'gemma2', 'qwen2.5']:
                print(f"Checking updates for: {model_name}")
                
                # 새로운 태그 확인 (예: qwen3:latest, gemma3:latest)
                latest_tag = f"{base_model}:latest"
                if self.try_pull_model(latest_tag, dry_run=True):
                    update_candidates.append((model_name, latest_tag))
        
        return update_candidates
    
    def try_pull_model(self, model_name, dry_run=False):
        """모델 다운로드 시도 (dry_run=True면 확인만)"""
        try:
            if dry_run:
                # 모델 존재 여부만 확인
                result = subprocess.run([self.ollama_path, "show", model_name],
                                      capture_output=True, text=True, timeout=30)
                return result.returncode == 0
            else:
                # 실제 다운로드
                result = subprocess.run([self.ollama_path, "pull", model_name],
                                      capture_output=True, text=True, timeout=600)
                return result.returncode == 0
        except:
            return False
    
    def update_model(self, old_model, new_model):
        """모델 업데이트 실행"""
        print(f"Updating {old_model} -> {new_model}")
        
        if self.try_pull_model(new_model):
            print(f"SUCCESS: {new_model} downloaded")
            
            # 성능 테스트
            if self.performance_test(new_model) > self.performance_test(old_model):
                print(f"UPGRADE: {new_model} performs better")
                return True
            else:
                print(f"KEEP: {old_model} still better")
                return False
        else:
            print(f"FAILED: Could not download {new_model}")
            return False
    
    def performance_test(self, model_name):
        """간단한 성능 테스트"""
        try:
            payload = {
                "model": model_name,
                "prompt": "안녕하세요. 한국어로 간단히 답변해주세요.",
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post("http://localhost:11434/api/generate",
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                end_time = time.time()
                result = response.json()
                answer = result.get('response', '')
                
                # 성능 점수 (응답 품질 + 속도)
                quality_score = len(answer) / 100  # 응답 길이 기반
                speed_score = 10 / max(end_time - start_time, 1)  # 속도 기반
                
                return quality_score * 0.7 + speed_score * 0.3
            else:
                return 0
        except:
            return 0
    
    def auto_update_check(self):
        """자동 업데이트 확인 및 실행"""
        print("Ollama Model Auto-Update System")
        print("=" * 40)
        print(f"Timestamp: {datetime.now()}")
        
        # 1. 현재 모델들 표시
        current_models = self.get_current_models()
        print(f"\\nCurrent Models ({len(current_models)}):")
        for name, modified, size in current_models:
            size_gb = size / (1024**3)
            print(f"  {name} ({size_gb:.1f}GB)")
        
        # 2. 업데이트 확인
        updates = self.check_for_updates()
        
        if updates:
            print(f"\\nFound {len(updates)} potential updates:")
            for old, new in updates:
                print(f"  {old} -> {new}")
                
            # 3. 업데이트 실행
            for old_model, new_model in updates:
                if self.update_model(old_model, new_model):
                    print(f"UPDATED: {old_model} -> {new_model}")
                else:
                    print(f"SKIPPED: {old_model} (no improvement)")
        else:
            print("\\nNo updates available")
        
        # 4. 최신 상태 확인
        print("\\n=== Final Model Status ===")
        final_models = self.get_current_models()
        for name, modified, size in final_models:
            size_gb = size / (1024**3)
            print(f"  {name} ({size_gb:.1f}GB)")
        
        print("\\nUpdate check completed!")
    
    def manual_install_latest(self):
        """수동으로 최신 모델들 설치"""
        print("Manual Latest Model Installation")
        print("=" * 40)
        
        # 확실히 존재하는 최신 모델들
        latest_models = [
            "qwen2.5:latest",   # Qwen 최신
            "llama3.1:latest",  # Llama 최신  
            "gemma2:latest",    # Gemma 최신
            "mistral:latest"    # Mistral 최신
        ]
        
        installed = []
        
        for model in latest_models:
            print(f"\\nInstalling: {model}")
            if self.try_pull_model(model):
                print(f"SUCCESS: {model}")
                installed.append(model)
            else:
                print(f"FAILED: {model}")
        
        if installed:
            print(f"\\nInstalled {len(installed)} latest models:")
            for model in installed:
                print(f"  - {model}")
        else:
            print("\\nNo new models installed")

def main():
    """메인 실행"""
    updater = OllamaModelUpdater()
    
    print("Ollama Model Update Options:")
    print("1. Auto-update check")
    print("2. Manual install latest")
    print("3. Current status only")
    
    # 자동으로 업데이트 확인 실행
    updater.auto_update_check()
    
    print("\\n" + "="*50)
    print("TIP: Ollama Model Updates")
    print("- 'ollama pull model:latest' for latest version")
    print("- 'ollama list' to see current models")
    print("- Models auto-update when available")
    print("- Always backup before major updates")

if __name__ == "__main__":
    main()