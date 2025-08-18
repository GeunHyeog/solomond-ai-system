#!/usr/bin/env python3
"""
무료 AI 대안 설정 가이드
비용 없이 고급 AI 기능 활용하기
"""

import os
import sys
import json
from pathlib import Path

def setup_ollama_integration():
    """Ollama 로컬 LLM 설정"""
    print("Ollama 로컬 LLM 설정")
    print("="*50)
    
    print("설치 방법:")
    print("1. https://ollama.ai 에서 Ollama 다운로드")
    print("2. 설치 후 터미널에서 실행:")
    print("   ollama pull llama3.2:3b        # 3B 모델 (빠름)")
    print("   ollama pull llama3.2:8b        # 8B 모델 (고품질)")
    print("   ollama pull qwen2.5:7b         # 다국어 지원")
    print("   ollama pull mistral:7b         # 코딩 특화")
    
    print("\n장점:")
    print("- 완전 무료")
    print("- 개인정보 보호 (로컬 실행)")
    print("- 인터넷 연결 불필요")
    print("- 무제한 사용")
    
    print("\n요구사항:")
    print("- RAM: 최소 8GB (3B 모델), 권장 16GB (8B 모델)")
    print("- 저장공간: 모델당 2-5GB")

def setup_huggingface_free():
    """Hugging Face 무료 API 설정"""
    print("\nHugging Face 무료 API 설정")
    print("="*50)
    
    print("📋 설정 방법:")
    print("1. https://huggingface.co 가입")
    print("2. Settings → Access Tokens에서 토큰 생성")
    print("3. 솔로몬드 AI 설정에서 토큰 입력")
    
    print("\n🆓 무료 사용 가능 모델:")
    print("- microsoft/DialoGPT-large (대화)")
    print("- facebook/bart-large-cnn (요약)")
    print("- google/flan-t5-large (범용)")
    print("- microsoft/Phi-3-mini (경량)")
    
    print("\n💡 무료 제한:")
    print("- 월 30,000 요청")
    print("- 응답 속도 제한")
    print("- 상용 사용 제한")

def setup_google_colab():
    """Google Colab 무료 GPU 활용"""
    print("\n☁️ Google Colab 무료 GPU 설정")
    print("="*50)
    
    print("📋 활용 방법:")
    print("1. Google Colab에서 솔로몬드 AI 실행")
    print("2. 무료 T4 GPU 사용 (15GB VRAM)")
    print("3. 대형 모델 실행 가능")
    
    print("\n💡 장점:")
    print("- 완전 무료 GPU 사용")
    print("- 강력한 하드웨어")
    print("- 클라우드 저장")
    
    print("\n⚠️ 제한사항:")
    print("- 세션당 최대 12시간")
    print("- 일일 사용 제한")
    print("- 네트워크 의존성")

def setup_local_models_config():
    """로컬 모델 최적화 설정"""
    print("\n🔧 현재 로컬 모델 최적화")
    print("="*50)
    
    config = {
        "ai_models": {
            "whisper": {
                "enabled": True,
                "model_size": "base",  # tiny, base, small, medium, large
                "language": "ko",
                "cost": 0.0
            },
            "easyocr": {
                "enabled": True,
                "languages": ["ko", "en"],
                "gpu": False,
                "cost": 0.0
            },
            "transformers": {
                "enabled": True,
                "model": "facebook/bart-large-cnn",
                "task": "summarization",
                "cost": 0.0
            }
        },
        "free_alternatives": {
            "ollama": {
                "enabled": False,
                "models": ["llama3.2:3b", "qwen2.5:7b"],
                "api_url": "http://localhost:11434"
            },
            "huggingface": {
                "enabled": False,
                "token": "",
                "models": [
                    "microsoft/DialoGPT-large",
                    "facebook/bart-large-cnn",
                    "google/flan-t5-large"
                ]
            }
        }
    }
    
    config_path = Path("free_ai_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📁 설정 파일 생성: {config_path}")
    print("📋 현재 활성화된 무료 AI 기능:")
    print("- ✅ Whisper STT (음성→텍스트)")
    print("- ✅ EasyOCR (이미지→텍스트)")
    print("- ✅ Transformers (텍스트 요약)")
    print("- ⏸️ Ollama LLM (설치 후 활성화 가능)")
    print("- ⏸️ Hugging Face API (토큰 설정 후 활성화)")

def show_cost_comparison():
    """비용 비교표"""
    print("\n💰 AI 서비스 비용 비교")
    print("="*60)
    
    print("📊 유료 서비스 (1M 토큰당):")
    print("- OpenAI GPT-4o:        $5.00")
    print("- OpenAI GPT-3.5:       $0.50")
    print("- Anthropic Claude:     $3.00")
    print("- Google Gemini:        $3.50")
    
    print("\n🆓 무료 대안:")
    print("- 솔로몬드 로컬 모델:    $0.00 (무제한)")
    print("- Ollama LLM:          $0.00 (무제한)")
    print("- Hugging Face:        $0.00 (월 30K 요청)")
    print("- Google Colab:        $0.00 (일일 제한)")
    
    print("\n📈 예상 월 사용 비용 (100회 분석 기준):")
    print("- OpenAI GPT-4o:        ~$15-30")
    print("- 솔로몬드 로컬:         $0")
    print("- Ollama:              $0")

def recommend_free_setup():
    """무료 설정 추천"""
    print("\n🎯 추천 무료 AI 설정")
    print("="*50)
    
    print("🥇 최고 추천 (현재 상태 유지):")
    print("- ✅ 솔로몬드 로컬 모델 (Whisper + EasyOCR + Transformers)")
    print("- 장점: 완전 무료, 빠른 속도, 개인정보 보호")
    print("- 현재 상태: 이미 완벽 구현됨")
    
    print("\n🥈 고급 기능 원할 시:")
    print("- 📥 Ollama 설치 → 로컬 LLM 추가")
    print("- 장점: GPT급 성능, 완전 무료")
    print("- 요구사항: RAM 8GB+")
    
    print("\n🥉 클라우드 활용:")
    print("- ☁️ Google Colab + 솔로몬드 AI")
    print("- 장점: 무료 GPU, 강력한 성능")
    print("- 단점: 세션 제한")
    
    print("\n💡 권장 순서:")
    print("1. 현재 로컬 시스템으로 테스트 (무료)")
    print("2. 필요시 Ollama 설치 (무료)")
    print("3. 고급 기능 필요시만 유료 API 고려")

def create_free_integration_guide():
    """무료 통합 가이드 생성"""
    guide_content = """
# 🆓 솔로몬드 AI 무료 활용 가이드

## 현재 무료 기능 (이미 구현됨)
- **Whisper STT**: 음성을 텍스트로 변환 (무료, 오프라인)
- **EasyOCR**: 이미지에서 텍스트 추출 (무료, 다국어)
- **Transformers**: 텍스트 요약 및 분석 (무료)

## 추가 무료 옵션

### 1. Ollama 로컬 LLM (강력 추천)
```bash
# 설치
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드
ollama pull llama3.2:3b    # 빠른 모델
ollama pull qwen2.5:7b     # 한국어 특화
```

### 2. Hugging Face 무료 API
```python
# 토큰 설정 후 사용
from transformers import pipeline
summarizer = pipeline("summarization", 
                     model="facebook/bart-large-cnn")
```

### 3. Google Colab 무료 GPU
- 솔로몬드 AI를 Colab에서 실행
- 무료 T4 GPU 사용 가능
- 대형 모델 실행 가능

## 비용 없는 고급 기능 활용법
1. **번역**: Google Translate API 무료 할당량
2. **음성 합성**: gTTS (Google Text-to-Speech) 무료
3. **이미지 생성**: Stable Diffusion 로컬 실행
4. **코드 분석**: 로컬 CodeT5 모델

## 성능 최적화 팁
- CPU 모드에서도 충분한 성능
- 배치 처리로 효율성 증대
- 모델 캐싱으로 속도 향상
- 스트리밍으로 메모리 절약
"""
    
    guide_path = Path("FREE_AI_GUIDE.md")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"\n📖 무료 활용 가이드 생성: {guide_path}")

def main():
    print("무료 솔로몬드 AI - 무료 AI 서비스 설정 가이드")
    print("="*60)
    
    show_cost_comparison()
    recommend_free_setup()
    setup_local_models_config()
    setup_ollama_integration()
    setup_huggingface_free()
    setup_google_colab()
    create_free_integration_guide()
    
    print("\n🎉 결론:")
    print("현재 솔로몬드 AI는 이미 완전 무료로 사용 가능합니다!")
    print("추가 기능이 필요한 경우에만 위 옵션들을 고려하세요.")
    
    print(f"\n📁 생성된 파일:")
    print(f"- free_ai_config.json (설정)")
    print(f"- FREE_AI_GUIDE.md (가이드)")

if __name__ == "__main__":
    main()