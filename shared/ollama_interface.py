#!/usr/bin/env python3
"""
🤖 공통 Ollama AI 인터페이스
모든 솔로몬드 AI 모듈에서 공통으로 사용하는 Ollama 통합 시스템

주요 기능:
- 모델별 최적화된 프롬프트 관리
- 주얼리 업계 특화 컨텍스트
- 다국어 지원 (한국어/영어)
- 스트리밍 응답 지원
- 에러 처리 및 폴백 시스템
"""

import requests
import json
import time
from typing import Dict, List, Optional, Generator
from datetime import datetime
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class OllamaInterface:
    """Ollama AI 모델 통합 인터페이스"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self._get_available_models()
        
        # 모듈별 추천 모델 설정 (속도 우선 최적화)
        self.module_models = {
            "conference_analysis": "qwen2.5:7b",      # 빠른 회의 분석
            "web_crawler": "qwen2.5:7b",             # 뉴스 요약 및 번역
            "gemstone_analysis": "qwen2.5:7b",       # 보석 전문 분석
            "cad_conversion": "llama3.2:3b",         # 빠른 이미지 처리
            "general": "qwen2.5:7b"                  # 일반 용도
        }
        
        # 주얼리 업계 특화 프롬프트 템플릿
        self.jewelry_prompts = {
            "news_summary": """
당신은 주얼리 업계 전문 애널리스트입니다. 다음 뉴스를 빠르게 요약해주세요.

요약 포맷:
- 📰 핵심 내용: (2-3문장)
- 💎 업계 영향: (상/중/하)
- 🔑 키워드: (보석명, 브랜드, 트렌드)
- ⭐ 중요도: (상/중/하)

기사:
{content}

**요약:**""",
            
            "conference_analysis": """
당신은 주얼리 업계 멀티모달 분석 전문가입니다. 이미지, 음성, 텍스트를 통합적으로 분석하여 깊이 있는 인사이트를 제공합니다.

🎯 **멀티모달 통합 분석 프레임워크**:

1. **시그널 추출 및 노이즈 필터링**:
   - 핵심 시그널: 반복 언급, 강조 표현, 시각적 강조점
   - 노이즈 제거: 일회성 언급, 부수적 내용, 기술적 문제
   - 크로스모달 검증: 여러 모달에서 확인되는 내용 우선순위

2. **상황적 컨텍스트 이해**:
   - 발화자별 핵심 메시지 구분
   - 시간 흐름에 따른 주제 변화 추적
   - 시각적 자료와 음성 내용의 일치도 분석

3. **업계 전문성 적용**:
   - 주얼리 시장 트렌드 컨텍스트
   - 기술 혁신 및 지속가능성 이슈
   - 소비자 행동 변화 패턴

4. **실행 가능한 인사이트 생성**:
   - 즉시 실행 가능한 액션 아이템
   - 중기 전략적 고려사항
   - 장기 업계 변화 전망

**분석 대상 내용**:
{content}

**통합 분석 결과** (각 섹션 3-4문장으로 간결하게):

🔍 **핵심 시그널 분석**:
- 

💡 **상황적 인사이트**:
- 

🎯 **업계 함의**:
- 

🚀 **실행 방안**:
- 

⚠️ **주의 사항**:
- 

📈 **미래 전망**:
- """,
            
            "gemstone_identification": """
당신은 보석학 전문가입니다. 다음 보석 이미지 정보를 바탕으로 산지를 분석해주세요.

분석 기준:
- 색상 특성
- 내포물 패턴
- 광학적 특성
- 지질학적 배경

이미지 정보:
{content}

산지 분석:""",
            
            "translation": """
다음 텍스트를 {target_language}로 번역해주세요. 주얼리 업계 전문 용어는 정확히 번역하고, 자연스러운 표현을 사용해주세요.

원문:
{content}

번역:"""
        }
    
    def _get_available_models(self) -> List[str]:
        """사용 가능한 Ollama 모델 목록 조회"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            return []
    
    def select_model(self, module_type: str = "general") -> str:
        """모듈 타입에 따른 최적 모델 선택"""
        recommended = self.module_models.get(module_type, "qwen2.5:7b")
        
        # 추천 모델이 사용 가능한지 확인
        if recommended in self.available_models:
            return recommended
        
        # 폴백 순서
        fallback_models = ["qwen2.5:7b", "llama3.2:3b", "gemma2:2b"]
        for model in fallback_models:
            if model in self.available_models:
                return model
        
        # 마지막 폴백: 첫 번째 사용 가능한 모델
        return self.available_models[0] if self.available_models else "llama3.2:3b"
    
    def generate_response(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        stream: bool = False,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """텍스트 생성 (기본 모드)"""
        
        if model is None:
            model = self.select_model()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_k": 40,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=300  # 5분으로 연장
            )
            
            if stream:
                return self._handle_stream_response(response)
            else:
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"API 오류: {response.status_code}")
                    return f"오류 발생: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            return f"AI 모델 오류: {str(e)}"
    
    def _handle_stream_response(self, response) -> Generator[str, None, None]:
        """스트리밍 응답 처리"""
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
        except Exception as e:
            logger.error(f"스트리밍 처리 오류: {e}")
            yield f"스트리밍 오류: {str(e)}"
    
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """채팅 완성 (대화형)"""
        
        if model is None:
            model = self.select_model()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=300  # 5분으로 연장
            )
            
            if stream:
                return self._handle_stream_response(response)
            else:
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
                else:
                    return f"채팅 오류: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"채팅 완성 실패: {e}")
            return f"채팅 오류: {str(e)}"
    
    def summarize_jewelry_news(self, content: str, model: Optional[str] = None) -> str:
        """주얼리 뉴스 요약 특화 함수"""
        if model is None:
            model = self.select_model("web_crawler")
        
        prompt = self.jewelry_prompts["news_summary"].format(content=content)
        return self.generate_response(prompt, model=model, temperature=0.3)
    
    def analyze_conference(self, content: str, model: Optional[str] = None) -> str:
        """컨퍼런스 분석 특화 함수 (청크 분할 처리)"""
        if model is None:
            model = self.select_model("conference_analysis")
        
        # 긴 텍스트는 청크로 분할
        if len(content) > 8000:  # 8000자 이상이면 분할
            return self._analyze_long_content(content, model)
        
        prompt = self.jewelry_prompts["conference_analysis"].format(content=content)
        return self.generate_response(prompt, model=model, temperature=0.5)
    
    def _analyze_long_content(self, content: str, model: str) -> str:
        """긴 텍스트 청크 분할 분석"""
        try:
            # 텍스트를 6000자 단위로 분할
            chunk_size = 6000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            
            chunk_analyses = []
            
            for i, chunk in enumerate(chunks):
                prompt = f"""
다음은 컨퍼런스 내용의 {i+1}/{len(chunks)} 부분입니다. 이 부분을 간단히 요약해주세요:

{chunk}

간단 요약:"""
                
                chunk_result = self.generate_response(
                    prompt, 
                    model=model, 
                    temperature=0.3,
                    max_tokens=500  # 청크당 짧게
                )
                chunk_analyses.append(chunk_result)
            
            # 전체 종합 분석
            combined_summary = "\n\n".join(chunk_analyses)
            final_prompt = self.jewelry_prompts["conference_analysis"].format(
                content=f"다음은 컨퍼런스 각 부분별 요약입니다:\n{combined_summary}"
            )
            
            return self.generate_response(final_prompt, model=model, temperature=0.5)
            
        except Exception as e:
            logger.error(f"긴 컨텐츠 분석 실패: {e}")
            return f"분석 중 오류 발생: {str(e)}"
    
    def identify_gemstone(self, image_info: str, model: Optional[str] = None) -> str:
        """보석 식별 특화 함수"""
        if model is None:
            model = self.select_model("gemstone_analysis")
        
        prompt = self.jewelry_prompts["gemstone_identification"].format(content=image_info)
        return self.generate_response(prompt, model=model, temperature=0.2)
    
    def translate_text(
        self, 
        content: str, 
        target_language: str = "한국어",
        model: Optional[str] = None
    ) -> str:
        """번역 특화 함수"""
        if model is None:
            model = self.select_model("general")
        
        prompt = self.jewelry_prompts["translation"].format(
            content=content, 
            target_language=target_language
        )
        return self.generate_response(prompt, model=model, temperature=0.1)
    
    def get_model_info(self) -> Dict[str, any]:
        """현재 설정 및 모델 정보 반환"""
        return {
            "available_models": self.available_models,
            "module_models": self.module_models,
            "base_url": self.base_url,
            "status": "연결됨" if self.available_models else "연결 실패"
        }
    
    def health_check(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_server_status(self) -> Dict[str, any]:
        """Ollama 서버 상세 상태 정보"""
        try:
            # 기본 연결 확인
            health = self.health_check()
            
            if not health:
                return {
                    "status": "disconnected",
                    "message": "Ollama 서버에 연결할 수 없습니다.",
                    "suggestion": "터미널에서 'ollama serve' 명령을 실행하세요."
                }
            
            # 모델 목록 확인
            models = self._get_available_models()
            
            return {
                "status": "connected",
                "available_models": models,
                "recommended_model": self.select_model("conference_analysis"),
                "message": f"정상 연결됨 ({len(models)}개 모델 사용 가능)"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"상태 확인 오류: {str(e)}",
                "suggestion": "Ollama 서버를 재시작해보세요."
            }

# 전역 인스턴스 생성
global_ollama = OllamaInterface()

# 편의 함수들
def quick_summary(text: str, model: str = None) -> str:
    """빠른 요약"""
    return global_ollama.summarize_jewelry_news(text, model)

def quick_translate(text: str, target: str = "한국어") -> str:
    """빠른 번역"""
    return global_ollama.translate_text(text, target)

def quick_analysis(text: str, model: str = None) -> str:
    """빠른 분석"""
    return global_ollama.analyze_conference(text, model)

def get_ollama_status() -> Dict[str, any]:
    """Ollama 상태 정보"""
    return global_ollama.get_server_status()

def get_ollama_models() -> Dict[str, any]:
    """Ollama 모델 정보"""
    return global_ollama.get_model_info()

if __name__ == "__main__":
    # 테스트 코드
    ollama = OllamaInterface()
    
    print("Ollama Interface Test")
    print("=" * 50)
    
    # 상태 확인
    print(f"Connection Status: {ollama.health_check()}")
    print(f"Available Models: {ollama.available_models}")
    
    # 간단한 테스트
    if ollama.health_check():
        test_prompt = "주얼리 업계의 미래 트렌드에 대해 간단히 설명해주세요."
        print(f"\nTest Prompt: {test_prompt}")
        print("=" * 50)
        
        response = ollama.generate_response(test_prompt)
        print(f"Response: {response[:200]}...")
    else:
        print("Cannot connect to Ollama server.")