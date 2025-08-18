#!/usr/bin/env python3
"""
🏆 SOLOMOND AI v3.0 - 고도화된 Ollama 인터페이스
최적화된 5개 모델을 활용한 차세대 AI 통합 시스템

모델 라인업:
- gpt-oss:20b (OpenAI o3-mini 수준) - 차세대 추론 엔진
- gemma3:27b (구글 최강) - 복잡한 분석
- qwen3:8b (최신 추론) - 메인 워크호스  
- qwen2.5:7b (안정성) - 백업 시스템
- gemma3:4b (경량) - 빠른 작업
"""

import requests
import json
import time
from typing import Dict, List, Optional, Generator, Tuple
from datetime import datetime
import logging
from enum import Enum
import asyncio

# 로깅 설정
logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """모델 성능 등급"""
    ULTIMATE = "ultimate"      # GPT-OSS-20B
    PREMIUM = "premium"        # Gemma3-27B  
    STANDARD = "standard"      # Qwen3-8B
    STABLE = "stable"         # Qwen2.5-7B
    FAST = "fast"             # Gemma3-4B

class TaskComplexity(Enum):
    """작업 복잡도"""
    SIMPLE = "simple"         # 간단한 요약, 번역
    MODERATE = "moderate"     # 일반적인 분석
    COMPLEX = "complex"       # 심화 추론, 복잡한 분석
    CRITICAL = "critical"     # 최고 품질 요구

class AdvancedOllamaInterface:
    """차세대 Ollama 통합 인터페이스"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self._get_available_models()
        
        # 🏆 최적화된 모델 라인업
        self.model_lineup = {
            ModelTier.ULTIMATE: "gpt-oss:20b",     # OpenAI 레벨 추론
            ModelTier.PREMIUM: "gemma3:27b",       # 구글 최강
            ModelTier.STANDARD: "qwen3:8b",        # 메인 워크호스
            ModelTier.STABLE: "qwen2.5:7b",        # 안정성 중시
            ModelTier.FAST: "gemma3:4b"            # 빠른 처리
        }
        
        # 🎯 작업별 모델 자동 선택 전략
        self.task_model_mapping = {
            # 컨퍼런스 분석
            "conference_simple": ModelTier.FAST,      # 빠른 요약
            "conference_standard": ModelTier.STANDARD, # 일반 분석
            "conference_deep": ModelTier.ULTIMATE,    # 심화 추론
            
            # 웹 크롤링 & 뉴스
            "news_summary": ModelTier.FAST,           # 빠른 요약
            "news_analysis": ModelTier.STANDARD,      # 트렌드 분석
            "news_insight": ModelTier.PREMIUM,        # 심화 인사이트
            
            # 보석 분석
            "gemstone_basic": ModelTier.STANDARD,     # 기본 식별
            "gemstone_expert": ModelTier.PREMIUM,     # 전문가 분석
            "gemstone_research": ModelTier.ULTIMATE,  # 연구 수준
            
            # 3D CAD
            "cad_simple": ModelTier.FAST,            # 빠른 변환
            "cad_complex": ModelTier.STANDARD,       # 복잡한 처리
            
            # 일반 작업
            "translation": ModelTier.FAST,           # 번역
            "coding": ModelTier.STANDARD,            # 코딩 지원
            "reasoning": ModelTier.ULTIMATE,         # 복잡한 추론
            "creative": ModelTier.PREMIUM            # 창작 작업
        }
        
        # 🧠 고도화된 프롬프트 시스템
        self.advanced_prompts = {
            "ultimate_analysis": """
🏆 당신은 세계 최고 수준의 AI 분석 전문가입니다. OpenAI o3-mini 수준의 추론 능력으로 다음을 분석해주세요.

🎯 **분석 목표**: {task_goal}

📋 **분석 프레임워크**:
1. 🔍 **심층 이해**: 내용의 핵심 의도와 숨겨진 패턴 파악
2. 💡 **통찰 도출**: 표면적 분석을 넘어선 독창적 인사이트
3. 🎯 **실행 방안**: 구체적이고 실행 가능한 액션 플랜
4. 🔮 **미래 예측**: 트렌드 전망 및 영향 분석

📊 **입력 데이터**:
{content}

🚀 **최고 품질 분석 결과**:""",

            "premium_synthesis": """
🔥 당신은 구글 최강 AI 모델 수준의 종합 분석가입니다. 복잡한 정보를 완벽히 통합해주세요.

🎪 **통합 분석 미션**: {task_goal}

⚡ **분석 차원**:
• 📈 **정량적 분석**: 데이터, 수치, 통계적 패턴
• 📝 **정성적 분석**: 의미, 맥락, 감정적 뉘앙스  
• 🌐 **시스템적 분석**: 전체적 구조와 상호관계
• ⚡ **시간적 분석**: 과거-현재-미래 연결고리

📚 **분석 대상**:
{content}

💎 **통합 분석 보고서**:""",

            "standard_processing": """
⚡ 당신은 Qwen3 최신 추론 엔진입니다. 빠르고 정확하게 처리해주세요.

🎯 **작업**: {task_goal}

📋 **처리 방식**:
✅ 핵심 포인트 위주 분석
✅ 명확하고 구조화된 결과  
✅ 실용적 관점 유지

📄 **입력**:
{content}

🚀 **처리 결과**:""",

            "fast_summary": """
🚀 빠른 처리 모드입니다. 효율적으로 요약해주세요.

목적: {task_goal}
내용: {content}

⚡ **간결한 결과**:"""
        }
    
    def _get_available_models(self) -> List[str]:
        """사용 가능한 모델 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"모델 목록 조회 실패: {e}")
            return []
    
    def smart_model_selection(
        self, 
        task_type: str, 
        content_length: int = 0,
        quality_priority: bool = False,
        speed_priority: bool = False
    ) -> Tuple[str, str]:
        """🧠 스마트 모델 선택 시스템"""
        
        # 1단계: 기본 작업 타입별 모델 선택
        base_tier = self.task_model_mapping.get(task_type, ModelTier.STANDARD)
        
        # 2단계: 컨텐츠 길이 고려
        if content_length > 10000:  # 긴 텍스트
            if base_tier == ModelTier.FAST:
                base_tier = ModelTier.STANDARD
        elif content_length > 50000:  # 매우 긴 텍스트
            base_tier = ModelTier.PREMIUM
        
        # 3단계: 사용자 우선순위 고려
        if quality_priority:
            if base_tier in [ModelTier.FAST, ModelTier.STANDARD]:
                base_tier = ModelTier.PREMIUM
            elif base_tier == ModelTier.STABLE:
                base_tier = ModelTier.ULTIMATE
        
        if speed_priority:
            if base_tier in [ModelTier.PREMIUM, ModelTier.ULTIMATE]:
                base_tier = ModelTier.STANDARD
        
        # 4단계: 실제 모델명 반환 + 프롬프트 템플릿 선택
        model_name = self.model_lineup[base_tier]
        
        # 프롬프트 템플릿 선택
        if base_tier == ModelTier.ULTIMATE:
            prompt_template = "ultimate_analysis"
        elif base_tier == ModelTier.PREMIUM:
            prompt_template = "premium_synthesis"
        elif base_tier == ModelTier.STANDARD:
            prompt_template = "standard_processing"
        else:
            prompt_template = "fast_summary"
        
        # 모델 사용 불가시 폴백
        if model_name not in self.available_models:
            fallback_models = [
                self.model_lineup[ModelTier.STANDARD],
                self.model_lineup[ModelTier.STABLE], 
                self.model_lineup[ModelTier.FAST]
            ]
            for fallback in fallback_models:
                if fallback in self.available_models:
                    return fallback, "standard_processing"
            
            return self.available_models[0] if self.available_models else "qwen2.5:7b", "fast_summary"
        
        return model_name, prompt_template
    
    def advanced_generate(
        self,
        task_type: str,
        content: str,
        task_goal: str = "",
        quality_priority: bool = False,
        speed_priority: bool = False,
        stream: bool = False,
        max_tokens: int = 3000,
        temperature: float = 0.7
    ) -> str:
        """🚀 고도화된 생성 함수"""
        
        # 스마트 모델 선택
        model, prompt_template = self.smart_model_selection(
            task_type=task_type,
            content_length=len(content),
            quality_priority=quality_priority,
            speed_priority=speed_priority
        )
        
        # 프롬프트 구성
        prompt = self.advanced_prompts[prompt_template].format(
            task_goal=task_goal or f"{task_type} 작업",
            content=content
        )
        
        # 모델별 최적화된 파라미터
        model_params = self._get_optimized_params(model, task_type)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": model_params["temperature"],
                "top_k": model_params["top_k"],
                "top_p": model_params["top_p"]
            }
        }
        
        try:
            logger.info(f"사용 모델: {model} | 작업: {task_type} | 템플릿: {prompt_template}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=600  # GPT-OSS는 시간이 더 필요
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
            logger.error(f"생성 실패: {e}")
            return f"AI 모델 오류: {str(e)}"
    
    def _get_optimized_params(self, model: str, task_type: str) -> Dict:
        """모델별 최적화 파라미터"""
        base_params = {
            "temperature": 0.7,
            "top_k": 40, 
            "top_p": 0.9
        }
        
        # GPT-OSS 최적화
        if "gpt-oss" in model:
            if "reasoning" in task_type or "deep" in task_type:
                base_params["temperature"] = 0.3  # 추론 작업은 낮은 온도
            else:
                base_params["temperature"] = 0.5
            base_params["top_k"] = 50
        
        # Gemma3 최적화
        elif "gemma3" in model:
            base_params["temperature"] = 0.6
            base_params["top_k"] = 30
        
        # Qwen 최적화  
        elif "qwen" in model:
            base_params["temperature"] = 0.7
            base_params["top_k"] = 40
        
        return base_params
    
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
    
    # 🎯 특화 함수들
    def ultimate_conference_analysis(self, content: str, context: str = "") -> str:
        """🏆 궁극 컨퍼런스 분석 (GPT-OSS 활용)"""
        return self.advanced_generate(
            task_type="conference_deep",
            content=content,
            task_goal=f"주얼리 업계 컨퍼런스 심층 분석{' - ' + context if context else ''}",
            quality_priority=True
        )
    
    def premium_market_insight(self, content: str) -> str:
        """💎 프리미엄 시장 인사이트 (Gemma3-27B 활용)"""
        return self.advanced_generate(
            task_type="news_insight", 
            content=content,
            task_goal="주얼리 시장 동향 및 미래 전망 분석",
            quality_priority=True
        )
    
    def fast_news_summary(self, content: str) -> str:
        """⚡ 빠른 뉴스 요약 (Gemma3-4B 활용)"""
        return self.advanced_generate(
            task_type="news_summary",
            content=content,
            task_goal="주얼리 뉴스 핵심 요약",
            speed_priority=True
        )
    
    def expert_gemstone_analysis(self, image_info: str, analysis_level: str = "standard") -> str:
        """💎 전문가급 보석 분석"""
        task_mapping = {
            "basic": "gemstone_basic",
            "standard": "gemstone_expert", 
            "research": "gemstone_research"
        }
        
        return self.advanced_generate(
            task_type=task_mapping.get(analysis_level, "gemstone_expert"),
            content=image_info,
            task_goal=f"보석 전문 분석 ({analysis_level} 수준)",
            quality_priority=(analysis_level in ["standard", "research"])
        )
    
    def intelligent_translation(self, content: str, target_lang: str = "한국어") -> str:
        """🌐 지능형 번역"""
        return self.advanced_generate(
            task_type="translation",
            content=content,
            task_goal=f"주얼리 전문 용어 정확 번역 ({target_lang})",
            speed_priority=True
        )
    
    def get_system_status(self) -> Dict[str, any]:
        """🔍 시스템 상태 확인"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "server_status": self.health_check(),
            "available_models": self.available_models,
            "model_lineup": {},
            "recommendations": {}
        }
        
        # 모델 라인업 상태
        for tier, model_name in self.model_lineup.items():
            is_available = model_name in self.available_models
            status["model_lineup"][tier.value] = {
                "model": model_name,
                "available": is_available,
                "status": "OK Available" if is_available else "NEED Install"
            }
        
        # 추천 사항
        missing_models = [model for model in self.model_lineup.values() 
                         if model not in self.available_models]
        
        if missing_models:
            status["recommendations"]["install"] = f"Install recommended: {', '.join(missing_models)}"
        else:
            status["recommendations"]["status"] = "All models are perfectly configured!"
        
        return status
    
    def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def benchmark_models(self, test_prompt: str = "주얼리 업계의 미래 트렌드를 설명해주세요.") -> Dict[str, any]:
        """🏁 모델 성능 벤치마크"""
        results = {}
        
        for tier, model_name in self.model_lineup.items():
            if model_name not in self.available_models:
                results[tier.value] = {"status": "모델 없음", "time": 0, "response": ""}
                continue
            
            start_time = time.time()
            try:
                response = self.advanced_generate(
                    task_type="standard_processing",
                    content=test_prompt,
                    task_goal="벤치마크 테스트"
                )
                end_time = time.time()
                
                results[tier.value] = {
                    "model": model_name,
                    "time": round(end_time - start_time, 2),
                    "response_length": len(response),
                    "status": "성공",
                    "preview": response[:100] + "..." if len(response) > 100 else response
                }
            except Exception as e:
                results[tier.value] = {
                    "model": model_name,
                    "time": 0,
                    "status": f"오류: {str(e)}",
                    "response": ""
                }
        
        return results

# 전역 인스턴스 
advanced_ollama = AdvancedOllamaInterface()

# 🚀 편의 함수들
def ultimate_analysis(content: str, context: str = "") -> str:
    """궁극 분석 (GPT-OSS)"""
    return advanced_ollama.ultimate_conference_analysis(content, context)

def premium_insight(content: str) -> str:
    """프리미엄 인사이트 (Gemma3-27B)"""
    return advanced_ollama.premium_market_insight(content)

def quick_summary(content: str) -> str:
    """빠른 요약 (Gemma3-4B)"""
    return advanced_ollama.fast_news_summary(content)

def smart_translate(content: str, target: str = "한국어") -> str:
    """스마트 번역"""
    return advanced_ollama.intelligent_translation(content, target)

def expert_gemstone(image_info: str, level: str = "standard") -> str:
    """전문가 보석 분석"""
    return advanced_ollama.expert_gemstone_analysis(image_info, level)

def get_system_info() -> Dict[str, any]:
    """시스템 정보"""
    return advanced_ollama.get_system_status()

def benchmark_all() -> Dict[str, any]:
    """전체 벤치마크"""
    return advanced_ollama.benchmark_models()

if __name__ == "__main__":
    # 테스트 및 벤치마크
    print("SOLOMOND AI v3.0 - Advanced Ollama Interface")
    print("=" * 60)
    
    # 시스템 상태
    status = get_system_info()
    print(f"서버 상태: {status['server_status']}")
    print(f"사용 가능 모델: {len(status['available_models'])}개")
    
    for tier, info in status['model_lineup'].items():
        print(f"{tier.upper()}: {info['model']} - {info['status']}")
    
    # 간단한 벤치마크 (사용 가능한 경우)
    if status['server_status']:
        print("\nRunning benchmark tests...")
        results = benchmark_all()
        
        for tier, result in results.items():
            if result['status'] == '성공':
                print(f"{tier.upper()}: {result['time']}s | {result['response_length']} chars")
            else:
                print(f"{tier.upper()}: {result['status']}")