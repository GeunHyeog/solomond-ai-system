#!/usr/bin/env python3
"""
Ollama 통합 엔진
로컬 LLM을 활용한 한국어 특화 분석 및 개인정보 보호
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from utils.logger import get_logger

class OllamaIntegrationEngine:
    """Ollama 로컬 LLM 통합 엔진"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.logger = self._setup_logging()
        
        # 사용할 모델들 - 2025년 최신 3세대 모델로 업데이트
        self.models = {
            "korean_chat": "qwen3:8b",          # 🥇 Qwen3 - 한국어 최강 3세대
            "emotion_analysis": "gemma3:4b",     # 🥈 GEMMA3 4B - 빠른 감정 분석
            "structured_output": "gemma3:27b",   # 🥉 GEMMA3 27B - 최고 성능 구조화
            "high_quality": "gemma3:27b",        # 🏆 최고 품질 분석용
            "fast_response": "gemma3:4b",        # ⚡ 빠른 응답용
            "backup_model": "gemma2:9b"          # 🔄 백업용
        }
        
        # 한국어 특화 프롬프트 템플릿
        self.korean_prompts = {
            "emotion_analysis": """
GEMMA3 고급 감정 분석: 다음 한국어 주얼리 상담 대화를 정밀 분석해주세요:

대화 내용:
{conversation}

GEMMA3 분석 요청:
1. 감정 상태 (긍정/부정/중립/관심/망설임/흥미/우려) - 정확도 95%+
2. 구매 의도 수준 (1-10점) + 근거
3. 핵심 키워드 추출 (우선순위별)
4. 고객 유형 분류 (신중형/적극형/가격민감형/품질중시형)
5. 최적 대응 전략

JSON 형태로 구조화하여 답변해주세요.
""",
            "conversation_summary": """
Qwen3 한국어 전문 분석: 다음 주얼리 상담 대화를 한국 문화에 맞게 요약해주세요:

{conversation}

Qwen3 요약 요구사항:
- 고객 핵심 요구사항 (명확한 니즈)
- 논의된 제품/서비스 (구체적 언급사항)
- 가격대 및 예산 상황 (민감도 포함)
- 의사결정 단계 (고민/확신/보류 등)
- 문화적 맥락 고려 (한국 주얼리 선호도)
- 제안할 다음 단계

한국어로 자연스럽고 전문적으로 요약해주세요.
""",
            "recommendation_generation": """
다음 상황에서 고객에게 최적의 제안을 생성해주세요:

고객 정보: {customer_info}
상담 내용: {conversation_summary}
시장 정보: {market_data}

다음 형태로 제안해주세요:
1. 맞춤 제품 추천 (3개)
2. 가격 혜택 제안
3. 추가 서비스 옵션
4. 결정을 위한 추가 정보

친근하고 전문적인 톤으로 작성해주세요.
"""
        }
        
        self.logger.info("🦙 Ollama 통합 엔진 초기화 완료")
    
    def _setup_logging(self):
        """로깅 설정"""
        return get_logger(f'{__name__}.OllamaIntegrationEngine')
    
    async def check_ollama_availability(self) -> Dict[str, Any]:
        """Ollama 서버 및 모델 가용성 확인"""
        
        status = {
            "server_available": False,
            "available_models": [],
            "missing_models": [],
            "recommendations": []
        }
        
        try:
            # Ollama 서버 연결 확인
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        status["server_available"] = True
                        data = await response.json()
                        installed_models = [model["name"] for model in data.get("models", [])]
                        status["available_models"] = installed_models
                        
                        # 필요한 모델 확인
                        for purpose, model_name in self.models.items():
                            if model_name not in installed_models:
                                status["missing_models"].append({
                                    "purpose": purpose,
                                    "model": model_name,
                                    "install_command": f"ollama pull {model_name}"
                                })
                        
                        # 설치 권장사항
                        if status["missing_models"]:
                            status["recommendations"] = [
                                "다음 명령어로 필요한 모델들을 설치하세요:",
                                *[m["install_command"] for m in status["missing_models"]]
                            ]
                        else:
                            status["recommendations"] = ["모든 필요한 모델이 설치되어 있습니다!"]
                    
        except Exception as e:
            self.logger.error(f"❌ Ollama 서버 연결 실패: {str(e)}")
            status["recommendations"] = [
                "Ollama를 설치하고 실행하세요:",
                "1. https://ollama.ai/ 에서 다운로드",
                "2. 설치 후 'ollama serve' 실행",
                "3. 필요한 모델들 설치"
            ]
        
        return status
    
    async def analyze_korean_conversation(self, conversation: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """한국어 대화 분석 - Ollama 활용"""
        
        try:
            # 감정 분석
            emotion_result = await self._call_ollama_model(
                "emotion_analysis",
                self.korean_prompts["emotion_analysis"].format(conversation=conversation)
            )
            
            # 대화 요약
            summary_result = await self._call_ollama_model(
                "korean_chat", 
                self.korean_prompts["conversation_summary"].format(conversation=conversation)
            )
            
            # 구조화된 결과 생성
            structured_result = await self._generate_structured_analysis(
                emotion_result, summary_result, context
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "emotion_analysis": emotion_result,
                "conversation_summary": summary_result,
                "structured_insights": structured_result,
                "processing_method": "ollama_korean_optimized"
            }
            
        except Exception as e:
            self.logger.error(f"❌ 한국어 대화 분석 실패: {str(e)}")
            return self._create_fallback_analysis(conversation)
    
    async def generate_personalized_recommendations(self, 
                                                  customer_info: Dict,
                                                  conversation_summary: str,
                                                  market_data: Dict = None) -> Dict[str, Any]:
        """개인화된 추천 생성"""
        
        try:
            prompt = self.korean_prompts["recommendation_generation"].format(
                customer_info=json.dumps(customer_info, ensure_ascii=False),
                conversation_summary=conversation_summary,
                market_data=json.dumps(market_data or {}, ensure_ascii=False)
            )
            
            recommendation = await self._call_ollama_model("korean_chat", prompt)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "personalized_recommendations": recommendation,
                "confidence": "high",
                "method": "ollama_korean_llm"
            }
            
        except Exception as e:
            self.logger.error(f"❌ 개인화 추천 생성 실패: {str(e)}")
            return {"error": str(e), "fallback": True}
    
    async def _call_ollama_model(self, model_purpose: str, prompt: str) -> str:
        """Ollama 모델 호출"""
        
        model_name = self.models.get(model_purpose, "llama3.1:8b")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "응답을 받을 수 없습니다.")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API 오류: {response.status} - {error_text}")
                        
        except Exception as e:
            self.logger.error(f"❌ Ollama 모델 호출 실패 ({model_name}): {str(e)}")
            raise
    
    async def _generate_structured_analysis(self, emotion_result: str, summary_result: str, context: Dict) -> Dict[str, Any]:
        """구조화된 분석 결과 생성"""
        
        try:
            # 감정 분석 결과 파싱 시도
            emotion_data = self._parse_emotion_analysis(emotion_result)
            
            # 요약 결과 구조화
            summary_data = self._parse_conversation_summary(summary_result)
            
            # 컨텍스트와 결합
            structured_insights = {
                "customer_emotions": emotion_data,
                "conversation_insights": summary_data,
                "context_integration": self._integrate_context(context),
                "ai_confidence": self._calculate_confidence(emotion_data, summary_data)
            }
            
            return structured_insights
            
        except Exception as e:
            self.logger.error(f"❌ 구조화된 분석 생성 실패: {str(e)}")
            return {"error": str(e), "raw_emotion": emotion_result, "raw_summary": summary_result}
    
    def _parse_emotion_analysis(self, emotion_text: str) -> Dict[str, Any]:
        """감정 분석 결과 파싱"""
        
        # JSON 형태로 응답이 온 경우 파싱 시도
        try:
            return json.loads(emotion_text)
        except:
            # 텍스트 형태인 경우 키워드 기반 파싱
            emotions = {
                "positive_indicators": [],
                "negative_indicators": [],
                "purchase_intent": 5  # 기본값
            }
            
            positive_words = ["좋다", "예쁘다", "마음에", "원한다", "관심"]
            negative_words = ["비싸다", "고민", "망설", "어렵다"]
            
            for word in positive_words:
                if word in emotion_text:
                    emotions["positive_indicators"].append(word)
            
            for word in negative_words:
                if word in emotion_text:
                    emotions["negative_indicators"].append(word)
            
            # 구매 의도 점수 계산
            positive_score = len(emotions["positive_indicators"])
            negative_score = len(emotions["negative_indicators"])
            emotions["purchase_intent"] = min(10, max(1, 5 + positive_score - negative_score))
            
            return emotions
    
    def _parse_conversation_summary(self, summary_text: str) -> Dict[str, Any]:
        """대화 요약 결과 파싱"""
        
        return {
            "summary": summary_text,
            "key_points": self._extract_key_points(summary_text),
            "action_items": self._extract_action_items(summary_text)
        }
    
    def _extract_key_points(self, text: str) -> List[str]:
        """핵심 포인트 추출"""
        
        key_indicators = ["요구사항", "제품", "가격", "결정", "관심"]
        key_points = []
        
        sentences = text.split('.')
        for sentence in sentences:
            if any(indicator in sentence for indicator in key_indicators):
                key_points.append(sentence.strip())
        
        return key_points[:5]  # 상위 5개
    
    def _extract_action_items(self, text: str) -> List[str]:
        """액션 아이템 추출"""
        
        action_indicators = ["다음", "필요", "준비", "제안", "추천"]
        action_items = []
        
        sentences = text.split('.')
        for sentence in sentences:
            if any(indicator in sentence for indicator in action_indicators):
                action_items.append(sentence.strip())
        
        return action_items[:3]  # 상위 3개
    
    def _integrate_context(self, context: Dict) -> Dict[str, Any]:
        """컨텍스트 통합"""
        
        if not context:
            return {"status": "no_context"}
        
        return {
            "participants": context.get('participants', ''),
            "situation": context.get('situation', ''),
            "keywords": context.get('keywords', ''),
            "integration_score": 0.8  # 기본 통합 점수
        }
    
    def _calculate_confidence(self, emotion_data: Dict, summary_data: Dict) -> float:
        """AI 신뢰도 계산"""
        
        confidence_factors = [
            len(emotion_data.get('positive_indicators', [])) > 0,
            len(summary_data.get('key_points', [])) > 2,
            emotion_data.get('purchase_intent', 0) > 0
        ]
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _create_fallback_analysis(self, conversation: str) -> Dict[str, Any]:
        """Ollama 실패 시 기본 분석"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "ollama_unavailable",
            "basic_analysis": {
                "conversation_length": len(conversation),
                "has_korean": any(ord(char) > 127 for char in conversation),
                "fallback_summary": "Ollama 연결 실패로 기본 분석 제공"
            },
            "recommendations": [
                "Ollama 서버 상태 확인 필요",
                "필요한 모델 설치 확인 필요"
            ]
        }

# 사용 예시 및 테스트
async def test_ollama_integration():
    """Ollama 통합 테스트"""
    
    engine = OllamaIntegrationEngine()
    
    # 1. 가용성 확인
    print("=== Ollama 가용성 확인 ===")
    status = await engine.check_ollama_availability()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    
    # 2. 한국어 대화 분석 (가용할 경우)
    if status["server_available"]:
        test_conversation = """
        고객: 결혼반지 좀 보고 싶어요. 예산은 200만원 정도 생각하고 있어요.
        상담사: 네, 좋은 선택이시네요. 어떤 스타일을 선호하시나요?
        고객: 너무 화려하지 않고 심플한 게 좋을 것 같아요. 다이아몬드는 꼭 필요한가요?
        상담사: 다이아몬드 없이도 아름다운 디자인들이 많이 있습니다. 보여드릴까요?
        """
        
        print("\n=== 한국어 대화 분석 ===")
        result = await engine.analyze_korean_conversation(test_conversation)
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test_ollama_integration())