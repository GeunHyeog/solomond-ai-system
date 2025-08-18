#!/usr/bin/env python3
"""
Enhanced Analysis Engine - 정확성 개선 시스템
사용자 요구사항: "분석의 결과가 정확하지 않은 것 같아. 개선할 수 있는 방법은?"
"""

import os
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAnalysisEngine:
    """정확성 개선된 분석 엔진"""
    
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.available_models = []
        self.analysis_history = {}
        self.quality_scores = {}
        
        # 품질 검증을 위한 다중 모델 시스템
        self.quality_models = [
            'qwen2.5:7b',   # 최고 품질
            'gemma2:9b',    # 교차 검증용
            'qwen2.5:3b',   # 빠른 검증용
        ]
        
        # 도메인별 전문 프롬프트 템플릿
        self.domain_prompts = {
            'jewelry': {
                'keywords': ['보석', '주얼리', '다이아몬드', '금', '은', '원석', '반지', '목걸이'],
                'template': """
주얼리/보석 분야 전문 분석:
- 제품명, 재질, 크기, 품질 등급
- 시장 가격 동향 및 투자 가치
- 제조 공정 및 기술적 특성
- 고객 선호도 및 트렌드 분석
"""
            },
            'conference': {
                'keywords': ['회의', '컨퍼런스', '미팅', '발표', '토론', '의견', '결정'],
                'template': """
컨퍼런스/회의 전문 분석:
- 핵심 의제 및 결정사항
- 참석자별 의견 및 입장
- 합의점과 이견 분석
- 후속 액션 플랜 도출
"""
            },
            'business': {
                'keywords': ['비즈니스', '사업', '매출', '수익', '전략', '마케팅', '고객'],
                'template': """
비즈니스 전문 분석:
- 사업 기회 및 위험 요소
- 시장 동향 및 경쟁 분석
- 수익성 및 성장 가능성
- 전략적 추천사항
"""
            }
        }
    
    def check_ollama_connection(self) -> bool:
        """Ollama 서버 연결 및 모델 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                logger.info(f"사용 가능한 모델: {len(self.available_models)}개")
                return True
        except Exception as e:
            logger.error(f"Ollama 연결 실패: {e}")
        return False
    
    def detect_domain(self, content: str) -> str:
        """내용을 분석하여 도메인 자동 감지"""
        content_lower = content.lower()
        
        domain_scores = {}
        for domain, config in self.domain_prompts.items():
            score = sum(1 for keyword in config['keywords'] if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
            logger.info(f"감지된 도메인: {best_domain} (점수: {domain_scores[best_domain]})")
            return best_domain
        
        return 'general'
    
    def create_enhanced_prompt(self, content: str, context: str = "", domain: str = None) -> str:
        """정확성 향상된 프롬프트 생성"""
        
        if not domain:
            domain = self.detect_domain(content)
        
        # 기본 분석 구조
        base_prompt = f"""
다음 내용을 정확하고 체계적으로 분석해주세요:

=== 분석 대상 ===
{content}

=== 컨텍스트 정보 ===
{context}

=== 분석 요구사항 ===
"""
        
        # 도메인별 전문 프롬프트 추가
        if domain in self.domain_prompts:
            base_prompt += self.domain_prompts[domain]['template']
        
        # 정확성 검증 지침 추가
        base_prompt += """

=== 정확성 검증 지침 ===
1. 핵심 메시지 추출:
   - 명시적으로 언급된 내용
   - 암시적/함축적 의미
   - 문맥상 중요한 포인트

2. 증거 기반 분석:
   - 구체적인 근거 제시
   - 추측과 사실 구분
   - 불확실한 내용은 명시

3. 구조화된 결과:
   - 핵심 요약 (3-5줄)
   - 상세 분석 (단계별)
   - 실행 가능한 제안

4. 품질 체크:
   - 논리적 일관성 확인
   - 중요 정보 누락 방지
   - 명확하고 구체적인 표현

상세하고 정확하게 한국어로 분석해주세요.
"""
        return base_prompt.strip()
    
    def get_best_models(self, num_models: int = 2) -> List[str]:
        """분석 품질을 위한 최적 모델 선택"""
        available_quality = [m for m in self.quality_models if m in self.available_models]
        
        if not available_quality:
            # fallback to available models
            available_quality = self.available_models[:num_models]
        
        return available_quality[:num_models]
    
    def analyze_with_model(self, model: str, prompt: str, timeout: int = 120) -> Dict:
        """단일 모델로 분석 실행"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,      # 정확성 우선
                    "num_predict": 1500,     # 충분한 응답 길이
                    "top_p": 0.85,          # 품질 있는 선택
                    "repeat_penalty": 1.1,   # 반복 방지
                    "top_k": 40             # 적절한 다양성
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                analysis_result = {
                    "success": True,
                    "model": model,
                    "response": result.get("response", ""),
                    "processing_time": processing_time,
                    "confidence_score": self.calculate_confidence(result.get("response", "")),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Model {model}: {processing_time:.2f}s, confidence: {analysis_result['confidence_score']:.2f}")
                return analysis_result
            else:
                return {
                    "success": False,
                    "model": model,
                    "error": f"HTTP {response.status_code}",
                    "processing_time": processing_time
                }
                
        except Exception as e:
            logger.error(f"Model {model} analysis failed: {e}")
            return {
                "success": False,
                "model": model,
                "error": str(e),
                "processing_time": 0
            }
    
    def calculate_confidence(self, response: str) -> float:
        """응답의 신뢰도 점수 계산"""
        if not response:
            return 0.0
        
        confidence_indicators = [
            ("구체적", "명확한", "정확한", "확실한"),  # 확실성 표현
            ("분석", "검토", "평가", "조사"),         # 분석적 표현  
            ("결론", "결과", "요약", "핵심"),         # 결론적 표현
            ("근거", "이유", "원인", "배경"),         # 근거 제시
            ("제안", "추천", "방향", "전략")          # 실행 가능성
        ]
        
        score = 0.0
        response_lower = response.lower()
        
        for indicator_group in confidence_indicators:
            if any(indicator in response_lower for indicator in indicator_group):
                score += 0.2
        
        # 길이 기반 가중치 (너무 짧거나 길지 않은 적정 길이)
        length = len(response)
        if 200 <= length <= 2000:
            score += 0.1
        elif length > 100:
            score += 0.05
        
        return min(score, 1.0)
    
    def cross_validate_results(self, results: List[Dict]) -> Dict:
        """다중 모델 결과 교차 검증"""
        successful_results = [r for r in results if r.get('success')]
        
        if not successful_results:
            return {
                "success": False,
                "error": "모든 모델 분석 실패",
                "validation": "FAILED"
            }
        
        # 신뢰도 점수 기반 최적 결과 선택
        best_result = max(successful_results, key=lambda r: r.get('confidence_score', 0))
        
        # 교차 검증 점수 계산
        avg_confidence = sum(r.get('confidence_score', 0) for r in successful_results) / len(successful_results)
        consistency_score = self.calculate_consistency(successful_results)
        
        validation_result = {
            "success": True,
            "primary_result": best_result,
            "validation": {
                "models_used": len(successful_results),
                "avg_confidence": avg_confidence,
                "consistency_score": consistency_score,
                "quality_level": self.get_quality_level(avg_confidence, consistency_score)
            },
            "all_results": successful_results
        }
        
        return validation_result
    
    def calculate_consistency(self, results: List[Dict]) -> float:
        """결과 간 일관성 점수 계산"""
        if len(results) < 2:
            return 1.0
        
        responses = [r.get('response', '') for r in results if r.get('response')]
        if not responses:
            return 0.0
        
        # 키워드 기반 일관성 체크
        all_keywords = set()
        response_keywords = []
        
        for response in responses:
            keywords = set(re.findall(r'\b\w{4,}\b', response.lower()))
            response_keywords.append(keywords)
            all_keywords.update(keywords)
        
        if not all_keywords:
            return 0.5
        
        # 교집합 비율 계산
        common_keywords = set.intersection(*response_keywords) if response_keywords else set()
        consistency = len(common_keywords) / len(all_keywords) if all_keywords else 0
        
        return min(consistency * 2, 1.0)  # 일관성 점수 정규화
    
    def get_quality_level(self, confidence: float, consistency: float) -> str:
        """품질 수준 판정"""
        overall_score = (confidence + consistency) / 2
        
        if overall_score >= 0.8:
            return "HIGH"
        elif overall_score >= 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def enhanced_analysis(self, content: str, context: str = "", domain: str = None, num_models: int = 2) -> Dict:
        """정확성 향상된 종합 분석"""
        
        if not self.check_ollama_connection():
            return {
                "success": False,
                "error": "Ollama 서버 연결 실패",
                "suggestion": "Ollama 서버가 실행 중인지 확인해주세요"
            }
        
        # 향상된 프롬프트 생성
        enhanced_prompt = self.create_enhanced_prompt(content, context, domain)
        
        # 최적 모델들 선택
        selected_models = self.get_best_models(num_models)
        
        if not selected_models:
            return {
                "success": False,
                "error": "사용 가능한 분석 모델이 없습니다"
            }
        
        logger.info(f"분석 시작: {len(selected_models)}개 모델 사용")
        
        # 병렬 분석 실행
        results = []
        with ThreadPoolExecutor(max_workers=min(len(selected_models), 3)) as executor:
            future_to_model = {
                executor.submit(self.analyze_with_model, model, enhanced_prompt): model 
                for model in selected_models
            }
            
            for future in as_completed(future_to_model, timeout=180):
                result = future.result()
                results.append(result)
        
        # 교차 검증 및 최종 결과 생성
        final_result = self.cross_validate_results(results)
        
        # 분석 이력 저장
        analysis_id = hashlib.md5(f"{content[:100]}{time.time()}".encode()).hexdigest()[:8]
        self.analysis_history[analysis_id] = {
            "content_preview": content[:200],
            "timestamp": datetime.now().isoformat(),
            "models_used": selected_models,
            "result": final_result
        }
        
        if final_result.get('success'):
            logger.info(f"분석 완료 (ID: {analysis_id}): 품질 수준 {final_result['validation']['quality_level']}")
        
        return final_result
    
    def get_analysis_report(self) -> Dict:
        """분석 성능 보고서 생성"""
        if not self.analysis_history:
            return {"message": "분석 이력이 없습니다"}
        
        total_analyses = len(self.analysis_history)
        successful = sum(1 for a in self.analysis_history.values() if a['result'].get('success'))
        
        quality_levels = [a['result']['validation']['quality_level'] 
                         for a in self.analysis_history.values() 
                         if a['result'].get('success')]
        
        high_quality = sum(1 for q in quality_levels if q == 'HIGH')
        
        return {
            "총_분석수": total_analyses,
            "성공률": f"{successful/total_analyses*100:.1f}%",
            "고품질_비율": f"{high_quality/len(quality_levels)*100:.1f}%" if quality_levels else "0%",
            "평균_신뢰도": f"{sum(a['result']['validation']['avg_confidence'] for a in self.analysis_history.values() if a['result'].get('success'))/successful:.2f}" if successful else "0",
            "사용_모델수": len(set(model for a in self.analysis_history.values() for model in a['models_used'])),
            "최근_분석": list(self.analysis_history.keys())[-5:] if self.analysis_history else []
        }

# 전역 인스턴스
enhanced_engine = EnhancedAnalysisEngine()

if __name__ == "__main__":
    # 테스트 실행
    test_content = """
    오늘 회의에서 새로운 주얼리 제품 라인에 대해 논의했습니다. 
    다이아몬드 반지 시리즈의 품질 기준과 가격 정책을 검토하고, 
    고객 피드백을 바탕으로 개선 방향을 설정했습니다.
    """
    
    result = enhanced_engine.enhanced_analysis(
        content=test_content,
        context="주얼리 회사 제품 기획 회의",
        num_models=2
    )
    
    print("=== 정확성 개선 분석 결과 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n=== 성능 보고서 ===")
    report = enhanced_engine.get_analysis_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))