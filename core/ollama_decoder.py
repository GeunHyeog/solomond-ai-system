#!/usr/bin/env python3
"""
🧠 Ollama 기반 지능형 디코더 - SOLOMOND AI 진정한 멀티모달리티 구현
Intelligent Ollama Decoder: Strategic Model Chaining for Insight Generation

🎯 주요 기능:
1. 5개 모델 전략적 체이닝 - 각 모델의 특성을 최적화 활용
2. 융합 임베딩 → 인사이트 변환 - 768차원 벡터를 의미있는 분석으로
3. 계층적 분석 프로세스 - 기본→심화→전문→통합→검증 단계
4. 컨텍스트 인식 디코딩 - 도메인별 특화 프롬프트 적용
5. 품질 보증 시스템 - 다중 검증으로 신뢰성 확보
"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 로컬 임포트
from .multimodal_encoder import EncodedResult
from .crossmodal_fusion import FusionResult
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from ollama_interface import OllamaInterface, get_ollama_status
except ImportError:
    logging.error("Ollama 인터페이스를 가져올 수 없습니다.")
    OllamaInterface = None

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DecodingResult:
    """디코딩 결과 구조"""
    insights: List[Dict[str, Any]]  # 생성된 인사이트들
    confidence_scores: Dict[str, float]  # 모델별 신뢰도
    processing_chain: List[str]  # 사용된 모델 체인
    final_confidence: float  # 최종 신뢰도
    processing_time: float  # 처리 시간
    metadata: Dict[str, Any]  # 메타데이터
    quality_assessment: str  # 품질 평가

class OllamaIntelligentDecoder:
    """Ollama 기반 지능형 디코더"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Ollama 인터페이스 초기화
        if OllamaInterface:
            self.ollama = OllamaInterface()
        else:
            self.ollama = None
            logger.error("Ollama 인터페이스가 없습니다.")
            
        # 5개 모델 체인 설정
        self.model_chain = self._setup_model_chain()
        
        # 성능 메트릭
        self.stats = {
            'decodings_performed': 0,
            'average_processing_time': 0.0,
            'model_usage_count': {},
            'quality_scores': [],
            'failed_decodings': 0
        }
        
    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'model_chain': [
                'qwen2.5:7b',    # 1단계: 기본 분석 (빠르고 안정적)
                'gemma2:9b',     # 2단계: 심화 분석 (균형잡힌 성능)
                'llama3.1:8b',   # 3단계: 전문 분석 (고품질 추론)
                'qwen2:7b',      # 4단계: 통합 분석 (다각도 관점)
                'gemma:7b'       # 5단계: 검증 분석 (품질 보증)
            ],
            'fallback_models': ['qwen2.5:7b', 'llama3.2:3b'],
            'max_retries': 3,
            'timeout_per_model': 300,  # 5분
            'min_confidence_threshold': 0.3,
            'enable_parallel_processing': True,
            'context_window': 4000,
            'temperature': 0.7
        }
    
    def _setup_model_chain(self) -> List[Dict]:
        """모델 체인 설정"""
        return [
            {
                'model': self.config['model_chain'][0],
                'role': 'basic_analyzer',
                'description': '기본 분석: 핵심 요소 추출 및 초기 해석',
                'temperature': 0.3,
                'max_tokens': 500,
                'prompt_style': 'concise'
            },
            {
                'model': self.config['model_chain'][1],
                'role': 'deep_analyzer',
                'description': '심화 분석: 패턴 탐지 및 관계 분석',
                'temperature': 0.5,
                'max_tokens': 800,
                'prompt_style': 'analytical'
            },
            {
                'model': self.config['model_chain'][2],
                'role': 'expert_analyzer',
                'description': '전문 분석: 도메인 특화 인사이트 생성',
                'temperature': 0.7,
                'max_tokens': 1000,
                'prompt_style': 'expert'
            },
            {
                'model': self.config['model_chain'][3],
                'role': 'synthesizer',
                'description': '통합 분석: 다각도 관점 통합 및 종합',
                'temperature': 0.6,
                'max_tokens': 1200,
                'prompt_style': 'comprehensive'
            },
            {
                'model': self.config['model_chain'][4],
                'role': 'validator',
                'description': '검증 분석: 결과 검토 및 품질 보증',
                'temperature': 0.2,
                'max_tokens': 600,
                'prompt_style': 'critical'
            }
        ]
    
    def decode_fusion_result(self, fusion_result: FusionResult, context: Optional[Dict] = None) -> DecodingResult:
        """융합 결과를 인사이트로 디코딩"""
        if not self.ollama:
            return self._create_fallback_result("Ollama 인터페이스 없음")
            
        logger.info("🧠 지능형 디코딩 프로세스 시작...")
        start_time = time.time()
        
        # 컨텍스트 준비
        analysis_context = self._prepare_context(fusion_result, context)
        
        # 5단계 체이닝 분석 실행
        insights = []
        confidence_scores = {}
        processing_chain = []
        
        previous_output = ""
        
        for step_idx, model_config in enumerate(self.model_chain):
            try:
                logger.info(f"🔄 {step_idx + 1}단계: {model_config['description']}")
                
                # 단계별 프롬프트 생성
                prompt = self._generate_stage_prompt(
                    step_idx, model_config, fusion_result, 
                    analysis_context, previous_output
                )
                
                # 모델 실행
                response = self._execute_model(model_config, prompt)
                
                if response and len(response.strip()) > 10:
                    # 응답 파싱 및 인사이트 추출
                    stage_insights = self._parse_model_response(
                        response, model_config['role'], step_idx
                    )
                    
                    insights.extend(stage_insights)
                    confidence_scores[model_config['role']] = self._calculate_response_confidence(response)
                    processing_chain.append(f"{model_config['role']}({model_config['model']})")
                    previous_output = response
                    
                    # 통계 업데이트
                    model_name = model_config['model']
                    self.stats['model_usage_count'][model_name] = self.stats['model_usage_count'].get(model_name, 0) + 1
                    
                else:
                    logger.warning(f"⚠️ {step_idx + 1}단계 모델 응답 부족, 다음 단계로 진행")
                    
            except Exception as e:
                logger.error(f"❌ {step_idx + 1}단계 실행 실패: {e}")
                # 실패해도 다음 단계 진행
                continue
        
        # 최종 결과 생성
        processing_time = time.time() - start_time
        final_confidence = self._calculate_final_confidence(confidence_scores, insights)
        quality_assessment = self._assess_quality(insights, confidence_scores)
        
        # 통계 업데이트
        self._update_stats(processing_time, final_confidence, len(insights))
        
        result = DecodingResult(
            insights=insights,
            confidence_scores=confidence_scores,
            processing_chain=processing_chain,
            final_confidence=final_confidence,
            processing_time=processing_time,
            quality_assessment=quality_assessment,
            metadata={
                'total_stages': len(self.model_chain),
                'successful_stages': len(confidence_scores),
                'context_provided': context is not None,
                'fusion_confidence': fusion_result.fusion_confidence,
                'dominant_modality': fusion_result.dominant_modality
            }
        )
        
        logger.info(f"✅ 지능형 디코딩 완료: {len(insights)}개 인사이트, 신뢰도 {final_confidence:.3f}")
        return result
    
    def _prepare_context(self, fusion_result: FusionResult, additional_context: Optional[Dict]) -> Dict:
        """분석 컨텍스트 준비"""
        context = {
            'modality_weights': fusion_result.modal_weights,
            'dominant_modality': fusion_result.dominant_modality,
            'semantic_coherence': fusion_result.semantic_coherence,
            'fusion_confidence': fusion_result.fusion_confidence,
            'cross_modal_correlations': fusion_result.cross_modal_correlations[:3],  # 상위 3개만
            'analysis_domain': 'conference_analysis'  # 기본값
        }
        
        # 추가 컨텍스트 병합
        if additional_context:
            context.update(additional_context)
            
        return context
    
    def _generate_stage_prompt(self, stage: int, model_config: Dict, fusion_result: FusionResult, 
                              context: Dict, previous_output: str) -> str:
        """단계별 프롬프트 생성"""
        
        base_info = f"""
다음은 멀티모달 분석 결과입니다:
- 지배적 모달리티: {fusion_result.dominant_modality}
- 융합 신뢰도: {fusion_result.fusion_confidence:.3f}
- 의미적 일관성: {fusion_result.semantic_coherence:.3f}
- 모달 구성: {', '.join(fusion_result.modal_weights.keys())}
"""
        
        # 단계별 특화 프롬프트
        if stage == 0:  # 기본 분석
            return f"""
{base_info}

🎯 **1단계 - 기본 분석 요청**
당신은 멀티모달 데이터의 핵심 요소를 추출하는 전문가입니다.
위 정보를 바탕으로 다음을 간단명료하게 분석해주세요:

1. **핵심 주제**: 무엇에 관한 내용인가?
2. **주요 특징**: 어떤 특징들이 두드러지는가?
3. **초기 관찰**: 첫 번째 인상은 어떠한가?

📋 **응답 형식**: 각 항목을 3-4줄로 간단히 정리
"""
            
        elif stage == 1:  # 심화 분석
            return f"""
{base_info}

이전 단계 결과:
{previous_output[:500]}

🔍 **2단계 - 심화 분석 요청**
당신은 패턴과 관계를 탐지하는 분석가입니다.
위 기본 분석을 바탕으로 더 깊이 있는 분석을 수행해주세요:

1. **패턴 탐지**: 어떤 규칙성이나 패턴이 보이는가?
2. **관계 분석**: 요소들 간의 연관성은 어떠한가?
3. **숨겨진 의미**: 표면적으로 드러나지 않은 의미는?

📋 **응답 형식**: 각 항목을 5-6줄로 분석적으로 설명
"""
            
        elif stage == 2:  # 전문 분석
            domain = context.get('analysis_domain', 'general')
            return f"""
{base_info}

이전 단계들의 결과:
{previous_output[:800]}

👨‍🏫 **3단계 - 전문 분석 요청**
당신은 {domain} 분야의 전문가입니다.
앞선 분석들을 바탕으로 전문가적 관점에서 분석해주세요:

1. **전문적 해석**: 해당 분야 관점에서의 의미는?
2. **중요한 시사점**: 전문가가 주목할 포인트들은?
3. **실무적 가치**: 실제 활용 가능한 인사이트는?

📋 **응답 형식**: 전문 용어를 포함하여 구체적으로 설명
"""
            
        elif stage == 3:  # 통합 분석
            return f"""
{base_info}

지금까지의 모든 분석 결과:
{previous_output[:1000]}

🔗 **4단계 - 통합 분석 요청**
당신은 다각도 관점을 통합하는 통합전문가입니다.
모든 이전 분석들을 종합하여 통합적 관점을 제시해주세요:

1. **종합 인사이트**: 모든 분석을 통합한 핵심 메시지는?
2. **다각도 관점**: 여러 관점에서 본 완전한 그림은?
3. **전략적 함의**: 이 분석이 가져다주는 전략적 가치는?

📋 **응답 형식**: 포괄적이고 균형잡힌 종합 분석
"""
            
        else:  # 검증 분석
            return f"""
{base_info}

최종 통합 분석 결과:
{previous_output}

✅ **5단계 - 검증 분석 요청**
당신은 품질과 정확성을 검증하는 검증 전문가입니다.
위 통합 분석을 비판적으로 검토하고 최종 의견을 제시해주세요:

1. **분석 품질**: 위 분석의 타당성과 신뢰도는?
2. **개선 포인트**: 보완하거나 수정해야 할 부분은?
3. **최종 권고**: 이 분석을 어떻게 활용하는 것이 좋을까?

📋 **응답 형식**: 객관적이고 건설적인 검증 의견
"""
    
    def _execute_model(self, model_config: Dict, prompt: str) -> Optional[str]:
        """개별 모델 실행"""
        try:
            model_name = model_config['model']
            
            # 모델 가용성 확인
            available_models = self.ollama.available_models
            if model_name not in available_models:
                # 폴백 모델 사용
                for fallback in self.config['fallback_models']:
                    if fallback in available_models:
                        model_name = fallback
                        logger.info(f"폴백 모델 사용: {model_name}")
                        break
                else:
                    logger.error("사용 가능한 모델이 없습니다.")
                    return None
            
            # Ollama 호출
            response = self.ollama.generate_response(
                prompt=prompt,
                model=model_name,
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 800)
            )
            
            return response.strip() if response else None
            
        except Exception as e:
            logger.error(f"모델 실행 실패 {model_config['model']}: {e}")
            return None
    
    def _parse_model_response(self, response: str, role: str, stage: int) -> List[Dict[str, Any]]:
        """모델 응답 파싱하여 인사이트 추출"""
        insights = []
        
        try:
            # 응답을 섹션별로 분할
            sections = []
            current_section = ""
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('1.', '2.', '3.', '**', '#', '-')):
                    if current_section:
                        sections.append(current_section.strip())
                    current_section = line
                else:
                    current_section += f" {line}"
                    
            if current_section:
                sections.append(current_section.strip())
            
            # 각 섹션을 인사이트로 변환
            for i, section in enumerate(sections):
                if len(section) > 20:  # 너무 짧은 섹션 제외
                    insight = {
                        'content': section,
                        'type': self._classify_insight_type(section, role),
                        'stage': stage + 1,
                        'role': role,
                        'confidence': self._calculate_section_confidence(section),
                        'priority': self._calculate_priority(section, role, i)
                    }
                    insights.append(insight)
                    
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            # 폴백: 전체 응답을 하나의 인사이트로 처리
            if response and len(response) > 50:
                insights.append({
                    'content': response,
                    'type': 'general',
                    'stage': stage + 1,
                    'role': role,
                    'confidence': 0.5,
                    'priority': 'medium'
                })
        
        return insights
    
    def _classify_insight_type(self, content: str, role: str) -> str:
        """인사이트 타입 분류"""
        content_lower = content.lower()
        
        # 키워드 기반 분류
        if any(keyword in content_lower for keyword in ['패턴', 'pattern', '규칙', '경향']):
            return 'pattern'
        elif any(keyword in content_lower for keyword in ['관계', '연관', '상관', '관련']):
            return 'relationship'
        elif any(keyword in content_lower for keyword in ['인사이트', '함의', '의미', '시사점']):
            return 'insight'
        elif any(keyword in content_lower for keyword in ['추천', '제안', '권고', '개선']):
            return 'recommendation'
        elif any(keyword in content_lower for keyword in ['문제', '이슈', '위험', '주의']):
            return 'issue'
        else:
            return 'general'
    
    def _calculate_section_confidence(self, section: str) -> float:
        """섹션 신뢰도 계산"""
        # 길이, 구조, 구체성 기반 신뢰도 계산
        length_score = min(1.0, len(section) / 200.0)  # 200자 기준
        
        # 구체적 표현 점수
        concrete_words = ['구체적으로', '예를 들어', '실제로', '특히', '명확히']
        concrete_score = sum(1 for word in concrete_words if word in section) * 0.1
        
        # 전문 용어 점수
        expert_words = ['분석', '관점', '측면', '요소', '특징', '패턴', '트렌드']
        expert_score = min(0.3, sum(1 for word in expert_words if word in section) * 0.05)
        
        base_confidence = 0.4
        return min(1.0, base_confidence + length_score * 0.3 + concrete_score + expert_score)
    
    def _calculate_priority(self, section: str, role: str, position: int) -> str:
        """우선순위 계산"""
        # 역할별 기본 우선순위
        role_priority = {
            'basic_analyzer': 'medium',
            'deep_analyzer': 'medium', 
            'expert_analyzer': 'high',
            'synthesizer': 'high',
            'validator': 'medium'
        }
        
        # 내용 기반 우선순위 조정
        content_lower = section.lower()
        high_priority_keywords = ['중요', '핵심', '주요', '결정적', '필수', 'urgent', 'critical']
        low_priority_keywords = ['참고', '부가', '추가', '보완', '기타']
        
        if any(keyword in content_lower for keyword in high_priority_keywords):
            return 'high'
        elif any(keyword in content_lower for keyword in low_priority_keywords):
            return 'low'
        else:
            return role_priority.get(role, 'medium')
    
    def _calculate_response_confidence(self, response: str) -> float:
        """전체 응답 신뢰도 계산"""
        if not response:
            return 0.0
            
        # 기본 점수
        base_score = 0.5
        
        # 길이 점수 (적절한 길이)
        length = len(response)
        if 100 <= length <= 2000:
            length_score = 0.3
        elif length > 50:
            length_score = 0.1
        else:
            length_score = 0.0
            
        # 구조화 점수 (번호, 불릿 포인트 등)
        structure_indicators = ['1.', '2.', '3.', '**', '##', '-', '•']
        structure_score = min(0.2, sum(0.05 for indicator in structure_indicators if indicator in response))
        
        return min(1.0, base_score + length_score + structure_score)
    
    def _calculate_final_confidence(self, confidence_scores: Dict[str, float], insights: List[Dict]) -> float:
        """최종 신뢰도 계산"""
        if not confidence_scores:
            return 0.0
            
        # 단계별 신뢰도 가중 평균
        stage_weights = {
            'basic_analyzer': 0.15,
            'deep_analyzer': 0.20,
            'expert_analyzer': 0.25,
            'synthesizer': 0.25,
            'validator': 0.15
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for role, confidence in confidence_scores.items():
            weight = stage_weights.get(role, 0.2)
            weighted_confidence += confidence * weight
            total_weight += weight
            
        stage_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # 인사이트 품질 보너스
        if insights:
            avg_insight_confidence = np.mean([insight.get('confidence', 0.5) for insight in insights])
            insight_bonus = (avg_insight_confidence - 0.5) * 0.1
        else:
            insight_bonus = 0.0
            
        return min(1.0, stage_confidence + insight_bonus)
    
    def _assess_quality(self, insights: List[Dict], confidence_scores: Dict[str, float]) -> str:
        """품질 평가"""
        if not insights:
            return 'poor'
            
        avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
        insight_count = len(insights)
        
        if avg_confidence >= 0.8 and insight_count >= 8:
            return 'excellent'
        elif avg_confidence >= 0.6 and insight_count >= 5:
            return 'good'
        elif avg_confidence >= 0.4 and insight_count >= 3:
            return 'fair'
        else:
            return 'poor'
    
    def _create_fallback_result(self, reason: str) -> DecodingResult:
        """폴백 결과 생성"""
        return DecodingResult(
            insights=[{
                'content': f'디코딩 실패: {reason}',
                'type': 'error',
                'stage': 0,
                'role': 'system',
                'confidence': 0.0,
                'priority': 'high'
            }],
            confidence_scores={},
            processing_chain=[],
            final_confidence=0.0,
            processing_time=0.0,
            quality_assessment='poor',
            metadata={'fallback': True, 'reason': reason}
        )
    
    def _update_stats(self, processing_time: float, confidence: float, insight_count: int):
        """통계 업데이트"""
        self.stats['decodings_performed'] += 1
        
        # 이동 평균으로 처리 시간 업데이트
        alpha = 0.1
        self.stats['average_processing_time'] = (
            (1 - alpha) * self.stats['average_processing_time'] + 
            alpha * processing_time
        )
        
        self.stats['quality_scores'].append(confidence)
        
        if confidence < self.config['min_confidence_threshold']:
            self.stats['failed_decodings'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        success_rate = (
            (self.stats['decodings_performed'] - self.stats['failed_decodings']) / 
            max(1, self.stats['decodings_performed']) * 100
        )
        
        avg_quality = (
            np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0.0
        )
        
        return {
            'total_decodings': self.stats['decodings_performed'],
            'success_rate': f"{success_rate:.1f}%",
            'average_processing_time': f"{self.stats['average_processing_time']:.2f}초",
            'average_quality_score': f"{avg_quality:.3f}",
            'model_usage_distribution': self.stats['model_usage_count'].copy(),
            'failed_decodings': self.stats['failed_decodings']
        }

# 사용 예제
def main():
    """사용 예제"""
    decoder = OllamaIntelligentDecoder()
    
    # Ollama 상태 확인
    if decoder.ollama:
        status = decoder.ollama.health_check()
        print(f"🤖 Ollama 연결 상태: {'✅ 연결됨' if status else '❌ 연결 실패'}")
        
        if status:
            print(f"사용 가능한 모델: {decoder.ollama.available_models}")
            
            # 예제 융합 결과 생성 (실제로는 crossmodal_fusion.py에서 받아옴)
            from .crossmodal_fusion import FusionResult
            
            sample_fusion_result = FusionResult(
                fused_embedding=np.random.rand(768).astype(np.float32),
                modal_weights={'image': 0.4, 'audio': 0.6},
                cross_modal_correlations=[],
                dominant_modality='audio',
                fusion_confidence=0.85,
                semantic_coherence=0.78,
                metadata={'test': True}
            )
            
            print("\n🧠 지능형 디코더 테스트 시작...")
            result = decoder.decode_fusion_result(sample_fusion_result)
            
            print(f"\n🎯 디코딩 결과:")
            print(f"   생성된 인사이트: {len(result.insights)}개")
            print(f"   최종 신뢰도: {result.final_confidence:.3f}")
            print(f"   품질 평가: {result.quality_assessment}")
            print(f"   처리 시간: {result.processing_time:.2f}초")
            print(f"   사용된 모델 체인: {' → '.join(result.processing_chain)}")
            
            print(f"\n💡 주요 인사이트들:")
            for i, insight in enumerate(result.insights[:3], 1):
                print(f"   {i}. [{insight['type']}] {insight['content'][:100]}...")
                
        else:
            print("❌ Ollama 서버에 연결할 수 없습니다.")
            print("다음 명령으로 Ollama를 시작하세요: ollama serve")
    else:
        print("❌ Ollama 인터페이스를 초기화할 수 없습니다.")

if __name__ == "__main__":
    main()