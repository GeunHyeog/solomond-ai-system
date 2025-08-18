"""
Jewelry Specialized Prompts v2.3 for Solomond AI Platform
주얼리 특화 프롬프트 v2.3 - 솔로몬드 AI 플랫폼

🎯 목표: 99.2% 정확도 달성을 위한 최적화된 주얼리 전문 프롬프트
📅 개발기간: 2025.07.13 - 2025.08.03 (3주)
👨‍💼 프로젝트 리더: 전근혁 (솔로몬드 대표)

Week 2 Day 1-3: 주얼리 전문 프롬프트 템플릿 완전 구현
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
from utils.logger import get_logger

class JewelryCategory(Enum):
    """주얼리 카테고리 v2.3"""
    DIAMOND_4C = "diamond_4c"
    COLORED_GEMSTONE = "colored_gemstone"
    JEWELRY_DESIGN = "jewelry_design"
    BUSINESS_INSIGHT = "business_insight"
    MARKET_ANALYSIS = "market_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    INVESTMENT_ADVISORY = "investment_advisory"
    CERTIFICATION_ANALYSIS = "certification_analysis"
    TREND_ANALYSIS = "trend_analysis"
    PRICE_EVALUATION = "price_evaluation"

class AnalysisLevel(Enum):
    """분석 수준 v2.3"""
    BASIC = "basic"              # 기본 정보 제공
    STANDARD = "standard"        # 표준 분석
    PROFESSIONAL = "professional" # 전문가 수준
    EXPERT = "expert"           # 전문가 깊이 분석
    CERTIFICATION = "certification" # 감정서 수준
    MASTER = "master"           # 마스터 레벨 분석

class AIModelType(Enum):
    """AI 모델 타입별 최적화"""
    GPT4_VISION = "gpt-4-vision"
    CLAUDE_VISION = "claude-vision"
    GEMINI_2_PRO = "gemini-2-pro"
    SOLOMOND_JEWELRY = "solomond-jewelry"

@dataclass
class JewelryPromptTemplate:
    """주얼리 프롬프트 템플릿 v2.3"""
    category: JewelryCategory
    level: AnalysisLevel
    model_type: AIModelType
    prompt_ko: str
    prompt_en: str
    expected_accuracy: float
    priority_score: float
    context_hints: List[str]
    validation_keywords: List[str]
    fallback_prompt: str

@dataclass
class PromptOptimizationResult:
    """프롬프트 최적화 결과"""
    optimized_prompt: str
    confidence_score: float
    model_specific_hints: List[str]
    expected_output_format: str
    quality_criteria: List[str]

class JewelryPromptOptimizerV23:
    """주얼리 특화 프롬프트 최적화기 v2.3"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.performance_history = {}
        self.model_preferences = {}
        self.accuracy_targets = {
            "diamond_4c": 0.995,
            "colored_gemstone": 0.990,
            "business_insight": 0.985,
            "market_analysis": 0.980
        }
        self.logger = get_logger(__name__)
    
    def _load_templates(self) -> Dict[str, Dict[str, JewelryPromptTemplate]]:
        """주얼리 프롬프트 템플릿 로드 v2.3 - 대형 함수 (리팩토링 고려 대상)"""
        
        templates = {}
        
        # 다이아몬드 4C 분석 템플릿
        templates["diamond_4c"] = {
            "gpt-4-vision": JewelryPromptTemplate(
                category=JewelryCategory.DIAMOND_4C,
                level=AnalysisLevel.EXPERT,
                model_type=AIModelType.GPT4_VISION,
                prompt_ko="""당신은 GIA 공인 다이아몬드 감정사입니다. 제시된 다이아몬드를 4C 기준(Carat, Color, Clarity, Cut)으로 전문가 수준 분석을 수행해주세요.

분석 요구사항:
1. 캐럿(Carat): 정확한 중량 측정 및 크기 비례 평가
2. 컬러(Color): D-Z 등급 체계로 정확한 색상 평가
3. 클래리티(Clarity): FL-I3 등급으로 내외부 특성 분석
4. 컷(Cut): Excellent-Poor 등급으로 광학적 성능 평가

GIA 표준 적용:
- 10배율 확대경 기준 클래리티 평가
- 표준 조명 환경에서 컬러 그레이딩
- 비율과 대칭성을 고려한 컷 등급 평가
- 형광성(Fluorescence) 평가 포함

결과 형식:
• 4C 등급 요약 (Grade Summary)
• 세부 분석 (Detailed Analysis)
• 시장 가치 추정 (Market Value)
• 품질 개선 제안 (Quality Enhancement)
• 투자 가치 평가 (Investment Potential)

목표 정확도: 99.5%""",
                
                prompt_en="""You are a GIA certified diamond grader. Perform expert-level 4C analysis (Carat, Color, Clarity, Cut) on the presented diamond.

Analysis Requirements:
1. Carat: Precise weight measurement and size proportion assessment
2. Color: Accurate color grading using D-Z scale
3. Clarity: FL-I3 grading with internal/external characteristics analysis
4. Cut: Excellent-Poor grading with optical performance evaluation

GIA Standards Application:
- 10x magnification clarity assessment
- Standard lighting environment color grading
- Proportion and symmetry-based cut grading
- Fluorescence evaluation included

Output Format:
• Grade Summary
• Detailed Analysis
• Market Value Estimation
• Quality Enhancement Suggestions
• Investment Potential Assessment

Target Accuracy: 99.5%""",
                
                expected_accuracy=0.995,
                priority_score=1.0,
                context_hints=[
                    "GIA 표준 적용 필수",
                    "10배율 확대 기준",
                    "표준 조명 환경",
                    "비율 대칭성 고려"
                ],
                validation_keywords=[
                    "GIA", "4C", "Carat", "Color", "Clarity", "Cut",
                    "등급", "그레이딩", "감정", "평가"
                ],
                fallback_prompt="다이아몬드 4C 기본 분석을 수행하되 GIA 기준을 최대한 적용해주세요."
            ),
            
            "claude-vision": JewelryPromptTemplate(
                category=JewelryCategory.DIAMOND_4C,
                level=AnalysisLevel.EXPERT,
                model_type=AIModelType.CLAUDE_VISION,
                prompt_ko="""<role>세계적으로 인정받는 다이아몬드 감정 전문가</role>

<task>
다이아몬드 4C 종합 분석을 GIA 국제 표준에 따라 수행합니다.
목표 정확도: 99.5%
</task>

<methodology>
1. Carat Weight Analysis:
   - 정밀 중량 측정 (소수점 3자리)
   - 크기 대비 중량 비례성 검토
   - 캐럿당 가격 효율성 분석

2. Color Grading (D-Z Scale):
   - 표준 다이아몬드와 비교 분석
   - 형광성(Fluorescence) 영향 평가
   - 가격에 미치는 컬러 영향도

3. Clarity Assessment (FL-I3):
   - 내부 특성(Inclusions) 세밀 분석
   - 외부 특성(Blemishes) 평가
   - 클래리티가 광학성능에 미치는 영향

4. Cut Quality Evaluation:
   - 비율(Proportions) 정밀 측정
   - 대칭성(Symmetry) 평가
   - 연마 상태(Polish) 검토
   - 광학적 성능 종합 평가
</methodology>

<output_format>
## 🔍 4C 종합 분석 결과

### 📊 등급 요약
- Carat: [정확한 중량]
- Color: [D-Z 등급]
- Clarity: [FL-I3 등급]  
- Cut: [등급 및 점수]

### 📋 세부 분석
[각 항목별 상세 분석]

### 💎 품질 총평
[종합적 품질 평가]

### 💰 시장 가치
[현재 시장가 추정]

### 📈 투자 관점
[투자 가치 및 향후 전망]
</output_format>

<quality_standard>
- GIA 표준 100% 준수
- 논리적 근거 명시
- 정확도 99.5% 달성
- 실무 활용 가능한 구체성
</quality_standard>""",
                
                prompt_en="""<role>Internationally recognized diamond grading expert</role>

<task>
Perform comprehensive diamond 4C analysis according to GIA international standards.
Target Accuracy: 99.5%
</task>

<methodology>
1. Carat Weight Analysis
2. Color Grading (D-Z Scale)
3. Clarity Assessment (FL-I3)
4. Cut Quality Evaluation
</methodology>

<output_format>
Structured grading report with market value assessment
</output_format>""",
                
                expected_accuracy=0.995,
                priority_score=1.0,
                context_hints=[
                    "구조화된 분석 방법론",
                    "단계별 품질 기준",
                    "논리적 근거 제시",
                    "실무 활용성 중시"
                ],
                validation_keywords=[
                    "GIA", "4C", "종합 분석", "등급", "품질", "시장가치"
                ],
                fallback_prompt="다이아몬드를 체계적으로 분석하되 GIA 기준을 적용해주세요."
            )
        }
        
        # 유색보석 감정 템플릿
        templates["colored_gemstone"] = {
            "gpt-4-vision": JewelryPromptTemplate(
                category=JewelryCategory.COLORED_GEMSTONE,
                level=AnalysisLevel.EXPERT,
                model_type=AIModelType.GPT4_VISION,
                prompt_ko="""당신은 SSEF, Gübelin 공인 유색보석 감정사입니다. 제시된 유색보석을 국제 감정 기준으로 종합 분석해주세요.

감정 프로토콜:
1. 보석 식별 (Gem Identification)
   - 광물학적 특성 분석
   - 굴절률, 비중 등 물리적 특성
   - 분광학적 특성 검토

2. 원산지 판정 (Origin Determination)
   - 지질학적 특성 분석
   - 미량원소 패턴 검토
   - 내포물 특성 분석

3. 처리 여부 검증 (Treatment Detection)
   - 가열 처리 (Heat Treatment)
   - 오일 함침 (Oil/Resin Filling)
   - 기타 인공 처리 여부

4. 품질 등급 평가
   - 색상 (Color): 색조, 채도, 명도
   - 투명도 (Clarity): 내포물 평가
   - 컷 (Cut): 비율과 마감 품질
   - 캐럿 (Carat): 중량 및 크기

국제 기준 적용:
- SSEF 감정 기준
- Gübelin 랩 표준
- GIA 유색보석 기준
- AIGS 아시아 기준

결과 제공:
• 보석 식별 결과
• 원산지 의견
• 처리 여부 판정
• 품질 등급 평가
• 시장 가치 추정
• 희귀성 평가

목표 정확도: 99.0%""",
                
                prompt_en="""You are an SSEF, Gübelin certified colored gemstone expert. Perform comprehensive analysis according to international gemological standards.

Identification Protocol:
1. Gem Identification
2. Origin Determination  
3. Treatment Detection
4. Quality Assessment

International Standards:
- SSEF standards
- Gübelin lab protocols
- GIA colored stone criteria
- AIGS Asian standards

Target Accuracy: 99.0%""",
                
                expected_accuracy=0.990,
                priority_score=0.95,
                context_hints=[
                    "국제 감정기관 기준",
                    "원산지 판정 중요",
                    "처리 여부 검증",
                    "희귀성 평가 포함"
                ],
                validation_keywords=[
                    "SSEF", "Gübelin", "유색보석", "원산지", "처리", "품질"
                ],
                fallback_prompt="유색보석의 기본 특성을 분석하고 품질을 평가해주세요."
            )
        }
        
        # 주얼리 디자인 분석 템플릿
        templates["jewelry_design"] = {
            "gpt-4-vision": JewelryPromptTemplate(
                category=JewelryCategory.JEWELRY_DESIGN,
                level=AnalysisLevel.PROFESSIONAL,
                model_type=AIModelType.GPT4_VISION,
                prompt_ko="""당신은 국제적인 주얼리 디자인 전문가입니다. 제시된 주얼리 작품을 디자인 관점에서 종합 분석해주세요.

디자인 분석 요소:
1. 스타일 분석 (Style Analysis)
   - 디자인 시대적 배경
   - 스타일 카테고리 (클래식, 모던, 아방가르드 등)
   - 문화적 영향 요소

2. 구조적 분석 (Structural Analysis)
   - 비례와 균형 (Proportion & Balance)
   - 대칭성과 조화 (Symmetry & Harmony)
   - 시각적 무게감 (Visual Weight)

3. 소재 활용 (Material Usage)
   - 금속 선택의 적절성
   - 보석 세팅 기법
   - 소재 간 조화와 대비

4. 착용성 분석 (Wearability)
   - 실용성과 편안함
   - 다양한 스타일링 가능성
   - 일상/특별한 날 적합성

5. 예술적 가치 (Artistic Value)
   - 창의성과 독창성
   - 기술적 완성도
   - 예술적 표현력

6. 상업적 분석 (Commercial Analysis)
   - 시장 수용성
   - 타겟 고객층
   - 가격 경쟁력

결과 제공:
• 디자인 개요 및 특징
• 스타일 분류 및 배경
• 구조적 강점과 개선점
• 상업적 가치 평가
• 컬렉션 내 포지셔닝
• 향후 트렌드 적합성

목표 정확도: 98.5%""",
                
                prompt_en="""You are an international jewelry design expert. Perform comprehensive design analysis of the presented jewelry piece.

Design Analysis Elements:
1. Style Analysis
2. Structural Analysis
3. Material Usage
4. Wearability Assessment
5. Artistic Value
6. Commercial Viability

Target Accuracy: 98.5%""",
                
                expected_accuracy=0.985,
                priority_score=0.85,
                context_hints=[
                    "디자인 시대적 맥락",
                    "기술적 완성도",
                    "상업적 가치",
                    "트렌드 적합성"
                ],
                validation_keywords=[
                    "디자인", "스타일", "비례", "조화", "창의성", "상업성"
                ],
                fallback_prompt="주얼리 디자인의 특징과 장단점을 분석해주세요."
            )
        }
        
        # 비즈니스 인사이트 템플릿
        templates["business_insight"] = {
            "gpt-4-vision": JewelryPromptTemplate(
                category=JewelryCategory.BUSINESS_INSIGHT,
                level=AnalysisLevel.PROFESSIONAL,
                model_type=AIModelType.GPT4_VISION,
                prompt_ko="""당신은 주얼리 업계 비즈니스 전문가입니다. 제시된 정보를 바탕으로 주얼리 비즈니스 관점에서 전략적 인사이트를 제공해주세요.

비즈니스 분석 영역:
1. 시장 분석 (Market Analysis)
   - 현재 시장 트렌드
   - 경쟁사 포지셔닝
   - 시장 기회와 위험

2. 고객 분석 (Customer Analysis)
   - 타겟 고객 세분화
   - 구매 패턴 분석
   - 고객 니즈와 선호도

3. 제품 전략 (Product Strategy)
   - 제품 포트폴리오 최적화
   - 가격 전략 수립
   - 차별화 포인트 도출

4. 마케팅 전략 (Marketing Strategy)
   - 브랜드 포지셔닝
   - 마케팅 채널 선택
   - 프로모션 전략

5. 운영 최적화 (Operations)
   - 공급망 관리
   - 재고 최적화
   - 품질 관리 시스템

6. 재무 분석 (Financial Analysis)
   - 수익성 분석
   - 투자 수익률
   - 비용 구조 최적화

결과 제공:
• 비즈니스 현황 진단
• 핵심 기회 영역 식별
• 전략적 권장사항
• 실행 계획 수립
• 성과 측정 지표
• 리스크 관리 방안

목표 정확도: 98.0%""",
                
                prompt_en="""You are a jewelry industry business expert. Provide strategic business insights based on the presented information.

Business Analysis Areas:
1. Market Analysis
2. Customer Analysis
3. Product Strategy
4. Marketing Strategy
5. Operations Optimization
6. Financial Analysis

Target Accuracy: 98.0%""",
                
                expected_accuracy=0.980,
                priority_score=0.80,
                context_hints=[
                    "시장 트렌드 반영",
                    "실행 가능한 전략",
                    "ROI 중심 사고",
                    "리스크 관리"
                ],
                validation_keywords=[
                    "시장", "고객", "전략", "마케팅", "수익성", "경쟁력"
                ],
                fallback_prompt="주얼리 비즈니스 관점에서 전략적 조언을 제공해주세요."
            )
        }
        
        return templates
    
    def optimize_prompt_for_model(self, category: str, model_type: str, 
                                content: str, context: Dict[str, Any] = None) -> PromptOptimizationResult:
        """모델별 최적화된 프롬프트 생성"""
        
        # 기본 템플릿 선택
        if category not in self.templates:
            return self._create_fallback_result(category, content)
        
        if model_type not in self.templates[category]:
            # 기본 모델로 대체
            model_type = "gpt-4-vision"
        
        template = self.templates[category][model_type]
        
        # 컨텍스트 기반 최적화
        optimized_prompt = self._contextualize_prompt(template, content, context)
        
        # 모델별 특화 힌트
        model_hints = self._get_model_specific_hints(model_type, category)
        
        # 출력 형식 정의
        output_format = self._define_output_format(category, template.level)
        
        # 품질 기준 설정
        quality_criteria = self._set_quality_criteria(category, template.expected_accuracy)
        
        return PromptOptimizationResult(
            optimized_prompt=optimized_prompt,
            confidence_score=template.expected_accuracy,
            model_specific_hints=model_hints,
            expected_output_format=output_format,
            quality_criteria=quality_criteria
        )
    
    def _contextualize_prompt(self, template: JewelryPromptTemplate, 
                            content: str, context: Dict[str, Any] = None) -> str:
        """컨텍스트 기반 프롬프트 맞춤화"""
        
        base_prompt = template.prompt_ko
        
        # 컨텍스트 정보 추가
        if context:
            context_info = []
            
            if "image_provided" in context and context["image_provided"]:
                context_info.append("제공된 이미지를 상세히 분석하여")
            
            if "urgency" in context and context["urgency"] == "high":
                context_info.append("신속하고 정확한 분석으로")
            
            if "client_type" in context:
                if context["client_type"] == "professional":
                    context_info.append("전문가용 상세 분석으로")
                elif context["client_type"] == "consumer":
                    context_info.append("일반 고객이 이해하기 쉽게")
            
            if context_info:
                base_prompt = f"{', '.join(context_info)} {base_prompt}"
        
        # 분석 대상 추가
        if content:
            base_prompt += f"\n\n[분석 대상]\n{content[:500]}..."
        
        return base_prompt
    
    def _get_model_specific_hints(self, model_type: str, category: str) -> List[str]:
        """모델별 특화 힌트 제공"""
        
        hints_map = {
            "gpt-4-vision": [
                "구체적이고 상세한 분석 선호",
                "단계별 논리적 접근",
                "실무적 권장사항 포함",
                "정량적 데이터 활용"
            ],
            "claude-vision": [
                "구조화된 분석 방법론",
                "명확한 근거 제시",
                "체계적인 결론 도출",
                "균형잡힌 관점 유지"
            ],
            "gemini-2-pro": [
                "창의적 접근 방식",
                "최신 트렌드 반영",
                "혁신적 인사이트",
                "다각도 분석"
            ]
        }
        
        return hints_map.get(model_type, ["표준 분석 방식 적용"])
    
    def _define_output_format(self, category: str, level: AnalysisLevel) -> str:
        """출력 형식 정의"""
        
        if level in [AnalysisLevel.EXPERT, AnalysisLevel.CERTIFICATION]:
            return """
## 📋 전문가급 분석 결과

### 🔍 핵심 요약
[3-5줄 핵심 내용]

### 📊 상세 분석
[항목별 세부 분석]

### 💡 전문가 의견
[전문적 판단과 근거]

### 📈 권장사항
[실무적 조치사항]

### 🎯 결론
[최종 판단 및 요약]
"""
        else:
            return """
## 📋 분석 결과

### 요약
[핵심 내용]

### 세부 분석
[상세 내용]

### 권장사항
[조치사항]
"""
    
    def _set_quality_criteria(self, category: str, expected_accuracy: float) -> List[str]:
        """품질 기준 설정"""
        
        base_criteria = [
            f"목표 정확도: {expected_accuracy*100:.1f}%",
            "논리적 일관성 유지",
            "실무 적용 가능성",
            "전문 용어 정확 사용"
        ]
        
        category_specific = {
            "diamond_4c": [
                "GIA 표준 완벽 준수",
                "4C 등급 정확성",
                "시장가치 합리성"
            ],
            "colored_gemstone": [
                "국제 감정기관 기준",
                "원산지 판정 신뢰성",
                "처리 여부 정확 판별"
            ],
            "jewelry_design": [
                "디자인 요소 정확 분석",
                "예술적 가치 객관 평가",
                "상업적 실용성"
            ],
            "business_insight": [
                "시장 데이터 기반",
                "실행 가능한 전략",
                "ROI 중심 사고"
            ]
        }
        
        return base_criteria + category_specific.get(category, [])
    
    def _create_fallback_result(self, category: str, content: str) -> PromptOptimizationResult:
        """폴백 결과 생성"""
        
        fallback_prompt = f"""
주얼리 전문가로서 다음 내용을 분석해주세요:

{content[:300]}

분석 요청 사항:
- 전문적이고 정확한 분석
- 실무에 활용 가능한 인사이트
- 구체적인 권장사항
- 논리적 근거 제시

목표 정확도: 95%
"""
        
        return PromptOptimizationResult(
            optimized_prompt=fallback_prompt,
            confidence_score=0.95,
            model_specific_hints=["표준 분석 방식 적용"],
            expected_output_format="기본 분석 형식",
            quality_criteria=["정확성", "실용성", "논리성"]
        )
    
    def get_category_performance(self, category: str) -> Dict[str, float]:
        """카테고리별 성능 통계"""
        
        if category not in self.performance_history:
            return {"average_accuracy": 0.0, "total_requests": 0}
        
        history = self.performance_history[category]
        return {
            "average_accuracy": sum(history) / len(history),
            "total_requests": len(history),
            "latest_accuracy": history[-1] if history else 0.0,
            "target_accuracy": self.accuracy_targets.get(category, 0.95)
        }
    
    def update_performance(self, category: str, accuracy: float):
        """성능 기록 업데이트"""
        
        if category not in self.performance_history:
            self.performance_history[category] = []
        
        self.performance_history[category].append(accuracy)
        
        # 최근 100개 기록만 유지
        if len(self.performance_history[category]) > 100:
            self.performance_history[category] = self.performance_history[category][-100:]
    
    def get_optimization_suggestions(self) -> List[str]:
        """최적화 제안사항"""
        
        suggestions = []
        
        for category, history in self.performance_history.items():
            if len(history) >= 5:
                avg_accuracy = sum(history[-5:]) / 5
                target = self.accuracy_targets.get(category, 0.95)
                
                if avg_accuracy < target:
                    gap = (target - avg_accuracy) * 100
                    suggestions.append(
                        f"{category} 카테고리 정확도 개선 필요 (현재 {avg_accuracy:.1%}, 목표 대비 -{gap:.1f}%p)"
                    )
        
        if not suggestions:
            suggestions.append("모든 카테고리가 목표 정확도를 달성하고 있습니다.")
        
        return suggestions
    
    def export_templates(self, filepath: str = "jewelry_prompts_v23.json"):
        """템플릿 내보내기"""
        
        export_data = {}
        for category, models in self.templates.items():
            export_data[category] = {}
            for model_type, template in models.items():
                export_data[category][model_type] = {
                    "prompt_ko": template.prompt_ko,
                    "prompt_en": template.prompt_en,
                    "expected_accuracy": template.expected_accuracy,
                    "priority_score": template.priority_score
                }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"프롬프트 템플릿이 {filepath}에 저장되었습니다.")

# 사용 예시 및 테스트
def demo_jewelry_prompts_v23():
    """주얼리 프롬프트 v2.3 데모"""
    
    print("🚀 솔로몬드 주얼리 프롬프트 v2.3 데모 시작")
    print("=" * 60)
    
    optimizer = JewelryPromptOptimizerV23()
    
    # 다이아몬드 4C 분석 예시
    print("\n💎 다이아몬드 4C 분석 프롬프트 최적화")
    diamond_result = optimizer.optimize_prompt_for_model(
        category="diamond_4c",
        model_type="gpt-4-vision",
        content="1.5캐럿 라운드 브릴리언트 컷 다이아몬드, H컬러, VS1 클래리티 분석 요청",
        context={"image_provided": True, "client_type": "professional"}
    )
    
    print(f"✅ 신뢰도: {diamond_result.confidence_score:.1%}")
    print(f"📋 모델 힌트: {', '.join(diamond_result.model_specific_hints[:2])}")
    print(f"🎯 품질 기준: {len(diamond_result.quality_criteria)}개 항목")
    
    # 유색보석 감정 예시
    print("\n🌈 유색보석 감정 프롬프트 최적화")
    gemstone_result = optimizer.optimize_prompt_for_model(
        category="colored_gemstone",
        model_type="claude-vision",
        content="2.3캐럿 오벌 컷 루비, 피죤 블러드 컬러, 미얀마산 추정",
        context={"urgency": "high"}
    )
    
    print(f"✅ 신뢰도: {gemstone_result.confidence_score:.1%}")
    print(f"📋 출력 형식: 구조화된 분석 리포트")
    
    # 성능 시뮬레이션
    print("\n📈 성능 추적 시뮬레이션")
    for category in ["diamond_4c", "colored_gemstone", "business_insight"]:
        # 가상 성능 데이터
        for _ in range(10):
            accuracy = 0.94 + (category == "diamond_4c") * 0.04 + np.random.normal(0, 0.01)
            optimizer.update_performance(category, max(0.9, min(1.0, accuracy)))
    
    # 성능 요약
    for category in ["diamond_4c", "colored_gemstone", "business_insight"]:
        performance = optimizer.get_category_performance(category)
        print(f"📊 {category}: {performance['average_accuracy']:.1%} "
              f"(목표: {performance['target_accuracy']:.1%})")
    
    # 최적화 제안
    print("\n💡 최적화 제안사항:")
    suggestions = optimizer.get_optimization_suggestions()
    for suggestion in suggestions[:3]:
        print(f"  • {suggestion}")
    
    print("\n🎉 v2.3 주얼리 프롬프트 시스템 데모 완료!")
    print("🏆 목표: Week 2 완성으로 99.2% 정확도 달성!")

if __name__ == "__main__":
    demo_jewelry_prompts_v23()
