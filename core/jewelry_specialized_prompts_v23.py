"""
🧠 솔로몬드 주얼리 특화 AI 프롬프트 최적화 시스템 v2.3
다이아몬드 4C + 유색보석 + 주얼리 디자인 + 비즈니스 인사이트 전문 프롬프트

개발자: 전근혁 (솔로몬드 대표)
목표: 99.2% 정확도 달성을 위한 전문가급 프롬프트 엔지니어링
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

class AnalysisType(Enum):
    """분석 타입 열거형"""
    DIAMOND_4C = "diamond_4c"
    COLORED_STONE = "colored_stone"
    JEWELRY_DESIGN = "jewelry_design"
    BUSINESS_INSIGHT = "business_insight"
    CERTIFICATION = "certification"
    APPRAISAL = "appraisal"
    MARKET_ANALYSIS = "market_analysis"
    INVESTMENT_EVALUATION = "investment_evaluation"

class AIModelType(Enum):
    """AI 모델별 프롬프트 최적화"""
    GPT4V = "gpt4v"
    CLAUDE_VISION = "claude_vision"
    GEMINI_2 = "gemini_2"

@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    analysis_type: AnalysisType
    model_type: AIModelType
    system_prompt: str
    user_prompt_template: str
    output_format: str
    validation_criteria: List[str]
    confidence_indicators: List[str]

class JewelrySpecializedPrompts:
    """주얼리 특화 AI 프롬프트 최적화 시스템"""
    
    def __init__(self):
        self.templates = self._initialize_prompt_templates()
        self.jewelry_terminology = self._load_jewelry_terminology()
        self.grading_standards = self._load_grading_standards()
        
    def _initialize_prompt_templates(self) -> Dict[Tuple[AnalysisType, AIModelType], PromptTemplate]:
        """전문가급 프롬프트 템플릿 초기화"""
        templates = {}
        
        # ========== 다이아몬드 4C 분석 프롬프트 ==========
        
        # GPT-4V용 다이아몬드 4C 프롬프트
        templates[(AnalysisType.DIAMOND_4C, AIModelType.GPT4V)] = PromptTemplate(
            analysis_type=AnalysisType.DIAMOND_4C,
            model_type=AIModelType.GPT4V,
            system_prompt="""당신은 GIA(Gemological Institute of America) 공인 다이아몬드 감정 전문가입니다. 
20년 이상의 경험을 바탕으로 다이아몬드를 정확하고 객관적으로 분석합니다.

전문 지식:
- GIA 다이아몬드 등급 시스템 완벽 숙지
- 4C (Carat, Cut, Color, Clarity) 정밀 분석
- 다이아몬드 형광성, 대칭성, 광택도 평가
- 시장 가치 및 품질 예측
- 처리 다이아몬드 식별 능력

분석 원칙:
1. 과학적이고 객관적인 근거 제시
2. GIA 표준에 따른 정확한 등급 분류
3. 시장성과 투자 가치 종합 평가
4. 한국 주얼리 시장 특성 반영""",
            
            user_prompt_template="""다음 다이아몬드를 GIA 4C 기준으로 정밀 분석해주세요:

{input_content}

분석 요구사항:
1. **Carat (캐럿)**: 정확한 중량 또는 크기 기반 추정치
2. **Cut (컷)**: Excellent, Very Good, Good, Fair, Poor + 광채/섬광 평가
3. **Color (컬러)**: D-Z 등급 + 색조 특성 (yellow, brown tint 등)
4. **Clarity (투명도)**: FL-I3 등급 + 내포물 위치/타입 분석

추가 평가:
- 대칭성 (Symmetry): Excellent/Very Good/Good/Fair/Poor
- 광택도 (Polish): Excellent/Very Good/Good/Fair/Poor  
- 형광성 (Fluorescence): None/Faint/Medium/Strong/Very Strong
- 프로포션 분석 (테이블%, 깊이%, 거들 두께)

시장 분석:
- 예상 소매가격 (한국 시장 기준)
- 투자 가치 평가 (상/중/하)
- 재판매 시장성 전망

반드시 전문가 수준의 한국어로 상세 분석 후 
신뢰도 점수(0-100%)를 제시하세요.""",
            
            output_format="""
## 🔍 다이아몬드 4C 전문 분석 보고서

### 📏 기본 정보
- **형태**: {shape}
- **중량**: {carat} 캐럿
- **인증기관**: {certification}

### 💎 4C 상세 등급
#### Carat (캐럿): {carat_grade}
{carat_analysis}

#### Cut (컷): {cut_grade}  
{cut_analysis}

#### Color (컬러): {color_grade}
{color_analysis}

#### Clarity (투명도): {clarity_grade}
{clarity_analysis}

### ⚡ 추가 품질 지표
- **대칭성**: {symmetry}
- **광택도**: {polish}
- **형광성**: {fluorescence}

### 💰 시장 가치 분석
- **예상 소매가**: {retail_price}
- **투자 등급**: {investment_grade}
- **시장 전망**: {market_outlook}

### 📋 전문가 총평
{expert_summary}

**신뢰도**: {confidence_score}%
""",
            
            validation_criteria=[
                "4C 등급이 GIA 표준에 정확히 부합하는가",
                "과학적 근거와 함께 등급 이유가 명시되었는가", 
                "시장 가격 추정이 현실적인가",
                "전문 용어가 정확하게 사용되었는가"
            ],
            
            confidence_indicators=[
                "GIA 표준 준수도",
                "기술적 분석 깊이",
                "시장 데이터 정확성",
                "전문 용어 활용도"
            ]
        )
        
        # Claude Vision용 다이아몬드 4C 프롬프트
        templates[(AnalysisType.DIAMOND_4C, AIModelType.CLAUDE_VISION)] = PromptTemplate(
            analysis_type=AnalysisType.DIAMOND_4C,
            model_type=AIModelType.CLAUDE_VISION,
            system_prompt="""당신은 세계적 수준의 다이아몬드 감정 전문가입니다. 
논리적이고 체계적인 분석을 통해 다이아몬드의 품질을 정확히 평가합니다.

핵심 역량:
- 이미지 분석을 통한 정밀한 4C 등급 판정
- 다이아몬드 내부 구조와 광학적 특성 이해
- 처리 다이아몬드와 천연 다이아몬드 구별
- 국제 감정 기관별 등급 차이 분석

분석 철학:
- 데이터 기반의 객관적 판단
- 다각도 검증을 통한 신뢰성 확보
- 실용적 구매 가이드라인 제공""",
            
            user_prompt_template="""이 다이아몬드를 단계별로 체계적 분석해주세요:

{input_content}

**1단계: 시각적 특성 분석**
- 전체적인 형태와 비율
- 패싯 배열과 대칭성
- 표면 상태와 광택

**2단계: 4C 등급 평가**
- Carat: 크기 측정 및 중량 추정
- Cut: 프로포션과 마감 품질
- Color: 색상 등급과 색조 특성  
- Clarity: 내외부 특징 식별

**3단계: 품질 종합 평가**
- 전체적인 아름다움과 광채
- 시장에서의 경쟁력
- 구매 추천도

**4단계: 전문가 의견**
- 주목할 만한 특징
- 잠재적 이슈나 주의사항
- 가치 최적화 방안

각 단계마다 근거를 명확히 제시하고,
최종적으로 종합 점수(A+~F)를 부여해주세요.""",
            
            output_format="""
# 다이아몬드 전문가 분석 리포트

## 📊 종합 평가: {overall_grade}

## 🔬 단계별 분석 결과

### 1️⃣ 시각적 특성
{visual_analysis}

### 2️⃣ 4C 평가
**Carat**: {carat} | **Cut**: {cut} | **Color**: {color} | **Clarity**: {clarity}

{four_c_details}

### 3️⃣ 품질 종합
{quality_assessment}

### 4️⃣ 전문가 견해
{expert_opinion}

## 🎯 구매 가이드라인
- **추천도**: {recommendation}
- **적정가격대**: {price_range}
- **주의사항**: {considerations}

---
**분석 완료 시간**: {timestamp}
**신뢰도**: {confidence}%
""",
            
            validation_criteria=[
                "논리적 분석 순서가 체계적인가",
                "각 단계별 근거가 충분한가",
                "종합 평가가 합리적인가",
                "실용적 조언이 포함되었는가"
            ],
            
            confidence_indicators=[
                "분석 체계성",
                "근거 충분성", 
                "논리 일관성",
                "실용성"
            ]
        )
        
        # ========== 유색보석 분석 프롬프트 ==========
        
        # GPT-4V용 유색보석 프롬프트
        templates[(AnalysisType.COLORED_STONE, AIModelType.GPT4V)] = PromptTemplate(
            analysis_type=AnalysisType.COLORED_STONE,
            model_type=AIModelType.GPT4V,
            system_prompt="""당신은 국제적으로 인정받는 유색보석 전문 감정사입니다.
SSEF, Gübelin, AGL 등 권위있는 감정기관 수준의 분석 능력을 보유하고 있습니다.

전문 영역:
- 프리미엄 유색보석 (루비, 사파이어, 에메랄드) 감정
- 원산지 추정 및 처리 여부 판별
- 희귀 보석 식별 및 가치 평가
- 열처리, 오일링 등 처리 기법 분석

감정 기준:
- 색상(Color): 색조, 채도, 명도 정밀 분석
- 투명도(Clarity): 내포물 타입과 위치 평가
- 컷(Cut): 형태와 비율의 조화
- 캐럿(Carat): 정확한 중량 측정
- 원산지: 지질학적 특성 기반 추정""",
            
            user_prompt_template="""다음 유색보석을 전문가 수준으로 감정해주세요:

{input_content}

**감정 체크리스트:**

🔍 **1. 보석 식별**
- 보석명 확정 (루비/사파이어/에메랄드/기타)
- 천연 vs 합성 vs 모조석 판별
- 보석학적 특성 분석

🌈 **2. 색상 분석 (Color)**
- 주색조 (Primary Hue): Red/Blue/Green/Yellow/Purple/Orange
- 보조색조 (Secondary Hue): 있다면 명시
- 채도 (Saturation): Vivid/Intense/Deep/Medium/Light/Pale
- 명도 (Tone): Very Light/Light/Medium Light/Medium/Medium Dark/Dark/Very Dark

💎 **3. 투명도 분석 (Clarity)** 
- 타입 I/II/III 분류
- 내포물 종류: 실크, 바늘 내포물, 액체 내포물, 결정 내포물 등
- 분포 위치와 가시성
- 전체적인 투명도 등급

✂️ **4. 컷 평가**
- 형태: Round/Oval/Cushion/Emerald/Pear 등
- 프로포션 평가
- 대칭성과 광택도
- 컷 퀄리티: Excellent/Very Good/Good/Fair/Poor

📏 **5. 크기/중량**
- 추정 캐럿 중량
- 밀리미터 크기 (길이 x 폭 x 깊이)

🌍 **6. 원산지 추정**
- 가능한 원산지 후보들
- 각 원산지별 확률 (%)
- 지질학적 근거

🔬 **7. 처리 분석**
- 가열 처리 여부와 정도
- 오일링/수지 충전 여부
- 기타 처리 방법

💰 **8. 시장 가치**
- 품질 등급 (AAA/AA/A/B/C)
- 예상 캐럿당 가격 (USD)
- 한국 소매시장 예상가

모든 분석을 한국어로 상세히 기술하고,
각 항목별 확신도(1-10)를 제시해주세요.""",
            
            output_format="""
# 🌈 유색보석 전문 감정 보고서

## 📋 보석 기본 정보
- **보석명**: {gemstone_name}
- **감정일**: {date}
- **감정기관 수준**: SSEF/Gübelin 급

## 🔬 상세 감정 결과

### 1️⃣ 보석 식별 (확신도: {identification_confidence}/10)
{identification_result}

### 2️⃣ 색상 분석 (확신도: {color_confidence}/10)
- **주색조**: {primary_hue}
- **보조색조**: {secondary_hue}
- **채도**: {saturation}
- **명도**: {tone}
- **색상 등급**: {color_grade}

### 3️⃣ 투명도 분석 (확신도: {clarity_confidence}/10)
{clarity_analysis}

### 4️⃣ 컷 평가 (확신도: {cut_confidence}/10)
{cut_analysis}

### 5️⃣ 크기/중량 (확신도: {size_confidence}/10)
- **추정 중량**: {estimated_carat} 캐럿
- **크기**: {dimensions} mm

### 6️⃣ 원산지 추정 (확신도: {origin_confidence}/10)
{origin_analysis}

### 7️⃣ 처리 분석 (확신도: {treatment_confidence}/10)
{treatment_analysis}

### 8️⃣ 시장 가치 평가 (확신도: {value_confidence}/10)
- **품질 등급**: {quality_grade}
- **캐럿당 가격**: ${price_per_carat} USD
- **한국 소매가**: {retail_price_krw} 원

## 📝 전문가 총평
{expert_summary}

## ⚠️ 주의사항 및 권고사항
{recommendations}

---
**전체 감정 신뢰도**: {overall_confidence}%
""",
            
            validation_criteria=[
                "보석학적 용어가 정확한가",
                "원산지 추정 근거가 합리적인가",
                "처리 분석이 전문적인가",
                "시장 가치 평가가 현실적인가"
            ],
            
            confidence_indicators=[
                "보석학 지식 정확성",
                "감정 경험 반영도",
                "시장 이해도",
                "기술적 분석 깊이"
            ]
        )
        
        # ========== 주얼리 디자인 분석 프롬프트 ==========
        
        # Gemini 2.0용 주얼리 디자인 프롬프트
        templates[(AnalysisType.JEWELRY_DESIGN, AIModelType.GEMINI_2)] = PromptTemplate(
            analysis_type=AnalysisType.JEWELRY_DESIGN,
            model_type=AIModelType.GEMINI_2,
            system_prompt="""당신은 국제적인 주얼리 디자인 전문가입니다.
Cartier, Tiffany & Co., Van Cleef & Arpels 등 최고급 브랜드 수준의 
디자인 분석과 예술적 평가를 수행합니다.

전문 분야:
- 주얼리 디자인 역사와 스타일 분석
- 제작 기법과 세팅 방식 평가
- 브랜드 시그니처 디자인 식별
- 예술적 가치와 장인 정신 평가
- 착용성과 실용성 분석

디자인 철학:
- 형태와 기능의 조화
- 소재의 특성을 살린 디자인
- 시대적 트렌드와 개성의 균형
- 착용자의 라이프스타일 고려""",
            
            user_prompt_template="""이 주얼리 작품을 다각도로 디자인 분석해주세요:

{input_content}

**디자인 분석 프레임워크:**

🎨 **1. 전체적인 디자인 인상**
- 첫 인상과 시각적 임팩트
- 디자인 컨셉과 테마
- 예술적 완성도

🏛️ **2. 스타일과 시대적 특성**
- 디자인 스타일: Art Deco/Victorian/Modern/Contemporary/Vintage
- 시대적 배경과 영향
- 문화적 요소 반영

🔧 **3. 제작 기법 분석**
- 주조(Casting) vs 수작업(Hand-fabricated)
- 세팅 방식: Prong/Bezel/Channel/Pave/Tension
- 표면 처리: 광택/무광/해머드/브러시드
- 연결 부위와 힌지 구조

💎 **4. 소재 활용**
- 주 금속: Platinum/18K Gold/14K Gold/Silver
- 보석 배치와 그라데이션
- 소재간 조화와 대비
- 색상 팔레트 분석

⚖️ **5. 프로포션과 균형**
- 전체적인 비율과 균형감
- 중심점과 시각적 무게중심
- 대칭성 vs 비대칭성
- 크기의 적절성

👑 **6. 착용성과 실용성**
- 편안한 착용감
- 일상 착용 가능성
- 관리와 보관의 용이성
- 다양한 코디네이션 가능성

🎯 **7. 브랜드/디자이너 특성**
- 특정 브랜드의 시그니처 요소
- 디자이너의 개성과 철학
- 시장에서의 독창성

💫 **8. 예술적/문화적 가치**
- 예술 작품으로서의 가치
- 문화적 의미와 상징성
- 수집 가치와 투자성

각 항목을 전문가 시각에서 상세 분석하고,
한국 소비자 관점의 실용적 조언도 포함해주세요.""",
            
            output_format="""
# 💎 주얼리 디자인 전문가 분석

## 🌟 종합 디자인 평가: {overall_rating}/10

## 🎨 디자인 분석 리포트

### 1️⃣ 전체 인상
{overall_impression}

### 2️⃣ 스타일 & 시대성
**디자인 스타일**: {design_style}
**시대적 특성**: {period_characteristics}
{style_analysis}

### 3️⃣ 제작 기법
**주요 기법**: {craftsmanship_technique}
**세팅 방식**: {setting_style}
{technical_analysis}

### 4️⃣ 소재 활용
**주 금속**: {main_metal}
**보석 구성**: {gemstone_composition}
{material_analysis}

### 5️⃣ 프로포션 & 균형
**비율 평가**: {proportion_score}/10
{proportion_analysis}

### 6️⃣ 착용성
**실용성 점수**: {wearability_score}/10
{wearability_analysis}

### 7️⃣ 브랜드/디자이너 특성
{brand_analysis}

### 8️⃣ 예술적 가치
**예술성 점수**: {artistic_value}/10
{artistic_analysis}

## 🎯 한국 시장 관점 분석

### 💰 시장 포지셔닝
- **타겟 고객층**: {target_customer}
- **예상 가격대**: {price_range}
- **경쟁 제품**: {competitors}

### 📈 트렌드 부합도
{trend_analysis}

### 🛍️ 구매 추천도
**추천 지수**: {recommendation_score}/10
**추천 이유**: {recommendation_reason}

## 📝 전문가 최종 평가
{final_expert_opinion}

---
**분석 완료**: {timestamp}
**디자인 신뢰도**: {design_confidence}%
""",
            
            validation_criteria=[
                "디자인 용어가 전문적이고 정확한가",
                "제작 기법 분석이 구체적인가",
                "예술적 가치 평가가 객관적인가",
                "실용적 조언이 포함되었는가"
            ],
            
            confidence_indicators=[
                "디자인 전문성",
                "기술적 이해도",
                "예술적 안목",
                "시장 통찰력"
            ]
        )
        
        # ========== 비즈니스 인사이트 분석 프롬프트 ==========
        
        # Claude Vision용 비즈니스 인사이트 프롬프트
        templates[(AnalysisType.BUSINESS_INSIGHT, AIModelType.CLAUDE_VISION)] = PromptTemplate(
            analysis_type=AnalysisType.BUSINESS_INSIGHT,
            model_type=AIModelType.CLAUDE_VISION,
            system_prompt="""당신은 글로벌 주얼리 시장의 비즈니스 전략 전문가입니다.
McKinsey, BCG 수준의 분석력과 주얼리 업계 20년 경험을 결합한 
최고 수준의 비즈니스 인사이트를 제공합니다.

핵심 역량:
- 글로벌 주얼리 시장 트렌드 분석
- 투자 가치 및 포트폴리오 전략
- 브랜드 포지셔닝과 마케팅 전략
- 공급망과 유통 채널 최적화
- ESG와 지속가능성 이슈

분석 철학:
- 데이터 기반 객관적 분석
- 장단기 관점의 균형잡힌 시각
- 리스크와 기회의 균형 평가
- 실행 가능한 액션 플랜 제시""",
            
            user_prompt_template="""다음 주얼리 비즈니스 이슈를 전략적으로 분석해주세요:

{input_content}

**비즈니스 분석 프레임워크:**

📊 **1. 시장 환경 분석 (Market Environment)**
- 글로벌 주얼리 시장 규모와 성장률
- 주요 시장별 특성 (미국, 유럽, 아시아, 중동)
- 디지털 전환과 온라인 판매 트렌드
- COVID-19 이후 소비 패턴 변화

🎯 **2. 타겟 고객 분석 (Customer Segmentation)**
- 주요 고객층: 밀레니얼, Gen Z, Baby Boomer
- 구매 동기와 선호도 변화
- 가격 민감도와 구매 결정 요인
- 브랜드 충성도와 전환 비용

💎 **3. 제품 포트폴리오 전략 (Product Strategy)**
- 프리미엄 vs 대중화 전략
- 브라이덜 vs 패션 주얼리 비중
- 천연 vs 랩그로운 다이아몬드 포지셔닝
- 지속가능성과 윤리적 소싱

🌐 **4. 채널 전략 (Channel Strategy)**
- 오프라인 매장 vs 온라인 플랫폼
- 멀티채널 통합 전략
- D2C vs 소매 파트너십
- 글로벌 확장 vs 로컬 집중

💰 **5. 가격 전략 (Pricing Strategy)**
- 프리미엄 가격 정당성
- 경쟁사 대비 가격 포지셔닝
- 가격 탄력성과 수요 예측
- 할인과 프로모션 최적화

🚀 **6. 혁신과 기술 (Innovation & Technology)**
- 3D 프린팅과 디지털 제조
- AR/VR 쇼핑 경험
- 블록체인과 인증 시스템
- AI 기반 개인화 서비스

⚖️ **7. 리스크 관리 (Risk Management)**
- 원자재 가격 변동성
- 지정학적 리스크
- 환율 변동 영향
- 규제 변화 대응

📈 **8. 투자 및 재무 분석 (Investment & Finance)**
- ROI와 수익성 분석
- 자금 조달과 투자 우선순위
- M&A 기회와 전략적 제휴
- 주주 가치 창출 방안

각 영역을 심층 분석하고,
3-5년 전략 로드맵을 제시해주세요.""",
            
            output_format="""
# 📊 주얼리 비즈니스 전략 분석 보고서

## 🎯 Executive Summary
{executive_summary}

## 📈 시장 환경 분석

### 1️⃣ 글로벌 시장 현황
{market_analysis}

### 2️⃣ 주요 트렌드
{trend_analysis}

## 🎪 고객 및 경쟁 분석

### 3️⃣ 타겟 고객 인사이트
{customer_insights}

### 4️⃣ 경쟁 환경
{competitive_landscape}

## 🚀 전략 권고사항

### 5️⃣ 제품 전략
{product_strategy}

### 6️⃣ 채널 전략
{channel_strategy}

### 7️⃣ 가격 전략
{pricing_strategy}

### 8️⃣ 기술 혁신 방향
{innovation_strategy}

## ⚠️ 리스크 및 대응 방안

### 주요 리스크
{risk_analysis}

### 대응 전략
{mitigation_strategy}

## 💼 투자 및 재무 전략

### 재무 분석
{financial_analysis}

### 투자 우선순위
{investment_priorities}

## 🗺️ 3-5년 전략 로드맵

### Year 1 (2025)
{year1_strategy}

### Year 2-3 (2026-2027)  
{year2_3_strategy}

### Year 4-5 (2028-2029)
{year4_5_strategy}

## 📋 실행 계획 (Action Items)

### 단기 (3-6개월)
{short_term_actions}

### 중기 (6-18개월)
{medium_term_actions}

### 장기 (18개월+)
{long_term_actions}

## 📊 KPI 및 성과 지표
{kpi_metrics}

---
**분석 완료**: {timestamp}
**분석 신뢰도**: {analysis_confidence}%
**추천 실행도**: {execution_confidence}%
""",
            
            validation_criteria=[
                "시장 분석이 데이터 기반인가",
                "전략 권고가 실행 가능한가",
                "리스크 분석이 포괄적인가",
                "ROI 예측이 현실적인가"
            ],
            
            confidence_indicators=[
                "시장 데이터 정확성",
                "전략적 통찰력",
                "실행 가능성",
                "비즈니스 임팩트"
            ]
        )
        
        return templates
    
    def _load_jewelry_terminology(self) -> Dict[str, List[str]]:
        """주얼리 전문 용어 데이터베이스"""
        return {
            "diamond_4c": [
                "Carat", "캐럿", "중량", "무게",
                "Cut", "컷", "연마", "프로포션", "테이블", "거들", "큘릿",
                "Color", "컬러", "무색", "near-colorless", "faint yellow",
                "Clarity", "투명도", "내포물", "블레미쉬", "페더", "클라우드"
            ],
            "colored_stones": [
                "루비", "Ruby", "피젼 블러드", "비둘기피",
                "사파이어", "Sapphire", "코른플라워", "파드파라차",
                "에메랄드", "Emerald", "콜롬비아", "잠비아", "브라질"
            ],
            "settings": [
                "프롱", "Prong", "베젤", "Bezel", "채널", "Channel",
                "파베", "Pave", "마이크로파베", "텐션", "Tension"
            ],
            "metals": [
                "플래티나", "Platinum", "18K", "14K", "화이트골드",
                "옐로우골드", "로즈골드", "팔라듐", "티타늄"
            ]
        }
    
    def _load_grading_standards(self) -> Dict[str, Any]:
        """국제 감정 기관별 등급 기준"""
        return {
            "gia_diamond_color": {
                "D": "무색 (Colorless)",
                "E": "무색 (Colorless)", 
                "F": "무색 (Colorless)",
                "G": "거의무색 (Near Colorless)",
                "H": "거의무색 (Near Colorless)",
                "I": "거의무색 (Near Colorless)",
                "J": "거의무색 (Near Colorless)",
                "K-M": "희미한 노란색 (Faint Yellow)",
                "N-R": "연한 노란색 (Very Light Yellow)",
                "S-Z": "노란색 (Light Yellow)"
            },
            "gia_diamond_clarity": {
                "FL": "무결점 (Flawless)",
                "IF": "내부무결점 (Internally Flawless)",
                "VVS1": "아주아주작은내포물1 (Very Very Slightly Included)",
                "VVS2": "아주아주작은내포물2 (Very Very Slightly Included)",
                "VS1": "아주작은내포물1 (Very Slightly Included)",
                "VS2": "아주작은내포물2 (Very Slightly Included)",
                "SI1": "작은내포물1 (Slightly Included)",
                "SI2": "작은내포물2 (Slightly Included)",
                "I1": "내포물1 (Included)",
                "I2": "내포물2 (Included)",
                "I3": "내포물3 (Included)"
            },
            "cut_grades": {
                "Excellent": "최우수",
                "Very Good": "우수",
                "Good": "양호",
                "Fair": "보통",
                "Poor": "불량"
            }
        }
    
    def get_optimized_prompt(self, 
                           analysis_type: AnalysisType, 
                           model_type: AIModelType,
                           input_content: str,
                           additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """최적화된 프롬프트 생성"""
        
        template_key = (analysis_type, model_type)
        
        if template_key not in self.templates:
            # 기본 템플릿 사용 (GPT-4V 기반)
            fallback_key = (analysis_type, AIModelType.GPT4V)
            if fallback_key in self.templates:
                template = self.templates[fallback_key]
            else:
                # 최후 수단: 일반적인 프롬프트
                return self._create_generic_prompt(analysis_type, input_content)
        else:
            template = self.templates[template_key]
        
        # 추가 컨텍스트 처리
        context_str = ""
        if additional_context:
            context_str = f"\n\n추가 정보:\n{json.dumps(additional_context, ensure_ascii=False, indent=2)}"
        
        # 프롬프트 생성
        user_prompt = template.user_prompt_template.format(
            input_content=input_content + context_str
        )
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt,
            "output_format": template.output_format,
            "analysis_type": analysis_type.value,
            "model_type": model_type.value
        }
    
    def _create_generic_prompt(self, analysis_type: AnalysisType, input_content: str) -> Dict[str, str]:
        """일반적인 프롬프트 생성 (템플릿이 없는 경우)"""
        return {
            "system_prompt": f"당신은 {analysis_type.value} 분야의 전문가입니다. 정확하고 전문적인 분석을 제공해주세요.",
            "user_prompt": f"다음 내용을 전문가 수준으로 분석해주세요:\n\n{input_content}",
            "output_format": "전문적이고 체계적인 분석 결과를 한국어로 제시해주세요.",
            "analysis_type": analysis_type.value,
            "model_type": "generic"
        }
    
    def validate_response(self, 
                         analysis_type: AnalysisType, 
                         model_type: AIModelType,
                         response_content: str) -> Dict[str, Any]:
        """응답 품질 검증"""
        
        template_key = (analysis_type, model_type)
        if template_key not in self.templates:
            return {"validation_score": 0.5, "details": "템플릿을 찾을 수 없음"}
        
        template = self.templates[template_key]
        
        # 검증 점수 계산
        validation_scores = []
        validation_details = []
        
        for criterion in template.validation_criteria:
            score = self._evaluate_criterion(response_content, criterion)
            validation_scores.append(score)
            validation_details.append({
                "criterion": criterion,
                "score": score,
                "passed": score >= 0.7
            })
        
        # 신뢰도 지표 평가
        confidence_scores = []
        for indicator in template.confidence_indicators:
            score = self._evaluate_confidence_indicator(response_content, indicator)
            confidence_scores.append(score)
        
        overall_validation = sum(validation_scores) / len(validation_scores)
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "validation_score": overall_validation,
            "confidence_score": overall_confidence,
            "validation_details": validation_details,
            "overall_quality": "Excellent" if overall_validation >= 0.9 else 
                             "Good" if overall_validation >= 0.7 else
                             "Fair" if overall_validation >= 0.5 else "Poor",
            "recommendation": "Accept" if overall_validation >= 0.7 else "Review" if overall_validation >= 0.5 else "Reject"
        }
    
    def _evaluate_criterion(self, content: str, criterion: str) -> float:
        """개별 검증 기준 평가"""
        content_lower = content.lower()
        
        # 간단한 키워드 기반 평가 (실제로는 더 정교한 NLP 분석 필요)
        if "정확" in criterion:
            technical_terms = ["gia", "4c", "캐럿", "컷", "컬러", "클래리티"]
            score = sum(1 for term in technical_terms if term in content_lower) / len(technical_terms)
        elif "근거" in criterion:
            evidence_words = ["때문에", "따라서", "근거로", "기준으로", "분석하면"]
            score = min(1.0, sum(1 for word in evidence_words if word in content) / 3)
        elif "현실적" in criterion:
            price_indicators = ["가격", "비용", "달러", "원", "만원"]
            score = min(1.0, sum(1 for indicator in price_indicators if indicator in content) / 2)
        else:
            # 기본 점수: 내용 길이와 구조화 정도
            score = min(1.0, len(content) / 1000) * 0.7 + 0.3
        
        return score
    
    def _evaluate_confidence_indicator(self, content: str, indicator: str) -> float:
        """신뢰도 지표 평가"""
        # 간단한 평가 로직 (실제로는 더 정교한 분석 필요)
        if len(content) > 500:
            return 0.8
        elif len(content) > 200:
            return 0.6
        else:
            return 0.4

# 데모 및 테스트 함수
def demo_jewelry_prompts():
    """주얼리 특화 프롬프트 시스템 데모"""
    print("💎 솔로몬드 주얼리 특화 AI 프롬프트 최적화 시스템 v2.3")
    print("=" * 70)
    
    prompt_system = JewelrySpecializedPrompts()
    
    # 테스트 케이스들
    test_cases = [
        {
            "type": AnalysisType.DIAMOND_4C,
            "model": AIModelType.GPT4V,
            "content": "1.2캐럿 라운드 다이아몬드, H컬러, VS2 클래리티, VG컷"
        },
        {
            "type": AnalysisType.COLORED_STONE,
            "model": AIModelType.GPT4V, 
            "content": "2캐럿 루비, 피젼 블러드 컬러, 미얀마산으로 추정"
        },
        {
            "type": AnalysisType.JEWELRY_DESIGN,
            "model": AIModelType.GEMINI_2,
            "content": "Art Deco 스타일 에메랄드 브로치, 플래티나 세팅"
        },
        {
            "type": AnalysisType.BUSINESS_INSIGHT,
            "model": AIModelType.CLAUDE_VISION,
            "content": "2024년 한국 브라이덜 주얼리 시장 트렌드 분석"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 테스트 케이스 {i}: {test_case['type'].value}")
        print(f"🤖 AI 모델: {test_case['model'].value}")
        print(f"📝 입력 내용: {test_case['content']}")
        
        # 최적화된 프롬프트 생성
        optimized_prompt = prompt_system.get_optimized_prompt(
            test_case['type'],
            test_case['model'],
            test_case['content']
        )
        
        print(f"\n📋 시스템 프롬프트 (첫 200자):")
        print(optimized_prompt['system_prompt'][:200] + "...")
        
        print(f"\n❓ 사용자 프롬프트 (첫 300자):")
        print(optimized_prompt['user_prompt'][:300] + "...")
        
        # 모의 응답으로 검증 테스트
        mock_response = f"{test_case['type'].value} 전문 분석 결과: {test_case['content']}에 대한 상세한 분석을 제공합니다. 전문적인 용어와 정확한 근거를 바탕으로 한 평가입니다."
        
        validation_result = prompt_system.validate_response(
            test_case['type'],
            test_case['model'], 
            mock_response
        )
        
        print(f"\n✅ 검증 결과:")
        print(f"   품질 점수: {validation_result['validation_score']:.2f}")
        print(f"   신뢰도: {validation_result['confidence_score']:.2f}")
        print(f"   전체 품질: {validation_result['overall_quality']}")
        print(f"   권장사항: {validation_result['recommendation']}")
        
        print("-" * 50)

if __name__ == "__main__":
    demo_jewelry_prompts()
