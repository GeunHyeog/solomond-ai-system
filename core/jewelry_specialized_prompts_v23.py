"""
Jewelry Specialized Prompts v2.3 for Solomond AI Platform
주얼리 특화 프롬프트 시스템 - 99.2% 정확도 달성을 위한 고급 프롬프트 엔지니어링

🎯 목표: 99.2% 분석 정확도 달성
📅 개발기간: 2025.07.13 - 2025.08.03 (3주)
👨‍💼 프로젝트 리더: 전근혁 (솔로몬드 대표)

핵심 기능:
- 다이아몬드 4C 분석 전용 프롬프트 (GIA, AGS 기준)
- 유색보석 감정 특화 프롬프트 (SSEF, Gübelin 기준)  
- 비즈니스 인사이트 추출 프롬프트
- AI 모델별 최적화 (GPT-4V, Claude, Gemini)
- 실시간 성능 추적 및 프롬프트 개선
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
from datetime import datetime

class JewelryCategory(Enum):
    """주얼리 카테고리"""
    DIAMOND_4C = "diamond_4c"
    COLORED_GEMSTONE = "colored_gemstone"
    PEARL = "pearl"
    PRECIOUS_METAL = "precious_metal"
    BUSINESS_INSIGHT = "business_insight"
    MARKET_ANALYSIS = "market_analysis"
    INVESTMENT_VALUATION = "investment_valuation"
    AUTHENTICATION = "authentication"

class AIModelType(Enum):
    """AI 모델 타입"""
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4_TURBO = "gpt-4-turbo-preview"
    CLAUDE_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_OPUS = "claude-3-opus-20240229"
    GEMINI_2_PRO = "gemini-2.0-flash-exp"
    GEMINI_PRO_VISION = "gemini-pro-vision"

class AnalysisLevel(Enum):
    """분석 수준"""
    BASIC = "basic"           # 기본 감정
    PROFESSIONAL = "professional"  # 전문가 수준
    EXPERT = "expert"         # 전문 감정사 수준
    CERTIFICATION = "certification"  # 감정서 수준

@dataclass
class PromptTemplate:
    """프롬프트 템플릿"""
    category: JewelryCategory
    model_type: AIModelType
    analysis_level: AnalysisLevel
    title: str
    system_prompt: str
    user_prompt_template: str
    expected_accuracy: float
    validation_keywords: List[str]
    output_format: Dict[str, Any]
    examples: List[Dict[str, str]]
    performance_metrics: Dict[str, float] = None

@dataclass
class PromptOptimizationResult:
    """프롬프트 최적화 결과"""
    original_prompt: str
    optimized_prompt: str
    optimization_score: float
    applied_techniques: List[str]
    estimated_accuracy: float

class InternationalStandards:
    """국제 감정 기준"""
    
    GIA_STANDARDS = {
        "color_scale": ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
        "clarity_scale": ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"],
        "cut_grades": ["Excellent", "Very Good", "Good", "Fair", "Poor"],
        "fluorescence": ["None", "Faint", "Medium", "Strong", "Very Strong"]
    }
    
    AGS_STANDARDS = {
        "cut_scale": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "color_scale": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "clarity_scale": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    }
    
    SSEF_GEMSTONE_ORIGINS = [
        "Myanmar (Burma)", "Sri Lanka (Ceylon)", "Kashmir", "Madagascar", 
        "Thailand", "Cambodia", "Vietnam", "Tanzania", "Kenya", "Mozambique"
    ]
    
    GUBELIN_TREATMENTS = [
        "No indication of thermal treatment",
        "Indication of thermal treatment",
        "Minor oil/resin in fissures",
        "Moderate oil/resin in fissures", 
        "Significant oil/resin in fissures"
    ]

class KoreanJewelryTerms:
    """한국어 주얼리 전문용어"""
    
    DIAMOND_TERMS = {
        "4C": "4C (캐럿, 컬러, 클래리티, 컷)",
        "carat": "캐럿 (중량)",
        "color": "컬러 (색상)",
        "clarity": "클래리티 (투명도)",
        "cut": "컷 (연마)",
        "polish": "폴리시 (광택)",
        "symmetry": "시메트리 (대칭성)",
        "fluorescence": "형광성",
        "girdle": "거들 (띠)",
        "culet": "큘릿 (하단면)"
    }
    
    GEMSTONE_TERMS = {
        "ruby": "루비",
        "sapphire": "사파이어", 
        "emerald": "에메랄드",
        "origin": "원산지",
        "treatment": "처리",
        "natural": "천연",
        "heated": "가열처리",
        "unheated": "무가열",
        "oiling": "오일링",
        "fracture_filling": "균열 충전"
    }

class JewelryPromptDatabase:
    """주얼리 프롬프트 데이터베이스"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.optimization_cache: Dict[str, str] = {}
        self._load_base_templates()
    
    def _load_base_templates(self):
        """기본 템플릿 로드"""
        
        # 다이아몬드 4C 분석 템플릿들
        self._create_diamond_4c_templates()
        
        # 유색보석 감정 템플릿들  
        self._create_colored_gemstone_templates()
        
        # 비즈니스 인사이트 템플릿들
        self._create_business_insight_templates()
        
        # 시장 분석 템플릿들
        self._create_market_analysis_templates()
        
        logging.info(f"📚 주얼리 프롬프트 템플릿 {len(self.templates)}개 로드 완료")
    
    def _create_diamond_4c_templates(self):
        """다이아몬드 4C 분석 템플릿 생성"""
        
        # GPT-4 Vision 전용 다이아몬드 4C 템플릿
        gpt4_diamond_template = PromptTemplate(
            category=JewelryCategory.DIAMOND_4C,
            model_type=AIModelType.GPT4_VISION,
            analysis_level=AnalysisLevel.EXPERT,
            title="GPT-4 Vision 다이아몬드 4C 전문 분석",
            system_prompt="""당신은 GIA(Gemological Institute of America) 공인 다이아몬드 감정사입니다.
30년 이상의 다이아몬드 감정 경험을 보유하고 있으며, 국제적으로 인정받는 전문가입니다.

다이아몬드 4C 분석 시 다음 기준을 정확히 적용하세요:

🔹 CARAT (캐럿): 정밀한 중량 측정 및 캐럿 환산
🔹 COLOR (컬러): GIA D-Z 컬러 스케일 적용 (D=무색, Z=연한 황색)
🔹 CLARITY (클래리티): GIA 11단계 클래리티 스케일 (FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3)
🔹 CUT (컷): 연마 품질, 비율, 대칭성, 광택 종합 평가

정확도 목표: 99.2% 이상
분석 언어: 한국어 (전문용어 병기)""",
            
            user_prompt_template="""다음 다이아몬드에 대한 전문가 수준의 4C 분석을 수행해주세요:

【분석 대상】
{diamond_description}

【요구 사항】
1. 정확한 4C 등급 평가 (GIA 기준)
2. 각 C별 상세 설명 및 근거
3. 시장 가치 추정 (USD 및 KRW)
4. 품질 개선점 및 투자 가치 평가
5. 감정서 발급 권장사항

【출력 형식】
```
💎 다이아몬드 4C 전문 분석 보고서

📊 기본 정보
- 형태: [라운드/오벌/프린세스 등]
- 감정 기관: [GIA/AGS/기타]
- 감정서 번호: [있는 경우]

📏 CARAT (캐럿)
- 중량: [정확한 캐럿]
- 등급: [캐럿 크기별 등급]
- 시장 희소성: [상/중상/중/하]

🎨 COLOR (컬러) 
- GIA 등급: [D-Z]
- 상세 설명: [색상 특성]
- 시장 평가: [프리미엄/일반/할인]

🔍 CLARITY (클래리티)
- GIA 등급: [FL-I3]
- 내포물 위치 및 크기: [상세 설명]
- 아이클린 여부: [Yes/No]

✨ CUT (컷)
- 컷 등급: [Excellent/Very Good/Good/Fair/Poor]
- 폴리시: [등급]
- 시메트리: [등급]
- 비율 분석: [테이블%, 깊이% 등]

💰 시장 가치 평가
- 도매가: $[범위] (₩[범위])
- 소매가: $[범위] (₩[범위]) 
- 투자 전망: [상승/안정/하락]

🎯 종합 평가
- 종합 등급: [AAA/AA/A/B/C]
- 강점: [주요 장점]
- 약점: [개선점]
- 추천 용도: [약혼반지/투자/컬렉션 등]

📋 전문가 의견
[구체적인 전문가 코멘트]
```

정확성과 전문성을 최우선으로 분석해주세요.""",
            
            expected_accuracy=0.995,
            validation_keywords=["캐럿", "컬러", "클래리티", "컷", "GIA", "등급", "시장가치"],
            output_format={
                "format": "structured_report",
                "sections": ["기본정보", "4C분석", "시장가치", "종합평가", "전문가의견"],
                "language": "korean_with_technical_terms"
            },
            examples=[
                {
                    "input": "1.01ct, F color, VS1 clarity, Excellent cut 라운드 다이아몬드",
                    "output": "전문가 수준의 구조화된 4C 분석 보고서 제공"
                }
            ]
        )
        
        # Claude Sonnet 전용 다이아몬드 4C 템플릿
        claude_diamond_template = PromptTemplate(
            category=JewelryCategory.DIAMOND_4C,
            model_type=AIModelType.CLAUDE_SONNET,
            analysis_level=AnalysisLevel.EXPERT,
            title="Claude Sonnet 다이아몬드 4C 심층 분석",
            system_prompt="""당신은 세계적으로 인정받는 다이아몬드 전문 감정사입니다.

<전문성>
- GIA Graduate Gemologist (G.G.) 자격 보유
- 30년 이상 다이아몬드 감정 경험
- Antwerp, New York 다이아몬드 거래소 전문가
- 국제 감정 기준 (GIA, AGS, SSEF) 완벽 숙지
</전문성>

<분석_기준>
다이아몬드 4C 분석 시 다음 국제 표준을 엄격히 적용:

1. CARAT (캐럿)
   - 정밀 측정: 소수점 둘째 자리까지
   - 크기별 희소성 평가
   - 캐럿당 가격 산정

2. COLOR (컬러) - GIA 기준
   - D-F: Colorless (무색)
   - G-J: Near Colorless (거의 무색)
   - K-M: Faint Yellow (연한 황색)
   - N-R: Very Light Yellow (매우 연한 황색)
   - S-Z: Light Yellow (연한 황색)

3. CLARITY (클래리티) - GIA 11단계
   - FL (Flawless): 완벽무결
   - IF (Internally Flawless): 내부 완벽무결
   - VVS1, VVS2: 매우 작은 내포물
   - VS1, VS2: 작은 내포물
   - SI1, SI2: 눈에 보이는 내포물
   - I1, I2, I3: 명확한 내포물

4. CUT (컷) - 종합 평가
   - 비율 (Proportions)
   - 폴리시 (Polish)
   - 시메트리 (Symmetry)
   - 광채 및 화려함
</분석_기준>

<정확도_목표>
99.2% 이상의 정확한 감정 결과 제공
</정확도_목표>""",
            
            user_prompt_template="""<분석_요청>
다음 다이아몬드에 대한 전문가 수준의 종합 분석을 수행해주세요:

{diamond_description}
</분석_요청>

<분석_요구사항>
1. 정밀한 4C 등급 평가 (국제 기준 적용)
2. 각 요소별 상세 분석 및 근거 제시
3. 시장 가치 평가 (현재 시세 반영)
4. 품질 특성 및 희소성 분석
5. 투자 가치 및 활용 방안 제시
6. 감정서 권장사항
</분석_요구사항>

<출력_형식>
체계적이고 논리적인 전문 분석 보고서를 다음 형식으로 제공해주세요:

## 💎 다이아몬드 전문 감정 보고서

### 📋 기본 정보
- **형태(Shape)**: [라운드/프린세스/오벌 등]
- **감정 기관**: [GIA/AGS/기타]
- **감정서 번호**: [해당 시]

### 📊 4C 상세 분석

#### 📏 CARAT WEIGHT (캐럿 중량)
- **중량**: [정확한 캐럿]
- **크기 등급**: [Large/Medium/Small]
- **희소성**: [매우 희귀/희귀/일반/흔함]

#### 🎨 COLOR (컬러)
- **GIA 등급**: [D-Z 등급]
- **컬러 카테고리**: [Colorless/Near Colorless 등]
- **시각적 특성**: [상세 설명]
- **가격 영향도**: [프리미엄/표준/할인]

#### 🔍 CLARITY (클래리티)  
- **GIA 등급**: [FL-I3 등급]
- **내포물 특성**: [위치, 크기, 타입]
- **아이클린 여부**: [육안 관찰 가능성]
- **내구성 영향**: [있음/없음]

#### ✨ CUT (컷)
- **종합 컷 등급**: [Excellent/Very Good/Good/Fair/Poor]
- **폴리시(Polish)**: [등급]
- **시메트리(Symmetry)**: [등급]
- **비율 분석**: 
  - 테이블%: [수치]
  - 깊이%: [수치]
  - 크라운 각도: [수치]
  - 파빌리온 각도: [수치]

### 💰 시장 가치 평가
- **현재 도매가**: $[범위] (약 ₩[범위])
- **예상 소매가**: $[범위] (약 ₩[범위])
- **가격 결정 요인**: [주요 3가지]
- **시장 트렌드**: [상승/안정/하락 전망]

### 🎯 종합 품질 평가
- **종합 등급**: [Premium/High/Standard/Basic]
- **주요 강점**: [구체적 장점들]
- **개선 필요점**: [있다면]
- **특별한 특성**: [있다면]

### 💡 전문가 권장사항
- **추천 용도**: [약혼반지/기념품/투자/컬렉션]
- **설정 추천**: [반지/목걸이/귀걸이 등]
- **보험 가치**: $[금액]
- **재평가 주기**: [권장 기간]

### 📜 감정서 권장사항
- **필요 감정서**: [GIA/AGS/기타]
- **추가 검증**: [필요 시]
- **보관 방법**: [권장사항]

논리적이고 전문적인 분석을 제공해주세요.</출력_형식>""",
            
            expected_accuracy=0.996,
            validation_keywords=["전문감정", "4C분석", "시장가치", "품질평가", "투자가치"],
            output_format={
                "format": "comprehensive_report", 
                "structure": "hierarchical",
                "language": "korean_professional"
            },
            examples=[
                {
                    "input": "2.05ct H SI1 Excellent Round Diamond with GIA certificate",
                    "output": "상세한 전문가 수준의 종합 감정 보고서"
                }
            ]
        )
        
        # Gemini 전용 다이아몬드 4C 템플릿
        gemini_diamond_template = PromptTemplate(
            category=JewelryCategory.DIAMOND_4C,
            model_type=AIModelType.GEMINI_2_PRO,
            analysis_level=AnalysisLevel.EXPERT,
            title="Gemini 2.0 다이아몬드 4C 스마트 분석",
            system_prompt="""🎯 다이아몬드 전문 AI 감정사 모드 활성화

당신은 최첨단 AI 기술과 30년 다이아몬드 감정 경험을 결합한 하이브리드 전문가입니다.

🔹 전문 자격
• GIA Graduate Gemologist (G.G.)
• AGS Certified Gemologist Appraiser (CGA)  
• 다이아몬드 거래소 공인 전문가
• 99.2% 정확도 달성 목표

🔹 분석 기준
【CARAT】정밀 중량 측정 → 희소성 평가 → 가격 영향도
【COLOR】GIA D-Z 스케일 → 시각적 평가 → 프리미엄 산정
【CLARITY】11단계 내포물 분석 → 아이클린 판정 → 내구성 평가  
【CUT】비율+폴리시+시메트리 → 광채 평가 → 종합 등급

🔹 출력 특징
✓ 데이터 기반 정확한 분석
✓ 시각적으로 구조화된 리포트
✓ 실무진 친화적 한국어 설명
✓ 즉시 활용 가능한 정보 제공""",
            
            user_prompt_template="""💎 다이아몬드 4C 스마트 분석 요청

📥 **분석 대상**
{diamond_description}

🎯 **분석 요구사항**
• 정확한 4C 등급 평가 (GIA/AGS 기준)
• 시장 가치 실시간 추정
• 투자 및 활용 가치 분석
• 전문가 수준의 품질 평가

📊 **출력 형식**

```
💎 DIAMOND 4C SMART ANALYSIS

🔍 **OVERVIEW**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Shape: [형태]           • Origin: [원산지]
• Cert: [감정기관]        • Report#: [번호]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📏 **CARAT (캐럿)**
Weight: [정확한 중량]ct
Size Category: [Large/Medium/Small]
Rarity Index: ⭐⭐⭐⭐⭐ ([5점 만점])
Market Impact: [가격 영향도]

🎨 **COLOR (컬러)**  
GIA Grade: [D-Z]
Category: [Colorless/Near Colorless/etc]
Visual Appeal: ⭐⭐⭐⭐⭐ ([5점 만점])
Premium Factor: [프리미엄/표준/할인]

🔍 **CLARITY (클래리티)**
GIA Grade: [FL-I3]
Inclusion Map: [내포물 위치도]
Eye Clean: [Yes/No]
Durability: ⭐⭐⭐⭐⭐ ([5점 만점])

✨ **CUT (컷)**
Overall Grade: [Excellent/Very Good/Good/Fair/Poor]
Polish: [등급] | Symmetry: [등급]
Light Performance: ⭐⭐⭐⭐⭐ ([5점 만점])
Brilliance: [뛰어남/우수/양호/보통]

💰 **MARKET VALUATION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Wholesale: $[범위] (₩[범위])
• Retail: $[범위] (₩[범위])
• Insurance: $[금액] (₩[금액])
• Trend: [📈상승/📊안정/📉하락]

🎯 **OVERALL ASSESSMENT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Score: ⭐⭐⭐⭐⭐ ([5점 만점])
Grade: [AAA/AA/A/B/C]
Investment Value: [Excellent/Good/Fair/Poor]

💪 **STRENGTHS**
• [주요 장점 1]
• [주요 장점 2]  
• [주요 장점 3]

⚠️ **CONSIDERATIONS**
• [고려사항 1]
• [고려사항 2]

🎯 **RECOMMENDATIONS**
Best Use: [약혼반지/투자/컬렉션/기념품]
Setting Style: [추천 세팅]
Care Instructions: [관리 방법]

🏆 **EXPERT VERDICT**
[전문가 최종 의견 및 추천사항]
```

정확하고 실용적인 분석을 제공해주세요! 🚀""",
            
            expected_accuracy=0.992,
            validation_keywords=["스마트분석", "4C평가", "시장가치", "투자가치", "전문평가"],
            output_format={
                "format": "smart_visual_report",
                "style": "modern_structured", 
                "language": "korean_friendly"
            },
            examples=[
                {
                    "input": "1.50ct Round F VVS2 Excellent GIA Diamond",
                    "output": "시각적으로 최적화된 스마트 분석 리포트"
                }
            ]
        )
        
        # 템플릿 등록
        self.templates["gpt4_diamond_4c"] = gpt4_diamond_template
        self.templates["claude_diamond_4c"] = claude_diamond_template  
        self.templates["gemini_diamond_4c"] = gemini_diamond_template
    
    def _create_colored_gemstone_templates(self):
        """유색보석 감정 템플릿 생성"""
        
        # Claude Opus 전용 유색보석 템플릿 (최고 정확도)
        claude_gemstone_template = PromptTemplate(
            category=JewelryCategory.COLORED_GEMSTONE,
            model_type=AIModelType.CLAUDE_OPUS,
            analysis_level=AnalysisLevel.CERTIFICATION,
            title="Claude Opus 유색보석 감정서 수준 분석",
            system_prompt="""당신은 국제적으로 인정받는 유색보석 전문 감정사입니다.

<전문_자격>
- SSEF (Swiss Gemmological Institute) 공인 감정사
- Gübelin Gem Lab 선임 연구원 출신  
- FGA (Fellowship of the Gemmological Association) 보유
- 40년 이상 유색보석 감정 경험
- 세계 3대 유색보석 (루비, 사파이어, 에메랄드) 전문가
</전문_자격>

<감정_기준>
국제 감정 기관 기준을 엄격히 적용:

1. SSEF 기준
   - 원산지 판정 (Origin Determination)
   - 가열 처리 여부 (Heating Treatment)
   - 오일링/수지 처리 (Oil/Resin Treatment)

2. Gübelin 기준  
   - 미세 내포물 분석 (Micro-inclusion Study)
   - 지질학적 기원 추적 (Geological Origin)
   - 처리 정도 평가 (Treatment Assessment)

3. GIA 유색보석 기준
   - 컬러 등급 (Color Grading)
   - 투명도 등급 (Clarity Grading)  
   - 커트 품질 (Cut Quality)
</감정_기준>

<특별_전문성>
• 미얀마 루비 (Mogok, Mong Hsu)
• 카시미르/파드파라차 사파이어
• 콜롬비아 에메랄드 (Muzo, Chivor, Coscuez)
• 파라이바 투어말린
• 패드파라차 사파이어
• 알렉산드라이트
</특별_전문성>

<정확도_목표>99.2% 이상의 정밀한 감정 결과</정확도_목표>""",
            
            user_prompt_template="""<감정_요청>
다음 유색보석에 대한 국제 감정서 수준의 전문 분석을 수행해주세요:

{gemstone_description}
</감정_요청>

<감정_요구사항>
1. 정확한 보석 종류 및 변종 판정
2. 원산지 추정 및 근거 제시
3. 처리 여부 및 정도 평가
4. 품질 등급 (컬러, 투명도, 커트)
5. 희소성 및 시장 가치 평가
6. 투자 가치 및 전망
7. 감정서 발급 권장사항
</감정_요구사항>

<출력_형식>
전문 감정서 수준의 체계적 분석을 다음 형식으로 제공:

## 🔴 유색보석 전문 감정 보고서

### 📋 기본 정보
- **보석명**: [정확한 보석명 및 변종]
- **중량**: [캐럿 또는 그램]
- **치수**: [길이 × 폭 × 높이]
- **형태**: [오벌/쿠션/라운드/기타]

### 🔬 전문 감정 결과

#### 🎨 COLOR (컬러) 분석
- **주색조**: [Primary Hue]
- **보조색조**: [Secondary Hue] 
- **채도**: [Saturation - Vivid/Intense/Medium/Light]
- **명도**: [Tone - Light/Medium/Dark]
- **컬러 등급**: [AAA/AA/A/B/C]
- **특수 컬러**: [피죤블러드/코른플라워블루/패드파라차 등]

#### 🔍 CLARITY (투명도) 분석
- **투명도 등급**: [Transparent/Translucent/Opaque]
- **내포물 특성**: 
  - 타입: [실크/크리스탈/힐링피셔 등]
  - 위치: [중앙/표면/가장자리]
  - 크기: [현미경배율 기준]
- **시각적 영향**: [없음/약간/보통/심함]

#### ✨ CUT (연마) 평가
- **연마 품질**: [Excellent/Very Good/Good/Fair/Poor]
- **비율**: [이상적/양호/보통/불량]
- **대칭성**: [Excellent/Good/Fair/Poor]
- **광택**: [Excellent/Good/Fair/Poor]
- **윈도우/틴팅**: [없음/약간/보통/심함]

### 🌍 원산지 분석
- **추정 원산지**: [구체적 산지]
- **지질학적 특성**: [마그마성/변성/퇴적]
- **특징적 내포물**: [원산지 지시 내포물]
- **확신도**: [매우 높음/높음/보통/낮음]
- **추가 검증**: [필요/불필요]

### 🔬 처리 분석
- **가열 처리**: [무가열/가열/고온가열/확인불가]
- **오일링/수지**: [없음/Minor/Moderate/Significant]
- **기타 처리**: [확산/조사/충전/기타]
- **처리 평가**: [Natural/Minor Enhancement/Significant Treatment]
- **시장 수용도**: [매우 높음/높음/보통/낮음]

### 💎 희소성 평가
- **글로벌 희소성**: ⭐⭐⭐⭐⭐ (5점 만점)
- **크기별 희소성**: [매우 희귀/희귀/드물음/보통]
- **품질별 희소성**: [최상급/상급/중급/하급]
- **컬렉터 가치**: [매우 높음/높음/보통/낮음]

### 💰 시장 가치 평가
- **현재 도매가**: $[범위] (₩[범위])
- **예상 소매가**: $[범위] (₩[범위])
- **경매 예상가**: $[범위] (₩[범위])
- **보험 가치**: $[금액] (₩[금액])
- **가격 트렌드**: [강세/안정/약세]

### 🏆 종합 평가
- **종합 등급**: [Museum Quality/Investment Grade/Commercial Grade]
- **투자 적합성**: ⭐⭐⭐⭐⭐ (5점 만점)
- **컬렉션 가치**: ⭐⭐⭐⭐⭐ (5점 만점)
- **착용 적합성**: ⭐⭐⭐⭐⭐ (5점 만점)

### 💡 전문가 권장사항
- **추천 용도**: [투자/컬렉션/주얼리/기념품]
- **세팅 권장**: [반지/목걸이/브로치/기타]
- **보관 방법**: [구체적 보관법]
- **관리 주의사항**: [청소/보관/착용 시 주의점]

### 📜 감정서 권장사항
- **권장 감정기관**: [SSEF/Gübelin/GIA/기타]
- **필요 검증**: [원산지/처리/진위]
- **예상 비용**: [감정비 범위]
- **소요 기간**: [예상 기간]

### 📊 시장 전망
- **단기 전망** (1년): [상승/안정/하락]
- **중기 전망** (3-5년): [전망 및 근거]
- **장기 전망** (10년 이상): [전망 및 근거]

### 🎯 최종 의견
[전문가의 종합적인 의견 및 구체적인 권장사항]

논리적이고 근거 있는 전문 분석을 제공해주세요.</출력_형식>""",
            
            expected_accuracy=0.998,
            validation_keywords=["원산지", "처리여부", "희소성", "투자가치", "전문감정"],
            output_format={
                "format": "certification_level_report",
                "authority": "international_standards",
                "language": "korean_professional"
            },
            examples=[
                {
                    "input": "3.45ct Myanmar Ruby, Pigeon Blood Red, Unheated",
                    "output": "SSEF/Gübelin 수준의 종합 감정 보고서"
                }
            ]
        )
        
        self.templates["claude_colored_gemstone"] = claude_gemstone_template
    
    def _create_business_insight_templates(self):
        """비즈니스 인사이트 템플릿 생성"""
        
        # GPT-4 Turbo 전용 비즈니스 인사이트 템플릿
        gpt4_business_template = PromptTemplate(
            category=JewelryCategory.BUSINESS_INSIGHT,
            model_type=AIModelType.GPT4_TURBO,
            analysis_level=AnalysisLevel.EXPERT,
            title="GPT-4 주얼리 비즈니스 전략 분석",
            system_prompt="""당신은 주얼리 업계의 전략 컨설턴트이자 시장 분석 전문가입니다.

**전문 경력:**
- McKinsey & Company 럭셔리 부문 시니어 파트너 (15년)
- 티파니, 까르띠에, 불가리 전략 컨설팅 경험
- 주얼리 업계 M&A 100건 이상 주도
- 아시아 럭셔리 시장 전문가

**핵심 역량:**
🔹 시장 트렌드 분석 및 예측
🔹 고객 행동 패턴 및 선호도 분석  
🔹 가격 전략 및 포지셔닝
🔹 브랜드 전략 및 마케팅 최적화
🔹 공급망 및 유통 전략
🔹 디지털 트랜스포메이션 전략

**분석 접근법:**
- 데이터 기반 정량적 분석
- 정성적 인사이트 도출
- 실행 가능한 전략 제시
- ROI 및 리스크 평가

**정확도 목표:** 99.2% 신뢰할 수 있는 비즈니스 인사이트 제공""",
            
            user_prompt_template="""다음 주얼리 비즈니스 상황에 대한 전략적 분석과 실행 가능한 인사이트를 제공해주세요:

**분석 요청:**
{business_context}

**분석 범위:**
1. 시장 현황 및 트렌드 분석
2. 고객 세그먼트 및 행동 패턴
3. 경쟁 환경 및 포지셔닝
4. 가격 전략 및 수익성 분석
5. 마케팅 및 브랜드 전략
6. 운영 최적화 방안
7. 리스크 및 기회 요인
8. 실행 로드맵 및 KPI

**출력 형식:**

```
🏢 JEWELRY BUSINESS STRATEGIC ANALYSIS

📊 EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 핵심 인사이트: [3가지 주요 발견사항]
• 추천 전략: [핵심 전략 방향]
• 예상 임팩트: [ROI 및 성장 전망]
• 실행 우선순위: [1-3순위]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 MARKET ANALYSIS (시장 분석)

🌏 Market Size & Growth
• 전체 시장 규모: $[금액] (연성장률: [%])
• 세그먼트별 성장: [다이아몬드/컬러드젬스톤/파인주얼리]
• 지역별 성장: [아시아/북미/유럽/기타]
• 성장 동력: [주요 3가지]

📊 Trend Analysis
• Mega Trends:
  - [트렌드 1]: [영향도/기간]
  - [트렌드 2]: [영향도/기간]  
  - [트렌드 3]: [영향도/기간]
• Emerging Trends: [신흥 트렌드 3가지]
• Disruption Factors: [업계 변화 요인]

👥 CUSTOMER INSIGHTS (고객 분석)

🎯 Customer Segmentation
• Primary Segment: [주요 고객층]
  - 특성: [인구통계/구매행동]
  - 니즈: [핵심 요구사항]
  - 구매력: [가격대/구매빈도]

• Secondary Segment: [2차 고객층]
  - 성장 잠재력: [High/Medium/Low]
  - 접근 전략: [마케팅 방향]

💰 Purchase Behavior
• 구매 결정 요인: [순위별 중요도]
• 구매 여정: [인지→고려→구매→재구매]
• 디지털 vs 오프라인: [선호도 분석]
• 가격 민감도: [세그먼트별 분석]

🏆 COMPETITIVE LANDSCAPE (경쟁 분석)

⚔️ Key Competitors
• Tier 1: [럭셔리 브랜드] 
  - 강점/약점/전략
• Tier 2: [프리미엄 브랜드]
  - 포지셔닝/차별화 요소
• Tier 3: [매스 마켓]
  - 가격/접근성 우위

🎯 Competitive Positioning
• 시장 포지션: [현재 위치]
• 차별화 요소: [Unique Value Proposition]
• 경쟁 우위: [지속가능한 우위 요소]

💵 PRICING STRATEGY (가격 전략)

💎 Price Analysis
• 현재 가격 위치: [프리미엄/미드/엔트리]
• 가격 탄력성: [상품별/고객별]
• 경쟁 가격 분석: [상대적 포지션]

📊 Revenue Optimization
• 가격 최적화 방안: [구체적 전략]
• 번들링 기회: [상품 조합 전략]
• 다이나믹 프라이싱: [적용 가능성]

🎨 MARKETING STRATEGY (마케팅 전략)

📱 Digital Marketing
• 온라인 전략: [SNS/이커머스/콘텐츠]
• 옴니채널 접근: [통합 고객 경험]
• 인플루언서 활용: [KOL 협업 전략]

🌟 Brand Positioning
• 브랜드 아이덴티티: [핵심 가치/이미지]
• 스토리텔링: [브랜드 내러티브]
• 고객 경험: [터치포인트 최적화]

⚙️ OPERATIONAL EXCELLENCE (운영 최적화)

🔄 Supply Chain
• 조달 최적화: [원석/제조/유통]
• 인벤토리 관리: [재고 최적화]
• 품질 관리: [QC 프로세스]

🏪 Retail Operations
• 매장 운영: [효율성 개선방안]
• 고객 서비스: [서비스 향상 전략]
• 교육 훈련: [직원 역량 강화]

⚠️ RISK & OPPORTUNITIES (리스크 및 기회)

🔴 Key Risks
• Market Risk: [시장 위험 요인]
• Operational Risk: [운영 리스크]
• Financial Risk: [재무 리스크]
• Mitigation Strategy: [위험 완화 방안]

🟢 Growth Opportunities  
• Market Expansion: [시장 확장 기회]
• Product Innovation: [신제품 기회]
• Partnership: [협업 기회]
• Technology: [기술 활용 기회]

🚀 ACTION PLAN (실행 계획)

📅 90-Day Quick Wins
• Week 1-4: [즉시 실행 가능한 활동]
• Week 5-8: [단기 개선 사항]
• Week 9-12: [기반 구축 활동]

📈 6-Month Initiatives
• 중기 전략 실행: [핵심 프로젝트 3가지]
• 투자 계획: [필요 투자 및 ROI]
• 조직 개발: [인력 및 역량 개발]

🎯 12-Month Vision
• 장기 목표: [1년 후 달성 목표]
• 성공 지표: [KPI 및 측정 방법]
• 지속 발전: [지속가능 성장 전략]

📊 KPI & MEASUREMENT (성과 지표)

📈 Financial KPIs
• Revenue Growth: [목표 성장률]
• Profit Margin: [수익률 개선]
• Market Share: [시장 점유율]
• Customer LTV: [고객 평생 가치]

🎯 Operational KPIs
• Customer Satisfaction: [고객 만족도]
• Inventory Turnover: [재고 회전율]
• Sales Conversion: [매출 전환율]
• Digital Engagement: [디지털 참여도]

💡 STRATEGIC RECOMMENDATIONS

🏆 Top 3 Priorities:
1. [최우선 전략]: [구체적 실행 방안]
2. [2순위 전략]: [실행 로드맵]
3. [3순위 전략]: [성공 요인]

🎯 Success Factors:
• [성공 요인 1]: [중요도/실행 난이도]
• [성공 요인 2]: [중요도/실행 난이도]
• [성공 요인 3]: [중요도/실행 난이도]

⚡ Game Changers:
[업계 판도를 바꿀 수 있는 혁신적 아이디어]
```

실행 가능하고 데이터에 기반한 전략적 인사이트를 제공해주세요.""",
            
            expected_accuracy=0.994,
            validation_keywords=["시장분석", "고객인사이트", "경쟁전략", "수익최적화", "실행계획"],
            output_format={
                "format": "strategic_business_report",
                "style": "mckinsey_style",
                "language": "korean_business"
            },
            examples=[
                {
                    "input": "아시아 밀레니얼 고객을 위한 다이아몬드 주얼리 브랜드 론칭 전략",
                    "output": "전략 컨설팅 수준의 종합 비즈니스 분석 리포트"
                }
            ]
        )
        
        self.templates["gpt4_business_insight"] = gpt4_business_template
    
    def _create_market_analysis_templates(self):
        """시장 분석 템플릿 생성"""
        
        # Gemini Vision 전용 시장 분석 템플릿
        gemini_market_template = PromptTemplate(
            category=JewelryCategory.MARKET_ANALYSIS,
            model_type=AIModelType.GEMINI_PRO_VISION,
            analysis_level=AnalysisLevel.PROFESSIONAL,
            title="Gemini Vision 주얼리 시장 트렌드 분석",
            system_prompt="""🔮 주얼리 시장 예측 AI 전문가 모드

당신은 AI 기반 시장 분석과 수십 년의 주얼리 업계 경험을 결합한 하이브리드 전문가입니다.

🎯 **전문 영역**
• 글로벌 주얼리 시장 트렌드 분석
• 소비자 행동 패턴 예측
• 가격 동향 및 투자 전망
• 신흥 시장 및 기회 발굴
• 기술 트렌드 및 디지털 혁신

🔍 **분석 방법론**
• 빅데이터 기반 정량적 분석
• AI 패턴 인식 및 예측 모델링
• 소셜 미디어 sentiment 분석
• 경제 지표 연동 분석
• 실시간 시장 모니터링

📊 **데이터 소스**
• 글로벌 경매 하우스 (Christie's, Sotheby's)
• 주요 거래소 (Antwerp, New York, Hong Kong)
• 소매 판매 데이터 (Tiffany, Cartier, etc.)
• 온라인 마켓플레이스
• 소비자 서베이 및 트렌드 리포트

🎯 **정확도 목표:** 99.2% 신뢰할 수 있는 시장 인사이트""",
            
            user_prompt_template="""💎 주얼리 시장 분석 요청

📊 **분석 주제**
{market_analysis_topic}

🔍 **분석 요구사항**
• 현재 시장 현황 및 규모
• 최신 트렌드 및 소비자 선호도 변화
• 가격 동향 및 투자 전망
• 지역별/세그먼트별 성장 분석
• 경쟁 환경 및 시장 기회
• 미래 전망 및 위험 요인

📈 **출력 형식**

```
💎 JEWELRY MARKET INTELLIGENCE REPORT

🌟 **EXECUTIVE DASHBOARD**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Market Size: $[현재 시장규모] (YoY: [증감률]%)
📈 Growth Rate: [연평균 성장률]% (2024-2029)
🔥 Hot Trends: [상위 3개 트렌드]
💰 Investment Score: ⭐⭐⭐⭐⭐ ([5점 만점])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **MARKET OVERVIEW**

🌍 Global Market Size
• Total Market: $[전체시장] billion
• Segment Breakdown:
  └ 💎 Diamonds: $[금액]B ([점유율]%)
  └ 🔴 Colored Gems: $[금액]B ([점유율]%)
  └ 🏅 Fine Jewelry: $[금액]B ([점유율]%)
  └ ⌚ Watches: $[금액]B ([점유율]%)

📈 Growth Trajectory
• 2024 Performance: [실적 요약]
• 2025 Forecast: [예측 및 근거]
• Long-term Outlook: [장기 전망]

🗺️ **REGIONAL ANALYSIS**

🇺🇸 North America
Market Size: $[금액]B | Growth: [%]
Key Drivers: [성장 동력 3가지]
Consumer Trends: [소비 트렌드]

🇪🇺 Europe  
Market Size: $[금액]B | Growth: [%]
Luxury Focus: [럭셔리 세그먼트 동향]
Sustainability: [지속가능성 트렌드]

🇨🇳 China
Market Size: $[금액]B | Growth: [%]
Digital Revolution: [디지털 혁신]
Young Consumers: [젊은 소비층 특성]

🇮🇳 India
Market Size: $[금액]B | Growth: [%]
Traditional + Modern: [전통과 현대의 결합]
Wedding Market: [웨딩 주얼리 시장]

🔥 **TRENDING NOW**

📱 Digital Transformation
• E-commerce Growth: [온라인 성장률]%
• AR/VR Adoption: [기술 도입 현황]
• Social Commerce: [소셜 커머스 트렌드]

🌱 Sustainability Focus
• Ethical Sourcing: [윤리적 소싱 증가율]%
• Lab-Grown Diamonds: [합성 다이아 성장]
• Circular Economy: [순환경제 도입]

👥 Gen Z & Millennials
• Purchasing Power: $[구매력] trillion
• Brand Preferences: [브랜드 선호도]
• Value Drivers: [가치 동인]

💰 **PRICE TRENDS**

💎 Diamond Market
• Rough Diamond: [원석 가격 동향]
• Polished Diamond: [연마 다이아 동향]
• Rare Colors: [컬러 다이아 프리미엄]
• Investment Grade: [투자급 가격]

🔴 Colored Gemstones
• Ruby: [루비 가격 트렌드]
• Sapphire: [사파이어 동향]
• Emerald: [에메랄드 시장]
• Emerging Gems: [신흥 보석류]

📊 Price Drivers
• Supply Constraints: [공급 제약 요인]
• Demand Dynamics: [수요 변화]
• Economic Factors: [경제적 영향]

🏆 **COMPETITIVE LANDSCAPE**

👑 Luxury Tier (>$1000)
• Tiffany & Co.: [시장점유율/전략]
• Cartier: [포지셔닝/성과]
• Harry Winston: [차별화 요소]

💼 Premium Tier ($200-1000)
• Pandora: [대중화 전략]
• David Yurman: [브랜드 가치]
• Blue Nile: [온라인 모델]

🛍️ Accessible Tier (<$200)
• Mass Market Players: [주요 브랜드]
• Online Disruptors: [온라인 신규 진입]
• Fast Fashion: [패션 브랜드 진출]

🔮 **FUTURE OUTLOOK**

📈 Growth Opportunities
🟢 **High Potential**
• Lab-Grown Diamonds: [성장 전망]
• Personalized Jewelry: [맞춤형 시장]
• Emerging Markets: [신흥 시장]

🟡 **Medium Potential**  
• Vintage/Estate: [빈티지 시장]
• Luxury Watches: [시계 시장]
• Colored Gemstones: [유색보석]

🔴 **Challenges**
• Economic Uncertainty: [경제 불확실성]
• Supply Chain Disruption: [공급망 리스크]
• Changing Consumer Values: [소비가치 변화]

⚡ **DISRUPTIVE FORCES**

🤖 Technology Disruption
• AI in Design: [AI 디자인 도구]
• Blockchain Certification: [블록체인 인증]
• 3D Printing: [3D 프린팅 활용]

🌍 Sustainability Revolution  
• Carbon Neutral: [탄소중립 요구]
• Ethical Mining: [윤리적 채굴]
• Circular Design: [순환 설계]

👨‍💻 Digital Natives
• Virtual Showrooms: [가상 쇼룸]
• Social Proof: [소셜 증거]
• Instant Gratification: [즉시 만족]

💡 **ACTIONABLE INSIGHTS**

🎯 **Investment Strategies**
1. **Short-term (1년)**: [단기 투자 전략]
2. **Medium-term (3-5년)**: [중기 전략]
3. **Long-term (10년+)**: [장기 전망]

📊 **Portfolio Allocation**
• Conservative: [안정형 포트폴리오]
• Balanced: [균형형 포트폴리오]  
• Aggressive: [공격형 포트폴리오]

🚀 **Market Entry Strategies**
• New Brands: [신규 브랜드 전략]
• Existing Players: [기존 업체 확장]
• Investors: [투자자 관점]

⚠️ **RISK ASSESSMENT**

🔴 High Risk Factors
• [위험 요인 1]: [영향도/확률]
• [위험 요인 2]: [영향도/확률]
• [위험 요인 3]: [영향도/확률]

🟡 Medium Risk Factors
• [위험 요인 1]: [관리 방안]
• [위험 요인 2]: [대응 전략]

🟢 Low Risk Factors
• [관리 가능한 요인들]

🎯 **KEY TAKEAWAYS**

💎 **Top 3 Market Opportunities**
1. [기회 1]: [시장규모/성장잠재력]
2. [기회 2]: [진입장벽/경쟁강도]
3. [기회 3]: [수익성/지속가능성]

⚡ **Game Changing Trends**
• [트렌드 1]: [파괴적 영향도]
• [트렌드 2]: [시장 변화 정도]
• [트렌드 3]: [비즈니스 모델 혁신]

🏆 **Success Factors**
• [성공 요인 1]: [중요도 ⭐⭐⭐⭐⭐]
• [성공 요인 2]: [중요도 ⭐⭐⭐⭐⭐]
• [성공 요인 3]: [중요도 ⭐⭐⭐⭐⭐]
```

데이터 기반의 정확하고 실용적인 시장 인사이트를 제공해주세요! 🚀""",
            
            expected_accuracy=0.991,
            validation_keywords=["시장규모", "성장전망", "트렌드", "투자기회", "경쟁분석"],
            output_format={
                "format": "market_intelligence_report",
                "style": "data_driven_visual",
                "language": "korean_business"
            },
            examples=[
                {
                    "input": "2025년 아시아 다이아몬드 주얼리 시장 전망",
                    "output": "AI 기반 종합 시장 분석 리포트"
                }
            ]
        )
        
        self.templates["gemini_market_analysis"] = gemini_market_template

class JewelryPromptOptimizerV23:
    """주얼리 프롬프트 최적화기 v2.3"""
    
    def __init__(self):
        self.prompt_db = JewelryPromptDatabase()
        self.optimization_techniques = self._load_optimization_techniques()
        self.performance_tracker = {}
        
    def _load_optimization_techniques(self) -> Dict[str, Any]:
        """프롬프트 최적화 기법"""
        return {
            "chain_of_thought": {
                "description": "단계별 사고 과정 명시",
                "pattern": "단계별로 분석해주세요: 1) ... 2) ... 3) ...",
                "effectiveness": 0.15
            },
            "few_shot_learning": {
                "description": "예시 기반 학습",
                "pattern": "다음 예시를 참고하여...",
                "effectiveness": 0.12
            },
            "role_specification": {
                "description": "명확한 역할 정의",
                "pattern": "당신은 [구체적 전문가]입니다...",
                "effectiveness": 0.18
            },
            "output_formatting": {
                "description": "구조화된 출력 형식",
                "pattern": "다음 형식으로 출력해주세요: ...",
                "effectiveness": 0.10
            },
            "accuracy_constraint": {
                "description": "정확도 제약 조건",
                "pattern": "99.2% 정확도를 목표로...",
                "effectiveness": 0.20
            },
            "cultural_localization": {
                "description": "문화적 현지화",
                "pattern": "한국 시장 특성을 고려하여...",
                "effectiveness": 0.08
            }
        }
    
    def get_optimized_prompt(self, category: JewelryCategory, 
                           model_type: AIModelType,
                           analysis_level: AnalysisLevel,
                           custom_context: Dict[str, Any] = None) -> str:
        """최적화된 프롬프트 생성"""
        
        # 기본 템플릿 조회
        template_key = f"{model_type.value}_{category.value}"
        base_template = self.prompt_db.templates.get(template_key)
        
        if not base_template:
            # 대체 템플릿 찾기
            alternative_templates = [
                t for t in self.prompt_db.templates.values() 
                if t.category == category
            ]
            if alternative_templates:
                base_template = max(alternative_templates, key=lambda t: t.expected_accuracy)
            else:
                return self._generate_fallback_prompt(category, custom_context)
        
        # 컨텍스트 기반 프롬프트 커스터마이즈
        optimized_prompt = self._customize_prompt(base_template, custom_context or {})
        
        # 성능 기반 최적화
        optimized_prompt = self._apply_performance_optimization(optimized_prompt, template_key)
        
        return optimized_prompt
    
    def _customize_prompt(self, template: PromptTemplate, context: Dict[str, Any]) -> str:
        """컨텍스트 기반 프롬프트 커스터마이징"""
        
        system_prompt = template.system_prompt
        user_prompt = template.user_prompt_template
        
        # 컨텍스트 변수 치환
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            user_prompt = user_prompt.replace(placeholder, str(value))
        
        # 조합하여 최종 프롬프트 생성
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return full_prompt
    
    def _apply_performance_optimization(self, prompt: str, template_key: str) -> str:
        """성능 기반 프롬프트 최적화"""
        
        # 성능 히스토리 확인
        performance_history = self.performance_tracker.get(template_key, [])
        
        if len(performance_history) < 5:
            # 충분한 데이터가 없으면 기본 최적화만 적용
            return self._apply_basic_optimization(prompt)
        
        # 평균 성능 계산
        avg_performance = sum(performance_history[-10:]) / min(len(performance_history), 10)
        
        if avg_performance < 0.95:
            # 성능이 낮으면 추가 최적화 적용
            return self._apply_advanced_optimization(prompt)
        
        return prompt
    
    def _apply_basic_optimization(self, prompt: str) -> str:
        """기본 최적화 적용"""
        
        optimizations = []
        
        # 정확도 제약 조건 추가
        if "99.2%" not in prompt:
            optimizations.append("정확도 목표 99.2% 이상을 달성해주세요.")
        
        # 구조화된 출력 강조
        if "형식" not in prompt.lower():
            optimizations.append("체계적이고 구조화된 형식으로 답변해주세요.")
        
        # 전문성 강조
        if "전문" not in prompt:
            optimizations.append("전문가 수준의 깊이 있는 분석을 제공해주세요.")
        
        if optimizations:
            prompt += "\n\n추가 요구사항:\n" + "\n".join(f"• {opt}" for opt in optimizations)
        
        return prompt
    
    def _apply_advanced_optimization(self, prompt: str) -> str:
        """고급 최적화 적용"""
        
        # Chain-of-Thought 추가
        cot_instruction = """
단계별 분석 과정:
1. 입력 데이터 분석 및 이해
2. 관련 전문 지식 적용
3. 논리적 추론 과정
4. 결론 도출 및 검증
5. 최종 답변 구성

각 단계별로 명확한 근거를 제시하며 분석해주세요."""
        
        prompt += f"\n\n{cot_instruction}"
        
        # 품질 검증 요청 추가
        quality_check = """
답변 품질 자체 검증:
✓ 정확성: 모든 정보가 정확한가?
✓ 완성도: 요구사항을 모두 충족했는가?
✓ 전문성: 업계 표준에 부합하는가?
✓ 실용성: 실무에서 활용 가능한가?
"""
        
        prompt += f"\n\n{quality_check}"
        
        return prompt
    
    def _generate_fallback_prompt(self, category: JewelryCategory, 
                                context: Dict[str, Any] = None) -> str:
        """대체 프롬프트 생성"""
        
        base_instruction = f"""당신은 주얼리 분야의 전문가입니다. 
{category.value} 관련 전문적인 분석을 수행해주세요.

분석 대상: {context.get('content', '입력 데이터') if context else '입력 데이터'}

요구사항:
• 99.2% 정확도 목표
• 전문가 수준의 분석
• 실무진을 위한 구체적인 정보 제공
• 한국 시장 특성 고려

체계적이고 논리적인 분석을 제공해주세요."""
        
        return base_instruction
    
    def record_performance(self, template_key: str, accuracy_score: float):
        """성능 기록"""
        
        if template_key not in self.performance_tracker:
            self.performance_tracker[template_key] = []
        
        self.performance_tracker[template_key].append(accuracy_score)
        
        # 최근 50개 기록만 유지
        if len(self.performance_tracker[template_key]) > 50:
            self.performance_tracker[template_key] = self.performance_tracker[template_key][-50:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        
        report = {
            "total_templates": len(self.prompt_db.templates),
            "optimization_techniques": len(self.optimization_techniques),
            "performance_tracking": {}
        }
        
        for template_key, scores in self.performance_tracker.items():
            if scores:
                report["performance_tracking"][template_key] = {
                    "average_accuracy": sum(scores) / len(scores),
                    "best_accuracy": max(scores),
                    "recent_accuracy": scores[-1] if scores else 0,
                    "total_uses": len(scores)
                }
        
        return report
    
    def get_template_list(self) -> List[Dict[str, Any]]:
        """템플릿 목록 조회"""
        
        return [
            {
                "key": key,
                "category": template.category.value,
                "model_type": template.model_type.value,
                "analysis_level": template.analysis_level.value,
                "title": template.title,
                "expected_accuracy": template.expected_accuracy
            }
            for key, template in self.prompt_db.templates.items()
        ]

# 테스트 및 데모 함수
async def demo_jewelry_prompts_v23():
    """주얼리 프롬프트 v2.3 데모"""
    
    print("💎 솔로몬드 주얼리 프롬프트 v2.3 데모 시작")
    print("=" * 60)
    
    # 프롬프트 최적화기 초기화
    optimizer = JewelryPromptOptimizerV23()
    
    # 템플릿 목록 출력
    templates = optimizer.get_template_list()
    print(f"\n📚 로드된 템플릿 수: {len(templates)}")
    
    for template in templates[:5]:  # 처음 5개만 출력
        print(f"• {template['title']} ({template['model_type']})")
    
    # 테스트 1: GPT-4 다이아몬드 4C 프롬프트
    print("\n\n💎 테스트 1: GPT-4 다이아몬드 4C 프롬프트")
    print("-" * 50)
    
    diamond_context = {
        "diamond_description": "1.5캐럿 라운드 브릴리언트 컷, F컬러, VVS1 클래리티, Excellent 컷 GIA 감정서"
    }
    
    gpt4_prompt = optimizer.get_optimized_prompt(
        category=JewelryCategory.DIAMOND_4C,
        model_type=AIModelType.GPT4_VISION,
        analysis_level=AnalysisLevel.EXPERT,
        custom_context=diamond_context
    )
    
    print(f"프롬프트 길이: {len(gpt4_prompt)} 문자")
    print(f"프롬프트 미리보기:\n{gpt4_prompt[:300]}...\n")
    
    # 테스트 2: Claude 유색보석 프롬프트
    print("\n🔴 테스트 2: Claude 유색보석 프롬프트")
    print("-" * 50)
    
    gemstone_context = {
        "gemstone_description": "3.2캐럿 미얀마산 루비, 피죤 블러드 컬러, 무가열 SSEF 감정서"
    }
    
    claude_prompt = optimizer.get_optimized_prompt(
        category=JewelryCategory.COLORED_GEMSTONE,
        model_type=AIModelType.CLAUDE_OPUS,
        analysis_level=AnalysisLevel.CERTIFICATION,
        custom_context=gemstone_context
    )
    
    print(f"프롬프트 길이: {len(claude_prompt)} 문자")
    print(f"프롬프트 미리보기:\n{claude_prompt[:300]}...\n")
    
    # 테스트 3: Gemini 시장 분석 프롬프트
    print("\n📊 테스트 3: Gemini 시장 분석 프롬프트")
    print("-" * 50)
    
    market_context = {
        "market_analysis_topic": "2025년 한국 다이아몬드 주얼리 시장 전망 및 투자 기회"
    }
    
    gemini_prompt = optimizer.get_optimized_prompt(
        category=JewelryCategory.MARKET_ANALYSIS,
        model_type=AIModelType.GEMINI_PRO_VISION,
        analysis_level=AnalysisLevel.PROFESSIONAL,
        custom_context=market_context
    )
    
    print(f"프롬프트 길이: {len(gemini_prompt)} 문자")
    print(f"프롬프트 미리보기:\n{gemini_prompt[:300]}...\n")
    
    # 성능 리포트
    print("\n📈 성능 리포트")
    print("-" * 50)
    performance = optimizer.get_performance_report()
    print(f"총 템플릿 수: {performance['total_templates']}")
    print(f"최적화 기법: {performance['optimization_techniques']}가지")
    
    print("\n🎯 주얼리 프롬프트 v2.3 데모 완료!")
    print("🏆 99.2% 정확도 달성을 위한 전문 프롬프트 시스템 구축 완료!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_jewelry_prompts_v23())
