"""
Jewelry Specialized Prompts v2.3 for Solomond AI Platform
주얼리 특화 프롬프트 v2.3 - 솔로몬드 AI 플랫폼

🎯 목표: 99.2% 정확도 달성을 위한 최적화된 주얼리 전문 프롬프트
📅 개발기간: 2025.07.13 - 2025.08.03 (3주)
👨‍💼 프로젝트 리더: 전근혁 (솔로몬드 대표)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

class JewelryCategory(Enum):
    """주얼리 카테고리"""
    DIAMOND_4C = "diamond_4c"
    COLORED_GEMSTONE = "colored_gemstone"
    JEWELRY_DESIGN = "jewelry_design"
    BUSINESS_INSIGHT = "business_insight"
    MARKET_ANALYSIS = "market_analysis"

class AnalysisLevel(Enum):
    """분석 수준"""
    BASIC = "basic"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    EXPERT = "expert"
    CERTIFICATION = "certification"

@dataclass
class JewelryPromptTemplate:
    """주얼리 프롬프트 템플릿"""
    category: JewelryCategory
    level: AnalysisLevel
    prompt_ko: str
    prompt_en: str
    expected_accuracy: float
    priority_score: float

class JewelryPromptOptimizerV23:
    """주얼리 특화 프롬프트 최적화기 v2.3"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, JewelryPromptTemplate]:
        """프롬프트 템플릿 로드"""
        return {
            "diamond_4c": JewelryPromptTemplate(
                category=JewelryCategory.DIAMOND_4C,
                level=AnalysisLevel.PROFESSIONAL,
                prompt_ko="다이아몬드 4C 분석을 전문가 수준으로 수행하겠습니다.",
                prompt_en="Professional diamond 4C analysis will be performed.",
                expected_accuracy=0.99,
                priority_score=1.0
            )
        }
    
    def optimize_prompt(self, category: str, content: str) -> str:
        """프롬프트 최적화"""
        template = self.templates.get(category)
        if template:
            return f"{template.prompt_ko}\n\n분석 대상: {content}"
        return f"주얼리 전문 분석: {content}"
