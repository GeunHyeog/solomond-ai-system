"""
💎 솔로몬드 AI v2.1.3 - 주얼리 특화 AI 모델
주얼리 업계 전문 지식을 활용한 AI 기반 품질 평가 및 감정 시스템

주요 기능:
- 다이아몬드 4C (Carat, Cut, Color, Clarity) AI 평가
- 보석 품질 자동 감정 및 등급 산정
- 가격 예측 AI 모델
- 시장 트렌드 분석 및 예측
- 주얼리 진품/모조품 판별 AI
- 맞춤형 추천 시스템
"""

import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import re
from collections import defaultdict, Counter

# 주얼리 업계 전문 상수 및 데이터
DIAMOND_COLOR_GRADES = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DIAMOND_CLARITY_GRADES = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1', 'I2', 'I3']
DIAMOND_CUT_GRADES = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
GEMSTONE_TYPES = ['Diamond', 'Ruby', 'Sapphire', 'Emerald', 'Topaz', 'Amethyst', 'Aquamarine', 'Pearl', 'Opal', 'Garnet']

@dataclass
class DiamondCharacteristics:
    """다이아몬드 특성"""
    carat: float
    cut_grade: str
    color_grade: str
    clarity_grade: str
    polish: str
    symmetry: str
    fluorescence: str
    measurements: Dict[str, float]  # length, width, depth
    table_percentage: float
    depth_percentage: float

@dataclass
class Gemstoneevaluation:
    """보석 평가 결과"""
    gemstone_type: str
    quality_score: float  # 0-100
    grade: str  # AAA, AA, A, B, C
    estimated_value_usd: float
    confidence: float
    characteristics: Dict[str, Any]
    market_comparison: Dict[str, float]
    certification_recommendation: str

@dataclass
class JewelryMarketAnalysis:
    """주얼리 시장 분석"""
    current_trend: str
    price_trend: str  # rising, stable, falling
    demand_level: str  # high, medium, low
    seasonal_factor: float
    market_conditions: Dict[str, Any]
    price_prediction_30days: float
    investment_recommendation: str

@dataclass
class AuthenticityAssessment:
    """진품/모조품 판별 결과"""
    authenticity_score: float  # 0-100 (100=확실한 진품)
    assessment: str  # authentic, suspicious, likely_fake
    evidence: List[str]
    red_flags: List[str]
    verification_methods: List[str]
    confidence: float

class DiamondAIEvaluator:
    """다이아몬드 AI 평가기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 다이아몬드 품질 기준 데이터
        self.color_values = {grade: idx for idx, grade in enumerate(DIAMOND_COLOR_GRADES)}
        self.clarity_values = {grade: idx for idx, grade in enumerate(DIAMOND_CLARITY_GRADES)}
        self.cut_values = {grade: idx for idx, grade in enumerate(DIAMOND_CUT_GRADES)}
        
        # 시장 가격 모델 (간소화된 버전)
        self.base_price_per_carat = {
            'D': {'FL': 15000, 'IF': 12000, 'VVS1': 10000, 'VVS2': 8500, 'VS1': 7000, 'VS2': 6000},
            'E': {'FL': 14000, 'IF': 11000, 'VVS1': 9000, 'VVS2': 7500, 'VS1': 6200, 'VS2': 5300},
            'F': {'FL': 13000, 'IF': 10000, 'VVS1': 8000, 'VVS2': 6800, 'VS1': 5600, 'VS2': 4800},
            'G': {'FL': 11000, 'IF': 8500, 'VVS1': 7000, 'VVS2': 6000, 'VS1': 5000, 'VS2': 4200},
            'H': {'FL': 9000, 'IF': 7000, 'VVS1': 6000, 'VVS2': 5200, 'VS1': 4400, 'VS2': 3800},
            'I': {'FL': 7000, 'IF': 5500, 'VVS1': 4800, 'VVS2': 4200, 'VS1': 3600, 'VS2': 3200},
            'J': {'FL': 5500, 'IF': 4500, 'VVS1': 4000, 'VVS2': 3500, 'VS1': 3000, 'VS2': 2600}
        }
        
        self.logger.info("다이아몬드 AI 평가기 초기화 완료")
    
    def evaluate_diamond_4c(self, characteristics: DiamondCharacteristics) -> Dict[str, Any]:
        """다이아몬드 4C 종합 평가"""
        try:
            # 각 C별 점수 계산
            carat_score = self._evaluate_carat(characteristics.carat)
            cut_score = self._evaluate_cut(characteristics)
            color_score = self._evaluate_color(characteristics.color_grade)
            clarity_score = self._evaluate_clarity(characteristics.clarity_grade)
            
            # 종합 점수 (가중 평균)
            weights = {'carat': 0.25, 'cut': 0.30, 'color': 0.25, 'clarity': 0.20}
            overall_score = (
                carat_score * weights['carat'] +
                cut_score * weights['cut'] +
                color_score * weights['color'] +
                clarity_score * weights['clarity']
            )
            
            # 등급 결정
            grade = self._determine_overall_grade(overall_score)
            
            # 가격 추정
            estimated_price = self._estimate_diamond_price(characteristics)
            
            # 투자 가치 분석
            investment_analysis = self._analyze_investment_potential(characteristics, overall_score)
            
            return {
                'overall_score': round(overall_score, 2),
                'grade': grade,
                'detailed_scores': {
                    'carat': round(carat_score, 2),
                    'cut': round(cut_score, 2),
                    'color': round(color_score, 2),
                    'clarity': round(clarity_score, 2)
                },
                'estimated_price_usd': estimated_price,
                'price_per_carat': round(estimated_price / characteristics.carat, 2),
                'investment_analysis': investment_analysis,
                'certification_required': overall_score > 80 or estimated_price > 5000,
                'market_position': self._analyze_market_position(characteristics, overall_score),
                'recommendations': self._generate_recommendations(characteristics, overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"다이아몬드 4C 평가 실패: {e}")
            return {'error': str(e)}
    
    def _evaluate_carat(self, carat: float) -> float:
        """캐럿 평가 (희소성 고려)"""
        if carat >= 3.0:
            return 100
        elif carat >= 2.0:
            return 90 + (carat - 2.0) * 10
        elif carat >= 1.0:
            return 75 + (carat - 1.0) * 15
        elif carat >= 0.5:
            return 50 + (carat - 0.5) * 50
        else:
            return carat * 100
    
    def _evaluate_cut(self, characteristics: DiamondCharacteristics) -> float:
        """컷 평가 (비율 포함)"""
        base_score = (5 - self.cut_values.get(characteristics.cut_grade, 4)) * 20
        
        # 테이블과 깊이 비율 보정
        table_optimal = 57  # 최적 테이블 비율
        depth_optimal = 61  # 최적 깊이 비율
        
        table_penalty = abs(characteristics.table_percentage - table_optimal) * 2
        depth_penalty = abs(characteristics.depth_percentage - depth_optimal) * 1.5
        
        adjusted_score = base_score - table_penalty - depth_penalty
        return max(0, min(100, adjusted_score))
    
    def _evaluate_color(self, color_grade: str) -> float:
        """컬러 평가"""
        color_index = self.color_values.get(color_grade, 10)
        if color_index <= 2:  # D, E, F (무색)
            return 100 - color_index * 2
        elif color_index <= 6:  # G, H, I, J (거의 무색)
            return 90 - (color_index - 2) * 5
        else:  # K 이하 (약간 노란색)
            return max(20, 70 - (color_index - 6) * 8)
    
    def _evaluate_clarity(self, clarity_grade: str) -> float:
        """투명도 평가"""
        clarity_index = self.clarity_values.get(clarity_grade, 8)
        if clarity_index <= 1:  # FL, IF
            return 100 - clarity_index * 2
        elif clarity_index <= 5:  # VVS1-VS2
            return 95 - (clarity_index - 1) * 5
        elif clarity_index <= 7:  # SI1, SI2
            return 70 - (clarity_index - 5) * 10
        else:  # I1-I3
            return max(10, 50 - (clarity_index - 7) * 15)
    
    def _determine_overall_grade(self, score: float) -> str:
        """종합 등급 결정"""
        if score >= 90:
            return "Exceptional"
        elif score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Very Good"
        elif score >= 60:
            return "Good"
        elif score >= 50:
            return "Fair"
        else:
            return "Poor"
    
    def _estimate_diamond_price(self, characteristics: DiamondCharacteristics) -> float:
        """다이아몬드 가격 추정"""
        try:
            color = characteristics.color_grade
            clarity = characteristics.clarity_grade
            
            # 기본 가격 (색상과 투명도 기준)
            if color in self.base_price_per_carat and clarity in self.base_price_per_carat[color]:
                base_price_per_carat = self.base_price_per_carat[color][clarity]
            else:
                # 기본값 사용
                base_price_per_carat = 3000
            
            # 캐럿 가중치 (큰 다이아몬드는 기하급수적으로 비쌈)
            carat_multiplier = characteristics.carat ** 1.8
            
            # 컷 보정
            cut_multipliers = {'Excellent': 1.2, 'Very Good': 1.1, 'Good': 1.0, 'Fair': 0.9, 'Poor': 0.7}
            cut_multiplier = cut_multipliers.get(characteristics.cut_grade, 1.0)
            
            # 형광성 보정 (강한 형광성은 가격 하락)
            fluorescence_adjustments = {'None': 1.0, 'Faint': 0.98, 'Medium': 0.95, 'Strong': 0.85, 'Very Strong': 0.75}
            fluorescence_multiplier = fluorescence_adjustments.get(characteristics.fluorescence, 1.0)
            
            estimated_price = (
                base_price_per_carat * 
                carat_multiplier * 
                cut_multiplier * 
                fluorescence_multiplier
            )
            
            return round(estimated_price, 2)
            
        except Exception as e:
            self.logger.error(f"가격 추정 실패: {e}")
            return 1000.0  # 기본값
    
    def _analyze_investment_potential(self, characteristics: DiamondCharacteristics, score: float) -> Dict[str, Any]:
        """투자 가치 분석"""
        potential = "Low"
        reasoning = []
        
        # 높은 점수
        if score >= 85:
            potential = "High"
            reasoning.append("최고급 품질로 투자 가치 우수")
        elif score >= 75:
            potential = "Medium"
            reasoning.append("양호한 품질로 적정 투자 가치")
        
        # 큰 캐럿
        if characteristics.carat >= 2.0:
            potential = "High" if potential != "High" else potential
            reasoning.append("대형 다이아몬드로 희소성 높음")
        elif characteristics.carat >= 1.0:
            reasoning.append("적절한 크기로 시장성 양호")
        
        # 컬러와 투명도
        if characteristics.color_grade in ['D', 'E', 'F'] and characteristics.clarity_grade in ['FL', 'IF', 'VVS1']:
            potential = "High"
            reasoning.append("최고급 컬러와 투명도")
        
        return {
            'potential': potential,
            'reasoning': reasoning,
            'hold_recommendation': potential == "High",
            'liquidity': "High" if score >= 70 else "Medium"
        }
    
    def _analyze_market_position(self, characteristics: DiamondCharacteristics, score: float) -> str:
        """시장 포지션 분석"""
        if score >= 90:
            return "Luxury Premium"
        elif score >= 80:
            return "High-End"
        elif score >= 70:
            return "Mid-Premium"
        elif score >= 60:
            return "Mid-Range"
        else:
            return "Entry-Level"
    
    def _generate_recommendations(self, characteristics: DiamondCharacteristics, score: float) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        if score >= 80:
            recommendations.append("공인 감정서 발급을 강력히 권장합니다")
            recommendations.append("보험 가입을 고려하세요")
        
        if characteristics.carat >= 1.0:
            recommendations.append("투자용으로 적합한 크기입니다")
        
        if characteristics.cut_grade not in ['Excellent', 'Very Good']:
            recommendations.append("더 나은 컷 등급의 다이아몬드를 고려해보세요")
        
        if characteristics.clarity_grade in ['SI1', 'SI2']:
            recommendations.append("내포물이 육안으로 보이지 않는지 확인하세요")
        
        if characteristics.fluorescence in ['Strong', 'Very Strong']:
            recommendations.append("강한 형광성이 외관에 영향을 줄 수 있으니 주의하세요")
        
        return recommendations

class GemstoneevaluatorAI:
    """보석 AI 평가기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 보석별 특성 기준
        self.gemstone_characteristics = {
            'Ruby': {
                'hardness': 9.0,
                'refractive_index': (1.762, 1.778),
                'specific_gravity': (3.97, 4.05),
                'color_range': ['Pinkish Red', 'Red', 'Purplish Red'],
                'key_inclusions': ['silk', 'needles', 'color_zoning']
            },
            'Sapphire': {
                'hardness': 9.0,
                'refractive_index': (1.762, 1.778),
                'specific_gravity': (3.95, 4.03),
                'color_range': ['Blue', 'Yellow', 'Pink', 'White', 'Green'],
                'key_inclusions': ['silk', 'needles', 'color_banding']
            },
            'Emerald': {
                'hardness': 7.5,
                'refractive_index': (1.565, 1.602),
                'specific_gravity': (2.67, 2.78),
                'color_range': ['Green', 'Bluish Green', 'Yellowish Green'],
                'key_inclusions': ['jardin', 'three_phase_inclusions']
            }
        }
        
        self.logger.info("보석 AI 평가기 초기화 완료")
    
    def evaluate_gemstone(self, gemstone_type: str, properties: Dict[str, Any]) -> Gemstoneevaluation:
        """보석 종합 평가"""
        try:
            # 기본 검증
            if gemstone_type not in self.gemstone_characteristics:
                raise ValueError(f"지원하지 않는 보석 타입: {gemstone_type}")
            
            standards = self.gemstone_characteristics[gemstone_type]
            
            # 각 특성별 평가
            color_score = self._evaluate_color_quality(gemstone_type, properties)
            clarity_score = self._evaluate_clarity_quality(properties)
            cut_score = self._evaluate_cut_quality(properties)
            size_score = self._evaluate_size_quality(properties)
            origin_score = self._evaluate_origin_premium(gemstone_type, properties)
            
            # 종합 점수
            weights = {'color': 0.35, 'clarity': 0.25, 'cut': 0.20, 'size': 0.10, 'origin': 0.10}
            overall_score = (
                color_score * weights['color'] +
                clarity_score * weights['clarity'] +
                cut_score * weights['cut'] +
                size_score * weights['size'] +
                origin_score * weights['origin']
            )
            
            # 등급 결정
            grade = self._determine_gemstone_grade(overall_score)
            
            # 가격 추정
            estimated_value = self._estimate_gemstone_value(gemstone_type, properties, overall_score)
            
            # 시장 비교
            market_comparison = self._analyze_market_comparison(gemstone_type, properties, overall_score)
            
            # 신뢰도 계산
            confidence = self._calculate_evaluation_confidence(properties)
            
            return Gemstoneevaluation(
                gemstone_type=gemstone_type,
                quality_score=round(overall_score, 2),
                grade=grade,
                estimated_value_usd=estimated_value,
                confidence=confidence,
                characteristics={
                    'color_score': round(color_score, 2),
                    'clarity_score': round(clarity_score, 2),
                    'cut_score': round(cut_score, 2),
                    'size_score': round(size_score, 2),
                    'origin_score': round(origin_score, 2)
                },
                market_comparison=market_comparison,
                certification_recommendation=self._get_certification_recommendation(overall_score, estimated_value)
            )
            
        except Exception as e:
            self.logger.error(f"보석 평가 실패: {e}")
            return Gemstoneevaluation(
                gemstone_type=gemstone_type,
                quality_score=0,
                grade="Unknown",
                estimated_value_usd=0,
                confidence=0,
                characteristics={},
                market_comparison={},
                certification_recommendation="전문가 감정 필요"
            )
    
    def _evaluate_color_quality(self, gemstone_type: str, properties: Dict[str, Any]) -> float:
        """색상 품질 평가"""
        color = properties.get('color', '')
        saturation = properties.get('saturation', 50)  # 0-100
        tone = properties.get('tone', 50)  # 0-100 (0=very light, 100=very dark)
        
        base_score = 50
        
        # 보석별 최적 색상
        if gemstone_type == 'Ruby':
            if 'Red' in color and saturation >= 80 and 40 <= tone <= 70:
                base_score = 95
            elif 'Red' in color and saturation >= 60:
                base_score = 80
            else:
                base_score = 60
        
        elif gemstone_type == 'Sapphire':
            if color == 'Blue' and saturation >= 75 and 45 <= tone <= 75:
                base_score = 90
            elif color in ['Blue', 'Pink', 'Yellow'] and saturation >= 60:
                base_score = 75
            else:
                base_score = 65
        
        elif gemstone_type == 'Emerald':
            if 'Green' in color and saturation >= 70 and 40 <= tone <= 75:
                base_score = 90
            elif 'Green' in color and saturation >= 50:
                base_score = 75
            else:
                base_score = 55
        
        return min(100, max(0, base_score))
    
    def _evaluate_clarity_quality(self, properties: Dict[str, Any]) -> float:
        """투명도 품질 평가"""
        clarity_grade = properties.get('clarity_grade', 'SI')
        inclusion_severity = properties.get('inclusion_severity', 'moderate')
        
        clarity_scores = {
            'FL': 100, 'VVS': 90, 'VS': 80, 'SI': 65, 'I': 40
        }
        
        base_score = clarity_scores.get(clarity_grade, 50)
        
        # 내포물 심각도에 따른 조정
        severity_adjustments = {
            'none': 5, 'minor': 0, 'moderate': -10, 'severe': -25
        }
        
        adjustment = severity_adjustments.get(inclusion_severity, -5)
        return min(100, max(0, base_score + adjustment))
    
    def _evaluate_cut_quality(self, properties: Dict[str, Any]) -> float:
        """컷 품질 평가"""
        cut_grade = properties.get('cut_grade', 'Good')
        symmetry = properties.get('symmetry', 'Good')
        polish = properties.get('polish', 'Good')
        
        cut_scores = {'Excellent': 95, 'Very Good': 85, 'Good': 75, 'Fair': 60, 'Poor': 40}
        
        base_score = cut_scores.get(cut_grade, 70)
        
        # 대칭성과 연마 상태 보정
        if symmetry == 'Excellent':
            base_score += 3
        elif symmetry == 'Poor':
            base_score -= 5
        
        if polish == 'Excellent':
            base_score += 2
        elif polish == 'Poor':
            base_score -= 5
        
        return min(100, max(0, base_score))
    
    def _evaluate_size_quality(self, properties: Dict[str, Any]) -> float:
        """크기 품질 평가"""
        carat_weight = properties.get('carat_weight', 1.0)
        
        if carat_weight >= 5.0:
            return 100
        elif carat_weight >= 3.0:
            return 90
        elif carat_weight >= 2.0:
            return 80
        elif carat_weight >= 1.0:
            return 70
        else:
            return 50 + (carat_weight * 20)
    
    def _evaluate_origin_premium(self, gemstone_type: str, properties: Dict[str, Any]) -> float:
        """원산지 프리미엄 평가"""
        origin = properties.get('origin', 'Unknown')
        
        premium_origins = {
            'Ruby': ['Burma', 'Myanmar', 'Mogok'],
            'Sapphire': ['Kashmir', 'Burma', 'Ceylon'],
            'Emerald': ['Colombia', 'Muzo', 'Chivor']
        }
        
        if gemstone_type in premium_origins:
            if origin in premium_origins[gemstone_type]:
                return 90
            elif origin != 'Unknown':
                return 70
        
        return 50
    
    def _determine_gemstone_grade(self, score: float) -> str:
        """보석 등급 결정"""
        if score >= 90:
            return "AAA"
        elif score >= 80:
            return "AA"
        elif score >= 70:
            return "A"
        elif score >= 60:
            return "B"
        else:
            return "C"
    
    def _estimate_gemstone_value(self, gemstone_type: str, properties: Dict[str, Any], score: float) -> float:
        """보석 가치 추정"""
        carat_weight = properties.get('carat_weight', 1.0)
        
        # 기본 가격 (캐럿당 USD)
        base_prices = {
            'Ruby': 1500, 'Sapphire': 800, 'Emerald': 1200,
            'Topaz': 100, 'Amethyst': 50, 'Aquamarine': 200
        }
        
        base_price_per_carat = base_prices.get(gemstone_type, 300)
        
        # 품질 점수에 따른 배수
        quality_multiplier = (score / 50) ** 1.5
        
        # 크기에 따른 배수 (큰 보석은 기하급수적으로 비쌈)
        size_multiplier = carat_weight ** 1.3
        
        estimated_value = base_price_per_carat * quality_multiplier * size_multiplier
        
        return round(estimated_value, 2)
    
    def _analyze_market_comparison(self, gemstone_type: str, properties: Dict[str, Any], score: float) -> Dict[str, float]:
        """시장 비교 분석"""
        return {
            'percentile_rank': min(95, score + 5),  # 시장에서의 위치
            'price_competitiveness': 85 if score >= 80 else 70,
            'rarity_factor': 90 if score >= 85 else 75,
            'investment_appeal': 80 if score >= 75 else 60
        }
    
    def _calculate_evaluation_confidence(self, properties: Dict[str, Any]) -> float:
        """평가 신뢰도 계산"""
        confidence = 70  # 기본 신뢰도
        
        # 데이터 완성도에 따른 조정
        required_fields = ['color', 'clarity_grade', 'cut_grade', 'carat_weight']
        available_fields = sum(1 for field in required_fields if field in properties)
        
        confidence += (available_fields / len(required_fields)) * 20
        
        # 전문적 분석 데이터가 있으면 신뢰도 상승
        if 'spectroscopy_data' in properties:
            confidence += 10
        if 'inclusion_analysis' in properties:
            confidence += 5
        
        return min(95, confidence)
    
    def _get_certification_recommendation(self, score: float, value: float) -> str:
        """감정서 발급 권장사항"""
        if score >= 85 or value >= 2000:
            return "GIA 또는 동급 국제 감정기관 감정서 발급 강력 권장"
        elif score >= 75 or value >= 1000:
            return "공인 감정기관 감정서 발급 권장"
        elif score >= 65:
            return "지역 감정기관 감정서 고려"
        else:
            return "감정서 불필요, 일반 거래용"

class JewelryMarketAnalyzer:
    """주얼리 시장 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 시장 데이터 (실제로는 외부 API에서 가져올 데이터)
        self.market_trends = {
            'diamond': {'trend': 'stable', 'change_3m': 2.5, 'demand': 'high'},
            'ruby': {'trend': 'rising', 'change_3m': 8.2, 'demand': 'very_high'},
            'sapphire': {'trend': 'rising', 'change_3m': 5.1, 'demand': 'high'},
            'emerald': {'trend': 'stable', 'change_3m': 1.8, 'demand': 'medium'}
        }
        
        self.seasonal_factors = {
            1: 0.85, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05,  # 1-5월
            6: 0.95, 7: 0.90, 8: 0.95, 9: 1.00, 10: 1.05,  # 6-10월
            11: 1.15, 12: 1.25  # 11-12월 (연말 성수기)
        }
        
        self.logger.info("주얼리 시장 분석기 초기화 완료")
    
    def analyze_market_conditions(self, jewelry_type: str, characteristics: Dict[str, Any]) -> JewelryMarketAnalysis:
        """시장 상황 종합 분석"""
        try:
            jewelry_type_key = jewelry_type.lower()
            current_month = datetime.now().month
            
            # 기본 트렌드 정보
            trend_data = self.market_trends.get(jewelry_type_key, {
                'trend': 'stable', 'change_3m': 0, 'demand': 'medium'
            })
            
            # 계절성 팩터
            seasonal_factor = self.seasonal_factors.get(current_month, 1.0)
            
            # 수요 레벨 분석
            demand_level = self._analyze_demand_level(jewelry_type, characteristics)
            
            # 가격 예측
            price_prediction = self._predict_price_movement(trend_data, seasonal_factor)
            
            # 투자 추천
            investment_recommendation = self._generate_investment_recommendation(
                trend_data, demand_level, characteristics
            )
            
            return JewelryMarketAnalysis(
                current_trend=trend_data['trend'],
                price_trend=self._determine_price_trend(trend_data['change_3m']),
                demand_level=demand_level,
                seasonal_factor=seasonal_factor,
                market_conditions={
                    'volatility': self._calculate_volatility(jewelry_type),
                    'liquidity': self._assess_liquidity(jewelry_type, characteristics),
                    'growth_potential': self._assess_growth_potential(trend_data)
                },
                price_prediction_30days=price_prediction,
                investment_recommendation=investment_recommendation
            )
            
        except Exception as e:
            self.logger.error(f"시장 분석 실패: {e}")
            return JewelryMarketAnalysis(
                current_trend="unknown",
                price_trend="stable",
                demand_level="medium",
                seasonal_factor=1.0,
                market_conditions={},
                price_prediction_30days=0,
                investment_recommendation="데이터 부족으로 분석 불가"
            )
    
    def _analyze_demand_level(self, jewelry_type: str, characteristics: Dict[str, Any]) -> str:
        """수요 레벨 분석"""
        base_demand = self.market_trends.get(jewelry_type.lower(), {}).get('demand', 'medium')
        
        # 특성에 따른 수요 조정
        if jewelry_type.lower() == 'diamond':
            carat = characteristics.get('carat', 1.0)
            if carat >= 2.0:
                return 'very_high'
            elif carat >= 1.0:
                return 'high'
        
        return base_demand
    
    def _determine_price_trend(self, change_3m: float) -> str:
        """가격 트렌드 결정"""
        if change_3m > 3:
            return 'rising'
        elif change_3m < -3:
            return 'falling'
        else:
            return 'stable'
    
    def _predict_price_movement(self, trend_data: Dict, seasonal_factor: float) -> float:
        """30일 가격 변동 예측"""
        base_change = trend_data.get('change_3m', 0) / 3  # 월간 변화율
        seasonal_adjustment = (seasonal_factor - 1) * 100
        
        # 간단한 예측 모델
        predicted_change = base_change + seasonal_adjustment
        return round(predicted_change, 2)
    
    def _generate_investment_recommendation(self, trend_data: Dict, demand_level: str, characteristics: Dict) -> str:
        """투자 추천 생성"""
        trend = trend_data.get('trend', 'stable')
        change = trend_data.get('change_3m', 0)
        
        if trend == 'rising' and demand_level in ['high', 'very_high']:
            return "매수 적극 권장 - 상승 트렌드와 높은 수요"
        elif trend == 'rising':
            return "매수 권장 - 상승 트렌드"
        elif trend == 'stable' and demand_level == 'high':
            return "보유 권장 - 안정적이고 수요 양호"
        elif trend == 'falling':
            return "관망 권장 - 하락 트렌드 주의"
        else:
            return "중립 - 시장 상황 지켜볼 필요"
    
    def _calculate_volatility(self, jewelry_type: str) -> str:
        """변동성 계산"""
        volatility_levels = {
            'diamond': 'low',
            'ruby': 'medium',
            'sapphire': 'medium',
            'emerald': 'high',
            'pearl': 'medium'
        }
        return volatility_levels.get(jewelry_type.lower(), 'medium')
    
    def _assess_liquidity(self, jewelry_type: str, characteristics: Dict) -> str:
        """유동성 평가"""
        if jewelry_type.lower() == 'diamond':
            carat = characteristics.get('carat', 1.0)
            if carat >= 1.0:
                return 'high'
            else:
                return 'medium'
        
        return 'medium'
    
    def _assess_growth_potential(self, trend_data: Dict) -> str:
        """성장 잠재력 평가"""
        change = trend_data.get('change_3m', 0)
        
        if change > 5:
            return 'high'
        elif change > 2:
            return 'medium'
        elif change > -2:
            return 'low'
        else:
            return 'negative'

class AuthenticityDetectorAI:
    """진품/모조품 판별 AI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 진품 판별 기준
        self.authenticity_criteria = {
            'diamond': {
                'thermal_conductivity': True,
                'hardness_test': True,
                'refractive_index': (2.417, 2.419),
                'specific_gravity': (3.50, 3.53)
            },
            'ruby': {
                'hardness': 9.0,
                'refractive_index': (1.762, 1.778),
                'pleochroism': True,
                'fluorescence_pattern': 'red'
            }
        }
        
        self.logger.info("진품 판별 AI 초기화 완료")
    
    def assess_authenticity(self, jewelry_type: str, test_results: Dict[str, Any]) -> AuthenticityAssessment:
        """진품성 종합 평가"""
        try:
            evidence = []
            red_flags = []
            authenticity_score = 50  # 기본 점수
            
            # 물리적 특성 검사
            physical_score = self._check_physical_properties(jewelry_type, test_results)
            authenticity_score += (physical_score - 50) * 0.4
            
            # 광학적 특성 검사
            optical_score = self._check_optical_properties(jewelry_type, test_results)
            authenticity_score += (optical_score - 50) * 0.3
            
            # 내포물 및 구조 검사
            inclusion_score = self._check_inclusions(jewelry_type, test_results)
            authenticity_score += (inclusion_score - 50) * 0.2
            
            # 가격 합리성 검사
            price_consistency = self._check_price_consistency(jewelry_type, test_results)
            authenticity_score += (price_consistency - 50) * 0.1
            
            # 증거와 의혹 사항 수집
            evidence, red_flags = self._collect_evidence_and_flags(jewelry_type, test_results)
            
            # 최종 평가
            authenticity_score = max(0, min(100, authenticity_score))
            assessment = self._determine_authenticity_assessment(authenticity_score)
            
            # 검증 방법 추천
            verification_methods = self._recommend_verification_methods(jewelry_type, authenticity_score)
            
            # 신뢰도 계산
            confidence = self._calculate_authenticity_confidence(test_results)
            
            return AuthenticityAssessment(
                authenticity_score=round(authenticity_score, 2),
                assessment=assessment,
                evidence=evidence,
                red_flags=red_flags,
                verification_methods=verification_methods,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"진품성 평가 실패: {e}")
            return AuthenticityAssessment(
                authenticity_score=0,
                assessment="unable_to_assess",
                evidence=[],
                red_flags=["평가 데이터 부족"],
                verification_methods=["전문기관 감정 필요"],
                confidence=0
            )
    
    def _check_physical_properties(self, jewelry_type: str, test_results: Dict[str, Any]) -> float:
        """물리적 특성 검사"""
        score = 50
        
        if jewelry_type.lower() == 'diamond':
            # 경도 테스트
            if test_results.get('hardness_test_passed', False):
                score += 20
            
            # 열전도도 테스트
            if test_results.get('thermal_conductivity_high', False):
                score += 25
            
            # 비중 검사
            specific_gravity = test_results.get('specific_gravity', 0)
            if 3.50 <= specific_gravity <= 3.53:
                score += 15
            elif specific_gravity > 0:
                score -= 20
        
        return min(100, max(0, score))
    
    def _check_optical_properties(self, jewelry_type: str, test_results: Dict[str, Any]) -> float:
        """광학적 특성 검사"""
        score = 50
        
        # 굴절률 검사
        refractive_index = test_results.get('refractive_index', 0)
        if jewelry_type.lower() == 'diamond':
            if 2.417 <= refractive_index <= 2.419:
                score += 30
            elif refractive_index > 0:
                score -= 25
        
        # 복굴절 검사
        if test_results.get('birefringence_test', False):
            score += 10
        
        # 분산 검사
        dispersion = test_results.get('dispersion', 0)
        if dispersion > 0.044:  # 다이아몬드의 높은 분산
            score += 10
        
        return min(100, max(0, score))
    
    def _check_inclusions(self, jewelry_type: str, test_results: Dict[str, Any]) -> float:
        """내포물 및 구조 검사"""
        score = 50
        
        inclusions = test_results.get('inclusions', [])
        
        if jewelry_type.lower() == 'diamond':
            # 자연 다이아몬드의 전형적인 내포물
            natural_inclusions = ['crystal', 'feather', 'cloud', 'pinpoint']
            synthetic_indicators = ['metallic_inclusion', 'hourglass_pattern']
            
            for inclusion in inclusions:
                if inclusion in natural_inclusions:
                    score += 5
                elif inclusion in synthetic_indicators:
                    score -= 15
        
        # 성장 패턴 검사
        growth_pattern = test_results.get('growth_pattern', '')
        if growth_pattern == 'natural':
            score += 15
        elif growth_pattern == 'synthetic':
            score -= 20
        
        return min(100, max(0, score))
    
    def _check_price_consistency(self, jewelry_type: str, test_results: Dict[str, Any]) -> float:
        """가격 합리성 검사"""
        score = 50
        
        claimed_value = test_results.get('claimed_value', 0)
        estimated_value = test_results.get('estimated_value', 0)
        
        if claimed_value > 0 and estimated_value > 0:
            ratio = claimed_value / estimated_value
            
            if 0.8 <= ratio <= 1.2:  # 합리적 범위
                score += 20
            elif 0.5 <= ratio <= 2.0:  # 허용 범위
                score += 10
            elif ratio > 3.0:  # 의심스럽게 비쌈
                score -= 30
            elif ratio < 0.3:  # 의심스럽게 쌈
                score -= 25
        
        return min(100, max(0, score))
    
    def _collect_evidence_and_flags(self, jewelry_type: str, test_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """증거와 의혹 사항 수집"""
        evidence = []
        red_flags = []
        
        # 긍정적 증거
        if test_results.get('thermal_conductivity_high', False):
            evidence.append("높은 열전도율 확인")
        
        if test_results.get('hardness_test_passed', False):
            evidence.append("경도 테스트 통과")
        
        if test_results.get('natural_inclusions', False):
            evidence.append("자연 내포물 발견")
        
        # 의혹 사항
        if test_results.get('price_too_low', False):
            red_flags.append("시장가 대비 비정상적으로 낮은 가격")
        
        if test_results.get('synthetic_indicators', False):
            red_flags.append("합성 보석 지시자 발견")
        
        if test_results.get('modern_cutting_style', False) and test_results.get('claimed_age', 0) > 100:
            red_flags.append("주장된 연대와 컷팅 스타일 불일치")
        
        return evidence, red_flags
    
    def _determine_authenticity_assessment(self, score: float) -> str:
        """진품성 평가 결정"""
        if score >= 85:
            return "authentic"
        elif score >= 60:
            return "likely_authentic"
        elif score >= 40:
            return "suspicious"
        else:
            return "likely_fake"
    
    def _recommend_verification_methods(self, jewelry_type: str, score: float) -> List[str]:
        """검증 방법 추천"""
        methods = []
        
        if score < 70:
            methods.append("공인 감정기관 정밀 검사")
            methods.append("X선 회절 분석")
            methods.append("분광 분석")
        
        if jewelry_type.lower() == 'diamond':
            methods.append("다이아몬드 테스터 검사")
            methods.append("자외선 형광 반응 검사")
        
        methods.append("현미경 내포물 분석")
        methods.append("전문가 육안 검사")
        
        return methods
    
    def _calculate_authenticity_confidence(self, test_results: Dict[str, Any]) -> float:
        """진품성 평가 신뢰도 계산"""
        confidence = 60  # 기본 신뢰도
        
        # 검사 항목 수에 따른 신뢰도 증가
        test_count = len([k for k, v in test_results.items() if v is not None and v != 0])
        confidence += min(30, test_count * 3)
        
        # 전문 장비 사용 시 신뢰도 증가
        if test_results.get('professional_equipment_used', False):
            confidence += 10
        
        return min(95, confidence)

class JewelryAIManager:
    """주얼리 AI 통합 관리자"""
    
    def __init__(self):
        self.diamond_evaluator = DiamondAIEvaluator()
        self.gemstone_evaluator = GemstoneevaluatorAI()
        self.market_analyzer = JewelryMarketAnalyzer()
        self.authenticity_detector = AuthenticityDetectorAI()
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_jewelry_analysis(self, 
                                     jewelry_type: str,
                                     characteristics: Dict[str, Any],
                                     test_results: Dict[str, Any] = None,
                                     market_analysis: bool = True) -> Dict[str, Any]:
        """주얼리 종합 AI 분석"""
        try:
            analysis_results = {
                'jewelry_type': jewelry_type,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_components': []
            }
            
            # 1. 품질 평가
            if jewelry_type.lower() == 'diamond':
                if all(key in characteristics for key in ['carat', 'cut_grade', 'color_grade', 'clarity_grade']):
                    diamond_chars = DiamondCharacteristics(**characteristics)
                    quality_analysis = self.diamond_evaluator.evaluate_diamond_4c(diamond_chars)
                    analysis_results['quality_analysis'] = quality_analysis
                    analysis_results['analysis_components'].append('diamond_4c_evaluation')
            else:
                gemstone_evaluation = self.gemstone_evaluator.evaluate_gemstone(jewelry_type, characteristics)
                analysis_results['quality_analysis'] = asdict(gemstone_evaluation)
                analysis_results['analysis_components'].append('gemstone_evaluation')
            
            # 2. 시장 분석
            if market_analysis:
                market_analysis_result = self.market_analyzer.analyze_market_conditions(jewelry_type, characteristics)
                analysis_results['market_analysis'] = asdict(market_analysis_result)
                analysis_results['analysis_components'].append('market_analysis')
            
            # 3. 진품성 평가
            if test_results:
                authenticity_assessment = self.authenticity_detector.assess_authenticity(jewelry_type, test_results)
                analysis_results['authenticity_assessment'] = asdict(authenticity_assessment)
                analysis_results['analysis_components'].append('authenticity_assessment')
            
            # 4. 종합 추천사항
            comprehensive_recommendations = self._generate_comprehensive_recommendations(analysis_results)
            analysis_results['comprehensive_recommendations'] = comprehensive_recommendations
            
            # 5. 투자 점수
            investment_score = self._calculate_investment_score(analysis_results)
            analysis_results['investment_score'] = investment_score
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"종합 주얼리 분석 실패: {e}")
            return {
                'error': str(e),
                'jewelry_type': jewelry_type,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _generate_comprehensive_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """종합 추천사항 생성"""
        recommendations = []
        
        # 품질 분석 기반 추천
        if 'quality_analysis' in analysis_results:
            quality = analysis_results['quality_analysis']
            
            if isinstance(quality, dict):
                overall_score = quality.get('overall_score', quality.get('quality_score', 0))
                
                if overall_score >= 85:
                    recommendations.append("🏆 최고급 품질 - 투자 및 수집용으로 추천")
                elif overall_score >= 75:
                    recommendations.append("💎 우수한 품질 - 장기 보유 권장")
                elif overall_score >= 60:
                    recommendations.append("📈 양호한 품질 - 적정 가격 확인 후 구매")
                else:
                    recommendations.append("⚠️ 품질 주의 - 신중한 검토 필요")
        
        # 시장 분석 기반 추천
        if 'market_analysis' in analysis_results:
            market = analysis_results['market_analysis']
            
            if market.get('price_trend') == 'rising':
                recommendations.append("📈 가격 상승 트렌드 - 조기 구매 고려")
            elif market.get('current_trend') == 'stable':
                recommendations.append("📊 안정적 시장 - 안전한 투자처")
        
        # 진품성 평가 기반 추천
        if 'authenticity_assessment' in analysis_results:
            auth = analysis_results['authenticity_assessment']
            assessment = auth.get('assessment', '')
            
            if assessment == 'authentic':
                recommendations.append("✅ 진품 확신 - 안전한 거래 가능")
            elif assessment in ['suspicious', 'likely_fake']:
                recommendations.append("🚨 진품성 의혹 - 전문 감정 필수")
        
        return recommendations
    
    def _calculate_investment_score(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """투자 점수 계산"""
        scores = {'quality': 0, 'market': 0, 'authenticity': 0, 'overall': 0}
        
        # 품질 점수
        if 'quality_analysis' in analysis_results:
            quality = analysis_results['quality_analysis']
            if isinstance(quality, dict):
                scores['quality'] = quality.get('overall_score', quality.get('quality_score', 0))
        
        # 시장 점수
        if 'market_analysis' in analysis_results:
            market = analysis_results['market_analysis']
            trend_score = 80 if market.get('price_trend') == 'rising' else 60
            demand_score = {'very_high': 90, 'high': 80, 'medium': 60, 'low': 40}.get(market.get('demand_level', 'medium'), 60)
            scores['market'] = (trend_score + demand_score) / 2
        
        # 진품성 점수
        if 'authenticity_assessment' in analysis_results:
            auth = analysis_results['authenticity_assessment']
            scores['authenticity'] = auth.get('authenticity_score', 0)
        
        # 종합 점수 (가중 평균)
        weights = {'quality': 0.4, 'market': 0.3, 'authenticity': 0.3}
        scores['overall'] = sum(scores[key] * weights[key] for key in weights if scores[key] > 0)
        
        return {k: round(v, 2) for k, v in scores.items()}

# 전역 주얼리 AI 관리자
global_jewelry_ai_manager = JewelryAIManager()

def jewelry_ai_enhanced(jewelry_type: str, include_market_analysis: bool = True):
    """주얼리 AI 향상 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 원본 함수 실행
            result = func(*args, **kwargs)
            
            # 결과에서 주얼리 특성 추출
            if isinstance(result, dict) and 'characteristics' in result:
                characteristics = result['characteristics']
                test_results = result.get('test_results')
                
                # AI 분석 실행
                ai_analysis = global_jewelry_ai_manager.comprehensive_jewelry_analysis(
                    jewelry_type=jewelry_type,
                    characteristics=characteristics,
                    test_results=test_results,
                    market_analysis=include_market_analysis
                )
                
                # 결과에 AI 분석 추가
                result['ai_analysis'] = ai_analysis
            
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트 실행
    print("💎 솔로몬드 AI v2.1.3 - 주얼리 특화 AI 모델")
    print("=" * 60)
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 다이아몬드 테스트
    print("\n💎 다이아몬드 AI 평가 테스트...")
    diamond_chars = DiamondCharacteristics(
        carat=1.5,
        cut_grade="Excellent",
        color_grade="G",
        clarity_grade="VS1",
        polish="Excellent",
        symmetry="Very Good",
        fluorescence="None",
        measurements={"length": 7.2, "width": 7.1, "depth": 4.4},
        table_percentage=57.0,
        depth_percentage=61.5
    )
    
    diamond_ai = DiamondAIEvaluator()
    diamond_result = diamond_ai.evaluate_diamond_4c(diamond_chars)
    
    print(f"종합 점수: {diamond_result['overall_score']}/100")
    print(f"등급: {diamond_result['grade']}")
    print(f"추정 가격: ${diamond_result['estimated_price_usd']:,.2f}")
    print(f"캐럿당 가격: ${diamond_result['price_per_carat']:,.2f}")
    
    # 루비 테스트
    print("\n🔴 루비 AI 평가 테스트...")
    ruby_props = {
        'color': 'Red',
        'saturation': 85,
        'tone': 60,
        'clarity_grade': 'VS',
        'cut_grade': 'Excellent',
        'carat_weight': 2.0,
        'origin': 'Burma',
        'symmetry': 'Excellent',
        'polish': 'Excellent'
    }
    
    gemstone_ai = GemstoneevaluatorAI()
    ruby_result = gemstone_ai.evaluate_gemstone('Ruby', ruby_props)
    
    print(f"품질 점수: {ruby_result.quality_score}/100")
    print(f"등급: {ruby_result.grade}")
    print(f"추정 가치: ${ruby_result.estimated_value_usd:,.2f}")
    print(f"신뢰도: {ruby_result.confidence:.1f}%")
    
    # 시장 분석 테스트
    print("\n📊 시장 분석 테스트...")
    market_analyzer = JewelryMarketAnalyzer()
    market_result = market_analyzer.analyze_market_conditions('ruby', ruby_props)
    
    print(f"현재 트렌드: {market_result.current_trend}")
    print(f"가격 트렌드: {market_result.price_trend}")
    print(f"수요 레벨: {market_result.demand_level}")
    print(f"30일 가격 예측: {market_result.price_prediction_30days:+.2f}%")
    print(f"투자 추천: {market_result.investment_recommendation}")
    
    # 진품 판별 테스트
    print("\n🔍 진품 판별 테스트...")
    test_results = {
        'thermal_conductivity_high': True,
        'hardness_test_passed': True,
        'refractive_index': 2.418,
        'specific_gravity': 3.52,
        'natural_inclusions': True,
        'claimed_value': 8000,
        'estimated_value': 7500
    }
    
    auth_detector = AuthenticityDetectorAI()
    auth_result = auth_detector.assess_authenticity('diamond', test_results)
    
    print(f"진품성 점수: {auth_result.authenticity_score}/100")
    print(f"평가: {auth_result.assessment}")
    print(f"증거: {', '.join(auth_result.evidence)}")
    print(f"신뢰도: {auth_result.confidence:.1f}%")
    
    # 종합 분석 테스트
    print("\n🎯 종합 AI 분석 테스트...")
    ai_manager = JewelryAIManager()
    
    comprehensive_result = ai_manager.comprehensive_jewelry_analysis(
        jewelry_type='diamond',
        characteristics=asdict(diamond_chars),
        test_results=test_results,
        market_analysis=True
    )
    
    print(f"분석 구성요소: {', '.join(comprehensive_result['analysis_components'])}")
    print(f"투자 점수: {comprehensive_result['investment_score']['overall']:.1f}/100")
    print("종합 추천사항:")
    for rec in comprehensive_result['comprehensive_recommendations']:
        print(f"  {rec}")
    
    # 데코레이터 테스트
    print("\n🎨 주얼리 AI 데코레이터 테스트...")
    
    @jewelry_ai_enhanced('diamond', include_market_analysis=True)
    def analyze_diamond_sample():
        return {
            'characteristics': asdict(diamond_chars),
            'test_results': test_results,
            'basic_analysis': '기본 분석 완료'
        }
    
    enhanced_result = analyze_diamond_sample()
    if 'ai_analysis' in enhanced_result:
        ai_investment_score = enhanced_result['ai_analysis']['investment_score']['overall']
        print(f"AI 향상 분석 완료 - 투자 점수: {ai_investment_score:.1f}/100")
    
    print("\n✅ 주얼리 특화 AI 모델 테스트 완료!")
    print("💎 다이아몬드부터 보석까지 모든 주얼리의 AI 분석이 가능합니다!")
