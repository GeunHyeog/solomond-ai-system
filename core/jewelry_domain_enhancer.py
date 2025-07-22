#!/usr/bin/env python3
"""
주얼리 도메인 특화 분석 강화 엔진
주얼리 업계 전문 지식을 활용한 고급 분석 및 인사이트 제공
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

class JewelryDomainEnhancer:
    """주얼리 도메인 특화 분석 강화"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # 주얼리 업계 전문 지식베이스
        self.jewelry_expertise = self._build_jewelry_expertise()
        
        # 시장 트렌드 분석
        self.market_trends = self._build_market_trends()
        
        # 고객 세그먼트 분석
        self.customer_segments = self._build_customer_segments()
        
        # 비즈니스 인사이트 패턴
        self.business_patterns = self._build_business_patterns()
        
        self.logger.info("💎 주얼리 도메인 특화 엔진 초기화 완료")
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _build_jewelry_expertise(self) -> Dict[str, Any]:
        """주얼리 업계 전문 지식베이스"""
        return {
            "precious_metals": {
                "gold": {
                    "purities": ["24K", "22K", "18K", "14K", "10K"],
                    "colors": ["옐로우 골드", "화이트 골드", "로즈 골드", "그린 골드"],
                    "price_factors": ["순도", "중량", "국제 금시세", "가공비"],
                    "characteristics": {
                        "24K": "순금, 가장 순수하지만 부드러움",
                        "18K": "일반적인 고급 주얼리 표준",
                        "14K": "내구성과 가치의 균형"
                    }
                },
                "silver": {
                    "purities": ["999 실버", "925 스털링 실버", "800 실버"],
                    "treatments": ["산화 방지 코팅", "로듐 도금"],
                    "characteristics": {
                        "925": "스털링 실버 표준, 내구성 좋음",
                        "999": "순은, 매우 부드러움"
                    }
                },
                "platinum": {
                    "purities": ["950 플래티넘", "900 플래티넘"],
                    "characteristics": "최고급 소재, 변색 없음, 내구성 최고"
                }
            },
            "gemstones": {
                "diamonds": {
                    "4c_grading": {
                        "cut": ["Excellent", "Very Good", "Good", "Fair", "Poor"],
                        "color": ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
                        "clarity": ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2"],
                        "carat": "캐럿 중량"
                    },
                    "shapes": ["라운드", "프린세스", "에메랄드", "아셔", "오벌", "마키즈", "페어", "하트", "쿠션"],
                    "certifications": ["GIA", "AGS", "SSEF", "Gübelin"]
                },
                "colored_gems": {
                    "ruby": {"origin": ["미얀마", "모잠비크", "태국"], "treatments": ["가열", "오일링"]},
                    "sapphire": {"colors": ["블루", "핑크", "옐로우", "패드파라차"], "origin": ["스리랑카", "마다가스카르"]},
                    "emerald": {"origin": ["콜롬비아", "잠비아", "브라질"], "treatments": ["오일링", "수지 처리"]}
                }
            },
            "jewelry_types": {
                "rings": {
                    "categories": ["약혼반지", "결혼반지", "패션반지", "시그넷반지"],
                    "settings": ["프롱", "베젤", "파베", "채널", "테이션"],
                    "sizing": "사이즈 조절 가능성 고려"
                },
                "necklaces": {
                    "lengths": ["초커(35-40cm)", "프린세스(42-48cm)", "마티네(50-60cm)", "오페라(70-90cm)"],
                    "chain_types": ["베네치안", "로프", "피가로", "큐브", "앙커"]
                },
                "earrings": {
                    "types": ["스터드", "드롭", "후프", "샹들리에", "클러스터"],
                    "backs": ["푸시백", "스크류백", "레버백", "피쉬훅"]
                }
            },
            "market_positioning": {
                "luxury": {
                    "brands": ["까르띠에", "티파니", "불가리", "샤넬", "반클리프"],
                    "price_range": "1000만원 이상",
                    "characteristics": "브랜드 프리미엄, 최고급 소재, 한정판"
                },
                "premium": {
                    "price_range": "100만원 - 1000만원",
                    "characteristics": "고품질 소재, 세련된 디자인, 브랜드 인지도"
                },
                "accessible": {
                    "price_range": "10만원 - 100만원",
                    "characteristics": "합리적 가격, 트렌디한 디자인, 실용성"
                }
            }
        }
    
    def _build_market_trends(self) -> Dict[str, Any]:
        """시장 트렌드 분석"""
        return {
            "2024_trends": {
                "popular_styles": ["미니멀", "빈티지", "레이어드", "컬러풀"],
                "hot_materials": ["로즈골드", "컬러드 다이아몬드", "랩그로운 다이아몬드"],
                "emerging_categories": ["지속가능 주얼리", "개인 맞춤", "스택 링"]
            },
            "seasonal_patterns": {
                "spring": ["파스텔 컬러", "플로럴 모티프", "라이트 톤"],
                "summer": ["브라이트 컬러", "비치 테마", "서머 체인"],
                "fall": ["어스 톤", "따뜻한 메탈", "레이어드 룩"],
                "winter": ["클래식", "럭셔리", "스파클링"]
            },
            "demographic_preferences": {
                "gen_z": ["개성 표현", "지속가능성", "합리적 가격", "SNS 친화적"],
                "millennials": ["투자 가치", "브랜드 스토리", "경험 중시"],
                "gen_x": ["실용성", "품질", "클래식 디자인"],
                "boomers": ["전통적 가치", "최고급 소재", "장인정신"]
            }
        }
    
    def _build_customer_segments(self) -> Dict[str, Any]:
        """고객 세그먼트 분석"""
        return {
            "bridal_customers": {
                "characteristics": ["약혼/결혼 준비", "일생일대 구매", "감정적 가치 중시"],
                "key_factors": ["다이아몬드 품질", "반지 사이즈", "예산 계획", "브랜드 신뢰"],
                "price_sensitivity": "중간 (품질 대비 가격)",
                "decision_timeline": "3-6개월"
            },
            "fashion_enthusiasts": {
                "characteristics": ["트렌드 추종", "다양한 스타일링", "컬렉션 구축"],
                "key_factors": ["디자인 독창성", "착용 편의성", "가격 접근성"],
                "price_sensitivity": "높음",
                "decision_timeline": "즉석 - 1개월"
            },
            "luxury_collectors": {
                "characteristics": ["고가 제품 선호", "브랜드 충성도", "투자 목적"],
                "key_factors": ["브랜드 프리미엄", "희소성", "재판매 가치"],
                "price_sensitivity": "낮음",
                "decision_timeline": "신중한 검토 (1-3개월)"
            },
            "gift_buyers": {
                "characteristics": ["타인을 위한 구매", "안전한 선택 선호"],
                "key_factors": ["무난함", "포장", "교환/환불 정책"],
                "price_sensitivity": "예산 범위 내",
                "decision_timeline": "긴급 - 2주"
            }
        }
    
    def _build_business_patterns(self) -> Dict[str, Any]:
        """비즈니스 인사이트 패턴"""
        return {
            "sales_indicators": {
                "high_intent": ["가격 문의", "사이즈 확인", "인증서 요청", "할부 문의"],
                "comparison_shopping": ["여러 옵션 비교", "경쟁사 언급", "가격 비교"],
                "objection_handling": ["가격 부담", "품질 의구심", "브랜드 인지도"],
                "closing_signals": ["구매 결정", "결제 방법", "배송 일정"]
            },
            "service_opportunities": {
                "customization": ["개인 맞춤", "각인 서비스", "디자인 변경"],
                "after_service": ["사이즈 조절", "청소", "수리", "업그레이드"],
                "education": ["관리 방법", "품질 설명", "트렌드 안내"]
            },
            "risk_factors": {
                "price_objections": "가격 대비 가치 설명 필요",
                "quality_concerns": "인증서 및 보증 강조",
                "competition": "차별화 포인트 부각"
            }
        }
    
    def analyze_jewelry_context(self, text: str, conversation_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """주얼리 컨텍스트 분석"""
        self.logger.info("💎 주얼리 도메인 컨텍스트 분석 시작")
        
        analysis = {
            "jewelry_mentions": self._extract_jewelry_mentions(text),
            "customer_segment": self._identify_customer_segment(text, conversation_data),
            "market_position": self._analyze_market_position(text),
            "business_insights": self._generate_business_insights(text),
            "trend_alignment": self._analyze_trend_alignment(text),
            "expertise_level": self._assess_expertise_level(text)
        }
        
        return analysis
    
    def _extract_jewelry_mentions(self, text: str) -> Dict[str, Any]:
        """주얼리 관련 언급 추출 및 분석"""
        mentions = {
            "products": [],
            "materials": [],
            "quality_terms": [],
            "price_references": [],
            "brands": [],
            "technical_specs": []
        }
        
        text_lower = text.lower()
        
        # 제품 유형 감지
        for category, items in self.jewelry_expertise["jewelry_types"].items():
            for item_type in items.get("categories", []):
                if item_type.lower() in text_lower:
                    mentions["products"].append({
                        "category": category,
                        "type": item_type,
                        "confidence": self._calculate_mention_confidence(item_type, text)
                    })
        
        # 소재 감지
        for metal_type, details in self.jewelry_expertise["precious_metals"].items():
            if metal_type in text_lower:
                mentions["materials"].append({
                    "type": metal_type,
                    "details": details,
                    "confidence": self._calculate_mention_confidence(metal_type, text)
                })
        
        # 품질 용어 감지
        quality_terms = ["4c", "gia", "인증서", "캐럿", "clarity", "color", "cut"]
        for term in quality_terms:
            if term.lower() in text_lower:
                mentions["quality_terms"].append(term)
        
        # 가격 관련 언급
        price_patterns = [
            r'\d+만원', r'\d+천원', r'\$\d+', r'₩\d+',
            r'할인', r'세일', r'프로모션', r'특가'
        ]
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            mentions["price_references"].extend(matches)
        
        return mentions
    
    def _identify_customer_segment(self, text: str, conversation_data: Dict = None) -> Dict[str, Any]:
        """고객 세그먼트 식별"""
        segment_scores = {}
        text_lower = text.lower()
        
        # 각 세그먼트별 키워드 매칭
        for segment, details in self.customer_segments.items():
            score = 0
            matched_keywords = []
            
            # 특성 기반 점수 계산
            for characteristic in details["characteristics"]:
                if any(word in text_lower for word in characteristic.lower().split()):
                    score += 2
                    matched_keywords.append(characteristic)
            
            # 핵심 요소 기반 점수
            for factor in details["key_factors"]:
                if any(word in text_lower for word in factor.lower().split()):
                    score += 3
                    matched_keywords.append(factor)
            
            if score > 0:
                segment_scores[segment] = {
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "characteristics": details
                }
        
        # 가장 높은 점수의 세그먼트 반환
        if segment_scores:
            primary_segment = max(segment_scores, key=lambda x: segment_scores[x]["score"])
            return {
                "primary_segment": primary_segment,
                "confidence": segment_scores[primary_segment]["score"] / 10,  # 0-1 스케일
                "all_segments": segment_scores
            }
        
        return {"primary_segment": "general", "confidence": 0.1, "all_segments": {}}
    
    def _analyze_market_position(self, text: str) -> Dict[str, Any]:
        """시장 포지션 분석"""
        text_lower = text.lower()
        position_indicators = {
            "luxury": ["명품", "럭셔리", "최고급", "프리미엄", "한정판", "까르띠에", "티파니"],
            "premium": ["고급", "품질", "브랜드", "인증", "보증"],
            "accessible": ["합리적", "저렴", "할인", "가성비", "실용적"]
        }
        
        position_scores = {}
        for position, keywords in position_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                position_scores[position] = score
        
        if position_scores:
            target_position = max(position_scores, key=position_scores.get)
            market_info = self.jewelry_expertise["market_positioning"][target_position]
            
            return {
                "target_position": target_position,
                "confidence": position_scores[target_position] / 5,
                "market_info": market_info,
                "all_scores": position_scores
            }
        
        return {"target_position": "premium", "confidence": 0.3, "market_info": {}}
    
    def _generate_business_insights(self, text: str) -> List[Dict[str, Any]]:
        """비즈니스 인사이트 생성"""
        insights = []
        text_lower = text.lower()
        
        # 판매 시그널 분석
        for signal_type, indicators in self.business_patterns["sales_indicators"].items():
            matches = [ind for ind in indicators if any(word in text_lower for word in ind.split())]
            if matches:
                insights.append({
                    "type": "sales_signal",
                    "category": signal_type,
                    "indicators": matches,
                    "recommendation": self._get_sales_recommendation(signal_type)
                })
        
        # 서비스 기회 분석
        for service_type, opportunities in self.business_patterns["service_opportunities"].items():
            matches = [opp for opp in opportunities if any(word in text_lower for word in opp.split())]
            if matches:
                insights.append({
                    "type": "service_opportunity",
                    "category": service_type,
                    "opportunities": matches,
                    "recommendation": self._get_service_recommendation(service_type)
                })
        
        return insights
    
    def _analyze_trend_alignment(self, text: str) -> Dict[str, Any]:
        """트렌드 정렬성 분석"""
        text_lower = text.lower()
        trend_matches = {
            "styles": [],
            "materials": [],
            "categories": []
        }
        
        # 2024 트렌드와 매칭
        trends = self.market_trends["2024_trends"]
        
        for style in trends["popular_styles"]:
            if style.lower() in text_lower:
                trend_matches["styles"].append(style)
        
        for material in trends["hot_materials"]:
            if material.lower() in text_lower:
                trend_matches["materials"].append(material)
        
        for category in trends["emerging_categories"]:
            if any(word in text_lower for word in category.split()):
                trend_matches["categories"].append(category)
        
        total_matches = sum(len(matches) for matches in trend_matches.values())
        
        return {
            "trend_alignment_score": min(total_matches / 3, 1.0),  # 0-1 스케일
            "matched_trends": trend_matches,
            "trend_recommendations": self._get_trend_recommendations(trend_matches)
        }
    
    def _assess_expertise_level(self, text: str) -> Dict[str, Any]:
        """전문성 수준 평가"""
        text_lower = text.lower()
        
        technical_terms = [
            "4c", "gia", "clarity", "cut", "carat", "color",
            "fluorescence", "inclusion", "pavilion", "crown",
            "girdle", "culet", "table", "depth"
        ]
        
        korean_technical_terms = [
            "투명도", "컷팅", "캐럿", "형광성", "내포물",
            "파빌리온", "크라운", "거들", "큘릿", "테이블"
        ]
        
        all_technical = technical_terms + korean_technical_terms
        
        technical_count = sum(1 for term in all_technical if term in text_lower)
        total_words = len(text.split())
        
        if total_words == 0:
            expertise_ratio = 0
        else:
            expertise_ratio = technical_count / total_words
        
        if expertise_ratio > 0.1:
            level = "expert"
        elif expertise_ratio > 0.05:
            level = "knowledgeable"
        elif expertise_ratio > 0.02:
            level = "informed"
        else:
            level = "beginner"
        
        return {
            "expertise_level": level,
            "technical_term_count": technical_count,
            "expertise_ratio": expertise_ratio,
            "detected_terms": [term for term in all_technical if term in text_lower]
        }
    
    def _calculate_mention_confidence(self, term: str, text: str) -> float:
        """언급 신뢰도 계산"""
        # 단순한 빈도 기반 신뢰도
        count = text.lower().count(term.lower())
        return min(count / 3, 1.0)  # 최대 1.0
    
    def _get_sales_recommendation(self, signal_type: str) -> str:
        """판매 시그널별 추천사항"""
        recommendations = {
            "high_intent": "즉시 구매 유도 - 재고 확인, 결제 옵션 안내, 특별 혜택 제공",
            "comparison_shopping": "차별화 포인트 강조 - 독점 장점, 추가 서비스, 가치 제안",
            "objection_handling": "우려사항 해결 - 상세 설명, 보증 정책, 고객 후기 제공",
            "closing_signals": "거래 마무리 - 계약서 준비, 배송 일정 확정, 사후 서비스 안내"
        }
        return recommendations.get(signal_type, "고객 니즈에 맞는 맞춤 상담 제공")
    
    def _get_service_recommendation(self, service_type: str) -> str:
        """서비스 기회별 추천사항"""
        recommendations = {
            "customization": "개인 맞춤 서비스 제안 - 디자인 상담, 각인 옵션, 맞춤 제작",
            "after_service": "지속적 관계 구축 - 정기 점검, 관리 서비스, 업그레이드 제안",
            "education": "전문성 활용 - 교육 자료 제공, 전문 상담, 트렌드 정보 공유"
        }
        return recommendations.get(service_type, "추가 서비스 기회 활용")
    
    def _get_trend_recommendations(self, trend_matches: Dict) -> List[str]:
        """트렌드 기반 추천사항"""
        recommendations = []
        
        if trend_matches["styles"]:
            recommendations.append(f"🔥 트렌드 스타일 활용: {', '.join(trend_matches['styles'])} 제품군 추천")
        
        if trend_matches["materials"]:
            recommendations.append(f"💎 인기 소재 강조: {', '.join(trend_matches['materials'])} 특장점 어필")
        
        if trend_matches["categories"]:
            recommendations.append(f"🆕 신규 카테고리 제안: {', '.join(trend_matches['categories'])} 관련 상품 소개")
        
        if not any(trend_matches.values()):
            recommendations.append("📈 최신 트렌드 정보 제공 - 2024년 인기 스타일 및 소재 안내")
        
        return recommendations
    
    def generate_domain_enhanced_summary(self, analysis_result: Dict[str, Any], text: str) -> Dict[str, Any]:
        """주얼리 도메인 특화 요약 생성"""
        
        # 주얼리 컨텍스트 분석
        jewelry_context = self.analyze_jewelry_context(text)
        
        # 전문성 기반 요약 레벨 결정
        expertise = jewelry_context["expertise_level"]["expertise_level"]
        
        if expertise in ["expert", "knowledgeable"]:
            summary_style = "technical"
        elif expertise == "informed":
            summary_style = "balanced" 
        else:
            summary_style = "accessible"
        
        # 고객 세그먼트 기반 인사이트
        customer_segment = jewelry_context["customer_segment"]["primary_segment"]
        
        # 비즈니스 기회 분석
        business_opportunities = self._identify_business_opportunities(jewelry_context, text)
        
        enhanced_summary = {
            "jewelry_analysis": jewelry_context,
            "business_opportunities": business_opportunities,
            "customer_profile": {
                "segment": customer_segment,
                "expertise_level": expertise,
                "recommended_approach": self._get_approach_strategy(customer_segment, expertise)
            },
            "actionable_insights": self._generate_actionable_insights(jewelry_context, business_opportunities),
            "domain_confidence": self._calculate_domain_confidence(jewelry_context)
        }
        
        return enhanced_summary
    
    def _identify_business_opportunities(self, jewelry_context: Dict, text: str) -> List[Dict[str, Any]]:
        """비즈니스 기회 식별"""
        opportunities = []
        
        # 고가치 제품 기회
        if jewelry_context["market_position"]["target_position"] == "luxury":
            opportunities.append({
                "type": "premium_upsell",
                "description": "럭셔리 제품군 상향 판매 기회",
                "priority": "high",
                "action": "프리미엄 컬렉션 및 한정판 제품 소개"
            })
        
        # 맞춤 서비스 기회
        if any("맞춤" in insight.get("opportunities", []) for insight in jewelry_context["business_insights"] if insight["type"] == "service_opportunity"):
            opportunities.append({
                "type": "customization_service",
                "description": "개인 맞춤 서비스 제공 기회",
                "priority": "medium",
                "action": "맞춤 제작 상담 및 디자인 서비스 제안"
            })
        
        # 교육 및 컨설팅 기회
        if jewelry_context["expertise_level"]["expertise_level"] == "beginner":
            opportunities.append({
                "type": "education_service",
                "description": "고객 교육을 통한 신뢰 구축 기회",
                "priority": "medium",
                "action": "주얼리 교육 자료 제공 및 전문 상담"
            })
        
        return opportunities
    
    def _get_approach_strategy(self, segment: str, expertise: str) -> str:
        """접근 전략 수립"""
        strategies = {
            ("bridal_customers", "beginner"): "감정적 가치 + 기본 교육 중심",
            ("bridal_customers", "informed"): "품질 보증 + 상세 스펙 제공",
            ("luxury_collectors", "expert"): "전문적 논의 + 투자 가치 강조",
            ("fashion_enthusiasts", "informed"): "트렌드 정보 + 스타일링 제안",
            ("gift_buyers", "beginner"): "안전한 선택 + 간편한 프로세스"
        }
        
        return strategies.get((segment, expertise), "고객 맞춤 상담 제공")
    
    def _generate_actionable_insights(self, jewelry_context: Dict, opportunities: List) -> List[str]:
        """실행 가능한 인사이트 생성"""
        insights = []
        
        # 트렌드 기반 인사이트
        if jewelry_context["trend_alignment"]["trend_alignment_score"] > 0.5:
            insights.append("🔥 고객이 최신 트렌드에 관심이 높음 - 신제품 및 한정판 제품 추천")
        
        # 전문성 기반 인사이트
        expertise = jewelry_context["expertise_level"]["expertise_level"]
        if expertise in ["expert", "knowledgeable"]:
            insights.append("🎓 고객의 전문성이 높음 - 기술적 세부사항과 품질 정보 중심 상담")
        elif expertise == "beginner":
            insights.append("📚 고객이 주얼리 초보자임 - 기본 교육과 신뢰 구축 우선")
        
        # 비즈니스 기회 기반 인사이트
        for opp in opportunities:
            if opp["priority"] == "high":
                insights.append(f"💰 {opp['description']} - {opp['action']}")
        
        return insights
    
    def _calculate_domain_confidence(self, jewelry_context: Dict) -> float:
        """도메인 신뢰도 계산"""
        factors = [
            len(jewelry_context["jewelry_mentions"]["products"]) > 0,
            len(jewelry_context["jewelry_mentions"]["materials"]) > 0,
            jewelry_context["customer_segment"]["confidence"] > 0.3,
            jewelry_context["market_position"]["confidence"] > 0.3,
            len(jewelry_context["business_insights"]) > 0
        ]
        
        confidence = sum(factors) / len(factors)
        return confidence

# 전역 주얼리 도메인 엔진
global_jewelry_enhancer = JewelryDomainEnhancer()

def enhance_with_jewelry_domain(analysis_result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """주얼리 도메인 특화 분석 적용"""
    if not text or len(text.strip()) < 10:
        return analysis_result
    
    try:
        # 주얼리 도메인 분석 수행
        domain_analysis = global_jewelry_enhancer.generate_domain_enhanced_summary(analysis_result, text)
        
        # 기존 결과에 도메인 특화 분석 추가
        enhanced_result = analysis_result.copy()
        enhanced_result['jewelry_domain_analysis'] = domain_analysis
        enhanced_result['domain_enhanced'] = True
        
        return enhanced_result
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"주얼리 도메인 분석 실패: {e}")
        analysis_result['domain_enhancement_error'] = str(e)
        return analysis_result

if __name__ == "__main__":
    # 테스트 실행
    print("💎 주얼리 도메인 특화 엔진 테스트")
    
    test_text = "이 다이아몬드 반지 18K 골드로 된 거 가격이 얼마인가요? GIA 인증서도 있나요? 약혼반지로 생각하고 있는데 할인도 가능한지 궁금해요."
    
    enhancer = JewelryDomainEnhancer()
    result = enhancer.analyze_jewelry_context(test_text)
    
    print(f"고객 세그먼트: {result['customer_segment']['primary_segment']}")
    print(f"전문성 레벨: {result['expertise_level']['expertise_level']}")
    print(f"시장 포지션: {result['market_position']['target_position']}")
    print(f"비즈니스 인사이트: {len(result['business_insights'])}개")
    
    domain_summary = enhancer.generate_domain_enhanced_summary({}, test_text)
    print(f"\n실행 인사이트: {domain_summary['actionable_insights']}")