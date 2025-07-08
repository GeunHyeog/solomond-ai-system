"""
🧠 주얼리 AI 분석 엔진 v2.0 - 인사이트 생성 시스템
Phase 2 Week 3 Day 3: 업계 특화 텍스트 분석과 비즈니스 인사이트 자동 생성

작성자: 전근혁 (솔로몬드 대표)
목적: 주얼리 업계 회의, 세미나, 상담 내용을 AI가 분석하여 실무 인사이트 제공
통합: 기존 solomond-ai-system과 완전 호환
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import statistics

@dataclass
class JewelryInsight:
    """주얼리 업계 인사이트 데이터 클래스"""
    category: str           # 카테고리 (가격, 품질, 트렌드, 고객반응 등)
    insight: str           # 핵심 인사이트
    confidence: float      # 신뢰도 (0-1)
    evidence: List[str]    # 근거 텍스트들
    priority: str          # 우선순위 (높음/중간/낮음)
    action_items: List[str] # 액션 아이템

class JewelryAIEngine:
    """주얼리 업계 특화 AI 분석 엔진"""
    
    def __init__(self):
        self.jewelry_patterns = self._load_jewelry_patterns()
        self.business_keywords = self._load_business_keywords()
        self.sentiment_patterns = self._load_sentiment_patterns()
        
    def _load_jewelry_patterns(self) -> Dict:
        """주얼리 업계 특화 패턴 로드 (기존 jewelry_database.py와 연동)"""
        return {
            # 보석 종류 및 등급
            'gemstones': {
                'diamond': ['다이아몬드', '다이야몬드', 'diamond', '다이어몬드', '다이몬드'],
                'ruby': ['루비', 'ruby', '홍옥', '루비석'],
                'sapphire': ['사파이어', 'sapphire', '청옥', '새파이어'],
                'emerald': ['에메랄드', 'emerald', '녹주석', '에메롤드'],
                'pearl': ['진주', 'pearl', '펄', '진주석']
            },
            
            # 4C 평가 기준 (GIA 표준)
            'four_c': {
                'cut': ['컷', 'cut', '커팅', '연마', '브릴리언트', '라운드', '프린세스'],
                'color': ['컬러', 'color', '색상', '등급', 'D급', 'E급', 'F급', 'G급', 'H급'],
                'clarity': ['클래리티', 'clarity', '투명도', '내포물', 'FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2'],
                'carat': ['캐럿', 'carat', '중량', 'ct', '무게', '크기']
            },
            
            # 가격 관련 (비즈니스 실무)
            'pricing': {
                'wholesale': ['도매가', '도매', 'wholesale', '벌크', '대량', '도매 가격'],
                'retail': ['소매가', '소매', 'retail', '개별', '단품', '소매 가격'],
                'margin': ['마진', 'margin', '수익률', '이익', '마크업', '수익성'],
                'discount': ['할인', 'discount', '세일', '프로모션', '특가', '바겐']
            },
            
            # 시장 트렌드 (2025년 업데이트)
            'trends': {
                'fashion': ['유행', '트렌드', 'trend', '패션', '스타일', '모던', '빈티지'],
                'seasonal': ['시즌', '계절', '봄', '여름', '가을', '겨울', '웨딩시즌'],
                'generation': ['세대', 'MZ', '밀레니얼', '젊은층', 'Z세대', '시니어']
            },
            
            # 감정기관 및 인증 (국제 표준)
            'certification': {
                'institutes': ['GIA', 'AGS', 'GÜBELIN', 'SSEF', '한국보석감정원'],
                'grades': ['인증서', '감정서', '보고서', 'certificate', 'report']
            }
        }
    
    def _load_business_keywords(self) -> Dict:
        """비즈니스 키워드 패턴 (실무 중심)"""
        return {
            'sales': ['매출', '판매', '수익', '실적', '성과', '영업', '거래'],
            'customer': ['고객', '구매자', '손님', '클라이언트', '바이어', '소비자'],
            'inventory': ['재고', '물량', '보유', '입고', '출고', '스톡', '인벤토리'],
            'competition': ['경쟁', '경쟁사', '라이벌', '시장점유율', '포지셔닝'],
            'marketing': ['마케팅', '홍보', '광고', '브랜딩', '프로모션', '캠페인'],
            'quality': ['품질', '퀄리티', '등급', '인증', '감정', '표준', '검증'],
            'export': ['수출', '무역', '글로벌', '해외', '국제', '바이어'],
            'technology': ['기술', '디지털', 'AI', '혁신', '자동화', '시스템']
        }
    
    def _load_sentiment_patterns(self) -> Dict:
        """감정 분석 패턴 (주얼리 업계 맥락)"""
        return {
            'positive': [
                '좋다', '만족', '훌륭', '우수', '성공', '증가', '상승', '호조',
                '성장', '개선', '긍정', '효과적', '우수', '탁월', '인기',
                '호평', '성공적', '만족스럽', '뛰어나', '강세'
            ],
            'negative': [
                '나쁘다', '불만', '문제', '실패', '감소', '하락', '우려',
                '약세', '부정', '어려움', '위험', '손실', '악화', '비관',
                '염려', '실망', '곤란', '위기', '침체'
            ],
            'neutral': [
                '보통', '일반적', '평균', '표준', '정상', '안정',
                '유지', '변동없음', '기본', '중간', '적정'
            ]
        }
    
    def analyze_text(self, text: str, context: str = "general") -> Dict:
        """텍스트 종합 분석 (기존 시스템과 연동)"""
        
        # 1. 기본 전처리
        cleaned_text = self._preprocess_text(text)
        
        # 2. 주얼리 업계 엔티티 추출
        entities = self._extract_jewelry_entities(cleaned_text)
        
        # 3. 가격 정보 분석
        price_analysis = self._analyze_pricing(cleaned_text)
        
        # 4. 감정 분석
        sentiment = self._analyze_sentiment(cleaned_text)
        
        # 5. 비즈니스 인사이트 생성
        insights = self._generate_insights(cleaned_text, entities, price_analysis, sentiment)
        
        # 6. 액션 아이템 추출
        action_items = self._extract_action_items(cleaned_text)
        
        # 7. 키워드 추출 (업계 특화)
        keywords = self._extract_jewelry_keywords(cleaned_text)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'original_text': text,
            'cleaned_text': cleaned_text,
            'entities': entities,
            'price_analysis': price_analysis,
            'sentiment': sentiment,
            'insights': [asdict(insight) for insight in insights],
            'action_items': action_items,
            'keywords': keywords,
            'summary': self._generate_summary(insights),
            'confidence_score': self._calculate_overall_confidence(insights),
            'recommendations': self._generate_recommendations(insights)
        }
    
    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (주얼리 업계 특화)"""
        # 숫자와 단위 정규화
        text = re.sub(r'(\d+)\s*캐럿', r'\1ct', text)
        text = re.sub(r'(\d+)\s*만원', r'\1만원', text)
        text = re.sub(r'(\d+)\s*달러', r'\1달러', text)
        text = re.sub(r'(\d+)\s*위안', r'\1위안', text)
        
        # 주얼리 브랜드명 정규화
        brand_mapping = {
            '티파니': 'Tiffany & Co.',
            '까르띠에': 'Cartier',
            '불가리': 'Bulgari',
            '쇼메': 'Chaumet',
            '반클리프': 'Van Cleef & Arpels',
            '해리윈스턴': 'Harry Winston',
            '그라프': 'Graff'
        }
        
        for korean, english in brand_mapping.items():
            text = text.replace(korean, english)
        
        # 감정기관명 정규화
        cert_mapping = {
            '지아이에이': 'GIA',
            '에이지에스': 'AGS',
            '귀벌린': 'GÜBELIN',
            '에스에스이에프': 'SSEF'
        }
        
        for korean, english in cert_mapping.items():
            text = text.replace(korean, english)
        
        return text
    
    def _extract_jewelry_entities(self, text: str) -> Dict:
        """주얼리 업계 엔티티 추출 (고도화)"""
        entities = {
            'gemstones': [],
            'grades': [],
            'certifications': [],
            'brands': [],
            'prices': [],
            'measurements': [],
            'quality_indicators': []
        }
        
        # 보석 종류 추출 (중복 제거)
        found_gemstones = set()
        for gem_type, keywords in self.jewelry_patterns['gemstones'].items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    found_gemstones.add(gem_type)
        entities['gemstones'] = list(found_gemstones)
        
        # 등급 정보 추출 (정규표현식 고도화)
        grade_patterns = [
            r'([A-Z]+)\s*급',  # D급, VVS급 등
            r'([0-9\.]+)\s*ct',  # 캐럿 정보
            r'(FL|IF|VVS[12]|VS[12]|SI[12]|I[123])',  # 클래리티
            r'([DEF])\s*컬러',  # 컬러 등급
            r'(Excellent|Very Good|Good|Fair|Poor)',  # 컷 등급
            r'(3EX|Triple\s*Excellent)'  # 최고 등급
        ]
        
        for pattern in grade_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['grades'].extend([match for match in matches if match])
        
        # 감정기관 및 인증서 정보
        for institute in self.jewelry_patterns['certification']['institutes']:
            if institute in text:
                entities['certifications'].append(institute)
        
        # 가격 정보 추출 (다양한 통화)
        price_patterns = [
            (r'(\d+(?:,\d+)*)\s*만원', '만원'),
            (r'\$(\d+(?:,\d+)*)', 'USD'),
            (r'(\d+(?:,\d+)*)\s*달러', 'USD'),
            (r'(\d+(?:,\d+)*)\s*위안', 'CNY'),
            (r'(\d+(?:,\d+)*)\s*엔', 'JPY'),
            (r'(\d+(?:,\d+)*)\s*원', 'KRW')
        ]
        
        for pattern, currency in price_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities['prices'].append({
                    'value': match.replace(',', ''),
                    'currency': currency,
                    'context': self._get_price_context(text, match)
                })
        
        return entities
    
    def _analyze_pricing(self, text: str) -> Dict:
        """가격 분석 (주얼리 시장 특화)"""
        price_analysis = {
            'mentioned_prices': [],
            'price_trends': None,
            'price_range': None,
            'margin_info': {},
            'competitive_pricing': {},
            'value_assessment': None
        }
        
        # 가격 추출 및 분석
        price_patterns = [
            (r'(\d+(?:,\d+)*)\s*만원', '만원'),
            (r'\$(\d+(?:,\d+)*)', '달러'),
            (r'(\d+(?:,\d+)*)\s*원', '원')
        ]
        
        prices = []
        for pattern, unit in price_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    price_value = int(match.replace(',', ''))
                    prices.append(price_value)
                    price_analysis['mentioned_prices'].append({
                        'value': price_value,
                        'unit': unit,
                        'context': self._get_price_context(text, match)
                    })
                except ValueError:
                    continue
        
        # 가격 범위 분석
        if prices:
            price_analysis['price_range'] = {
                'min': min(prices),
                'max': max(prices),
                'average': sum(prices) / len(prices),
                'count': len(prices)
            }
            
            # 가격대 분류
            avg_price = price_analysis['price_range']['average']
            if avg_price > 50000000:  # 5천만원 이상
                price_analysis['value_assessment'] = 'ultra_luxury'
            elif avg_price > 10000000:  # 1천만원 이상
                price_analysis['value_assessment'] = 'luxury'
            elif avg_price > 1000000:  # 100만원 이상
                price_analysis['value_assessment'] = 'premium'
            else:
                price_analysis['value_assessment'] = 'standard'
        
        # 가격 트렌드 분석
        trend_keywords = {
            'increasing': ['상승', '오름', '인상', '증가', '올라', '비싸', '급등'],
            'decreasing': ['하락', '내림', '인하', '감소', '떨어', '싸', '급락'],
            'stable': ['안정', '유지', '변동없음', '고정', '일정']
        }
        
        for trend, keywords in trend_keywords.items():
            if any(keyword in text for keyword in keywords):
                price_analysis['price_trends'] = trend
                break
        
        return price_analysis
    
    def _get_price_context(self, text: str, price: str) -> str:
        """가격의 맥락 정보 추출"""
        price_index = text.find(price)
        if price_index != -1:
            start = max(0, price_index - 50)
            end = min(len(text), price_index + 50)
            context = text[start:end].strip()
            return context
        return ""
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """감정 분석 (주얼리 업계 맥락 고려)"""
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_details = {'positive': [], 'negative': [], 'neutral': []}
        
        # 감정 키워드 매칭 및 점수 계산
        for sentiment_type, keywords in self.sentiment_patterns.items():
            for keyword in keywords:
                count = text.lower().count(keyword.lower())
                if count > 0:
                    sentiment_scores[sentiment_type] += count
                    sentiment_details[sentiment_type].extend([keyword] * count)
        
        # 비즈니스 맥락 고려 (주얼리 업계 특화)
        business_positive = ['매출 증가', '고객 만족', '품질 향상', '시장 확대']
        business_negative = ['재고 과다', '품질 문제', '매출 감소', '경쟁 심화']
        
        for phrase in business_positive:
            if phrase in text:
                sentiment_scores['positive'] += 2  # 가중치 적용
        
        for phrase in business_negative:
            if phrase in text:
                sentiment_scores['negative'] += 2  # 가중치 적용
        
        # 감정 비율 계산
        total_sentiment = sum(sentiment_scores.values())
        if total_sentiment > 0:
            sentiment_percentages = {
                k: (v / total_sentiment) * 100 
                for k, v in sentiment_scores.items()
            }
        else:
            sentiment_percentages = {'positive': 33.3, 'negative': 33.3, 'neutral': 33.3}
        
        # 주요 감정 결정 (임계값 적용)
        primary_sentiment = max(sentiment_percentages, key=sentiment_percentages.get)
        confidence = max(sentiment_percentages.values()) / 100
        
        # 감정 강도 분류
        if confidence > 0.7:
            intensity = 'strong'
        elif confidence > 0.5:
            intensity = 'moderate'
        else:
            intensity = 'weak'
        
        return {
            'primary': primary_sentiment,
            'scores': sentiment_percentages,
            'confidence': confidence,
            'intensity': intensity,
            'details': sentiment_details,
            'business_context': self._analyze_business_sentiment(text)
        }
    
    def _analyze_business_sentiment(self, text: str) -> Dict:
        """비즈니스 맥락 감정 분석"""
        business_indicators = {
            'growth': ['성장', '확장', '증가', '향상', '개선'],
            'concern': ['위험', '문제', '우려', '하락', '감소'],
            'opportunity': ['기회', '가능성', '잠재력', '신시장', '혁신'],
            'stability': ['안정', '유지', '지속', '견고', '일관']
        }
        
        business_sentiment = {}
        for category, keywords in business_indicators.items():
            score = sum(text.lower().count(keyword.lower()) for keyword in keywords)
            business_sentiment[category] = score
        
        return business_sentiment
    
    def _generate_insights(self, text: str, entities: Dict, price_analysis: Dict, sentiment: Dict) -> List[JewelryInsight]:
        """비즈니스 인사이트 생성 (고도화)"""
        insights = []
        
        # 1. 가격 관련 인사이트
        if price_analysis['mentioned_prices']:
            avg_price = price_analysis['price_range']['average']
            value_assessment = price_analysis['value_assessment']
            
            if value_assessment == 'ultra_luxury':
                insight = JewelryInsight(
                    category="초고가 시장 분석",
                    insight=f"초고급 제품 중심 논의 (평균 {avg_price:,.0f}원) - VIP 마케팅 필요",
                    confidence=0.95,
                    evidence=[f"평균 가격: {avg_price:,.0f}원", f"가격대: {value_assessment}"],
                    priority="최고",
                    action_items=["VIP 고객 전용 서비스 개발", "프리미엄 마케팅 전략 수립", "고급 매장 환경 조성"]
                )
                insights.append(insight)
            elif value_assessment == 'luxury':
                insight = JewelryInsight(
                    category="럭셔리 시장 동향",
                    insight=f"고급 제품 선호 트렌드 (평균 {avg_price:,.0f}원)",
                    confidence=0.9,
                    evidence=[f"언급된 가격 범위: {price_analysis['price_range']}"],
                    priority="높음",
                    action_items=["럭셔리 라인 확대", "브랜드 포지셔닝 강화"]
                )
                insights.append(insight)
        
        # 2. 보석 종류별 트렌드 인사이트
        if entities['gemstones']:
            gemstone_counts = Counter(entities['gemstones'])
            if gemstone_counts:
                most_mentioned = gemstone_counts.most_common(1)[0]
                
                # 보석별 특화 인사이트
                gemstone_insights = {
                    'diamond': "다이아몬드 시장 활성화 - 4C 등급 관리 중요",
                    'ruby': "루비 관심 증가 - 아시아 시장 확대 기회",
                    'sapphire': "사파이어 수요 상승 - 컬러 다양성 어필",
                    'emerald': "에메랄드 프리미엄 시장 - 품질 인증 강화 필요",
                    'pearl': "진주 클래식 회귀 - 젊은층 마케팅 필요"
                }
                
                insight_text = gemstone_insights.get(most_mentioned[0], 
                                                   f"{most_mentioned[0]} 중심 논의")
                
                insight = JewelryInsight(
                    category="제품 트렌드 분석",
                    insight=f"{insight_text} ({most_mentioned[1]}회 언급)",
                    confidence=0.85,
                    evidence=[f"언급된 보석: {list(gemstone_counts.keys())}"],
                    priority="높음",
                    action_items=[
                        f"{most_mentioned[0]} 재고 전략적 확보",
                        "관련 제품 라인 확장 검토",
                        "전문 스태프 교육 강화"
                    ]
                )
                insights.append(insight)
        
        # 3. 감정 기반 인사이트 (고도화)
        if sentiment['confidence'] > 0.6:
            if sentiment['primary'] == 'positive':
                business_context = sentiment['business_context']
                
                if business_context.get('growth', 0) > 0:
                    insight = JewelryInsight(
                        category="시장 기회",
                        insight="긍정적 성장 신호 감지 - 확장 전략 수립 적기",
                        confidence=sentiment['confidence'],
                        evidence=["긍정적 키워드 다수 발견", "성장 관련 언급 포함"],
                        priority="높음",
                        action_items=["시장 확장 계획 수립", "투자 계획 검토", "마케팅 예산 증액 고려"]
                    )
                    insights.append(insight)
                else:
                    insight = JewelryInsight(
                        category="고객 반응",
                        insight="긍정적 고객 피드백 확인 - 성공 사례 활용",
                        confidence=sentiment['confidence'],
                        evidence=["긍정적 키워드 다수 발견"],
                        priority="중간",
                        action_items=["성공 사례 문서화", "마케팅 소재 활용", "고객 추천 프로그램 강화"]
                    )
                    insights.append(insight)
            
            elif sentiment['primary'] == 'negative':
                business_context = sentiment['business_context']
                
                if business_context.get('concern', 0) > 0:
                    insight = JewelryInsight(
                        category="리스크 관리",
                        insight="시장 우려사항 감지 - 즉시 대응 필요",
                        confidence=sentiment['confidence'],
                        evidence=["부정적 키워드 및 우려사항 발견"],
                        priority="최고",
                        action_items=["긴급 대응팀 구성", "리스크 분석 실시", "고객 소통 강화"]
                    )
                    insights.append(insight)
                else:
                    insight = JewelryInsight(
                        category="개선 기회",
                        insight="개선 필요 영역 식별 - 품질 향상 기회",
                        confidence=sentiment['confidence'],
                        evidence=["부정적 피드백 발견"],
                        priority="높음",
                        action_items=["문제점 상세 분석", "개선 방안 수립", "고객 만족도 조사"]
                    )
                    insights.append(insight)
        
        # 4. 비즈니스 기회 인사이트 (확장)
        business_opportunities = self._identify_business_opportunities(text, entities)
        insights.extend(business_opportunities)
        
        # 5. 기술 혁신 기회
        tech_opportunities = self._identify_tech_opportunities(text)
        insights.extend(tech_opportunities)
        
        return insights
    
    def _identify_business_opportunities(self, text: str, entities: Dict) -> List[JewelryInsight]:
        """비즈니스 기회 식별 (확장)"""
        opportunities = []
        
        # 시장 확장 기회
        market_keywords = ['신시장', '확장', '글로벌', '수출', '새로운', '해외', '국제']
        if any(keyword in text for keyword in market_keywords):
            opportunity = JewelryInsight(
                category="시장 확장",
                insight="해외 시장 진출 기회 - 글로벌 전략 수립",
                confidence=0.75,
                evidence=["시장 확장 관련 키워드 발견"],
                priority="높음",
                action_items=["해외 시장 조사", "수출 전략 수립", "국제 인증 준비"]
            )
            opportunities.append(opportunity)
        
        # 디지털 전환 기회
        digital_keywords = ['온라인', '디지털', '플랫폼', 'AI', '자동화', '시스템']
        if any(keyword in text for keyword in digital_keywords):
            opportunity = JewelryInsight(
                category="디지털 혁신",
                insight="디지털 전환 기회 - 기술 도입 검토",
                confidence=0.7,
                evidence=["디지털 관련 키워드 발견"],
                priority="중간",
                action_items=["디지털 전환 로드맵 수립", "기술 도입 비용 분석", "직원 교육 계획"]
            )
            opportunities.append(opportunity)
        
        # 고객 세그먼트 기회
        customer_keywords = ['MZ', '젊은층', '시니어', '밀레니얼', '신혼부부']
        if any(keyword in text for keyword in customer_keywords):
            opportunity = JewelryInsight(
                category="고객 세분화",
                insight="타겟 고객층 확장 기회 - 세분화 전략",
                confidence=0.8,
                evidence=["특정 고객층 언급"],
                priority="중간",
                action_items=["고객 세분화 분석", "맞춤형 상품 개발", "타겟 마케팅 전략"]
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_tech_opportunities(self, text: str) -> List[JewelryInsight]:
        """기술 혁신 기회 식별"""
        opportunities = []
        
        # AI/ML 기회
        ai_keywords = ['AI', '인공지능', '머신러닝', '딥러닝', '자동화']
        if any(keyword in text for keyword in ai_keywords):
            opportunity = JewelryInsight(
                category="AI 혁신",
                insight="AI 기술 활용 기회 - 업무 자동화 가능",
                confidence=0.7,
                evidence=["AI 관련 기술 언급"],
                priority="중간",
                action_items=["AI 도입 가능성 검토", "자동화 프로세스 설계", "ROI 분석"]
            )
            opportunities.append(opportunity)
        
        # 블록체인/인증 기회
        blockchain_keywords = ['블록체인', '인증', '추적', '원산지', '진위']
        if any(keyword in text for keyword in blockchain_keywords):
            opportunity = JewelryInsight(
                category="블록체인 인증",
                insight="블록체인 기반 인증 시스템 도입 기회",
                confidence=0.6,
                evidence=["인증/추적 관련 언급"],
                priority="낮음",
                action_items=["블록체인 기술 조사", "인증 시스템 설계", "파일럿 프로젝트 계획"]
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    def _extract_action_items(self, text: str) -> List[str]:
        """액션 아이템 추출 (고도화)"""
        action_patterns = [
            r'해야\s*한다',
            r'필요하다',
            r'검토\s*하[다겠자]',
            r'계획\s*[이을하한]',
            r'준비\s*[하할해]',
            r'진행\s*[하할해]',
            r'개선\s*[하할해]',
            r'도입\s*[하할해]',
            r'확대\s*[하할해]',
            r'강화\s*[하할해]'
        ]
        
        action_items = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 너무 짧은 문장 제외
                continue
                
            for pattern in action_patterns:
                if re.search(pattern, sentence):
                    # 액션 아이템 정제
                    cleaned_action = self._clean_action_item(sentence)
                    if cleaned_action and len(cleaned_action) > 5:
                        action_items.append(cleaned_action)
                    break
        
        # 중복 제거 및 우선순위 정렬
        unique_actions = list(dict.fromkeys(action_items))  # 순서 유지하며 중복 제거
        return unique_actions[:8]  # 상위 8개만 반환
    
    def _clean_action_item(self, sentence: str) -> str:
        """액션 아이템 정제"""
        # 불필요한 접두사 제거
        prefixes_to_remove = ['그래서', '따라서', '그러므로', '이에', '또한', '그리고']
        for prefix in prefixes_to_remove:
            if sentence.startswith(prefix):
                sentence = sentence[len(prefix):].strip()
        
        # 문장 끝 정리
        sentence = sentence.rstrip('다고하겠습니다.')
        
        return sentence
    
    def _extract_jewelry_keywords(self, text: str) -> Dict:
        """주얼리 관련 키워드 추출"""
        keywords = {
            'gemstones': [],
            'technical': [],
            'business': [],
            'quality': [],
            'market': []
        }
        
        # 각 카테고리별 키워드 추출
        for category, patterns in self.jewelry_patterns.items():
            if category in keywords:
                for subcategory, terms in patterns.items():
                    for term in terms:
                        if term.lower() in text.lower():
                            keywords[category].append(term)
        
        # 비즈니스 키워드
        for keyword_type, terms in self.business_keywords.items():
            for term in terms:
                if term in text:
                    keywords['business'].append(term)
        
        # 중복 제거
        for category in keywords:
            keywords[category] = list(set(keywords[category]))
        
        return keywords
    
    def _generate_summary(self, insights: List[JewelryInsight]) -> str:
        """인사이트 요약 생성 (고도화)"""
        if not insights:
            return "분석할 인사이트가 없습니다."
        
        high_priority = [i for i in insights if i.priority in ["최고", "높음"]]
        categories = list(set(i.category for i in insights))
        
        summary = f"총 {len(insights)}개 인사이트 발견. "
        summary += f"주요 분야: {', '.join(categories[:3])}{'...' if len(categories) > 3 else ''}. "
        
        if high_priority:
            summary += f"긴급/우선 대응 필요 항목 {len(high_priority)}개. "
        
        # 신뢰도 분석
        avg_confidence = sum(i.confidence for i in insights) / len(insights)
        summary += f"평균 신뢰도 {avg_confidence:.1%}."
        
        return summary
    
    def _calculate_overall_confidence(self, insights: List[JewelryInsight]) -> float:
        """전체 신뢰도 계산 (가중평균)"""
        if not insights:
            return 0.0
        
        # 우선순위별 가중치
        priority_weights = {"최고": 1.0, "높음": 0.8, "중간": 0.6, "낮음": 0.4}
        
        weighted_sum = 0
        total_weight = 0
        
        for insight in insights:
            weight = priority_weights.get(insight.priority, 0.5)
            weighted_sum += insight.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, insights: List[JewelryInsight]) -> List[str]:
        """추천 사항 생성"""
        recommendations = []
        
        # 우선순위 높은 인사이트 기반 추천
        high_priority_insights = [i for i in insights if i.priority in ["최고", "높음"]]
        
        for insight in high_priority_insights[:3]:  # 상위 3개
            if insight.action_items:
                recommendations.append(f"[{insight.category}] {insight.action_items[0]}")
        
        # 일반적 추천사항
        general_recommendations = [
            "정기적인 시장 동향 모니터링 강화",
            "고객 피드백 수집 시스템 개선",
            "직원 전문성 교육 확대"
        ]
        
        # 기존 추천과 중복되지 않는 일반 추천 추가
        for rec in general_recommendations:
            if len(recommendations) < 5 and not any(rec in existing for existing in recommendations):
                recommendations.append(rec)
        
        return recommendations[:5]
    
    def generate_business_report(self, analysis_results: Dict) -> str:
        """비즈니스 리포트 생성 (고도화)"""
        insights = analysis_results['insights']
        entities = analysis_results['entities']
        sentiment = analysis_results['sentiment']
        price_analysis = analysis_results['price_analysis']
        
        report = f"""
🧠 주얼리 AI 분석 리포트 v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 생성일시: {analysis_results['timestamp'][:19]}
🎯 신뢰도: {analysis_results['confidence_score']:.1%}
📊 분석 맥락: {analysis_results['context']}

💎 핵심 요약
{analysis_results['summary']}

🔍 발견된 요소
├─ 보석 종류: {', '.join(entities['gemstones']) if entities['gemstones'] else '언급 없음'}
├─ 품질 등급: {', '.join(str(g) for g in entities['grades'][:3]) if entities['grades'] else '언급 없음'}
├─ 인증기관: {', '.join(entities['certifications']) if entities['certifications'] else '언급 없음'}
└─ 가격 정보: {len(entities['prices'])}건 발견

📈 감정 분석
주요 감정: {sentiment['primary']} ({sentiment['confidence']:.1%} 신뢰도, {sentiment['intensity']} 강도)
비즈니스 맥락: {max(sentiment['business_context'].items(), key=lambda x: x[1])[0] if sentiment['business_context'] else 'N/A'}
"""

        # 가격 분석 추가
        if price_analysis.get('price_range'):
            pr = price_analysis['price_range']
            report += f"""
💰 가격 분석
├─ 평균: {pr['average']:,.0f}원 ({price_analysis['value_assessment']})
├─ 범위: {pr['min']:,.0f}원 ~ {pr['max']:,.0f}원
└─ 트렌드: {price_analysis.get('price_trends', '불명')}
"""

        # 주요 인사이트
        report += f"\n🎯 주요 인사이트\n"
        for i, insight in enumerate(insights[:5], 1):
            priority_emoji = {"최고": "🔴", "높음": "🟠", "중간": "🟡", "낮음": "🟢"}.get(insight['priority'], "⚪")
            report += f"{i}. {priority_emoji} [{insight['category']}] {insight['insight']}\n"
            if insight['action_items']:
                report += f"   💡 권장 액션: {insight['action_items'][0]}\n"

        # 액션 아이템
        if analysis_results['action_items']:
            report += f"\n📋 액션 아이템\n"
            for i, item in enumerate(analysis_results['action_items'][:5], 1):
                report += f"{i}. {item}\n"

        # 추천사항
        if analysis_results.get('recommendations'):
            report += f"\n💡 추천 사항\n"
            for i, rec in enumerate(analysis_results['recommendations'][:3], 1):
                report += f"{i}. {rec}\n"

        report += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🏢 솔로몬드 AI 시스템 | 주얼리 업계 특화 분석 플랫폼"
        
        return report

# 기존 시스템과의 통합을 위한 호환성 함수
def integrate_with_existing_system(text: str, enhanced_text: str = None) -> Dict:
    """기존 jewelry_enhancer.py와 연동"""
    ai_engine = JewelryAIEngine()
    
    # 기존 시스템에서 처리된 텍스트가 있으면 우선 사용
    analysis_text = enhanced_text if enhanced_text else text
    
    results = ai_engine.analyze_text(analysis_text, context="integrated_analysis")
    
    # 기존 시스템과 호환되는 형태로 결과 포맷팅
    return {
        'ai_insights': results,
        'business_report': ai_engine.generate_business_report(results),
        'compatibility': 'v2.0_integrated'
    }

# 테스트 및 예시
if __name__ == "__main__":
    # AI 엔진 초기화
    ai_engine = JewelryAIEngine()
    
    # 테스트 텍스트 (실제 주얼리 업계 회의 내용)
    test_text = """
    오늘 2025년 상반기 다이아몬드 시장 동향 회의에서 논의된 내용입니다.
    3캐럿 D컬러 VVS1 등급 GIA 인증 다이아몬드 가격이 5천만원대로 상승했습니다.
    특히 Tiffany와 Cartier 등 럭셔리 브랜드에서 프리미엄 제품 선호가 높아지고 있습니다.
    고객들의 반응은 대체로 긍정적이며, 특히 MZ세대 고객층에서 관심이 높습니다.
    루비와 사파이어 매출도 20% 증가했으며, 해외 수출도 늘어나고 있습니다.
    다음 분기에는 에메랄드 라인 확장을 검토해야 하고, AI 기반 재고관리 시스템 도입도 필요합니다.
    온라인 플랫폼 강화를 통해 디지털 전환을 가속화할 계획입니다.
    """
    
    # 분석 실행
    print("🧠 주얼리 AI 분석 엔진 v2.0 테스트")
    print("=" * 50)
    
    results = ai_engine.analyze_text(test_text, context="2025_상반기_시장동향회의")
    
    # 리포트 생성 및 출력
    report = ai_engine.generate_business_report(results)
    print(report)
    
    print("\n" + "=" * 50)
    print("📊 상세 분석 결과 (JSON)")
    print(json.dumps(results, ensure_ascii=False, indent=2))