#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
솔로몬드 AI v2.1.3 - AI 분석 정확도 검증 시스템
🚨 긴급 패치: 통합 분석 요약이 사실과 다른 내용 생성 문제 해결

목적: AI 분석 결과의 정확성과 신뢰성을 검증하는 시스템
작성자: 전근혁 (솔로몬드 대표)
생성일: 2025.07.13
"""

import re
import json
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """검증 결과 데이터 클래스"""
    accuracy_score: float
    confidence_level: str
    issues_found: List[str]
    suggestions: List[str]
    verified_facts: List[str]
    questionable_claims: List[str]
    jewelry_term_accuracy: float
    factual_consistency: float

class JewelryFactChecker:
    """주얼리 전문 지식 기반 팩트 체커"""
    
    def __init__(self):
        # 주얼리 전문 용어 사전 (확장된 버전)
        self.jewelry_terms = {
            # 다이아몬드 4C
            "carat": ["캐럿", "ct", "carat", "carats"],
            "clarity": ["투명도", "내포물", "VVS", "VS", "SI", "I", "FL", "IF"],
            "color": ["색상", "D", "E", "F", "G", "H", "I", "J", "K", "컬러"],
            "cut": ["컷", "커팅", "브릴리언트", "라운드", "프린세스", "에메랄드"],
            
            # 보석 종류
            "diamond": ["다이아몬드", "다이야몬드", "diamond"],
            "ruby": ["루비", "홍옥", "ruby"],
            "sapphire": ["사파이어", "청옥", "sapphire"],
            "emerald": ["에메랄드", "녹주석", "emerald"],
            
            # 감정 기관
            "gia": ["GIA", "지아", "미국보석연구소"],
            "ags": ["AGS", "미국보석학회"],
            "grs": ["GRS", "젬리서치", "스위스랩"],
            "ssef": ["SSEF", "스위스감정기관"],
            
            # 가격 및 거래
            "price": ["가격", "price", "cost", "pricing", "비용"],
            "wholesale": ["도매", "wholesale", "도매가"],
            "retail": ["소매", "retail", "소매가"],
            "discount": ["할인", "discount", "디스카운트"],
            
            # 품질 등급
            "grade": ["등급", "grade", "grading"],
            "certificate": ["감정서", "인증서", "certificate", "cert"],
            "natural": ["천연", "natural", "내추럴"],
            "synthetic": ["합성", "synthetic", "인조", "lab-grown"]
        }
        
        # 팩트 체킹 규칙
        self.fact_rules = {
            "price_ranges": {
                "1ct_diamond_vvs1": (8000, 25000),  # USD
                "ruby_premium": (1000, 15000),  # per carat
                "sapphire_premium": (500, 10000)   # per carat
            },
            "impossible_claims": [
                "무료 다이아몬드",
                "100% 할인",
                "가짜 GIA 감정서",
                "인공 천연석"
            ],
            "suspicious_patterns": [
                r"(\d+)캐럿.*(\d+)원",  # 비현실적 가격
                r"100%.*천연.*합성",    # 모순된 표현
                r"무료.*다이아몬드"      # 의심스러운 제안
            ]
        }

    def extract_jewelry_terms(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 주얼리 전문용어 추출"""
        found_terms = {}
        
        for category, terms in self.jewelry_terms.items():
            found = []
            for term in terms:
                if term.lower() in text.lower():
                    found.append(term)
            if found:
                found_terms[category] = found
        
        return found_terms

    def check_price_consistency(self, text: str) -> List[str]:
        """가격 정보의 일관성 검증"""
        issues = []
        
        # 가격 패턴 추출
        price_patterns = [
            r'(\d+(?:,\d+)*)\s*(?:달러|USD|\$)',
            r'(\d+(?:,\d+)*)\s*(?:원|KRW|₩)',
            r'(\d+(?:,\d+)*)\s*(?:유로|EUR|€)'
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            prices.extend([int(p.replace(',', '')) for p in matches])
        
        # 비현실적 가격 검증
        for price in prices:
            if price < 100:  # 너무 저렴한 다이아몬드
                issues.append(f"의심스러운 저가격: {price}")
            elif price > 1000000:  # 너무 비싼 가격
                issues.append(f"비현실적 고가격: {price}")
        
        return issues

    def verify_technical_claims(self, text: str) -> List[str]:
        """기술적 주장의 정확성 검증"""
        issues = []
        
        # 모순된 표현 검출
        contradictions = [
            (r'천연.*합성', "천연과 합성이 동시에 언급됨"),
            (r'무료.*다이아몬드', "무료 다이아몬드는 의심스러움"),
            (r'100%.*할인', "100% 할인은 불가능"),
            (r'가짜.*GIA', "가짜 GIA 감정서 언급")
        ]
        
        for pattern, issue in contradictions:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(issue)
        
        return issues

class ContentConsistencyChecker:
    """다중 소스 간 내용 일관성 검증"""
    
    def __init__(self):
        self.fact_checker = JewelryFactChecker()

    def check_cross_source_consistency(self, sources: List[Dict]) -> Dict[str, Any]:
        """여러 소스 간 일관성 검증"""
        consistency_report = {
            "overall_consistency": 0.0,
            "conflicting_info": [],
            "supporting_info": [],
            "missing_info": []
        }
        
        # 키워드 추출 및 비교
        all_keywords = {}
        for i, source in enumerate(sources):
            content = source.get('content', '')
            keywords = self.fact_checker.extract_jewelry_terms(content)
            all_keywords[f"source_{i}"] = keywords
        
        # 일관성 점수 계산
        if len(sources) > 1:
            consistent_terms = 0
            total_terms = 0
            
            for category in self.fact_checker.jewelry_terms.keys():
                sources_with_category = [s for s in all_keywords.values() if category in s]
                if len(sources_with_category) > 1:
                    total_terms += 1
                    # 같은 카테고리의 용어가 여러 소스에서 일치하는지 확인
                    if len(set(str(terms) for terms in [s[category] for s in sources_with_category])) == 1:
                        consistent_terms += 1
            
            if total_terms > 0:
                consistency_report["overall_consistency"] = consistent_terms / total_terms
        
        return consistency_report

class AccuracyVerifierV213:
    """v2.1.3 AI 분석 정확도 검증 시스템"""
    
    def __init__(self):
        self.fact_checker = JewelryFactChecker()
        self.consistency_checker = ContentConsistencyChecker()
        self.min_accuracy_threshold = 0.7
        
    def verify_analysis_result(self, 
                             original_sources: List[Dict], 
                             analysis_result: Dict) -> VerificationResult:
        """AI 분석 결과의 정확성 종합 검증"""
        
        issues_found = []
        suggestions = []
        verified_facts = []
        questionable_claims = []
        
        # 1. 주얼리 전문용어 정확성 검증
        jewelry_accuracy = self._verify_jewelry_terminology(
            original_sources, analysis_result, issues_found, verified_facts
        )
        
        # 2. 팩트 체킹
        fact_accuracy = self._verify_factual_claims(
            analysis_result, issues_found, questionable_claims
        )
        
        # 3. 다중 소스 간 일관성 검증
        consistency_score = self._verify_cross_source_consistency(
            original_sources, analysis_result, issues_found
        )
        
        # 4. 전체 정확도 계산
        overall_accuracy = (jewelry_accuracy + fact_accuracy + consistency_score) / 3
        
        # 5. 신뢰도 레벨 결정
        confidence_level = self._determine_confidence_level(overall_accuracy)
        
        # 6. 개선 제안 생성
        suggestions = self._generate_suggestions(overall_accuracy, issues_found)
        
        return VerificationResult(
            accuracy_score=overall_accuracy,
            confidence_level=confidence_level,
            issues_found=issues_found,
            suggestions=suggestions,
            verified_facts=verified_facts,
            questionable_claims=questionable_claims,
            jewelry_term_accuracy=jewelry_accuracy,
            factual_consistency=consistency_score
        )
    
    def _verify_jewelry_terminology(self, sources: List[Dict], 
                                  analysis: Dict, 
                                  issues: List[str], 
                                  verified: List[str]) -> float:
        """주얼리 전문용어 정확성 검증"""
        
        # 원본 소스에서 주얼리 용어 추출
        source_terms = set()
        for source in sources:
            content = source.get('content', '') + ' ' + source.get('summary', '')
            terms = self.fact_checker.extract_jewelry_terms(content)
            for category_terms in terms.values():
                source_terms.update(category_terms)
        
        # 분석 결과에서 주얼리 용어 추출
        analysis_text = str(analysis.get('summary', '')) + ' ' + str(analysis.get('jewelry_terms', []))
        analysis_terms = self.fact_checker.extract_jewelry_terms(analysis_text)
        analysis_term_set = set()
        for category_terms in analysis_terms.values():
            analysis_term_set.update(category_terms)
        
        # 정확도 계산
        if source_terms:
            correct_terms = source_terms.intersection(analysis_term_set)
            accuracy = len(correct_terms) / len(source_terms)
            
            # 검증된 용어 기록
            verified.extend(list(correct_terms))
            
            # 누락된 용어 확인
            missing_terms = source_terms - analysis_term_set
            if missing_terms:
                issues.append(f"누락된 주얼리 용어: {', '.join(list(missing_terms)[:5])}")
            
            return accuracy
        else:
            return 0.8  # 기본값
    
    def _verify_factual_claims(self, analysis: Dict, 
                             issues: List[str], 
                             questionable: List[str]) -> float:
        """팩트 체킹 수행"""
        
        analysis_text = json.dumps(analysis, ensure_ascii=False)
        
        # 가격 일관성 검증
        price_issues = self.fact_checker.check_price_consistency(analysis_text)
        issues.extend(price_issues)
        
        # 기술적 주장 검증
        tech_issues = self.fact_checker.verify_technical_claims(analysis_text)
        issues.extend(tech_issues)
        questionable.extend(tech_issues)
        
        # 의심스러운 패턴 검출
        for pattern in self.fact_checker.fact_rules["suspicious_patterns"]:
            if re.search(pattern, analysis_text):
                questionable.append(f"의심스러운 패턴 발견: {pattern}")
        
        # 팩트 정확도 점수 계산
        total_checks = len(price_issues) + len(tech_issues) + len(questionable)
        if total_checks == 0:
            return 0.9  # 문제 없음
        else:
            # 문제가 많을수록 점수 하락
            return max(0.3, 1.0 - (total_checks * 0.2))
    
    def _verify_cross_source_consistency(self, sources: List[Dict], 
                                       analysis: Dict, 
                                       issues: List[str]) -> float:
        """다중 소스 간 일관성 검증"""
        
        if len(sources) <= 1:
            return 0.8  # 단일 소스인 경우 기본값
        
        consistency_report = self.consistency_checker.check_cross_source_consistency(sources)
        consistency_score = consistency_report["overall_consistency"]
        
        # 일관성 문제 보고
        if consistency_score < 0.7:
            issues.append(f"소스 간 일관성 부족: {consistency_score:.1%}")
        
        if consistency_report["conflicting_info"]:
            issues.extend([f"상충 정보: {info}" for info in consistency_report["conflicting_info"]])
        
        return consistency_score
    
    def _determine_confidence_level(self, accuracy: float) -> str:
        """정확도 기반 신뢰도 레벨 결정"""
        if accuracy >= 0.9:
            return "매우 높음"
        elif accuracy >= 0.8:
            return "높음"
        elif accuracy >= 0.7:
            return "보통"
        elif accuracy >= 0.6:
            return "낮음"
        else:
            return "매우 낮음"
    
    def _generate_suggestions(self, accuracy: float, issues: List[str]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        if accuracy < self.min_accuracy_threshold:
            suggestions.append("⚠️ 분석 결과의 정확도가 기준치 이하입니다. 재분석을 권장합니다.")
        
        if any("가격" in issue for issue in issues):
            suggestions.append("💰 가격 정보를 실시간 시세와 재확인하세요.")
        
        if any("용어" in issue for issue in issues):
            suggestions.append("💎 주얼리 전문용어 사전을 업데이트하세요.")
        
        if any("일관성" in issue for issue in issues):
            suggestions.append("🔄 여러 소스 간 내용 불일치를 확인하세요.")
        
        if len(issues) > 5:
            suggestions.append("🚨 다수의 문제가 발견되었습니다. 원본 데이터를 재확인하세요.")
        
        return suggestions

# 🚨 긴급 패치용 간단 래퍼 함수
def emergency_verify_analysis(original_content: str, analysis_result: Dict) -> Dict:
    """긴급 패치용 간단 검증 함수"""
    try:
        verifier = AccuracyVerifierV213()
        
        # 원본 내용을 소스 형태로 변환
        sources = [{
            'content': original_content,
            'type': 'mixed',
            'summary': ''
        }]
        
        # 검증 수행
        result = verifier.verify_analysis_result(sources, analysis_result)
        
        return {
            "accuracy_score": result.accuracy_score,
            "confidence_level": result.confidence_level,
            "is_reliable": result.accuracy_score >= 0.7,
            "issues_count": len(result.issues_found),
            "suggestions": result.suggestions[:3],  # 최대 3개만
            "verification_status": "✅ 검증 완료" if result.accuracy_score >= 0.7 else "⚠️ 검증 실패"
        }
        
    except Exception as e:
        logger.error(f"검증 시스템 오류: {e}")
        return {
            "accuracy_score": 0.5,
            "confidence_level": "알 수 없음",
            "is_reliable": False,
            "issues_count": 1,
            "suggestions": ["검증 시스템 오류가 발생했습니다."],
            "verification_status": "❌ 검증 오류"
        }

# 테스트 함수
def test_verification_system():
    """검증 시스템 테스트"""
    
    # 테스트 데이터
    test_content = "1캐럿 VVS1 다이아몬드 가격은 $15,000 입니다. GIA 감정서가 포함됩니다."
    test_analysis = {
        "summary": "다이아몬드 거래에 관한 내용입니다. 가격은 합리적입니다.",
        "jewelry_terms": ["다이아몬드", "캐럿", "VVS1", "GIA"],
        "key_topics": ["가격", "감정서"]
    }
    
    # 검증 실행
    result = emergency_verify_analysis(test_content, test_analysis)
    
    print("🧪 검증 시스템 테스트 결과:")
    print(f"정확도: {result['accuracy_score']:.1%}")
    print(f"신뢰도: {result['confidence_level']}")
    print(f"상태: {result['verification_status']}")
    
    return result

if __name__ == "__main__":
    # 테스트 실행
    test_verification_system()
