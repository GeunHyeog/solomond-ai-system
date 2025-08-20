#!/usr/bin/env python3
"""
💡 SOLOMOND AI 자동 인사이트 생성 엔진
Automated Insight Generation Engine with Advanced Analytics

🎯 주요 기능:
1. 통합 분석 - 멀티모달 + 온톨로지 + 외부 데이터
2. 실시간 인사이트 생성 - 분석 중 즉시 패턴 탐지
3. 컨텍스트 인식 추론 - 상황 맞춤 인사이트
4. 신뢰도 기반 랭킹 - 품질 보증 시스템
5. 액션 가능한 제안 - 구체적 실행 방안 제시
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
from collections import defaultdict, Counter
import re
import statistics

# 내부 모듈 import
try:
    from .knowledge_ontology import KnowledgeOntology, KnowledgeNode
    from .multimodal_pipeline import MultimodalPipeline, MultimodalResult
except ImportError:
    try:
        from core.knowledge_ontology import KnowledgeOntology, KnowledgeNode
        from core.multimodal_pipeline import MultimodalPipeline, MultimodalResult
    except ImportError as e:
        print(f"⚠️ 내부 모듈 import 실패: {e}")
        
        # 기본 폴백 클래스들
        class KnowledgeOntology:
            def __init__(self, domain):
                self.domain = domain
            def add_knowledge_from_analysis(self, data): pass
            def infer_insights(self): return []
            def get_knowledge_summary(self): return {}
        
        class KnowledgeNode: pass

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class InsightItem:
    """인사이트 아이템 구조"""
    id: str
    type: str
    title: str
    description: str
    confidence: float
    priority: str  # "high", "medium", "low"
    category: str
    evidence: List[str]
    actionable_suggestions: List[str]
    related_data: Dict[str, Any]
    generated_at: str
    validity_period: Optional[str] = None

class InsightGenerator:
    """자동 인사이트 생성 엔진"""
    
    def __init__(self, domain: str = "conference_analysis"):
        self.domain = domain
        self.ontology = KnowledgeOntology(domain)
        self.insight_patterns = self._initialize_insight_patterns()
        self.quality_thresholds = self._initialize_quality_thresholds()
        
        # 생성된 인사이트 저장
        self.insights_history: List[InsightItem] = []
        self.insights_cache: Dict[str, InsightItem] = {}
        
        # 성능 메트릭
        self.performance_stats = {
            'total_insights_generated': 0,
            'high_confidence_insights': 0,
            'actionable_insights': 0,
            'average_generation_time': 0.0,
            'pattern_detection_accuracy': 0.0
        }
        
        logger.info(f"💡 인사이트 생성 엔진 초기화 - 도메인: {domain}")
    
    def _initialize_insight_patterns(self) -> Dict[str, Dict]:
        """인사이트 패턴 정의 초기화"""
        patterns = {
            # 트렌드 패턴
            "trend_detection": {
                "pattern": "시간에 따른 변화 패턴",
                "indicators": ["증가", "감소", "상승", "하락", "성장", "decline", "growth"],
                "confidence_weight": 0.8,
                "min_evidence_count": 3,
                "category": "trend_analysis"
            },
            
            # 상관관계 패턴
            "correlation_insight": {
                "pattern": "두 개 이상 요소 간 상관관계",
                "indicators": ["영향", "관련", "연관", "correlation", "relationship"],
                "confidence_weight": 0.75,
                "min_evidence_count": 2,
                "category": "relationship_analysis"
            },
            
            # 이상 패턴 탐지
            "anomaly_detection": {
                "pattern": "예상과 다른 특이한 패턴",
                "indicators": ["예외", "특이", "이상", "unusual", "unexpected", "anomaly"],
                "confidence_weight": 0.9,
                "min_evidence_count": 1,
                "category": "anomaly_analysis"
            },
            
            # 기회 식별
            "opportunity_identification": {
                "pattern": "개선이나 활용 가능한 기회",
                "indicators": ["기회", "가능성", "잠재", "opportunity", "potential", "chance"],
                "confidence_weight": 0.7,
                "min_evidence_count": 2,
                "category": "opportunity_analysis"
            },
            
            # 리스크 탐지
            "risk_assessment": {
                "pattern": "위험 요소나 주의사항",
                "indicators": ["위험", "문제", "우려", "risk", "concern", "issue"],
                "confidence_weight": 0.85,
                "min_evidence_count": 1,
                "category": "risk_analysis"
            },
            
            # 성과 분석
            "performance_analysis": {
                "pattern": "성과나 결과에 대한 분석",
                "indicators": ["성과", "결과", "효과", "performance", "result", "effectiveness"],
                "confidence_weight": 0.8,
                "min_evidence_count": 2,
                "category": "performance_analysis"
            }
        }
        
        return patterns
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """품질 임계값 설정"""
        return {
            'min_confidence': 0.6,
            'high_confidence': 0.8,
            'min_evidence_strength': 0.5,
            'max_similarity_threshold': 0.85,  # 중복 방지
            'temporal_relevance_decay': 0.95  # 시간 경과에 따른 관련성 감소
        }
    
    async def generate_comprehensive_insights(
        self, 
        multimodal_results: List[MultimodalResult],
        cross_modal_insights: Dict[str, Any],
        external_context: Optional[Dict[str, Any]] = None
    ) -> List[InsightItem]:
        """종합적인 인사이트 생성"""
        
        logger.info("🚀 종합 인사이트 생성 시작...")
        start_time = datetime.now()
        
        # 1. 온톨로지 지식 베이스 업데이트
        analysis_data = {
            "multimodal_results": [asdict(result) for result in multimodal_results],
            "cross_modal_insights": cross_modal_insights
        }
        
        self.ontology.add_knowledge_from_analysis(analysis_data)
        
        # 2. 다각도 인사이트 생성
        insights = []
        
        # 멀티모달 분석 기반 인사이트
        modal_insights = await self._generate_multimodal_insights(multimodal_results)
        insights.extend(modal_insights)
        
        # 크로스 모달 분석 기반 인사이트  
        cross_insights = await self._generate_cross_modal_insights(cross_modal_insights)
        insights.extend(cross_insights)
        
        # 온톨로지 기반 추론 인사이트
        ontology_insights = await self._generate_ontology_insights()
        insights.extend(ontology_insights)
        
        # 외부 컨텍스트 통합 인사이트
        if external_context:
            context_insights = await self._generate_context_insights(external_context)
            insights.extend(context_insights)
        
        # 3. 인사이트 품질 필터링 및 랭킹
        filtered_insights = self._filter_and_rank_insights(insights)
        
        # 4. 액션 가능한 제안 생성
        actionable_insights = await self._add_actionable_suggestions(filtered_insights)
        
        # 5. 중복 제거 및 최종 정제
        final_insights = self._deduplicate_insights(actionable_insights)
        
        # 성능 통계 업데이트
        generation_time = (datetime.now() - start_time).total_seconds()
        self._update_performance_stats(len(final_insights), generation_time)
        
        # 히스토리에 저장
        self.insights_history.extend(final_insights)
        
        logger.info(f"✅ 인사이트 생성 완료: {len(final_insights)}개 (처리시간: {generation_time:.2f}초)")
        return final_insights
    
    async def _generate_multimodal_insights(self, results: List[MultimodalResult]) -> List[InsightItem]:
        """멀티모달 결과 기반 인사이트 생성"""
        insights = []
        
        # 모달별 품질 분석
        modality_quality = self._analyze_modality_quality(results)
        if modality_quality['insight_worthy']:
            insights.append(
                self._create_insight(
                    type="quality_analysis",
                    title="멀티모달 데이터 품질 분석",
                    description=modality_quality['description'],
                    confidence=modality_quality['confidence'],
                    evidence=modality_quality['evidence'],
                    category="data_quality"
                )
            )
        
        # 컨텐츠 일관성 분석
        consistency_analysis = self._analyze_content_consistency(results)
        if consistency_analysis['has_insights']:
            insights.append(
                self._create_insight(
                    type="consistency_analysis",
                    title="컨텐츠 일관성 평가",
                    description=consistency_analysis['description'],
                    confidence=consistency_analysis['confidence'],
                    evidence=consistency_analysis['evidence'],
                    category="content_analysis"
                )
            )
        
        # 처리 효율성 인사이트
        efficiency_insights = self._analyze_processing_efficiency(results)
        insights.extend(efficiency_insights)
        
        return insights
    
    def _analyze_modality_quality(self, results: List[MultimodalResult]) -> Dict[str, Any]:
        """모달별 품질 분석"""
        modality_stats = defaultdict(list)
        
        for result in results:
            modality_stats[result.file_type].append(result.confidence)
        
        quality_analysis = {
            'insight_worthy': False,
            'description': "",
            'confidence': 0.0,
            'evidence': []
        }
        
        if len(modality_stats) > 1:
            # 모달별 평균 신뢰도 계산
            modal_averages = {
                modal: statistics.mean(confidences) 
                for modal, confidences in modality_stats.items()
            }
            
            best_modal = max(modal_averages, key=modal_averages.get)
            worst_modal = min(modal_averages, key=modal_averages.get)
            
            quality_gap = modal_averages[best_modal] - modal_averages[worst_modal]
            
            if quality_gap > 0.2:  # 20% 이상 차이
                quality_analysis.update({
                    'insight_worthy': True,
                    'description': f"{best_modal} 데이터 품질이 {worst_modal}보다 {quality_gap:.1%} 높음. 분석 신뢰도 개선 방안 필요.",
                    'confidence': 0.8,
                    'evidence': [
                        f"{best_modal} 평균 신뢰도: {modal_averages[best_modal]:.3f}",
                        f"{worst_modal} 평균 신뢰도: {modal_averages[worst_modal]:.3f}"
                    ]
                })
        
        return quality_analysis
    
    def _analyze_content_consistency(self, results: List[MultimodalResult]) -> Dict[str, Any]:
        """컨텐츠 일관성 분석"""
        analysis = {
            'has_insights': False,
            'description': "",
            'confidence': 0.0,
            'evidence': []
        }
        
        if len(results) < 2:
            return analysis
        
        # 컨텐츠 길이 분산 분석
        content_lengths = [len(result.content) for result in results if result.content]
        
        if content_lengths and len(content_lengths) > 1:
            avg_length = statistics.mean(content_lengths)
            std_length = statistics.stdev(content_lengths)
            cv = std_length / avg_length if avg_length > 0 else 0  # 변동계수
            
            if cv > 1.0:  # 높은 변동성
                analysis.update({
                    'has_insights': True,
                    'description': f"컨텐츠 길이 편차가 큼 (변동계수: {cv:.2f}). 일부 파일에서 정보 추출이 부족할 가능성.",
                    'confidence': 0.7,
                    'evidence': [
                        f"평균 컨텐츠 길이: {avg_length:.0f}자",
                        f"표준편차: {std_length:.0f}자",
                        f"최대/최소 비율: {max(content_lengths)/min(content_lengths):.1f}"
                    ]
                })
        
        return analysis
    
    def _analyze_processing_efficiency(self, results: List[MultimodalResult]) -> List[InsightItem]:
        """처리 효율성 분석"""
        insights = []
        
        if not results:
            return insights
        
        # 처리 시간 분석
        processing_times = [result.processing_time for result in results]
        avg_time = statistics.mean(processing_times)
        
        # 비효율적인 파일 식별
        slow_files = [
            result for result in results 
            if result.processing_time > avg_time * 2  # 평균의 2배 이상
        ]
        
        if slow_files:
            insights.append(
                self._create_insight(
                    type="efficiency_analysis",
                    title="처리 효율성 개선 기회",
                    description=f"{len(slow_files)}개 파일의 처리 시간이 평균보다 {len(slow_files)/len(results):.1%} 길음. 최적화 필요.",
                    confidence=0.75,
                    evidence=[
                        f"평균 처리 시간: {avg_time:.2f}초",
                        f"최대 처리 시간: {max(processing_times):.2f}초",
                        f"비효율 파일 수: {len(slow_files)}개"
                    ],
                    category="performance_optimization"
                )
            )
        
        return insights
    
    async def _generate_cross_modal_insights(self, cross_insights: Dict[str, Any]) -> List[InsightItem]:
        """크로스 모달 인사이트 생성"""
        insights = []
        
        # 상관관계 분석
        correlation_pairs = cross_insights.get("high_correlation_pairs", [])
        if correlation_pairs:
            for pair in correlation_pairs[:3]:  # 상위 3개
                insights.append(
                    self._create_insight(
                        type="cross_modal_correlation",
                        title="강한 크로스 모달 상관관계 발견",
                        description=f"파일 간 {pair['correlation_score']:.1%} 상관관계 탐지. 연관된 내용일 가능성 높음.",
                        confidence=pair['correlation_score'],
                        evidence=[f"상관계수: {pair['correlation_score']:.3f}"],
                        category="relationship_analysis"
                    )
                )
        
        # 주요 테마 분석
        themes = cross_insights.get("dominant_themes", [])
        if len(themes) >= 3:
            insights.append(
                self._create_insight(
                    type="theme_analysis",
                    title="주요 테마 식별",
                    description=f"분석 데이터에서 {len(themes)}개 핵심 테마 발견: {', '.join(themes[:5])}",
                    confidence=0.8,
                    evidence=[f"테마 수: {len(themes)}", f"상위 테마: {themes[:3]}"],
                    category="content_analysis"
                )
            )
        
        return insights
    
    async def _generate_ontology_insights(self) -> List[InsightItem]:
        """온톨로지 기반 인사이트 생성"""
        insights = []
        
        # 온톨로지에서 추론된 인사이트 가져오기
        ontology_insights = self.ontology.infer_insights()
        
        for ont_insight in ontology_insights[:5]:  # 상위 5개
            insights.append(
                self._create_insight(
                    type="ontology_inference",
                    title=f"지식 기반 추론: {ont_insight.get('type', '분석')}",
                    description=ont_insight.get('description', ''),
                    confidence=ont_insight.get('confidence', 0.7),
                    evidence=ont_insight.get('evidence', []),
                    category=ont_insight.get('category', 'knowledge_inference')
                )
            )
        
        return insights
    
    async def _generate_context_insights(self, external_context: Dict[str, Any]) -> List[InsightItem]:
        """외부 컨텍스트 기반 인사이트 생성"""
        insights = []
        
        # 시간 컨텍스트 분석
        if 'timestamp' in external_context:
            time_insights = self._analyze_temporal_context(external_context['timestamp'])
            insights.extend(time_insights)
        
        # 업계 컨텍스트 분석
        if 'industry_data' in external_context:
            industry_insights = self._analyze_industry_context(external_context['industry_data'])
            insights.extend(industry_insights)
        
        return insights
    
    def _analyze_temporal_context(self, timestamp: str) -> List[InsightItem]:
        """시간적 컨텍스트 분석"""
        insights = []
        
        try:
            analysis_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            
            # 분석 시점의 특성 파악
            hour = analysis_time.hour
            weekday = analysis_time.weekday()
            
            time_context = ""
            if hour < 9 or hour > 18:
                time_context = "업무 시간 외"
            elif weekday >= 5:
                time_context = "주말"
            else:
                time_context = "업무 시간 중"
            
            if time_context in ["업무 시간 외", "주말"]:
                insights.append(
                    self._create_insight(
                        type="temporal_context",
                        title="분석 시점 특성",
                        description=f"{time_context} 분석으로, 공식적 업무 관련 내용일 가능성 낮음",
                        confidence=0.6,
                        evidence=[f"분석 시점: {analysis_time.strftime('%Y-%m-%d %H:%M')}"],
                        category="context_analysis"
                    )
                )
                
        except Exception as e:
            logger.warning(f"시간 컨텍스트 분석 실패: {e}")
        
        return insights
    
    def _analyze_industry_context(self, industry_data: Dict[str, Any]) -> List[InsightItem]:
        """업계 컨텍스트 분석"""
        insights = []
        
        # 업계 트렌드와 분석 내용 비교
        if 'trends' in industry_data:
            trends = industry_data['trends']
            
            insights.append(
                self._create_insight(
                    type="industry_alignment",
                    title="업계 트렌드 정렬도",
                    description=f"현재 업계 {len(trends)}개 주요 트렌드와의 연관성 분석 필요",
                    confidence=0.7,
                    evidence=[f"업계 트렌드: {', '.join(trends[:3])}"],
                    category="market_analysis"
                )
            )
        
        return insights
    
    def _filter_and_rank_insights(self, insights: List[InsightItem]) -> List[InsightItem]:
        """인사이트 품질 필터링 및 랭킹"""
        # 최소 신뢰도 필터링
        filtered = [
            insight for insight in insights 
            if insight.confidence >= self.quality_thresholds['min_confidence']
        ]
        
        # 우선순위 계산
        for insight in filtered:
            insight.priority = self._calculate_priority(insight)
        
        # 신뢰도와 우선순위 기준 정렬
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        
        filtered.sort(
            key=lambda x: (priority_weights.get(x.priority, 0), x.confidence), 
            reverse=True
        )
        
        return filtered[:20]  # 상위 20개만 유지
    
    def _calculate_priority(self, insight: InsightItem) -> str:
        """인사이트 우선순위 계산"""
        score = insight.confidence
        
        # 카테고리별 가중치
        category_weights = {
            'risk_analysis': 0.3,
            'opportunity_analysis': 0.25,
            'anomaly_analysis': 0.2,
            'performance_optimization': 0.15,
            'trend_analysis': 0.1
        }
        
        score += category_weights.get(insight.category, 0)
        
        # 증거 수에 따른 보정
        score += min(0.1, len(insight.evidence) * 0.02)
        
        if score >= 0.9:
            return "high"
        elif score >= 0.7:
            return "medium"
        else:
            return "low"
    
    async def _add_actionable_suggestions(self, insights: List[InsightItem]) -> List[InsightItem]:
        """액션 가능한 제안 추가"""
        for insight in insights:
            suggestions = self._generate_action_suggestions(insight)
            insight.actionable_suggestions = suggestions
        
        return insights
    
    def _generate_action_suggestions(self, insight: InsightItem) -> List[str]:
        """인사이트별 액션 제안 생성"""
        suggestions = []
        
        # 인사이트 타입별 제안
        if insight.type == "quality_analysis":
            suggestions.extend([
                "저품질 데이터 소스 개선 방안 검토",
                "데이터 전처리 프로세스 최적화",
                "품질 모니터링 시스템 구축"
            ])
        
        elif insight.type == "efficiency_analysis":
            suggestions.extend([
                "비효율적인 파일 처리 프로세스 최적화",
                "하드웨어 리소스 업그레이드 검토",
                "배치 처리 크기 조정"
            ])
        
        elif insight.type == "cross_modal_correlation":
            suggestions.extend([
                "상관관계 있는 데이터 통합 분석",
                "연관 패턴 활용한 예측 모델 구축",
                "데이터 수집 전략 개선"
            ])
        
        elif insight.type == "theme_analysis":
            suggestions.extend([
                "주요 테마 기반 심화 분석 수행",
                "테마별 전문가 리뷰 요청",
                "관련 외부 데이터 수집"
            ])
        
        # 카테고리별 공통 제안
        if insight.category == "risk_analysis":
            suggestions.append("리스크 완화 계획 수립 및 실행")
        
        elif insight.category == "opportunity_analysis":
            suggestions.append("기회 활용 전략 개발 및 우선순위 설정")
        
        return suggestions[:3]  # 최대 3개 제안
    
    def _deduplicate_insights(self, insights: List[InsightItem]) -> List[InsightItem]:
        """중복 인사이트 제거"""
        unique_insights = []
        seen_signatures = set()
        
        for insight in insights:
            # 인사이트 서명 생성 (제목 + 카테고리 기반)
            signature = f"{insight.title}_{insight.category}"
            
            if signature not in seen_signatures:
                unique_insights.append(insight)
                seen_signatures.add(signature)
        
        return unique_insights
    
    def _create_insight(
        self,
        type: str,
        title: str, 
        description: str,
        confidence: float,
        evidence: List[str],
        category: str,
        validity_hours: int = 24
    ) -> InsightItem:
        """인사이트 아이템 생성 헬퍼"""
        
        insight_id = f"{type}_{datetime.now().timestamp()}"
        validity_period = (datetime.now() + timedelta(hours=validity_hours)).isoformat()
        
        return InsightItem(
            id=insight_id,
            type=type,
            title=title,
            description=description,
            confidence=min(1.0, max(0.0, confidence)),  # 0-1 범위로 클램핑
            priority="medium",  # 기본값, 나중에 계산됨
            category=category,
            evidence=evidence,
            actionable_suggestions=[],  # 나중에 추가됨
            related_data={},
            generated_at=datetime.now().isoformat(),
            validity_period=validity_period
        )
    
    def _update_performance_stats(self, insight_count: int, generation_time: float) -> None:
        """성능 통계 업데이트"""
        self.performance_stats['total_insights_generated'] += insight_count
        self.performance_stats['high_confidence_insights'] += sum(
            1 for insight in self.insights_history[-insight_count:] 
            if insight.confidence >= self.quality_thresholds['high_confidence']
        )
        self.performance_stats['actionable_insights'] += sum(
            1 for insight in self.insights_history[-insight_count:] 
            if insight.actionable_suggestions
        )
        
        # 평균 생성 시간 업데이트 (이동 평균)
        current_avg = self.performance_stats['average_generation_time']
        total_runs = self.performance_stats['total_insights_generated'] // insight_count
        
        if total_runs == 1:
            self.performance_stats['average_generation_time'] = generation_time
        else:
            # 지수 이동 평균
            alpha = 0.1
            self.performance_stats['average_generation_time'] = (
                alpha * generation_time + (1 - alpha) * current_avg
            )
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """인사이트 생성 요약"""
        if not self.insights_history:
            return {"message": "생성된 인사이트가 없습니다."}
        
        # 카테고리별 분포
        category_dist = Counter(insight.category for insight in self.insights_history)
        
        # 우선순위별 분포
        priority_dist = Counter(insight.priority for insight in self.insights_history)
        
        # 최근 생성된 인사이트들
        recent_insights = [
            insight for insight in self.insights_history 
            if (datetime.now() - datetime.fromisoformat(insight.generated_at)).days <= 1
        ]
        
        return {
            'total_insights': len(self.insights_history),
            'recent_insights': len(recent_insights),
            'category_distribution': dict(category_dist),
            'priority_distribution': dict(priority_dist),
            'average_confidence': statistics.mean(
                insight.confidence for insight in self.insights_history
            ),
            'performance_stats': self.performance_stats,
            'ontology_summary': self.ontology.get_knowledge_summary()
        }

# 사용 예제
async def main():
    """사용 예제"""
    generator = InsightGenerator("conference_analysis")
    
    # 샘플 멀티모달 결과
    from multimodal_pipeline import MultimodalResult
    sample_results = [
        MultimodalResult(
            file_path="sample1.wav",
            file_type="audio",
            content="주얼리 시장의 디지털 전환이 가속화되고 있습니다",
            confidence=0.85,
            processing_time=2.3,
            metadata={"duration": 45.2}
        ),
        MultimodalResult(
            file_path="sample2.png", 
            file_type="image",
            content="Digital Transformation Jewelry Market Growth 2024",
            confidence=0.92,
            processing_time=1.8,
            metadata={"image_size": (1920, 1080)}
        )
    ]
    
    # 샘플 크로스 모달 인사이트
    cross_insights = {
        "high_correlation_pairs": [
            {
                "primary_file": "sample1.wav",
                "correlation_score": 0.87,
                "related_files": [{"file": "sample2.png", "similarity": 0.87}]
            }
        ],
        "dominant_themes": ["디지털전환", "주얼리", "시장성장", "기술혁신"]
    }
    
    # 종합 인사이트 생성
    insights = await generator.generate_comprehensive_insights(
        sample_results, 
        cross_insights
    )
    
    print("💡 생성된 인사이트:")
    for i, insight in enumerate(insights[:5], 1):
        print(f"\n{i}. [{insight.priority.upper()}] {insight.title}")
        print(f"   설명: {insight.description}")
        print(f"   신뢰도: {insight.confidence:.2f}")
        print(f"   제안: {'; '.join(insight.actionable_suggestions[:2])}")
    
    print(f"\n📊 요약: {generator.get_insights_summary()}")

if __name__ == "__main__":
    asyncio.run(main())