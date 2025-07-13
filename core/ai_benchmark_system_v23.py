"""
📊 솔로몬드 AI 성능 벤치마크 시스템 v2.3
99.2% 정확도 달성을 위한 종합적 성능 측정 및 A/B 테스트 자동화 시스템

📅 개발일: 2025.07.13
🎯 목표: 실시간 성능 측정으로 99.2% 정확도 달성 추적
🔥 주요 기능:
- A/B 테스트 자동화 시스템
- 다차원 성능 메트릭 분석
- 모델 성능 비교 및 랭킹
- 정확도 달성도 실시간 추적
- 자동 최적화 권장사항 생성
- 성능 트렌드 분석 및 예측

연동 시스템:
- hybrid_llm_manager_v23.py
- ai_quality_validator_v23.py  
- jewelry_specialized_prompts_v23.py
"""

import asyncio
import logging
import time
import json
import statistics
import random
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import concurrent.futures
from pathlib import Path
import csv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AIBenchmark_v23')

class BenchmarkType(Enum):
    """벤치마크 유형"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    COST_EFFICIENCY = "cost_efficiency"
    JEWELRY_EXPERTISE = "jewelry_expertise"
    COMPREHENSIVE = "comprehensive"
    A_B_TEST = "a_b_test"
    STRESS_TEST = "stress_test"
    REAL_TIME_MONITORING = "real_time_monitoring"

class MetricType(Enum):
    """성능 지표 유형"""
    ACCURACY_SCORE = "accuracy_score"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    COST_PER_REQUEST = "cost_per_request"
    JEWELRY_RELEVANCE = "jewelry_relevance"
    USER_SATISFACTION = "user_satisfaction"
    QUALITY_CONSISTENCY = "quality_consistency"
    ERROR_RATE = "error_rate"

class TestScenario(Enum):
    """테스트 시나리오"""
    DIAMOND_4C_ANALYSIS = "diamond_4c_analysis"
    RUBY_GRADING = "ruby_grading"
    EMERALD_EVALUATION = "emerald_evaluation"
    SAPPHIRE_ASSESSMENT = "sapphire_assessment"
    MARKET_VALUATION = "market_valuation"
    INVESTMENT_ANALYSIS = "investment_analysis"
    INSURANCE_APPRAISAL = "insurance_appraisal"
    COMPLEX_MULTIMODAL = "complex_multimodal"

@dataclass
class BenchmarkMetric:
    """벤치마크 지표"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestCase:
    """테스트 케이스"""
    id: str
    scenario: TestScenario
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    difficulty_level: float = 0.5  # 0.0 (쉬움) ~ 1.0 (어려움)
    priority: str = "normal"  # low, normal, high, critical
    tags: List[str] = field(default_factory=list)

@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    test_id: str
    model_name: str
    scenario: TestScenario
    metrics: Dict[MetricType, BenchmarkMetric]
    overall_score: float
    
    # 상세 정보
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    # 품질 분석
    quality_analysis: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ABTestResult:
    """A/B 테스트 결과"""
    test_name: str
    model_a: str
    model_b: str
    
    # 성능 비교
    model_a_metrics: Dict[MetricType, float]
    model_b_metrics: Dict[MetricType, float]
    
    # 통계적 유의성
    statistical_significance: bool
    confidence_level: float
    p_value: float
    
    # 결론
    winner: Optional[str] = None
    improvement_percentage: float = 0.0
    recommendation: str = ""
    
    test_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    sample_size: int = 100
    timestamp: datetime = field(default_factory=datetime.now)

class TestCaseGenerator:
    """테스트 케이스 생성기"""
    
    def __init__(self):
        self.scenario_templates = {
            TestScenario.DIAMOND_4C_ANALYSIS: {
                "templates": [
                    "2.50캐럿 라운드 브릴리언트 다이아몬드의 4C 분석",
                    "1.75캐럿 프린세스 컷 다이아몬드 감정",
                    "3.25캐럿 에메랄드 컷 다이아몬드 등급 평가",
                    "5.10캐럿 쿠션 컷 다이아몬드 투자 가치 분석"
                ],
                "variables": {
                    "carat": [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 5.00],
                    "cut_type": ["라운드", "프린세스", "에메랄드", "쿠션", "오벌", "마키즈"],
                    "color_grade": ["D", "E", "F", "G", "H", "I", "J"],
                    "clarity_grade": ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1"]
                }
            },
            
            TestScenario.RUBY_GRADING: {
                "templates": [
                    "버마산 루비의 원산지 감정 및 품질 평가",
                    "태국 루비와 미얀마 루비의 비교 분석",
                    "무가열 루비의 시장 가치 평가",
                    "Pigeon Blood 루비의 진위성 검증"
                ],
                "variables": {
                    "origin": ["미얀마", "태국", "마다가스카르", "모잠비크"],
                    "treatment": ["무가열", "가열", "불명"],
                    "color_quality": ["Pigeon Blood", "Red", "Purplish Red", "Pink Red"],
                    "size": ["1-3캐럿", "3-5캐럿", "5-10캐럿", "10캐럿 이상"]
                }
            },
            
            TestScenario.MARKET_VALUATION: {
                "templates": [
                    "국제 다이아몬드 시장 동향 분석",
                    "유색보석 투자 시장 전망",
                    "주얼리 경매 시장 가격 분석",
                    "코로나 이후 보석 시장 변화"
                ],
                "variables": {
                    "market_type": ["다이아몬드", "루비", "사파이어", "에메랄드"],
                    "time_period": ["최근 1년", "최근 3년", "최근 5년", "장기 전망"],
                    "region": ["글로벌", "아시아", "북미", "유럽"],
                    "segment": ["투자등급", "상업적품질", "수집가급"]
                }
            }
        }
    
    def generate_test_cases(self, count: int = 10, 
                          scenarios: Optional[List[TestScenario]] = None) -> List[TestCase]:
        """테스트 케이스 생성"""
        
        if not scenarios:
            scenarios = list(TestScenario)
        
        test_cases = []
        
        for i in range(count):
            scenario = random.choice(scenarios)
            test_case = self._generate_single_test_case(f"TEST_{i+1:03d}", scenario)
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_single_test_case(self, test_id: str, scenario: TestScenario) -> TestCase:
        """단일 테스트 케이스 생성"""
        
        template_data = self.scenario_templates.get(scenario, {})
        templates = template_data.get("templates", ["기본 분석 요청"])
        variables = template_data.get("variables", {})
        
        # 랜덤 템플릿 선택
        base_template = random.choice(templates)
        
        # 변수 치환
        input_text = base_template
        for var_name, var_values in variables.items():
            if f"{{{var_name}}}" in input_text:
                input_text = input_text.replace(f"{{{var_name}}}", random.choice(var_values))
        
        # 난이도 설정
        difficulty_factors = {
            TestScenario.DIAMOND_4C_ANALYSIS: 0.6,
            TestScenario.RUBY_GRADING: 0.8,
            TestScenario.EMERALD_EVALUATION: 0.7,
            TestScenario.MARKET_VALUATION: 0.9,
            TestScenario.COMPLEX_MULTIMODAL: 1.0
        }
        
        difficulty = difficulty_factors.get(scenario, 0.5)
        difficulty += random.uniform(-0.1, 0.1)  # 약간의 랜덤성
        difficulty = max(0.0, min(1.0, difficulty))
        
        input_data = {
            "text": input_text,
            "analysis_type": scenario.value,
            "gemstone_type": self._infer_gemstone_type(input_text),
            "priority": random.choice(["normal", "high", "normal", "normal"]),  # normal 우선
            "context": {
                "test_case": True,
                "difficulty": difficulty,
                "generated_at": datetime.now().isoformat()
            }
        }
        
        # 태그 설정
        tags = [scenario.value]
        if difficulty > 0.8:
            tags.append("high_difficulty")
        if "투자" in input_text or "가치" in input_text:
            tags.append("investment_related")
        if "감정" in input_text or "등급" in input_text:
            tags.append("grading_related")
        
        return TestCase(
            id=test_id,
            scenario=scenario,
            input_data=input_data,
            difficulty_level=difficulty,
            priority="high" if difficulty > 0.8 else "normal",
            tags=tags
        )
    
    def _infer_gemstone_type(self, text: str) -> str:
        """텍스트에서 보석 타입 추론"""
        
        text_lower = text.lower()
        
        if "다이아몬드" in text_lower or "diamond" in text_lower:
            return "diamond"
        elif "루비" in text_lower or "ruby" in text_lower:
            return "ruby"
        elif "사파이어" in text_lower or "sapphire" in text_lower:
            return "sapphire"
        elif "에메랄드" in text_lower or "emerald" in text_lower:
            return "emerald"
        else:
            return "general"

class PerformanceAnalyzer:
    """성능 분석기"""
    
    def __init__(self):
        self.target_accuracy = 0.992  # 99.2%
        self.performance_thresholds = {
            MetricType.ACCURACY_SCORE: 0.992,
            MetricType.RESPONSE_TIME: 15.0,  # 15초 이내
            MetricType.COST_PER_REQUEST: 0.05,  # 5센트 이내
            MetricType.JEWELRY_RELEVANCE: 0.90,
            MetricType.QUALITY_CONSISTENCY: 0.95,
            MetricType.ERROR_RATE: 0.02  # 2% 이하
        }
    
    def analyze_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """성능 분석"""
        
        if not results:
            return {"error": "분석할 결과가 없습니다"}
        
        analysis = {
            "summary": {},
            "detailed_metrics": {},
            "target_achievement": {},
            "trends": {},
            "recommendations": []
        }
        
        # 1. 기본 통계
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_execution_time = statistics.mean(r.execution_time for r in results)
        
        analysis["summary"] = {
            "total_tests": len(results),
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "test_period": f"{results[0].timestamp} ~ {results[-1].timestamp}"
        }
        
        # 2. 메트릭별 상세 분석
        metric_aggregates = defaultdict(list)
        
        for result in results:
            for metric_type, metric in result.metrics.items():
                metric_aggregates[metric_type].append(metric.value)
        
        for metric_type, values in metric_aggregates.items():
            if values:
                analysis["detailed_metrics"][metric_type.value] = {
                    "average": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "samples": len(values)
                }
        
        # 3. 목표 달성도 분석
        for metric_type, threshold in self.performance_thresholds.items():
            if metric_type in metric_aggregates:
                values = metric_aggregates[metric_type]
                
                if metric_type in [MetricType.RESPONSE_TIME, MetricType.COST_PER_REQUEST, MetricType.ERROR_RATE]:
                    # 낮을수록 좋은 지표
                    achievement_rate = sum(1 for v in values if v <= threshold) / len(values)
                else:
                    # 높을수록 좋은 지표
                    achievement_rate = sum(1 for v in values if v >= threshold) / len(values)
                
                analysis["target_achievement"][metric_type.value] = {
                    "threshold": threshold,
                    "achievement_rate": achievement_rate,
                    "status": "달성" if achievement_rate >= 0.8 else "미달성"
                }
        
        # 4. 트렌드 분석 (시간순 정렬된 최근 결과들)
        if len(results) >= 5:
            recent_results = sorted(results, key=lambda r: r.timestamp)[-10:]
            
            for metric_type in metric_aggregates.keys():
                recent_values = []
                for result in recent_results:
                    if metric_type in result.metrics:
                        recent_values.append(result.metrics[metric_type].value)
                
                if len(recent_values) >= 3:
                    # 선형 회귀로 트렌드 계산
                    x = list(range(len(recent_values)))
                    slope = self._calculate_trend_slope(x, recent_values)
                    
                    trend_direction = "상승" if slope > 0.01 else ("하락" if slope < -0.01 else "안정")
                    
                    analysis["trends"][metric_type.value] = {
                        "direction": trend_direction,
                        "slope": slope,
                        "recent_average": statistics.mean(recent_values)
                    }
        
        # 5. 권장사항 생성
        recommendations = self._generate_recommendations(analysis)
        analysis["recommendations"] = recommendations
        
        return analysis
    
    def _calculate_trend_slope(self, x: List[float], y: List[float]) -> float:
        """트렌드 기울기 계산 (단순 선형 회귀)"""
        
        n = len(x)
        if n < 2:
            return 0.0
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        
        recommendations = []
        
        # 성공률 기반 권장사항
        success_rate = analysis["summary"]["success_rate"]
        if success_rate < 0.95:
            recommendations.append(f"성공률 개선 필요 (현재: {success_rate:.1%}, 목표: 95% 이상)")
        
        # 목표 달성도 기반 권장사항
        target_achievement = analysis.get("target_achievement", {})
        
        for metric_name, data in target_achievement.items():
            if data["status"] == "미달성":
                recommendations.append(
                    f"{metric_name} 개선 필요 (달성률: {data['achievement_rate']:.1%})"
                )
        
        # 트렌드 기반 권장사항
        trends = analysis.get("trends", {})
        
        declining_metrics = [
            metric for metric, trend in trends.items()
            if trend["direction"] == "하락"
        ]
        
        if declining_metrics:
            recommendations.append(
                f"성능 하락 추세 모니터링 필요: {', '.join(declining_metrics)}"
            )
        
        # 99.2% 목표 관련 권장사항
        accuracy_data = analysis["detailed_metrics"].get("accuracy_score")
        if accuracy_data and accuracy_data["average"] < self.target_accuracy:
            gap = self.target_accuracy - accuracy_data["average"]
            recommendations.append(
                f"99.2% 정확도 목표까지 {gap:.1%} 추가 개선 필요"
            )
        
        # 응답 시간 관련 권장사항
        response_time_data = analysis["detailed_metrics"].get("response_time")
        if response_time_data and response_time_data["average"] > 15.0:
            recommendations.append("응답 시간 최적화 필요 (목표: 15초 이내)")
        
        # 일반적인 권장사항
        if not recommendations:
            recommendations.append("현재 성능이 목표 수준을 만족하고 있습니다.")
            recommendations.append("지속적인 모니터링을 통한 성능 유지를 권장합니다.")
        
        return recommendations

class ABTestManager:
    """A/B 테스트 관리자"""
    
    def __init__(self):
        self.active_tests = {}
        self.completed_tests = []
        self.statistical_confidence = 0.95
    
    async def create_ab_test(self, test_name: str, model_a: str, model_b: str,
                           test_cases: List[TestCase], 
                           duration_hours: int = 24) -> str:
        """A/B 테스트 생성"""
        
        test_id = f"AB_{test_name}_{int(time.time())}"
        
        ab_test = {
            "test_id": test_id,
            "name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "test_cases": test_cases,
            "duration": timedelta(hours=duration_hours),
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=duration_hours),
            "results_a": [],
            "results_b": [],
            "status": "running"
        }
        
        self.active_tests[test_id] = ab_test
        
        logger.info(f"A/B 테스트 시작: {test_name} ({model_a} vs {model_b})")
        
        return test_id
    
    async def execute_ab_test(self, test_id: str, 
                            execution_func: Callable) -> ABTestResult:
        """A/B 테스트 실행"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"존재하지 않는 테스트: {test_id}")
        
        test_config = self.active_tests[test_id]
        
        logger.info(f"A/B 테스트 실행 중: {test_config['name']}")
        
        # 테스트 케이스를 두 그룹으로 분할
        test_cases = test_config["test_cases"]
        random.shuffle(test_cases)
        
        mid_point = len(test_cases) // 2
        cases_a = test_cases[:mid_point]
        cases_b = test_cases[mid_point:]
        
        # 동시 실행
        results_a = []
        results_b = []
        
        # 모델 A 테스트
        for case in cases_a:
            try:
                result = await execution_func(test_config["model_a"], case)
                results_a.append(result)
            except Exception as e:
                logger.error(f"모델 A 테스트 실패: {e}")
        
        # 모델 B 테스트
        for case in cases_b:
            try:
                result = await execution_func(test_config["model_b"], case)
                results_b.append(result)
            except Exception as e:
                logger.error(f"모델 B 테스트 실패: {e}")
        
        # 결과 분석
        ab_result = self._analyze_ab_test_results(
            test_config, results_a, results_b
        )
        
        # 테스트 완료 처리
        test_config["status"] = "completed"
        test_config["results_a"] = results_a
        test_config["results_b"] = results_b
        
        self.completed_tests.append(ab_result)
        del self.active_tests[test_id]
        
        logger.info(f"A/B 테스트 완료: {test_config['name']}")
        
        return ab_result
    
    def _analyze_ab_test_results(self, test_config: Dict[str, Any],
                                results_a: List[BenchmarkResult],
                                results_b: List[BenchmarkResult]) -> ABTestResult:
        """A/B 테스트 결과 분석"""
        
        # 메트릭별 평균 계산
        metrics_a = self._calculate_average_metrics(results_a)
        metrics_b = self._calculate_average_metrics(results_b)
        
        # 통계적 유의성 검정 (단순화된 버전)
        significance_results = self._test_statistical_significance(
            results_a, results_b
        )
        
        # 승자 결정
        winner = None
        improvement = 0.0
        
        # 주요 지표 (정확도) 기준으로 승자 결정
        if MetricType.ACCURACY_SCORE in metrics_a and MetricType.ACCURACY_SCORE in metrics_b:
            accuracy_a = metrics_a[MetricType.ACCURACY_SCORE]
            accuracy_b = metrics_b[MetricType.ACCURACY_SCORE]
            
            if accuracy_a > accuracy_b:
                winner = test_config["model_a"]
                improvement = ((accuracy_a - accuracy_b) / accuracy_b) * 100
            else:
                winner = test_config["model_b"]
                improvement = ((accuracy_b - accuracy_a) / accuracy_a) * 100
        
        # 권장사항 생성
        recommendation = self._generate_ab_recommendation(
            winner, improvement, significance_results
        )
        
        return ABTestResult(
            test_name=test_config["name"],
            model_a=test_config["model_a"],
            model_b=test_config["model_b"],
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            statistical_significance=significance_results["significant"],
            confidence_level=significance_results["confidence"],
            p_value=significance_results["p_value"],
            winner=winner,
            improvement_percentage=improvement,
            recommendation=recommendation,
            test_duration=test_config["duration"],
            sample_size=len(results_a) + len(results_b)
        )
    
    def _calculate_average_metrics(self, results: List[BenchmarkResult]) -> Dict[MetricType, float]:
        """평균 메트릭 계산"""
        
        metric_sums = defaultdict(list)
        
        for result in results:
            for metric_type, metric in result.metrics.items():
                metric_sums[metric_type].append(metric.value)
        
        averages = {}
        for metric_type, values in metric_sums.items():
            if values:
                averages[metric_type] = statistics.mean(values)
        
        return averages
    
    def _test_statistical_significance(self, results_a: List[BenchmarkResult],
                                     results_b: List[BenchmarkResult]) -> Dict[str, Any]:
        """통계적 유의성 검정 (단순화된 t-test)"""
        
        # 정확도 점수를 기준으로 검정
        scores_a = []
        scores_b = []
        
        for result in results_a:
            if MetricType.ACCURACY_SCORE in result.metrics:
                scores_a.append(result.metrics[MetricType.ACCURACY_SCORE].value)
        
        for result in results_b:
            if MetricType.ACCURACY_SCORE in result.metrics:
                scores_b.append(result.metrics[MetricType.ACCURACY_SCORE].value)
        
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {
                "significant": False,
                "confidence": 0.0,
                "p_value": 1.0,
                "reason": "샘플 크기 부족"
            }
        
        # 단순화된 t-test
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0
        
        # 효과 크기 계산
        effect_size = abs(mean_a - mean_b) / max(std_a, std_b, 0.001)
        
        # 단순화된 유의성 판정
        significant = effect_size > 0.5 and abs(mean_a - mean_b) > 0.02
        confidence = min(0.95, effect_size / 2.0)
        p_value = max(0.01, 1.0 - confidence)
        
        return {
            "significant": significant,
            "confidence": confidence,
            "p_value": p_value,
            "effect_size": effect_size
        }
    
    def _generate_ab_recommendation(self, winner: Optional[str], 
                                  improvement: float,
                                  significance: Dict[str, Any]) -> str:
        """A/B 테스트 권장사항 생성"""
        
        if not significance["significant"]:
            return "통계적으로 유의한 차이가 발견되지 않았습니다. 추가 테스트가 필요합니다."
        
        if winner and improvement > 5.0:
            return f"{winner} 모델 채택 권장 (성능 개선: {improvement:.1f}%)"
        elif improvement > 0:
            return f"{winner} 모델이 약간 우수하나, 추가 검증 권장"
        else:
            return "두 모델의 성능이 유사합니다. 다른 요소를 고려하여 선택하세요."

class AIBenchmarkSystemV23:
    """AI 벤치마크 시스템 v2.3"""
    
    def __init__(self):
        self.version = "2.3.0"
        self.target_accuracy = 0.992  # 99.2%
        
        # 핵심 컴포넌트
        self.test_generator = TestCaseGenerator()
        self.performance_analyzer = PerformanceAnalyzer()
        self.ab_test_manager = ABTestManager()
        
        # 결과 저장소
        self.benchmark_history = deque(maxlen=10000)
        self.performance_trends = defaultdict(deque)
        
        # 실시간 모니터링
        self.monitoring_active = False
        self.monitoring_interval = 60  # 초
        
        # 성능 통계
        self.system_stats = {
            "total_benchmarks": 0,
            "total_ab_tests": 0,
            "models_tested": set(),
            "scenarios_covered": set(),
            "achievement_history": []
        }
        
        logger.info(f"📊 AI 벤치마크 시스템 v{self.version} 초기화 완료")
        logger.info(f"🎯 목표 정확도: {self.target_accuracy * 100}%")
    
    async def run_comprehensive_benchmark(self, 
                                        models: List[str],
                                        test_count: int = 50,
                                        scenarios: Optional[List[TestScenario]] = None) -> Dict[str, Any]:
        """종합 벤치마크 실행"""
        
        start_time = time.time()
        
        logger.info(f"📊 종합 벤치마크 시작: {len(models)}개 모델, {test_count}개 테스트")
        
        # 테스트 케이스 생성
        test_cases = self.test_generator.generate_test_cases(test_count, scenarios)
        
        # 모델별 벤치마크 실행
        all_results = []
        model_summaries = {}
        
        for model in models:
            logger.info(f"🔍 모델 테스트 중: {model}")
            
            model_results = await self._execute_model_benchmark(model, test_cases)
            all_results.extend(model_results)
            
            # 모델별 요약
            model_summary = self._summarize_model_performance(model_results)
            model_summaries[model] = model_summary
            
            # 시스템 통계 업데이트
            self.system_stats["models_tested"].add(model)
        
        # 전체 성능 분석
        overall_analysis = self.performance_analyzer.analyze_performance(all_results)
        
        # 모델 순위
        model_rankings = self._rank_models(model_summaries)
        
        # 99.2% 목표 달성도 분석
        target_analysis = self._analyze_target_achievement(model_summaries)
        
        # 벤치마크 결과 저장
        benchmark_result = {
            "benchmark_id": f"COMP_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - start_time,
            "models_tested": models,
            "test_cases_count": test_count,
            "scenarios": [s.value for s in (scenarios or list(TestScenario))],
            
            "model_summaries": model_summaries,
            "model_rankings": model_rankings,
            "overall_analysis": overall_analysis,
            "target_achievement": target_analysis,
            
            "recommendations": self._generate_benchmark_recommendations(
                model_rankings, target_analysis
            )
        }
        
        # 결과 저장
        self.benchmark_history.append(benchmark_result)
        self.system_stats["total_benchmarks"] += 1
        
        execution_time = time.time() - start_time
        logger.info(f"✅ 종합 벤치마크 완료 ({execution_time:.2f}초)")
        
        return benchmark_result
    
    async def _execute_model_benchmark(self, model: str, 
                                     test_cases: List[TestCase]) -> List[BenchmarkResult]:
        """개별 모델 벤치마크 실행"""
        
        results = []
        
        for test_case in test_cases:
            try:
                # 실제 구현에서는 모델 실행 로직이 들어감
                # 여기서는 시뮬레이션
                result = await self._simulate_model_execution(model, test_case)
                results.append(result)
                
            except Exception as e:
                logger.error(f"테스트 실행 실패 {model} - {test_case.id}: {e}")
                
                # 실패 결과 생성
                error_result = BenchmarkResult(
                    test_id=test_case.id,
                    model_name=model,
                    scenario=test_case.scenario,
                    metrics={},
                    overall_score=0.0,
                    execution_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    async def _simulate_model_execution(self, model: str, 
                                      test_case: TestCase) -> BenchmarkResult:
        """모델 실행 시뮬레이션 (실제 구현에서는 실제 모델 호출)"""
        
        start_time = time.time()
        
        # 모델별 성능 특성 시뮬레이션
        model_characteristics = {
            "jewelry_specialized_v22": {
                "accuracy_base": 0.95,
                "speed_factor": 1.2,
                "jewelry_expertise": 0.98,
                "cost_factor": 0.8
            },
            "gpt4v": {
                "accuracy_base": 0.92,
                "speed_factor": 0.8,
                "jewelry_expertise": 0.75,
                "cost_factor": 1.5
            },
            "claude_vision": {
                "accuracy_base": 0.90,
                "speed_factor": 1.0,
                "jewelry_expertise": 0.78,
                "cost_factor": 1.2
            },
            "gemini_2.0": {
                "accuracy_base": 0.88,
                "speed_factor": 1.8,
                "jewelry_expertise": 0.70,
                "cost_factor": 0.5
            }
        }
        
        char = model_characteristics.get(model, {
            "accuracy_base": 0.85,
            "speed_factor": 1.0,
            "jewelry_expertise": 0.65,
            "cost_factor": 1.0
        })
        
        # 시뮬레이션된 처리 시간
        base_time = 5.0 + test_case.difficulty_level * 10.0
        actual_time = base_time / char["speed_factor"]
        actual_time += random.uniform(-2.0, 2.0)  # 랜덤 변동
        actual_time = max(1.0, actual_time)
        
        await asyncio.sleep(actual_time / 100)  # 시뮬레이션용 단축
        
        # 성능 지표 시뮬레이션
        difficulty_penalty = test_case.difficulty_level * 0.1
        
        # 정확도 점수
        accuracy = char["accuracy_base"] - difficulty_penalty
        accuracy += random.uniform(-0.05, 0.05)  # 랜덤 변동
        accuracy = max(0.0, min(1.0, accuracy))
        
        # 주얼리 전문성
        jewelry_relevance = char["jewelry_expertise"] - difficulty_penalty * 0.5
        jewelry_relevance += random.uniform(-0.03, 0.03)
        jewelry_relevance = max(0.0, min(1.0, jewelry_relevance))
        
        # 비용
        base_cost = 0.02
        cost = base_cost * char["cost_factor"] * (1 + test_case.difficulty_level)
        
        # 품질 일관성
        consistency = accuracy * 0.95 + random.uniform(-0.02, 0.02)
        consistency = max(0.0, min(1.0, consistency))
        
        # 에러율
        error_rate = (1 - accuracy) * 0.1 + random.uniform(0.0, 0.01)
        error_rate = max(0.0, min(0.1, error_rate))
        
        # 전체 점수 계산
        overall_score = (
            accuracy * 0.4 +
            jewelry_relevance * 0.3 +
            consistency * 0.2 +
            (1 - error_rate) * 0.1
        )
        
        # 메트릭 생성
        metrics = {
            MetricType.ACCURACY_SCORE: BenchmarkMetric(
                MetricType.ACCURACY_SCORE, accuracy, "score"
            ),
            MetricType.RESPONSE_TIME: BenchmarkMetric(
                MetricType.RESPONSE_TIME, actual_time, "seconds"
            ),
            MetricType.COST_PER_REQUEST: BenchmarkMetric(
                MetricType.COST_PER_REQUEST, cost, "USD"
            ),
            MetricType.JEWELRY_RELEVANCE: BenchmarkMetric(
                MetricType.JEWELRY_RELEVANCE, jewelry_relevance, "score"
            ),
            MetricType.QUALITY_CONSISTENCY: BenchmarkMetric(
                MetricType.QUALITY_CONSISTENCY, consistency, "score"
            ),
            MetricType.ERROR_RATE: BenchmarkMetric(
                MetricType.ERROR_RATE, error_rate, "rate"
            )
        }
        
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            test_id=test_case.id,
            model_name=model,
            scenario=test_case.scenario,
            metrics=metrics,
            overall_score=overall_score,
            execution_time=execution_time,
            success=True,
            quality_analysis={
                "difficulty_level": test_case.difficulty_level,
                "scenario": test_case.scenario.value
            }
        )
    
    def _summarize_model_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """모델 성능 요약"""
        
        if not results:
            return {"error": "결과가 없습니다"}
        
        # 기본 통계
        success_results = [r for r in results if r.success]
        success_rate = len(success_results) / len(results)
        
        if not success_results:
            return {
                "success_rate": success_rate,
                "overall_score": 0.0,
                "error": "성공한 테스트가 없습니다"
            }
        
        # 메트릭별 평균
        metric_averages = {}
        metric_aggregates = defaultdict(list)
        
        for result in success_results:
            for metric_type, metric in result.metrics.items():
                metric_aggregates[metric_type].append(metric.value)
        
        for metric_type, values in metric_aggregates.items():
            metric_averages[metric_type.value] = {
                "average": statistics.mean(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values)
            }
        
        # 전체 평균 점수
        overall_scores = [r.overall_score for r in success_results]
        avg_overall_score = statistics.mean(overall_scores)
        
        # 시나리오별 성능
        scenario_performance = defaultdict(list)
        for result in success_results:
            scenario_performance[result.scenario.value].append(result.overall_score)
        
        scenario_averages = {
            scenario: statistics.mean(scores)
            for scenario, scores in scenario_performance.items()
        }
        
        return {
            "success_rate": success_rate,
            "overall_score": avg_overall_score,
            "total_tests": len(results),
            "successful_tests": len(success_results),
            "metric_averages": metric_averages,
            "scenario_performance": scenario_averages,
            "score_distribution": {
                "mean": avg_overall_score,
                "std_dev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                "min": min(overall_scores),
                "max": max(overall_scores)
            }
        }
    
    def _rank_models(self, model_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모델 순위 결정"""
        
        rankings = []
        
        for model, summary in model_summaries.items():
            if "error" in summary:
                continue
            
            # 순위 점수 계산 (정확도 우선)
            rank_score = summary["overall_score"]
            
            # 99.2% 목표 근접도 보너스
            accuracy_avg = summary["metric_averages"].get("accuracy_score", {}).get("average", 0)
            if accuracy_avg >= self.target_accuracy:
                rank_score += 0.1  # 목표 달성 보너스
            
            # 성공률 보정
            rank_score *= summary["success_rate"]
            
            rankings.append({
                "model": model,
                "rank_score": rank_score,
                "overall_score": summary["overall_score"],
                "accuracy": accuracy_avg,
                "success_rate": summary["success_rate"]
            })
        
        # 점수 기준 정렬
        rankings.sort(key=lambda x: x["rank_score"], reverse=True)
        
        # 순위 번호 추가
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _analyze_target_achievement(self, model_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """99.2% 목표 달성도 분석"""
        
        analysis = {
            "target_accuracy": self.target_accuracy,
            "achievement_status": {},
            "best_performing_model": None,
            "gap_analysis": {},
            "recommendations": []
        }
        
        best_accuracy = 0.0
        best_model = None
        
        for model, summary in model_summaries.items():
            if "error" in summary:
                continue
            
            accuracy = summary["metric_averages"].get("accuracy_score", {}).get("average", 0)
            
            # 목표 달성 여부
            achieved = accuracy >= self.target_accuracy
            gap = self.target_accuracy - accuracy if not achieved else 0
            
            analysis["achievement_status"][model] = {
                "accuracy": accuracy,
                "target_achieved": achieved,
                "gap": gap,
                "gap_percentage": (gap / self.target_accuracy * 100) if gap > 0 else 0
            }
            
            # 최고 성능 모델 추적
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        analysis["best_performing_model"] = best_model
        analysis["best_accuracy"] = best_accuracy
        
        # 전체 권장사항
        if best_accuracy >= self.target_accuracy:
            analysis["recommendations"].append(f"✅ 목표 달성! {best_model}이 99.2% 정확도 기준을 만족합니다.")
        else:
            remaining_gap = self.target_accuracy - best_accuracy
            analysis["recommendations"].append(
                f"🎯 목표까지 {remaining_gap:.1%} 추가 개선 필요 (최고: {best_model})"
            )
        
        return analysis
    
    def _generate_benchmark_recommendations(self, rankings: List[Dict[str, Any]], 
                                          target_analysis: Dict[str, Any]) -> List[str]:
        """벤치마크 권장사항 생성"""
        
        recommendations = []
        
        if not rankings:
            return ["모델 테스트 결과가 없습니다."]
        
        # 최고 성능 모델 권장
        best_model = rankings[0]
        recommendations.append(
            f"🥇 최고 성능: {best_model['model']} (전체 점수: {best_model['overall_score']:.3f})"
        )
        
        # 99.2% 목표 관련 권장
        if target_analysis.get("best_accuracy", 0) >= self.target_accuracy:
            recommendations.append("✅ 99.2% 정확도 목표 달성 가능한 모델 확인됨")
        else:
            gap = self.target_accuracy - target_analysis.get("best_accuracy", 0)
            recommendations.append(f"⚡ 99.2% 목표까지 {gap:.1%} 추가 최적화 필요")
        
        # 성능 격차 분석
        if len(rankings) >= 2:
            score_gap = rankings[0]["rank_score"] - rankings[1]["rank_score"]
            if score_gap > 0.1:
                recommendations.append("📊 모델 간 성능 격차가 큼 - 최고 모델 우선 사용 권장")
            else:
                recommendations.append("⚖️ 모델 간 성능이 비슷함 - 비용/속도 등 추가 요소 고려")
        
        # 개선 우선순위
        improvement_areas = []
        
        # 정확도가 낮은 모델들 식별
        low_accuracy_models = [
            r["model"] for r in rankings
            if r.get("accuracy", 0) < 0.90
        ]
        
        if low_accuracy_models:
            improvement_areas.append(f"정확도 개선 필요: {', '.join(low_accuracy_models)}")
        
        if improvement_areas:
            recommendations.extend(improvement_areas)
        
        # 일반 권장사항
        recommendations.append("🔄 정기적 벤치마크를 통한 지속적 성능 모니터링 권장")
        
        return recommendations
    
    async def run_ab_test(self, test_name: str, model_a: str, model_b: str,
                         test_count: int = 30,
                         scenarios: Optional[List[TestScenario]] = None) -> ABTestResult:
        """A/B 테스트 실행"""
        
        logger.info(f"🧪 A/B 테스트 시작: {model_a} vs {model_b}")
        
        # 테스트 케이스 생성
        test_cases = self.test_generator.generate_test_cases(test_count, scenarios)
        
        # A/B 테스트 생성
        test_id = await self.ab_test_manager.create_ab_test(
            test_name, model_a, model_b, test_cases
        )
        
        # 실행 함수 정의
        async def execute_model(model: str, test_case: TestCase) -> BenchmarkResult:
            return await self._simulate_model_execution(model, test_case)
        
        # A/B 테스트 실행
        ab_result = await self.ab_test_manager.execute_ab_test(test_id, execute_model)
        
        # 시스템 통계 업데이트
        self.system_stats["total_ab_tests"] += 1
        self.system_stats["models_tested"].add(model_a)
        self.system_stats["models_tested"].add(model_b)
        
        logger.info(f"✅ A/B 테스트 완료: {ab_result.winner or '무승부'}")
        
        return ab_result
    
    async def start_real_time_monitoring(self, models: List[str],
                                       monitoring_interval: int = 300):
        """실시간 성능 모니터링 시작"""
        
        self.monitoring_active = True
        self.monitoring_interval = monitoring_interval
        
        logger.info(f"👁️ 실시간 모니터링 시작: {len(models)}개 모델, {monitoring_interval}초 간격")
        
        while self.monitoring_active:
            try:
                # 간단한 테스트 케이스로 모니터링
                test_cases = self.test_generator.generate_test_cases(5)
                
                for model in models:
                    results = await self._execute_model_benchmark(model, test_cases)
                    
                    # 성능 트렌드 업데이트
                    if results:
                        avg_score = statistics.mean(r.overall_score for r in results if r.success)
                        self.performance_trends[model].append({
                            "timestamp": datetime.now(),
                            "score": avg_score
                        })
                        
                        # 최근 데이터만 유지 (최대 100개)
                        if len(self.performance_trends[model]) > 100:
                            self.performance_trends[model].popleft()
                
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 대기
    
    def stop_real_time_monitoring(self):
        """실시간 모니터링 중지"""
        
        self.monitoring_active = False
        logger.info("👁️ 실시간 모니터링 중지됨")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """종합 성능 리포트 생성"""
        
        report = {
            "system_info": {
                "version": self.version,
                "target_accuracy": f"{self.target_accuracy * 100}%",
                "total_benchmarks": self.system_stats["total_benchmarks"],
                "total_ab_tests": self.system_stats["total_ab_tests"],
                "models_tested": len(self.system_stats["models_tested"]),
                "scenarios_covered": len(self.system_stats["scenarios_covered"])
            },
            
            "recent_performance": {},
            "trending_analysis": {},
            "achievement_summary": {},
            
            "recommendations": [
                "정기적 종합 벤치마크 실행",
                "A/B 테스트를 통한 모델 비교",
                "실시간 모니터링으로 성능 추적"
            ]
        }
        
        # 최근 벤치마크 결과 요약
        if self.benchmark_history:
            latest_benchmark = self.benchmark_history[-1]
            report["recent_performance"] = {
                "timestamp": latest_benchmark["timestamp"],
                "best_model": latest_benchmark["model_rankings"][0]["model"] if latest_benchmark["model_rankings"] else "없음",
                "target_achievement": latest_benchmark["target_achievement"].get("best_accuracy", 0) >= self.target_accuracy
            }
        
        # 성능 트렌드 분석
        for model, trend_data in self.performance_trends.items():
            if len(trend_data) >= 3:
                recent_scores = [d["score"] for d in list(trend_data)[-10:]]
                trend_direction = "상승" if recent_scores[-1] > recent_scores[0] else "하락"
                
                report["trending_analysis"][model] = {
                    "direction": trend_direction,
                    "current_score": recent_scores[-1],
                    "score_change": recent_scores[-1] - recent_scores[0]
                }
        
        return report
    
    async def export_benchmark_data(self, filepath: str, format: str = "json"):
        """벤치마크 데이터 내보내기"""
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "version": self.version,
                "format": format
            },
            "system_stats": {
                **self.system_stats,
                "models_tested": list(self.system_stats["models_tested"]),
                "scenarios_covered": list(self.system_stats["scenarios_covered"])
            },
            "benchmark_history": list(self.benchmark_history),
            "performance_trends": {
                model: list(trend) for model, trend in self.performance_trends.items()
            }
        }
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        elif format.lower() == "csv":
            # CSV 형태로 주요 메트릭만 내보내기
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 헤더
                writer.writerow([
                    "timestamp", "benchmark_id", "model", "overall_score", 
                    "accuracy", "response_time", "success_rate"
                ])
                
                # 데이터
                for benchmark in self.benchmark_history:
                    for model, summary in benchmark.get("model_summaries", {}).items():
                        if "error" not in summary:
                            writer.writerow([
                                benchmark["timestamp"],
                                benchmark["benchmark_id"],
                                model,
                                summary["overall_score"],
                                summary["metric_averages"].get("accuracy_score", {}).get("average", 0),
                                summary["metric_averages"].get("response_time", {}).get("average", 0),
                                summary["success_rate"]
                            ])
        
        logger.info(f"📄 벤치마크 데이터 내보내기 완료: {filepath}")

# 테스트 및 데모 함수들

async def test_benchmark_system_v23():
    """AI 벤치마크 시스템 v2.3 테스트"""
    
    print("📊 솔로몬드 AI 벤치마크 시스템 v2.3 테스트")
    print("=" * 60)
    
    # 시스템 초기화
    benchmark_system = AIBenchmarkSystemV23()
    
    # 테스트 케이스 1: 종합 벤치마크
    print("\n🔹 테스트 1: 종합 벤치마크 실행")
    
    test_models = [
        "jewelry_specialized_v22",
        "gpt4v",
        "claude_vision",
        "gemini_2.0"
    ]
    
    benchmark_result = await benchmark_system.run_comprehensive_benchmark(
        models=test_models,
        test_count=20,
        scenarios=[TestScenario.DIAMOND_4C_ANALYSIS, TestScenario.RUBY_GRADING]
    )
    
    print(f"실행 시간: {benchmark_result['execution_time']:.2f}초")
    print(f"테스트된 모델: {len(benchmark_result['models_tested'])}개")
    print(f"테스트 케이스: {benchmark_result['test_cases_count']}개")
    
    print(f"\n🏆 모델 순위:")
    for ranking in benchmark_result['model_rankings'][:3]:
        print(f"  {ranking['rank']}. {ranking['model']} - 점수: {ranking['rank_score']:.3f}")
    
    print(f"\n🎯 99.2% 목표 달성도:")
    target_analysis = benchmark_result['target_achievement']
    print(f"최고 성능 모델: {target_analysis['best_performing_model']}")
    print(f"최고 정확도: {target_analysis['best_accuracy']:.1%}")
    
    # 테스트 케이스 2: A/B 테스트
    print("\n🔹 테스트 2: A/B 테스트 실행")
    
    ab_result = await benchmark_system.run_ab_test(
        test_name="전문성_vs_범용성",
        model_a="jewelry_specialized_v22",
        model_b="gpt4v",
        test_count=15
    )
    
    print(f"테스트 모델: {ab_result.model_a} vs {ab_result.model_b}")
    print(f"샘플 크기: {ab_result.sample_size}")
    print(f"통계적 유의성: {'유의함' if ab_result.statistical_significance else '유의하지 않음'}")
    print(f"승자: {ab_result.winner or '무승부'}")
    
    if ab_result.winner:
        print(f"성능 개선: {ab_result.improvement_percentage:.1f}%")
    
    print(f"권장사항: {ab_result.recommendation}")
    
    # 테스트 케이스 3: 실시간 모니터링 (짧은 시간)
    print("\n🔹 테스트 3: 실시간 모니터링 (5초 시연)")
    
    # 모니터링 시작 (백그라운드)
    monitoring_task = asyncio.create_task(
        benchmark_system.start_real_time_monitoring(
            models=["jewelry_specialized_v22", "gpt4v"],
            monitoring_interval=2
        )
    )
    
    # 5초 대기
    await asyncio.sleep(5)
    
    # 모니터링 중지
    benchmark_system.stop_real_time_monitoring()
    
    try:
        await asyncio.wait_for(monitoring_task, timeout=1.0)
    except asyncio.TimeoutError:
        monitoring_task.cancel()
    
    print("✅ 실시간 모니터링 테스트 완료")
    
    # 종합 리포트
    print("\n📈 종합 성능 리포트:")
    comprehensive_report = benchmark_system.get_comprehensive_report()
    
    print(f"시스템 버전: {comprehensive_report['system_info']['version']}")
    print(f"목표 정확도: {comprehensive_report['system_info']['target_accuracy']}")
    print(f"총 벤치마크: {comprehensive_report['system_info']['total_benchmarks']}회")
    print(f"총 A/B 테스트: {comprehensive_report['system_info']['total_ab_tests']}회")
    print(f"테스트된 모델: {comprehensive_report['system_info']['models_tested']}개")
    
    if comprehensive_report.get('recent_performance'):
        recent = comprehensive_report['recent_performance']
        print(f"\n최근 성능:")
        print(f"  최고 모델: {recent['best_model']}")
        print(f"  목표 달성: {'달성' if recent['target_achievement'] else '미달성'}")
    
    print(f"\n📋 권장사항:")
    for recommendation in comprehensive_report['recommendations']:
        print(f"  • {recommendation}")
    
    # 데이터 내보내기 테스트
    print(f"\n💾 데이터 내보내기 테스트:")
    
    export_path = f"benchmark_data_{int(time.time())}.json"
    await benchmark_system.export_benchmark_data(export_path, "json")
    print(f"JSON 내보내기 완료: {export_path}")
    
    print("\n" + "=" * 60)
    print("✅ AI 벤치마크 시스템 v2.3 테스트 완료!")
    
    return benchmark_system

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_benchmark_system_v23())
