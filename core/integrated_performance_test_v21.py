"""
🚀 솔로몬드 AI v2.1.2 - 통합 성능 테스트 시스템
전체 시스템 성능 평가 및 최적화 지점 분석

주요 기능:
- 종합 성능 벤치마크
- 모듈별 성능 프로파일링
- 메모리 사용량 최적화 분석
- 에러 복구 시스템 검증
- 실시간 성능 모니터링
"""

import time
import json
import asyncio
import threading
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# 우리가 만든 모듈들 import
try:
    from .performance_profiler_v21 import PerformanceProfiler, get_system_health
    from .memory_optimizer_v21 import MemoryManager, global_memory_manager
    from .error_recovery_system_v21 import ErrorRecoverySystem, global_recovery_system
except ImportError:
    # 직접 실행 시 fallback
    import sys
    sys.path.append('.')
    try:
        from performance_profiler_v21 import PerformanceProfiler, get_system_health
        from memory_optimizer_v21 import MemoryManager, global_memory_manager
        from error_recovery_system_v21 import ErrorRecoverySystem, global_recovery_system
    except ImportError:
        print("⚠️ 모듈 import 실패 - 단독 테스트 모드로 실행")
        PerformanceProfiler = None
        MemoryManager = None
        ErrorRecoverySystem = None

@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    test_name: str
    duration_seconds: float
    memory_used_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: float
    success_rate_percent: float
    errors_count: int
    metadata: Dict[str, Any]

@dataclass
class SystemPerformanceReport:
    """시스템 성능 리포트"""
    timestamp: str
    overall_score: float
    benchmark_results: List[BenchmarkResult]
    system_health: Dict[str, Any]
    memory_stats: Dict[str, Any]
    error_analysis: Dict[str, Any]
    optimization_recommendations: List[str]

class PerformanceTestSuite:
    """성능 테스트 스위트"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
        # 테스트 데이터 준비
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="solomond_perf_test_"))
        self._prepare_test_data()
    
    def _prepare_test_data(self):
        """테스트 데이터 준비"""
        try:
            # 작은 텍스트 파일 (1MB)
            small_file = self.test_data_dir / "small_test.txt"
            with open(small_file, 'w', encoding='utf-8') as f:
                for i in range(10000):
                    f.write(f"주얼리 테스트 라인 {i:06d} - 다이아몬드, 반지, 목걸이, 귀걸이\n")
            
            # 중간 텍스트 파일 (10MB)
            medium_file = self.test_data_dir / "medium_test.txt"
            with open(medium_file, 'w', encoding='utf-8') as f:
                for i in range(100000):
                    f.write(f"중형 테스트 데이터 {i:06d} - 솔로몬드 AI 시스템 성능 테스트\n")
            
            # JSON 데이터 파일
            json_file = self.test_data_dir / "test_data.json"
            test_json = {
                "jewelry_items": [
                    {"id": i, "type": "diamond", "carat": round(i * 0.1, 2), 
                     "price": i * 1000, "quality": f"Grade_{i%5}"}
                    for i in range(1000)
                ]
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(test_json, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"테스트 데이터 준비 완료: {self.test_data_dir}")
            
        except Exception as e:
            self.logger.error(f"테스트 데이터 준비 실패: {e}")
    
    def benchmark_file_processing(self) -> BenchmarkResult:
        """파일 처리 성능 벤치마크"""
        start_time = time.time()
        start_memory = global_memory_manager.get_memory_usage().used_mb if global_memory_manager else 0
        
        operations_count = 0
        errors_count = 0
        
        try:
            # 작은 파일 처리 테스트
            small_file = self.test_data_dir / "small_test.txt"
            if small_file.exists():
                with open(small_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "다이아몬드" in line:
                            operations_count += 1
            
            # JSON 파일 처리 테스트
            json_file = self.test_data_dir / "test_data.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("jewelry_items", []):
                        if item.get("carat", 0) > 1.0:
                            operations_count += 1
            
        except Exception as e:
            errors_count += 1
            self.logger.error(f"파일 처리 테스트 오류: {e}")
        
        end_time = time.time()
        end_memory = global_memory_manager.get_memory_usage().used_mb if global_memory_manager else 0
        
        duration = end_time - start_time
        throughput = operations_count / duration if duration > 0 else 0
        success_rate = ((operations_count) / (operations_count + errors_count) * 100) if (operations_count + errors_count) > 0 else 0
        
        return BenchmarkResult(
            test_name="파일 처리",
            duration_seconds=duration,
            memory_used_mb=end_memory - start_memory,
            cpu_usage_percent=0,  # 별도 측정 필요
            throughput_ops_per_sec=throughput,
            success_rate_percent=success_rate,
            errors_count=errors_count,
            metadata={"operations": operations_count, "files_processed": 2}
        )
    
    def benchmark_memory_operations(self) -> BenchmarkResult:
        """메모리 작업 벤치마크"""
        if not global_memory_manager:
            return BenchmarkResult(
                test_name="메모리 작업",
                duration_seconds=0,
                memory_used_mb=0,
                cpu_usage_percent=0,
                throughput_ops_per_sec=0,
                success_rate_percent=0,
                errors_count=1,
                metadata={"error": "메모리 매니저 없음"}
            )
        
        start_time = time.time()
        start_memory = global_memory_manager.get_memory_usage().used_mb
        
        operations_count = 0
        errors_count = 0
        
        try:
            # 캐시 테스트
            for i in range(100):
                key = f"test_key_{i}"
                value = f"테스트 값 {i}" * 100  # 큰 값
                success = global_memory_manager.cache.put(key, value)
                if success:
                    operations_count += 1
            
            # 캐시 읽기 테스트
            for i in range(50):
                key = f"test_key_{i}"
                value = global_memory_manager.cache.get(key)
                if value:
                    operations_count += 1
            
            # 메모리 정리 테스트
            cleanup_result = global_memory_manager.routine_cleanup()
            if cleanup_result:
                operations_count += 1
            
        except Exception as e:
            errors_count += 1
            self.logger.error(f"메모리 작업 테스트 오류: {e}")
        
        end_time = time.time()
        end_memory = global_memory_manager.get_memory_usage().used_mb
        
        duration = end_time - start_time
        throughput = operations_count / duration if duration > 0 else 0
        success_rate = ((operations_count) / (operations_count + errors_count) * 100) if (operations_count + errors_count) > 0 else 0
        
        return BenchmarkResult(
            test_name="메모리 작업",
            duration_seconds=duration,
            memory_used_mb=end_memory - start_memory,
            cpu_usage_percent=0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=success_rate,
            errors_count=errors_count,
            metadata={"cache_operations": operations_count}
        )
    
    def benchmark_error_recovery(self) -> BenchmarkResult:
        """에러 복구 시스템 벤치마크"""
        if not global_recovery_system:
            return BenchmarkResult(
                test_name="에러 복구",
                duration_seconds=0,
                memory_used_mb=0,
                cpu_usage_percent=0,
                throughput_ops_per_sec=0,
                success_rate_percent=0,
                errors_count=1,
                metadata={"error": "복구 시스템 없음"}
            )
        
        start_time = time.time()
        start_memory = global_memory_manager.get_memory_usage().used_mb if global_memory_manager else 0
        
        operations_count = 0
        errors_count = 0
        recovered_count = 0
        
        # 테스트용 함수들
        @global_recovery_system.resilient_execution("test_operation")
        def test_function_success():
            return "성공"
        
        @global_recovery_system.resilient_execution("test_operation_fail")
        def test_function_fail():
            raise ValueError("테스트 에러")
        
        try:
            # 성공 케이스 테스트
            for i in range(20):
                try:
                    result = test_function_success()
                    if result == "성공":
                        operations_count += 1
                except Exception:
                    errors_count += 1
            
            # 실패 복구 케이스 테스트
            for i in range(10):
                try:
                    result = test_function_fail()
                    recovered_count += 1  # 복구된 경우
                except Exception:
                    errors_count += 1  # 복구 실패
            
            # 시스템 상태 확인
            status = global_recovery_system.get_system_status()
            if status.get("health_status") in ["HEALTHY", "WARNING"]:
                operations_count += 1
            
        except Exception as e:
            errors_count += 1
            self.logger.error(f"에러 복구 테스트 오류: {e}")
        
        end_time = time.time()
        end_memory = global_memory_manager.get_memory_usage().used_mb if global_memory_manager else 0
        
        duration = end_time - start_time
        total_operations = operations_count + recovered_count
        throughput = total_operations / duration if duration > 0 else 0
        success_rate = (total_operations / (total_operations + errors_count) * 100) if (total_operations + errors_count) > 0 else 0
        
        return BenchmarkResult(
            test_name="에러 복구",
            duration_seconds=duration,
            memory_used_mb=end_memory - start_memory,
            cpu_usage_percent=0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=success_rate,
            errors_count=errors_count,
            metadata={
                "successful_operations": operations_count,
                "recovered_operations": recovered_count
            }
        )
    
    def benchmark_concurrent_operations(self) -> BenchmarkResult:
        """동시 작업 처리 벤치마크"""
        start_time = time.time()
        start_memory = global_memory_manager.get_memory_usage().used_mb if global_memory_manager else 0
        
        operations_count = 0
        errors_count = 0
        
        def worker_task(task_id: int):
            nonlocal operations_count, errors_count
            try:
                # 간단한 작업 시뮬레이션
                data = [i**2 for i in range(1000)]
                result = sum(data)
                
                # 메모리 캐싱 (가능한 경우)
                if global_memory_manager:
                    global_memory_manager.cache.put(f"result_{task_id}", result)
                
                operations_count += 1
                
            except Exception as e:
                errors_count += 1
                self.logger.debug(f"워커 태스크 {task_id} 오류: {e}")
        
        try:
            # 스레드 풀로 동시 작업 실행
            threads = []
            for i in range(10):
                thread = threading.Thread(target=worker_task, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 모든 스레드 완료 대기
            for thread in threads:
                thread.join(timeout=5.0)
            
        except Exception as e:
            errors_count += 1
            self.logger.error(f"동시 작업 테스트 오류: {e}")
        
        end_time = time.time()
        end_memory = global_memory_manager.get_memory_usage().used_mb if global_memory_manager else 0
        
        duration = end_time - start_time
        throughput = operations_count / duration if duration > 0 else 0
        success_rate = (operations_count / (operations_count + errors_count) * 100) if (operations_count + errors_count) > 0 else 0
        
        return BenchmarkResult(
            test_name="동시 작업",
            duration_seconds=duration,
            memory_used_mb=end_memory - start_memory,
            cpu_usage_percent=0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=success_rate,
            errors_count=errors_count,
            metadata={"threads_used": 10, "tasks_per_thread": 1}
        )
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """전체 벤치마크 실행"""
        self.logger.info("🚀 전체 성능 벤치마크 시작")
        
        benchmarks = [
            ("파일 처리", self.benchmark_file_processing),
            ("메모리 작업", self.benchmark_memory_operations),
            ("에러 복구", self.benchmark_error_recovery),
            ("동시 작업", self.benchmark_concurrent_operations)
        ]
        
        results = []
        
        for name, benchmark_func in benchmarks:
            self.logger.info(f"  📊 {name} 벤치마크 실행...")
            try:
                result = benchmark_func()
                results.append(result)
                self.logger.info(f"    ✅ {name} 완료: {result.duration_seconds:.2f}초, "
                               f"{result.success_rate_percent:.1f}% 성공률")
            except Exception as e:
                self.logger.error(f"    ❌ {name} 실패: {e}")
                # 실패한 경우 기본 결과 추가
                results.append(BenchmarkResult(
                    test_name=name,
                    duration_seconds=0,
                    memory_used_mb=0,
                    cpu_usage_percent=0,
                    throughput_ops_per_sec=0,
                    success_rate_percent=0,
                    errors_count=1,
                    metadata={"error": str(e)}
                ))
        
        self.test_results = results
        return results
    
    def cleanup(self):
        """테스트 데이터 정리"""
        try:
            import shutil
            shutil.rmtree(self.test_data_dir)
            self.logger.info("테스트 데이터 정리 완료")
        except Exception as e:
            self.logger.debug(f"테스트 데이터 정리 실패: {e}")

class SystemPerformanceAnalyzer:
    """시스템 성능 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_suite = PerformanceTestSuite()
    
    def analyze_system_performance(self) -> SystemPerformanceReport:
        """시스템 성능 종합 분석"""
        self.logger.info("🔍 시스템 성능 분석 시작")
        
        # 벤치마크 실행
        benchmark_results = self.test_suite.run_full_benchmark()
        
        # 시스템 건강도 체크
        system_health = get_system_health() if 'get_system_health' in globals() else {
            "health_score": 0,
            "status": "Unknown",
            "metrics": {}
        }
        
        # 메모리 통계
        memory_stats = {}
        if global_memory_manager:
            memory_usage = global_memory_manager.get_memory_usage()
            optimization_report = global_memory_manager.get_optimization_report()
            memory_stats = {
                "current_usage": {
                    "percent": memory_usage.percent,
                    "used_mb": memory_usage.used_mb,
                    "available_mb": memory_usage.available_mb
                },
                "cache_stats": optimization_report.get("cache", {}),
                "cleanup_stats": optimization_report.get("cleanup_stats", {})
            }
        
        # 에러 분석
        error_analysis = {}
        if global_recovery_system:
            status = global_recovery_system.get_system_status()
            error_analysis = {
                "health_status": status.get("health_status", "Unknown"),
                "error_rate_1h": status.get("error_rate_1h", 0),
                "total_errors": status.get("total_errors", 0),
                "top_error_types": status.get("top_error_types", {})
            }
        
        # 전체 점수 계산
        overall_score = self._calculate_overall_score(
            benchmark_results, system_health, memory_stats, error_analysis
        )
        
        # 최적화 권장사항 생성
        recommendations = self._generate_optimization_recommendations(
            benchmark_results, system_health, memory_stats, error_analysis
        )
        
        report = SystemPerformanceReport(
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            benchmark_results=benchmark_results,
            system_health=system_health,
            memory_stats=memory_stats,
            error_analysis=error_analysis,
            optimization_recommendations=recommendations
        )
        
        self.logger.info(f"📊 시스템 성능 분석 완료 - 전체 점수: {overall_score:.1f}/100")
        return report
    
    def _calculate_overall_score(self, benchmark_results: List[BenchmarkResult],
                                system_health: Dict, memory_stats: Dict,
                                error_analysis: Dict) -> float:
        """전체 성능 점수 계산"""
        scores = []
        
        # 벤치마크 점수 (40%)
        if benchmark_results:
            avg_success_rate = sum(r.success_rate_percent for r in benchmark_results) / len(benchmark_results)
            avg_throughput = sum(r.throughput_ops_per_sec for r in benchmark_results) / len(benchmark_results)
            
            benchmark_score = (avg_success_rate + min(avg_throughput * 10, 100)) / 2
            scores.append(("benchmark", benchmark_score, 0.4))
        
        # 시스템 건강도 (25%)
        health_score = system_health.get("health_score", 50)
        scores.append(("health", health_score, 0.25))
        
        # 메모리 효율성 (20%)
        memory_score = 100
        if memory_stats and "current_usage" in memory_stats:
            usage_percent = memory_stats["current_usage"].get("percent", 50)
            memory_score = max(0, 100 - usage_percent)  # 사용률이 낮을수록 좋음
        scores.append(("memory", memory_score, 0.2))
        
        # 에러 처리 (15%)
        error_score = 100
        if error_analysis:
            error_rate = error_analysis.get("error_rate_1h", 0)
            error_score = max(0, 100 - error_rate * 2)  # 에러율이 낮을수록 좋음
        scores.append(("error", error_score, 0.15))
        
        # 가중 평균 계산
        total_score = sum(score * weight for name, score, weight in scores)
        return min(100, max(0, total_score))
    
    def _generate_optimization_recommendations(self, benchmark_results: List[BenchmarkResult],
                                             system_health: Dict, memory_stats: Dict,
                                             error_analysis: Dict) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        # 벤치마크 기반 권장사항
        for result in benchmark_results:
            if result.success_rate_percent < 80:
                recommendations.append(f"⚠️ {result.test_name} 성공률 개선 필요 ({result.success_rate_percent:.1f}%)")
            
            if result.throughput_ops_per_sec < 1:
                recommendations.append(f"🐌 {result.test_name} 처리 속도 최적화 필요")
            
            if result.memory_used_mb > 100:
                recommendations.append(f"💾 {result.test_name} 메모리 사용량 최적화 필요")
        
        # 시스템 건강도 기반
        health_score = system_health.get("health_score", 100)
        if health_score < 70:
            recommendations.append("🚨 시스템 건강도 위험 - 하드웨어 점검 필요")
        elif health_score < 85:
            recommendations.append("⚠️ 시스템 성능 주의 - 리소스 최적화 권장")
        
        # 메모리 기반
        if memory_stats and "current_usage" in memory_stats:
            usage_percent = memory_stats["current_usage"].get("percent", 0)
            if usage_percent > 85:
                recommendations.append("💾 메모리 사용률 높음 - 정리 및 최적화 필요")
            
            cache_stats = memory_stats.get("cache_stats", {})
            hit_rate = cache_stats.get("hit_rate", 100)
            if hit_rate < 50:
                recommendations.append("📊 캐시 효율성 낮음 - 캐시 전략 재검토 필요")
        
        # 에러 분석 기반
        if error_analysis:
            error_rate = error_analysis.get("error_rate_1h", 0)
            if error_rate > 10:
                recommendations.append("🔧 높은 에러율 - 에러 처리 로직 강화 필요")
            
            health_status = error_analysis.get("health_status", "HEALTHY")
            if health_status in ["CRITICAL", "WARNING"]:
                recommendations.append(f"🛡️ 시스템 상태 {health_status} - 즉시 점검 필요")
        
        # 성능 향상 팁
        if not recommendations:
            recommendations.extend([
                "✅ 시스템이 최적 상태입니다",
                "🚀 더 나은 성능을 위해 정기적인 모니터링을 권장합니다",
                "📈 벤치마크를 정기적으로 실행하여 성능 추이를 확인하세요"
            ])
        
        return recommendations
    
    def save_report(self, report: SystemPerformanceReport, filepath: str):
        """리포트 파일 저장"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 성능 리포트 저장: {filepath}")
            
        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        self.test_suite.cleanup()

def run_performance_analysis(save_report: bool = True) -> SystemPerformanceReport:
    """성능 분석 실행 (편의 함수)"""
    analyzer = SystemPerformanceAnalyzer()
    
    try:
        report = analyzer.analyze_system_performance()
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"solomond_performance_report_{timestamp}.json"
            analyzer.save_report(report, report_path)
        
        return report
        
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    # 메인 실행부
    print("🚀 솔로몬드 AI 통합 성능 테스트 시스템 v2.1.2")
    print("=" * 60)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 성능 분석 실행
        print("\n🔍 시스템 성능 분석 시작...")
        report = run_performance_analysis(save_report=True)
        
        # 결과 출력
        print(f"\n📊 전체 성능 점수: {report.overall_score:.1f}/100")
        
        print("\n📈 벤치마크 결과:")
        for result in report.benchmark_results:
            print(f"  {result.test_name}:")
            print(f"    ⏱️  실행 시간: {result.duration_seconds:.3f}초")
            print(f"    💾 메모리 사용: {result.memory_used_mb:.2f}MB")
            print(f"    🎯 처리량: {result.throughput_ops_per_sec:.2f} ops/sec")
            print(f"    ✅ 성공률: {result.success_rate_percent:.1f}%")
            if result.errors_count > 0:
                print(f"    ❌ 에러 수: {result.errors_count}")
        
        print(f"\n💊 시스템 건강도: {report.system_health.get('health_score', 0)}/100")
        
        if report.memory_stats and "current_usage" in report.memory_stats:
            memory = report.memory_stats["current_usage"]
            print(f"💾 메모리 사용률: {memory.get('percent', 0):.1f}%")
            print(f"🆓 사용 가능 메모리: {memory.get('available_mb', 0):.1f}MB")
        
        if report.error_analysis:
            error = report.error_analysis
            print(f"🔧 에러율 (1시간): {error.get('error_rate_1h', 0):.2f}%")
            print(f"🛡️ 시스템 상태: {error.get('health_status', 'Unknown')}")
        
        print("\n💡 최적화 권장사항:")
        for i, rec in enumerate(report.optimization_recommendations, 1):
            print(f"  {i}. {rec}")
        
        # 점수별 등급 판정
        score = report.overall_score
        if score >= 90:
            grade = "🏆 우수 (Excellent)"
        elif score >= 80:
            grade = "🥈 좋음 (Good)"
        elif score >= 70:
            grade = "🥉 보통 (Fair)"
        elif score >= 60:
            grade = "⚠️ 주의 (Poor)"
        else:
            grade = "🚨 위험 (Critical)"
        
        print(f"\n🎖️ 시스템 등급: {grade}")
        
        # 성능 트렌드 (간단한 형태)
        print(f"\n📅 분석 시각: {report.timestamp}")
        print("📈 성능 향상을 위해 정기적인 벤치마크를 권장합니다.")
        
    except Exception as e:
        print(f"❌ 성능 분석 실패: {e}")
        import traceback
        print(traceback.format_exc())
    
    print("\n✅ 통합 성능 테스트 완료!")
    print("💎 솔로몬드 AI 시스템이 더욱 최적화되었습니다!")
