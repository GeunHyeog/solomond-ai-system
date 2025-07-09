# Phase 2 Week 3 Day 4-5: 통합 테스트 및 성능 벤치마크
# 스트리밍 + 복구 + 고급검증 시스템 통합 테스트

import asyncio
import time
import json
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import tempfile
import os
import numpy as np
from datetime import datetime
import threading
import traceback

# 개발한 모듈들 임포트
from core.streaming_processor import StreamingProcessor, StreamingConfig, MemoryMonitor
from core.recovery_manager import RecoveryManager, RecoveryConfig, RecoveryLevel
from core.advanced_validator import AdvancedCrossValidator, ValidationLevel

class BenchmarkLevel(Enum):
    """벤치마크 레벨"""
    LIGHT = "light"       # 가벼운 테스트
    STANDARD = "standard" # 표준 테스트
    HEAVY = "heavy"       # 무거운 테스트
    EXTREME = "extreme"   # 극한 테스트

@dataclass
class TestScenario:
    """테스트 시나리오"""
    scenario_id: str
    name: str
    description: str
    file_count: int
    total_size_mb: float
    file_types: List[str]
    expected_duration: float
    stress_factors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """성능 지표"""
    scenario_id: str
    
    # 처리 성능
    total_processing_time: float
    average_file_time: float
    throughput_mbps: float
    
    # 메모리 성능
    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency: float  # 처리량 대비 메모리 사용량
    
    # 검증 성능
    validation_accuracy: float
    anomaly_detection_rate: float
    false_positive_rate: float
    
    # 복구 성능
    error_recovery_rate: float
    recovery_time: float
    checkpoint_overhead: float
    
    # 안정성 지표
    success_rate: float
    error_count: int
    critical_errors: int
    
    # 사용자 경험
    responsiveness_score: float  # UI 반응성
    progress_accuracy: float     # 진행률 정확도

@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    benchmark_id: str
    timestamp: datetime
    system_info: Dict[str, Any]
    test_scenarios: List[TestScenario]
    performance_metrics: List[PerformanceMetrics]
    overall_score: float
    recommendations: List[str]
    detailed_logs: Dict[str, Any]

class SystemResourceMonitor:
    """시스템 리소스 모니터"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.resource_history = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring = True
        self.resource_history.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # 메모리 사용률
                memory = psutil.virtual_memory()
                
                # 디스크 I/O
                disk_io = psutil.disk_io_counters()
                
                # 네트워크 I/O (가능한 경우)
                try:
                    net_io = psutil.net_io_counters()
                except:
                    net_io = None
                
                resource_data = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_used_mb": memory.used / 1024 / 1024,
                    "memory_percent": memory.percent,
                    "disk_read_mb": disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                    "disk_write_mb": disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
                    "net_sent_mb": net_io.bytes_sent / 1024 / 1024 if net_io else 0,
                    "net_recv_mb": net_io.bytes_recv / 1024 / 1024 if net_io else 0
                }
                
                self.resource_history.append(resource_data)
                
                # 메모리 절약을 위해 오래된 데이터 제거 (최근 1000개 유지)
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logging.error(f"리소스 모니터링 오류: {e}")
                time.sleep(self.sampling_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """리소스 사용 요약"""
        if not self.resource_history:
            return {}
        
        cpu_values = [r["cpu_percent"] for r in self.resource_history]
        memory_values = [r["memory_used_mb"] for r in self.resource_history]
        
        return {
            "cpu": {
                "average": np.mean(cpu_values),
                "peak": np.max(cpu_values),
                "min": np.min(cpu_values)
            },
            "memory": {
                "average_mb": np.mean(memory_values),
                "peak_mb": np.max(memory_values),
                "min_mb": np.min(memory_values)
            },
            "samples_count": len(self.resource_history),
            "duration_seconds": (self.resource_history[-1]["timestamp"] - self.resource_history[0]["timestamp"]) if len(self.resource_history) > 1 else 0
        }

class TestDataGenerator:
    """테스트 데이터 생성기"""
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def generate_test_files(self, scenario: TestScenario) -> List[str]:
        """시나리오에 맞는 테스트 파일 생성"""
        test_files = []
        
        # 파일 크기 분배
        file_sizes = self._calculate_file_sizes(scenario.total_size_mb, scenario.file_count)
        
        for i in range(scenario.file_count):
            file_type = scenario.file_types[i % len(scenario.file_types)]
            file_size_mb = file_sizes[i]
            
            file_path = self._generate_test_file(
                file_type=file_type,
                size_mb=file_size_mb,
                file_index=i,
                scenario_id=scenario.scenario_id
            )
            
            test_files.append(file_path)
        
        return test_files
    
    def _calculate_file_sizes(self, total_mb: float, file_count: int) -> List[float]:
        """파일 크기 분배 계산"""
        if file_count == 1:
            return [total_mb]
        
        # 로그 정규분포를 사용하여 현실적인 파일 크기 분포 생성
        base_size = total_mb / file_count
        
        # 일부는 크고, 일부는 작게
        sizes = []
        remaining_mb = total_mb
        
        for i in range(file_count - 1):
            # 랜덤한 크기 (평균 주변으로 변동)
            variation = np.random.uniform(0.5, 2.0)
            size = min(base_size * variation, remaining_mb * 0.8)
            size = max(size, 1.0)  # 최소 1MB
            
            sizes.append(size)
            remaining_mb -= size
        
        # 마지막 파일은 남은 크기
        sizes.append(max(1.0, remaining_mb))
        
        return sizes
    
    def _generate_test_file(self, file_type: str, size_mb: float, file_index: int, scenario_id: str) -> str:
        """단일 테스트 파일 생성"""
        
        # 파일 확장자 결정
        extensions = {
            "audio": ".wav",
            "video": ".mp4", 
            "document": ".txt",
            "image": ".jpg"
        }
        
        ext = extensions.get(file_type, ".bin")
        filename = f"{scenario_id}_file_{file_index:03d}{ext}"
        file_path = self.temp_dir / filename
        
        # 파일 크기를 바이트로 변환
        size_bytes = int(size_mb * 1024 * 1024)
        
        # 파일 타입별 내용 생성
        if file_type == "document":
            content = self._generate_jewelry_text_content(size_bytes)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # 바이너리 파일 (실제 오디오/비디오/이미지 대신 랜덤 데이터)
            with open(file_path, 'wb') as f:
                # 청크 단위로 작성하여 메모리 효율성 확보
                chunk_size = 1024 * 1024  # 1MB 청크
                written = 0
                
                while written < size_bytes:
                    chunk_size_actual = min(chunk_size, size_bytes - written)
                    chunk_data = os.urandom(chunk_size_actual)
                    f.write(chunk_data)
                    written += chunk_size_actual
        
        return str(file_path)
    
    def _generate_jewelry_text_content(self, target_bytes: int) -> str:
        """주얼리 관련 텍스트 내용 생성"""
        
        # 주얼리 관련 템플릿 텍스트들
        jewelry_texts = [
            "이 다이아몬드는 1.2캐럿 D컬러 VVS1 등급으로 GIA 감정서가 함께 제공됩니다. 라운드 브릴리언트 컷으로 가공되어 뛰어난 광채를 보여줍니다.",
            "루비의 품질은 색상, 투명도, 컷, 캐럿으로 평가됩니다. 이 미얀마산 루비는 비둘기피 색상을 보이며 2.5캐럿의 무게를 가집니다.",
            "사파이어는 코런덤 계열의 보석으로, 블루 사파이어가 가장 유명합니다. 스리랑카산 사파이어는 로얄 블루 색상으로 유명합니다.",
            "에메랄드는 베릴 계열의 보석으로 콜롬비아산이 최고급으로 인정받습니다. 자르딘이라 불리는 내포물이 천연 에메랄드의 특징입니다.",
            "프린세스 컷 다이아몬드는 정사각형 모양의 브릴리언트 컷으로, 라운드 컷 다음으로 인기가 높습니다.",
            "플래티넘은 백금족 금속으로 변색되지 않아 고급 주얼리 제작에 사용됩니다. PT950, PT900 등의 순도로 구분됩니다.",
            "도매가격과 소매가격의 차이는 일반적으로 50-100% 정도입니다. 브랜드 프리미엄에 따라 더 큰 차이를 보일 수 있습니다.",
            "AGS 0등급은 최고 등급의 컷을 의미하며, 트리플 제로(000)라고도 불립니다.",
            "히트 트리트먼트는 보석의 색상과 투명도를 개선하는 가열 처리 방법입니다.",
            "인클루전은 보석 내부의 내포물을 의미하며, 천연 보석임을 증명하는 중요한 특징입니다."
        ]
        
        content = ""
        current_bytes = 0
        
        while current_bytes < target_bytes:
            # 랜덤하게 텍스트 선택 및 변형
            base_text = np.random.choice(jewelry_texts)
            
            # 텍스트 변형 (문장 반복, 단어 추가 등)
            variations = [
                base_text,
                base_text + " 전문가의 감정을 통해 확인된 정보입니다.",
                f"세부 사항: {base_text}",
                f"추가 정보로는 {base_text.lower()}",
                base_text + " 이는 업계 표준에 따른 평가입니다."
            ]
            
            selected_text = np.random.choice(variations)
            content += selected_text + "\n\n"
            current_bytes = len(content.encode('utf-8'))
        
        return content[:target_bytes]  # 정확한 크기로 자르기
    
    def cleanup_test_files(self, file_paths: List[str]):
        """테스트 파일 정리"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logging.warning(f"파일 삭제 실패: {file_path}, {e}")

class IntegratedBenchmarkSuite:
    """통합 벤치마크 스위트"""
    
    def __init__(self, benchmark_level: BenchmarkLevel = BenchmarkLevel.STANDARD):
        self.benchmark_level = benchmark_level
        self.test_data_generator = TestDataGenerator(Path("benchmark_temp"))
        self.resource_monitor = SystemResourceMonitor(sampling_interval=0.5)
        self.logger = logging.getLogger(__name__)
        
        # 벤치마크 ID 생성
        self.benchmark_id = f"benchmark_{int(time.time())}"
        
        # 테스트 시나리오 정의
        self.test_scenarios = self._define_test_scenarios()
        
        # 결과 저장
        self.results: List[PerformanceMetrics] = []
        self.detailed_logs = {}
    
    def _define_test_scenarios(self) -> List[TestScenario]:
        """벤치마크 레벨에 따른 테스트 시나리오 정의"""
        
        if self.benchmark_level == BenchmarkLevel.LIGHT:
            return [
                TestScenario(
                    scenario_id="light_mixed",
                    name="가벼운 혼합 테스트",
                    description="소규모 다중 파일 처리",
                    file_count=3,
                    total_size_mb=50.0,
                    file_types=["audio", "document", "image"],
                    expected_duration=30.0
                )
            ]
        
        elif self.benchmark_level == BenchmarkLevel.STANDARD:
            return [
                TestScenario(
                    scenario_id="standard_audio",
                    name="표준 오디오 테스트", 
                    description="중간 규모 오디오 파일 처리",
                    file_count=5,
                    total_size_mb=150.0,
                    file_types=["audio"],
                    expected_duration=60.0
                ),
                TestScenario(
                    scenario_id="standard_mixed",
                    name="표준 혼합 테스트",
                    description="다양한 파일 타입 혼합 처리",
                    file_count=8,
                    total_size_mb=200.0,
                    file_types=["audio", "video", "document", "image"],
                    expected_duration=90.0
                )
            ]
        
        elif self.benchmark_level == BenchmarkLevel.HEAVY:
            return [
                TestScenario(
                    scenario_id="heavy_batch",
                    name="대용량 배치 테스트",
                    description="대용량 파일 배치 처리",
                    file_count=15,
                    total_size_mb=500.0,
                    file_types=["audio", "video"],
                    expected_duration=180.0
                ),
                TestScenario(
                    scenario_id="heavy_stress",
                    name="스트레스 테스트",
                    description="메모리 및 CPU 부하 테스트", 
                    file_count=20,
                    total_size_mb=800.0,
                    file_types=["audio", "video", "document"],
                    expected_duration=300.0,
                    stress_factors={"memory_pressure": True, "concurrent_load": True}
                )
            ]
        
        else:  # EXTREME
            return [
                TestScenario(
                    scenario_id="extreme_volume",
                    name="극한 볼륨 테스트",
                    description="최대 용량 파일 처리",
                    file_count=25,
                    total_size_mb=1500.0,
                    file_types=["audio", "video"],
                    expected_duration=600.0,
                    stress_factors={"memory_pressure": True, "disk_io_pressure": True}
                ),
                TestScenario(
                    scenario_id="extreme_reliability",
                    name="극한 안정성 테스트",
                    description="오류 주입 및 복구 테스트",
                    file_count=20,
                    total_size_mb=1000.0,
                    file_types=["audio", "video", "document", "image"],
                    expected_duration=400.0,
                    stress_factors={"inject_errors": True, "network_instability": True}
                )
            ]
    
    async def run_comprehensive_benchmark(self) -> BenchmarkResult:
        """종합 벤치마크 실행"""
        
        print(f"🚀 통합 벤치마크 시작 (레벨: {self.benchmark_level.value})")
        print("=" * 80)
        
        # 시스템 정보 수집
        system_info = self._collect_system_info()
        
        # 리소스 모니터링 시작
        self.resource_monitor.start_monitoring()
        
        overall_start_time = time.time()
        
        try:
            # 각 시나리오 실행
            for i, scenario in enumerate(self.test_scenarios, 1):
                print(f"\n📋 시나리오 {i}/{len(self.test_scenarios)}: {scenario.name}")
                print("-" * 60)
                
                metrics = await self._run_scenario_benchmark(scenario)
                self.results.append(metrics)
                
                # 중간 결과 출력
                self._print_scenario_results(metrics)
                
                # 시나리오 간 쿨다운 (메모리 정리)
                if i < len(self.test_scenarios):
                    print("🧹 시스템 정리 중...")
                    await asyncio.sleep(5)
                    import gc
                    gc.collect()
        
        finally:
            # 리소스 모니터링 중지
            self.resource_monitor.stop_monitoring()
        
        # 전체 실행 시간
        total_benchmark_time = time.time() - overall_start_time
        
        # 전체 점수 계산
        overall_score = self._calculate_overall_score()
        
        # 추천사항 생성
        recommendations = self._generate_recommendations()
        
        # 결과 생성
        benchmark_result = BenchmarkResult(
            benchmark_id=self.benchmark_id,
            timestamp=datetime.now(),
            system_info=system_info,
            test_scenarios=self.test_scenarios,
            performance_metrics=self.results,
            overall_score=overall_score,
            recommendations=recommendations,
            detailed_logs=self.detailed_logs
        )
        
        # 최종 리포트 출력
        self._print_final_report(benchmark_result, total_benchmark_time)
        
        return benchmark_result
    
    async def _run_scenario_benchmark(self, scenario: TestScenario) -> PerformanceMetrics:
        """단일 시나리오 벤치마크 실행"""
        
        scenario_start_time = time.time()
        
        # 테스트 파일 생성
        print(f"📁 테스트 파일 생성 중... ({scenario.file_count}개, {scenario.total_size_mb:.1f}MB)")
        test_files = self.test_data_generator.generate_test_files(scenario)
        
        try:
            # 통합 시스템 설정
            streaming_config = StreamingConfig(
                chunk_size_mb=25.0,
                max_memory_mb=512.0,
                compression_enabled=True,
                max_concurrent_chunks=3
            )
            
            recovery_config = RecoveryConfig(
                max_retry_attempts=3,
                auto_recovery_enabled=True,
                recovery_level=RecoveryLevel.FULL
            )
            
            # 컴포넌트 초기화
            streaming_processor = StreamingProcessor(streaming_config)
            recovery_manager = RecoveryManager(recovery_config, Path("benchmark_recovery"))
            validator = AdvancedCrossValidator(ValidationLevel.COMPREHENSIVE)
            
            await streaming_processor.start_monitoring()
            
            # 처리 실행
            processing_results = []
            error_count = 0
            critical_errors = 0
            
            print(f"⚡ 파일 처리 시작...")
            
            for i, file_path in enumerate(test_files):
                try:
                    print(f"   처리 중: {i+1}/{len(test_files)} ({os.path.basename(file_path)})")
                    
                    # 스트리밍 처리 (복구 기능 포함)
                    async def process_with_recovery():
                        return await streaming_processor.process_file_streaming(
                            file_path=file_path,
                            file_id=f"test_file_{i}",
                            processor_func=self._mock_jewelry_processor
                        )
                    
                    result = await recovery_manager.execute_with_recovery(
                        operation_id=f"process_file_{i}",
                        operation_func=process_with_recovery,
                        session_id=scenario.scenario_id
                    )
                    
                    processing_results.append(result)
                    
                except Exception as e:
                    error_count += 1
                    if "critical" in str(e).lower() or "memory" in str(e).lower():
                        critical_errors += 1
                    self.logger.error(f"파일 처리 실패: {e}")
            
            # 크로스 검증 실행
            print(f"🔍 크로스 검증 실행...")
            validation_items = [
                {
                    "id": f"item_{i}",
                    "content": result.get("result", {}).get("merged_content", ""),
                    "quality": result.get("result", {}).get("average_confidence", 0.8),
                    "reliability": 0.9 if result.get("success", False) else 0.5
                }
                for i, result in enumerate(processing_results)
                if result.get("success", False)
            ]
            
            validation_result = await validator.validate_cross_consistency(validation_items)
            
            # 성능 지표 계산
            metrics = self._calculate_performance_metrics(
                scenario, processing_results, validation_result, 
                error_count, critical_errors, scenario_start_time
            )
            
            return metrics
            
        finally:
            # 정리
            await streaming_processor.stop_monitoring()
            self.test_data_generator.cleanup_test_files(test_files)
    
    async def _mock_jewelry_processor(self, chunk_data: bytes, chunk) -> Dict[str, Any]:
        """모의 주얼리 처리 함수"""
        
        # 처리 시간 시뮬레이션 (파일 크기에 비례)
        processing_time = len(chunk_data) / (1024 * 1024) * 0.1  # MB당 0.1초
        await asyncio.sleep(min(processing_time, 2.0))  # 최대 2초
        
        # 주얼리 관련 키워드 생성 (랜덤)
        jewelry_keywords = [
            "다이아몬드", "루비", "사파이어", "에메랄드", "4C", "GIA", 
            "캐럿", "컬러", "클래리티", "컷", "브릴리언트", "프린세스"
        ]
        
        selected_keywords = np.random.choice(
            jewelry_keywords, 
            size=np.random.randint(2, 6), 
            replace=False
        ).tolist()
        
        # 신뢰도 계산 (파일 크기와 처리 품질에 기반)
        base_confidence = 0.85
        size_factor = min(1.0, len(chunk_data) / (10 * 1024 * 1024))  # 10MB 기준
        confidence = base_confidence + (size_factor * 0.1)
        
        return {
            "content": f"주얼리 관련 내용 분석 결과 (청크: {chunk.chunk_id}): {', '.join(selected_keywords)}",
            "confidence": confidence,
            "keywords": selected_keywords,
            "chunk_info": {
                "chunk_id": chunk.chunk_id,
                "size_mb": chunk.size_mb,
                "processing_time": processing_time
            }
        }
    
    def _calculate_performance_metrics(
        self, 
        scenario: TestScenario,
        processing_results: List[Dict],
        validation_result,
        error_count: int,
        critical_errors: int,
        start_time: float
    ) -> PerformanceMetrics:
        """성능 지표 계산"""
        
        total_time = time.time() - start_time
        successful_results = [r for r in processing_results if r.get("success", False)]
        
        # 처리 성능
        total_processing_time = sum(r.get("processing_time", 0) for r in successful_results)
        average_file_time = total_processing_time / max(len(successful_results), 1)
        throughput_mbps = scenario.total_size_mb / max(total_time, 0.1)
        
        # 메모리 성능
        resource_summary = self.resource_monitor.get_summary()
        peak_memory = resource_summary.get("memory", {}).get("peak_mb", 0)
        avg_memory = resource_summary.get("memory", {}).get("average_mb", 0)
        memory_efficiency = throughput_mbps / max(avg_memory, 1)
        
        # 검증 성능
        validation_accuracy = validation_result.overall_score if validation_result else 0.0
        anomaly_count = len(validation_result.anomalies) if validation_result else 0
        anomaly_detection_rate = anomaly_count / max(len(successful_results), 1)
        
        # 복구 성능 (간단화)
        error_recovery_rate = (scenario.file_count - error_count) / scenario.file_count
        
        # 안정성 지표
        success_rate = len(successful_results) / scenario.file_count
        
        # 사용자 경험 점수 (처리 시간 기반)
        expected_time = scenario.expected_duration
        time_ratio = total_time / expected_time
        responsiveness_score = max(0.0, min(1.0, 2.0 - time_ratio))
        
        return PerformanceMetrics(
            scenario_id=scenario.scenario_id,
            total_processing_time=total_processing_time,
            average_file_time=average_file_time,
            throughput_mbps=throughput_mbps,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            memory_efficiency=memory_efficiency,
            validation_accuracy=validation_accuracy,
            anomaly_detection_rate=anomaly_detection_rate,
            false_positive_rate=0.05,  # 가정값
            error_recovery_rate=error_recovery_rate,
            recovery_time=1.5,  # 가정값
            checkpoint_overhead=0.1,  # 가정값
            success_rate=success_rate,
            error_count=error_count,
            critical_errors=critical_errors,
            responsiveness_score=responsiveness_score,
            progress_accuracy=0.95  # 가정값
        )
    
    def _calculate_overall_score(self) -> float:
        """전체 점수 계산"""
        if not self.results:
            return 0.0
        
        # 주요 지표들의 가중 평균
        weights = {
            "throughput": 0.25,
            "memory_efficiency": 0.20,
            "validation_accuracy": 0.20,
            "success_rate": 0.15,
            "responsiveness": 0.10,
            "error_recovery": 0.10
        }
        
        scores = []
        for metrics in self.results:
            # 각 지표를 0-1 범위로 정규화
            normalized_throughput = min(1.0, metrics.throughput_mbps / 10.0)  # 10MB/s를 최대로
            normalized_memory_eff = min(1.0, metrics.memory_efficiency / 0.1)  # 0.1을 최대로
            
            scenario_score = (
                weights["throughput"] * normalized_throughput +
                weights["memory_efficiency"] * normalized_memory_eff +
                weights["validation_accuracy"] * metrics.validation_accuracy +
                weights["success_rate"] * metrics.success_rate +
                weights["responsiveness"] * metrics.responsiveness_score +
                weights["error_recovery"] * metrics.error_recovery_rate
            )
            
            scores.append(scenario_score)
        
        return np.mean(scores)
    
    def _generate_recommendations(self) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        if not self.results:
            return ["벤치마크 결과가 없습니다."]
        
        # 성능 분석
        avg_throughput = np.mean([m.throughput_mbps for m in self.results])
        avg_memory = np.mean([m.peak_memory_mb for m in self.results])
        avg_success_rate = np.mean([m.success_rate for m in self.results])
        
        # 처리량 기반 추천
        if avg_throughput < 2.0:
            recommendations.append("처리 속도가 낮습니다. 청크 크기 늘리기 또는 동시 처리 수 증가를 검토하세요.")
        elif avg_throughput > 8.0:
            recommendations.append("우수한 처리 속도입니다. 현재 설정을 유지하세요.")
        
        # 메모리 기반 추천
        if avg_memory > 1000:
            recommendations.append("메모리 사용량이 높습니다. 압축 기능 활성화 또는 청크 크기 줄이기를 검토하세요.")
        
        # 성공률 기반 추천
        if avg_success_rate < 0.9:
            recommendations.append("처리 성공률이 낮습니다. 에러 복구 시스템 강화가 필요합니다.")
        elif avg_success_rate > 0.95:
            recommendations.append("높은 성공률을 보이고 있습니다. 안정성이 우수합니다.")
        
        # 검증 정확도 기반 추천
        avg_validation = np.mean([m.validation_accuracy for m in self.results])
        if avg_validation < 0.8:
            recommendations.append("검증 정확도가 낮습니다. 크로스 검증 알고리즘 개선이 필요합니다.")
        
        if not recommendations:
            recommendations.append("전반적으로 양호한 성능을 보이고 있습니다.")
        
        return recommendations
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.platform(),
            "benchmark_level": self.benchmark_level.value,
            "timestamp": datetime.now().isoformat()
        }
    
    def _print_scenario_results(self, metrics: PerformanceMetrics):
        """시나리오 결과 출력"""
        print(f"✅ 처리 완료!")
        print(f"   📊 처리량: {metrics.throughput_mbps:.2f} MB/s")
        print(f"   🧠 피크 메모리: {metrics.peak_memory_mb:.1f} MB")
        print(f"   ✅ 성공률: {metrics.success_rate:.1%}")
        print(f"   🔍 검증 정확도: {metrics.validation_accuracy:.1%}")
        print(f"   ⚡ 응답성: {metrics.responsiveness_score:.1%}")
    
    def _print_final_report(self, result: BenchmarkResult, total_time: float):
        """최종 리포트 출력"""
        print("\n" + "=" * 80)
        print("🎯 통합 벤치마크 최종 결과")
        print("=" * 80)
        
        print(f"🆔 벤치마크 ID: {result.benchmark_id}")
        print(f"⏱️ 총 실행 시간: {total_time:.1f}초")
        print(f"🎯 전체 점수: {result.overall_score:.3f}/1.000")
        
        # 시스템 정보
        print(f"\n💻 시스템 정보:")
        print(f"   CPU 코어: {result.system_info['cpu_count']}개")
        print(f"   메모리: {result.system_info['memory_total_gb']:.1f}GB")
        print(f"   벤치마크 레벨: {result.system_info['benchmark_level']}")
        
        # 시나리오별 요약
        print(f"\n📋 시나리오별 성과:")
        for i, metrics in enumerate(result.performance_metrics):
            scenario = result.test_scenarios[i]
            print(f"   {i+1}. {scenario.name}")
            print(f"      처리량: {metrics.throughput_mbps:.2f} MB/s")
            print(f"      성공률: {metrics.success_rate:.1%}")
            print(f"      메모리: {metrics.peak_memory_mb:.1f}MB")
        
        # 추천사항
        print(f"\n💡 추천사항:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)
        print("✅ 통합 벤치마크 완료!")

# 사용 예시
async def demo_integrated_benchmark():
    """통합 벤치마크 데모"""
    
    # 다양한 레벨로 테스트
    benchmark_levels = [BenchmarkLevel.LIGHT, BenchmarkLevel.STANDARD]
    
    for level in benchmark_levels:
        print(f"\n🚀 {level.value.upper()} 레벨 벤치마크 시작")
        
        benchmark_suite = IntegratedBenchmarkSuite(level)
        result = await benchmark_suite.run_comprehensive_benchmark()
        
        # 결과 저장 (옵션)
        result_file = f"benchmark_result_{level.value}_{int(time.time())}.json"
        with open(result_file, 'w') as f:
            # dataclass를 JSON으로 변환 (간단화)
            json.dump({
                "benchmark_id": result.benchmark_id,
                "overall_score": result.overall_score,
                "recommendations": result.recommendations
            }, f, indent=2)
        
        print(f"📄 결과 저장: {result_file}")

if __name__ == "__main__":
    import sys
    import platform
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 데모 실행
    asyncio.run(demo_integrated_benchmark())
