                "memory_efficiency_score": self._calculate_memory_efficiency(performance),
                "speed_performance_score": self._calculate_speed_performance(performance),
                "stability_score": self._calculate_stability_score(performance),
                "recommendations": self._generate_recommendations(test_results)
            }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_memory_efficiency(self, performance: Dict[str, Any]) -> float:
        """메모리 효율성 점수 계산"""
        try:
            summary = performance.get("summary", {})
            peak_memory = summary.get("peak_memory_usage_mb", 0)
            avg_memory = summary.get("average_memory_usage_mb", 0)
            
            # 1GB 기준으로 점수 계산
            memory_limit = 1000  # MB
            
            if peak_memory <= memory_limit * 0.8:  # 80% 이하
                return 95.0
            elif peak_memory <= memory_limit:  # 100% 이하
                return 85.0
            else:  # 초과
                return max(50.0, 100 - (peak_memory - memory_limit) / memory_limit * 50)
                
        except Exception:
            return 50.0
    
    def _calculate_speed_performance(self, performance: Dict[str, Any]) -> float:
        """속도 성능 점수 계산"""
        try:
            summary = performance.get("summary", {})
            avg_speed = summary.get("average_processing_speed_mb_per_sec", 0)
            
            # 속도 기준: 1MB/s 이상이면 좋음
            if avg_speed >= 2.0:
                return 95.0
            elif avg_speed >= 1.0:
                return 85.0
            elif avg_speed >= 0.5:
                return 75.0
            else:
                return max(30.0, avg_speed * 60)  # 최소 30점
                
        except Exception:
            return 50.0
    
    def _calculate_stability_score(self, performance: Dict[str, Any]) -> float:
        """안정성 점수 계산"""
        try:
            analysis = performance.get("performance_analysis", {})
            
            stability_factors = {
                "memory_stability": 1.0 if analysis.get("memory_stability") == "안정" else 0.5,
                "cpu_efficiency": 1.0 if analysis.get("cpu_efficiency") == "효율적" else 0.7,
                "throughput_consistency": 1.0 if analysis.get("throughput_consistency") == "일정" else 0.6
            }
            
            total_score = sum(stability_factors.values()) / len(stability_factors) * 100
            return total_score
            
        except Exception:
            return 50.0
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        try:
            performance = test_results.get("performance_report", {})
            summary = performance.get("summary", {})
            analysis = performance.get("performance_analysis", {})
            
            # 메모리 권장사항
            peak_memory = summary.get("peak_memory_usage_mb", 0)
            if peak_memory > 800:
                recommendations.append("메모리 사용량이 높습니다. 청크 크기를 줄이거나 병렬 처리 수를 조정하세요.")
            
            # CPU 권장사항
            if analysis.get("cpu_efficiency") == "높음":
                recommendations.append("CPU 사용률이 높습니다. 프로세스 수를 줄이거나 처리 방식을 최적화하세요.")
            
            # 속도 권장사항
            avg_speed = summary.get("average_processing_speed_mb_per_sec", 0)
            if avg_speed < 1.0:
                recommendations.append("처리 속도가 느립니다. 하드웨어 업그레이드나 알고리즘 최적화를 고려하세요.")
            
            # 안정성 권장사항
            if analysis.get("memory_stability") == "불안정":
                recommendations.append("메모리 사용량이 불안정합니다. 메모리 관리 로직을 개선하세요.")
            
            # 기본 권장사항
            if not recommendations:
                recommendations.append("전반적으로 안정적인 성능을 보입니다. 현재 설정을 유지하세요.")
            
        except Exception as e:
            recommendations.append(f"권장사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _print_test_summary(self, test_results: Dict[str, Any]):
        """테스트 요약 출력"""
        print("\n" + "="*60)
        print("📊 테스트 결과 요약")
        print("="*60)
        
        try:
            summary = test_results.get("test_summary", {})
            processing = test_results.get("processing_results", {})
            performance = test_results.get("performance_report", {})
            
            # 전체 성공 여부
            success = summary.get("overall_success", False)
            print(f"🎯 전체 테스트 결과: {'✅ 성공' if success else '❌ 실패'}")
            
            # 처리 통계
            print(f"\n📈 처리 통계:")
            proc_summary = processing.get("processing_summary", {})
            print(f"   - 총 청크: {proc_summary.get('total_chunks', 0)}개")
            print(f"   - 성공: {proc_summary.get('successful_chunks', 0)}개")
            print(f"   - 실패: {proc_summary.get('failed_chunks', 0)}개")
            print(f"   - 처리 시간: {proc_summary.get('total_processing_time', 0):.1f}초")
            
            # 성능 점수
            print(f"\n⚡ 성능 점수:")
            print(f"   - 메모리 효율성: {summary.get('memory_efficiency_score', 0):.1f}점")
            print(f"   - 처리 속도: {summary.get('speed_performance_score', 0):.1f}점")
            print(f"   - 안정성: {summary.get('stability_score', 0):.1f}점")
            
            # 메모리 사용량
            perf_summary = performance.get("summary", {})
            print(f"\n💾 메모리 사용량:")
            print(f"   - 최대: {perf_summary.get('peak_memory_usage_mb', 0):.1f}MB")
            print(f"   - 평균: {perf_summary.get('average_memory_usage_mb', 0):.1f}MB")
            
            # 처리 속도
            print(f"\n🚄 처리 속도:")
            print(f"   - 평균: {perf_summary.get('average_processing_speed_mb_per_sec', 0):.1f}MB/s")
            print(f"   - 최대: {perf_summary.get('peak_processing_speed_mb_per_sec', 0):.1f}MB/s")
            
            # 권장사항
            recommendations = summary.get("recommendations", [])
            if recommendations:
                print(f"\n💡 권장사항:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            # 성능 등급
            analysis = performance.get("performance_analysis", {})
            grade = analysis.get("performance_grade", "N/A")
            print(f"\n🏆 종합 성능 등급: {grade}")
            
            # 파일 정보
            file_gen = test_results.get("file_generation", {})
            total_size = file_gen.get("video_size_mb", 0) + file_gen.get("total_image_size_mb", 0)
            print(f"\n📁 처리된 데이터:")
            print(f"   - 총 크기: {total_size:.1f}MB")
            print(f"   - 비디오: {file_gen.get('video_size_mb', 0):.1f}MB")
            print(f"   - 이미지: {file_gen.get('image_files_count', 0)}개 ({file_gen.get('total_image_size_mb', 0):.1f}MB)")
            
        except Exception as e:
            print(f"⚠️ 요약 출력 오류: {e}")
        
        print("="*60)

async def run_performance_stress_test():
    """성능 스트레스 테스트 실행"""
    print("🧪 고용량 파일 성능 스트레스 테스트")
    print("이 테스트는 1시간 비디오 + 30개 이미지를 동시 처리합니다.")
    print("메모리 사용량, CPU 효율성, 처리 속도를 실시간 모니터링합니다.")
    
    # 사용자 확인
    response = input("\n테스트를 시작하시겠습니까? (y/N): ").strip().lower()
    if response != 'y':
        print("테스트가 취소되었습니다.")
        return
    
    # 테스트 실행
    tester = LargeFileRealTest()
    
    try:
        results = await tester.run_full_test()
        
        print(f"\n🎉 테스트 완료!")
        print(f"📂 결과 파일: {tester.test_dir}")
        print(f"📊 상세 리포트: {tester.test_dir}/final_test_report.json")
        print(f"📈 성능 차트: {tester.test_dir}/performance_charts.png")
        
        # 간단한 성공/실패 판정
        success = results.get("test_summary", {}).get("overall_success", False)
        if success:
            print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        else:
            print("\n⚠️ 일부 테스트에서 문제가 발생했습니다. 상세 리포트를 확인하세요.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        print("상세 오류 정보:")
        import traceback
        traceback.print_exc()
        return None

def quick_system_check():
    """빠른 시스템 체크"""
    print("🔍 시스템 사전 체크...")
    
    # 메모리 체크
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"💾 메모리: {memory_gb:.1f}GB 총용량, {available_gb:.1f}GB 사용가능")
    
    if available_gb < 2.0:
        print("⚠️ 경고: 사용 가능한 메모리가 부족합니다. (최소 2GB 권장)")
        return False
    
    # 디스크 체크
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    
    print(f"💿 디스크: {disk_free_gb:.1f}GB 여유공간")
    
    if disk_free_gb < 5.0:
        print("⚠️ 경고: 디스크 여유공간이 부족합니다. (최소 5GB 권장)")
        return False
    
    # CPU 체크
    cpu_count = psutil.cpu_count()
    print(f"🖥️ CPU: {cpu_count}코어")
    
    # 필수 라이브러리 체크
    required_libs = ['cv2', 'librosa', 'whisper', 'pytesseract']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"⚠️ 누락된 라이브러리: {', '.join(missing_libs)}")
        print("pip install 명령으로 설치해주세요.")
        return False
    
    print("✅ 시스템 체크 완료 - 테스트 실행 가능")
    return True

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('large_file_test.log'),
            logging.StreamHandler()
        ]
    )
    
    print("🎯 주얼리 AI 플랫폼 v2.1 - 고용량 파일 실전 테스트")
    print("=" * 60)
    
    # 시스템 사전 체크
    if not quick_system_check():
        print("\n❌ 시스템 요구사항을 만족하지 않습니다.")
        sys.exit(1)
    
    # 테스트 실행
    try:
        asyncio.run(run_performance_stress_test())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        logging.error(f"테스트 실행 오류: {e}", exc_info=True)
