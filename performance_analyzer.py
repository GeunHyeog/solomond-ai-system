#!/usr/bin/env python3
"""
Performance Analyzer - SOLOMOND AI 시스템 병목지점 자동 분석
"""

import time
import psutil
import requests
import os
import json
import threading
from datetime import datetime
import subprocess

class PerformanceAnalyzer:
    def __init__(self):
        self.start_time = None
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'network_speed': [],
            'disk_io': [],
            'ollama_response_times': [],
            'api_response_times': [],
            'file_processing_times': {},
            'bottlenecks': []
        }
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """실시간 시스템 모니터링 시작"""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        print("🔍 성능 모니터링 시작됨")
        
    def stop_monitoring(self):
        """모니터링 중지 및 분석 결과 출력"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self._analyze_bottlenecks()
        return self._generate_report()
        
    def _monitor_system(self):
        """시스템 리소스 실시간 모니터링"""
        while self.monitoring:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].append({
                    'time': time.time() - self.start_time,
                    'value': cpu_percent
                })
                
                # 메모리 사용률
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append({
                    'time': time.time() - self.start_time,
                    'value': memory.percent,
                    'available_gb': memory.available / (1024**3)
                })
                
                # 디스크 I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io'].append({
                        'time': time.time() - self.start_time,
                        'read_mb': disk_io.read_bytes / (1024**2),
                        'write_mb': disk_io.write_bytes / (1024**2)
                    })
                
                # 네트워크 I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.metrics['network_speed'].append({
                        'time': time.time() - self.start_time,
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv
                    })
                    
                time.sleep(1)  # 1초마다 측정
                
            except Exception as e:
                print(f"⚠️ 모니터링 오류: {e}")
                
    def test_ollama_performance(self):
        """Ollama AI 성능 테스트"""
        print("🤖 Ollama 성능 테스트 중...")
        
        test_prompts = [
            "Hello, test response",
            "분석해주세요: 테스트 데이터입니다.",
            "다음 내용을 요약해주세요: " + "테스트 " * 100  # 긴 텍스트
        ]
        
        for i, prompt in enumerate(test_prompts):
            start_time = time.time()
            try:
                response = requests.post('http://localhost:8000/api/analyze', 
                    json={
                        'model': 'qwen2.5:7b',
                        'context': 'Performance Test',
                        'image_texts': [prompt],
                        'audio_texts': []
                    },
                    timeout=60
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                self.metrics['ollama_response_times'].append({
                    'test_id': i + 1,
                    'prompt_length': len(prompt),
                    'response_time': response_time,
                    'success': response.status_code == 200
                })
                
                print(f"  테스트 {i+1}: {response_time:.2f}초")
                
            except Exception as e:
                print(f"  테스트 {i+1} 실패: {e}")
                self.metrics['ollama_response_times'].append({
                    'test_id': i + 1,
                    'prompt_length': len(prompt),
                    'response_time': -1,
                    'success': False,
                    'error': str(e)
                })
                
    def test_api_performance(self):
        """API 서버 성능 테스트"""
        print("🌐 API 서버 성능 테스트 중...")
        
        endpoints = [
            {'url': 'http://localhost:8000/api/health', 'name': 'Health Check'},
            {'url': 'http://localhost:8000/', 'name': 'Main Page'},
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            try:
                response = requests.get(endpoint['url'], timeout=10)
                end_time = time.time()
                response_time = end_time - start_time
                
                self.metrics['api_response_times'].append({
                    'endpoint': endpoint['name'],
                    'response_time': response_time,
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                })
                
                print(f"  {endpoint['name']}: {response_time:.3f}초 (HTTP {response.status_code})")
                
            except Exception as e:
                print(f"  {endpoint['name']} 실패: {e}")
                self.metrics['api_response_times'].append({
                    'endpoint': endpoint['name'],
                    'response_time': -1,
                    'success': False,
                    'error': str(e)
                })
    
    def test_file_processing(self):
        """파일 처리 성능 테스트"""
        print("📁 파일 처리 성능 테스트 중...")
        
        # 테스트용 더미 파일 생성
        test_files = {
            'small_text.txt': "테스트 텍스트 " * 100,
            'medium_text.txt': "분석용 데이터 " * 1000,
            'large_text.txt': "대용량 텍스트 분석 데이터 " * 10000
        }
        
        for filename, content in test_files.items():
            try:
                # 파일 생성
                start_time = time.time()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                # 파일 읽기 테스트
                with open(filename, 'r', encoding='utf-8') as f:
                    data = f.read()
                    
                end_time = time.time()
                processing_time = end_time - start_time
                
                self.metrics['file_processing_times'][filename] = {
                    'size_bytes': len(content.encode('utf-8')),
                    'processing_time': processing_time,
                    'speed_mb_per_sec': (len(content.encode('utf-8')) / (1024*1024)) / processing_time if processing_time > 0 else 0
                }
                
                print(f"  {filename}: {processing_time:.4f}초 ({len(content.encode('utf-8'))/1024:.1f}KB)")
                
                # 테스트 파일 정리
                os.remove(filename)
                
            except Exception as e:
                print(f"  {filename} 테스트 실패: {e}")
    
    def test_system_resources(self):
        """시스템 리소스 상태 확인"""
        print("💻 시스템 리소스 분석 중...")
        
        # CPU 정보
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        
        # 디스크 정보
        disk = psutil.disk_usage('C:')
        
        # GPU 정보 (NVIDIA-SMI 있을 경우)
        gpu_info = self._get_gpu_info()
        
        system_info = {
            'cpu': {
                'cores': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else 'Unknown',
                'usage_percent': cpu_percent
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': (disk.used / disk.total) * 100
            },
            'gpu': gpu_info
        }
        
        self.metrics['system_info'] = system_info
        
        print(f"  CPU: {cpu_count}코어, {cpu_percent:.1f}% 사용중")
        print(f"  메모리: {memory.available/(1024**3):.1f}GB 사용 가능 ({memory.percent:.1f}% 사용중)")
        print(f"  디스크: {disk.free/(1024**3):.1f}GB 여유공간")
        if gpu_info:
            print(f"  GPU: {gpu_info}")
            
    def _get_gpu_info(self):
        """GPU 정보 가져오기"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "No NVIDIA GPU detected"
    
    def _analyze_bottlenecks(self):
        """병목지점 자동 분석"""
        bottlenecks = []
        
        # CPU 병목 분석
        if self.metrics['cpu_usage']:
            avg_cpu = sum(m['value'] for m in self.metrics['cpu_usage']) / len(self.metrics['cpu_usage'])
            if avg_cpu > 80:
                bottlenecks.append({
                    'type': 'CPU',
                    'severity': 'HIGH',
                    'description': f'높은 CPU 사용률 ({avg_cpu:.1f}%)',
                    'recommendation': 'CPU 집약적 작업 최적화 또는 병렬처리 개선 필요'
                })
        
        # 메모리 병목 분석
        if self.metrics['memory_usage']:
            avg_memory = sum(m['value'] for m in self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
            min_available = min(m['available_gb'] for m in self.metrics['memory_usage'])
            
            if avg_memory > 85:
                bottlenecks.append({
                    'type': 'MEMORY',
                    'severity': 'HIGH',
                    'description': f'높은 메모리 사용률 ({avg_memory:.1f}%)',
                    'recommendation': '메모리 사용 최적화 또는 RAM 증설 권장'
                })
            elif min_available < 2:
                bottlenecks.append({
                    'type': 'MEMORY',
                    'severity': 'MEDIUM',
                    'description': f'낮은 사용 가능 메모리 ({min_available:.1f}GB)',
                    'recommendation': '대용량 파일 처리시 메모리 관리 개선 필요'
                })
        
        # Ollama AI 응답 시간 분석
        if self.metrics['ollama_response_times']:
            successful_tests = [t for t in self.metrics['ollama_response_times'] if t['success']]
            if successful_tests:
                avg_response_time = sum(t['response_time'] for t in successful_tests) / len(successful_tests)
                
                if avg_response_time > 30:
                    bottlenecks.append({
                        'type': 'AI_PROCESSING',
                        'severity': 'HIGH',
                        'description': f'느린 AI 응답 시간 ({avg_response_time:.1f}초)',
                        'recommendation': 'Ollama 모델 최적화 또는 GPU 가속 활성화 권장'
                    })
                elif avg_response_time > 10:
                    bottlenecks.append({
                        'type': 'AI_PROCESSING',
                        'severity': 'MEDIUM',
                        'description': f'보통 AI 응답 시간 ({avg_response_time:.1f}초)',
                        'recommendation': '더 빠른 모델 사용 또는 프롬프트 최적화 고려'
                    })
            
            failed_tests = [t for t in self.metrics['ollama_response_times'] if not t['success']]
            if failed_tests:
                bottlenecks.append({
                    'type': 'AI_CONNECTION',
                    'severity': 'CRITICAL',
                    'description': f'AI 연결 실패 ({len(failed_tests)}개 테스트)',
                    'recommendation': 'Ollama 서버 상태 확인 및 재시작 필요'
                })
        
        # API 응답 시간 분석
        if self.metrics['api_response_times']:
            slow_apis = [api for api in self.metrics['api_response_times'] 
                        if api['success'] and api['response_time'] > 2]
            if slow_apis:
                bottlenecks.append({
                    'type': 'API_RESPONSE',
                    'severity': 'MEDIUM',
                    'description': f'느린 API 응답 ({len(slow_apis)}개 엔드포인트)',
                    'recommendation': 'API 서버 최적화 또는 캐싱 구현 권장'
                })
        
        self.metrics['bottlenecks'] = bottlenecks
    
    def _generate_report(self):
        """성능 분석 보고서 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_test_time': time.time() - self.start_time if self.start_time else 0,
            'summary': {
                'total_bottlenecks': len(self.metrics['bottlenecks']),
                'critical_issues': len([b for b in self.metrics['bottlenecks'] if b['severity'] == 'CRITICAL']),
                'high_priority': len([b for b in self.metrics['bottlenecks'] if b['severity'] == 'HIGH']),
                'medium_priority': len([b for b in self.metrics['bottlenecks'] if b['severity'] == 'MEDIUM'])
            },
            'bottlenecks': self.metrics['bottlenecks'],
            'metrics': self.metrics,
            'recommendations': self._get_optimization_recommendations()
        }
        
        return report
    
    def _get_optimization_recommendations(self):
        """최적화 권장사항 생성"""
        recommendations = []
        
        # 병목지점별 맞춤 권장사항
        bottleneck_types = [b['type'] for b in self.metrics['bottlenecks']]
        
        if 'CPU' in bottleneck_types:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'System Optimization',
                'action': 'CPU 사용량 최적화',
                'details': [
                    '병렬 처리를 위한 ThreadPoolExecutor 스레드 수 조정',
                    'CPU 집약적 작업을 백그라운드로 이동',
                    'AI 모델 추론 시 배치 크기 최적화'
                ]
            })
        
        if 'MEMORY' in bottleneck_types:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Memory Management',
                'action': '메모리 사용량 최적화',
                'details': [
                    '큰 파일 스트리밍 처리 구현',
                    '사용하지 않는 데이터 즉시 해제',
                    'garbage collection 최적화'
                ]
            })
        
        if 'AI_PROCESSING' in bottleneck_types:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'AI Performance',
                'action': 'AI 처리 속도 개선',
                'details': [
                    'GPU 가속 활성화 (CUDA 설정)',
                    '더 빠른 모델로 변경 (예: gemma2:2b)',
                    'AI 응답 캐싱 시스템 구축'
                ]
            })
        
        # 일반적인 최적화 권장사항
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'General Performance',
            'action': '전반적 성능 향상',
            'details': [
                '실시간 진행률 표시로 사용자 경험 개선',
                '백그라운드 작업 우선순위 조정',
                '네트워크 요청 타임아웃 최적화',
                '임시 파일 자동 정리 시스템'
            ]
        })
        
        return recommendations

def run_performance_analysis():
    """전체 성능 분석 실행"""
    analyzer = PerformanceAnalyzer()
    
    print("🚀 SOLOMOND AI 시스템 성능 분석 시작")
    print("=" * 50)
    
    # 모니터링 시작
    analyzer.start_monitoring()
    
    try:
        # 각종 성능 테스트 수행
        analyzer.test_system_resources()
        print()
        
        analyzer.test_api_performance() 
        print()
        
        analyzer.test_ollama_performance()
        print()
        
        analyzer.test_file_processing()
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 분석 중 오류: {e}")
    finally:
        # 모니터링 중지 및 분석 결과 생성
        print("📊 분석 결과 생성 중...")
        report = analyzer.stop_monitoring()
        
        # 보고서 저장
        with open('performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 결과 출력
        print_analysis_results(report)
        
        return report

def print_analysis_results(report):
    """분석 결과를 보기 좋게 출력"""
    print("\n" + "=" * 60)
    print("📊 성능 분석 결과")
    print("=" * 60)
    
    # 요약 정보
    summary = report['summary']
    print(f"🔍 총 테스트 시간: {report['total_test_time']:.1f}초")
    print(f"⚠️ 발견된 병목지점: {summary['total_bottlenecks']}개")
    
    if summary['critical_issues'] > 0:
        print(f"🚨 긴급 문제: {summary['critical_issues']}개")
    if summary['high_priority'] > 0:
        print(f"⚡ 고우선순위: {summary['high_priority']}개") 
    if summary['medium_priority'] > 0:
        print(f"📋 중우선순위: {summary['medium_priority']}개")
    
    # 병목지점 상세 정보
    if report['bottlenecks']:
        print("\n🎯 발견된 병목지점:")
        for i, bottleneck in enumerate(report['bottlenecks'], 1):
            severity_emoji = {'CRITICAL': '🚨', 'HIGH': '⚡', 'MEDIUM': '📋', 'LOW': '💡'}
            emoji = severity_emoji.get(bottleneck['severity'], '📋')
            
            print(f"\n{i}. {emoji} {bottleneck['type']}")
            print(f"   문제: {bottleneck['description']}")
            print(f"   해결방안: {bottleneck['recommendation']}")
    
    # 최적화 권장사항
    print("\n💡 최적화 권장사항:")
    for i, rec in enumerate(report['recommendations'], 1):
        priority_emoji = {'HIGH': '🔥', 'MEDIUM': '⚡', 'LOW': '💡'}
        emoji = priority_emoji.get(rec['priority'], '💡')
        
        print(f"\n{i}. {emoji} {rec['action']} ({rec['category']})")
        for detail in rec['details']:
            print(f"   • {detail}")
    
    print(f"\n💾 상세 보고서가 'performance_report.json'에 저장되었습니다.")
    print("=" * 60)

if __name__ == '__main__':
    run_performance_analysis()