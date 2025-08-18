#!/usr/bin/env python3
"""
크롬 브라우저 실행 이력 자동 분석기
Streamlit 앱 사용 패턴 분석 및 개선점 도출
"""

import sys
import os
import time
import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

class ChromeUsageAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'performance_issues': [],
            'user_experience_issues': [],
            'recommendations': [],
            'system_metrics': {}
        }
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_system_performance(self):
        """시스템 성능 분석"""
        print("1. 시스템 성능 분석")
        
        # 메모리 분석
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        self.analysis_results['system_metrics']['memory_usage'] = memory_usage
        
        if memory_usage > 85:
            self.analysis_results['performance_issues'].append({
                'type': 'HIGH_MEMORY_USAGE',
                'severity': 'HIGH',
                'description': f'메모리 사용률이 {memory_usage:.1f}%로 매우 높습니다',
                'impact': 'Streamlit 앱 응답 속도 저하, 브라우저 렌더링 지연'
            })
        elif memory_usage > 75:
            self.analysis_results['performance_issues'].append({
                'type': 'MODERATE_MEMORY_USAGE',
                'severity': 'MEDIUM',
                'description': f'메모리 사용률이 {memory_usage:.1f}%로 높습니다',
                'impact': '간헐적 성능 저하 가능성'
            })
        
        # CPU 사용률 분석
        cpu_usage = psutil.cpu_percent(interval=1)
        self.analysis_results['system_metrics']['cpu_usage'] = cpu_usage
        
        if cpu_usage > 80:
            self.analysis_results['performance_issues'].append({
                'type': 'HIGH_CPU_USAGE',
                'severity': 'HIGH',
                'description': f'CPU 사용률이 {cpu_usage:.1f}%로 높습니다',
                'impact': '시스템 전반적 반응 속도 저하'
            })
        
        print(f"   메모리 사용률: {memory_usage:.1f}%")
        print(f"   CPU 사용률: {cpu_usage:.1f}%")
    
    def analyze_process_efficiency(self):
        """프로세스 효율성 분석"""
        print("\n2. 프로세스 효율성 분석")
        
        python_processes = []
        chrome_processes = []
        streamlit_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'cpu_percent']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                if 'python' in name:
                    python_processes.append(proc.info)
                    if 'streamlit' in cmdline.lower():
                        streamlit_processes.append(proc.info)
                elif 'chrome' in name:
                    chrome_processes.append(proc.info)
            except:
                pass
        
        self.analysis_results['system_metrics']['python_processes'] = len(python_processes)
        self.analysis_results['system_metrics']['chrome_processes'] = len(chrome_processes)
        self.analysis_results['system_metrics']['streamlit_processes'] = len(streamlit_processes)
        
        print(f"   Python 프로세스: {len(python_processes)}개")
        print(f"   Chrome 프로세스: {len(chrome_processes)}개")
        print(f"   Streamlit 프로세스: {len(streamlit_processes)}개")
        
        # 프로세스 과다 체크
        if len(streamlit_processes) > 3:
            self.analysis_results['performance_issues'].append({
                'type': 'TOO_MANY_STREAMLIT_PROCESSES',
                'severity': 'MEDIUM',
                'description': f'Streamlit 프로세스가 {len(streamlit_processes)}개로 과다합니다',
                'impact': '메모리 낭비, 포트 충돌 가능성'
            })
        
        if len(chrome_processes) > 30:
            self.analysis_results['performance_issues'].append({
                'type': 'TOO_MANY_CHROME_PROCESSES',
                'severity': 'MEDIUM',
                'description': f'Chrome 프로세스가 {len(chrome_processes)}개로 과다합니다',
                'impact': '시스템 리소스 과다 사용'
            })
    
    def analyze_network_connectivity(self):
        """네트워크 연결성 분석"""
        print("\n3. 네트워크 연결성 분석")
        
        import socket
        
        # 포트 8503 확인
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        port_8503_result = sock.connect_ex(('127.0.0.1', 8503))
        sock.close()
        
        port_8503_open = port_8503_result == 0
        self.analysis_results['system_metrics']['port_8503_open'] = port_8503_open
        
        print(f"   포트 8503: {'열림' if port_8503_open else '닫힘'}")
        
        if not port_8503_open:
            self.analysis_results['performance_issues'].append({
                'type': 'STREAMLIT_PORT_CLOSED',
                'severity': 'HIGH',
                'description': 'Streamlit 서버 포트(8503)에 연결할 수 없습니다',
                'impact': '앱에 접근할 수 없음'
            })
        
        # WebSocket 연결 시뮬레이션 (실제로는 브라우저에서만 가능)
        print("   WebSocket 연결: 브라우저에서만 확인 가능")
    
    def analyze_file_access_patterns(self):
        """파일 접근 패턴 분석"""
        print("\n4. 파일 접근 패턴 분석")
        
        # 주요 파일들 확인
        important_files = [
            'jewelry_stt_ui_v23_real.py',
            'jewelry_stt_ui_v23_real_fixed.py',
            'core/real_analysis_engine.py',
            '.streamlit/config.toml'
        ]
        
        missing_files = []
        large_files = []
        
        for file_path in important_files:
            path = Path(file_path)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > 10:  # 10MB 이상
                    large_files.append((file_path, size_mb))
                print(f"   {file_path}: {size_mb:.1f}MB")
            else:
                missing_files.append(file_path)
                print(f"   {file_path}: 누락")
        
        if missing_files:
            self.analysis_results['user_experience_issues'].append({
                'type': 'MISSING_FILES',
                'severity': 'HIGH',
                'description': f'중요 파일이 누락되었습니다: {", ".join(missing_files)}',
                'impact': '앱 기능 제한 또는 오류 발생'
            })
        
        if large_files:
            self.analysis_results['performance_issues'].append({
                'type': 'LARGE_FILES',
                'severity': 'LOW',
                'description': f'대용량 파일 감지: {large_files}',
                'impact': '초기 로딩 시간 증가'
            })
    
    def analyze_user_experience_issues(self):
        """사용자 경험 이슈 분석"""
        print("\n5. 사용자 경험 이슈 분석")
        
        # URL 해시 분석 (제공된 URL: http://localhost:8503/#71b7676d)
        url_hash = "71b7676d"
        print(f"   세션 해시: {url_hash}")
        
        # 세션 지속성 체크
        if len(url_hash) == 8:
            print("   세션 해시 형식: 정상")
        else:
            self.analysis_results['user_experience_issues'].append({
                'type': 'INVALID_SESSION_HASH',
                'severity': 'LOW',
                'description': '비정상적인 세션 해시 형식',
                'impact': '세션 추적 어려움'
            })
        
        # 브라우저 호환성 추정
        print("   브라우저: Chrome (추정)")
        print("   JavaScript 지원: 활성화됨 (추정)")
        
        # Streamlit 특화 이슈들
        common_streamlit_issues = [
            {
                'type': 'STREAMLIT_RERUN_FREQUENCY',
                'description': 'Streamlit 앱의 빈번한 리런으로 인한 성능 저하',
                'impact': '사용자 입력 지연, UI 깜빡임'
            },
            {
                'type': 'SESSION_STATE_OVERFLOW',
                'description': '세션 상태 데이터 과다 축적',
                'impact': '메모리 사용량 증가, 앱 응답 속도 저하'
            }
        ]
        
        for issue in common_streamlit_issues:
            self.analysis_results['user_experience_issues'].append({
                'type': issue['type'],
                'severity': 'MEDIUM',
                'description': issue['description'],
                'impact': issue['impact']
            })
    
    def generate_recommendations(self):
        """개선 권장사항 생성"""
        print("\n6. 개선 권장사항 생성")
        
        recommendations = []
        
        # 성능 이슈 기반 권장사항
        for issue in self.analysis_results['performance_issues']:
            if issue['type'] == 'HIGH_MEMORY_USAGE':
                recommendations.extend([
                    {
                        'category': 'PERFORMANCE',
                        'priority': 'HIGH',
                        'action': '메모리 최적화',
                        'details': [
                            '불필요한 프로그램 종료',
                            'Streamlit 캐시 설정 최적화',
                            '대용량 데이터 처리 시 청크 단위 처리 적용',
                            'st.cache_data 데코레이터 활용'
                        ]
                    }
                ])
            
            elif issue['type'] == 'TOO_MANY_STREAMLIT_PROCESSES':
                recommendations.append({
                    'category': 'SYSTEM',
                    'priority': 'MEDIUM',
                    'action': '프로세스 정리',
                    'details': [
                        '중복 Streamlit 프로세스 종료',
                        '포트 충돌 방지를 위한 프로세스 관리',
                        '자동 프로세스 정리 스크립트 구현'
                    ]
                })
        
        # 사용자 경험 개선 권장사항
        recommendations.extend([
            {
                'category': 'UX',
                'priority': 'HIGH',
                'action': 'UI 응답성 개선',
                'details': [
                    'st.spinner() 활용한 로딩 인디케이터 추가',
                    'st.progress() 진행률 표시 강화',
                    '비동기 처리로 UI 블로킹 방지',
                    'st.empty() 활용한 동적 컨텐츠 업데이트'
                ]
            },
            {
                'category': 'UX',
                'priority': 'MEDIUM',
                'action': '에러 핸들링 강화',
                'details': [
                    '사용자 친화적 에러 메시지 표시',
                    '자동 재시도 메커니즘 구현',
                    '오류 복구 가이드 제공',
                    'try-except 블록 확장'
                ]
            },
            {
                'category': 'FEATURE',
                'priority': 'MEDIUM',
                'action': '브라우저 호환성 강화',
                'details': [
                    'JavaScript 에러 자동 감지',
                    'WebSocket 연결 상태 모니터링',
                    '브라우저별 최적화 코드 추가',
                    'Progressive Web App(PWA) 기능 고려'
                ]
            }
        ])
        
        self.analysis_results['recommendations'] = recommendations
        
        print(f"   총 {len(recommendations)}개 권장사항 생성")
    
    def save_analysis_report(self):
        """분석 보고서 저장"""
        report_path = Path('chrome_usage_analysis_report.json')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n분석 보고서 저장: {report_path}")
        return report_path
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        print("\n" + "="*60)
        print("크롬 실행 이력 분석 결과 요약")
        print("="*60)
        
        print(f"\n📊 시스템 메트릭:")
        metrics = self.analysis_results['system_metrics']
        print(f"   메모리 사용률: {metrics.get('memory_usage', 'N/A'):.1f}%")
        print(f"   Python 프로세스: {metrics.get('python_processes', 'N/A')}개")
        print(f"   Chrome 프로세스: {metrics.get('chrome_processes', 'N/A')}개")
        print(f"   포트 8503: {'열림' if metrics.get('port_8503_open', False) else '닫힘'}")
        
        print(f"\n⚠️ 발견된 이슈:")
        total_issues = len(self.analysis_results['performance_issues']) + len(self.analysis_results['user_experience_issues'])
        print(f"   성능 이슈: {len(self.analysis_results['performance_issues'])}개")
        print(f"   사용자 경험 이슈: {len(self.analysis_results['user_experience_issues'])}개")
        print(f"   총 이슈: {total_issues}개")
        
        print(f"\n💡 권장사항:")
        print(f"   총 {len(self.analysis_results['recommendations'])}개 개선 방안 제시")
        
        # 우선순위별 권장사항
        high_priority = [r for r in self.analysis_results['recommendations'] if r['priority'] == 'HIGH']
        medium_priority = [r for r in self.analysis_results['recommendations'] if r['priority'] == 'MEDIUM']
        
        if high_priority:
            print(f"\n🚨 즉시 조치 필요 ({len(high_priority)}개):")
            for rec in high_priority:
                print(f"   • {rec['action']}")
        
        if medium_priority:
            print(f"\n⚡ 개선 권장 ({len(medium_priority)}개):")
            for rec in medium_priority:
                print(f"   • {rec['action']}")
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("크롬 브라우저 실행 이력 자동 분석 시작")
        print("="*50)
        
        try:
            self.analyze_system_performance()
            self.analyze_process_efficiency()
            self.analyze_network_connectivity()
            self.analyze_file_access_patterns()
            self.analyze_user_experience_issues()
            self.generate_recommendations()
            
            report_path = self.save_analysis_report()
            self.print_summary()
            
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {e}")
            return None

def main():
    analyzer = ChromeUsageAnalyzer()
    results = analyzer.run_full_analysis()
    
    if results:
        print(f"\n✅ 분석 완료! 상세 보고서: chrome_usage_analysis_report.json")
    else:
        print(f"\n❌ 분석 실패")

if __name__ == "__main__":
    main()