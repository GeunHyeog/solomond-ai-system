#!/usr/bin/env python3
"""
솔로몬드 AI 자동 에러 점검 및 개선 시스템
브라우저 실행 중 발생하는 에러들을 자동으로 감지하고 해결
"""

import sys
import os
import traceback
import logging
from pathlib import Path
import psutil
import time
from datetime import datetime
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class AutoErrorChecker:
    def __init__(self):
        self.setup_logging()
        self.error_count = 0
        self.fixed_count = 0
        self.issues_found = []
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('auto_error_check.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_system_resources(self):
        """시스템 리소스 상태 점검"""
        print("=== 시스템 리소스 점검 ===")
        
        # 메모리 확인
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        if memory_usage > 85:
            issue = {
                'type': 'HIGH_MEMORY_USAGE',
                'severity': 'WARNING',
                'description': f'메모리 사용률이 {memory_usage:.1f}%로 높습니다.',
                'solution': '불필요한 프로세스 종료 및 메모리 최적화 필요'
            }
            self.issues_found.append(issue)
            print(f"⚠️ 경고: {issue['description']}")
        else:
            print(f"✅ 메모리 사용률: {memory_usage:.1f}% (정상)")
        
        # 디스크 확인
        disk = psutil.disk_usage('C:')
        disk_usage = disk.used / disk.total * 100
        
        if disk_usage > 90:
            issue = {
                'type': 'HIGH_DISK_USAGE',
                'severity': 'WARNING',
                'description': f'디스크 사용률이 {disk_usage:.1f}%로 높습니다.',
                'solution': '임시 파일 정리 및 디스크 공간 확보 필요'
            }
            self.issues_found.append(issue)
            print(f"⚠️ 경고: {issue['description']}")
        else:
            print(f"✅ 디스크 사용률: {disk_usage:.1f}% (정상)")

    def check_streamlit_processes(self):
        """Streamlit 프로세스 상태 점검"""
        print("\n=== Streamlit 프로세스 점검 ===")
        
        streamlit_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent', 'status']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'streamlit' in cmdline.lower():
                        streamlit_processes.append({
                            'pid': proc.info['pid'],
                            'memory': proc.info['memory_percent'],
                            'status': proc.info['status'],
                            'cmd': cmdline
                        })
            except:
                pass
        
        if len(streamlit_processes) == 0:
            issue = {
                'type': 'NO_STREAMLIT_PROCESS',
                'severity': 'ERROR',
                'description': 'Streamlit 프로세스가 실행되지 않고 있습니다.',
                'solution': 'Streamlit 서버를 다시 시작해야 합니다.'
            }
            self.issues_found.append(issue)
            print(f"❌ 오류: {issue['description']}")
        elif len(streamlit_processes) > 5:
            issue = {
                'type': 'TOO_MANY_STREAMLIT_PROCESSES',
                'severity': 'WARNING',
                'description': f'Streamlit 프로세스가 {len(streamlit_processes)}개로 너무 많습니다.',
                'solution': '중복 프로세스 정리가 필요합니다.'
            }
            self.issues_found.append(issue)
            print(f"⚠️ 경고: {issue['description']}")
        else:
            print(f"✅ Streamlit 프로세스: {len(streamlit_processes)}개 (정상)")
            
        for proc in streamlit_processes:
            if proc['memory'] > 10:
                print(f"⚠️ 프로세스 PID {proc['pid']}: 메모리 {proc['memory']:.1f}% (높음)")
            else:
                print(f"✅ 프로세스 PID {proc['pid']}: 메모리 {proc['memory']:.1f}% (정상)")

    def check_port_status(self):
        """포트 상태 확인"""
        print("\n=== 포트 상태 점검 ===")
        
        import socket
        
        def check_port(host, port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        
        # 포트 8503 확인
        if check_port('127.0.0.1', 8503):
            print("✅ 포트 8503: 열림 (정상)")
        else:
            issue = {
                'type': 'PORT_8503_CLOSED',
                'severity': 'ERROR',
                'description': '포트 8503이 닫혀있습니다.',
                'solution': 'Streamlit 서버를 시작해야 합니다.'
            }
            self.issues_found.append(issue)
            print(f"❌ 오류: {issue['description']}")

    def check_dependencies(self):
        """의존성 라이브러리 확인"""
        print("\n=== 의존성 라이브러리 점검 ===")
        
        critical_dependencies = [
            'streamlit', 'numpy', 'pandas', 'plotly', 
            'whisper', 'easyocr', 'transformers',
            'opencv-python', 'librosa', 'psutil'
        ]
        
        missing_deps = []
        for dep in critical_dependencies:
            try:
                __import__(dep.replace('-', '_'))
                print(f"✅ {dep}: 설치됨")
            except ImportError:
                missing_deps.append(dep)
                print(f"❌ {dep}: 누락")
        
        if missing_deps:
            issue = {
                'type': 'MISSING_DEPENDENCIES',
                'severity': 'ERROR',
                'description': f'필수 라이브러리가 누락되었습니다: {", ".join(missing_deps)}',
                'solution': f'pip install {" ".join(missing_deps)} 실행 필요'
            }
            self.issues_found.append(issue)

    def check_file_permissions(self):
        """파일 권한 확인"""
        print("\n=== 파일 권한 점검 ===")
        
        important_files = [
            'jewelry_stt_ui_v23_real.py',
            'core/real_analysis_engine.py',
            'core/user_settings_manager.py'
        ]
        
        for file_path in important_files:
            full_path = project_root / file_path
            if full_path.exists():
                if os.access(full_path, os.R_OK):
                    print(f"✅ {file_path}: 읽기 권한 있음")
                else:
                    issue = {
                        'type': 'FILE_PERMISSION_ERROR',
                        'severity': 'ERROR',
                        'description': f'{file_path} 파일에 읽기 권한이 없습니다.',
                        'solution': '파일 권한 설정을 확인해야 합니다.'
                    }
                    self.issues_found.append(issue)
                    print(f"❌ {file_path}: 읽기 권한 없음")
            else:
                issue = {
                    'type': 'MISSING_FILE',
                    'severity': 'ERROR',
                    'description': f'{file_path} 파일이 없습니다.',
                    'solution': '파일이 삭제되었거나 경로가 잘못되었습니다.'
                }
                self.issues_found.append(issue)
                print(f"❌ {file_path}: 파일 없음")

    def check_browser_compatibility(self):
        """브라우저 호환성 확인"""
        print("\n=== 브라우저 호환성 점검 ===")
        
        # 브라우저 관련 JavaScript 에러 체크
        js_compatibility_issues = [
            {
                'check': 'WebSocket 지원',
                'description': 'Streamlit은 WebSocket을 사용합니다.',
                'solution': '최신 브라우저 버전을 사용하세요.'
            },
            {
                'check': 'JavaScript 활성화',
                'description': 'Streamlit UI는 JavaScript가 필요합니다.',
                'solution': '브라우저에서 JavaScript를 활성화하세요.'
            }
        ]
        
        for check in js_compatibility_issues:
            print(f"ℹ️ {check['check']}: {check['description']}")

    def apply_automatic_fixes(self):
        """자동 수정 적용"""
        print("\n=== 자동 수정 적용 ===")
        
        for issue in self.issues_found:
            if issue['type'] == 'HIGH_MEMORY_USAGE':
                print("🔧 메모리 최적화 적용 중...")
                self.optimize_memory()
                self.fixed_count += 1
                
            elif issue['type'] == 'TOO_MANY_STREAMLIT_PROCESSES':
                print("🔧 중복 프로세스 정리 중...")
                self.cleanup_duplicate_processes()
                self.fixed_count += 1
                
            elif issue['type'] == 'HIGH_DISK_USAGE':
                print("🔧 임시 파일 정리 중...")
                self.cleanup_temp_files()
                self.fixed_count += 1

    def optimize_memory(self):
        """메모리 최적화"""
        try:
            # Python 가비지 컬렉션 강제 실행
            import gc
            gc.collect()
            
            # 임시 파일 정리
            temp_dir = Path(tempfile.gettempdir())
            temp_files = list(temp_dir.glob("tmp*"))
            for temp_file in temp_files[:10]:  # 최대 10개만
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except:
                    pass
                    
            print("✅ 메모리 최적화 완료")
        except Exception as e:
            print(f"❌ 메모리 최적화 실패: {e}")

    def cleanup_duplicate_processes(self):
        """중복 프로세스 정리"""
        try:
            streamlit_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'streamlit' in cmdline.lower():
                            streamlit_pids.append(proc.info['pid'])
                except:
                    pass
            
            # 가장 오래된 것들 제외하고 나머지 종료 (첫 2개는 유지)
            if len(streamlit_pids) > 2:
                for pid in streamlit_pids[2:]:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        print(f"🔧 프로세스 PID {pid} 종료")
                    except:
                        pass
                        
            print("✅ 중복 프로세스 정리 완료")
        except Exception as e:
            print(f"❌ 프로세스 정리 실패: {e}")

    def cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            # 프로젝트 내 임시 파일들
            temp_patterns = ['*.tmp', '*.log.old', '__pycache__', '*.pyc']
            
            for pattern in temp_patterns:
                for temp_file in project_root.rglob(pattern):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir() and pattern == '__pycache__':
                            import shutil
                            shutil.rmtree(temp_file)
                    except:
                        pass
                        
            print("✅ 임시 파일 정리 완료")
        except Exception as e:
            print(f"❌ 임시 파일 정리 실패: {e}")

    def generate_error_report(self):
        """에러 리포트 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': len(self.issues_found),
            'fixed_issues': self.fixed_count,
            'issues': self.issues_found,
            'system_info': {
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('C:').used / psutil.disk_usage('C:').total * 100,
                'python_version': sys.version
            }
        }
        
        report_path = project_root / 'error_check_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 에러 리포트 생성: {report_path}")
        return report

    def run_full_check(self):
        """전체 점검 실행"""
        print("🔍 솔로몬드 AI 시스템 자동 에러 점검 시작")
        print("=" * 50)
        
        try:
            self.check_system_resources()
            self.check_streamlit_processes()
            self.check_port_status()
            self.check_dependencies()
            self.check_file_permissions()
            self.check_browser_compatibility()
            
            if self.issues_found:
                print(f"\n⚠️ 총 {len(self.issues_found)}개의 문제가 발견되었습니다.")
                self.apply_automatic_fixes()
            else:
                print("\n✅ 모든 시스템이 정상 상태입니다!")
            
            report = self.generate_error_report()
            
            print("\n" + "=" * 50)
            print("🎯 점검 결과 요약:")
            print(f"- 발견된 문제: {len(self.issues_found)}개")
            print(f"- 자동 수정: {self.fixed_count}개")
            print(f"- 메모리 사용률: {report['system_info']['memory_usage']:.1f}%")
            print(f"- 디스크 사용률: {report['system_info']['disk_usage']:.1f}%")
            
            if len(self.issues_found) > self.fixed_count:
                print(f"\n💡 수동 해결이 필요한 문제: {len(self.issues_found) - self.fixed_count}개")
                for issue in self.issues_found:
                    if issue['severity'] == 'ERROR':
                        print(f"❌ {issue['description']}")
                        print(f"   해결방법: {issue['solution']}")
            
        except Exception as e:
            print(f"❌ 점검 중 오류 발생: {e}")
            traceback.print_exc()

def main():
    checker = AutoErrorChecker()
    checker.run_full_check()

if __name__ == "__main__":
    main()