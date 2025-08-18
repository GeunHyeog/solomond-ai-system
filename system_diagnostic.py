#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 자동 진단 도구
사용자 테스트 완료 후 시스템 문제점 자동 검출
"""

import os
import sys
import json
import glob
import subprocess
import time
from datetime import datetime
from pathlib import Path

class SystemDiagnostic:
    """시스템 자동 진단"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def log_issue(self, category, severity, title, description, solution=""):
        """문제점 기록"""
        issue = {
            "category": category,
            "severity": severity,  # critical, high, medium, low
            "title": title,
            "description": description,
            "solution": solution,
            "timestamp": datetime.now().isoformat()
        }
        
        if severity in ["critical", "high"]:
            self.issues.append(issue)
        else:
            self.warnings.append(issue)
    
    def check_streamlit_logs(self):
        """Streamlit 로그 분석"""
        print("1. Streamlit 로그 분석 중...")
        
        try:
            # 최근 로그 파일들 확인
            log_patterns = [
                "*.log",
                "streamlit*.log", 
                "error*.log",
                "debug*.log"
            ]
            
            error_keywords = [
                "error", "exception", "traceback", "failed",
                "timeout", "memory", "unable", "cannot"
            ]
            
            log_files_found = False
            for pattern in log_patterns:
                log_files = glob.glob(pattern)
                if log_files:
                    log_files_found = True
                    for log_file in log_files[-3:]:  # 최근 3개만
                        try:
                            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read().lower()
                                for keyword in error_keywords:
                                    if keyword in content:
                                        self.log_issue(
                                            "logs", "medium",
                                            f"로그에서 {keyword} 감지",
                                            f"{log_file}에서 '{keyword}' 관련 문제 발견",
                                            f"{log_file} 파일을 직접 확인하여 상세 오류 분석 필요"
                                        )
                                        break
                        except Exception as e:
                            self.log_issue("logs", "low", "로그 파일 읽기 실패", str(e))
            
            if not log_files_found:
                self.log_issue("logs", "medium", "로그 파일 없음", 
                             "시스템 로그 파일이 발견되지 않음", 
                             "로깅 시스템 설정 확인 필요")
                
        except Exception as e:
            self.log_issue("logs", "high", "로그 분석 실패", f"로그 분석 중 오류: {e}")
    
    def check_process_status(self):
        """프로세스 상태 확인"""
        print("2. 프로세스 상태 확인 중...")
        
        try:
            # Streamlit 프로세스 확인
            result = subprocess.run([
                'wmic', 'process', 'where', 
                "name='python.exe' and CommandLine like '%streamlit%'", 
                'get', 'ProcessId,PageFileUsage,WorkingSetSize'
            ], capture_output=True, text=True, timeout=10)
            
            if "ProcessId" in result.stdout:
                lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                if len(lines) > 1:
                    data_line = lines[1] 
                    if data_line and not data_line.startswith("No Instance"):
                        parts = data_line.split()
                        if len(parts) >= 3:
                            memory_kb = float(parts[0]) if parts[0].isdigit() else 0
                            working_set = float(parts[1]) if parts[1].isdigit() else 0 
                            memory_mb = memory_kb / 1024
                            working_mb = working_set / 1024
                            
                            if memory_mb > 4000:  # 4GB 이상
                                self.log_issue("performance", "high",
                                             "높은 메모리 사용량",
                                             f"현재 메모리 사용량: {memory_mb:.1f}MB",
                                             "메모리 누수 가능성, 프로세스 재시작 권장")
                            elif memory_mb > 2000:  # 2GB 이상  
                                self.log_issue("performance", "medium",
                                             "메모리 사용량 경고",
                                             f"현재 메모리 사용량: {memory_mb:.1f}MB",
                                             "메모리 사용량 모니터링 필요")
                            
                            print(f"   Streamlit 메모리: {memory_mb:.1f}MB (Working: {working_mb:.1f}MB)")
                        else:
                            self.log_issue("process", "medium", "프로세스 정보 불완전", 
                                         "Streamlit 프로세스 정보를 완전히 읽을 수 없음")
                    else:
                        self.log_issue("process", "critical", "Streamlit 프로세스 없음",
                                     "Streamlit 서버가 실행되지 않고 있음",
                                     "Streamlit 서버 재시작 필요")
            else:
                self.log_issue("process", "critical", "프로세스 확인 실패",
                             "시스템 프로세스 조회 실패")
                
        except Exception as e:
            self.log_issue("process", "high", "프로세스 체크 오류", f"프로세스 확인 중 오류: {e}")
    
    def check_file_system(self):
        """파일 시스템 상태 확인"""
        print("3. 파일 시스템 확인 중...")
        
        try:
            # 필수 파일들 확인
            essential_files = [
                "jewelry_stt_ui_v23_real.py",
                "core/real_analysis_engine.py", 
                "core/document_processor.py",
                "core/comprehensive_message_extractor.py"
            ]
            
            for file_path in essential_files:
                if not os.path.exists(file_path):
                    self.log_issue("filesystem", "critical",
                                 f"필수 파일 누락: {file_path}",
                                 f"{file_path} 파일이 존재하지 않음",
                                 "누락된 파일 복구 필요")
                else:
                    # 파일 크기 확인
                    size = os.path.getsize(file_path)
                    if size < 100:  # 100바이트 미만
                        self.log_issue("filesystem", "high",
                                     f"파일 크기 이상: {file_path}",
                                     f"{file_path} 파일 크기가 {size}바이트로 너무 작음",
                                     "파일 내용 확인 및 복구 필요")
            
            # 임시 파일 정리 확인
            temp_patterns = ["tmp*", "temp*", "*.tmp", "analysis_timing_*.json"]
            temp_count = 0
            for pattern in temp_patterns:
                temp_count += len(glob.glob(pattern))
            
            if temp_count > 20:
                self.log_issue("filesystem", "medium",
                             "임시 파일 과다",
                             f"임시 파일 {temp_count}개 발견",
                             "불필요한 임시 파일 정리 권장")
                             
        except Exception as e:
            self.log_issue("filesystem", "high", "파일시스템 체크 오류", f"파일시스템 확인 중 오류: {e}")
    
    def check_dependencies(self):
        """의존성 라이브러리 확인"""
        print("4. 의존성 라이브러리 확인 중...")
        
        critical_imports = [
            ("streamlit", "Streamlit UI 프레임워크"),
            ("whisper", "음성 인식"),
            ("easyocr", "이미지 텍스트 추출"),
            ("transformers", "AI 모델"),
            ("torch", "PyTorch 프레임워크")
        ]
        
        optional_imports = [
            ("opencv-python", "영상 처리"),
            ("moviepy", "동영상 편집"),
            ("yt-dlp", "YouTube 다운로드"),
            ("psutil", "시스템 모니터링")
        ]
        
        for module, description in critical_imports:
            try:
                __import__(module)
                print(f"   ✓ {module} ({description})")
            except ImportError:
                self.log_issue("dependencies", "critical",
                             f"필수 라이브러리 누락: {module}",
                             f"{description} 라이브러리가 설치되지 않음",
                             f"pip install {module} 실행 필요")
        
        for module, description in optional_imports:
            try:
                __import__(module)
                print(f"   ✓ {module} ({description})")
            except ImportError:
                self.log_issue("dependencies", "medium",
                             f"선택적 라이브러리 누락: {module}",
                             f"{description} 라이브러리 미설치로 일부 기능 제한",
                             f"pip install {module} 실행 권장")
    
    def check_performance_logs(self):
        """성능 로그 분석"""
        print("5. 성능 로그 분석 중...")
        
        try:
            timing_files = glob.glob("analysis_timing_*.json") + glob.glob("performance_log_*.json")
            
            if timing_files:
                slow_analyses = []
                memory_issues = []
                
                for log_file in timing_files[-10:]:  # 최근 10개
                    try:
                        with open(log_file, 'r') as f:
                            data = json.load(f)
                        
                        total_time = data.get('total_seconds', data.get('total_time_seconds', 0))
                        memory_delta = data.get('total_memory_delta_mb', data.get('memory_usage_mb', 0))
                        
                        if total_time > 300:  # 5분 이상
                            slow_analyses.append((log_file, total_time))
                        
                        if memory_delta > 1000:  # 1GB 이상
                            memory_issues.append((log_file, memory_delta))
                            
                    except Exception as e:
                        continue
                
                if slow_analyses:
                    self.log_issue("performance", "high",
                                 "느린 분석 속도 감지",
                                 f"{len(slow_analyses)}개의 분석이 5분 이상 소요됨",
                                 "AI 모델 최적화 또는 하드웨어 업그레이드 검토 필요")
                
                if memory_issues:
                    self.log_issue("performance", "high", 
                                 "메모리 사용량 과다",
                                 f"{len(memory_issues)}개의 분석에서 1GB 이상 메모리 사용",
                                 "메모리 누수 점검 및 배치 크기 조정 필요")
            else:
                self.log_issue("performance", "low",
                             "성능 로그 없음",
                             "분석 성능 로그가 발견되지 않음",
                             "성능 모니터링 활성화 권장")
                             
        except Exception as e:
            self.log_issue("performance", "medium", "성능 로그 분석 실패", f"성능 로그 분석 중 오류: {e}")
    
    def run_full_diagnostic(self):
        """전체 진단 실행"""
        print("=" * 60)
        print("SOLOMOND AI SYSTEM DIAGNOSTIC STARTED")
        print(f"Diagnostic Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 각 진단 항목 실행
        self.check_streamlit_logs()
        self.check_process_status()
        self.check_file_system()
        self.check_dependencies()
        self.check_performance_logs()
        
        # 결과 정리
        print("\n" + "=" * 60)
        print("DIAGNOSTIC RESULTS SUMMARY")
        print("=" * 60)
        
        total_issues = len(self.issues) + len(self.warnings)
        critical_count = len([i for i in self.issues if i['severity'] == 'critical'])
        high_count = len([i for i in self.issues if i['severity'] == 'high'])
        
        print(f"총 발견된 문제: {total_issues}개")
        print(f"심각한 문제: {critical_count}개")
        print(f"높은 우선순위: {high_count}개")
        
        if critical_count > 0:
            print("\nCRITICAL ISSUES (IMMEDIATE ACTION REQUIRED):")
            for issue in [i for i in self.issues if i['severity'] == 'critical']:
                print(f"   - {issue['title']}: {issue['description']}")
                if issue['solution']:
                    print(f"     SOLUTION: {issue['solution']}")
        
        if high_count > 0:
            print("\nHIGH PRIORITY ISSUES:")
            for issue in [i for i in self.issues if i['severity'] == 'high']:
                print(f"   - {issue['title']}: {issue['description']}")
                if issue['solution']:
                    print(f"     SOLUTION: {issue['solution']}")
        
        # 진단 결과 저장
        diagnostic_result = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": total_issues,
            "critical_issues": critical_count,
            "high_priority_issues": high_count,
            "issues": self.issues,
            "warnings": self.warnings
        }
        
        result_file = f"diagnostic_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(diagnostic_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 상세 진단 결과 저장: {result_file}")
        print("=" * 60)
        
        return diagnostic_result

if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    result = diagnostic.run_full_diagnostic()
    
    if result['critical_issues'] > 0:
        print("\nSTATUS: CRITICAL ISSUES FOUND!")
        print("Follow the solutions above to fix critical problems.")
    elif result['high_priority_issues'] > 0:
        print("\nSTATUS: SYSTEM OPTIMIZATION NEEDED")
        print("Resolving high priority issues will improve performance.")
    else:
        print("\nSTATUS: SYSTEM OPERATING NORMALLY")