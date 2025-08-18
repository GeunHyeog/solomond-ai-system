#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Serena 자동 수정 도구
SOLOMOND AI 시스템의 일반적인 문제들을 자동으로 수정

주요 기능:
1. ThreadPool 에러 자동 수정
2. 메모리 누수 방지 코드 추가
3. Streamlit 성능 최적화
4. GPU/CPU 메모리 관리 개선
5. 에러 처리 강화
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

@dataclass
class FixRule:
    """수정 규칙"""
    name: str
    pattern: str
    replacement: str
    description: str
    priority: int  # 1=highest, 5=lowest
    file_types: List[str]

class SerenaAutoFixer:
    """Serena 자동 수정 도구"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.fix_rules = self._initialize_fix_rules()
        self.backup_suffix = ".serena_backup"
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("SerenaAutoFixer")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("serena_auto_fixer.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _initialize_fix_rules(self) -> List[FixRule]:
        """수정 규칙 초기화"""
        return [
            # ThreadPool 문제 수정
            FixRule(
                name="threadpool_context_manager",
                pattern=r'(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n((?:(?!\nexecutor\.shutdown).*\n)*)',
                replacement=r'\1with ThreadPoolExecutor(\3) as executor:\n\1    # ThreadPool 작업들\n\1    pass  # 실제 작업 코드를 여기에 배치\n',
                description="ThreadPoolExecutor를 with 문으로 감싸서 자동 정리",
                priority=1,
                file_types=[".py"]
            ),
            
            # GPU 메모리 정리 추가
            FixRule(
                name="gpu_memory_cleanup",
                pattern=r'(import torch.*?\n)',
                replacement=r'\1import gc\n\ndef cleanup_gpu_memory():\n    """GPU 메모리 정리"""\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()\n        gc.collect()\n\n',
                description="GPU 메모리 정리 함수 추가",
                priority=2,
                file_types=[".py"]
            ),
            
            # Streamlit 캐시 최적화
            FixRule(
                name="streamlit_cache_optimization",
                pattern=r'@st\.cache\b',
                replacement='@st.cache_data',
                description="오래된 st.cache를 st.cache_data로 업데이트",
                priority=2,
                file_types=[".py"]
            ),
            
            # 메모리 누수 방지
            FixRule(
                name="large_object_cleanup",
                pattern=r'(\s*)(.*(?:large_data|big_array|huge_list|massive_dict)\s*=\s*.*)\n',
                replacement=r'\1\2\n\1# 메모리 누수 방지: 사용 후 정리\n\1# del large_data; gc.collect()  # 필요시 주석 해제\n',
                description="대용량 객체에 정리 코멘트 추가",
                priority=3,
                file_types=[".py"]
            ),
            
            # 예외 처리 개선
            FixRule(
                name="empty_except_handler",
                pattern=r'(\s*except.*?:\s*\n\s*)pass\s*\n',
                replacement=r'\1logging.warning("Exception occurred but was ignored")\n\1pass\n',
                description="빈 except 블록에 로깅 추가",
                priority=2,
                file_types=[".py"]
            ),
            
            # 파일 처리 with 문 사용
            FixRule(
                name="file_context_manager",
                pattern=r'(\s*)f\s*=\s*open\(([^)]+)\)\s*\n',
                replacement=r'\1with open(\2) as f:\n\1    # 파일 작업을 여기에서 수행\n\1    pass\n',
                description="파일 열기를 with 문으로 변경",
                priority=2,
                file_types=[".py"]
            )
        ]
    
    def analyze_file(self, file_path: str) -> Dict[str, List[Dict]]:
        """파일 분석하여 적용 가능한 수정사항 찾기"""
        fixes_needed = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for rule in self.fix_rules:
                if any(file_path.endswith(ft) for ft in rule.file_types):
                    matches = list(re.finditer(rule.pattern, content, re.MULTILINE | re.DOTALL))
                    
                    if matches:
                        fixes_needed[rule.name] = [
                            {
                                'rule': rule,
                                'match': match,
                                'line_number': content[:match.start()].count('\n') + 1,
                                'matched_text': match.group()
                            }
                            for match in matches
                        ]
                        
        except Exception as e:
            self.logger.error(f"파일 분석 실패 {file_path}: {e}")
            
        return fixes_needed
    
    def create_backup(self, file_path: str) -> str:
        """파일 백업 생성"""
        backup_path = file_path + self.backup_suffix
        
        try:
            with open(file_path, 'r', encoding='utf-8') as original:
                content = original.read()
            
            with open(backup_path, 'w', encoding='utf-8') as backup:
                backup.write(content)
                
            self.logger.info(f"백업 생성: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"백업 생성 실패 {file_path}: {e}")
            raise
    
    def apply_fix(self, file_path: str, rule_name: str) -> bool:
        """특정 수정 규칙 적용"""
        rule = next((r for r in self.fix_rules if r.name == rule_name), None)
        if not rule:
            self.logger.error(f"알 수 없는 규칙: {rule_name}")
            return False
        
        try:
            # 백업 생성
            self.create_backup(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 패턴 매칭 및 교체
            original_content = content
            content = re.sub(rule.pattern, rule.replacement, content, flags=re.MULTILINE | re.DOTALL)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.logger.info(f"수정 적용 완료: {file_path} - {rule.name}")
                return True
            else:
                self.logger.info(f"수정 사항 없음: {file_path} - {rule.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"수정 적용 실패 {file_path} - {rule_name}: {e}")
            return False
    
    def apply_all_fixes(self, file_path: str, priority_threshold: int = 3) -> Dict[str, bool]:
        """우선순위가 임계값 이하인 모든 수정사항 적용"""
        fixes_needed = self.analyze_file(file_path)
        results = {}
        
        # 우선순위 순으로 정렬
        applicable_rules = [
            rule for rule in self.fix_rules 
            if rule.priority <= priority_threshold and rule.name in fixes_needed
        ]
        applicable_rules.sort(key=lambda r: r.priority)
        
        for rule in applicable_rules:
            result = self.apply_fix(file_path, rule.name)
            results[rule.name] = result
            
        return results
    
    def fix_threadpool_specifically(self, file_path: str) -> bool:
        """ThreadPool 문제 특별 수정"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 백업 생성
            self.create_backup(file_path)
            
            # ThreadPoolExecutor 패턴들 찾기
            patterns_and_fixes = [
                # 패턴 1: 기본 ThreadPoolExecutor 사용
                (
                    r'(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n(.*?)(\n\s*executor\.shutdown\(\))',
                    r'\1with ThreadPoolExecutor() as executor:\n\3'
                ),
                
                # 패턴 2: submit 사용하는 경우
                (
                    r'(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n((?:.*?executor\.submit.*?\n)*)(.*?)(\n\s*(?:executor\.shutdown\(\)|del\s+executor))',
                    r'\1with ThreadPoolExecutor() as executor:\n\3\4'
                ),
                
                # 패턴 3: map 사용하는 경우
                (
                    r'(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n((?:.*?executor\.map.*?\n)*)(.*?)(\n\s*(?:executor\.shutdown\(\)|del\s+executor))',
                    r'\1with ThreadPoolExecutor() as executor:\n\3\4'
                )
            ]
            
            modified = False
            for pattern, replacement in patterns_and_fixes:
                new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
                if new_content != content:
                    content = new_content
                    modified = True
            
            # ThreadPoolExecutor import 확인 및 추가
            if 'ThreadPoolExecutor' in content and 'from concurrent.futures import ThreadPoolExecutor' not in content:
                if 'import concurrent.futures' not in content:
                    # import 추가
                    import_line = 'from concurrent.futures import ThreadPoolExecutor\n'
                    
                    # 다른 import 문 뒤에 추가
                    import_match = re.search(r'((?:^import .*?\n|^from .*? import .*?\n)+)', content, re.MULTILINE)
                    if import_match:
                        content = content[:import_match.end()] + import_line + content[import_match.end():]
                    else:
                        content = import_line + content
                    modified = True
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.logger.info(f"ThreadPool 수정 완료: {file_path}")
                return True
            else:
                self.logger.info(f"ThreadPool 수정 사항 없음: {file_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"ThreadPool 수정 실패 {file_path}: {e}")
            return False
    
    def generate_fix_report(self, directory: str) -> Dict[str, Any]:
        """디렉토리 전체 수정 보고서 생성"""
        directory = Path(directory)
        report = {
            'timestamp': str(datetime.now()),
            'total_files': 0,
            'files_with_issues': 0,
            'total_issues': 0,
            'issues_by_type': {},
            'files_analyzed': []
        }
        
        for py_file in directory.glob("**/*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', '.git']):
                continue
            
            report['total_files'] += 1
            fixes_needed = self.analyze_file(str(py_file))
            
            if fixes_needed:
                report['files_with_issues'] += 1
                
                file_report = {
                    'file': str(py_file),
                    'issues': {}
                }
                
                for rule_name, matches in fixes_needed.items():
                    issue_count = len(matches)
                    report['total_issues'] += issue_count
                    
                    if rule_name not in report['issues_by_type']:
                        report['issues_by_type'][rule_name] = 0
                    report['issues_by_type'][rule_name] += issue_count
                    
                    file_report['issues'][rule_name] = {
                        'count': issue_count,
                        'locations': [match['line_number'] for match in matches]
                    }
                
                report['files_analyzed'].append(file_report)
        
        return report

def fix_solomond_threadpool_issues():
    """SOLOMOND AI 시스템의 ThreadPool 이슈 일괄 수정"""
    fixer = SerenaAutoFixer()
    
    # 주요 파일들 수정
    important_files = [
        'conference_analysis_COMPLETE_WORKING.py',
        'hybrid_compute_manager.py',
        'core/multimodal_pipeline.py',
        'core/batch_processing_engine.py'
    ]
    
    results = {}
    
    for file_name in important_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"🔧 {file_name} ThreadPool 이슈 수정 중...")
            success = fixer.fix_threadpool_specifically(str(file_path))
            results[file_name] = success
            
            if success:
                print(f"✅ {file_name} 수정 완료")
            else:
                print(f"ℹ️  {file_name} 수정 사항 없음")
        else:
            print(f"❌ {file_name} 파일을 찾을 수 없음")
            results[file_name] = False
    
    return results

if __name__ == "__main__":
    from datetime import datetime
    
    print("🧠 Serena 자동 수정 도구 시작")
    print("🎯 SOLOMOND AI ThreadPool 이슈 수정 중...")
    
    results = fix_solomond_threadpool_issues()
    
    print("\n📊 수정 결과:")
    for file_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패/불필요"
        print(f"  {file_name}: {status}")
    
    print(f"\n💾 로그 파일: serena_auto_fixer.log")
    print("🎉 Serena 자동 수정 완료!")