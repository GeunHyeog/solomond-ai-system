#!/usr/bin/env python3
"""
고급 코드 품질 분석기 v2.5
2025 최신 코드 품질 표준 반영
"""

import ast
import os
import sys
import re
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class CodeQualityIssue:
    """코드 품질 이슈 데이터 클래스"""
    file_path: str
    line_number: int
    issue_type: str  # 'error', 'warning', 'style', 'complexity'
    severity: str   # 'critical', 'high', 'medium', 'low'
    message: str
    rule_name: str
    suggestion: str

@dataclass
class CodeQualityReport:
    """코드 품질 보고서 데이터 클래스"""
    timestamp: str
    project_name: str
    files_analyzed: int
    total_lines: int
    issues_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    complexity_score: float
    maintainability_index: float
    test_coverage: float
    issues: List[CodeQualityIssue]
    recommendations: List[str]

class CodeQualityAnalyzer:
    """고급 코드 품질 분석기"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
        # 분석 규칙 설정
        self.complexity_thresholds = {
            'function_lines': 50,
            'class_lines': 200,
            'cyclomatic_complexity': 10,
            'nesting_depth': 4
        }
        
        # 제외할 파일/디렉토리 패턴
        self.exclude_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            '.venv',
            'node_modules',
            '*.pyc',
            '.DS_Store'
        ]
        
        # 코딩 표준 규칙
        self.style_rules = {
            'max_line_length': 120,
            'indentation': 4,
            'naming_conventions': {
                'function': r'^[a-z_][a-z0-9_]*$',
                'class': r'^[A-Z][a-zA-Z0-9]*$',
                'constant': r'^[A-Z_][A-Z0-9_]*$',
                'variable': r'^[a-z_][a-z0-9_]*$'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger(f'{__name__}.CodeQualityAnalyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze_project(self) -> CodeQualityReport:
        """프로젝트 전체 분석"""
        self.logger.info(f"🔍 프로젝트 코드 품질 분석 시작: {self.project_root}")
        
        # Python 파일 수집
        python_files = self._collect_python_files()
        self.logger.info(f"📁 분석 대상 파일: {len(python_files)}개")
        
        if not python_files:
            self.logger.warning("⚠️ 분석할 Python 파일이 없습니다")
            return self._create_empty_report()
        
        # 병렬 분석 실행
        all_issues = []
        total_lines = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(self._analyze_file, file_path): file_path 
                for file_path in python_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_issues, file_lines = future.result()
                    all_issues.extend(file_issues)
                    total_lines += file_lines
                except Exception as e:
                    self.logger.error(f"❌ 파일 분석 실패 {file_path}: {e}")
        
        # 보고서 생성
        return self._generate_report(python_files, all_issues, total_lines)
    
    def _collect_python_files(self) -> List[Path]:
        """Python 파일 수집"""
        python_files = []
        
        for file_path in self.project_root.rglob("*.py"):
            # 제외 패턴 확인
            should_exclude = False
            for pattern in self.exclude_patterns:
                if pattern in str(file_path):
                    should_exclude = True
                    break
            
            if not should_exclude:
                python_files.append(file_path)
        
        return sorted(python_files)
    
    def _analyze_file(self, file_path: Path) -> Tuple[List[CodeQualityIssue], int]:
        """개별 파일 분석"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # AST 파싱
            try:
                tree = ast.parse(content, filename=str(file_path))
                
                # 다양한 분석 수행
                issues.extend(self._analyze_complexity(file_path, tree, lines))
                issues.extend(self._analyze_style(file_path, lines))
                issues.extend(self._analyze_structure(file_path, tree))
                issues.extend(self._analyze_imports(file_path, tree))
                issues.extend(self._analyze_security(file_path, content))
                
            except SyntaxError as e:
                issues.append(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 1,
                    issue_type='error',
                    severity='critical',
                    message=f"구문 오류: {e.msg}",
                    rule_name='syntax_error',
                    suggestion="구문 오류를 수정하세요"
                ))
            
            return issues, len(lines)
            
        except Exception as e:
            self.logger.error(f"❌ 파일 읽기 실패 {file_path}: {e}")
            return [], 0
    
    def _analyze_complexity(self, file_path: Path, tree: ast.AST, lines: List[str]) -> List[CodeQualityIssue]:
        """복잡도 분석"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 함수 길이 검사
                func_lines = node.end_lineno - node.lineno + 1
                if func_lines > self.complexity_thresholds['function_lines']:
                    issues.append(CodeQualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type='complexity',
                        severity='medium',
                        message=f"함수 '{node.name}'이 너무 깁니다 ({func_lines}줄)",
                        rule_name='function_length',
                        suggestion=f"함수를 {self.complexity_thresholds['function_lines']}줄 이하로 분할하세요"
                    ))
                
                # 사이클로매틱 복잡도 검사
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > self.complexity_thresholds['cyclomatic_complexity']:
                    issues.append(CodeQualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type='complexity',
                        severity='high',
                        message=f"함수 '{node.name}'의 복잡도가 높습니다 (복잡도: {complexity})",
                        rule_name='cyclomatic_complexity',
                        suggestion="함수를 더 작은 단위로 분할하거나 로직을 단순화하세요"
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # 클래스 길이 검사
                if node.end_lineno:
                    class_lines = node.end_lineno - node.lineno + 1
                    if class_lines > self.complexity_thresholds['class_lines']:
                        issues.append(CodeQualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type='complexity',
                            severity='medium',
                            message=f"클래스 '{node.name}'이 너무 큽니다 ({class_lines}줄)",
                            rule_name='class_length',
                            suggestion=f"클래스를 {self.complexity_thresholds['class_lines']}줄 이하로 분할하세요"
                        ))
        
        return issues
    
    def _analyze_style(self, file_path: Path, lines: List[str]) -> List[CodeQualityIssue]:
        """스타일 분석"""
        issues = []
        
        for line_num, line in enumerate(lines, 1):
            # 줄 길이 검사
            if len(line) > self.style_rules['max_line_length']:
                issues.append(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    issue_type='style',
                    severity='low',
                    message=f"줄이 너무 깁니다 ({len(line)}자)",
                    rule_name='line_length',
                    suggestion=f"줄을 {self.style_rules['max_line_length']}자 이하로 줄이세요"
                ))
            
            # 탭/공백 혼용 검사
            if '\t' in line and ' ' * self.style_rules['indentation'] in line:
                issues.append(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    issue_type='style',
                    severity='medium',
                    message="탭과 공백이 혼용되었습니다",
                    rule_name='mixed_indentation',
                    suggestion="일관된 들여쓰기 방식을 사용하세요"
                ))
            
            # 끝 공백 검사
            if line.endswith(' ') or line.endswith('\t'):
                issues.append(CodeQualityIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    issue_type='style',
                    severity='low',
                    message="줄 끝에 불필요한 공백이 있습니다",
                    rule_name='trailing_whitespace',
                    suggestion="줄 끝 공백을 제거하세요"
                ))
        
        return issues
    
    def _analyze_structure(self, file_path: Path, tree: ast.AST) -> List[CodeQualityIssue]:
        """구조 분석"""
        issues = []
        
        # 중복 코드 검사
        function_bodies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 함수 바디의 AST 덤프로 중복 검사
                body_dump = ast.dump(node)
                if body_dump in function_bodies:
                    issues.append(CodeQualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type='structure',
                        severity='medium',
                        message=f"중복된 함수 구조 발견: '{node.name}'",
                        rule_name='duplicate_code',
                        suggestion="중복 코드를 공통 함수로 추출하세요"
                    ))
                else:
                    function_bodies.append(body_dump)
        
        return issues
    
    def _analyze_imports(self, file_path: Path, tree: ast.AST) -> List[CodeQualityIssue]:
        """임포트 분석"""
        issues = []
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
        
        # 중복 임포트 검사
        import_names = []
        for imp in imports:
            if isinstance(imp, ast.Import):
                for alias in imp.names:
                    if alias.name in import_names:
                        issues.append(CodeQualityIssue(
                            file_path=str(file_path),
                            line_number=imp.lineno,
                            issue_type='structure',
                            severity='low',
                            message=f"중복 임포트: {alias.name}",
                            rule_name='duplicate_import',
                            suggestion="중복된 임포트를 제거하세요"
                        ))
                    else:
                        import_names.append(alias.name)
        
        return issues
    
    def _analyze_security(self, file_path: Path, content: str) -> List[CodeQualityIssue]:
        """보안 분석"""
        issues = []
        
        # 잠재적 보안 문제 패턴
        security_patterns = [
            (r'eval\s*\(', 'eval 사용', 'eval 대신 안전한 대안을 사용하세요'),
            (r'exec\s*\(', 'exec 사용', 'exec 대신 안전한 대안을 사용하세요'),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell=True 사용', 'shell=True 대신 인수 리스트를 사용하세요'),
            (r'password\s*=\s*["\'][^"\']*["\']', '하드코딩된 비밀번호', '비밀번호를 환경변수나 설정 파일로 분리하세요'),
            (r'api_key\s*=\s*["\'][^"\']*["\']', '하드코딩된 API 키', 'API 키를 환경변수로 분리하세요')
        ]
        
        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            for pattern, message, suggestion in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeQualityIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type='security',
                        severity='high',
                        message=message,
                        rule_name='security_violation',
                        suggestion=suggestion
                    ))
        
        return issues
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """사이클로매틱 복잡도 계산"""
        complexity = 1  # 기본 복잡도
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _generate_report(self, files: List[Path], issues: List[CodeQualityIssue], total_lines: int) -> CodeQualityReport:
        """보고서 생성"""
        
        # 심각도별 이슈 카운트
        critical_count = len([i for i in issues if i.severity == 'critical'])
        high_count = len([i for i in issues if i.severity == 'high'])
        medium_count = len([i for i in issues if i.severity == 'medium'])
        low_count = len([i for i in issues if i.severity == 'low'])
        
        # 복잡도 점수 계산 (0-100, 높을수록 좋음)
        complexity_score = max(0, 100 - (len(issues) / max(total_lines, 1) * 1000))
        
        # 유지보수성 지수 계산 (0-100, 높을수록 좋음)
        maintainability_index = max(0, 100 - (critical_count * 20 + high_count * 10 + medium_count * 5 + low_count * 2))
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(issues)
        
        return CodeQualityReport(
            timestamp=datetime.now().isoformat(),
            project_name=self.project_root.name,
            files_analyzed=len(files),
            total_lines=total_lines,
            issues_found=len(issues),
            critical_issues=critical_count,
            high_issues=high_count,
            medium_issues=medium_count,
            low_issues=low_count,
            complexity_score=round(complexity_score, 2),
            maintainability_index=round(maintainability_index, 2),
            test_coverage=0.0,  # TODO: 테스트 커버리지 계산 구현
            issues=issues,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, issues: List[CodeQualityIssue]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 심각도별 권장사항
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append(f"🚨 {len(critical_issues)}개의 심각한 문제를 즉시 해결하세요")
        
        high_issues = [i for i in issues if i.severity == 'high']
        if high_issues:
            recommendations.append(f"⚠️ {len(high_issues)}개의 높은 우선순위 문제를 해결하세요")
        
        # 패턴별 권장사항
        complexity_issues = [i for i in issues if i.issue_type == 'complexity']
        if len(complexity_issues) > 5:
            recommendations.append("🔄 복잡한 함수/클래스를 더 작은 단위로 리팩토링하세요")
        
        style_issues = [i for i in issues if i.issue_type == 'style']
        if len(style_issues) > 10:
            recommendations.append("🎨 코드 포맷터(black, autopep8)를 사용하여 스타일을 일관성 있게 유지하세요")
        
        security_issues = [i for i in issues if i.issue_type == 'security']
        if security_issues:
            recommendations.append("🔒 보안 취약점을 즉시 해결하고 정기적인 보안 검토를 실시하세요")
        
        # 일반적인 권장사항
        if len(issues) > 50:
            recommendations.append("📊 정기적인 코드 품질 모니터링을 위해 CI/CD에 품질 검사를 통합하세요")
        
        recommendations.append("📝 코드 리뷰 프로세스를 강화하여 품질을 지속적으로 개선하세요")
        
        return recommendations
    
    def _create_empty_report(self) -> CodeQualityReport:
        """빈 보고서 생성"""
        return CodeQualityReport(
            timestamp=datetime.now().isoformat(),
            project_name=self.project_root.name,
            files_analyzed=0,
            total_lines=0,
            issues_found=0,
            critical_issues=0,
            high_issues=0,
            medium_issues=0,
            low_issues=0,
            complexity_score=100.0,
            maintainability_index=100.0,
            test_coverage=0.0,
            issues=[],
            recommendations=["프로젝트에 Python 파일이 없습니다"]
        )
    
    def export_report(self, report: CodeQualityReport, output_path: str) -> None:
        """보고서 내보내기"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 코드 품질 보고서 저장됨: {output_path}")
    
    def print_summary(self, report: CodeQualityReport) -> None:
        """요약 출력"""
        print("\n" + "="*60)
        print(f"📊 코드 품질 분석 결과 - {report.project_name}")
        print("="*60)
        print(f"📁 분석된 파일: {report.files_analyzed}개")
        print(f"📏 총 코드 라인: {report.total_lines:,}줄")
        print(f"⚠️ 발견된 이슈: {report.issues_found}개")
        print(f"   🚨 심각: {report.critical_issues}개")
        print(f"   ⚠️ 높음: {report.high_issues}개")
        print(f"   📝 보통: {report.medium_issues}개")
        print(f"   💡 낮음: {report.low_issues}개")
        print(f"📈 복잡도 점수: {report.complexity_score}/100")
        print(f"🔧 유지보수성: {report.maintainability_index}/100")
        
        if report.recommendations:
            print("\n🎯 권장사항:")
            for rec in report.recommendations:
                print(f"   • {rec}")
        
        print("="*60)

# CLI 실행
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='코드 품질 분석기')
    parser.add_argument('project_path', help='분석할 프로젝트 경로')
    parser.add_argument('--output', '-o', help='보고서 출력 파일 경로')
    
    args = parser.parse_args()
    
    analyzer = CodeQualityAnalyzer(args.project_path)
    report = analyzer.analyze_project()
    
    analyzer.print_summary(report)
    
    if args.output:
        analyzer.export_report(report, args.output)
        print(f"✅ 상세 보고서가 저장되었습니다: {args.output}")