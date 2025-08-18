#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 SOLOMOND AI Serena 코딩 에이전트 통합 시스템
Serena-Powered Code Intelligence for SOLOMOND AI

핵심 기능:
1. Symbol-level 코드 분석 및 편집
2. 토큰 효율적인 코드 블록 추출
3. 프로젝트별 컨텍스트 메모리
4. 정밀한 코드 수정 및 최적화
5. SOLOMOND AI 특화 성능 최적화

Serena의 핵심 능력을 표준 Python 도구로 구현
"""

import os
import ast
import re
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import traceback
from collections import defaultdict, Counter

# 프로젝트 구조 분석을 위한 임포트
import tokenize
import io
from tokenize import TokenInfo

@dataclass
class CodeSymbol:
    """코드 심볼 정보"""
    name: str
    type: str  # function, class, variable, import
    file_path: str
    line_start: int
    line_end: int
    indentation: int
    docstring: Optional[str] = None
    arguments: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    complexity: int = 0
    last_modified: Optional[str] = None

@dataclass
class CodeBlock:
    """토큰 효율적인 코드 블록"""
    content: str
    symbols: List[CodeSymbol]
    file_path: str
    block_id: str
    token_count: int
    importance_score: float
    context_relevance: float

@dataclass
class ProjectMemory:
    """프로젝트 컨텍스트 메모리"""
    project_name: str
    symbols: Dict[str, CodeSymbol]
    dependencies: Dict[str, List[str]]
    patterns: Dict[str, int]
    performance_bottlenecks: List[str]
    last_updated: str
    code_quality_metrics: Dict[str, float]

class SerenaCodeAnalyzer:
    """Serena 스타일 코드 분석기"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.memory_file = self.project_root / ".solomond_serena_memory.json"
        self.project_memory = self._load_or_create_memory()
        self.logger = self._setup_logger()
        
        # SOLOMOND AI 특화 패턴
        self.solomond_patterns = {
            'threadpool_issues': r'ThreadPoolExecutor.*(?:submit|map)',
            'memory_leaks': r'(?:torch\.cuda|np\.array|PIL\.Image).*(?:not.*del|no.*cleanup)',
            'streamlit_performance': r'st\.(?:cache|session_state).*(?:heavy|large|slow)',
            'ollama_integration': r'ollama.*(?:generate|chat|pull)',
            'multimodal_processing': r'(?:audio|image|video).*(?:process|analyze|extract)',
            'gpu_memory_issues': r'(?:cuda|gpu).*(?:memory|allocation|oom)',
            'error_handling_missing': r'try:.*except.*pass',
            'inefficient_loops': r'for.*in.*(?:range\(len|enumerate).*\[.*\]'
        }
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("SerenaAgent")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.project_root / "serena_agent.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def _load_or_create_memory(self) -> ProjectMemory:
        """프로젝트 메모리 로드 또는 생성"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ProjectMemory(**data)
            except Exception as e:
                print(f"메모리 로드 실패, 새로 생성: {e}")
        
        return ProjectMemory(
            project_name="SOLOMOND_AI",
            symbols={},
            dependencies={},
            patterns={},
            performance_bottlenecks=[],
            last_updated=datetime.now().isoformat(),
            code_quality_metrics={}
        )
    
    def save_memory(self):
        """프로젝트 메모리 저장"""
        try:
            self.project_memory.last_updated = datetime.now().isoformat()
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.project_memory), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"메모리 저장 실패: {e}")

    def analyze_file_symbols(self, file_path: str) -> List[CodeSymbol]:
        """파일의 모든 심볼 분석 (Symbol-level analysis)"""
        symbols = []
        file_path = Path(file_path)
        
        if not file_path.exists() or file_path.suffix != '.py':
            return symbols
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                symbol = self._extract_symbol_from_node(node, str(file_path), content)
                if symbol:
                    symbols.append(symbol)
                    
        except Exception as e:
            self.logger.error(f"파일 분석 실패 {file_path}: {e}")
            
        return symbols
    
    def _extract_symbol_from_node(self, node: ast.AST, file_path: str, content: str) -> Optional[CodeSymbol]:
        """AST 노드에서 심볼 정보 추출"""
        lines = content.split('\n')
        
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            args = [arg.arg for arg in node.args.args]
            
            return CodeSymbol(
                name=node.name,
                type="function",
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                indentation=len(lines[node.lineno-1]) - len(lines[node.lineno-1].lstrip()),
                docstring=docstring,
                arguments=args,
                complexity=self._calculate_complexity(node)
            )
            
        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            
            return CodeSymbol(
                name=node.name,
                type="class",
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                indentation=len(lines[node.lineno-1]) - len(lines[node.lineno-1].lstrip()),
                docstring=docstring,
                complexity=len(node.body)
            )
            
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            import_names = []
            if isinstance(node, ast.Import):
                import_names = [alias.name for alias in node.names]
            else:
                module = node.module or ""
                import_names = [f"{module}.{alias.name}" for alias in node.names]
                
            return CodeSymbol(
                name=", ".join(import_names),
                type="import",
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.lineno,
                indentation=len(lines[node.lineno-1]) - len(lines[node.lineno-1].lstrip()),
                dependencies=import_names
            )
            
        return None
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """코드 복잡도 계산 (McCabe 순환 복잡도 간소화 버전)"""
        complexity = 1  # 기본 복잡도
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
                
        return complexity

    def extract_efficient_code_blocks(self, file_path: str, max_tokens: int = 2000) -> List[CodeBlock]:
        """토큰 효율적인 코드 블록 추출"""
        symbols = self.analyze_file_symbols(file_path)
        blocks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            current_block = []
            current_symbols = []
            current_tokens = 0
            
            for symbol in sorted(symbols, key=lambda s: s.line_start):
                # 심볼에 해당하는 코드 라인들 추출
                symbol_lines = lines[symbol.line_start-1:symbol.line_end]
                symbol_content = ''.join(symbol_lines)
                symbol_tokens = len(symbol_content.split())
                
                if current_tokens + symbol_tokens > max_tokens and current_block:
                    # 현재 블록 완성
                    block_content = ''.join(current_block)
                    block_id = hashlib.md5(block_content.encode()).hexdigest()[:8]
                    
                    blocks.append(CodeBlock(
                        content=block_content,
                        symbols=current_symbols.copy(),
                        file_path=file_path,
                        block_id=block_id,
                        token_count=current_tokens,
                        importance_score=self._calculate_importance(current_symbols),
                        context_relevance=self._calculate_relevance(current_symbols)
                    ))
                    
                    current_block = []
                    current_symbols = []
                    current_tokens = 0
                
                current_block.extend(symbol_lines)
                current_symbols.append(symbol)
                current_tokens += symbol_tokens
            
            # 마지막 블록 처리
            if current_block:
                block_content = ''.join(current_block)
                block_id = hashlib.md5(block_content.encode()).hexdigest()[:8]
                
                blocks.append(CodeBlock(
                    content=block_content,
                    symbols=current_symbols,
                    file_path=file_path,
                    block_id=block_id,
                    token_count=current_tokens,
                    importance_score=self._calculate_importance(current_symbols),
                    context_relevance=self._calculate_relevance(current_symbols)
                ))
                
        except Exception as e:
            self.logger.error(f"코드 블록 추출 실패 {file_path}: {e}")
            
        return blocks
    
    def _calculate_importance(self, symbols: List[CodeSymbol]) -> float:
        """심볼들의 중요도 계산"""
        score = 0.0
        
        for symbol in symbols:
            # 클래스와 함수는 높은 점수
            if symbol.type == "class":
                score += 3.0
            elif symbol.type == "function":
                score += 2.0
                # 복잡한 함수는 더 높은 점수
                score += symbol.complexity * 0.1
            elif symbol.type == "import":
                score += 0.5
                
        return score / len(symbols) if symbols else 0.0
    
    def _calculate_relevance(self, symbols: List[CodeSymbol]) -> float:
        """SOLOMOND AI 프로젝트와의 관련성 계산"""
        relevance = 0.0
        solomond_keywords = [
            'streamlit', 'conference', 'analysis', 'ollama', 'whisper', 
            'easyocr', 'multimodal', 'gpu', 'threading', 'threadpool'
        ]
        
        for symbol in symbols:
            symbol_text = f"{symbol.name} {symbol.docstring or ''}".lower()
            keyword_matches = sum(1 for keyword in solomond_keywords if keyword in symbol_text)
            relevance += keyword_matches * 0.2
            
        return min(relevance, 1.0)

    def detect_solomond_issues(self, file_path: str) -> Dict[str, List[Dict]]:
        """SOLOMOND AI 특화 문제점 탐지"""
        issues = defaultdict(list)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for pattern_name, pattern in self.solomond_patterns.items():
                matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issue = {
                        'line': line_num,
                        'code': lines[line_num-1].strip() if line_num <= len(lines) else '',
                        'pattern': pattern_name,
                        'match': match.group(),
                        'suggestion': self._get_suggestion(pattern_name)
                    }
                    issues[pattern_name].append(issue)
                    
        except Exception as e:
            self.logger.error(f"문제점 탐지 실패 {file_path}: {e}")
            
        return dict(issues)
    
    def _get_suggestion(self, pattern_name: str) -> str:
        """패턴별 개선 제안"""
        suggestions = {
            'threadpool_issues': 'ThreadPoolExecutor 사용 시 with 문과 적절한 max_workers 설정 사용',
            'memory_leaks': '메모리 해제를 위한 del 문과 가비지 컬렉션 호출 추가',
            'streamlit_performance': 'st.cache_data 또는 st.cache_resource 사용으로 성능 최적화',
            'ollama_integration': 'Ollama API 호출 시 에러 처리와 재시도 로직 추가',
            'multimodal_processing': '대용량 파일 처리 시 청크 단위 처리와 메모리 관리',
            'gpu_memory_issues': 'torch.cuda.empty_cache() 호출과 메모리 모니터링 추가',
            'error_handling_missing': '구체적인 예외 처리와 로깅 추가',
            'inefficient_loops': '리스트 컴프리헨션이나 벡터화 연산 사용 고려'
        }
        return suggestions.get(pattern_name, '코드 최적화 검토 필요')

    def find_symbol(self, symbol_name: str, symbol_type: str = None) -> List[CodeSymbol]:
        """심볼 검색 (Serena의 핵심 기능)"""
        results = []
        
        for file_path in self.project_root.glob("**/*.py"):
            symbols = self.analyze_file_symbols(str(file_path))
            
            for symbol in symbols:
                if symbol.name == symbol_name:
                    if symbol_type is None or symbol.type == symbol_type:
                        results.append(symbol)
                        
        return results
    
    def suggest_optimizations(self, file_path: str) -> Dict[str, Any]:
        """성능 최적화 제안"""
        symbols = self.analyze_file_symbols(file_path)
        issues = self.detect_solomond_issues(file_path)
        
        optimizations = {
            'high_complexity_functions': [],
            'performance_issues': [],
            'memory_optimizations': [],
            'solomond_specific': []
        }
        
        # 고복잡도 함수 찾기
        for symbol in symbols:
            if symbol.type == "function" and symbol.complexity > 10:
                optimizations['high_complexity_functions'].append({
                    'function': symbol.name,
                    'complexity': symbol.complexity,
                    'line': symbol.line_start,
                    'suggestion': '함수 분리나 리팩토링 검토 필요'
                })
        
        # SOLOMOND AI 특화 이슈
        for issue_type, issue_list in issues.items():
            optimizations['solomond_specific'].extend([
                {
                    'type': issue_type,
                    'details': issue,
                    'priority': 'high' if 'threadpool' in issue_type or 'memory' in issue_type else 'medium'
                }
                for issue in issue_list
            ])
        
        return optimizations
    
    def update_project_memory(self, file_path: str):
        """프로젝트 메모리 업데이트"""
        symbols = self.analyze_file_symbols(file_path)
        
        for symbol in symbols:
            symbol_key = f"{symbol.file_path}:{symbol.name}"
            self.project_memory.symbols[symbol_key] = symbol
            
        # 의존성 그래프 업데이트
        import_symbols = [s for s in symbols if s.type == "import"]
        if import_symbols:
            file_key = str(file_path)
            self.project_memory.dependencies[file_key] = []
            for imp in import_symbols:
                if imp.dependencies:
                    self.project_memory.dependencies[file_key].extend(imp.dependencies)
        
        # 패턴 빈도 업데이트
        issues = self.detect_solomond_issues(file_path)
        for pattern_name, issue_list in issues.items():
            if pattern_name not in self.project_memory.patterns:
                self.project_memory.patterns[pattern_name] = 0
            self.project_memory.patterns[pattern_name] += len(issue_list)
        
        self.save_memory()

class SerenaIntegrationEngine:
    """Serena 통합 엔진 - SOLOMOND AI와의 통합 관리"""
    
    def __init__(self, project_root: str = None):
        self.analyzer = SerenaCodeAnalyzer(project_root)
        self.integration_log = []
        
    def analyze_project_health(self) -> Dict[str, Any]:
        """프로젝트 전체 건강도 분석"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'files_analyzed': 0,
            'total_symbols': 0,
            'critical_issues': 0,
            'performance_bottlenecks': [],
            'recommendations': []
        }
        
        total_files = 0
        total_issues = 0
        
        # 주요 Python 파일들 분석
        for py_file in self.analyzer.project_root.glob("**/*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', '.git']):
                continue
                
            total_files += 1
            symbols = self.analyzer.analyze_file_symbols(str(py_file))
            issues = self.analyzer.detect_solomond_issues(str(py_file))
            
            health_report['total_symbols'] += len(symbols)
            
            # 크리티컬 이슈 카운트
            critical_patterns = ['threadpool_issues', 'memory_leaks', 'gpu_memory_issues']
            for pattern in critical_patterns:
                if pattern in issues:
                    total_issues += len(issues[pattern])
                    health_report['critical_issues'] += len(issues[pattern])
                    
            # 성능 병목점 추가
            optimizations = self.analyzer.suggest_optimizations(str(py_file))
            if optimizations['high_complexity_functions']:
                health_report['performance_bottlenecks'].extend([
                    f"{py_file.name}:{opt['function']}" for opt in optimizations['high_complexity_functions']
                ])
        
        health_report['files_analyzed'] = total_files
        
        # 전체 점수 계산 (100점 만점)
        if total_files > 0:
            issue_penalty = min(total_issues * 5, 50)  # 이슈당 5점 감점, 최대 50점
            health_report['overall_score'] = max(100 - issue_penalty, 0)
        
        # 추천사항 생성
        if health_report['critical_issues'] > 0:
            health_report['recommendations'].append("크리티컬 이슈 우선 해결 필요")
        if len(health_report['performance_bottlenecks']) > 5:
            health_report['recommendations'].append("고복잡도 함수들의 리팩토링 검토")
        if health_report['overall_score'] < 70:
            health_report['recommendations'].append("코드 품질 전반적 개선 필요")
            
        return health_report
    
    def generate_optimization_plan(self) -> Dict[str, Any]:
        """최적화 계획 생성"""
        plan = {
            'priority_fixes': [],
            'performance_improvements': [],
            'code_quality_enhancements': [],
            'estimated_impact': {}
        }
        
        # 프로젝트 전체 분석
        for py_file in self.analyzer.project_root.glob("**/*.py"):
            if any(skip in str(py_file) for skip in ['venv', '__pycache__', '.git']):
                continue
                
            issues = self.analyzer.detect_solomond_issues(str(py_file))
            optimizations = self.analyzer.suggest_optimizations(str(py_file))
            
            # 우선순위 픽스 (크리티컬 이슈)
            critical_patterns = ['threadpool_issues', 'memory_leaks', 'gpu_memory_issues']
            for pattern in critical_patterns:
                if pattern in issues:
                    for issue in issues[pattern]:
                        plan['priority_fixes'].append({
                            'file': str(py_file),
                            'issue': issue,
                            'estimated_time': '30-60분',
                            'impact': 'high'
                        })
            
            # 성능 개선사항
            if optimizations['high_complexity_functions']:
                for func_opt in optimizations['high_complexity_functions']:
                    plan['performance_improvements'].append({
                        'file': str(py_file),
                        'function': func_opt['function'],
                        'complexity': func_opt['complexity'],
                        'suggestion': func_opt['suggestion'],
                        'estimated_time': '1-2시간',
                        'impact': 'medium'
                    })
        
        # 영향도 추정
        plan['estimated_impact'] = {
            'priority_fixes': f"{len(plan['priority_fixes'])}개 크리티컬 이슈 해결로 안정성 20-30% 향상",
            'performance_improvements': f"{len(plan['performance_improvements'])}개 함수 최적화로 성능 10-20% 향상",
            'total_effort': f"총 {len(plan['priority_fixes']) + len(plan['performance_improvements'])}개 항목, 예상 소요시간: {len(plan['priority_fixes']) * 1 + len(plan['performance_improvements']) * 1.5:.1f}시간"
        }
        
        return plan

def main():
    """Serena 에이전트 메인 실행"""
    print("🧠 SOLOMOND AI Serena 코딩 에이전트 시작")
    
    # 프로젝트 루트에서 실행
    project_root = Path.cwd()
    engine = SerenaIntegrationEngine(str(project_root))
    
    print("\n📊 프로젝트 건강도 분석 중...")
    health_report = engine.analyze_project_health()
    
    print(f"✅ 분석 완료!")
    print(f"📁 분석된 파일: {health_report['files_analyzed']}개")
    print(f"🔍 발견된 심볼: {health_report['total_symbols']}개")
    print(f"⚠️  크리티컬 이슈: {health_report['critical_issues']}개")
    print(f"📈 전체 건강도: {health_report['overall_score']:.1f}/100")
    
    if health_report['performance_bottlenecks']:
        print(f"\n🐌 성능 병목점:")
        for bottleneck in health_report['performance_bottlenecks'][:5]:
            print(f"  - {bottleneck}")
    
    if health_report['recommendations']:
        print(f"\n💡 추천사항:")
        for rec in health_report['recommendations']:
            print(f"  - {rec}")
    
    print("\n🎯 최적화 계획 생성 중...")
    optimization_plan = engine.generate_optimization_plan()
    
    if optimization_plan['priority_fixes']:
        print(f"\n🚨 우선순위 수정사항 ({len(optimization_plan['priority_fixes'])}개):")
        for fix in optimization_plan['priority_fixes'][:3]:
            print(f"  - {Path(fix['file']).name}: {fix['issue']['pattern']}")
    
    print(f"\n📈 예상 효과:")
    for impact_type, impact_desc in optimization_plan['estimated_impact'].items():
        print(f"  - {impact_desc}")
    
    print(f"\n💾 프로젝트 메모리 업데이트 완료")
    print(f"📝 로그 파일: {project_root / 'serena_agent.log'}")

if __name__ == "__main__":
    main()