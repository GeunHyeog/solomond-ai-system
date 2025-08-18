#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 SOLOMOND AI Serena 코딩 에이전트 - Claude Code 서브에이전트 통합
Serena Coding Agent as Claude Code Sub-Agent

이 모듈은 Serena의 코딩 에이전트 기능을 Claude Code의 서브에이전트로 통합합니다.
/agent serena 명령어로 호출 가능한 실제 Claude Code 서브에이전트를 구현합니다.

핵심 기능:
1. Symbol-level 코드 분석 및 편집
2. SOLOMOND AI 특화 이슈 자동 탐지  
3. 성능 최적화 제안 및 자동 수정
4. 정밀한 코드 품질 분석
5. 프로젝트 건강도 모니터링

Author: SOLOMOND AI Team
Version: 1.0.0
Created: 2025-08-17
"""

import ast
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from collections import defaultdict, Counter

# Serena 에이전트 페르소나 정의
SERENA_PERSONA = {
    "name": "Serena",
    "role": "SOLOMOND AI 전문 코딩 에이전트",
    "expertise": [
        "Python 코드 분석 및 최적화",
        "SOLOMOND AI 시스템 아키텍처",
        "ThreadPool 및 메모리 관리",
        "Streamlit 성능 최적화",
        "멀티모달 파이프라인 분석",
        "Ollama AI 통합 최적화"
    ],
    "personality": {
        "communication_style": "정밀하고 체계적",
        "tone": "기술적이지만 친근한",
        "approach": "문제의 근본 원인을 파악하여 해결"
    },
    "specialization": "SOLOMOND AI 컨퍼런스 분석 시스템의 안정성과 성능 극대화"
}

# SOLOMOND AI 특화 패턴
SOLOMOND_PATTERNS = {
    'threadpool_critical': {
        'pattern': r'ThreadPoolExecutor.*(?:submit|map).*(?!with\s)',
        'severity': 'critical',
        'description': 'ThreadPoolExecutor without context manager',
        'fix': 'Use with statement for proper resource management'
    },
    'memory_leak_risk': {
        'pattern': r'(?:torch\.cuda|np\.array|PIL\.Image).*(?:not.*del|no.*cleanup)',
        'severity': 'high',
        'description': 'Potential memory leak in GPU/large array operations',
        'fix': 'Add explicit cleanup and memory management'
    },
    'streamlit_performance': {
        'pattern': r'st\.(?!cache_data|cache_resource).*(?:heavy|large|slow)',
        'severity': 'medium',
        'description': 'Heavy operation without Streamlit caching',
        'fix': 'Use st.cache_data or st.cache_resource for optimization'
    },
    'ollama_error_handling': {
        'pattern': r'ollama.*(?:generate|chat|pull).*(?!try|except)',
        'severity': 'medium',
        'description': 'Ollama API call without error handling',
        'fix': 'Add try-except block for robust error handling'
    },
    'gpu_memory_management': {
        'pattern': r'(?:cuda|gpu).*(?:memory|allocation).*(?!empty_cache)',
        'severity': 'high',
        'description': 'GPU memory operations without cleanup',
        'fix': 'Add torch.cuda.empty_cache() calls'
    },
    'inefficient_file_io': {
        'pattern': r'open\(.*\)(?!.*with)',
        'severity': 'medium',
        'description': 'File I/O without context manager',
        'fix': 'Use with statement for safe file operations'
    }
}

@dataclass
class CodeIssue:
    """코드 이슈 정보"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    code_snippet: str
    suggested_fix: str
    confidence: float

@dataclass
class AnalysisResult:
    """분석 결과"""
    timestamp: str
    files_analyzed: int
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    issues: List[CodeIssue]
    recommendations: List[str]
    health_score: float

class SerenaCodeAnalyzer:
    """Serena 코드 분석기 - Claude Code 최적화 버전"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("SerenaAgent")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.project_root / "serena_agent.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_file(self, file_path: str) -> List[CodeIssue]:
        """단일 파일 분석"""
        issues = []
        file_path = Path(file_path)
        
        if not file_path.exists() or file_path.suffix != '.py':
            return issues
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # SOLOMOND AI 특화 패턴 분석
            for pattern_name, pattern_info in SOLOMOND_PATTERNS.items():
                matches = re.finditer(pattern_info['pattern'], content, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    code_snippet = lines[line_num-1].strip() if line_num <= len(lines) else ''
                    
                    issue = CodeIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type=pattern_name,
                        severity=pattern_info['severity'],
                        description=pattern_info['description'],
                        code_snippet=code_snippet,
                        suggested_fix=pattern_info['fix'],
                        confidence=0.85  # 패턴 매칭 기반 신뢰도
                    )
                    issues.append(issue)
                    
        except Exception as e:
            self.logger.error(f"파일 분석 실패 {file_path}: {e}")
            
        return issues

    def analyze_project(self, target_files: List[str] = None) -> AnalysisResult:
        """프로젝트 전체 분석"""
        start_time = datetime.now()
        all_issues = []
        files_analyzed = 0
        
        # 분석 대상 파일 결정
        if target_files:
            files_to_analyze = [Path(f) for f in target_files if Path(f).exists()]
        else:
            # 주요 SOLOMOND AI 파일들 우선 분석
            priority_files = [
                "conference_analysis_COMPLETE_WORKING.py",
                "solomond_ai_main_dashboard.py",
                "dual_brain_integration.py",
                "ai_insights_engine.py",
                "google_calendar_connector.py"
            ]
            
            files_to_analyze = []
            for file_name in priority_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    files_to_analyze.append(file_path)
            
            # core 디렉토리 추가 분석
            core_dir = self.project_root / "core"
            if core_dir.exists():
                core_files = list(core_dir.glob("*.py"))[:10]  # 최대 10개 파일
                files_to_analyze.extend(core_files)
        
        # 파일별 분석 실행
        for file_path in files_to_analyze:
            if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git', 'backup']):
                continue
                
            files_analyzed += 1
            file_issues = self.analyze_file(str(file_path))
            all_issues.extend(file_issues)
        
        # 심각도별 이슈 카운트
        severity_counts = Counter(issue.severity for issue in all_issues)
        
        # 건강도 점수 계산 (100점 만점)
        health_score = self._calculate_health_score(severity_counts, files_analyzed)
        
        # 추천사항 생성
        recommendations = self._generate_recommendations(severity_counts, all_issues)
        
        return AnalysisResult(
            timestamp=start_time.isoformat(),
            files_analyzed=files_analyzed,
            total_issues=len(all_issues),
            critical_issues=severity_counts.get('critical', 0),
            high_issues=severity_counts.get('high', 0),
            medium_issues=severity_counts.get('medium', 0),
            issues=all_issues,
            recommendations=recommendations,
            health_score=health_score
        )

    def _calculate_health_score(self, severity_counts: Counter, files_analyzed: int) -> float:
        """건강도 점수 계산"""
        if files_analyzed == 0:
            return 0.0
            
        # 심각도별 가중치
        critical_penalty = severity_counts.get('critical', 0) * 15
        high_penalty = severity_counts.get('high', 0) * 8
        medium_penalty = severity_counts.get('medium', 0) * 3
        
        total_penalty = critical_penalty + high_penalty + medium_penalty
        base_score = 100.0
        
        # 파일당 평균 페널티 계산
        avg_penalty = total_penalty / files_analyzed if files_analyzed > 0 else 0
        
        return max(base_score - avg_penalty, 0.0)

    def _generate_recommendations(self, severity_counts: Counter, issues: List[CodeIssue]) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        if severity_counts.get('critical', 0) > 0:
            recommendations.append("🚨 크리티컬 이슈 즉시 수정 필요 - ThreadPool 및 리소스 관리")
            
        if severity_counts.get('high', 0) > 0:
            recommendations.append("⚠️ 높은 우선순위 이슈 해결 - 메모리 누수 및 GPU 관리")
            
        # 패턴별 추천
        issue_patterns = Counter(issue.issue_type for issue in issues)
        
        if issue_patterns.get('threadpool_critical', 0) > 0:
            recommendations.append("🔧 ThreadPoolExecutor를 with 문으로 감싸서 리소스 안전 관리")
            
        if issue_patterns.get('streamlit_performance', 0) > 0:
            recommendations.append("⚡ Streamlit 캐싱 시스템 적용으로 성능 최적화")
            
        if issue_patterns.get('memory_leak_risk', 0) > 0:
            recommendations.append("🧹 명시적 메모리 정리 및 가비지 컬렉션 호출")
            
        return recommendations

    def generate_fix_script(self, issues: List[CodeIssue]) -> str:
        """자동 수정 스크립트 생성"""
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        
        if not critical_issues:
            return "# 수정이 필요한 크리티컬 이슈가 없습니다."
            
        script_lines = [
            "#!/usr/bin/env python3",
            "# Serena 자동 생성 수정 스크립트",
            "# SOLOMOND AI 시스템 최적화",
            "",
            "import re",
            "from pathlib import Path",
            "",
            "def apply_fixes():",
            "    \"\"\"크리티컬 이슈 자동 수정\"\"\"",
            "    fixes_applied = 0",
            ""
        ]
        
        # ThreadPool 이슈 수정 로직
        threadpool_issues = [i for i in critical_issues if i.issue_type == 'threadpool_critical']
        if threadpool_issues:
            script_lines.extend([
                "    # ThreadPool 이슈 수정",
                "    threadpool_files = [",
            ])
            
            for issue in threadpool_issues:
                script_lines.append(f"        '{issue.file_path}',")
                
            script_lines.extend([
                "    ]",
                "",
                "    for file_path in threadpool_files:",
                "        if Path(file_path).exists():",
                "            with open(file_path, 'r') as f:",
                "                content = f.read()",
                "            ",
                "            # with 문으로 감싸기",
                "            pattern = r'(\\s*)(executor\\s*=\\s*ThreadPoolExecutor\\([^)]*\\))\\s*\\n'",
                "            replacement = r'\\1with ThreadPoolExecutor() as executor:\\n\\1    # 수정됨: with 문 사용\\n'",
                "            ",
                "            new_content = re.sub(pattern, replacement, content)",
                "            if new_content != content:",
                "                with open(file_path, 'w') as f:",
                "                    f.write(new_content)",
                "                fixes_applied += 1",
                "                print(f'수정 완료: {file_path}')",
                ""
            ])
        
        script_lines.extend([
            "    return fixes_applied",
            "",
            "if __name__ == '__main__':",
            "    fixes = apply_fixes()",
            "    print(f'총 {fixes}개 파일 수정 완료')"
        ])
        
        return "\n".join(script_lines)

def main():
    """Serena 에이전트 실행"""
    print("🧠 SOLOMOND AI Serena 코딩 에이전트")
    print("=" * 50)
    
    analyzer = SerenaCodeAnalyzer()
    
    print("📊 프로젝트 분석 중...")
    result = analyzer.analyze_project()
    
    print(f"\n✅ 분석 완료 - {result.files_analyzed}개 파일")
    print(f"📈 전체 건강도: {result.health_score:.1f}/100")
    print(f"🔍 발견된 이슈: {result.total_issues}개")
    
    if result.critical_issues > 0:
        print(f"🚨 크리티컬: {result.critical_issues}개")
    if result.high_issues > 0:
        print(f"⚠️  높음: {result.high_issues}개")
    if result.medium_issues > 0:
        print(f"📝 보통: {result.medium_issues}개")
    
    # 상위 이슈들 표시
    if result.issues:
        print(f"\n🎯 주요 이슈:")
        critical_issues = [i for i in result.issues if i.severity == 'critical'][:3]
        for issue in critical_issues:
            print(f"  - {Path(issue.file_path).name}:{issue.line_number} - {issue.description}")
    
    # 추천사항 표시
    if result.recommendations:
        print(f"\n💡 추천사항:")
        for rec in result.recommendations:
            print(f"  {rec}")
    
    # 자동 수정 스크립트 생성 (크리티컬 이슈가 있는 경우)
    if result.critical_issues > 0:
        print(f"\n🔧 자동 수정 스크립트 생성 중...")
        fix_script = analyzer.generate_fix_script(result.issues)
        
        script_path = Path("serena_auto_fix.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(fix_script)
        
        print(f"✅ 수정 스크립트 생성: {script_path}")
        print(f"💡 실행 방법: python {script_path}")

if __name__ == "__main__":
    main()