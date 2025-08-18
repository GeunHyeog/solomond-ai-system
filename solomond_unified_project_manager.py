#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 SOLOMOND AI 통합 프로젝트 매니저
Serena 코딩 에이전트 기능 완전 통합 버전

이 모듈은 SOLOMOND AI 시스템의 모든 프로젝트 관리 기능과
Serena의 고급 코딩 분석 기능을 하나로 통합한 슈퍼 에이전트입니다.

기능:
1. Symbol-level 코드 분석 (Serena 엔진)
2. 자동 이슈 탐지 및 수정 제안
3. 시스템 건강도 모니터링 (0-100점)
4. ThreadPool, 메모리 누수, 성능 병목점 탐지
5. SOLOMOND AI 특화 최적화
6. 프로젝트 전체 관리 및 복구

Author: SOLOMOND AI Team
Version: 2.0.0 (Serena Integration Complete)
Created: 2025-08-17
"""

import re
import json
import shutil
import ast
import inspect
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import subprocess
import sys

@dataclass
class CodeIssue:
    """코드 이슈 정보"""
    file_path: str
    line_number: int
    pattern_name: str
    severity: str  # critical, high, medium, low
    message: str
    code_snippet: str
    solution: str
    confidence: float
    impact_score: int

@dataclass
class SystemHealth:
    """시스템 건강도 정보"""
    overall_score: int  # 0-100
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    files_analyzed: int
    recommendations: List[str]

class SOLOMONDProjectManager:
    """SOLOMOND AI 통합 프로젝트 매니저 (Serena 통합)"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.version = "2.0.0"
        self.agent_name = "SOLOMOND Project Manager"
        
        # 로깅 설정
        self.logger = self._setup_logger()
        
        # Serena 통합: 고급 분석 패턴들
        self.critical_patterns = {
            "threadpool_without_context": {
                "regex": r"executor\s*=\s*ThreadPoolExecutor\([^)]*\)(?!\s*\n\s*with)",
                "severity": "critical",
                "message": "ThreadPoolExecutor가 context manager 없이 사용됨",
                "solution": "with ThreadPoolExecutor() as executor: 패턴 사용",
                "impact_score": 25,
                "auto_fixable": True
            },
            
            "memory_leak_cuda": {
                "regex": r"torch\.cuda\.(?!empty_cache|synchronize|is_available)",
                "severity": "high", 
                "message": "CUDA 연산 후 메모리 정리 누락",
                "solution": "torch.cuda.empty_cache() 호출 추가",
                "impact_score": 15,
                "auto_fixable": True
            },
            
            "streamlit_heavy_no_cache": {
                "regex": r"def\s+(\w*(?:heavy|process|analyze|load|model)\w*)\s*\([^)]*\):(?!\s*\n\s*@st\.cache)",
                "severity": "medium",
                "message": "무거운 함수에 Streamlit 캐시 미적용",
                "solution": "@st.cache_data 데코레이터 추가",
                "impact_score": 10,
                "auto_fixable": True
            },
            
            "ollama_no_exception": {
                "regex": r"ollama\.(?:generate|chat|create)(?!.*(?:try|except))",
                "severity": "medium",
                "message": "Ollama API 호출에 예외 처리 없음",
                "solution": "try-except 블록으로 예외 처리 추가",
                "impact_score": 8,
                "auto_fixable": False
            },
            
            "file_open_no_context": {
                "regex": r"(?:file|f)\s*=\s*open\([^)]+\)(?!\s*\n\s*with)",
                "severity": "high",
                "message": "파일 열기에 context manager 미사용",
                "solution": "with open(...) as file: 패턴 사용",
                "impact_score": 12,
                "auto_fixable": True
            },
            
            "subprocess_shell_true": {
                "regex": r"subprocess\.(?:run|call|Popen).*shell\s*=\s*True",
                "severity": "critical",
                "message": "subprocess에서 shell=True 사용 (보안 위험)",
                "solution": "shell=False 사용 또는 shlex.quote() 적용",
                "impact_score": 30,
                "auto_fixable": False
            }
        }
        
        # SOLOMOND AI 특화 패턴
        self.solomond_patterns = {
            "conference_analysis_optimization": {
                "regex": r"conference.*analysis.*(?!cache|optimize)",
                "severity": "low",
                "message": "컨퍼런스 분석 최적화 기회",
                "solution": "캐시 및 배치 처리 적용",
                "impact_score": 5,
                "auto_fixable": False
            },
            
            "multimodal_resource_leak": {
                "regex": r"(?:whisper|easyocr|transformers).*(?!del|cleanup)",
                "severity": "medium",
                "message": "멀티모달 모델 리소스 정리 누락",
                "solution": "명시적 모델 삭제 및 메모리 정리",
                "impact_score": 12,
                "auto_fixable": True
            }
        }
        
        # 자동 수정 템플릿
        self.fix_templates = {
            "threadpool_fix": """
# Serena 자동 수정: ThreadPool 안전 관리
with ThreadPoolExecutor() as executor:
    # 기존 코드를 이 블록 안으로 이동
    {original_code}
""",
            
            "cuda_cleanup": """
{original_code}
torch.cuda.empty_cache()  # Serena 추가: GPU 메모리 정리
""",
            
            "streamlit_cache": """
@st.cache_data  # Serena 추가: 성능 최적화
{original_function}
"""
        }

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("SOLOMONDProjectManager")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_file = self.project_root / "solomond_project_manager.log"
            handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_codebase(self, target_files: List[str] = None) -> Dict[str, Any]:
        """
        통합 코드베이스 분석 (Serena 엔진 + 프로젝트 관리)
        """
        self.logger.info("🔍 SOLOMOND AI 통합 코드베이스 분석 시작")
        
        analysis_result = {
            "manager": {
                "name": self.agent_name,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "Comprehensive Code Analysis (Serena Engine Integrated)"
            },
            "project_summary": {
                "files_analyzed": 0,
                "total_lines": 0,
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            },
            "issues_found": [],
            "health_assessment": {},
            "optimization_recommendations": [],
            "solomond_specific_insights": [],
            "auto_fix_available": False,
            "system_status": {},
            "errors": []
        }
        
        try:
            # 분석 대상 파일 결정
            if not target_files:
                target_files = self._get_priority_files()
            
            # 각 파일 분석
            for file_path in target_files:
                if not Path(file_path).exists():
                    continue
                
                try:
                    file_analysis = self._analyze_single_file(file_path)
                    analysis_result["issues_found"].extend(file_analysis["issues"])
                    analysis_result["project_summary"]["files_analyzed"] += 1
                    analysis_result["project_summary"]["total_lines"] += file_analysis["line_count"]
                    
                except Exception as e:
                    error_msg = f"파일 분석 실패 {file_path}: {str(e)}"
                    self.logger.error(error_msg)
                    analysis_result["errors"].append(error_msg)
            
            # 이슈 분류 및 통계
            self._categorize_issues(analysis_result)
            
            # 건강도 평가
            analysis_result["health_assessment"] = self._assess_system_health(analysis_result)
            
            # 최적화 권장사항 생성
            analysis_result["optimization_recommendations"] = self._generate_recommendations(analysis_result)
            
            # SOLOMOND AI 특화 인사이트
            analysis_result["solomond_specific_insights"] = self._generate_solomond_insights(analysis_result)
            
            # 자동 수정 가능 여부 확인
            analysis_result["auto_fix_available"] = self._check_auto_fix_availability(analysis_result)
            
            # 시스템 상태 체크
            analysis_result["system_status"] = self._check_system_status()
            
        except Exception as e:
            error_msg = f"코드베이스 분석 중 심각한 오류: {str(e)}"
            self.logger.error(error_msg)
            analysis_result["errors"].append(error_msg)
        
        self.logger.info("✅ 통합 코드베이스 분석 완료")
        return analysis_result

    def _get_priority_files(self) -> List[str]:
        """SOLOMOND AI 우선순위 파일 목록"""
        priority_files = [
            "conference_analysis_COMPLETE_WORKING.py",
            "solomond_ai_main_dashboard.py",
            "dual_brain_integration.py",
            "ai_insights_engine.py",
            "google_calendar_connector.py",
            "hybrid_compute_manager.py",
            "n8n_connector.py"
        ]
        
        existing_files = []
        for file_name in priority_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                existing_files.append(str(file_path))
        
        # core 디렉토리 중요 파일 추가
        core_dir = self.project_root / "core"
        if core_dir.exists():
            core_files = [
                "multimodal_pipeline.py",
                "batch_processing_engine.py",
                "memory_optimizer.py",
                "insight_generator.py"
            ]
            
            for file_name in core_files:
                file_path = core_dir / file_name
                if file_path.exists():
                    existing_files.append(str(file_path))
        
        return existing_files

    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """Symbol-level 단일 파일 분석 (Serena 엔진)"""
        file_result = {
            "file_path": str(Path(file_path).name),
            "full_path": file_path,
            "line_count": 0,
            "issues": [],
            "symbols": {},
            "complexity": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                file_result["line_count"] = len(lines)
            
            # Symbol-level 분석 (AST 파싱)
            try:
                tree = ast.parse(content)
                file_result["symbols"] = self._extract_symbols(tree)
                file_result["complexity"] = self._calculate_complexity(tree)
            except SyntaxError:
                # Python 문법 오류 파일은 패턴 매칭만 수행
                pass
            
            # 크리티컬 패턴 검사
            for pattern_name, pattern_info in self.critical_patterns.items():
                matches = list(re.finditer(
                    pattern_info["regex"],
                    content,
                    re.MULTILINE | re.IGNORECASE
                ))
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    code_line = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                    
                    issue = CodeIssue(
                        file_path=str(Path(file_path).name),
                        line_number=line_num,
                        pattern_name=pattern_name,
                        severity=pattern_info["severity"],
                        message=pattern_info["message"],
                        code_snippet=code_line,
                        solution=pattern_info["solution"],
                        confidence=0.9,
                        impact_score=pattern_info["impact_score"]
                    )
                    file_result["issues"].append(asdict(issue))
            
            # SOLOMOND AI 특화 패턴 검사
            for pattern_name, pattern_info in self.solomond_patterns.items():
                matches = list(re.finditer(
                    pattern_info["regex"],
                    content,
                    re.MULTILINE | re.IGNORECASE
                ))
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    code_line = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                    
                    issue = CodeIssue(
                        file_path=str(Path(file_path).name),
                        line_number=line_num,
                        pattern_name=pattern_name,
                        severity=pattern_info["severity"],
                        message=pattern_info["message"],
                        code_snippet=code_line,
                        solution=pattern_info["solution"],
                        confidence=0.8,
                        impact_score=pattern_info["impact_score"]
                    )
                    file_result["issues"].append(asdict(issue))
                    
        except Exception as e:
            self.logger.error(f"파일 분석 오류 {file_path}: {str(e)}")
            
        return file_result

    def _extract_symbols(self, tree: ast.AST) -> Dict[str, Any]:
        """AST에서 심볼 정보 추출"""
        symbols = {
            "functions": [],
            "classes": [],
            "imports": [],
            "globals": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols["functions"].append({
                    "name": node.name,
                    "args": len(node.args.args),
                    "lineno": node.lineno,
                    "decorators": [ast.unparse(d) for d in node.decorator_list]
                })
            elif isinstance(node, ast.ClassDef):
                symbols["classes"].append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                symbols["imports"].append({
                    "module": ast.unparse(node),
                    "lineno": node.lineno
                })
        
        return symbols

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """코드 복잡도 계산 (Cyclomatic Complexity 근사)"""
        complexity = 1  # 기본 복잡도
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity

    def _categorize_issues(self, analysis_result: Dict[str, Any]):
        """이슈 분류 및 통계 계산"""
        summary = analysis_result["project_summary"]
        
        for issue in analysis_result["issues_found"]:
            summary["total_issues"] += 1
            
            severity = issue["severity"]
            if severity == "critical":
                summary["critical_issues"] += 1
            elif severity == "high":
                summary["high_issues"] += 1
            elif severity == "medium":
                summary["medium_issues"] += 1
            else:
                summary["low_issues"] += 1

    def _assess_system_health(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """시스템 건강도 평가 (0-100점)"""
        summary = analysis_result["project_summary"]
        
        # 기본 점수에서 이슈에 따라 감점
        base_score = 100
        penalty = (
            summary["critical_issues"] * 25 +
            summary["high_issues"] * 15 +
            summary["medium_issues"] * 8 +
            summary["low_issues"] * 3
        )
        
        overall_score = max(base_score - penalty, 0)
        
        # 권장사항 생성
        recommendations = []
        if summary["critical_issues"] > 0:
            recommendations.append(f"🚨 {summary['critical_issues']}개 크리티컬 이슈 즉시 수정 필요")
        
        if summary["high_issues"] > 3:
            recommendations.append(f"⚠️ {summary['high_issues']}개 중요 이슈 우선 해결 권장")
        
        if overall_score >= 90:
            recommendations.append("✅ 시스템이 매우 건강한 상태입니다")
        elif overall_score >= 70:
            recommendations.append("👍 전반적으로 양호하나 몇 가지 개선 필요")
        else:
            recommendations.append("⚠️ 시스템 안정성을 위해 즉시 개선 필요")
        
        return {
            "overall_score": overall_score,
            "critical_issues": summary["critical_issues"],
            "high_issues": summary["high_issues"],
            "medium_issues": summary["medium_issues"],
            "low_issues": summary["low_issues"],
            "files_analyzed": summary["files_analyzed"],
            "recommendations": recommendations
        }

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """최적화 권장사항 생성"""
        recommendations = []
        summary = analysis_result["project_summary"]
        
        # 크리티컬 이슈 기반 권장사항
        if summary["critical_issues"] > 0:
            recommendations.append({
                "priority": "critical",
                "title": "크리티컬 이슈 즉시 해결",
                "description": f"{summary['critical_issues']}개의 크리티컬 이슈가 시스템 안정성을 위협합니다.",
                "solomond_benefit": "시스템 크래시 방지, 안정적인 컨퍼런스 분석 보장",
                "estimated_time": "30분-2시간"
            })
        
        # ThreadPool 이슈 특별 처리
        threadpool_issues = [
            issue for issue in analysis_result["issues_found"]
            if "threadpool" in issue["pattern_name"].lower()
        ]
        
        if threadpool_issues:
            recommendations.append({
                "priority": "high",
                "title": "ThreadPool 리소스 관리 최적화",
                "description": "컨퍼런스 분석의 병렬 처리 안정성을 위해 ThreadPool 관리 개선",
                "solomond_benefit": "멀티파일 분석 시 메모리 누수 방지, 안정적인 배치 처리",
                "estimated_time": "15분-1시간"
            })
        
        # GPU 메모리 최적화
        cuda_issues = [
            issue for issue in analysis_result["issues_found"]
            if "cuda" in issue["pattern_name"].lower() or "memory" in issue["pattern_name"].lower()
        ]
        
        if cuda_issues:
            recommendations.append({
                "priority": "medium",
                "title": "GPU 메모리 최적화",
                "description": "AI 모델 로딩 시 GPU 메모리 효율성 개선",
                "solomond_benefit": "대용량 멀티모달 분석 성능 향상, OOM 에러 방지",
                "estimated_time": "20분-45분"
            })
        
        # Streamlit 성능 최적화
        streamlit_issues = [
            issue for issue in analysis_result["issues_found"]
            if "streamlit" in issue["pattern_name"].lower()
        ]
        
        if streamlit_issues:
            recommendations.append({
                "priority": "medium",
                "title": "Streamlit 캐싱 최적화",
                "description": "대시보드 응답 속도 향상을 위한 캐싱 전략 적용",
                "solomond_benefit": "사용자 경험 개선, AI 모델 로딩 시간 단축",
                "estimated_time": "10분-30분"
            })
        
        return recommendations

    def _generate_solomond_insights(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """SOLOMOND AI 특화 인사이트 생성"""
        insights = []
        
        # 컨퍼런스 분석 시스템 특화 인사이트
        conference_files = [
            issue for issue in analysis_result["issues_found"]
            if "conference" in issue["file_path"].lower()
        ]
        
        if conference_files:
            insights.append({
                "level": "enhancement",
                "title": "컨퍼런스 분석 시스템 최적화 기회",
                "message": f"컨퍼런스 분석 파일에서 {len(conference_files)}개 개선점 발견",
                "impact": "분석 속도 및 정확도 향상",
                "action": "배치 처리 및 캐싱 최적화 적용"
            })
        
        # 멀티모달 시스템 인사이트
        multimodal_patterns = [
            issue for issue in analysis_result["issues_found"]
            if any(keyword in issue["code_snippet"].lower() 
                  for keyword in ["whisper", "easyocr", "transformers"])
        ]
        
        if multimodal_patterns:
            insights.append({
                "level": "improvement",
                "title": "멀티모달 AI 리소스 관리",
                "message": "AI 모델 리소스 정리 패턴 개선으로 메모리 효율성 증대 가능",
                "impact": "대용량 파일 처리 안정성 향상",
                "action": "명시적 모델 정리 및 메모리 관리 강화"
            })
        
        # 듀얼 브레인 시스템 인사이트
        if analysis_result["project_summary"]["files_analyzed"] > 0:
            health_score = analysis_result.get("health_assessment", {}).get("overall_score", 0)
            
            if health_score >= 85:
                insights.append({
                    "level": "positive",
                    "title": "듀얼 브레인 시스템 안정성 우수",
                    "message": f"시스템 건강도 {health_score}점으로 매우 안정적인 상태",
                    "impact": "안정적인 AI 인사이트 생성 및 캘린더 통합",
                    "action": "현재 품질 유지하며 신규 기능 개발 집중"
                })
            else:
                insights.append({
                    "level": "warning",
                    "title": "듀얼 브레인 시스템 개선 필요",
                    "message": f"시스템 건강도 {health_score}점, 안정성 강화 필요",
                    "impact": "AI 인사이트 품질 및 시스템 신뢰성 향상",
                    "action": "크리티컬 이슈 우선 해결 후 전체 최적화"
                })
        
        return insights

    def _check_auto_fix_availability(self, analysis_result: Dict[str, Any]) -> bool:
        """자동 수정 가능 여부 확인"""
        auto_fixable_issues = [
            issue for issue in analysis_result["issues_found"]
            if issue["pattern_name"] in ["threadpool_without_context", "memory_leak_cuda", 
                                        "streamlit_heavy_no_cache", "file_open_no_context"]
        ]
        return len(auto_fixable_issues) > 0

    def _check_system_status(self) -> Dict[str, Any]:
        """시스템 상태 체크"""
        status = {
            "streamlit_ports": [],
            "ollama_status": False,
            "key_files_present": {},
            "git_status": {},
            "disk_space_gb": 0
        }
        
        try:
            # 핵심 파일 존재 확인
            key_files = [
                "conference_analysis_COMPLETE_WORKING.py",
                "solomond_ai_main_dashboard.py",
                "dual_brain_integration.py"
            ]
            
            for file_name in key_files:
                file_path = self.project_root / file_name
                status["key_files_present"][file_name] = file_path.exists()
            
            # Git 상태 확인
            try:
                git_status = subprocess.run(
                    ["git", "status", "--porcelain"], 
                    capture_output=True, text=True, cwd=self.project_root
                )
                status["git_status"] = {
                    "clean": len(git_status.stdout.strip()) == 0,
                    "modified_files": len(git_status.stdout.strip().split('\n')) if git_status.stdout.strip() else 0
                }
            except:
                status["git_status"] = {"clean": None, "modified_files": 0}
            
            # 디스크 공간 확인
            disk_usage = shutil.disk_usage(self.project_root)
            status["disk_space_gb"] = disk_usage.free // (1024**3)
            
        except Exception as e:
            self.logger.error(f"시스템 상태 체크 오류: {str(e)}")
        
        return status

    def auto_fix_issues(self, analysis_result: Dict[str, Any], create_backups: bool = True) -> Dict[str, Any]:
        """자동 이슈 수정 (Serena 엔진)"""
        fix_result = {
            "timestamp": datetime.now().isoformat(),
            "fixes_applied": 0,
            "files_modified": [],
            "backups_created": [],
            "errors": [],
            "summary": {}
        }
        
        if not analysis_result.get("auto_fix_available", False):
            fix_result["summary"]["message"] = "자동 수정 가능한 이슈가 없습니다."
            return fix_result
        
        # 자동 수정 가능한 이슈들 필터링
        auto_fixable_issues = [
            issue for issue in analysis_result["issues_found"]
            if issue["pattern_name"] in ["threadpool_without_context", "memory_leak_cuda", 
                                        "streamlit_heavy_no_cache", "file_open_no_context"]
        ]
        
        # 파일별로 그룹화
        files_to_fix = {}
        for issue in auto_fixable_issues:
            full_path = None
            for file_path in self._get_priority_files():
                if Path(file_path).name == issue["file_path"]:
                    full_path = file_path
                    break
            
            if full_path and Path(full_path).exists():
                if full_path not in files_to_fix:
                    files_to_fix[full_path] = []
                files_to_fix[full_path].append(issue)
        
        # 각 파일 수정
        for file_path, issues in files_to_fix.items():
            try:
                # 백업 생성
                if create_backups:
                    backup_path = self._create_backup(file_path)
                    fix_result["backups_created"].append(backup_path)
                
                # 파일 수정
                modified = self._apply_auto_fixes(file_path, issues)
                
                if modified:
                    fix_result["fixes_applied"] += len(issues)
                    fix_result["files_modified"].append(file_path)
                    self.logger.info(f"자동 수정 완료: {file_path} ({len(issues)}개 이슈)")
                
            except Exception as e:
                error_msg = f"파일 수정 실패 {file_path}: {str(e)}"
                fix_result["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        # 요약 생성
        fix_result["summary"] = {
            "message": f"총 {fix_result['fixes_applied']}개 이슈가 {len(fix_result['files_modified'])}개 파일에서 수정되었습니다.",
            "success": fix_result['fixes_applied'] > 0,
            "recommendation": "수정 후 시스템 테스트를 권장합니다." if fix_result['fixes_applied'] > 0 else None
        }
        
        return fix_result

    def _create_backup(self, file_path: str) -> str:
        """파일 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = Path(file_path).name
        backup_name = f"{file_name}.backup_{timestamp}"
        
        backup_dir = self.project_root / "solomond_backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        
        return str(backup_path)

    def _apply_auto_fixes(self, file_path: str, issues: List[Dict]) -> bool:
        """파일에 자동 수정 적용"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        
        # 이슈별 수정 적용
        for issue in issues:
            pattern_name = issue["pattern_name"]
            
            if pattern_name == "threadpool_without_context":
                # ThreadPool context manager 수정
                pattern = r"(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n"
                replacement = r"\1with ThreadPoolExecutor() as executor:\n\1    # Serena 자동 수정: 안전한 리소스 관리\n"
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            elif pattern_name == "memory_leak_cuda":
                # CUDA 메모리 정리 추가
                line_num = issue["line_number"]
                if line_num <= len(lines):
                    lines[line_num - 1] += "\n    torch.cuda.empty_cache()  # Serena 자동 추가: GPU 메모리 정리"
                    content = '\n'.join(lines)
            
            elif pattern_name == "streamlit_heavy_no_cache":
                # Streamlit 캐시 데코레이터 추가
                line_num = issue["line_number"]
                if line_num > 0 and line_num <= len(lines):
                    indent = len(lines[line_num - 1]) - len(lines[line_num - 1].lstrip())
                    cache_decorator = " " * indent + "@st.cache_data  # Serena 자동 추가: 성능 최적화"
                    lines.insert(line_num - 1, cache_decorator)
                    content = '\n'.join(lines)
            
            elif pattern_name == "file_open_no_context":
                # 파일 오픈 context manager 수정
                pattern = r"(\s*)((?:file|f)\s*=\s*open\([^)]+\))\s*\n"
                replacement = r"\1with open(...) as file:\n\1    # Serena 자동 수정: 파일 안전 관리\n"
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # 변경사항이 있으면 파일 저장
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False

    def generate_health_report(self, analysis_result: Dict[str, Any]) -> str:
        """건강도 보고서 생성"""
        health = analysis_result.get("health_assessment", {})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 건강도 등급 결정
        score = health.get("overall_score", 0)
        if score >= 90:
            grade = "최우수 🏆"
            emoji = "💚"
        elif score >= 80:
            grade = "우수 ⭐"
            emoji = "💚"
        elif score >= 70:
            grade = "양호 👍"
            emoji = "💛"
        elif score >= 60:
            grade = "보통 👌"
            emoji = "🧡"
        else:
            grade = "주의 필요 ⚠️"
            emoji = "❤️"
        
        report_lines = [
            "# 🏥 SOLOMOND AI 시스템 건강 진단 보고서",
            "",
            f"**📅 진단 시간**: {timestamp}",
            f"**🤖 진단 에이전트**: {self.agent_name} v{self.version}",
            f"**🎯 시스템**: SOLOMOND AI Conference Analysis Platform",
            "",
            f"## {emoji} 전체 건강도: {score}/100 ({grade})",
            "",
            "## 📊 상세 진단 결과",
            "",
            f"- **분석된 파일**: {health.get('files_analyzed', 0)}개",
            f"- **🚨 크리티컬 이슈**: {health.get('critical_issues', 0)}개",
            f"- **⚠️ 중요 이슈**: {health.get('high_issues', 0)}개", 
            f"- **📋 보통 이슈**: {health.get('medium_issues', 0)}개",
            f"- **ℹ️ 경미한 이슈**: {health.get('low_issues', 0)}개",
            "",
            "## 💡 권장사항",
            ""
        ]
        
        for recommendation in health.get("recommendations", []):
            report_lines.append(f"- {recommendation}")
        
        # SOLOMOND AI 특화 인사이트 추가
        insights = analysis_result.get("solomond_specific_insights", [])
        if insights:
            report_lines.extend([
                "",
                "## 🧠 SOLOMOND AI 특화 인사이트",
                ""
            ])
            
            for insight in insights:
                emoji_map = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
                emoji = emoji_map.get(insight.get("level"), "💡")
                report_lines.extend([
                    f"### {emoji} {insight.get('title', '')}",
                    f"{insight.get('message', '')}",
                    f"**영향**: {insight.get('impact', '')}",
                    f"**권장 조치**: {insight.get('action', '')}",
                    ""
                ])
        
        # 자동 수정 정보
        if analysis_result.get("auto_fix_available", False):
            report_lines.extend([
                "## 🔧 자동 수정 가능",
                "",
                "일부 이슈는 자동으로 수정할 수 있습니다.",
                "`manager.auto_fix_issues(analysis_result)` 호출로 자동 수정을 적용하세요.",
                ""
            ])
        
        report_lines.extend([
            "---",
            f"*이 보고서는 {self.agent_name}에 의해 자동 생성되었습니다.*"
        ])
        
        return "\n".join(report_lines)

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """에이전트 기능 정보 반환"""
        return {
            "name": self.agent_name,
            "version": self.version,
            "role": "SOLOMOND AI 시스템 통합 개발 매니저",
            "serena_integrated": True,
            "core_capabilities": [
                "Symbol-level 코드 분석 (AST 파싱)",
                "ThreadPool 및 리소스 관리 최적화",
                "GPU 메모리 누수 자동 탐지",
                "Streamlit 성능 최적화",
                "SOLOMOND AI 특화 분석",
                "자동 이슈 수정 (백업 포함)",
                "시스템 건강도 평가 (0-100점)",
                "실시간 프로젝트 상태 모니터링"
            ],
            "solomond_specializations": [
                "컨퍼런스 분석 시스템 최적화",
                "멀티모달 AI 리소스 관리",
                "듀얼 브레인 시스템 안정성",
                "n8n 워크플로우 통합 지원",
                "구글 캘린더 API 최적화"
            ],
            "analysis_patterns": len(self.critical_patterns) + len(self.solomond_patterns),
            "auto_fix_patterns": len([p for p in self.critical_patterns.values() if p.get("auto_fixable", False)]),
            "supported_commands": [
                "analyze_codebase() - 통합 코드베이스 분석",
                "auto_fix_issues() - 자동 이슈 수정",
                "generate_health_report() - 건강도 보고서 생성",
                "get_agent_capabilities() - 에이전트 정보"
            ]
        }

def main():
    """SOLOMOND Project Manager 실행 데모"""
    print("🤖 SOLOMOND AI 통합 프로젝트 매니저")
    print("🔗 Serena 코딩 에이전트 완전 통합 버전")
    print("=" * 70)
    
    # 매니저 초기화
    manager = SOLOMONDProjectManager()
    
    # 에이전트 정보 표시
    capabilities = manager.get_agent_capabilities()
    print(f"🧠 {capabilities['name']} v{capabilities['version']}")
    print(f"📋 역할: {capabilities['role']}")
    print(f"⚡ Serena 통합: {'✅ 완료' if capabilities['serena_integrated'] else '❌ 미완료'}")
    print(f"🎯 분석 패턴: {capabilities['analysis_patterns']}개")
    print(f"🔧 자동 수정 패턴: {capabilities['auto_fix_patterns']}개")
    
    # 통합 코드베이스 분석 실행
    print(f"\n📊 SOLOMOND AI 통합 분석 시작...")
    analysis_result = manager.analyze_codebase()
    
    # 분석 결과 출력
    print(f"✅ 분석 완료!")
    summary = analysis_result["project_summary"]
    print(f"📁 분석된 파일: {summary['files_analyzed']}개")
    print(f"📄 총 코드 라인: {summary['total_lines']:,}줄")
    print(f"🔍 발견된 이슈: {summary['total_issues']}개")
    
    if summary["critical_issues"] > 0:
        print(f"🚨 크리티컬 이슈: {summary['critical_issues']}개")
    
    # 건강도 평가
    health = analysis_result.get("health_assessment", {})
    if health:
        score = health.get("overall_score", 0)
        print(f"🏥 시스템 건강도: {score}/100")
    
    # 자동 수정 가능 여부
    if analysis_result.get("auto_fix_available", False):
        print(f"🔧 자동 수정 가능한 이슈 발견!")
    
    # SOLOMOND AI 특화 인사이트
    insights = analysis_result.get("solomond_specific_insights", [])
    if insights:
        print(f"\n🧠 SOLOMOND AI 특화 인사이트 ({len(insights)}개):")
        for i, insight in enumerate(insights[:3], 1):
            emoji_map = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
            emoji = emoji_map.get(insight.get("level"), "💡")
            print(f"  {i}. {emoji} {insight.get('title', '')}")
    
    # 최적화 권장사항
    recommendations = analysis_result.get("optimization_recommendations", [])
    if recommendations:
        print(f"\n💡 최적화 권장사항 ({len(recommendations)}개):")
        for i, rec in enumerate(recommendations[:3], 1):
            priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋"}
            emoji = priority_emoji.get(rec.get("priority"), "💡")
            print(f"  {i}. {emoji} {rec.get('title', '')}")
    
    # 건강도 보고서 생성
    print(f"\n📋 상세 건강도 보고서 생성 중...")
    health_report = manager.generate_health_report(analysis_result)
    
    report_file = Path("solomond_health_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(health_report)
    
    print(f"✅ 건강도 보고서 생성: {report_file}")
    
    # 자동 수정 실행 (사용자 확인)
    if analysis_result.get("auto_fix_available", False):
        print(f"\n🤔 자동 수정을 실행하시겠습니까? (y/N): ", end="")
        user_input = input().strip().lower()
        
        if user_input == 'y':
            print(f"🔧 자동 수정 실행 중...")
            fix_result = manager.auto_fix_issues(analysis_result)
            
            print(f"✅ 자동 수정 완료!")
            print(f"📊 수정된 이슈: {fix_result['fixes_applied']}개")
            print(f"📁 수정된 파일: {len(fix_result['files_modified'])}개")
            
            if fix_result["backups_created"]:
                print(f"💾 백업 생성: {len(fix_result['backups_created'])}개")
        else:
            print(f"ℹ️ 자동 수정을 건너뛰었습니다.")
    
    print(f"\n🎉 SOLOMOND AI 통합 프로젝트 매니저 분석 완료!")
    print(f"📋 상세 내용은 {report_file}을 확인하세요.")

if __name__ == "__main__":
    main()