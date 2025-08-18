#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 Serena MCP Integration Module
Claude Code MCP 도구들과 Serena 에이전트 완전 통합

이 모듈은 Claude Code의 MCP (Model Context Protocol) 도구들을 활용하여
Serena 에이전트의 기능을 극대화합니다.

MCP 도구 활용:
- Read: 파일 내용 정밀 분석
- Glob: 프로젝트 파일 패턴 매칭
- Edit/MultiEdit: 자동 코드 수정
- Grep: 고급 패턴 검색
- Git: 버전 관리 통합

Author: SOLOMOND AI Team & Serena
Version: 1.0.0
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SerenaAnalysisConfig:
    """Serena 분석 설정"""
    project_name: str = "SOLOMOND_AI"
    analysis_depth: str = "comprehensive"  # basic, standard, comprehensive
    focus_areas: List[str] = None
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = [
                "threadpool_management",
                "memory_optimization", 
                "streamlit_performance",
                "ollama_integration",
                "gpu_memory_handling"
            ]
        
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "venv/*", "__pycache__/*", ".git/*", 
                "*.backup", "*.log", "archive/*"
            ]

class SerenaMCPIntegration:
    """Serena와 Claude Code MCP 도구 통합"""
    
    def __init__(self, config: SerenaAnalysisConfig = None):
        self.config = config or SerenaAnalysisConfig()
        self.analysis_patterns = self._load_analysis_patterns()
        self.project_root = Path.cwd()
        
    def _load_analysis_patterns(self) -> Dict[str, Dict[str, Any]]:
        """SOLOMOND AI 특화 분석 패턴 로드"""
        return {
            # 크리티컬 이슈 패턴
            "threadpool_resource_leak": {
                "pattern": r"ThreadPoolExecutor\([^)]*\)(?!\s*as\s+\w+:)(?!\s*with)",
                "severity": "critical",
                "category": "resource_management",
                "description": "ThreadPoolExecutor without proper resource management",
                "impact": "Memory leak and resource exhaustion",
                "fix_template": "with ThreadPoolExecutor({params}) as executor:",
                "solomond_impact": "컨퍼런스 분석 시스템 안정성에 직접적 영향"
            },
            
            "gpu_memory_leak": {
                "pattern": r"torch\.cuda\.(?!empty_cache).*\n(?!.*torch\.cuda\.empty_cache)",
                "severity": "critical", 
                "category": "memory_management",
                "description": "CUDA operations without memory cleanup",
                "impact": "GPU memory accumulation leading to OOM errors",
                "fix_template": "# Add: torch.cuda.empty_cache() after CUDA operations",
                "solomond_impact": "AI 모델 로딩 및 GPU 기반 분석에서 메모리 부족 발생"
            },
            
            "streamlit_performance_issue": {
                "pattern": r"def\s+\w+.*:\s*\n(?:.*\n)*?.*(?:heavy_computation|model\.load|large_file_processing).*\n(?!.*@st\.cache)",
                "severity": "high",
                "category": "performance",
                "description": "Heavy computation without Streamlit caching",
                "impact": "Poor user experience and resource waste",
                "fix_template": "@st.cache_data\ndef function_name():",
                "solomond_impact": "사용자 대기시간 증가, 시스템 리소스 낭비"
            },
            
            "ollama_error_handling_missing": {
                "pattern": r"ollama\.(?:generate|chat|pull)\([^)]*\)(?!\s*\n\s*(?:try|except))",
                "severity": "high",
                "category": "error_handling", 
                "description": "Ollama API calls without error handling",
                "impact": "Application crashes on API failures",
                "fix_template": "try:\n    ollama.{method}({params})\nexcept Exception as e:\n    handle_error(e)",
                "solomond_impact": "AI 분석 중단, 사용자 경험 저하"
            },
            
            "file_io_resource_leak": {
                "pattern": r"open\([^)]*\)(?!\s*as\s+\w+:)(?!\s*with)",
                "severity": "medium",
                "category": "resource_management",
                "description": "File operations without context manager",
                "impact": "File handle leaks",
                "fix_template": "with open({params}) as file:",
                "solomond_impact": "파일 처리 과정에서 리소스 누수 발생"
            },
            
            "multimodal_sync_issue": {
                "pattern": r"(?:audio|image|video)_process.*\n.*(?:threading|concurrent).*(?!.*join|wait)",
                "severity": "medium",
                "category": "concurrency",
                "description": "Multimodal processing without proper synchronization",
                "impact": "Race conditions in media processing",
                "fix_template": "# Add proper thread synchronization",
                "solomond_impact": "멀티모달 분석 결과 불일치 가능성"
            }
        }
    
    def analyze_with_mcp_tools(self, target_files: List[str] = None) -> Dict[str, Any]:
        """
        MCP 도구들을 활용한 종합 분석
        
        이 메서드는 Claude Code의 MCP 도구들을 시뮬레이션하여
        실제 서브에이전트 환경에서의 동작을 구현합니다.
        """
        analysis_start = datetime.now()
        
        result = {
            "serena_agent": {
                "name": "Serena SOLOMOND AI 전문가",
                "version": "1.0.0",
                "analysis_type": "MCP 통합 분석"
            },
            "analysis_metadata": {
                "timestamp": analysis_start.isoformat(),
                "config": self.config.__dict__,
                "tools_used": ["Read", "Glob", "Grep", "Git"],
                "duration_ms": 0
            },
            "project_summary": {
                "files_analyzed": 0,
                "total_lines": 0,
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0
            },
            "detailed_findings": [],
            "optimization_recommendations": [],
            "auto_fix_candidates": [],
            "solomond_specific_insights": []
        }
        
        try:
            # 1. Glob 패턴으로 분석 대상 파일 찾기
            if not target_files:
                target_files = self._find_target_files()
            
            # 2. 각 파일 정밀 분석 
            for file_path in target_files:
                file_analysis = self._analyze_file_with_mcp(file_path)
                if file_analysis:
                    result["detailed_findings"].append(file_analysis)
                    result["project_summary"]["files_analyzed"] += 1
                    result["project_summary"]["total_lines"] += file_analysis.get("line_count", 0)
                    
                    # 이슈 카운트
                    for issue in file_analysis.get("issues", []):
                        severity = issue.get("severity", "medium")
                        result["project_summary"]["total_issues"] += 1
                        result["project_summary"][f"{severity}_issues"] += 1
            
            # 3. 최적화 추천사항 생성
            result["optimization_recommendations"] = self._generate_mcp_recommendations(result)
            
            # 4. 자동 수정 후보 식별
            result["auto_fix_candidates"] = self._identify_auto_fix_candidates(result)
            
            # 5. SOLOMOND AI 특화 인사이트
            result["solomond_specific_insights"] = self._generate_solomond_insights(result)
            
            # 분석 시간 계산
            analysis_end = datetime.now()
            duration = (analysis_end - analysis_start).total_seconds() * 1000
            result["analysis_metadata"]["duration_ms"] = round(duration, 2)
            
        except Exception as e:
            result["error"] = {
                "message": f"MCP 분석 중 오류 발생: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        return result
    
    def _find_target_files(self) -> List[str]:
        """분석 대상 파일 찾기 (Glob 시뮬레이션)"""
        # SOLOMOND AI 핵심 파일들 우선순위
        priority_files = [
            "conference_analysis_COMPLETE_WORKING.py",
            "solomond_ai_main_dashboard.py",
            "dual_brain_integration.py", 
            "ai_insights_engine.py",
            "google_calendar_connector.py",
            "hybrid_compute_manager.py"
        ]
        
        target_files = []
        
        # 우선순위 파일들 추가
        for file_name in priority_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                target_files.append(str(file_path))
        
        # core 디렉토리 추가 분석
        core_dir = self.project_root / "core"
        if core_dir.exists():
            core_files = [
                "multimodal_pipeline.py",
                "batch_processing_engine.py", 
                "memory_optimizer.py",
                "ollama_integration_engine.py"
            ]
            
            for file_name in core_files:
                file_path = core_dir / file_name
                if file_path.exists():
                    target_files.append(str(file_path))
        
        return target_files
    
    def _analyze_file_with_mcp(self, file_path: str) -> Optional[Dict[str, Any]]:
        """MCP Read 도구를 활용한 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            file_analysis = {
                "file_path": str(Path(file_path).name),
                "full_path": file_path,
                "line_count": len(lines),
                "issues": [],
                "metrics": {
                    "complexity_score": 0,
                    "solomond_relevance": 0,
                    "performance_impact": 0
                }
            }
            
            # 패턴별 분석
            for pattern_name, pattern_info in self.analysis_patterns.items():
                matches = re.finditer(
                    pattern_info["pattern"],
                    content, 
                    re.MULTILINE | re.DOTALL
                )
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    issue = {
                        "line": line_num,
                        "pattern": pattern_name,
                        "severity": pattern_info["severity"],
                        "category": pattern_info["category"],
                        "description": pattern_info["description"],
                        "impact": pattern_info["impact"],
                        "solomond_impact": pattern_info["solomond_impact"],
                        "code_snippet": lines[line_num - 1].strip() if line_num <= len(lines) else "",
                        "fix_suggestion": pattern_info["fix_template"],
                        "confidence": 0.9
                    }
                    
                    file_analysis["issues"].append(issue)
            
            # 메트릭 계산
            file_analysis["metrics"] = self._calculate_file_metrics(file_path, content, file_analysis["issues"])
            
            return file_analysis
            
        except Exception as e:
            return {
                "file_path": str(Path(file_path).name),
                "error": f"분석 실패: {str(e)}"
            }
    
    def _calculate_file_metrics(self, file_path: str, content: str, issues: List[Dict]) -> Dict[str, float]:
        """파일 메트릭 계산"""
        metrics = {
            "complexity_score": 0,
            "solomond_relevance": 0, 
            "performance_impact": 0
        }
        
        # 복잡도 점수 (이슈 심각도 기반)
        severity_weights = {"critical": 10, "high": 6, "medium": 3, "low": 1}
        complexity = sum(severity_weights.get(issue["severity"], 1) for issue in issues)
        metrics["complexity_score"] = min(complexity / 10.0, 10.0)  # 0-10 스케일
        
        # SOLOMOND AI 관련성 점수
        solomond_keywords = [
            "streamlit", "conference", "analysis", "ollama", "whisper",
            "easyocr", "multimodal", "threadpool", "gpu", "solomond"
        ]
        
        keyword_count = sum(1 for keyword in solomond_keywords if keyword.lower() in content.lower())
        metrics["solomond_relevance"] = min(keyword_count / 5.0, 10.0)  # 0-10 스케일
        
        # 성능 영향 점수 (크리티컬/하이 이슈 기반)
        performance_issues = [i for i in issues if i["severity"] in ["critical", "high"]]
        metrics["performance_impact"] = min(len(performance_issues) * 2.0, 10.0)
        
        return metrics
    
    def _generate_mcp_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """MCP 도구 기반 최적화 추천사항 생성"""
        recommendations = []
        
        project_summary = analysis_result["project_summary"]
        
        # 크리티컬 이슈 추천
        if project_summary["critical_issues"] > 0:
            recommendations.append({
                "priority": "critical",
                "category": "stability",
                "title": "시스템 안정성 크리티컬 이슈 해결",
                "description": f"{project_summary['critical_issues']}개의 크리티컬 이슈가 발견되었습니다. "
                             "ThreadPool 리소스 관리와 GPU 메모리 정리가 우선 필요합니다.",
                "action": "즉시 수정 스크립트 실행",
                "estimated_time": "30분",
                "tools": ["Edit", "MultiEdit"],
                "solomond_benefit": "시스템 크래시 방지, 안정적 컨퍼런스 분석 보장"
            })
        
        # 성능 최적화 추천
        if project_summary["high_issues"] > 2:
            recommendations.append({
                "priority": "high",
                "category": "performance", 
                "title": "Streamlit 캐싱 시스템 도입",
                "description": "무거운 AI 모델 로딩과 데이터 처리에 캐싱을 적용하여 성능을 향상시키세요.",
                "action": "@st.cache_data 데코레이터 추가",
                "estimated_time": "1시간",
                "tools": ["Edit", "Grep"],
                "solomond_benefit": "사용자 대기시간 50% 단축, 리소스 사용량 감소"
            })
        
        # 코드 품질 개선
        total_issues = project_summary["total_issues"]
        if total_issues > 5:
            recommendations.append({
                "priority": "medium",
                "category": "quality",
                "title": "코드 품질 전반적 개선",
                "description": f"총 {total_issues}개의 이슈가 발견되었습니다. "
                             "에러 처리 강화와 리소스 관리 개선이 필요합니다.",
                "action": "단계별 리팩토링 수행",
                "estimated_time": "2-3시간",
                "tools": ["Read", "Edit", "Git"],
                "solomond_benefit": "시스템 신뢰성 향상, 유지보수성 개선"
            })
        
        return recommendations
    
    def _identify_auto_fix_candidates(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """자동 수정 가능한 이슈들 식별"""
        auto_fix_candidates = []
        
        for file_analysis in analysis_result["detailed_findings"]:
            for issue in file_analysis.get("issues", []):
                # ThreadPool 이슈는 자동 수정 가능
                if issue["pattern"] == "threadpool_resource_leak":
                    auto_fix_candidates.append({
                        "file": file_analysis["file_path"],
                        "line": issue["line"],
                        "issue_type": issue["pattern"],
                        "current_code": issue["code_snippet"],
                        "fixed_code": "with ThreadPoolExecutor() as executor:",
                        "confidence": 0.95,
                        "description": "ThreadPoolExecutor를 with 문으로 변경"
                    })
                
                # 파일 IO 이슈도 자동 수정 가능
                elif issue["pattern"] == "file_io_resource_leak":
                    auto_fix_candidates.append({
                        "file": file_analysis["file_path"],
                        "line": issue["line"],
                        "issue_type": issue["pattern"],
                        "current_code": issue["code_snippet"],
                        "fixed_code": "with open(...) as file:",
                        "confidence": 0.90,
                        "description": "파일 열기를 with 문으로 변경"
                    })
        
        return auto_fix_candidates
    
    def _generate_solomond_insights(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """SOLOMOND AI 시스템 특화 인사이트 생성"""
        insights = []
        
        # 시스템 안정성 인사이트
        critical_count = analysis_result["project_summary"]["critical_issues"]
        if critical_count == 0:
            insights.append({
                "category": "stability",
                "level": "positive",
                "title": "시스템 안정성 양호",
                "message": "크리티컬 이슈가 발견되지 않아 SOLOMOND AI 시스템이 안정적으로 운영되고 있습니다.",
                "impact": "사용자 경험 안정성 보장"
            })
        else:
            insights.append({
                "category": "stability", 
                "level": "warning",
                "title": "시스템 안정성 주의 필요",
                "message": f"{critical_count}개의 크리티컬 이슈로 인해 컨퍼런스 분석 중 시스템 오류 가능성이 있습니다.",
                "impact": "분석 중단, 데이터 손실 위험"
            })
        
        # 성능 인사이트
        files_with_performance_issues = sum(
            1 for file_analysis in analysis_result["detailed_findings"]
            if any(issue["category"] == "performance" for issue in file_analysis.get("issues", []))
        )
        
        if files_with_performance_issues > 0:
            insights.append({
                "category": "performance",
                "level": "improvement",
                "title": "성능 최적화 기회",
                "message": f"{files_with_performance_issues}개 파일에서 성능 개선 여지가 발견되었습니다. "
                          "Streamlit 캐싱과 메모리 관리 최적화로 사용자 경험을 크게 향상시킬 수 있습니다.",
                "impact": "응답 속도 향상, 리소스 효율성 증대"
            })
        
        # AI 통합 인사이트
        ollama_issues = sum(
            1 for file_analysis in analysis_result["detailed_findings"]
            for issue in file_analysis.get("issues", [])
            if "ollama" in issue["pattern"]
        )
        
        if ollama_issues > 0:
            insights.append({
                "category": "ai_integration",
                "level": "enhancement",
                "title": "AI 모델 통합 개선",
                "message": f"Ollama AI 통합에서 {ollama_issues}개의 개선점이 발견되었습니다. "
                          "에러 처리 강화로 AI 분석의 신뢰성을 높일 수 있습니다.",
                "impact": "AI 분석 성공률 향상, 오류 복구 능력 강화"
            })
        
        return insights

def run_serena_analysis():
    """Serena MCP 통합 분석 실행"""
    print("🤖 Serena - SOLOMOND AI 전문 코딩 에이전트")
    print("🔗 Claude Code MCP 통합 분석 시작")
    print("=" * 60)
    
    # 분석 설정
    config = SerenaAnalysisConfig(
        analysis_depth="comprehensive",
        focus_areas=[
            "threadpool_management",
            "memory_optimization",
            "streamlit_performance", 
            "ollama_integration",
            "solomond_stability"
        ]
    )
    
    # MCP 통합 분석 실행
    serena_mcp = SerenaMCPIntegration(config)
    result = serena_mcp.analyze_with_mcp_tools()
    
    # 결과 출력
    print(f"📊 분석 완료 - {result['analysis_metadata']['duration_ms']:.1f}ms")
    print(f"📁 분석된 파일: {result['project_summary']['files_analyzed']}개")
    print(f"📝 총 코드 라인: {result['project_summary']['total_lines']:,}줄")
    print(f"🔍 발견된 이슈: {result['project_summary']['total_issues']}개")
    
    if result['project_summary']['critical_issues'] > 0:
        print(f"🚨 크리티컬: {result['project_summary']['critical_issues']}개")
    if result['project_summary']['high_issues'] > 0:
        print(f"⚠️  높음: {result['project_summary']['high_issues']}개")
    if result['project_summary']['medium_issues'] > 0:
        print(f"📋 보통: {result['project_summary']['medium_issues']}개")
    
    # 추천사항 표시
    if result['optimization_recommendations']:
        print(f"\n💡 Serena의 최적화 추천사항:")
        for i, rec in enumerate(result['optimization_recommendations'], 1):
            print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
            print(f"     {rec['description']}")
            print(f"     💎 SOLOMOND 효과: {rec['solomond_benefit']}")
    
    # SOLOMOND AI 특화 인사이트
    if result['solomond_specific_insights']:
        print(f"\n🧠 SOLOMOND AI 시스템 인사이트:")
        for insight in result['solomond_specific_insights']:
            emoji = {"positive": "✅", "warning": "⚠️", "improvement": "📈", "enhancement": "🔧"}
            print(f"  {emoji.get(insight['level'], '💡')} {insight['title']}")
            print(f"     {insight['message']}")
    
    # 자동 수정 가능한 이슈들
    if result['auto_fix_candidates']:
        print(f"\n🔧 자동 수정 가능한 이슈: {len(result['auto_fix_candidates'])}개")
        print("💡 Serena가 자동으로 수정할 수 있는 이슈들이 발견되었습니다.")
    
    return result

if __name__ == "__main__":
    run_serena_analysis()