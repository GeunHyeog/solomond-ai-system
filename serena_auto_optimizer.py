#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Serena Auto Optimizer
SOLOMOND AI 시스템 자동 최적화 엔진

이 모듈은 Serena 에이전트의 핵심 기능인 자동 이슈 탐지와 최적화를
Claude Code 환경에서 실행 가능한 형태로 구현합니다.

핵심 기능:
1. 실시간 코드 분석 및 이슈 탐지
2. 자동 수정 스크립트 생성 및 실행
3. 성능 최적화 제안 및 적용
4. SOLOMOND AI 시스템 건강도 모니터링

Author: Serena & SOLOMOND AI Team
Version: 1.0.0
Created: 2025-08-17
"""

import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class AutoFixResult:
    """자동 수정 결과"""
    file_path: str
    fixes_applied: int
    backup_created: bool
    success: bool
    error_message: Optional[str] = None
    modified_lines: List[int] = None

class SerenaAutoOptimizer:
    """Serena 자동 최적화 엔진"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_dir = self.project_root / "serena_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.logger = self._setup_logger()
        
        # 자동 수정 패턴들
        self.auto_fix_patterns = {
            "threadpool_fix": {
                "description": "ThreadPoolExecutor를 context manager로 변경",
                "pattern": r"(\s*)(executor\s*=\s*ThreadPoolExecutor\([^)]*\))\s*\n",
                "replacement": r"\1with ThreadPoolExecutor() as executor:\n\1    # Serena 자동 수정: with 문으로 리소스 안전 관리\n",
                "confidence": 0.95,
                "impact": "critical"
            },
            
            "file_io_fix": {
                "description": "파일 열기를 context manager로 변경",
                "pattern": r"(\s*)((?:file|f)\s*=\s*open\([^)]+\))\s*\n",
                "replacement": r"\1with open(...) as file:\n\1    # Serena 자동 수정: with 문으로 파일 안전 관리\n",
                "confidence": 0.85,
                "impact": "medium"
            },
            
            "gpu_memory_cleanup": {
                "description": "CUDA 연산 후 메모리 정리 추가",
                "pattern": r"(.*torch\.cuda\..*)\n(?!\s*torch\.cuda\.empty_cache)",
                "replacement": r"\1\n    torch.cuda.empty_cache()  # Serena 자동 추가: GPU 메모리 정리\n",
                "confidence": 0.90,
                "impact": "high"
            },
            
            "streamlit_cache_add": {
                "description": "무거운 함수에 Streamlit 캐시 추가",
                "pattern": r"(def\s+\w+.*heavy.*\([^)]*\):\s*\n)(?!\s*@st\.cache)",
                "replacement": r"@st.cache_data  # Serena 자동 추가: 성능 최적화\n\1",
                "confidence": 0.80,
                "impact": "medium"
            }
        }
        
        # 임포트 자동 추가 패턴
        self.import_patterns = {
            "threadpool": "from concurrent.futures import ThreadPoolExecutor",
            "torch_cuda": "import torch",
            "streamlit_cache": "import streamlit as st"
        }

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("SerenaAutoOptimizer")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.project_root / "serena_auto_fixer.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def analyze_and_fix_project(self, target_files: List[str] = None, auto_apply: bool = False) -> Dict[str, Any]:
        """
        프로젝트 분석 및 자동 수정
        
        Args:
            target_files: 대상 파일 리스트 (None이면 자동 선택)
            auto_apply: True면 자동으로 수정 적용, False면 분석만
        """
        results = {
            "serena_optimizer": {
                "name": "Serena Auto Optimizer",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            },
            "analysis_summary": {
                "files_analyzed": 0,
                "fixable_issues": 0,
                "critical_fixes": 0,
                "auto_fixes_applied": 0 if not auto_apply else 0
            },
            "file_results": [],
            "optimization_report": [],
            "errors": []
        }
        
        try:
            # 대상 파일 결정
            if not target_files:
                target_files = self._get_priority_files()
            
            # 각 파일 분석 및 수정
            for file_path in target_files:
                if not Path(file_path).exists():
                    continue
                    
                file_result = self._analyze_and_fix_file(file_path, auto_apply)
                results["file_results"].append(file_result)
                results["analysis_summary"]["files_analyzed"] += 1
                
                if file_result["fixable_issues"] > 0:
                    results["analysis_summary"]["fixable_issues"] += file_result["fixable_issues"]
                    
                if file_result["critical_fixes"] > 0:
                    results["analysis_summary"]["critical_fixes"] += file_result["critical_fixes"]
                    
                if auto_apply and file_result["fixes_applied"] > 0:
                    results["analysis_summary"]["auto_fixes_applied"] += file_result["fixes_applied"]
            
            # 최적화 보고서 생성
            results["optimization_report"] = self._generate_optimization_report(results)
            
        except Exception as e:
            error_msg = f"프로젝트 분석 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
        
        return results

    def _get_priority_files(self) -> List[str]:
        """우선순위 파일 목록 반환"""
        priority_files = [
            "conference_analysis_COMPLETE_WORKING.py",
            "solomond_ai_main_dashboard.py",
            "hybrid_compute_manager.py",
            "dual_brain_integration.py",
            "ai_insights_engine.py"
        ]
        
        existing_files = []
        for file_name in priority_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                existing_files.append(str(file_path))
        
        # core 디렉토리의 중요 파일들 추가
        core_dir = self.project_root / "core"
        if core_dir.exists():
            core_files = [
                "multimodal_pipeline.py",
                "batch_processing_engine.py",
                "memory_optimizer.py"
            ]
            
            for file_name in core_files:
                file_path = core_dir / file_name
                if file_path.exists():
                    existing_files.append(str(file_path))
        
        return existing_files

    def _analyze_and_fix_file(self, file_path: str, auto_apply: bool) -> Dict[str, Any]:
        """단일 파일 분석 및 수정"""
        file_result = {
            "file_path": str(Path(file_path).name),
            "full_path": file_path,
            "fixable_issues": 0,
            "critical_fixes": 0,
            "fixes_applied": 0,
            "backup_created": False,
            "issues_found": [],
            "modifications": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            modified_content = original_content
            lines = original_content.split('\n')
            
            # 각 패턴별로 분석 및 수정
            for pattern_name, pattern_info in self.auto_fix_patterns.items():
                matches = list(re.finditer(
                    pattern_info["pattern"],
                    modified_content,
                    re.MULTILINE
                ))
                
                for match in matches:
                    line_num = original_content[:match.start()].count('\n') + 1
                    
                    issue = {
                        "line": line_num,
                        "pattern": pattern_name,
                        "description": pattern_info["description"],
                        "impact": pattern_info["impact"],
                        "confidence": pattern_info["confidence"],
                        "code_snippet": lines[line_num - 1].strip() if line_num <= len(lines) else ""
                    }
                    
                    file_result["issues_found"].append(issue)
                    file_result["fixable_issues"] += 1
                    
                    if pattern_info["impact"] == "critical":
                        file_result["critical_fixes"] += 1
                    
                    # 자동 수정 적용
                    if auto_apply:
                        if not file_result["backup_created"]:
                            self._create_backup(file_path)
                            file_result["backup_created"] = True
                        
                        # 패턴 수정 적용
                        modified_content = re.sub(
                            pattern_info["pattern"],
                            pattern_info["replacement"],
                            modified_content,
                            count=1  # 한 번에 하나씩 수정
                        )
                        
                        file_result["fixes_applied"] += 1
                        file_result["modifications"].append({
                            "line": line_num,
                            "pattern": pattern_name,
                            "description": pattern_info["description"]
                        })
            
            # 필요한 임포트 추가
            if auto_apply and file_result["fixes_applied"] > 0:
                modified_content = self._add_required_imports(modified_content, file_result["modifications"])
                
                # 수정된 내용 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                self.logger.info(f"파일 수정 완료: {file_path} ({file_result['fixes_applied']}개 수정)")
                
        except Exception as e:
            error_msg = f"파일 분석/수정 실패 {file_path}: {str(e)}"
            self.logger.error(error_msg)
            file_result["error"] = error_msg
        
        return file_result

    def _create_backup(self, file_path: str) -> str:
        """파일 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = Path(file_path).name
        backup_name = f"{file_name}.backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.logger.info(f"백업 생성: {backup_path}")
        
        return str(backup_path)

    def _add_required_imports(self, content: str, modifications: List[Dict]) -> str:
        """필요한 임포트 자동 추가"""
        lines = content.split('\n')
        imports_to_add = set()
        
        # 수정사항에 따라 필요한 임포트 결정
        for mod in modifications:
            pattern = mod["pattern"]
            if "threadpool" in pattern and "ThreadPoolExecutor" in content:
                imports_to_add.add(self.import_patterns["threadpool"])
            elif "gpu_memory" in pattern and "torch.cuda" in content:
                imports_to_add.add(self.import_patterns["torch_cuda"])
            elif "streamlit_cache" in pattern and "@st.cache" in content:
                imports_to_add.add(self.import_patterns["streamlit_cache"])
        
        # 기존 임포트가 없는 경우에만 추가
        for import_statement in imports_to_add:
            if import_statement not in content:
                # 다른 임포트 문 뒤에 추가
                import_added = False
                for i, line in enumerate(lines):
                    if line.startswith(("import ", "from ")) and not import_added:
                        continue
                    elif not line.startswith(("import ", "from ")) and i > 0 and not import_added:
                        lines.insert(i, import_statement)
                        import_added = True
                        break
                
                if not import_added:
                    lines.insert(0, import_statement)
        
        return '\n'.join(lines)

    def _generate_optimization_report(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """최적화 보고서 생성"""
        report = []
        
        summary = results["analysis_summary"]
        
        # 전체 요약
        report.append({
            "category": "summary",
            "title": "Serena 자동 최적화 요약",
            "content": f"총 {summary['files_analyzed']}개 파일을 분석하여 "
                      f"{summary['fixable_issues']}개의 수정 가능한 이슈를 발견했습니다.",
            "metrics": {
                "files_analyzed": summary["files_analyzed"],
                "total_issues": summary["fixable_issues"],
                "critical_issues": summary["critical_fixes"],
                "fixes_applied": summary["auto_fixes_applied"]
            }
        })
        
        # 크리티컬 이슈 보고
        if summary["critical_fixes"] > 0:
            report.append({
                "category": "critical",
                "title": "크리티컬 이슈 해결",
                "content": f"{summary['critical_fixes']}개의 크리티컬 이슈가 발견되었습니다. "
                          "이러한 이슈들은 시스템 안정성에 직접적인 영향을 미칩니다.",
                "recommendation": "즉시 수정을 적용하여 SOLOMOND AI 시스템의 안정성을 확보하세요.",
                "impact": "시스템 크래시 방지, 리소스 누수 해결"
            })
        
        # 성능 최적화 보고
        performance_fixes = sum(
            1 for file_result in results["file_results"]
            for issue in file_result.get("issues_found", [])
            if issue.get("impact") in ["medium", "high"]
        )
        
        if performance_fixes > 0:
            report.append({
                "category": "performance",
                "title": "성능 최적화 기회",
                "content": f"{performance_fixes}개의 성능 최적화 기회가 발견되었습니다.",
                "recommendation": "Streamlit 캐싱과 메모리 관리 최적화를 적용하세요.",
                "impact": "응답 속도 향상, 리소스 사용량 감소"
            })
        
        # 코드 품질 보고
        if summary["fixable_issues"] > 0:
            quality_score = max(100 - (summary["critical_fixes"] * 20 + performance_fixes * 5), 0)
            report.append({
                "category": "quality",
                "title": "코드 품질 평가",
                "content": f"현재 코드 품질 점수: {quality_score}/100",
                "recommendation": "발견된 이슈들을 수정하여 코드 품질을 향상시키세요.",
                "next_steps": [
                    "크리티컬 이슈 우선 수정",
                    "성능 최적화 패턴 적용",
                    "정기적인 코드 리뷰 수행"
                ]
            })
        
        return report

    def generate_auto_fix_script(self, analysis_results: Dict[str, Any]) -> str:
        """실행 가능한 자동 수정 스크립트 생성"""
        script_lines = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            "\"\"\"",
            "🔧 Serena 자동 생성 수정 스크립트",
            "SOLOMOND AI 시스템 자동 최적화",
            f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\"\"\"",
            "",
            "import re",
            "import shutil",
            "from pathlib import Path",
            "from datetime import datetime",
            "",
            "def backup_file(file_path):",
            "    \"\"\"파일 백업 생성\"\"\"",
            "    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')",
            "    backup_path = f'{file_path}.serena_backup_{timestamp}'",
            "    shutil.copy2(file_path, backup_path)",
            "    print(f'✅ 백업 생성: {backup_path}')",
            "    return backup_path",
            "",
        ]
        
        # 수정할 파일들 추출
        files_with_critical_issues = [
            file_result for file_result in analysis_results["file_results"]
            if file_result["critical_fixes"] > 0
        ]
        
        if files_with_critical_issues:
            script_lines.extend([
                "def fix_critical_issues():",
                "    \"\"\"크리티컬 이슈 자동 수정\"\"\"",
                "    fixes_applied = 0",
                "    files_modified = []",
                ""
            ])
            
            for file_result in files_with_critical_issues:
                file_path = file_result["full_path"]
                script_lines.extend([
                    f"    # {file_result['file_path']} 수정",
                    f"    file_path = '{file_path}'",
                    "    if Path(file_path).exists():",
                    "        backup_file(file_path)",
                    "        ",
                    "        with open(file_path, 'r', encoding='utf-8') as f:",
                    "            content = f.read()",
                    "        ",
                    "        original_content = content",
                    ""
                ])
                
                # 각 이슈별 수정 로직 추가
                for issue in file_result["issues_found"]:
                    if issue["impact"] == "critical":
                        if "threadpool" in issue["pattern"]:
                            script_lines.extend([
                                "        # ThreadPool 리소스 관리 수정",
                                "        pattern = r'(\\s*)(executor\\s*=\\s*ThreadPoolExecutor\\([^)]*\\))\\s*\\n'",
                                "        replacement = r'\\1with ThreadPoolExecutor() as executor:\\n\\1    # Serena 수정: 안전한 리소스 관리\\n'",
                                "        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)",
                                ""
                            ])
                
                script_lines.extend([
                    "        if content != original_content:",
                    "            with open(file_path, 'w', encoding='utf-8') as f:",
                    "                f.write(content)",
                    "            fixes_applied += 1",
                    "            files_modified.append(file_path)",
                    f"            print(f'🔧 수정 완료: {file_result['file_path']}')",
                    "    ",
                    ""
                ])
            
            script_lines.extend([
                "    return fixes_applied, files_modified",
                ""
            ])
        
        # 메인 실행 부분
        script_lines.extend([
            "def main():",
            "    \"\"\"메인 실행 함수\"\"\"",
            "    print('🤖 Serena 자동 수정 시스템 시작')",
            "    print('🎯 SOLOMOND AI 시스템 최적화')",
            "    print('=' * 50)",
            "    ",
            "    total_fixes = 0",
            "    modified_files = []",
            "    ",
        ])
        
        if files_with_critical_issues:
            script_lines.extend([
                "    # 크리티컬 이슈 수정",
                "    critical_fixes, critical_files = fix_critical_issues()",
                "    total_fixes += critical_fixes",
                "    modified_files.extend(critical_files)",
                "    ",
            ])
        
        script_lines.extend([
            "    print(f'\\n✅ 수정 완료!')",
            "    print(f'📊 총 {total_fixes}개 파일 수정됨')",
            "    print(f'📁 수정된 파일: {len(modified_files)}개')",
            "    ",
            "    if total_fixes > 0:",
            "        print('\\n💡 권장사항:')",
            "        print('1. 시스템을 재시작하여 변경사항 확인')",
            "        print('2. 백업 파일들이 생성되었으므로 필요시 복원 가능')",
            "        print('3. 수정 후 기능 테스트 수행')",
            "    else:",
            "        print('ℹ️  수정이 필요한 크리티컬 이슈가 발견되지 않았습니다.')",
            "",
            "if __name__ == '__main__':",
            "    main()"
        ])
        
        return "\n".join(script_lines)

def main():
    """Serena Auto Optimizer 실행"""
    print("🚀 Serena Auto Optimizer")
    print("🎯 SOLOMOND AI 시스템 자동 최적화 엔진")
    print("=" * 60)
    
    optimizer = SerenaAutoOptimizer()
    
    # 분석만 실행 (자동 수정은 사용자 확인 후)
    print("📊 프로젝트 분석 중...")
    results = optimizer.analyze_and_fix_project(auto_apply=False)
    
    # 결과 출력
    summary = results["analysis_summary"]
    print(f"✅ 분석 완료!")
    print(f"📁 분석된 파일: {summary['files_analyzed']}개")
    print(f"🔍 수정 가능한 이슈: {summary['fixable_issues']}개")
    
    if summary["critical_fixes"] > 0:
        print(f"🚨 크리티컬 이슈: {summary['critical_fixes']}개")
    
    # 최적화 보고서 표시
    if results["optimization_report"]:
        print(f"\n📋 Serena 최적화 보고서:")
        for report in results["optimization_report"]:
            if report["category"] == "summary":
                continue
                
            emoji = {"critical": "🚨", "performance": "⚡", "quality": "📈"}
            print(f"\n{emoji.get(report['category'], '💡')} {report['title']}")
            print(f"   {report['content']}")
            
            if "recommendation" in report:
                print(f"   💡 권장사항: {report['recommendation']}")
    
    # 자동 수정 스크립트 생성
    if summary["critical_fixes"] > 0:
        print(f"\n🔧 자동 수정 스크립트 생성 중...")
        fix_script = optimizer.generate_auto_fix_script(results)
        
        script_path = Path("serena_auto_fix_generated.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(fix_script)
        
        print(f"✅ 자동 수정 스크립트 생성: {script_path}")
        print(f"💡 실행 방법: python {script_path}")
        print(f"⚠️  주의: 실행 전 중요 파일을 백업하세요!")

if __name__ == "__main__":
    main()