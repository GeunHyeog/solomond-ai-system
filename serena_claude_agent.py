#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Serena Claude Code Sub-Agent
SOLOMOND AI 전용 Serena 에이전트 - Claude Code 네이티브 통합

이 모듈은 Claude Code의 MCP 도구들을 활용하여 Serena 에이전트의 기능을
네이티브 Claude Code 환경에 통합합니다.

사용법:
/agent serena [command] [options]

Commands:
- analyze: 코드 분석 수행
- fix: 자동 수정 적용  
- health: 프로젝트 건강도 체크
- optimize: 성능 최적화 제안

Author: SOLOMOND AI Team
Version: 1.0.0
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

class SerenaClaudeAgent:
    """Serena Claude Code 서브에이전트"""
    
    def __init__(self):
        self.agent_name = "serena"
        self.version = "1.0.0"
        self.project_root = Path.cwd()
        
        # Serena 에이전트 설정
        self.config = {
            "name": "Serena",
            "role": "SOLOMOND AI 코딩 전문가",
            "expertise": [
                "Symbol-level 코드 분석",
                "ThreadPool 최적화", 
                "메모리 누수 탐지",
                "Streamlit 성능 향상",
                "Ollama AI 통합 최적화"
            ],
            "response_style": "정밀하고 실용적인 기술 조언"
        }
        
        # SOLOMOND AI 특화 패턴
        self.analysis_patterns = {
            "threadpool_critical": {
                "regex": r"ThreadPoolExecutor.*(?:submit|map).*(?!with\s)",
                "severity": "critical",
                "message": "ThreadPoolExecutor가 context manager 없이 사용됨",
                "solution": "with ThreadPoolExecutor() as executor: 패턴 사용"
            },
            "memory_leak_gpu": {
                "regex": r"torch\.cuda.*(?!empty_cache)",
                "severity": "high", 
                "message": "GPU 메모리 정리 없이 CUDA 연산 수행",
                "solution": "torch.cuda.empty_cache() 호출 추가"
            },
            "streamlit_no_cache": {
                "regex": r"st\.(?!cache_data|cache_resource).*(?:heavy|process|analyze)",
                "severity": "medium",
                "message": "무거운 연산에 Streamlit 캐시 미사용",
                "solution": "@st.cache_data 또는 @st.cache_resource 데코레이터 추가"
            },
            "ollama_no_error_handling": {
                "regex": r"ollama.*(?:generate|chat).*(?!try|except)",
                "severity": "medium",
                "message": "Ollama API 호출에 에러 처리 없음",
                "solution": "try-except 블록으로 예외 처리 추가"
            }
        }

    def analyze_project(self, target_files: List[str] = None) -> Dict[str, Any]:
        """
        프로젝트 분석 수행
        Claude Code의 Read, Glob 도구를 활용
        """
        analysis_result = {
            "agent": "Serena",
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "SOLOMOND AI 특화 코드 분석",
            "files_analyzed": 0,
            "issues_found": [],
            "health_score": 0,
            "recommendations": [],
            "critical_fixes": []
        }
        
        try:
            # 분석 대상 파일 결정
            if not target_files:
                # SOLOMOND AI 핵심 파일들
                target_files = [
                    "conference_analysis_COMPLETE_WORKING.py",
                    "solomond_ai_main_dashboard.py", 
                    "dual_brain_integration.py",
                    "ai_insights_engine.py",
                    "google_calendar_connector.py"
                ]
            
            analyzed_files = 0
            total_issues = 0
            critical_issues = 0
            
            for file_path in target_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    analyzed_files += 1
                    file_issues = self._analyze_single_file(str(full_path))
                    
                    for issue in file_issues:
                        analysis_result["issues_found"].append(issue)
                        total_issues += 1
                        
                        if issue["severity"] == "critical":
                            critical_issues += 1
                            analysis_result["critical_fixes"].append({
                                "file": file_path,
                                "line": issue["line"],
                                "fix": issue["solution"]
                            })
            
            analysis_result["files_analyzed"] = analyzed_files
            
            # 건강도 점수 계산
            if analyzed_files > 0:
                penalty = (critical_issues * 20) + (total_issues * 5)
                analysis_result["health_score"] = max(100 - penalty, 0)
            
            # 추천사항 생성
            analysis_result["recommendations"] = self._generate_recommendations(
                total_issues, critical_issues, analysis_result["issues_found"]
            )
            
        except Exception as e:
            analysis_result["error"] = f"분석 중 오류 발생: {str(e)}"
        
        return analysis_result

    def _analyze_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """단일 파일 분석"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 패턴별 분석
            for pattern_name, pattern_info in self.analysis_patterns.items():
                matches = re.finditer(
                    pattern_info["regex"], 
                    content, 
                    re.MULTILINE | re.IGNORECASE
                )
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    code_line = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                    
                    issue = {
                        "file": str(Path(file_path).name),
                        "line": line_num,
                        "pattern": pattern_name,
                        "severity": pattern_info["severity"],
                        "message": pattern_info["message"],
                        "code": code_line,
                        "solution": pattern_info["solution"],
                        "confidence": 0.85
                    }
                    issues.append(issue)
                    
        except Exception as e:
            issues.append({
                "file": str(Path(file_path).name),
                "line": 0,
                "pattern": "analysis_error",
                "severity": "error",
                "message": f"파일 분석 실패: {str(e)}",
                "code": "",
                "solution": "파일 접근 권한 및 인코딩 확인",
                "confidence": 1.0
            })
        
        return issues

    def _generate_recommendations(self, total_issues: int, critical_issues: int, issues: List[Dict]) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        # 기본 추천사항
        if critical_issues > 0:
            recommendations.append(
                f"🚨 {critical_issues}개의 크리티컬 이슈를 즉시 수정해야 합니다. "
                "특히 ThreadPool 리소스 관리는 시스템 안정성에 직접적 영향을 줍니다."
            )
        
        if total_issues > 5:
            recommendations.append(
                "📊 발견된 이슈가 많습니다. 코드 리뷰 프로세스 강화를 권장합니다."
            )
        
        # 패턴별 추천사항
        pattern_counts = {}
        for issue in issues:
            pattern = issue.get("pattern", "unknown")
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if pattern_counts.get("threadpool_critical", 0) > 0:
            recommendations.append(
                "🔧 ThreadPoolExecutor 사용 시 항상 with 문을 사용하세요. "
                "이는 SOLOMOND AI 시스템의 안정성에 필수적입니다."
            )
        
        if pattern_counts.get("memory_leak_gpu", 0) > 0:
            recommendations.append(
                "🧹 GPU 메모리 관리를 강화하세요. "
                "torch.cuda.empty_cache()를 정기적으로 호출하여 메모리 누수를 방지하세요."
            )
        
        if pattern_counts.get("streamlit_no_cache", 0) > 0:
            recommendations.append(
                "⚡ Streamlit 캐싱을 적극 활용하세요. "
                "무거운 AI 모델 로딩과 데이터 처리에 캐시를 사용하면 성능이 크게 향상됩니다."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ 주요 패턴에서 심각한 문제가 발견되지 않았습니다. "
                "현재 코드 품질이 양호한 상태입니다."
            )
        
        return recommendations

    def generate_fix_script(self, analysis_result: Dict[str, Any]) -> str:
        """자동 수정 스크립트 생성"""
        if not analysis_result.get("critical_fixes"):
            return "# 수정이 필요한 크리티컬 이슈가 없습니다.\nprint('코드가 안정적입니다!')"
        
        script_lines = [
            "#!/usr/bin/env python3",
            "# Serena 자동 생성 수정 스크립트",
            "# SOLOMOND AI 시스템 자동 최적화",
            f"# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "import re",
            "import shutil",
            "from pathlib import Path",
            "",
            "def backup_file(file_path):",
            "    \"\"\"파일 백업\"\"\"",
            "    backup_path = f'{file_path}.serena_backup'",
            "    shutil.copy2(file_path, backup_path)",
            "    print(f'백업 생성: {backup_path}')",
            "",
            "def fix_threadpool_issues():",
            "    \"\"\"ThreadPool 이슈 자동 수정\"\"\"",
            "    fixes_applied = 0",
            ""
        ]
        
        # ThreadPool 수정 로직 추가
        threadpool_files = [
            fix["file"] for fix in analysis_result["critical_fixes"] 
            if "threadpool" in fix.get("fix", "").lower()
        ]
        
        if threadpool_files:
            script_lines.extend([
                "    files_to_fix = [",
                *[f"        '{file}'," for file in set(threadpool_files)],
                "    ]",
                "",
                "    for file_path in files_to_fix:",
                "        if Path(file_path).exists():",
                "            backup_file(file_path)",
                "            ",
                "            with open(file_path, 'r', encoding='utf-8') as f:",
                "                content = f.read()",
                "            ",
                "            # ThreadPoolExecutor를 with 문으로 감싸기",
                "            original_content = content",
                "            ",
                "            # 패턴 1: 기본 ThreadPoolExecutor 할당",
                "            pattern1 = r'(\\s*)(executor\\s*=\\s*ThreadPoolExecutor\\([^)]*\\))\\s*\\n'",
                "            replacement1 = r'\\1with ThreadPoolExecutor() as executor:\\n\\1    # Serena 수정: with 문 사용으로 리소스 안전 관리\\n'",
                "            content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)",
                "            ",
                "            if content != original_content:",
                "                with open(file_path, 'w', encoding='utf-8') as f:",
                "                    f.write(content)",
                "                fixes_applied += 1",
                "                print(f'ThreadPool 수정 완료: {file_path}')",
                "    ",
                "    return fixes_applied",
                ""
            ])
        
        script_lines.extend([
            "def main():",
            "    \"\"\"메인 실행\"\"\"",
            "    print('🔧 Serena 자동 수정 시스템 시작')",
            "    print('=' * 50)",
            "    ",
            "    total_fixes = 0",
            "    ",
            "    # ThreadPool 이슈 수정",
            "    threadpool_fixes = fix_threadpool_issues()",
            "    total_fixes += threadpool_fixes",
            "    ",
            "    print(f'\\n✅ 수정 완료: 총 {total_fixes}개 파일')",
            "    ",
            "    if total_fixes > 0:",
            "        print('💡 수정 후 시스템을 재시작하여 변경사항을 확인하세요.')",
            "        print('📝 백업 파일들이 생성되었으므로 필요시 복원 가능합니다.')",
            "    else:",
            "        print('ℹ️  수정이 필요한 이슈가 발견되지 않았습니다.')",
            "",
            "if __name__ == '__main__':",
            "    main()"
        ])
        
        return "\n".join(script_lines)

    def get_agent_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "name": self.config["name"],
            "version": self.version,
            "role": self.config["role"],
            "expertise": self.config["expertise"],
            "response_style": self.config["response_style"],
            "capabilities": [
                "SOLOMOND AI 특화 코드 분석",
                "ThreadPool 최적화 자동 수정",
                "메모리 누수 패턴 탐지",
                "Streamlit 성능 최적화 제안",
                "프로젝트 건강도 평가",
                "자동 수정 스크립트 생성"
            ],
            "supported_commands": [
                "analyze - 코드 분석 수행",
                "fix - 자동 수정 적용",
                "health - 프로젝트 건강도 체크", 
                "optimize - 성능 최적화 제안",
                "info - 에이전트 정보 표시"
            ]
        }

def main():
    """테스트 실행"""
    print("🤖 Serena Claude Code Sub-Agent")
    print("=" * 50)
    
    serena = SerenaClaudeAgent()
    
    # 에이전트 정보 표시
    info = serena.get_agent_info()
    print(f"🧠 {info['name']} v{info['version']}")
    print(f"📋 역할: {info['role']}")
    print(f"⚡ 전문 분야: {', '.join(info['expertise'][:3])}...")
    
    # 프로젝트 분석 실행
    print(f"\n📊 SOLOMOND AI 프로젝트 분석 중...")
    result = serena.analyze_project()
    
    print(f"✅ 분석 완료!")
    print(f"📁 분석된 파일: {result['files_analyzed']}개")
    print(f"📈 건강도 점수: {result['health_score']}/100")
    print(f"🔍 발견된 이슈: {len(result['issues_found'])}개")
    
    if result['critical_fixes']:
        print(f"🚨 크리티컬 수정사항: {len(result['critical_fixes'])}개")
    
    # 추천사항 표시
    if result['recommendations']:
        print(f"\n💡 Serena의 추천사항:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    # 자동 수정 스크립트 생성 (크리티컬 이슈가 있는 경우)
    if result['critical_fixes']:
        print(f"\n🔧 자동 수정 스크립트 생성 중...")
        fix_script = serena.generate_fix_script(result)
        
        script_path = Path("serena_auto_fix.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(fix_script)
        
        print(f"✅ 자동 수정 스크립트: {script_path}")
        print(f"💡 실행: python {script_path}")

if __name__ == "__main__":
    main()