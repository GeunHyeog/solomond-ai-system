#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 개발 품질 향상 도구
통합 툴킷과 MCP를 활용한 자동화된 개발 프로세스
"""

import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 우리가 만든 통합 툴킷 활용
from integrated_development_toolkit import IntegratedDevelopmentToolkit

class DevelopmentQualityEnhancer:
    """개발 품질 향상 자동화 시스템"""
    
    def __init__(self):
        self.toolkit = IntegratedDevelopmentToolkit()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path("C:/Users/PC_58410/solomond-ai-system")
        
        print(f"[QUALITY] 개발 품질 향상 시스템 초기화 - Session: {self.session_id}")
    
    async def analyze_code_quality(self) -> Dict[str, Any]:
        """코드 품질 자동 분석"""
        
        print("[QUALITY] 코드 품질 분석 시작...")
        
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "python_files": [],
            "quality_issues": [],
            "recommendations": [],
            "github_issues": [],
            "documentation_gaps": []
        }
        
        # 1. Python 파일들 스캔
        python_files = list(self.project_root.glob("*.py"))
        core_files = list((self.project_root / "core").glob("*.py"))
        all_files = python_files + core_files
        
        analysis_result["python_files"] = [str(f.relative_to(self.project_root)) for f in all_files]
        print(f"[QUALITY] Python 파일 {len(all_files)}개 발견")
        
        # 2. GitHub 이슈 분석
        try:
            issues = self.toolkit.list_issues('GeunHyeog', 'solomond-ai-system')
            analysis_result["github_issues"] = [
                {
                    "number": issue["number"],
                    "title": issue["title"],
                    "created_at": issue["created_at"],
                    "labels": [label["name"] for label in issue["labels"]]
                }
                for issue in issues
            ]
            print(f"[QUALITY] GitHub 이슈 {len(issues)}개 분석 완료")
        except Exception as e:
            print(f"[ERROR] GitHub 이슈 분석 실패: {e}")
        
        # 3. 코드 품질 이슈 탐지
        quality_issues = await self._detect_quality_issues(all_files)
        analysis_result["quality_issues"] = quality_issues
        
        # 4. 개선 권장사항 생성
        recommendations = self._generate_recommendations(analysis_result)
        analysis_result["recommendations"] = recommendations
        
        return analysis_result
    
    async def _detect_quality_issues(self, files: List[Path]) -> List[Dict[str, Any]]:
        """코드 품질 이슈 자동 탐지"""
        
        issues = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = []
                
                # 기본적인 품질 체크
                lines = content.split('\n')
                
                # 1. 긴 함수 탐지 (50줄 이상)
                in_function = False
                function_start = 0
                function_name = ""
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        if in_function and i - function_start > 50:
                            file_issues.append({
                                "type": "long_function",
                                "function": function_name,
                                "lines": i - function_start,
                                "line_number": function_start
                            })
                        
                        in_function = True
                        function_start = i
                        function_name = line.strip().split('(')[0].replace('def ', '')
                    
                    elif line.strip().startswith('class '):
                        in_function = False
                
                # 2. TODO/FIXME 주석 탐지
                for i, line in enumerate(lines):
                    if 'TODO' in line.upper() or 'FIXME' in line.upper():
                        file_issues.append({
                            "type": "todo_comment",
                            "content": line.strip(),
                            "line_number": i + 1
                        })
                
                # 3. 하드코딩된 값 탐지 (간단한 패턴)
                for i, line in enumerate(lines):
                    if 'localhost:' in line or 'http://' in line:
                        file_issues.append({
                            "type": "hardcoded_url",
                            "content": line.strip(),
                            "line_number": i + 1
                        })
                
                if file_issues:
                    issues.append({
                        "file": str(file_path.relative_to(self.project_root)),
                        "issues": file_issues
                    })
                    
            except Exception as e:
                print(f"[WARNING] 파일 분석 실패 {file_path}: {e}")
                continue
        
        return issues
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """개선 권장사항 자동 생성"""
        
        recommendations = []
        
        # 1. 코드 품질 기반 권장사항
        total_issues = sum(len(file_data["issues"]) for file_data in analysis["quality_issues"])
        
        if total_issues > 10:
            recommendations.append({
                "category": "code_quality",
                "priority": "high",
                "title": "코드 품질 개선 필요",
                "description": f"{total_issues}개의 품질 이슈 발견. 리팩토링 권장",
                "action": "automated_refactoring"
            })
        
        # 2. GitHub 이슈 기반 권장사항
        if len(analysis["github_issues"]) > 3:
            recommendations.append({
                "category": "project_management",
                "priority": "medium", 
                "title": "이슈 관리 개선",
                "description": f"{len(analysis['github_issues'])}개의 열린 이슈. 우선순위 정리 필요",
                "action": "issue_prioritization"
            })
        
        # 3. 자동화 개선 권장사항
        recommendations.append({
            "category": "automation",
            "priority": "high",
            "title": "CI/CD 파이프라인 구축",
            "description": "자동화된 테스트 및 배포 시스템 구축 권장",
            "action": "setup_automation"
        })
        
        # 4. 문서화 개선 권장사항
        if len(analysis["python_files"]) > 20:
            recommendations.append({
                "category": "documentation",
                "priority": "medium",
                "title": "API 문서 자동화",
                "description": "대규모 프로젝트에 자동 문서 생성 시스템 권장",
                "action": "auto_documentation"
            })
        
        return recommendations
    
    async def implement_quality_improvements(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """품질 개선사항 자동 구현"""
        
        print("[QUALITY] 품질 개선사항 구현 시작...")
        
        implementation_result = {
            "implemented": [],
            "failed": [],
            "created_files": []
        }
        
        # 1. 자동화 스크립트 생성
        automation_script = await self._create_automation_script()
        if automation_script:
            implementation_result["created_files"].append("quality_automation.py")
            implementation_result["implemented"].append("automation_script")
        
        # 2. GitHub 이슈 정리
        issue_organization = await self._organize_github_issues(analysis["github_issues"])
        if issue_organization:
            implementation_result["implemented"].append("issue_organization")
        
        # 3. 코드 품질 대시보드 생성
        dashboard = await self._create_quality_dashboard(analysis)
        if dashboard:
            implementation_result["created_files"].append("quality_dashboard.html")
            implementation_result["implemented"].append("quality_dashboard")
        
        return implementation_result
    
    async def _create_automation_script(self) -> bool:
        """품질 자동화 스크립트 생성"""
        
        script_content = '''#!/usr/bin/env python3
"""
솔로몬드 AI 자동화 품질 검사 스크립트
매일 실행되어 코드 품질을 모니터링하고 GitHub에 보고
"""

import sys
import subprocess
from pathlib import Path

def run_quality_checks():
    """품질 검사 실행"""
    
    print("🔍 솔로몬드 AI 품질 검사 시작...")
    
    checks = [
        ("Python 문법 검사", "python -m py_compile *.py"),
        ("Import 검사", "python -c 'import jewelry_stt_ui_v23_real'"),
        ("테스트 실행", "python -m pytest test_* -v"),
    ]
    
    results = []
    
    for name, command in checks:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                results.append(f"✅ {name}: 성공")
            else:
                results.append(f"❌ {name}: 실패 - {result.stderr[:100]}")
        except Exception as e:
            results.append(f"⚠️ {name}: 오류 - {str(e)}")
    
    # 결과 저장
    with open("quality_report.txt", "w", encoding="utf-8") as f:
        f.write("\\n".join(results))
    
    print("📊 품질 검사 완료 - quality_report.txt 확인")

if __name__ == "__main__":
    run_quality_checks()
'''
        
        try:
            script_path = self.project_root / "quality_automation.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            print("[SUCCESS] 자동화 스크립트 생성 완료")
            return True
        except Exception as e:
            print(f"[ERROR] 자동화 스크립트 생성 실패: {e}")
            return False
    
    async def _organize_github_issues(self, issues: List[Dict]) -> bool:
        """GitHub 이슈 자동 정리"""
        
        if not issues:
            return True
        
        # 이슈 우선순위 분석
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for issue in issues:
            labels = issue.get("labels", [])
            title = issue["title"].lower()
            
            if any(label in ["bug", "critical", "urgent"] for label in labels):
                high_priority.append(issue)
            elif any(label in ["enhancement", "feature"] for label in labels):
                medium_priority.append(issue)
            else:
                low_priority.append(issue)
        
        # 우선순위 보고서 생성
        report = f"""# GitHub 이슈 우선순위 분석

## 🔴 높은 우선순위 ({len(high_priority)}개)
{chr(10).join(f"- #{issue['number']}: {issue['title']}" for issue in high_priority)}

## 🟡 중간 우선순위 ({len(medium_priority)}개)  
{chr(10).join(f"- #{issue['number']}: {issue['title']}" for issue in medium_priority)}

## 🟢 낮은 우선순위 ({len(low_priority)}개)
{chr(10).join(f"- #{issue['number']}: {issue['title']}" for issue in low_priority)}

생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        try:
            report_path = self.project_root / "issue_priority_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print("[SUCCESS] GitHub 이슈 우선순위 보고서 생성 완료")
            return True
        except Exception as e:
            print(f"[ERROR] 이슈 정리 실패: {e}")
            return False
    
    async def _create_quality_dashboard(self, analysis: Dict[str, Any]) -> bool:
        """품질 대시보드 HTML 생성"""
        
        dashboard_html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>솔로몬드 AI 품질 대시보드</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #e3f2fd; border-radius: 8px; }}
        .issue {{ padding: 10px; margin: 5px 0; background: #fff3e0; border-left: 4px solid #ff9800; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background: #e8f5e8; border-left: 4px solid #4caf50; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 솔로몬드 AI 품질 대시보드</h1>
        <p>생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>📊 프로젝트 메트릭</h2>
        <div class="metric">
            <h3>Python 파일</h3>
            <div style="font-size: 2em; color: #2196f3;">{len(analysis["python_files"])}</div>
        </div>
        <div class="metric">
            <h3>GitHub 이슈</h3>
            <div style="font-size: 2em; color: #ff9800;">{len(analysis["github_issues"])}</div>
        </div>
        <div class="metric">
            <h3>품질 이슈</h3>
            <div style="font-size: 2em; color: #f44336;">{sum(len(f["issues"]) for f in analysis["quality_issues"])}</div>
        </div>
        
        <h2>🔧 개선 권장사항</h2>
        {''.join(f'<div class="recommendation"><strong>{rec["title"]}</strong><br>{rec["description"]}</div>' for rec in analysis["recommendations"])}
        
        <h2>📈 다음 단계</h2>
        <ul>
            <li>자동화 스크립트 실행: <code>python quality_automation.py</code></li>
            <li>GitHub 이슈 우선순위 검토</li>
            <li>코드 리팩토링 계획 수립</li>
        </ul>
    </div>
</body>
</html>'''
        
        try:
            dashboard_path = self.project_root / "quality_dashboard.html"
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            print("[SUCCESS] 품질 대시보드 생성 완료")
            return True  
        except Exception as e:
            print(f"[ERROR] 대시보드 생성 실패: {e}")
            return False

# 메인 실행 함수
async def enhance_solomond_quality():
    """솔로몬드 AI 시스템 품질 향상 메인 프로세스"""
    
    enhancer = DevelopmentQualityEnhancer()
    
    # 1. 코드 품질 분석
    analysis = await enhancer.analyze_code_quality()
    
    # 2. 개선사항 구현
    implementation = await enhancer.implement_quality_improvements(analysis)
    
    # 3. 결과 저장 (Supabase 로그)
    enhancer.toolkit.save_development_log(
        action="quality_enhancement", 
        details={"analysis": analysis, "implementation": implementation}
    )
    
    print("🎉 솔로몬드 AI 품질 향상 프로세스 완료!")
    return {"analysis": analysis, "implementation": implementation}

if __name__ == "__main__":
    asyncio.run(enhance_solomond_quality())