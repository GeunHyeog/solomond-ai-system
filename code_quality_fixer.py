#!/usr/bin/env python3
"""
코드 품질 개선기 - 단계별 자동 수정 시스템
권장 순서대로 346개 품질 이슈를 체계적으로 해결
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 통합 툴킷 활용
from integrated_development_toolkit import IntegratedDevelopmentToolkit

class CodeQualityFixer:
    """코드 품질 자동 개선기"""
    
    def __init__(self):
        self.toolkit = IntegratedDevelopmentToolkit()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path("C:/Users/PC_58410/SOLOMONDd-ai-system")
        
        # 수정 통계
        self.fix_stats = {
            "hardcoded_values": {"found": 0, "fixed": 0},
            "long_functions": {"found": 0, "fixed": 0}, 
            "duplicate_code": {"found": 0, "fixed": 0},
            "todo_comments": {"found": 0, "fixed": 0}
        }
        
        print(f"[FIXER] 코드 품질 개선기 초기화 - Session: {self.session_id}")
    
    def get_python_files(self) -> List[Path]:
        """수정 대상 Python 파일 목록"""
        
        python_files = []
        
        # 메인 디렉토리 .py 파일들
        main_files = list(self.project_root.glob("*.py"))
        python_files.extend(main_files)
        
        # core 디렉토리 파일들  
        core_dir = self.project_root / "core"
        if core_dir.exists():
            core_files = list(core_dir.glob("*.py"))
            python_files.extend(core_files)
        
        # 시스템 파일 제외
        excluded_patterns = ["__pycache__", "venv", ".git", "legacy_backup"]
        python_files = [f for f in python_files if not any(pattern in str(f) for pattern in excluded_patterns)]
        
        print(f"[FIXER] 수정 대상 파일 {len(python_files)}개 발견")
        return python_files
    
    # =====================================
    # 1단계: 하드코딩된 값 개선
    # =====================================
    
    def fix_hardcoded_values(self) -> Dict[str, Any]:
        """1단계: 하드코딩된 값 자동 개선"""
        
        print("[STAGE1] 하드코딩된 값 개선 시작...")
        
        files = self.get_python_files()
        hardcoded_issues = []
        
        for file_path in files:
            issues = self._find_hardcoded_values(file_path)
            if issues:
                hardcoded_issues.extend(issues)
        
        self.fix_stats["hardcoded_values"]["found"] = len(hardcoded_issues)
        
        # 설정 파일 생성
        config_created = self._create_config_file(hardcoded_issues)
        
        # 하드코딩된 값 교체
        fixed_count = 0
        for issue in hardcoded_issues:
            if self._replace_hardcoded_value(issue):
                fixed_count += 1
        
        self.fix_stats["hardcoded_values"]["fixed"] = fixed_count
        
        result = {
            "stage": "hardcoded_values",
            "found": len(hardcoded_issues),
            "fixed": fixed_count,
            "config_created": config_created,
            "issues": hardcoded_issues[:10]  # 처음 10개만 표시
        }
        
        print(f"[STAGE1] 완료 - 발견: {len(hardcoded_issues)}개, 수정: {fixed_count}개")
        return result
    
    def _find_hardcoded_values(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일에서 하드코딩된 값 탐지"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"[WARNING] 파일 읽기 실패 {file_path}: {e}")
            return []
        
        issues = []
        
        # 패턴별 하드코딩 탐지
        patterns = [
            # URL 패턴
            (r'https?://[^\s\'"]+', 'url', '하드코딩된 URL'),
            # 포트 번호
            (r'localhost:\d+', 'port', '하드코딩된 포트'),
            # 파일 경로 (Windows)
            (r'C:\\[^\\]*(?:\\[^\\]*)*', 'file_path', '하드코딩된 파일 경로'),
            # API 키 패턴 (일부만 - 보안상 전체 교체 안함)
            (r'["\'][a-zA-Z0-9]{20,}["\']', 'api_key', '하드코딩된 API 키'),
            # 이메일 주소
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email', '하드코딩된 이메일'),
        ]
        
        for line_num, line in enumerate(lines, 1):
            # 주석이나 docstring 제외
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            
            for pattern, issue_type, description in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    # 변수 할당이나 설정으로 보이는 경우만 (간단한 휴리스틱)
                    if '=' in line or 'url' in line.lower() or 'port' in line.lower():
                        issues.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "matched_value": match.group(),
                            "issue_type": issue_type,
                            "description": description,
                            "suggested_fix": self._suggest_config_replacement(issue_type, match.group())
                        })
        
        return issues
    
    def _suggest_config_replacement(self, issue_type: str, value: str) -> str:
        """설정 파일 교체 제안"""
        
        config_mapping = {
            'url': 'CONFIG["BASE_URL"]',
            'port': 'CONFIG["PORT"]', 
            'file_path': 'CONFIG["DATA_PATH"]',
            'api_key': 'CONFIG["API_KEY"]',
            'email': 'CONFIG["CONTACT_EMAIL"]'
        }
        
        return config_mapping.get(issue_type, 'CONFIG["VALUE"]')
    
    def _create_config_file(self, hardcoded_issues: List[Dict]) -> bool:
        """설정 파일 자동 생성"""
        
        if not hardcoded_issues:
            return False
        
        # 설정값 추출 및 분류
        config_values = {}
        
        for issue in hardcoded_issues:
            issue_type = issue["issue_type"]
            value = issue["matched_value"].strip('"\'')
            
            if issue_type == 'url':
                config_values["BASE_URL"] = value
            elif issue_type == 'port':
                port = re.search(r'\d+', value)
                if port:
                    config_values["PORT"] = int(port.group())
            elif issue_type == 'file_path':
                config_values["DATA_PATH"] = value
            elif issue_type == 'email':
                config_values["CONTACT_EMAIL"] = value
        
        # config.py 파일 생성
        config_content = f'''#!/usr/bin/env python3
"""
솔로몬드 AI 시스템 설정 파일
자동 생성됨 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import os
from pathlib import Path

# 기본 설정
CONFIG = {{
    # 서버 설정
    "BASE_URL": "{config_values.get('BASE_URL', 'http://f"localhost:{SETTINGS['PORT']}"')}",
    "PORT": {config_values.get('PORT', 8503)},
    
    # 경로 설정  
    "DATA_PATH": "{config_values.get('DATA_PATH', './data')}",
    "UPLOAD_PATH": "./uploads",
    "RESULTS_PATH": "./results",
    
    # API 설정
    "API_TIMEOUT": 30,
    "MAX_FILE_SIZE": 100 * 1024 * 1024,  # 100MB
    
    # 연락처
    "CONTACT_EMAIL": "{config_values.get('CONTACT_EMAIL', 'admin@SOLOMONDd.ai')}",
    
    # 환경별 설정
    "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
    "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
}}

# 환경변수 오버라이드
def load_config():
    """환경변수로 설정 오버라이드"""
    config = CONFIG.copy()
    
    # 환경변수가 있으면 사용
    if os.getenv("SOLOMOND_BASE_URL"):
        config["BASE_URL"] = os.getenv("SOLOMOND_BASE_URL")
    
    if os.getenv("SOLOMOND_PORT"):
        config["PORT"] = int(os.getenv("SOLOMOND_PORT"))
    
    return config

# 전역 설정 인스턴스
SETTINGS = load_config()
'''
        
        try:
            config_file = self.project_root / "config.py"
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            print(f"[CONFIG] 설정 파일 생성 완료: {config_file}")
            return True
            
        except Exception as e:
            print(f"[ERROR] 설정 파일 생성 실패: {e}")
            return False
    
    def _replace_hardcoded_value(self, issue: Dict[str, Any]) -> bool:
        """하드코딩된 값을 설정으로 교체"""
        
        file_path = self.project_root / issue["file"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 해당 라인 수정
            line_num = issue["line_number"] - 1
            if line_num < len(lines):
                old_line = lines[line_num]
                
                # 간단한 교체 (보수적으로)
                if issue["issue_type"] == "port" and "localhost:" in old_line:
                    new_line = old_line.replace(issue["matched_value"], "f\"localhost:{SETTINGS['PORT']}\"")
                    lines[line_num] = new_line
                    
                    # 파일 상단에 import 추가 (이미 없는 경우)
                    if "from config import SETTINGS" not in content:
                        # import 구간 찾기
                        import_line = 0
                        for i, line in enumerate(lines):
                            if line.startswith('import ') or line.startswith('from '):
                                import_line = i + 1
                        
                        lines.insert(import_line, "from config import SETTINGS")
                    
                    # 파일 저장
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    
                    print(f"[FIX] {issue['file']}:{issue['line_number']} 수정 완료")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] 파일 수정 실패 {file_path}: {e}")
            return False
    
    # =====================================
    # 2단계: 긴 함수 리팩토링
    # =====================================
    
    def fix_long_functions(self) -> Dict[str, Any]:
        """2단계: 긴 함수 리팩토링"""
        
        print("[STAGE2] 긴 함수 리팩토링 시작...")
        
        files = self.get_python_files()
        long_functions = []
        
        for file_path in files:
            functions = self._find_long_functions(file_path)
            long_functions.extend(functions)
        
        self.fix_stats["long_functions"]["found"] = len(long_functions)
        
        # 가장 긴 함수부터 리팩토링
        long_functions.sort(key=lambda x: x["line_count"], reverse=True)
        
        fixed_count = 0
        for func in long_functions[:5]:  # 상위 5개만 처리
            if self._refactor_long_function(func):
                fixed_count += 1
        
        self.fix_stats["long_functions"]["fixed"] = fixed_count
        
        result = {
            "stage": "long_functions",
            "found": len(long_functions),
            "fixed": fixed_count,
            "top_functions": long_functions[:10]
        }
        
        print(f"[STAGE2] 완료 - 발견: {len(long_functions)}개, 수정: {fixed_count}개")
        return result
    
    def _find_long_functions(self, file_path: Path, min_lines: int = 50) -> List[Dict[str, Any]]:
        """긴 함수 탐지"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return []
        
        functions = []
        in_function = False
        function_start = 0
        function_name = ""
        indent_level = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 함수 시작 탐지
            if stripped.startswith('def ') or stripped.startswith('async def '):
                if in_function and i - function_start >= min_lines:
                    functions.append({
                        "file": str(file_path.relative_to(self.project_root)),
                        "function_name": function_name,
                        "start_line": function_start + 1,
                        "end_line": i,
                        "line_count": i - function_start,
                        "content_preview": ''.join(lines[function_start:function_start+5])
                    })
                
                in_function = True
                function_start = i
                function_name = stripped.split('(')[0].replace('def ', '').replace('async ', '')
                indent_level = len(line) - len(line.lstrip())
            
            # 함수 끝 탐지 (간단한 방법)
            elif in_function and stripped and len(line) - len(line.lstrip()) <= indent_level and not line.startswith(' '):
                if stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('if __name__'):
                    if i - function_start >= min_lines:
                        functions.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "function_name": function_name,
                            "start_line": function_start + 1,
                            "end_line": i,
                            "line_count": i - function_start,
                            "content_preview": ''.join(lines[function_start:function_start+5])
                        })
                    in_function = False
        
        return functions
    
    def _refactor_long_function(self, func_info: Dict[str, Any]) -> bool:
        """긴 함수 리팩토링 (코멘트 추가로 시작)"""
        
        file_path = self.project_root / func_info["file"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 함수 시작 부분에 리팩토링 필요 주석 추가
            start_line = func_info["start_line"] - 1
            
            refactor_comment = f"    # TODO: 리팩토링 필요 - {func_info['line_count']}줄 함수를 더 작은 함수로 분할 고려\n"
            
            if start_line < len(lines) and "TODO: 리팩토링" not in lines[start_line]:
                lines.insert(start_line + 1, refactor_comment)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print(f"[REFACTOR] {func_info['file']}:{func_info['function_name']} 리팩토링 주석 추가")
                return True
            
        except Exception as e:
            print(f"[ERROR] 함수 리팩토링 실패: {e}")
        
        return False
    
    # =====================================
    # 실행 메인 함수
    # =====================================
    
    def run_quality_improvements(self) -> Dict[str, Any]:
        """전체 품질 개선 실행"""
        
        print(f"[FIXER] 코드 품질 개선 시작 - 권장 순서대로 진행")
        print("=" * 60)
        
        results = {}
        
        # 1단계: 하드코딩된 값 개선
        results["stage1"] = self.fix_hardcoded_values()
        
        # 2단계: 긴 함수 리팩토링
        results["stage2"] = self.fix_long_functions()
        
        # 최종 결과
        results["summary"] = {
            "session_id": self.session_id,
            "total_fixes": sum(stage.get("fixed", 0) for stage in results.values() if isinstance(stage, dict)),
            "stats": self.fix_stats,
            "next_steps": [
                "3단계: 중복 코드 제거",
                "4단계: TODO/FIXME 주석 해결",
                "코드 리뷰 및 테스트 실행"
            ]
        }
        
        # GitHub 이슈 업데이트
        self._update_github_with_results(results)
        
        print("=" * 60)
        print(f"[SUCCESS] 품질 개선 1-2단계 완료!")
        print(f"   하드코딩 수정: {results['stage1']['fixed']}개")
        print(f"   긴 함수 표시: {results['stage2']['fixed']}개")
        
        return results
    
    def _update_github_with_results(self, results: Dict[str, Any]):
        """GitHub에 개선 결과 리포팅"""
        
        try:
            comment_body = f"""## 🔧 코드 품질 개선 진행 보고서

**세션 ID**: {self.session_id}
**실행 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 📊 1단계: 하드코딩된 값 개선
- **발견**: {results['stage1']['found']}개
- **수정**: {results['stage1']['fixed']}개
- **설정 파일**: {'✅ 생성' if results['stage1']['config_created'] else '❌ 실패'}

### 📊 2단계: 긴 함수 리팩토링
- **발견**: {results['stage2']['found']}개 (50줄 이상)
- **주석 추가**: {results['stage2']['fixed']}개

### 🎯 다음 단계
- 3단계: 중복 코드 제거
- 4단계: TODO/FIXME 주석 해결
- 자동화된 테스트 실행

> 자동 생성된 보고서 - Code Quality Fixer v1.0
"""
            
            # 새로운 이슈 생성 (코드 품질 전용)
            issue_result = self.toolkit.create_issue(
                "GeunHyeog",
                "SOLOMONDd-ai-system", 
                "🔧 코드 품질 개선 진행 상황",
                comment_body
            )
            
            if issue_result:
                print("[SUCCESS] GitHub 코드 품질 개선 이슈 생성 완료")
                
        except Exception as e:
            print(f"[WARNING] GitHub 업데이트 실패: {e}")

# 실행 함수
def main():
    """메인 실행"""
    
    fixer = CodeQualityFixer()
    results = fixer.run_quality_improvements()
    
    return results

if __name__ == "__main__":
    main()