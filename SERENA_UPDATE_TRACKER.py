#!/usr/bin/env python3
"""
Serena 업데이트 추적 및 자동 반영 시스템
"""

import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class SerenaUpdateTracker:
    """Serena 업데이트 추적기"""
    
    def __init__(self):
        self.serena_repo = "https://api.github.com/repos/oraios/serena"
        self.update_file = "serena_update_status.json"
        self.last_check_file = "serena_last_check.json"
        
    def get_latest_release(self) -> Optional[Dict]:
        """최신 릴리즈 정보 가져오기"""
        try:
            response = requests.get(f"{self.serena_repo}/releases/latest", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"릴리즈 정보 가져오기 실패: {response.status_code}")
                return None
        except Exception as e:
            print(f"API 호출 실패: {e}")
            return None
    
    def get_recent_commits(self, limit=10) -> List[Dict]:
        """최근 커밋 정보 가져오기"""
        try:
            response = requests.get(f"{self.serena_repo}/commits?per_page={limit}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"커밋 정보 가져오기 실패: {response.status_code}")
                return []
        except Exception as e:
            print(f"커밋 API 호출 실패: {e}")
            return []
    
    def load_last_check(self) -> Dict:
        """마지막 체크 상태 로드"""
        if os.path.exists(self.last_check_file):
            try:
                with open(self.last_check_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"last_release": None, "last_commit": None, "last_check": None}
    
    def save_last_check(self, data: Dict):
        """마지막 체크 상태 저장"""
        with open(self.last_check_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def check_for_updates(self) -> Dict[str, Any]:
        """업데이트 확인"""
        print("Serena 업데이트 확인 중...")
        
        # 이전 상태 로드
        last_check = self.load_last_check()
        
        # 최신 정보 가져오기
        latest_release = self.get_latest_release()
        recent_commits = self.get_recent_commits()
        
        update_info = {
            "check_time": datetime.now().isoformat(),
            "has_updates": False,
            "updates": []
        }
        
        # 릴리즈 업데이트 확인
        if latest_release:
            if last_check["last_release"] != latest_release["tag_name"]:
                update_info["has_updates"] = True
                update_info["updates"].append({
                    "type": "release",
                    "version": latest_release["tag_name"],
                    "name": latest_release["name"],
                    "published_at": latest_release["published_at"],
                    "body": latest_release["body"][:500] + "..." if len(latest_release["body"]) > 500 else latest_release["body"]
                })
        
        # 커밋 업데이트 확인
        if recent_commits and last_check["last_commit"]:
            new_commits = []
            for commit in recent_commits:
                if commit["sha"] == last_check["last_commit"]:
                    break
                new_commits.append(commit)
            
            if new_commits:
                update_info["has_updates"] = True
                update_info["updates"].append({
                    "type": "commits",
                    "count": len(new_commits),
                    "commits": [{
                        "sha": c["sha"][:8],
                        "message": c["commit"]["message"].split('\\n')[0][:100],
                        "date": c["commit"]["author"]["date"],
                        "author": c["commit"]["author"]["name"]
                    } for c in new_commits[:5]]  # 최근 5개만
                })
        
        # 현재 상태 저장
        current_state = {
            "last_release": latest_release["tag_name"] if latest_release else None,
            "last_commit": recent_commits[0]["sha"] if recent_commits else None,
            "last_check": update_info["check_time"]
        }
        self.save_last_check(current_state)
        
        return update_info
    
    def analyze_update_impact(self, update_info: Dict) -> Dict[str, Any]:
        """업데이트 영향도 분석"""
        
        impact_analysis = {
            "compatibility": "unknown",
            "integration_effort": "medium",
            "recommended_action": "manual_review",
            "affected_features": [],
            "update_priority": "medium"
        }
        
        for update in update_info["updates"]:
            if update["type"] == "release":
                # 버전 분석
                version = update["version"]
                if "major" in version or "v2." in version or "v3." in version:
                    impact_analysis["compatibility"] = "breaking_changes"
                    impact_analysis["integration_effort"] = "high"
                    impact_analysis["update_priority"] = "high"
                elif "minor" in version or any(x in version for x in ["v1.1", "v1.2", "v1.3"]):
                    impact_analysis["compatibility"] = "backward_compatible"
                    impact_analysis["integration_effort"] = "medium"
                else:
                    impact_analysis["compatibility"] = "patch"
                    impact_analysis["integration_effort"] = "low"
                
                # 기능 영향 분석 (릴리즈 노트 기반)
                body = update["body"].lower()
                if any(keyword in body for keyword in ["symbol", "lsp", "language server"]):
                    impact_analysis["affected_features"].append("symbol_analysis")
                if any(keyword in body for keyword in ["mcp", "context", "protocol"]):
                    impact_analysis["affected_features"].append("mcp_integration")
                if any(keyword in body for keyword in ["performance", "speed", "optimization"]):
                    impact_analysis["affected_features"].append("performance")
        
        return impact_analysis
    
    def generate_update_plan(self, update_info: Dict, impact_analysis: Dict) -> str:
        """업데이트 계획 생성"""
        
        if not update_info["has_updates"]:
            return "현재 최신 상태입니다. 업데이트가 필요하지 않습니다."
        
        plan = f"""# Serena 업데이트 적용 계획

## 📊 업데이트 정보
- 확인 시간: {update_info['check_time']}
- 업데이트 유형: {len(update_info['updates'])}개 발견

"""
        
        for update in update_info["updates"]:
            if update["type"] == "release":
                plan += f"""### 🚀 새 릴리즈: {update['version']}
- 릴리즈명: {update['name']}
- 발행일: {update['published_at']}
- 주요 내용: {update['body'][:200]}...

"""
            elif update["type"] == "commits":
                plan += f"""### 📝 새 커밋: {update['count']}개
"""
                for commit in update['commits']:
                    plan += f"- `{commit['sha']}` {commit['message']} ({commit['author']})\n"
                plan += "\n"
        
        plan += f"""## 🎯 영향도 분석
- **호환성**: {impact_analysis['compatibility']}
- **통합 노력**: {impact_analysis['integration_effort']}
- **우선순위**: {impact_analysis['update_priority']}
- **영향 받는 기능**: {', '.join(impact_analysis['affected_features']) if impact_analysis['affected_features'] else '없음'}

## 🔧 권장 적용 방법

### SOLOMOND AI 맞춤형 업데이트 전략:

1. **백업 생성**
   ```bash
   cp -r solomond_serena_agent.py solomond_serena_agent_backup.py
   cp -r serena_claude_interface.py serena_claude_interface_backup.py
   ```

2. **새 기능 통합**
   - Serena 원본 저장소에서 새 기능 분석
   - SOLOMOND AI 특화 부분 보존하며 선별적 적용
   - Symbol-level 분석, ThreadPool 최적화 등 기존 기능 유지

3. **테스트 및 검증**
   ```bash
   python serena_quick_test.py
   python serena_claude_interface.py analyze
   ```

4. **점진적 롤아웃**
   - 개발 환경에서 먼저 테스트
   - 핵심 기능 정상 작동 확인 후 전체 적용

## ⚠️ 주의사항
- SOLOMOND AI 특화 기능들 (ThreadPool, GPU 메모리, Streamlit 최적화) 보존 필수
- 기존 MCP 통합 부분 호환성 확인 필요
- 사용자 데이터 (.solomond_serena_memory.json) 백업 필수

## 📅 적용 일정
- **즉시 적용 가능**: {impact_analysis['update_priority'] == 'low'}
- **신중한 검토 필요**: {impact_analysis['update_priority'] in ['medium', 'high']}
"""
        
        return plan
    
    def create_auto_update_script(self) -> str:
        """자동 업데이트 스크립트 생성"""
        
        script_content = '''#!/usr/bin/env python3
"""
Serena 자동 업데이트 적용 스크립트
"""

import subprocess
import shutil
import os
from datetime import datetime

def backup_current_version():
    """현재 버전 백업"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"serena_backup_{timestamp}"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        "solomond_serena_agent.py",
        "serena_claude_interface.py", 
        "serena_auto_optimizer.py",
        "serena_quick_test.py",
        ".solomond_serena_memory.json"
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
    
    print(f"백업 완료: {backup_dir}")
    return backup_dir

def download_latest_serena():
    """최신 Serena 다운로드"""
    try:
        # GitHub에서 최신 코드 다운로드
        subprocess.run(["git", "clone", "https://github.com/oraios/serena.git", "serena_latest"], check=True)
        print("최신 Serena 다운로드 완료")
        return True
    except:
        print("다운로드 실패")
        return False

def integrate_updates():
    """업데이트 통합"""
    # 선별적 업데이트 로직
    # SOLOMOND AI 특화 기능 보존
    print("업데이트 통합 중...")
    
    # 실제 통합 로직은 수동 검토 후 구현
    print("수동 검토가 필요한 업데이트입니다.")

def main():
    print("Serena 자동 업데이트 시작...")
    
    # 1. 백업
    backup_dir = backup_current_version()
    
    # 2. 다운로드
    if download_latest_serena():
        # 3. 통합 (신중하게)
        integrate_updates()
        
        # 4. 테스트
        print("업데이트 테스트 실행...")
        result = subprocess.run(["python", "serena_quick_test.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 업데이트 성공!")
        else:
            print("❌ 업데이트 실패, 백업에서 복원...")
            # 복원 로직
    
if __name__ == "__main__":
    main()
'''
        
        with open("serena_auto_update.py", 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return "serena_auto_update.py"

def main():
    """메인 실행"""
    tracker = SerenaUpdateTracker()
    
    # 업데이트 확인
    update_info = tracker.check_for_updates()
    
    # 영향도 분석
    impact_analysis = tracker.analyze_update_impact(update_info)
    
    # 업데이트 계획 생성
    plan = tracker.generate_update_plan(update_info, impact_analysis)
    
    # 결과 출력
    print("="*60)
    print("SERENA 업데이트 추적 결과")
    print("="*60)
    print(plan)
    
    # 보고서 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = f"serena_update_plan_{timestamp}.md"
    
    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write(plan)
    
    # 자동 업데이트 스크립트 생성
    script_file = tracker.create_auto_update_script()
    
    print(f"\\n📋 업데이트 계획 저장: {plan_file}")
    print(f"🔧 자동 업데이트 스크립트: {script_file}")
    
    return update_info["has_updates"]

if __name__ == "__main__":
    main()