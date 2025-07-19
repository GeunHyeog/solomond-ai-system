#!/usr/bin/env python3
"""
지속적 메모리 동기화 시스템
재접속 후에도 개발 단계를 완벽하게 기억하기 위한 자동화 시스템
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class PersistentMemoryManager:
    """지속적 메모리 관리 시스템"""
    
    def __init__(self):
        self.project_root = Path("/home/solomond/claude/solomond-ai-system")
        self.memory_file = self.project_root / "memory_state.json"
        self.claude_md = self.project_root / "CLAUDE.md"
        
    def capture_current_state(self) -> Dict[str, Any]:
        """현재 개발 상태 캐쳐"""
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "git_status": self.get_git_status(),
            "file_structure": self.get_key_files_status(),
            "running_processes": self.get_running_processes(),
            "recent_activities": self.get_recent_activities(),
            "development_phase": self.determine_development_phase(),
            "next_steps": self.get_next_steps()
        }
        
        return state
    
    def get_git_status(self) -> Dict[str, Any]:
        """Git 상태 수집"""
        try:
            # 최근 커밋 정보
            recent_commits = subprocess.run(
                ["git", "log", "--oneline", "-5"], 
                cwd=self.project_root, 
                capture_output=True, 
                text=True
            )
            
            # 현재 상태
            git_status = subprocess.run(
                ["git", "status", "--porcelain"], 
                cwd=self.project_root, 
                capture_output=True, 
                text=True
            )
            
            # 브랜치 정보
            branch_info = subprocess.run(
                ["git", "branch", "--show-current"], 
                cwd=self.project_root, 
                capture_output=True, 
                text=True
            )
            
            return {
                "recent_commits": recent_commits.stdout.strip().split('\n'),
                "uncommitted_changes": git_status.stdout.strip().split('\n') if git_status.stdout.strip() else [],
                "current_branch": branch_info.stdout.strip(),
                "last_commit_hash": recent_commits.stdout.split()[0] if recent_commits.stdout else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_key_files_status(self) -> Dict[str, Any]:
        """핵심 파일들 상태 확인"""
        key_files = [
            "jewelry_stt_ui_v23_real.py",
            "core/real_analysis_engine.py", 
            "windows_demo_monitor.py",
            "demo_capture_system.py",
            "collect_debug_info.py",
            "CLAUDE.md"
        ]
        
        files_status = {}
        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                stat = full_path.stat()
                files_status[file_path] = {
                    "exists": True,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "lines": self.count_lines(full_path)
                }
            else:
                files_status[file_path] = {"exists": False}
        
        return files_status
    
    def count_lines(self, file_path: Path) -> int:
        """파일 라인 수 계산"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def get_running_processes(self) -> Dict[str, Any]:
        """실행 중인 관련 프로세스 확인"""
        try:
            # Streamlit 프로세스
            streamlit_ps = subprocess.run(
                ["pgrep", "-f", "streamlit"], 
                capture_output=True, 
                text=True
            )
            
            processes = {
                "streamlit_running": bool(streamlit_ps.stdout.strip()),
                "streamlit_pids": streamlit_ps.stdout.strip().split('\n') if streamlit_ps.stdout.strip() else []
            }
            
            return processes
        except Exception as e:
            return {"error": str(e)}
    
    def get_recent_activities(self) -> List[str]:
        """최근 활동 내역 추출"""
        activities = []
        
        # 최근 파일 변경사항
        try:
            recent_files = subprocess.run(
                ["find", ".", "-name", "*.py", "-mtime", "-1", "-type", "f"], 
                cwd=self.project_root, 
                capture_output=True, 
                text=True
            )
            
            if recent_files.stdout.strip():
                activities.append(f"최근 24시간 내 수정된 Python 파일: {len(recent_files.stdout.strip().split())}")
        except:
            pass
        
        # Git 활동
        git_status = self.get_git_status()
        if git_status.get("recent_commits"):
            activities.append(f"최근 커밋: {git_status['recent_commits'][0]}")
        
        return activities
    
    def determine_development_phase(self) -> str:
        """현재 개발 단계 판단"""
        git_status = self.get_git_status()
        files_status = self.get_key_files_status()
        
        # 최근 커밋 메시지 분석
        if git_status.get("recent_commits"):
            latest_commit = git_status["recent_commits"][0]
            
            if "브라우저 모니터링" in latest_commit:
                return "브라우저 모니터링 시스템 완성 단계"
            elif "실제 분석" in latest_commit:
                return "실제 AI 분석 시스템 완성 단계"
            elif "Playwright" in latest_commit or "MCP" in latest_commit:
                return "MCP 통합 및 자동화 단계"
        
        # 파일 존재 여부로 판단
        if files_status.get("windows_demo_monitor.py", {}).get("exists"):
            if files_status.get("demo_capture_system.py", {}).get("exists"):
                return "브라우저 모니터링 완성 + Playwright MCP 연동 준비"
        
        return "개발 진행 중"
    
    def get_next_steps(self) -> List[str]:
        """다음 단계 제안"""
        phase = self.determine_development_phase()
        
        if "브라우저 모니터링 완성" in phase:
            return [
                "Playwright MCP 함수 활용 테스트",
                "자동 브라우저 에러 캐쳐 구현", 
                "음성 파일 m4a → wav 변환 해결",
                "성능 최적화 및 안정성 향상"
            ]
        
        return [
            "현재 개발 상태 재평가",
            "미완성 기능 식별",
            "테스트 및 디버깅 수행"
        ]
    
    def save_state(self):
        """현재 상태를 JSON 파일로 저장"""
        state = self.capture_current_state()
        
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"💾 개발 상태 저장: {self.memory_file}")
        return state
    
    def load_state(self) -> Dict[str, Any]:
        """이전 상태 로드"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def update_claude_md(self, state: Dict[str, Any]):
        """CLAUDE.md 파일 자동 업데이트"""
        
        # 현재 CLAUDE.md 읽기
        if self.claude_md.exists():
            with open(self.claude_md, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = ""
        
        # 마지막 업데이트 시간 갱신
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M KST")
        phase = state.get("development_phase", "Unknown")
        
        # 업데이트 섹션 추가/교체
        update_section = f"""
---
**Last Updated**: {timestamp}  
**Version**: v2.3-dev  
**Status**: {phase}  
**Session ID**: {state.get("session_id", "Unknown")}
**Git Status**: {len(state.get("git_status", {}).get("uncommitted_changes", []))} uncommitted changes
**Next Session Goal**: {', '.join(state.get("next_steps", ["개발 계속"])[:2])}"""
        
        # 기존 업데이트 섹션 교체
        if "---\n**Last Updated**" in content:
            content = content.split("---\n**Last Updated**")[0] + update_section
        else:
            content += update_section
        
        # CLAUDE.md 저장
        with open(self.claude_md, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📝 CLAUDE.md 업데이트 완료")
    
    def generate_session_summary(self, state: Dict[str, Any]) -> str:
        """세션 요약 생성"""
        
        summary = f"""
🧠 개발 세션 요약 - {state.get("session_id", "Unknown")}

📊 현재 상태:
- 개발 단계: {state.get("development_phase", "Unknown")}
- Git 상태: {len(state.get("git_status", {}).get("uncommitted_changes", []))}개 uncommitted 변경사항
- 실행 중인 프로세스: {"Streamlit 실행 중" if state.get("running_processes", {}).get("streamlit_running") else "Streamlit 중지"}

🎯 다음 단계:
{chr(10).join(f"- {step}" for step in state.get("next_steps", ["계속 개발"]))}

📁 핵심 파일 상태:
{chr(10).join(f"- {file}: {'✅' if info.get('exists') else '❌'}" for file, info in state.get("file_structure", {}).items())}

💡 재접속 시 확인사항:
1. Streamlit 실행 상태 점검
2. MCP 서버 연결 확인 (특히 Playwright MCP)
3. 최신 Git 상태 확인
4. MCP Memory에서 이전 컨텍스트 검색
"""
        
        return summary.strip()

def main():
    """메인 실행 함수"""
    print("🧠 지속적 메모리 동기화 시스템")
    print("=" * 50)
    
    manager = PersistentMemoryManager()
    
    # 현재 상태 캐쳐
    state = manager.save_state()
    
    # CLAUDE.md 업데이트
    manager.update_claude_md(state)
    
    # 세션 요약 출력
    summary = manager.generate_session_summary(state)
    print(summary)
    
    print("\n✅ 메모리 동기화 완료!")
    print("📋 재접속 시 이 정보를 참조하여 개발을 이어갈 수 있습니다.")

if __name__ == "__main__":
    main()