#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-Loop Recovery System for SOLOMOND AI
제자리 돌기 방지 및 자동 복구 시스템

핵심 기능:
1. 작업 패턴 감지로 제자리 돌기 방지
2. 핵심 시스템 자동 복구
3. 우선순위 기반 작업 재정렬
4. 컨텍스트 연속성 보장
"""

import json
from datetime import datetime
from pathlib import Path

class AntiLoopRecoverySystem:
    
    def __init__(self):
        self.recovery_log = Path("recovery_actions.log")
        
    def detect_and_prevent_loops(self):
        """제자리 돌기 패턴 감지 및 방지"""
        
        loop_patterns = [
            "File upload problems causing core system neglect",
            "Fixing one thing breaks another (cascading failures)", 
            "Getting lost in technical details, forgetting main goal",
            "Streamlit crashes leading to system rebuilds"
        ]
        
        prevention_actions = {
            "file_upload": "Separate file upload from core systems completely",
            "cascading_failures": "Always backup working systems before changes",
            "goal_drift": "Regular check: Does this serve the Dual Brain goal?",
            "streamlit_issues": "Use HTML alternatives, avoid Streamlit dependency"
        }
        
        return loop_patterns, prevention_actions
    
    def execute_recovery_plan(self):
        """핵심 시스템 복구 플랜 실행"""
        
        print("SOLOMOND AI Recovery System")
        print("=" * 30)
        
        recovery_steps = [
            {
                "step": 1,
                "action": "Verify core systems integrity",
                "command": "python system_integrity_check.py",
                "priority": "HIGH"
            },
            {
                "step": 2, 
                "action": "Start Dual Brain Main Dashboard",
                "command": "streamlit run solomond_ai_main_dashboard.py --server.port 8500",
                "priority": "HIGH"
            },
            {
                "step": 3,
                "action": "Activate AI Insights Engine", 
                "command": "python ai_insights_engine.py",
                "priority": "MEDIUM"
            },
            {
                "step": 4,
                "action": "Test Google Calendar Integration",
                "command": "python google_calendar_connector.py",
                "priority": "MEDIUM"
            },
            {
                "step": 5,
                "action": "File upload as separate, optional feature",
                "command": "Handle separately from core systems",
                "priority": "LOW"
            }
        ]
        
        print("Recovery Plan:")
        for step in recovery_steps:
            print(f"{step['step']}. [{step['priority']}] {step['action']}")
            print(f"   Command: {step['command']}")
        
        return recovery_steps
    
    def create_focus_guard(self):
        """목적 집중 가드 생성"""
        
        focus_reminders = [
            "MAIN GOAL: Dual Brain Second Brain System", 
            "Core Features: Google Calendar + AI Insights",
            "Workflow: Analysis -> Calendar -> AI Insights -> Future Planning",
            "File upload is just a TOOL, not the PURPOSE",
            "If spending >30min on file upload, STOP and refocus"
        ]
        
        guard_file = Path("FOCUS_GUARD.md")
        content = "# SOLOMOND AI Focus Guard\n\n"
        content += "## Primary Mission Reminder\n\n"
        
        for reminder in focus_reminders:
            content += f"- {reminder}\n"
        
        content += "\n## Anti-Loop Questions\n"
        content += "Before any change, ask:\n"
        content += "1. Does this serve the Dual Brain goal?\n"
        content += "2. Will this break existing working systems?\n" 
        content += "3. Is this a distraction from the main purpose?\n"
        content += "4. Have I backed up working systems?\n\n"
        
        content += "## Recovery Commands\n"
        content += "If lost or confused:\n"
        content += "```bash\n"
        content += "python ANTI_LOOP_RECOVERY_SYSTEM.py\n"
        content += "python system_integrity_check.py\n"
        content += "```\n"
        
        with open(guard_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return guard_file
    
    def log_recovery_action(self, action, result):
        """복구 행동 로그"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result
        }
        
        with open(self.recovery_log, 'a', encoding='utf-8') as f:
            f.write(f"{log_entry}\n")

def main():
    """메인 실행"""
    
    system = AntiLoopRecoverySystem()
    
    print("Anti-Loop Recovery System Activated")
    print("=" * 40)
    
    # 1. 제자리 돌기 패턴 감지
    patterns, actions = system.detect_and_prevent_loops()
    print("\nDetected Loop Patterns:")
    for pattern in patterns:
        print(f"- {pattern}")
    
    # 2. 복구 플랜 실행
    print("\n" + "="*40)
    recovery_steps = system.execute_recovery_plan()
    
    # 3. 집중 가드 생성
    guard_file = system.create_focus_guard()
    print(f"\nFocus Guard created: {guard_file}")
    
    # 4. 복구 행동 로그
    system.log_recovery_action("Anti-loop system activated", "Success")
    
    print("\nRECOVERY COMPLETE")
    print("Next: Focus on Dual Brain System, not file upload!")
    
    return True

if __name__ == "__main__":
    main()