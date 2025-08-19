#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Preservation System for SOLOMOND AI
컨텍스트 보존 및 연속성 보장 시스템

목적:
1. 핵심 시스템 상태 자동 백업
2. 작업 중단 시점 기록 및 복구
3. 개발 컨텍스트 영구 보존
4. 제자리 돌기 방지
"""

import json
import time
from datetime import datetime
from pathlib import Path

class ContextPreservationSystem:
    
    def __init__(self):
        self.context_file = Path("SESSION_CONTEXT_PRESERVATION.json")
        self.backup_dir = Path("context_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def save_current_context(self, user_goal, current_task, system_state):
        """현재 컨텍스트 저장"""
        
        context = {
            "timestamp": datetime.now().isoformat(),
            "user_goal": user_goal,
            "current_task": current_task,
            "system_state": system_state,
            "core_systems_status": self.check_core_systems(),
            "important_notes": [
                "Dual Brain System = Google Calendar + AI Insights",
                "Main Dashboard = Calendar Widget + Analysis History", 
                "File upload is just a tool, not the main purpose",
                "Core workflow: Analysis -> Calendar -> AI Insights -> Future Planning"
            ],
            "recovery_instructions": {
                "main_dashboard": "streamlit run solomond_ai_main_dashboard.py --server.port 8500",
                "dual_brain": "python dual_brain_integration.py", 
                "ai_insights": "python ai_insights_engine.py",
                "calendar_connector": "python google_calendar_connector.py"
            }
        }
        
        # 현재 컨텍스트 저장
        with open(self.context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        
        # 타임스탬프 백업도 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"context_{timestamp}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
            
        return context
    
    def check_core_systems(self):
        """핵심 시스템 빠른 체크"""
        systems = {
            "ai_insights_engine.py": Path("ai_insights_engine.py").exists(),
            "dual_brain_integration.py": Path("dual_brain_integration.py").exists(),
            "solomond_ai_main_dashboard.py": Path("solomond_ai_main_dashboard.py").exists(),
            "google_calendar_connector.py": Path("google_calendar_connector.py").exists()
        }
        return systems
    
    def load_latest_context(self):
        """최신 컨텍스트 로드"""
        if self.context_file.exists():
            with open(self.context_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def generate_recovery_guide(self):
        """복구 가이드 생성"""
        context = self.load_latest_context()
        if not context:
            return "No context found"
            
        guide = f"""
# SOLOMOND AI Context Recovery Guide

## Last Session Info
- Timestamp: {context['timestamp']}
- User Goal: {context['user_goal']}  
- Current Task: {context['current_task']}

## Core Systems Status
"""
        for system, status in context['core_systems_status'].items():
            status_icon = "OK" if status else "MISSING"
            guide += f"- {system}: {status_icon}\n"
            
        guide += f"""
## Recovery Commands
"""
        for name, command in context['recovery_instructions'].items():
            guide += f"- {name}: {command}\n"
            
        guide += f"""
## Important Reminders
"""
        for note in context['important_notes']:
            guide += f"- {note}\n"
            
        return guide

def save_session_context():
    """현재 세션 컨텍스트 저장"""
    
    cps = ContextPreservationSystem()
    
    # 현재 상황 정의
    user_goal = "Dual Brain Second Brain System with Google Calendar + AI Insights"
    current_task = "File upload issue caused deviation from core mission"
    system_state = "Core systems intact, need to refocus on main objective"
    
    context = cps.save_current_context(user_goal, current_task, system_state)
    
    print("Context Preservation Complete")
    print("=" * 30)
    print(f"User Goal: {user_goal}")
    print(f"Current Task: {current_task}")
    print(f"Saved to: {cps.context_file}")
    
    # 복구 가이드 생성
    guide = cps.generate_recovery_guide()
    with open("CONTEXT_RECOVERY_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(guide)
    
    print("Recovery Guide: CONTEXT_RECOVERY_GUIDE.md")
    
    return context

if __name__ == "__main__":
    save_session_context()