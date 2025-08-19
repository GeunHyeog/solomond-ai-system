#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from pathlib import Path

def check_core_systems():
    """핵심 시스템 상태 검사"""
    
    print("SOLOMOND AI System Integrity Check")
    print("=" * 40)
    
    # 핵심 시스템 목록
    core_systems = [
        "ai_insights_engine.py",
        "google_calendar_connector.py", 
        "dual_brain_integration.py",
        "solomond_ai_main_dashboard.py",
        "shared/ollama_interface.py",
        "database_adapter.py"
    ]
    
    working = []
    missing = []
    
    for system in core_systems:
        if Path(system).exists():
            working.append(system)
            print(f"OK {system}")
        else:
            missing.append(system)
            print(f"MISSING {system}")
    
    print()
    print(f"Working systems: {len(working)}")
    print(f"Missing systems: {len(missing)}")
    
    # 상태 저장
    status = {
        "timestamp": datetime.now().isoformat(),
        "working": working,
        "missing": missing,
        "total": len(core_systems)
    }
    
    with open("system_status.json", "w") as f:
        json.dump(status, f, indent=2)
    
    # 복구 가이드 생성
    if missing:
        print("\nRecovery needed for:")
        for system in missing:
            print(f"- {system}")
            
        print("\nTo recover:")
        print("1. Check system_integrity_backups/ folder")
        print("2. Restore missing files from backup")
        print("3. Run this check again")
    else:
        print("\nAll core systems are present!")
        print("Safe to proceed with operations.")
    
    return status

if __name__ == "__main__":
    check_core_systems()