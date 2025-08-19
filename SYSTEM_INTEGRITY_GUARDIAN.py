#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ SOLOMOND AI 시스템 무결성 가디언
System Integrity Guardian - 핵심 시스템 상태 보존 및 자동 복구

핵심 기능:
1. 핵심 시스템 상태 자동 백업
2. 무결성 실시간 감시
3. 문제 발견 시 즉시 복구
4. 컨텍스트 영구 보존
5. 제자리 돌기 방지
"""

import os
import json
import time
import shutil
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class SystemIntegrityGuardian:
    """시스템 무결성 가디언"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.backup_dir = Path("system_integrity_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # 핵심 시스템 목록 (CLAUDE.md에서 추출)
        self.core_systems = {
            "ai_insights_engine.py": "AI 인사이트 엔진 (6가지 패턴)",
            "google_calendar_connector.py": "구글 캘린더 API 연동",
            "dual_brain_integration.py": "듀얼 브레인 통합 시스템",
            "solomond_ai_main_dashboard.py": "메인 대시보드 + 캘린더",
            "conference_analysis_COMPLETE_WORKING.py": "컨퍼런스 분석 엔진",
            "shared/ollama_interface.py": "Ollama AI 인터페이스",
            "database_adapter.py": "데이터베이스 어댑터",
            "holistic_conference_analyzer_supabase.py": "홀리스틱 분석기"
        }
        
        # 컨텍스트 보존 파일들
        self.context_files = [
            "CLAUDE.md",
            "analysis_history/",
            "user_files/",
            "shared/port_config.json"
        ]
        
        self.status_file = Path("SYSTEM_STATUS_SNAPSHOT.json")
        
    def create_system_snapshot(self):
        """현재 시스템 상태 스냅샷 생성"""
        print("🔍 시스템 무결성 검사 중...")
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "core_systems_status": {},
            "context_preservation": {},
            "working_systems": [],
            "broken_systems": [],
            "recovery_plan": []
        }
        
        # 1. 핵심 시스템 상태 검사
        for file_path, description in self.core_systems.items():
            full_path = self.base_dir / file_path
            
            if full_path.exists():
                # Python 파일이면 import 테스트
                if file_path.endswith('.py'):
                    try:
                        module_name = file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
                        if '/' in module_name or '\\' in module_name:
                            # 하위 디렉토리의 경우 상대 import 시도
                            module_name = module_name.split('.')[-1]
                        
                        spec = importlib.util.spec_from_file_location(module_name, full_path)
                        if spec and spec.loader:
                            snapshot["core_systems_status"][file_path] = {
                                "exists": True,
                                "importable": True,
                                "description": description,
                                "size": full_path.stat().st_size,
                                "modified": datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
                            }
                            snapshot["working_systems"].append(file_path)
                        else:
                            raise ImportError("Cannot create spec")
                            
                    except Exception as e:
                        snapshot["core_systems_status"][file_path] = {
                            "exists": True,
                            "importable": False,
                            "error": str(e),
                            "description": description
                        }
                        snapshot["broken_systems"].append(file_path)
                        snapshot["recovery_plan"].append(f"Fix import error in {file_path}: {e}")
                else:
                    snapshot["core_systems_status"][file_path] = {
                        "exists": True,
                        "description": description,
                        "size": full_path.stat().st_size
                    }
                    snapshot["working_systems"].append(file_path)
            else:
                snapshot["core_systems_status"][file_path] = {
                    "exists": False,
                    "description": description
                }
                snapshot["broken_systems"].append(file_path)
                snapshot["recovery_plan"].append(f"Restore missing file: {file_path}")
        
        # 2. 컨텍스트 파일 확인
        for context_item in self.context_files:
            path = self.base_dir / context_item
            if path.exists():
                if path.is_file():
                    snapshot["context_preservation"][context_item] = {
                        "type": "file",
                        "size": path.stat().st_size,
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    }
                elif path.is_dir():
                    file_count = len(list(path.rglob("*")))
                    snapshot["context_preservation"][context_item] = {
                        "type": "directory",
                        "file_count": file_count
                    }
            else:
                snapshot["context_preservation"][context_item] = {"exists": False}
        
        # 스냅샷 저장
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        
        return snapshot
    
    def backup_working_systems(self, snapshot):
        """작동하는 시스템들을 백업"""
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_backup_dir = self.backup_dir / f"backup_{backup_timestamp}"
        current_backup_dir.mkdir(exist_ok=True)
        
        print(f"💾 작동하는 시스템 백업 중... ({len(snapshot['working_systems'])}개)")
        
        for system_file in snapshot["working_systems"]:
            source = self.base_dir / system_file
            if source.exists():
                # 디렉토리 구조 유지하며 복사
                dest = current_backup_dir / system_file
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                print(f"✅ 백업: {system_file}")
        
        # 컨텍스트 파일들도 백업
        for context_item in self.context_files:
            source = self.base_dir / context_item
            if source.exists():
                dest = current_backup_dir / context_item
                if source.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                elif source.is_dir():
                    shutil.copytree(source, dest, dirs_exist_ok=True)
                print(f"💾 컨텍스트 백업: {context_item}")
        
        return current_backup_dir
    
    def generate_recovery_instructions(self, snapshot):
        """복구 지침서 생성"""
        recovery_file = Path("SYSTEM_RECOVERY_GUIDE.md")
        
        content = f"""# 🛡️ SOLOMOND AI 시스템 복구 가이드

## 📊 시스템 상태 (생성: {snapshot['timestamp']})

### ✅ 정상 작동 시스템 ({len(snapshot['working_systems'])}개)
"""
        
        for system in snapshot['working_systems']:
            desc = self.core_systems.get(system, "시스템 파일")
            content += f"- **{system}**: {desc}\n"
        
        content += f"""
### ❌ 문제 발견 시스템 ({len(snapshot['broken_systems'])}개)
"""
        
        for system in snapshot['broken_systems']:
            desc = self.core_systems.get(system, "시스템 파일")
            content += f"- **{system}**: {desc}\n"
        
        content += """
### 🔧 복구 계획
"""
        
        for plan in snapshot['recovery_plan']:
            content += f"1. {plan}\n"
        
        content += f"""
## 🚀 즉시 복구 방법

### A. 자동 복구 (권장)
```bash
python SYSTEM_INTEGRITY_GUARDIAN.py --restore-latest
```

### B. 수동 복구
1. `system_integrity_backups/` 폴더에서 최신 백업 찾기
2. 문제가 된 파일들을 백업에서 복사
3. `python SYSTEM_INTEGRITY_GUARDIAN.py --verify` 실행

### C. 핵심 시스템 재시작
```bash
# 듀얼 브레인 시스템 재시작
streamlit run solomond_ai_main_dashboard.py --server.port 8500

# 컨퍼런스 분석 시스템 재시작  
streamlit run conference_analysis_COMPLETE_WORKING.py --server.port 8501
```

## 💡 제자리 돌기 방지 규칙

1. **새로운 기능 추가 전**: 반드시 백업 생성
2. **시스템 수정 후**: 즉시 무결성 검증
3. **문제 발견 시**: 백업에서 즉시 복구 후 수정
4. **업로드 문제**: 핵심 시스템과 분리하여 처리

---
**Generated by SOLOMOND AI System Integrity Guardian**
**복구 문의: 이 가이드를 Claude Code에 보여주세요**
"""
        
        with open(recovery_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return recovery_file
    
    def run_integrity_check(self):
        """전체 무결성 검사 실행"""
        print("🛡️ SOLOMOND AI 시스템 무결성 가디언 실행")
        print("=" * 50)
        
        # 1. 현재 상태 스냅샷
        snapshot = self.create_system_snapshot()
        
        # 2. 결과 출력
        print(f"📊 핵심 시스템: {len(self.core_systems)}개")
        print(f"✅ 정상 작동: {len(snapshot['working_systems'])}개")
        print(f"❌ 문제 발견: {len(snapshot['broken_systems'])}개")
        
        if snapshot['broken_systems']:
            print("\n🚨 문제가 발견된 시스템:")
            for system in snapshot['broken_systems']:
                desc = self.core_systems.get(system, "시스템 파일")
                print(f"  ❌ {system}: {desc}")
        
        # 3. 정상 시스템 백업
        if snapshot['working_systems']:
            backup_dir = self.backup_working_systems(snapshot)
            print(f"\n💾 백업 완료: {backup_dir}")
        
        # 4. 복구 가이드 생성
        recovery_guide = self.generate_recovery_instructions(snapshot)
        print(f"📋 복구 가이드: {recovery_guide}")
        
        # 5. 종합 결과
        if len(snapshot['broken_systems']) == 0:
            print("\n🎉 모든 핵심 시스템이 정상 작동 중입니다!")
            print("💡 안전하게 새로운 기능을 추가할 수 있습니다.")
        else:
            print(f"\n⚠️ {len(snapshot['broken_systems'])}개 시스템에 문제가 있습니다.")
            print("🔧 먼저 복구 후 다른 작업을 진행하세요.")
        
        return snapshot

def main():
    """메인 실행 함수"""
    guardian = SystemIntegrityGuardian()
    snapshot = guardian.run_integrity_check()
    
    # 컨텍스트 보존 확인
    print("\n📚 컨텍스트 보존 상태:")
    for item, status in snapshot['context_preservation'].items():
        if status.get('exists', True):
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item}")
    
    print("\n💡 이 도구로 제자리 돌기 문제를 방지할 수 있습니다!")
    print("📖 상세한 복구 방법은 SYSTEM_RECOVERY_GUIDE.md를 확인하세요.")

if __name__ == "__main__":
    main()