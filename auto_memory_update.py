#!/usr/bin/env python3
"""
자동 메모리 업데이트 시스템
Git 커밋과 개발 활동을 자동으로 MCP Memory에 동기화
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def get_recent_commits(project_root: str, count: int = 5) -> List[Dict[str, str]]:
    """최근 커밋 정보 수집"""
    try:
        result = subprocess.run(
            ["git", "log", f"--oneline", f"-{count}", "--pretty=format:%H|%s|%ad", "--date=iso"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                hash_val, message, date = line.split('|', 2)
                commits.append({
                    "hash": hash_val,
                    "message": message,
                    "date": date,
                    "timestamp": datetime.now().isoformat()
                })
        
        return commits
    except Exception as e:
        print(f"Git 정보 수집 실패: {e}")
        return []

def analyze_commit_patterns(commits: List[Dict[str, str]]) -> Dict[str, Any]:
    """커밋 패턴 분석"""
    
    patterns = {
        "브라우저 모니터링": ["브라우저", "모니터링", "캐쳐", "스크린샷"],
        "실제 분석 시스템": ["실제", "분석", "Whisper", "EasyOCR"],
        "MCP 통합": ["MCP", "Playwright", "메모리", "통합"],
        "디버깅 시스템": ["디버깅", "에러", "수집", "로그"],
        "성능 최적화": ["최적화", "성능", "메모리", "CPU"]
    }
    
    analysis = {
        "total_commits": len(commits),
        "recent_themes": [],
        "development_trends": [],
        "last_major_milestone": None
    }
    
    for commit in commits:
        message = commit["message"].lower()
        
        for theme, keywords in patterns.items():
            if any(keyword.lower() in message for keyword in keywords):
                analysis["recent_themes"].append({
                    "theme": theme,
                    "commit": commit["message"],
                    "date": commit["date"]
                })
                break
    
    # 최근 주요 마일스톤 식별
    major_keywords = ["완성", "구현", "완료", "통합"]
    for commit in commits:
        if any(keyword in commit["message"] for keyword in major_keywords):
            analysis["last_major_milestone"] = commit
            break
    
    return analysis

def create_development_timeline(commits: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """개발 타임라인 생성"""
    timeline = []
    
    for commit in commits:
        timeline_entry = {
            "timestamp": commit["date"],
            "hash": commit["hash"][:8],
            "milestone": commit["message"],
            "impact": "major" if any(word in commit["message"].lower() 
                                   for word in ["완성", "구현", "완료", "시스템"]) else "minor"
        }
        timeline.append(timeline_entry)
    
    return timeline

def extract_current_capabilities() -> List[str]:
    """현재 시스템 기능 추출"""
    project_root = Path("/home/solomond/claude/solomond-ai-system")
    
    capabilities = []
    
    # 주요 파일 존재 여부로 기능 판단
    capability_files = {
        "jewelry_stt_ui_v23_real.py": "실제 AI 분석 UI",
        "core/real_analysis_engine.py": "Whisper STT + EasyOCR 엔진",
        "windows_demo_monitor.py": "윈도우 브라우저 모니터링",
        "demo_capture_system.py": "Playwright 자동 캐쳐",
        "collect_debug_info.py": "자동 디버깅 정보 수집",
        "sync_windows_captures.py": "WSL-윈도우 데이터 동기화"
    }
    
    for file_path, capability in capability_files.items():
        if (project_root / file_path).exists():
            capabilities.append(capability)
    
    return capabilities

def generate_memory_entities() -> List[Dict[str, Any]]:
    """MCP Memory용 엔티티 생성"""
    
    project_root = "/home/solomond/claude/solomond-ai-system"
    commits = get_recent_commits(project_root, 10)
    analysis = analyze_commit_patterns(commits)
    timeline = create_development_timeline(commits)
    capabilities = extract_current_capabilities()
    
    current_session = {
        "name": f"개발세션_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "entityType": "development_session",
        "observations": [
            f"세션 시작: {datetime.now().isoformat()}",
            f"총 {analysis['total_commits']}개 최근 커밋 분석",
            f"현재 시스템 기능: {len(capabilities)}개 주요 컴포넌트",
            f"마지막 주요 마일스톤: {analysis['last_major_milestone']['message'] if analysis['last_major_milestone'] else '없음'}",
            "지속적 메모리 동기화 시스템 활성화",
            "Playwright MCP 통합 완료 상태",
            "브라우저 모니터링 및 자동 캐쳐 시스템 운영"
        ]
    }
    
    system_status = {
        "name": "솔로몬드AI_시스템상태_2025",
        "entityType": "system_status", 
        "observations": [
            f"활성 기능: {', '.join(capabilities)}",
            f"개발 트렌드: {', '.join([theme['theme'] for theme in analysis['recent_themes'][:3]])}",
            "실제 AI 분석 시스템으로 완전 전환 완료",
            "25개 파일 동시 처리 테스트 완료 (이미지 23개 성공, 음성 2개 포맷 이슈)",
            "GPU 메모리 최적화로 CPU 모드 안정적 작동",
            "MCP 기반 완전 자동화 브라우저 모니터링 준비",
            f"GitHub 최신 동기화: {commits[0]['hash'][:8]} - {commits[0]['message']}"
        ]
    }
    
    development_roadmap = {
        "name": "개발로드맵_2025Q3",
        "entityType": "roadmap",
        "observations": [
            "1단계: Playwright MCP 완전 활용 (진행 중)",
            "2단계: 음성 파일 호환성 문제 해결 (m4a → wav)",
            "3단계: 대용량 파일 처리 성능 최적화",
            "4단계: 모바일 지원 및 크로스 플랫폼 확장",
            "5단계: AI 분석 정확도 향상 및 특화 모델 통합",
            f"타임라인 마일스톤: {len(timeline)}개 주요 개발 포인트 추적 중"
        ]
    }
    
    return [current_session, system_status, development_roadmap]

def generate_memory_relations() -> List[Dict[str, str]]:
    """MCP Memory용 관계 생성"""
    return [
        {
            "from": f"개발세션_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "to": "솔로몬드AI_시스템상태_2025",
            "relationType": "업데이트함"
        },
        {
            "from": "솔로몬드AI_시스템상태_2025", 
            "to": "개발로드맵_2025Q3",
            "relationType": "진행함"
        },
        {
            "from": "개발로드맵_2025Q3",
            "to": f"개발세션_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "relationType": "가이드함"
        }
    ]

def main():
    """메인 실행 함수"""
    print("🔄 자동 메모리 업데이트 시스템")
    print("=" * 50)
    
    # 엔티티 및 관계 생성
    entities = generate_memory_entities()
    relations = generate_memory_relations()
    
    print("📊 생성된 메모리 엔티티:")
    for entity in entities:
        print(f"- {entity['name']} ({entity['entityType']})")
        for obs in entity['observations'][:3]:
            print(f"  • {obs}")
        print()
    
    print("🔗 생성된 관계:")
    for relation in relations:
        print(f"- {relation['from']} → {relation['to']} ({relation['relationType']})")
    
    # JSON 파일로 저장 (MCP Memory 수동 입력용)
    memory_data = {
        "entities": entities,
        "relations": relations,
        "generated_at": datetime.now().isoformat(),
        "session_summary": "자동 메모리 동기화 완료 - 개발 단계 추적 활성화"
    }
    
    output_file = Path("/home/solomond/claude/solomond-ai-system/memory_update.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(memory_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 메모리 데이터 저장: {output_file}")
    print("✅ 자동 메모리 업데이트 완료!")
    
    return memory_data

if __name__ == "__main__":
    main()