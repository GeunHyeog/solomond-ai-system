#!/usr/bin/env python3
"""
윈도우 캐쳐 결과를 WSL로 동기화하는 스크립트
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def sync_windows_captures():
    """윈도우 캐쳐 결과를 WSL demo_captures로 동기화"""
    
    print("🔄 윈도우 캐쳐 결과 동기화")
    print("=" * 50)
    
    # 경로 설정
    windows_captures_path = Path("/mnt/c/Users/PC_58410/solomond-ai-system/windows_captures")
    wsl_captures_path = Path("/home/solomond/claude/solomond-ai-system/demo_captures")
    
    # 대체 경로들
    alternative_paths = [
        Path("/mnt/c/Users/PC_58410/Desktop/solomond-ai-system/windows_captures"),
        Path("/mnt/c/Users/PC_58410/Documents/solomond-ai-system/windows_captures"),
        Path("/mnt/c/Users/PC_58410/Downloads/solomond-ai-system/windows_captures"),
        Path("./windows_captures"),  # 현재 디렉토리
    ]
    
    # 윈도우 캐쳐 폴더 찾기
    source_path = None
    
    if windows_captures_path.exists():
        source_path = windows_captures_path
    else:
        print("🔍 윈도우 캐쳐 폴더 검색 중...")
        for alt_path in alternative_paths:
            if alt_path.exists():
                print(f"✅ 발견: {alt_path}")
                source_path = alt_path
                break
    
    if not source_path:
        print("❌ 윈도우 캐쳐 폴더를 찾을 수 없습니다.")
        print("📁 다음 위치들을 확인했습니다:")
        for path in [windows_captures_path] + alternative_paths:
            print(f"   - {path}")
        print()
        print("💡 수동 동기화 방법:")
        print("   1. 윈도우에서 windows_captures 폴더 위치 확인")
        print("   2. 파일을 WSL로 복사:")
        print("      copy windows_captures\\*.* \\\\wsl$\\Ubuntu\\home\\solomond\\claude\\solomond-ai-system\\demo_captures\\")
        return False
    
    # WSL 대상 폴더 생성
    wsl_captures_path.mkdir(exist_ok=True)
    
    print(f"📂 소스: {source_path}")
    print(f"📂 대상: {wsl_captures_path}")
    print()
    
    # 파일 동기화
    synced_files = []
    skipped_files = []
    
    for file_path in source_path.glob("*"):
        if file_path.is_file():
            dest_path = wsl_captures_path / file_path.name
            
            # 이미 존재하는 파일 확인
            if dest_path.exists():
                # 파일 크기와 수정 시간 비교
                src_stat = file_path.stat()
                dest_stat = dest_path.stat()
                
                if src_stat.st_size == dest_stat.st_size and src_stat.st_mtime <= dest_stat.st_mtime:
                    skipped_files.append(file_path.name)
                    continue
            
            try:
                shutil.copy2(str(file_path), str(dest_path))
                synced_files.append(file_path.name)
                print(f"✅ 동기화: {file_path.name}")
            except Exception as e:
                print(f"❌ 실패: {file_path.name} - {e}")
    
    print()
    print("📊 동기화 결과:")
    print(f"   ✅ 성공: {len(synced_files)}개")
    print(f"   ⏭️ 건너뜀: {len(skipped_files)}개")
    
    if synced_files:
        print(f"\n📁 새로 동기화된 파일:")
        for file_name in synced_files[:10]:  # 처음 10개만 표시
            print(f"   - {file_name}")
        if len(synced_files) > 10:
            print(f"   ... 외 {len(synced_files) - 10}개")
    
    # 최신 세션 리포트 분석
    analyze_latest_session(wsl_captures_path)
    
    return True

def analyze_latest_session(captures_path):
    """최신 세션 리포트 분석"""
    
    # 윈도우 세션 리포트 찾기
    reports = list(captures_path.glob("windows_session_report_*.json"))
    
    if not reports:
        print("\n💡 윈도우 세션 리포트가 없습니다.")
        return
    
    # 최신 리포트 선택
    latest_report = max(reports, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_report, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        print(f"\n📊 최신 윈도우 세션 분석: {latest_report.name}")
        print("-" * 50)
        
        session_info = report_data.get('session_info', {})
        activity = report_data.get('activity_summary', {})
        
        print(f"🎯 세션 정보:")
        print(f"   • 세션 ID: {session_info.get('session_id', 'N/A')}")
        print(f"   • 총 캐쳐: {session_info.get('total_captures', 0)}개")
        print(f"   • 소요 시간: {session_info.get('duration', 'N/A')}")
        print(f"   • 플랫폼: {session_info.get('platform', 'N/A')}")
        
        print(f"\n🖥️ 활동 분석:")
        print(f"   • Streamlit 상호작용: {activity.get('streamlit_interactions', 0)}회")
        print(f"   • 상호작용 비율: {activity.get('streamlit_interaction_rate', '0%')}")
        print(f"   • 브라우저 사용: {', '.join(activity.get('browser_usage', {}).keys()) or '없음'}")
        print(f"   • 윈도우 전환: {activity.get('unique_windows', 0)}개")
        
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            print(f"\n💡 시스템 평가:")
            for rec in recommendations[:3]:
                print(f"   {rec}")
        
        print()
        
    except Exception as e:
        print(f"⚠️ 리포트 분석 실패: {e}")

def create_unified_report():
    """WSL과 윈도우 캐쳐 결과를 통합한 리포트 생성"""
    
    captures_path = Path("/home/solomond/claude/solomond-ai-system/demo_captures")
    
    # WSL 리포트 찾기
    wsl_reports = list(captures_path.glob("session_report_*.json"))
    windows_reports = list(captures_path.glob("windows_session_report_*.json"))
    
    if not wsl_reports and not windows_reports:
        print("📭 생성할 리포트가 없습니다.")
        return
    
    unified_data = {
        'unified_report_timestamp': datetime.now().isoformat(),
        'wsl_sessions': [],
        'windows_sessions': [],
        'summary': {}
    }
    
    # WSL 세션 데이터 수집
    for report_path in wsl_reports:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                unified_data['wsl_sessions'].append({
                    'file': report_path.name,
                    'platform': 'wsl',
                    'session_info': data.get('session_info', {}),
                    'activity_summary': data.get('activity_summary', {})
                })
        except Exception as e:
            print(f"⚠️ WSL 리포트 읽기 실패 {report_path.name}: {e}")
    
    # 윈도우 세션 데이터 수집
    for report_path in windows_reports:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                unified_data['windows_sessions'].append({
                    'file': report_path.name,
                    'platform': 'windows',
                    'session_info': data.get('session_info', {}),
                    'activity_summary': data.get('activity_summary', {})
                })
        except Exception as e:
            print(f"⚠️ 윈도우 리포트 읽기 실패 {report_path.name}: {e}")
    
    # 요약 정보 생성
    total_sessions = len(unified_data['wsl_sessions']) + len(unified_data['windows_sessions'])
    total_captures = sum(session['session_info'].get('total_captures', 0) 
                        for session in unified_data['wsl_sessions'] + unified_data['windows_sessions'])
    
    unified_data['summary'] = {
        'total_sessions': total_sessions,
        'total_captures': total_captures,
        'wsl_sessions_count': len(unified_data['wsl_sessions']),
        'windows_sessions_count': len(unified_data['windows_sessions']),
        'platforms_used': ['wsl'] if wsl_reports else [] + ['windows'] if windows_reports else []
    }
    
    # 통합 리포트 저장
    unified_report_path = captures_path / f"unified_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(unified_report_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 통합 리포트 생성: {unified_report_path}")
    print(f"   • 총 세션: {total_sessions}개")
    print(f"   • 총 캐쳐: {total_captures}개")
    print(f"   • WSL 세션: {len(unified_data['wsl_sessions'])}개")
    print(f"   • 윈도우 세션: {len(unified_data['windows_sessions'])}개")

def main():
    """메인 실행 함수"""
    
    print("🔄 솔로몬드 AI - 캐쳐 결과 동기화 시스템")
    print("=" * 60)
    
    # 1. 윈도우 결과 동기화
    sync_success = sync_windows_captures()
    
    if sync_success:
        print("\n📊 통합 리포트 생성 중...")
        create_unified_report()
    
    print("\n✅ 동기화 완료!")
    print("📁 모든 캐쳐 결과는 demo_captures/ 폴더에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()