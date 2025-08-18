#!/usr/bin/env python3
"""
자동 시연 캐쳐 (비대화형)
바로 실행되는 시연 캐쳐
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from demo_capture_system import DemoCaptureSystem
from config import SETTINGS

async def auto_demo_capture():
    """자동 시연 캐쳐 (5분간)"""
    
    print("🎭 솔로몬드 AI 자동 시연 캐쳐")
    print("=" * 60)
    print("💎 브라우저에서 시연한 내용을 5분간 자동 캡처합니다!")
    print()
    
    # 기본 설정
    url = "http://f"localhost:{SETTINGS['PORT']}""
    duration = 5  # 5분
    
    print(f"🚀 자동 설정:")
    print(f"   📍 URL: {url}")
    print(f"   ⏰ 시간: {duration}분")
    print(f"   📸 간격: 3초")
    print()
    
    # 캐쳐 시스템 시작
    capture_system = DemoCaptureSystem(streamlit_url=url)
    
    print("🌐 브라우저를 열고 시연을 시작합니다...")
    print("💡 자유롭게 시연하세요! 자동으로 캐쳐됩니다.")
    print("🛑 중간에 중단하려면 Ctrl+C를 누르세요.")
    print()
    
    try:
        # 캐쳐 세션 시작
        session_report = await capture_system.start_capture_session(duration_minutes=duration)
        
        if session_report:
            print("\n" + "=" * 60)
            print("📊 시연 캐쳐 완료!")
            print("=" * 60)
            
            # 요약 정보 출력
            display_summary(session_report, capture_system.session_id)
            
        else:
            print("❌ 시연 캐쳐가 실패했습니다.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
        
        # 부분 리포트 생성
        if capture_system.captures:
            print("📊 부분 리포트 생성 중...")
            session_report = await capture_system.generate_session_report()
            display_summary(session_report, capture_system.session_id)
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

def display_summary(session_report: dict, session_id: str):
    """요약 정보 표시"""
    
    session_info = session_report['session_info']
    activity = session_report['activity_summary']
    files = session_report['file_uploads']
    results = session_report['analysis_results']
    
    print(f"📈 캐쳐 통계:")
    print(f"   • 총 캐쳐: {session_info['total_captures']}개")
    print(f"   • 소요 시간: {session_info['duration']}")
    print(f"   • 세션 ID: {session_info['session_id']}")
    print()
    
    print(f"🎯 시연 활동:")
    print(f"   • 사용한 탭: {', '.join(activity['tabs_used']) if activity['tabs_used'] else '없음'}")
    print(f"   • 총 상호작용: {activity['total_interactions']}회")
    print()
    
    print(f"📁 파일 업로드:")
    print(f"   • 총 파일: {files['total_files']}개")
    print(f"   • 음성 파일: {len(files['audio_files'])}개")
    print(f"   • 이미지 파일: {len(files['image_files'])}개")
    if files['all_files']:
        print(f"   • 파일 목록: {', '.join(files['all_files'])}")
    print()
    
    print(f"🔍 분석 결과:")
    print(f"   • 성공한 분석: {results['success_count']}회")
    print(f"   • 실패한 분석: {results['error_count']}회")
    if (results['success_count'] + results['error_count']) > 0:
        success_rate = results['success_count'] / (results['success_count'] + results['error_count']) * 100
        print(f"   • 성공률: {success_rate:.1f}%")
    print()
    
    print("💡 추천사항:")
    for rec in session_report['recommendations']:
        print(f"   {rec}")
    print()
    
    print(f"📄 저장된 파일:")
    print(f"   📊 리포트: demo_captures/session_report_{session_id}.json")
    print(f"   📸 스크린샷: demo_captures/screenshot_{session_id}_*.png")
    print()
    
    # Claude 분석용 요약
    claude_summary = f"""
🎭 솔로몬드 AI 시연 캐쳐 결과

📊 캐쳐 통계: {session_info['total_captures']}개 캐쳐, {session_info['duration']} 소요
🎯 활동: {', '.join(activity['tabs_used']) if activity['tabs_used'] else '탭 사용 없음'}
📁 파일: 음성 {len(files['audio_files'])}개, 이미지 {len(files['image_files'])}개
🔍 성과: 성공 {results['success_count']}회, 실패 {results['error_count']}회

📝 성공 사례:
{chr(10).join(f"- {msg}" for msg in results['success_messages'][:3]) if results['success_messages'] else "- 없음"}

⚠️ 오류 사례:
{chr(10).join(f"- {msg}" for msg in results['error_messages'][:2]) if results['error_messages'] else "- 없음"}

💡 평가: {', '.join(session_report['recommendations'])}
"""
    
    print("🤖 Claude 전달용 요약:")
    print("-" * 50)
    print(claude_summary.strip())
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(auto_demo_capture())