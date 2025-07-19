#!/usr/bin/env python3
"""
시연 캐쳐 실행 스크립트
간단하게 시연 캐쳐를 시작할 수 있는 스크립트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from demo_capture_system import DemoCaptureSystem

async def quick_demo_capture():
    """빠른 시연 캐쳐"""
    
    print("🎭 솔로몬드 AI 시연 캐쳐 시스템")
    print("=" * 60)
    print("💎 브라우저에서 시연한 내용을 자동으로 캡처합니다!")
    print()
    
    # 사용자 입력
    print("⚙️ 캐쳐 설정:")
    
    # URL 확인
    url = input("Streamlit URL (엔터시 기본값: http://localhost:8503): ").strip()
    if not url:
        url = "http://localhost:8503"
    
    # 시간 설정
    duration_input = input("캐쳐 시간 (분, 엔터시 기본값: 5분): ").strip()
    try:
        duration = int(duration_input) if duration_input else 5
    except ValueError:
        duration = 5
    
    print(f"\n🚀 설정 완료:")
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
            
            # 요약 정보
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
            print()
            
            print("💡 추천사항:")
            for rec in session_report['recommendations']:
                print(f"   {rec}")
            print()
            
            print(f"📄 상세 리포트:")
            print(f"   📁 demo_captures/session_report_{capture_system.session_id}.json")
            print(f"   📸 스크린샷들: demo_captures/screenshot_{capture_system.session_id}_*.png")
            print()
            
            # Claude에게 전달할 요약
            claude_summary = generate_claude_summary(session_report)
            print("🤖 Claude 분석용 요약:")
            print("-" * 40)
            print(claude_summary)
            print("-" * 40)
            
        else:
            print("❌ 시연 캐쳐가 실패했습니다.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
        
        # 부분 리포트 생성
        if capture_system.captures:
            print("📊 부분 리포트 생성 중...")
            session_report = await capture_system.generate_session_report()
            
            claude_summary = generate_claude_summary(session_report)
            print("\n🤖 Claude 분석용 부분 요약:")
            print("-" * 40)
            print(claude_summary)
            print("-" * 40)
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

def generate_claude_summary(session_report: dict) -> str:
    """Claude에게 전달할 요약 생성"""
    
    session_info = session_report['session_info']
    activity = session_report['activity_summary']
    files = session_report['file_uploads']
    results = session_report['analysis_results']
    
    summary = f"""
🎭 솔로몬드 AI 시연 결과 요약

📊 기본 정보:
- 세션 ID: {session_info['session_id']}
- 캐쳐 횟수: {session_info['total_captures']}개
- 시연 시간: {session_info['duration']}
- 시작: {session_info['start_time']}
- 종료: {session_info['end_time']}

🎯 사용자 활동:
- 사용한 탭: {', '.join(activity['tabs_used']) if activity['tabs_used'] else '없음'}
- 총 상호작용: {activity['total_interactions']}회

📁 파일 처리:
- 총 업로드 파일: {files['total_files']}개
- 음성 파일 ({len(files['audio_files'])}개): {', '.join(files['audio_files']) if files['audio_files'] else '없음'}
- 이미지 파일 ({len(files['image_files'])}개): {', '.join(files['image_files']) if files['image_files'] else '없음'}

🔍 분석 성과:
- 성공한 분석: {results['success_count']}회
- 실패한 분석: {results['error_count']}회
- 성공률: {(results['success_count']/(results['success_count']+results['error_count'])*100) if (results['success_count']+results['error_count']) > 0 else 0:.1f}%

✅ 성공 메시지:
{chr(10).join(f"- {msg}" for msg in results['success_messages'][:5]) if results['success_messages'] else "- 없음"}

❌ 오류 메시지:
{chr(10).join(f"- {msg}" for msg in results['error_messages'][:3]) if results['error_messages'] else "- 없음"}

💡 시스템 평가:
{chr(10).join(f"- {rec}" for rec in session_report['recommendations'])}

📈 메트릭 변화:
{f"- 총 {len(results['metrics_evolution'])}회 메트릭 변화 감지" if results['metrics_evolution'] else "- 메트릭 변화 없음"}
"""
    
    return summary.strip()

if __name__ == "__main__":
    print("시연 캐쳐를 시작하려면 아래 명령어를 실행하세요:")
    print("python3 start_demo_capture.py")
    print()
    asyncio.run(quick_demo_capture())