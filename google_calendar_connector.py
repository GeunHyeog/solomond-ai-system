#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📅 구글 캘린더 API 연동 시스템
Google Calendar API Integration for SOLOMOND AI Dual Brain

사용자: solomond.jgh@gmail.com
목적: 컨퍼런스 분석 결과를 구글 캘린더에 자동 기록
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import streamlit as st

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    st.warning("⚠️ Google APIs 라이브러리가 설치되지 않았습니다.")

class GoogleCalendarConnector:
    """구글 캘린더 연동 관리"""
    
    def __init__(self):
        self.credentials_file = Path("google_credentials.json")
        self.token_file = Path("google_token.json")
        self.scopes = ['https://www.googleapis.com/auth/calendar']
        self.service = None
        
    def setup_credentials(self):
        """구글 API 자격 증명 설정 안내"""
        st.subheader("🔐 구글 캘린더 API 설정")
        
        if not GOOGLE_APIS_AVAILABLE:
            st.error("❌ Google APIs 클라이언트 라이브러리가 필요합니다")
            st.code("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False
        
        if not self.credentials_file.exists():
            st.warning("📋 구글 API 자격 증명 파일이 필요합니다")
            
            with st.expander("📖 설정 가이드"):
                st.markdown("""
                ### 구글 캘린더 API 설정 방법
                
                1. **Google Cloud Console** 접속
                   - https://console.cloud.google.com
                
                2. **프로젝트 생성 또는 선택**
                
                3. **Calendar API 활성화**
                   - API 및 서비스 > 라이브러리
                   - "Google Calendar API" 검색 후 활성화
                
                4. **OAuth 2.0 클라이언트 ID 생성**
                   - API 및 서비스 > 사용자 인증 정보
                   - "사용자 인증 정보 만들기" > "OAuth 클라이언트 ID"
                   - 애플리케이션 유형: "데스크톱 애플리케이션"
                
                5. **자격 증명 파일 다운로드**
                   - JSON 파일을 다운로드
                   - 파일명을 `google_credentials.json`으로 변경
                   - 이 시스템의 루트 폴더에 저장
                """)
            
            # 파일 업로드 위젯
            uploaded_file = st.file_uploader(
                "구글 API 자격 증명 파일 업로드 (google_credentials.json)",
                type=['json']
            )
            
            if uploaded_file is not None:
                try:
                    credentials_data = json.load(uploaded_file)
                    with open(self.credentials_file, 'w') as f:
                        json.dump(credentials_data, f)
                    st.success("✅ 자격 증명 파일이 저장되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 파일 저장 실패: {e}")
            
            return False
        
        return True
    
    def authenticate(self):
        """구글 계정 인증"""
        if not GOOGLE_APIS_AVAILABLE:
            return False
        
        creds = None
        
        # 기존 토큰 확인
        if self.token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_file), self.scopes)
            except Exception as e:
                st.warning(f"⚠️ 기존 토큰 로드 실패: {e}")
        
        # 토큰 갱신 또는 새로 생성
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    st.success("🔄 토큰이 갱신되었습니다!")
                except Exception as e:
                    st.warning(f"⚠️ 토큰 갱신 실패: {e}")
                    creds = None
            
            if not creds:
                st.info("🔐 구글 계정 인증이 필요합니다")
                
                if st.button("🌐 구글 계정으로 로그인"):
                    try:
                        flow = Flow.from_client_secrets_file(
                            str(self.credentials_file), self.scopes
                        )
                        flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                        
                        auth_url, _ = flow.authorization_url(prompt='consent')
                        
                        st.markdown(f"""
                        ### 인증 단계:
                        1. [이 링크를 클릭하여 구글 로그인]({auth_url})
                        2. 권한을 승인하고 인증 코드를 복사
                        3. 아래에 인증 코드를 입력
                        """)
                        
                        auth_code = st.text_input("인증 코드 입력", type="password")
                        
                        if auth_code and st.button("인증 완료"):
                            flow.fetch_token(code=auth_code)
                            creds = flow.credentials
                            
                            # 토큰 저장
                            with open(self.token_file, 'w') as f:
                                f.write(creds.to_json())
                            
                            st.success("✅ 인증이 완료되었습니다!")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ 인증 실패: {e}")
                        return False
                
                return False
        
        # 토큰 저장
        if creds:
            with open(self.token_file, 'w') as f:
                f.write(creds.to_json())
        
        # Calendar API 서비스 생성
        try:
            self.service = build('calendar', 'v3', credentials=creds)
            return True
        except Exception as e:
            st.error(f"❌ Calendar API 서비스 생성 실패: {e}")
            return False
    
    def create_analysis_event(self, analysis_data: Dict[str, Any]) -> bool:
        """분석 결과를 캘린더 이벤트로 생성"""
        if not self.service:
            st.error("❌ 캘린더 서비스가 연결되지 않았습니다")
            return False
        
        try:
            # 이벤트 시간 설정 (분석 완료 시점)
            analysis_time = datetime.fromisoformat(analysis_data["timestamp"])
            start_time = analysis_time.isoformat()
            end_time = (analysis_time + timedelta(hours=1)).isoformat()
            
            # 이벤트 제목 및 설명
            conference_name = analysis_data["pre_info"].get("conference_name", "Unknown Conference")
            success_rate = f"{analysis_data['success_count']}/{analysis_data['total_files']}"
            
            event = {
                'summary': f'🎯 컨퍼런스 분석: {conference_name}',
                'description': f"""
📊 솔로몬드 AI 분석 결과

🎯 컨퍼런스: {conference_name}
📅 분석일시: {analysis_time.strftime('%Y년 %m월 %d일 %H시 %M분')}
📁 분석파일: {analysis_data['total_files']}개
✅ 성공률: {success_rate} ({analysis_data['success_count']/analysis_data['total_files']*100:.1f}%)

📋 추가 정보:
- 분석 ID: {analysis_data['analysis_id']}
- 장소: {analysis_data['pre_info'].get('conference_location', '미지정')}
- 업계: {analysis_data['pre_info'].get('industry_field', '미지정')}
- 파일 유형: {', '.join(analysis_data['file_types'])}

🤖 Generated by SOLOMOND AI Dual Brain System
                """.strip(),
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'Asia/Seoul',
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'Asia/Seoul',
                },
                'colorId': '9',  # 파란색
            }
            
            # 이벤트 생성
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            
            st.success(f"✅ 캘린더 이벤트가 생성되었습니다!")
            st.info(f"🔗 이벤트 링크: {event.get('htmlLink')}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ 캘린더 이벤트 생성 실패: {e}")
            return False
    
    def get_calendar_events(self, days_back: int = 30) -> List[Dict]:
        """최근 캘린더 이벤트 조회"""
        if not self.service:
            return []
        
        try:
            # 날짜 범위 설정
            now = datetime.now()
            time_min = (now - timedelta(days=days_back)).isoformat() + 'Z'
            time_max = now.isoformat() + 'Z'
            
            # 이벤트 조회
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=50,
                singleEvents=True,
                orderBy='startTime',
                q='솔로몬드'  # 솔로몬드 관련 이벤트만
            ).execute()
            
            events = events_result.get('items', [])
            return events
            
        except Exception as e:
            st.error(f"❌ 캘린더 이벤트 조회 실패: {e}")
            return []

def test_calendar_integration():
    """캘린더 연동 테스트"""
    st.subheader("🧪 구글 캘린더 연동 테스트")
    
    connector = GoogleCalendarConnector()
    
    # 자격 증명 설정
    if not connector.setup_credentials():
        return
    
    # 인증
    if not connector.authenticate():
        return
    
    st.success("✅ 구글 캘린더 연동이 완료되었습니다!")
    
    # 테스트 이벤트 생성
    if st.button("🧪 테스트 이벤트 생성"):
        test_data = {
            "analysis_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "pre_info": {
                "conference_name": "솔로몬드 AI 테스트 컨퍼런스",
                "conference_location": "테스트 장소",
                "industry_field": "AI/기술"
            },
            "total_files": 5,
            "success_count": 4,
            "file_types": ["image", "audio", "video"]
        }
        
        if connector.create_analysis_event(test_data):
            st.balloons()
    
    # 최근 이벤트 조회
    st.subheader("📅 최근 캘린더 이벤트")
    events = connector.get_calendar_events()
    
    if events:
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            st.write(f"📅 {start}: {event['summary']}")
    else:
        st.info("📊 아직 솔로몬드 관련 캘린더 이벤트가 없습니다")

if __name__ == "__main__":
    st.set_page_config(
        page_title="구글 캘린더 연동",
        page_icon="📅",
        layout="wide"
    )
    
    st.title("📅 구글 캘린더 API 연동")
    st.markdown("**solomond.jgh@gmail.com 계정과 연동**")
    
    test_calendar_integration()