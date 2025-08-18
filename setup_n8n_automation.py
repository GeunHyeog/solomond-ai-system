#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 SOLOMOND AI - n8n 자동화 시스템 완전 설치 스크립트
n8n 설치부터 워크플로우 배포까지 원클릭 자동화

사용법: python setup_n8n_automation.py
"""

import sys
import os
import subprocess
import json
import requests
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

# Windows UTF-8 인코딩 강제 설정
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class N8nAutomationSetup:
    """n8n 자동화 시스템 완전 설치 클래스"""
    
    def __init__(self):
        self.n8n_url = "http://localhost:5678"
        self.n8n_dir = Path("n8n-solomond")
        self.setup_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """설정 로그 기록"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        print(log_entry)
    
    def check_node_installation(self) -> bool:
        """Node.js 설치 확인"""
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log(f"✅ Node.js 설치됨: {version}")
                return True
            else:
                self.log("❌ Node.js가 설치되지 않음")
                return False
        except Exception as e:
            self.log(f"❌ Node.js 확인 실패: {str(e)}")
            return False
    
    def install_n8n(self) -> bool:
        """n8n 설치"""
        try:
            self.log("🚀 n8n 설치 시작...")
            
            # n8n 전역 설치
            cmd = ['npm', 'install', '-g', 'n8n']
            if sys.platform.startswith('win'):
                cmd = ['cmd', '/c'] + cmd
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log("✅ n8n 설치 완료")
                return True
            else:
                self.log(f"❌ n8n 설치 실패: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ n8n 설치 중 오류: {str(e)}")
            return False
    
    def start_n8n_server(self) -> subprocess.Popen:
        """n8n 서버 시작"""
        try:
            self.log("🌟 n8n 서버 시작 중...")
            
            cmd = ['n8n', 'start']
            if sys.platform.startswith('win'):
                # Windows에서 새 창으로 실행
                process = subprocess.Popen(
                    ['cmd', '/c', 'start', 'cmd', '/k'] + cmd,
                    shell=True
                )
            else:
                process = subprocess.Popen(cmd)
            
            # 서버 시작 대기
            self.wait_for_n8n_server()
            
            self.log("✅ n8n 서버 시작됨")
            return process
            
        except Exception as e:
            self.log(f"❌ n8n 서버 시작 실패: {str(e)}")
            return None
    
    def wait_for_n8n_server(self, max_wait: int = 60):
        """n8n 서버 시작 대기"""
        for i in range(max_wait):
            try:
                response = requests.get(f"{self.n8n_url}/healthz", timeout=5)
                if response.status_code == 200:
                    self.log(f"✅ n8n 서버 준비 완료 (대기시간: {i+1}초)")
                    return True
            except:
                pass
            
            time.sleep(1)
            if i % 10 == 9:
                self.log(f"⏳ n8n 서버 시작 대기 중... ({i+1}/{max_wait}초)")
        
        self.log("❌ n8n 서버 시작 시간 초과")
        return False
    
    async def deploy_solomond_workflows(self) -> bool:
        """SOLOMOND AI 워크플로우 자동 배포"""
        try:
            self.log("📋 SOLOMOND AI 워크플로우 배포 시작...")
            
            # n8n_connector를 통해 워크플로우 생성
            from n8n_connector import N8nConnector
            
            connector = N8nConnector()
            
            # 서버 상태 확인
            if not connector.check_n8n_status():
                self.log("❌ n8n 서버 연결 실패")
                return False
            
            # 모든 워크플로우 설정
            results = await connector.setup_solomond_workflows()
            
            success_count = 0
            for workflow_name, result in results.items():
                if not result.startswith("error"):
                    self.log(f"✅ 워크플로우 생성 성공: {workflow_name} -> {result}")
                    success_count += 1
                else:
                    self.log(f"❌ 워크플로우 생성 실패: {workflow_name} -> {result}")
            
            total_workflows = len(results)
            self.log(f"📊 워크플로우 배포 결과: {success_count}/{total_workflows} 성공")
            
            return success_count > 0
            
        except Exception as e:
            self.log(f"❌ 워크플로우 배포 중 오류: {str(e)}")
            return False
    
    def create_startup_scripts(self):
        """시작 스크립트들 생성"""
        try:
            self.log("📝 시작 스크립트 생성 중...")
            
            # Windows 시작 스크립트
            if sys.platform.startswith('win'):
                startup_script = '''@echo off
chcp 65001 > nul
echo 🚀 SOLOMOND AI - n8n 자동화 시스템 시작
echo.

echo ⏳ n8n 서버 시작 중...
start "n8n Server" cmd /k "n8n start"

echo ⏳ 서버 초기화 대기 중... (30초)
timeout /t 30 /nobreak > nul

echo 🌐 n8n 대시보드 열기...
start http://localhost:5678

echo.
echo ✅ n8n 자동화 시스템이 시작되었습니다!
echo 🌐 n8n 대시보드: http://localhost:5678
echo 📋 SOLOMOND AI 워크플로우가 자동으로 배포되었습니다.
echo.
pause
'''
            else:
                # Linux/Mac 시작 스크립트
                startup_script = '''#!/bin/bash
echo "🚀 SOLOMOND AI - n8n 자동화 시스템 시작"
echo ""

echo "⏳ n8n 서버 시작 중..."
n8n start &

echo "⏳ 서버 초기화 대기 중... (30초)"
sleep 30

echo "🌐 n8n 대시보드 열기..."
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:5678
elif command -v open > /dev/null; then
    open http://localhost:5678
fi

echo ""
echo "✅ n8n 자동화 시스템이 시작되었습니다!"
echo "🌐 n8n 대시보드: http://localhost:5678"
echo "📋 SOLOMOND AI 워크플로우가 자동으로 배포되었습니다."
echo ""
read -p "Press any key to continue..."
'''
            
            script_path = Path("start_n8n_system.bat" if sys.platform.startswith('win') else "start_n8n_system.sh")
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(startup_script)
            
            if not sys.platform.startswith('win'):
                os.chmod(script_path, 0o755)
            
            self.log(f"✅ 시작 스크립트 생성됨: {script_path}")
            
        except Exception as e:
            self.log(f"❌ 시작 스크립트 생성 실패: {str(e)}")
    
    def create_google_calendar_guide(self):
        """Google Calendar API 인증 가이드 생성"""
        guide_content = '''# 📅 Google Calendar API 인증 설정 가이드

## 🔧 1단계: Google Cloud Console 설정

1. **Google Cloud Console 접속**
   - https://console.cloud.google.com/ 접속
   - 새 프로젝트 생성 또는 기존 프로젝트 선택

2. **Google Calendar API 활성화**
   - "API 및 서비스" → "라이브러리" 이동
   - "Google Calendar API" 검색 후 활성화

3. **OAuth 2.0 인증 정보 생성**
   - "API 및 서비스" → "사용자 인증 정보" 이동
   - "+ 사용자 인증 정보 만들기" → "OAuth 클라이언트 ID" 선택
   - 애플리케이션 유형: "데스크톱 애플리케이션"
   - 이름: "SOLOMOND AI Calendar Integration"

4. **인증 파일 다운로드**
   - 생성된 OAuth 클라이언트에서 JSON 파일 다운로드
   - `credentials.json`으로 저장하여 프로젝트 루트에 배치

## 🔧 2단계: n8n에서 Google Calendar 연동

1. **n8n 대시보드 접속**: http://localhost:5678
2. **새 워크플로우 생성**
3. **Google Calendar 노드 추가**
   - 노드 팔레트에서 "Google Calendar" 검색
   - 노드를 워크플로우에 드래그앤드롭
4. **인증 설정**
   - Google Calendar 노드 클릭 → "Create New" credential 선택
   - OAuth2 방식 선택
   - Client ID, Client Secret 입력 (credentials.json에서 확인)
   - Authorization URL: `https://accounts.google.com/o/oauth2/auth`
   - Access Token URL: `https://oauth2.googleapis.com/token`
   - Scope: `https://www.googleapis.com/auth/calendar`

## 🔧 3단계: 자동 인증 (선택사항)

SOLOMOND AI 시스템에서 자동 인증을 원하는 경우:

1. **환경 변수 설정**
   ```bash
   set GOOGLE_CLIENT_ID=your_client_id
   set GOOGLE_CLIENT_SECRET=your_client_secret
   set GOOGLE_REDIRECT_URI=http://localhost:8080/callback
   ```

2. **credentials.json 파일 위치**
   - 프로젝트 루트: `C:\\Users\\PC_58410\\solomond-ai-system\\credentials.json`

## ✅ 테스트 방법

1. **n8n 워크플로우에서 테스트**
   - Google Calendar 노드 설정 완료 후
   - "Test step" 버튼 클릭하여 인증 및 연결 확인

2. **SOLOMOND AI에서 테스트**
   - 컨퍼런스 분석 완료 후
   - 듀얼 브레인 시스템 실행 시 자동으로 캘린더 이벤트 생성 확인

## 🛠️ 문제 해결

### 인증 오류 시:
- `credentials.json` 파일이 올바른 위치에 있는지 확인
- Google Cloud Console에서 OAuth 동의 화면 설정 확인
- 리다이렉트 URI가 정확히 설정되었는지 확인

### 권한 오류 시:
- Google Calendar API가 활성화되어 있는지 확인
- OAuth 스코프가 올바르게 설정되었는지 확인
- Google 계정에서 캘린더 권한이 부여되었는지 확인

---
💡 **자동화 완료 후**: 컨퍼런스 분석 → 구글 캘린더 이벤트 자동 생성 → AI 인사이트 추가
'''
        
        guide_path = Path("GOOGLE_CALENDAR_SETUP_GUIDE.md")
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        self.log(f"✅ Google Calendar 설정 가이드 생성됨: {guide_path}")
    
    def save_setup_log(self):
        """설치 로그 저장"""
        log_path = Path("n8n_setup_log.txt")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.setup_log))
        
        self.log(f"📋 설치 로그 저장됨: {log_path}")
    
    async def run_complete_setup(self):
        """완전 자동 설치 실행"""
        self.log("🚀 SOLOMOND AI - n8n 자동화 시스템 완전 설치 시작")
        self.log("=" * 60)
        
        try:
            # 1. Node.js 확인
            if not self.check_node_installation():
                self.log("❌ Node.js가 필요합니다. https://nodejs.org 에서 설치하세요.")
                return False
            
            # 2. n8n 설치
            if not self.install_n8n():
                return False
            
            # 3. n8n 서버 시작
            n8n_process = self.start_n8n_server()
            if not n8n_process:
                return False
            
            # 4. 워크플로우 배포
            workflow_success = await self.deploy_solomond_workflows()
            
            # 5. 시작 스크립트 생성
            self.create_startup_scripts()
            
            # 6. Google Calendar 가이드 생성
            self.create_google_calendar_guide()
            
            # 7. 설치 로그 저장
            self.save_setup_log()
            
            # 최종 결과
            self.log("=" * 60)
            if workflow_success:
                self.log("🎉 SOLOMOND AI - n8n 자동화 시스템 설치 완료!")
                self.log("📋 다음 단계:")
                self.log("   1. GOOGLE_CALENDAR_SETUP_GUIDE.md를 참조하여 구글 캘린더 연동")
                self.log("   2. start_n8n_system.bat 실행하여 자동화 시스템 시작")
                self.log("   3. 컨퍼런스 분석 완료 후 자동화 워크플로우 확인")
                self.log(f"🌐 n8n 대시보드: {self.n8n_url}")
                return True
            else:
                self.log("⚠️ 일부 설치가 완료되지 않았습니다. 로그를 확인하세요.")
                return False
                
        except Exception as e:
            self.log(f"❌ 설치 중 치명적 오류: {str(e)}")
            return False

async def main():
    """메인 실행 함수"""
    setup = N8nAutomationSetup()
    success = await setup.run_complete_setup()
    
    if success:
        print("\n✅ 설치 완료! 다음 명령어로 시작하세요:")
        if sys.platform.startswith('win'):
            print("start_n8n_system.bat")
        else:
            print("./start_n8n_system.sh")
    else:
        print("\n❌ 설치 실패. n8n_setup_log.txt를 확인하세요.")

if __name__ == "__main__":
    asyncio.run(main())