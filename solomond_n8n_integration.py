#!/usr/bin/env python3
"""
🎯 SOLOMOND AI - n8n 통합 브릿지
기존 시스템에 n8n 워크플로우 트리거 기능 추가
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import httpx
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolomondN8nIntegration:
    """SOLOMOND AI 시스템과 n8n 워크플로우 통합 클래스"""
    
    def __init__(self):
        self.n8n_base_url = "http://localhost:5678"
        self.webhook_urls = {
            "analysis_complete": f"{self.n8n_base_url}/webhook/analysis-complete",
            "file_upload": f"{self.n8n_base_url}/webhook/file-upload",
            "system_alert": f"{self.n8n_base_url}/webhook/system-alert"
        }
        
    def trigger_dual_brain_workflow(self, analysis_data: Dict[str, Any]) -> bool:
        """듀얼 브레인 워크플로우 트리거 (분석 완료 시 호출)"""
        try:
            # 분석 데이터 포맷팅
            payload = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "analysis_title": analysis_data.get("title", "컨퍼런스 분석"),
                "summary": analysis_data.get("summary", "분석이 완료되었습니다."),
                "file_count": analysis_data.get("file_count", 0),
                "analysis_type": analysis_data.get("type", "conference"),
                "insights_data": analysis_data
            }
            
            # n8n 웹훅 호출
            response = requests.post(
                self.webhook_urls["analysis_complete"],
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 듀얼 브레인 워크플로우 트리거 성공: {analysis_data.get('title', 'Unknown')}")
                return True
            else:
                logger.warning(f"⚠️ 워크플로우 트리거 실패 (HTTP {response.status_code}): {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ n8n 연결 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 워크플로우 트리거 에러: {e}")
            return False
    
    def trigger_file_analysis_workflow(self, file_data: Dict[str, Any]) -> bool:
        """파일 분석 워크플로우 트리거 (파일 업로드 시 호출)"""
        try:
            payload = {
                "file_name": file_data.get("name", "unknown"),
                "file_type": file_data.get("type", "unknown"),
                "file_size": file_data.get("size", 0),
                "upload_timestamp": datetime.now().isoformat(),
                "file_path": file_data.get("path", ""),
                "metadata": file_data.get("metadata", {})
            }
            
            response = requests.post(
                self.webhook_urls["file_upload"],
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 파일 분석 워크플로우 트리거 성공: {file_data.get('name', 'Unknown')}")
                return True
            else:
                logger.warning(f"⚠️ 파일 워크플로우 트리거 실패 (HTTP {response.status_code})")
                return False
                
        except Exception as e:
            logger.error(f"❌ 파일 워크플로우 트리거 에러: {e}")
            return False
    
    def send_system_alert(self, alert_data: Dict[str, Any]) -> bool:
        """시스템 알림 발송"""
        try:
            payload = {
                "alert_type": alert_data.get("type", "info"),
                "message": alert_data.get("message", "시스템 알림"),
                "timestamp": datetime.now().isoformat(),
                "severity": alert_data.get("severity", "low"),
                "system": "SOLOMOND AI",
                "details": alert_data.get("details", {})
            }
            
            response = requests.post(
                self.webhook_urls["system_alert"],
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 시스템 알림 발송 성공: {alert_data.get('message', 'Unknown')}")
                return True
            else:
                logger.warning(f"⚠️ 시스템 알림 발송 실패 (HTTP {response.status_code})")
                return False
                
        except Exception as e:
            logger.error(f"❌ 시스템 알림 에러: {e}")
            return False
    
    def check_n8n_connection(self) -> bool:
        """n8n 서버 연결 상태 확인"""
        try:
            response = requests.get(f"{self.n8n_base_url}/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_webhook_test_data(self, workflow_type: str) -> Dict[str, Any]:
        """테스트용 데이터 생성"""
        test_data = {
            "dual_brain": {
                "title": "테스트 컨퍼런스 분석",
                "summary": "n8n 연동 테스트를 위한 샘플 분석 데이터입니다.",
                "file_count": 5,
                "type": "conference",
                "duration": 120,
                "participants": ["Speaker A", "Speaker B"],
                "key_topics": ["AI", "주얼리", "분석"]
            },
            "file_upload": {
                "name": "test_audio.wav",
                "type": "audio/wav",
                "size": 1024000,
                "path": "/test/audio.wav",
                "metadata": {
                    "duration": 300,
                    "quality": "high",
                    "source": "conference"
                }
            },
            "system_alert": {
                "type": "warning",
                "message": "시스템 리소스 사용량이 높습니다",
                "severity": "medium",
                "details": {
                    "cpu_usage": 85,
                    "memory_usage": 90,
                    "disk_space": 75
                }
            }
        }
        
        return test_data.get(workflow_type, {})

# 기존 SOLOMOND 시스템과의 통합을 위한 유틸리티 함수들
def integrate_with_conference_analysis():
    """컨퍼런스 분석 시스템에 n8n 통합"""
    integration = SolomondN8nIntegration()
    
    # 컨퍼런스 분석 완료 후 호출할 함수
    def on_analysis_complete(analysis_result):
        """분석 완료 시 n8n 워크플로우 트리거"""
        if integration.check_n8n_connection():
            integration.trigger_dual_brain_workflow(analysis_result)
        else:
            logger.warning("⚠️ n8n 서버에 연결할 수 없습니다. 워크플로우를 건너뜁니다.")
    
    return on_analysis_complete

def integrate_with_file_upload():
    """파일 업로드 시스템에 n8n 통합"""
    integration = SolomondN8nIntegration()
    
    def on_file_upload(file_info):
        """파일 업로드 시 n8n 워크플로우 트리거"""
        if integration.check_n8n_connection():
            integration.trigger_file_analysis_workflow(file_info)
        else:
            logger.warning("⚠️ n8n 서버에 연결할 수 없습니다. 파일 워크플로우를 건너뜁니다.")
    
    return on_file_upload

# 테스트 함수
def test_n8n_integration():
    """n8n 통합 테스트"""
    integration = SolomondN8nIntegration()
    
    print("🧪 SOLOMOND AI - n8n 통합 테스트 시작")
    print("=" * 50)
    
    # 연결 테스트
    if integration.check_n8n_connection():
        print("✅ n8n 서버 연결 성공")
    else:
        print("❌ n8n 서버 연결 실패")
        return
    
    # 각 워크플로우 테스트
    workflows = ["dual_brain", "file_upload", "system_alert"]
    
    for workflow in workflows:
        print(f"\n🔧 {workflow} 워크플로우 테스트...")
        test_data = integration.get_webhook_test_data(workflow)
        
        if workflow == "dual_brain":
            result = integration.trigger_dual_brain_workflow(test_data)
        elif workflow == "file_upload":
            result = integration.trigger_file_analysis_workflow(test_data)
        elif workflow == "system_alert":
            result = integration.send_system_alert(test_data)
        
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   결과: {status}")
    
    print("\n🎯 n8n 통합 테스트 완료")

if __name__ == "__main__":
    # 통합 테스트 실행
    test_n8n_integration()