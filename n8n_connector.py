#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔗 SOLOMOND AI - n8n 워크플로우 연동 시스템
n8n과 SOLOMOND AI 시스템을 연결하는 API 브릿지

Windows 호환성 완전 지원 및 UTF-8 인코딩 강제
"""

import sys
import os

# Windows UTF-8 인코딩 강제 설정
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    # Windows 콘솔 CP949 문제 해결
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import asyncio
import httpx
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class N8nConnector:
    """n8n 워크플로우와 SOLOMOND AI 시스템 연동 클래스"""
    
    def __init__(self, n8n_url: str = "http://localhost:5678"):
        self.n8n_url = n8n_url
        self.api_base = f"{n8n_url}/api/v1"
        self.webhook_base = f"{n8n_url}/webhook"
        
        # SOLOMOND AI 시스템 정보
        self.solomond_ports = {
            "main_dashboard": 8500,
            "conference_analysis": 8501,
            "web_crawler": 8502,
            "gemstone_analysis": 8503,
            "cad_conversion": 8504
        }
        
        # n8n 워크플로우 템플릿
        self.workflow_templates = {
            "dual_brain_pipeline": self._create_dual_brain_workflow(),
            "google_calendar_sync": self._create_calendar_sync_workflow(),
            "monitoring_alerts": self._create_monitoring_workflow(),
            "analysis_pipeline": self._create_analysis_pipeline_workflow()
        }
    
    def _create_dual_brain_workflow(self) -> Dict[str, Any]:
        """듀얼 브레인 워크플로우 템플릿 생성"""
        return {
            "name": "SOLOMOND Dual Brain Pipeline",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "analysis-trigger",
                        "options": {}
                    },
                    "name": "Analysis Trigger",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [240, 300]
                },
                {
                    "parameters": {
                        "url": f"http://localhost:{self.solomond_ports['conference_analysis']}/api/analyze",
                        "sendBody": True,
                        "bodyContentType": "json",
                        "jsonBody": "={{ $json }}"
                    },
                    "name": "Conference Analysis",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [460, 300]
                },
                {
                    "parameters": {
                        "conditions": {
                            "string": [
                                {
                                    "value1": "={{ $json.status }}",
                                    "operation": "equal",
                                    "value2": "completed"
                                }
                            ]
                        }
                    },
                    "name": "Check Analysis Status",
                    "type": "n8n-nodes-base.if",
                    "typeVersion": 1,
                    "position": [680, 300]
                },
                {
                    "parameters": {
                        "url": "http://localhost:8580/api/generate-insights",
                        "sendBody": True,
                        "bodyContentType": "json",
                        "jsonBody": "={{ $json }}"
                    },
                    "name": "AI Insights Generation",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [900, 200]
                },
                {
                    "parameters": {
                        "url": "https://www.googleapis.com/calendar/v3/calendars/solomond.jgh@gmail.com/events",
                        "sendBody": True,
                        "bodyContentType": "json",
                        "jsonBody": "={{ $json.calendar_event }}"
                    },
                    "name": "Google Calendar Sync",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [900, 400]
                }
            ],
            "connections": {
                "Analysis Trigger": {
                    "main": [
                        [
                            {
                                "node": "Conference Analysis",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Conference Analysis": {
                    "main": [
                        [
                            {
                                "node": "Check Analysis Status",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                },
                "Check Analysis Status": {
                    "main": [
                        [
                            {
                                "node": "AI Insights Generation",
                                "type": "main",
                                "index": 0
                            },
                            {
                                "node": "Google Calendar Sync",
                                "type": "main",
                                "index": 0
                            }
                        ],
                        []
                    ]
                }
            }
        }
    
    def _create_calendar_sync_workflow(self) -> Dict[str, Any]:
        """구글 캘린더 동기화 워크플로우"""
        return {
            "name": "Google Calendar Sync",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "calendar-event",
                        "options": {}
                    },
                    "name": "Calendar Event Trigger",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [240, 300]
                },
                {
                    "parameters": {
                        "resource": "event",
                        "operation": "create",
                        "calendarId": "solomond.jgh@gmail.com",
                        "summary": "={{ $json.title }}",
                        "description": "={{ $json.description }}",
                        "startDate": "={{ $json.start_date }}",
                        "endDate": "={{ $json.end_date }}"
                    },
                    "name": "Create Calendar Event",
                    "type": "n8n-nodes-base.googleCalendar",
                    "typeVersion": 1,
                    "position": [460, 300]
                }
            ],
            "connections": {
                "Calendar Event Trigger": {
                    "main": [
                        [
                            {
                                "node": "Create Calendar Event",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                }
            }
        }
    
    def _create_monitoring_workflow(self) -> Dict[str, Any]:
        """시스템 모니터링 워크플로우"""
        return {
            "name": "SOLOMOND System Monitor",
            "nodes": [
                {
                    "parameters": {
                        "rule": {
                            "interval": [
                                {
                                    "field": "minutes",
                                    "minutesInterval": 5
                                }
                            ]
                        }
                    },
                    "name": "Monitor Schedule",
                    "type": "n8n-nodes-base.cron",
                    "typeVersion": 1,
                    "position": [240, 300]
                },
                {
                    "parameters": {
                        "url": f"http://localhost:{self.solomond_ports['main_dashboard']}/health",
                        "options": {
                            "timeout": 5000
                        }
                    },
                    "name": "Check Dashboard",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [460, 200]
                },
                {
                    "parameters": {
                        "url": f"http://localhost:{self.solomond_ports['conference_analysis']}/health",
                        "options": {
                            "timeout": 5000
                        }
                    },
                    "name": "Check Conference Analysis",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [460, 400]
                }
            ]
        }
    
    def _create_analysis_pipeline_workflow(self) -> Dict[str, Any]:
        """분석 파이프라인 워크플로우"""
        return {
            "name": "SOLOMOND Analysis Pipeline",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "file-upload",
                        "options": {}
                    },
                    "name": "File Upload Trigger",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [240, 300]
                },
                {
                    "parameters": {
                        "conditions": {
                            "string": [
                                {
                                    "value1": "={{ $json.file_type }}",
                                    "operation": "contains",
                                    "value2": "audio"
                                }
                            ]
                        }
                    },
                    "name": "Check File Type",
                    "type": "n8n-nodes-base.if",
                    "typeVersion": 1,
                    "position": [460, 300]
                },
                {
                    "parameters": {
                        "url": f"http://localhost:{self.solomond_ports['conference_analysis']}/api/process-audio",
                        "sendBody": True,
                        "bodyContentType": "json"
                    },
                    "name": "Process Audio",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [680, 200]
                },
                {
                    "parameters": {
                        "url": f"http://localhost:{self.solomond_ports['conference_analysis']}/api/process-image",
                        "sendBody": True,
                        "bodyContentType": "json"
                    },
                    "name": "Process Image",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 1,
                    "position": [680, 400]
                }
            ]
        }
    
    async def create_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """n8n에서 워크플로우 생성"""
        if workflow_name not in self.workflow_templates:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow_data = self.workflow_templates[workflow_name]
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_base}/workflows",
                    json=workflow_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"✅ Workflow '{workflow_name}' created successfully: {result.get('id')}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to create workflow '{workflow_name}': {e}")
                raise
    
    async def trigger_workflow(self, webhook_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """웹훅을 통해 워크플로우 트리거"""
        webhook_url = f"{self.webhook_base}/{webhook_path}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    webhook_url,
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json() if response.content else {"status": "triggered"}
                
                logger.info(f"✅ Workflow triggered via '{webhook_path}': {result}")
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to trigger workflow '{webhook_path}': {e}")
                raise
    
    def check_n8n_status(self) -> bool:
        """n8n 서버 상태 확인"""
        try:
            response = requests.get(f"{self.n8n_url}/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def setup_solomond_workflows(self) -> Dict[str, str]:
        """SOLOMOND AI용 모든 워크플로우 설정"""
        results = {}
        
        for workflow_name in self.workflow_templates.keys():
            try:
                result = await self.create_workflow(workflow_name)
                results[workflow_name] = result.get('id', 'unknown')
                
                # 워크플로우 활성화
                if 'id' in result:
                    await self.activate_workflow(result['id'])
                    
            except Exception as e:
                logger.error(f"Failed to setup workflow {workflow_name}: {e}")
                results[workflow_name] = f"error: {e}"
        
        return results
    
    async def activate_workflow(self, workflow_id: str) -> bool:
        """워크플로우 활성화"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.patch(
                    f"{self.api_base}/workflows/{workflow_id}",
                    json={"active": True}
                )
                response.raise_for_status()
                logger.info(f"✅ Workflow {workflow_id} activated")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to activate workflow {workflow_id}: {e}")
                return False

# 메인 실행 함수
async def main():
    """n8n 연동 시스템 초기화"""
    connector = N8nConnector()
    
    # n8n 서버 상태 확인
    if not connector.check_n8n_status():
        logger.error("❌ n8n server is not running on localhost:5678")
        return
    
    logger.info("✅ n8n server is running")
    
    # SOLOMOND AI 워크플로우 설정
    logger.info("🚀 Setting up SOLOMOND AI workflows...")
    results = await connector.setup_solomond_workflows()
    
    print("\n🎯 SOLOMOND AI - n8n Integration Results:")
    for workflow, result in results.items():
        status = "✅" if not result.startswith("error") else "❌"
        print(f"{status} {workflow}: {result}")
    
    print(f"\n🌐 n8n Dashboard: {connector.n8n_url}")
    print("🔗 SOLOMOND AI 시스템과 n8n이 연동되었습니다!")

if __name__ == "__main__":
    asyncio.run(main())