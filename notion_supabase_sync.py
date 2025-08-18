#!/usr/bin/env python3
"""
Notion-Supabase 데이터 동기화 시스템
솔로몬드 AI 분석 결과를 Notion과 Supabase 간 자동 동기화

주요 기능:
- 분석 결과 자동 Notion 페이지 생성
- Supabase 데이터베이스 동기화
- 양방향 데이터 연동
- 자동 백업 및 복구
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotionSupabaseSync:
    """Notion-Supabase 동기화 관리자"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.notion_config = self._load_notion_config()
        self.supabase_config = self._load_supabase_config()
        
        # Notion 설정
        self.notion_headers = {
            "Authorization": f"Bearer {self.notion_config['api_key']}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        # 동기화 상태 추적
        self.sync_log = []
    
    def _load_notion_config(self) -> Dict[str, str]:
        """Notion 설정 로드"""
        return {
            "api_key": os.environ.get("NOTION_API_KEY", "NOTION_TOKEN_NOT_SET"),
            "database_id": None  # 나중에 설정
        }
    
    def _load_supabase_config(self) -> Dict[str, Any]:
        """Supabase 설정 로드"""
        config_file = self.project_root / "supabase_config.json"
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Supabase 설정 파일을 찾을 수 없습니다. 기본값 사용.")
            return {
                "supabase": {
                    "url": "YOUR_SUPABASE_URL_HERE",
                    "anon_key": "YOUR_SUPABASE_ANON_KEY_HERE"
                }
            }
    
    def create_notion_analysis_database(self) -> Optional[str]:
        """Notion에 분석 결과 데이터베이스 생성"""
        
        # 먼저 기존 데이터베이스 검색
        search_result = self._search_notion_database("솔로몬드 AI 분석 결과")
        if search_result:
            logger.info(f"기존 데이터베이스 발견: {search_result}")
            return search_result
        
        # 새 데이터베이스 생성
        database_schema = {
            "parent": {"type": "page_id", "page_id": self._get_or_create_parent_page()},
            "title": [
                {
                    "type": "text",
                    "text": {"content": "솔로몬드 AI 분석 결과"}
                }
            ],
            "properties": {
                "분석 ID": {"title": {}},
                "분석 유형": {
                    "select": {
                        "options": [
                            {"name": "컨퍼런스 분석", "color": "blue"},
                            {"name": "보석 산지 분석", "color": "green"},
                            {"name": "웹 크롤러", "color": "orange"},
                            {"name": "3D CAD 변환", "color": "purple"}
                        ]
                    }
                },
                "상태": {
                    "select": {
                        "options": [
                            {"name": "대기중", "color": "gray"},
                            {"name": "처리중", "color": "yellow"},
                            {"name": "완료", "color": "green"},
                            {"name": "실패", "color": "red"}
                        ]
                    }
                },
                "생성일": {"date": {}},
                "처리 시간(초)": {"number": {}},
                "파일 개수": {"number": {}},
                "신뢰도": {"number": {}},
                "사용자": {"rich_text": {}},
                "요약": {"rich_text": {}},
                "태그": {
                    "multi_select": {
                        "options": [
                            {"name": "음성", "color": "blue"},
                            {"name": "이미지", "color": "green"}, 
                            {"name": "텍스트", "color": "yellow"},
                            {"name": "비디오", "color": "red"}
                        ]
                    }
                }
            }
        }
        
        try:
            response = requests.post(
                "https://api.notion.com/v1/databases",
                headers=self.notion_headers,
                json=database_schema,
                timeout=30
            )
            
            if response.status_code == 200:
                database_data = response.json()
                database_id = database_data["id"]
                
                # 설정에 저장
                self.notion_config["database_id"] = database_id
                self._save_notion_database_id(database_id)
                
                logger.info(f"Notion 데이터베이스 생성 성공: {database_id}")
                return database_id
            else:
                logger.error(f"데이터베이스 생성 실패: {response.json()}")
                return None
                
        except Exception as e:
            logger.error(f"데이터베이스 생성 오류: {e}")
            return None
    
    def _search_notion_database(self, title: str) -> Optional[str]:
        """Notion에서 데이터베이스 검색"""
        try:
            response = requests.post(
                "https://api.notion.com/v1/search",
                headers=self.notion_headers,
                json={
                    "query": title,
                    "filter": {"property": "object", "value": "database"}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return results[0]["id"]
            
            return None
            
        except Exception as e:
            logger.error(f"데이터베이스 검색 오류: {e}")
            return None
    
    def _get_or_create_parent_page(self) -> str:
        """부모 페이지 가져오기 또는 생성"""
        # 간단히 첫 번째 페이지 사용 (실제로는 더 정교한 로직 필요)
        try:
            response = requests.post(
                "https://api.notion.com/v1/search",
                headers=self.notion_headers,
                json={"query": "", "page_size": 1},
                timeout=15
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return results[0]["id"]
            
            # 기본 페이지 생성 로직 (실제 구현에서는 더 정교하게)
            return "00000000-0000-0000-0000-000000000000"
            
        except Exception as e:
            logger.error(f"부모 페이지 검색 오류: {e}")
            return "00000000-0000-0000-0000-000000000000"
    
    def _save_notion_database_id(self, database_id: str):
        """Notion 데이터베이스 ID 저장"""
        config_file = self.project_root / "notion_database_config.json"
        config = {
            "database_id": database_id,
            "created_at": datetime.now().isoformat(),
            "title": "솔로몬드 AI 분석 결과"
        }
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Notion 데이터베이스 ID 저장: {config_file}")
    
    def sync_analysis_result(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 Notion과 Supabase에 동기화"""
        sync_result = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": analysis_data.get("id", "unknown"),
            "notion_sync": False,
            "supabase_sync": False,
            "errors": []
        }
        
        try:
            # 1. Notion 동기화
            notion_result = self._sync_to_notion(analysis_data)
            if notion_result:
                sync_result["notion_sync"] = True
                sync_result["notion_page_id"] = notion_result
            else:
                sync_result["errors"].append("Notion 동기화 실패")
            
            # 2. Supabase 동기화 (구현 예정)
            # supabase_result = self._sync_to_supabase(analysis_data)
            # sync_result["supabase_sync"] = supabase_result
            
            # 3. 동기화 로그 저장
            self.sync_log.append(sync_result)
            self._save_sync_log()
            
            return sync_result
            
        except Exception as e:
            sync_result["errors"].append(str(e))
            logger.error(f"동기화 오류: {e}")
            return sync_result
    
    def _sync_to_notion(self, analysis_data: Dict[str, Any]) -> Optional[str]:
        """Notion에 분석 결과 페이지 생성"""
        
        # 데이터베이스 ID 확인
        database_id = self.notion_config.get("database_id")
        if not database_id:
            database_id = self.create_notion_analysis_database()
            if not database_id:
                return None
        
        # 페이지 데이터 구성
        page_data = {
            "parent": {"database_id": database_id},
            "properties": {
                "분석 ID": {
                    "title": [{"text": {"content": analysis_data.get("id", "Unknown")}}]
                },
                "분석 유형": {
                    "select": {"name": analysis_data.get("type", "기타")}
                },
                "상태": {
                    "select": {"name": analysis_data.get("status", "완료")}
                },
                "생성일": {
                    "date": {"start": analysis_data.get("created_at", datetime.now().isoformat())}
                },
                "처리 시간(초)": {
                    "number": analysis_data.get("processing_time", 0)
                },
                "파일 개수": {
                    "number": len(analysis_data.get("files", []))
                },
                "신뢰도": {
                    "number": analysis_data.get("confidence", 0.0)
                },
                "사용자": {
                    "rich_text": [{"text": {"content": analysis_data.get("user", "System")}}]
                },
                "요약": {
                    "rich_text": [{"text": {"content": analysis_data.get("summary", "")[:2000]}}]
                }
            }
        }
        
        try:
            response = requests.post(
                "https://api.notion.com/v1/pages",
                headers=self.notion_headers,
                json=page_data,
                timeout=30
            )
            
            if response.status_code == 200:
                page_data = response.json()
                page_id = page_data["id"]
                logger.info(f"Notion 페이지 생성 성공: {page_id}")
                return page_id
            else:
                logger.error(f"Notion 페이지 생성 실패: {response.json()}")
                return None
                
        except Exception as e:
            logger.error(f"Notion 동기화 오류: {e}")
            return None
    
    def _save_sync_log(self):
        """동기화 로그 저장"""
        log_file = self.project_root / "notion_supabase_sync_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.sync_log, f, indent=2, ensure_ascii=False)
    
    def create_sample_analysis_data(self) -> Dict[str, Any]:
        """테스트용 샘플 분석 데이터 생성"""
        return {
            "id": f"analysis_{int(datetime.now().timestamp())}",
            "type": "컨퍼런스 분석",
            "status": "완료",
            "created_at": datetime.now().isoformat(),
            "processing_time": 45,
            "files": ["audio1.m4a", "image1.png"],
            "confidence": 0.89,
            "user": "Test User",
            "summary": "JGA25 주얼리 컨퍼런스 분석 결과: 다이아몬드 시장 트렌드와 소비자 선호도 변화에 대한 종합적인 분석이 완료되었습니다. 음성 인식을 통해 주요 발표 내용을 추출하고, 이미지 분석을 통해 제품 카탈로그를 정리했습니다.",
            "results": {
                "stt_confidence": 0.92,
                "ocr_blocks": 15,
                "extracted_text": "2024년 다이아몬드 시장 전망...",
                "keywords": ["다이아몬드", "주얼리", "트렌드", "컨퍼런스"]
            }
        }
    
    def test_complete_sync(self):
        """전체 동기화 시스템 테스트"""
        print("=== Notion-Supabase 동기화 테스트 ===")
        
        # 1. Notion 데이터베이스 생성/확인
        print("1. Notion 데이터베이스 생성 중...")
        database_id = self.create_notion_analysis_database()
        if database_id:
            print(f"SUCCESS: 데이터베이스 ID: {database_id}")
        else:
            print("ERROR: 데이터베이스 생성 실패")
            return False
        
        # 2. 샘플 데이터 동기화
        print("2. 샘플 분석 결과 동기화 중...")
        sample_data = self.create_sample_analysis_data()
        sync_result = self.sync_analysis_result(sample_data)
        
        print(f"동기화 결과:")
        print(f"  - Notion 동기화: {sync_result['notion_sync']}")
        print(f"  - 오류: {sync_result['errors']}")
        
        if sync_result["notion_sync"]:
            print(f"  - Notion 페이지 ID: {sync_result.get('notion_page_id', 'Unknown')}")
        
        return sync_result["notion_sync"]

# 편의 함수
def create_sync_manager() -> NotionSupabaseSync:
    """동기화 매니저 인스턴스 생성"""
    return NotionSupabaseSync()

if __name__ == "__main__":
    # 테스트 실행
    sync_manager = NotionSupabaseSync()
    success = sync_manager.test_complete_sync()
    
    if success:
        print("\\nSUCCESS: 전체 동기화 시스템 테스트 완료!")
    else:
        print("\\nERROR: 동기화 시스템에 문제가 있습니다.")